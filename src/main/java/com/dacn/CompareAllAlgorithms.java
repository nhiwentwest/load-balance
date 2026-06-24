package com.dacn;

import org.cloudsimplus.brokers.DatacenterBroker;
import org.cloudsimplus.brokers.DatacenterBrokerSimple;
import org.cloudsimplus.cloudlets.Cloudlet;
import org.cloudsimplus.cloudlets.CloudletSimple;
import org.cloudsimplus.core.CloudSimPlus;
import org.cloudsimplus.datacenters.Datacenter;
import org.cloudsimplus.datacenters.DatacenterSimple;
import org.cloudsimplus.hosts.Host;
import org.cloudsimplus.hosts.HostSimple;
import org.cloudsimplus.resources.Pe;
import org.cloudsimplus.resources.PeSimple;
import org.cloudsimplus.schedulers.cloudlet.CloudletSchedulerSpaceShared;
import org.cloudsimplus.schedulers.vm.VmSchedulerTimeShared;
import org.cloudsimplus.utilizationmodels.UtilizationModelDynamic;
import org.cloudsimplus.vms.Vm;
import org.cloudsimplus.vms.VmSimple;
import org.cloudsimplus.listeners.EventInfo;
import org.cloudsimplus.listeners.EventListener;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Compare ALL 5 VM Placement Algorithms:
 * 1. FFD + Power-Aware
 * 2. PEAP (Power Efficient Allocation Policy)
 * 3. Tabu Search
 * 4. ACO (Ant Colony Optimization)
 * 5. PSO (Particle Swarm Optimization)
 */
public class CompareAllAlgorithms {
    
    // Configuration
    private static final int NUM_HOSTS = 30;
    private static final int HOST_MIPS = 1000;
    private static final int HOST_PES = 4;
    private static final int HOST_RAM = 8192;
    private static final int HOST_BW = 100000;
    
    private static final int NUM_VMS = 25;
    
    // Power Model
    private static final double HOST_MAX_POWER = 200;
    private static final double HOST_STATIC_POWER = 50;
    
    private static final double TIME_INTERVAL = 100.0;
    
    public static void main(String[] args) {
        System.out.println("============================================================");
        System.out.println("   COMPARING ALL 5 VM PLACEMENT ALGORITHMS");
        System.out.println("============================================================");
        System.out.println("Algorithms:");
        System.out.println("  1. FFD + Power-Aware");
        System.out.println("  2. PEAP (Power Efficient)");
        System.out.println("  3. Tabu Search");
        System.out.println("  4. ACO (Ant Colony)");
        System.out.println("  5. PSO (Particle Swarm)");
        System.out.println("Config: " + NUM_HOSTS + " Hosts, " + NUM_VMS + " VMs");
        System.out.println("------------------------------------------------------------\n");
        
        // Run all algorithms
        Result resultFFD = runAlgorithm("FFD");
        Result resultPEAP = runAlgorithm("PEAP");
        Result resultTabu = runAlgorithm("Tabu");
        Result resultACO = runAlgorithm("ACO");
        Result resultPSO = runAlgorithm("PSO");
        
        // Print comparison
        printComparison(resultFFD, resultPEAP, resultTabu, resultACO, resultPSO);
        
        // Print ranking
        printRanking(resultFFD, resultPEAP, resultTabu, resultACO, resultPSO);
    }
    
    private static Result runAlgorithm(String algorithm) {
        System.out.println("\n>>> Running " + algorithm + "...");
        
        // Reset counters
        VmAllocationPolicyFfdPowerAware.resetCounters();
        VmAllocationPolicyTabuSearch.resetCounters();
        VmAllocationPolicyPEAP.resetCounters();
        VmAllocationPolicyACO.resetCounters();
        VmAllocationPolicyPSO.resetCounters();
        
        double totalEnergy = 0;
        double previousTime = 0;
        
        CloudSimPlus simulation = new CloudSimPlus();
        Datacenter datacenter = createDatacenter(simulation, algorithm);
        DatacenterBroker broker = new DatacenterBrokerSimple(simulation);
        
        // Create VMs
        List<Vm> vmList = createVms(NUM_VMS);
        
        // FFD: Sort VMs
        if (algorithm.equals("FFD")) {
            vmList.sort((v1, v2) -> Double.compare(v2.getMips(), v1.getMips()));
        }
        
        // Create cloudlets
        List<Cloudlet> cloudletList = createCloudlets(NUM_VMS);
        
        // Energy tracking
        final double[] energyTracker = {0};
        final double[] previousTimeArr = {0};
        simulation.addOnClockTickListener(new EventListener<>() {
            @Override
            public void update(EventInfo info) {
                double currentTime = info.getTime();
                if (currentTime - previousTimeArr[0] >= TIME_INTERVAL) {
                    double timeDelta = currentTime - previousTimeArr[0];
                    if (timeDelta > 0) {
                        double currentPower = calculateTotalPower(datacenter);
                        double energyDelta = (currentPower * (timeDelta / 3600.0)) / 1000.0;
                        energyTracker[0] += energyDelta;
                        previousTimeArr[0] = currentTime;
                    }
                }
            }
        });
        
        broker.submitVmList(vmList);
        broker.submitCloudletList(cloudletList);
        
        for (int i = 0; i < cloudletList.size(); i++) {
            cloudletList.get(i).setVm(vmList.get(i % vmList.size()));
        }
        
        simulation.start();
        
        totalEnergy = energyTracker[0];
        
        int activeHosts = 0;
        for (Host host : datacenter.getHostList()) {
            if (!host.getVmList().isEmpty()) {
                activeHosts++;
            }
        }
        
        Result result = new Result();
        result.algorithm = algorithm;
        result.energy = totalEnergy;
        result.activeHosts = activeHosts;
        result.vmsPlaced = (int) vmList.stream().filter(vm -> vm.getHost() != null).count();
        
        System.out.println("  " + algorithm + " Energy: " + String.format("%.4f", result.energy) + " kWh");
        
        return result;
    }
    
    private static Datacenter createDatacenter(CloudSimPlus simulation, String algorithm) {
        List<Host> hostList = new ArrayList<>();
        
        for (int i = 0; i < NUM_HOSTS; i++) {
            List<Pe> peList = new ArrayList<>();
            for (int j = 0; j < HOST_PES; j++) {
                peList.add(new PeSimple(HOST_MIPS));
            }
            
            Host host = new HostSimple(HOST_RAM, HOST_BW, 1000000, peList);
            host.setVmScheduler(new VmSchedulerTimeShared());
            hostList.add(host);
        }
        
        switch (algorithm) {
            case "FFD":
                return new DatacenterSimple(simulation, hostList, new VmAllocationPolicyFfdPowerAware());
            case "PEAP":
                return new DatacenterSimple(simulation, hostList, new VmAllocationPolicyPEAP());
            case "Tabu":
                return new DatacenterSimple(simulation, hostList, new VmAllocationPolicyTabuSearch());
            case "ACO":
                return new DatacenterSimple(simulation, hostList, new VmAllocationPolicyACO());
            case "PSO":
                return new DatacenterSimple(simulation, hostList, new VmAllocationPolicyPSO());
            default:
                return new DatacenterSimple(simulation, hostList);
        }
    }
    
    private static List<Vm> createVms(int count) {
        List<Vm> vmList = new ArrayList<>();
        
        int[] mipsValues = {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000};
        
        for (int i = 0; i < count; i++) {
            int mips = mipsValues[i % mipsValues.length];
            Vm vm = new VmSimple(mips, 1);
            vm.setRam(512).setBw(1000).setSize(5000);
            vm.setCloudletScheduler(new CloudletSchedulerSpaceShared());
            vmList.add(vm);
        }
        
        return vmList;
    }
    
    private static List<Cloudlet> createCloudlets(int count) {
        List<Cloudlet> cloudletList = new ArrayList<>();
        
        for (int i = 0; i < count; i++) {
            Cloudlet cloudlet = new CloudletSimple(5000 + i * 500, 1);
            cloudlet.setUtilizationModelCpu(new UtilizationModelDynamic(0.3 + Math.random() * 0.5));
            cloudletList.add(cloudlet);
        }
        
        return cloudletList;
    }
    
    private static double calculateTotalPower(Datacenter datacenter) {
        double totalPower = 0;
        
        for (Host host : datacenter.getHostList()) {
            if (!host.getVmList().isEmpty()) {
                double totalMips = host.getPeList().stream()
                    .mapToDouble(Pe::getCapacity)
                    .sum();
                double allocatedMips = host.getVmList().stream()
                    .mapToDouble(vm -> vm.getMips())
                    .sum();
                double utilization = Math.min(1.0, allocatedMips / totalMips);
                
                double power = HOST_STATIC_POWER + (utilization * (HOST_MAX_POWER - HOST_STATIC_POWER));
                totalPower += power;
            }
        }
        
        return totalPower;
    }
    
    private static void printComparison(Result ffd, Result peap, Result tabu, Result aco, Result pso) {
        System.out.println("\n");
        System.out.println("########################################################");
        System.out.println("              ALL ALGORITHMS COMPARISON                  ");
        System.out.println("########################################################");
        System.out.println();
        System.out.println(String.format("%-15s %12s %12s", "Algorithm", "Energy(kWh)", "ActiveHosts"));
        System.out.println(String.format("%-15s %12s %12s", "---------------", "------------", "------------"));
        System.out.println(String.format("%-15s %12.4f %12d", "FFD+Power", ffd.energy, ffd.activeHosts));
        System.out.println(String.format("%-15s %12.4f %12d", "PEAP", peap.energy, peap.activeHosts));
        System.out.println(String.format("%-15s %12.4f %12d", "Tabu Search", tabu.energy, tabu.activeHosts));
        System.out.println(String.format("%-15s %12.4f %12d", "ACO", aco.energy, aco.activeHosts));
        System.out.println(String.format("%-15s %12.4f %12d", "PSO", pso.energy, pso.activeHosts));
        System.out.println("########################################################\n");
    }
    
    private static void printRanking(Result ffd, Result peap, Result tabu, Result aco, Result pso) {
        // Create ranking
        List<Result> results = new ArrayList<>();
        results.add(ffd);
        results.add(peap);
        results.add(tabu);
        results.add(aco);
        results.add(pso);
        
        // Sort by energy (ascending - lower is better)
        results.sort((r1, r2) -> Double.compare(r1.energy, r2.energy));
        
        System.out.println("============================================================");
        System.out.println("                    RANKING (by Energy)                    ");
        System.out.println("============================================================");
        System.out.println();
        
        Result best = results.get(0);
        double baseline = best.energy;
        
        for (int i = 0; i < results.size(); i++) {
            Result r = results.get(i);
            double improvement = ((r.energy - baseline) / baseline) * 100;
            
            String medal;
            if (i == 0) medal = "🥇";
            else if (i == 1) medal = "🥈";
            else if (i == 2) medal = "🥉";
            else medal = "  ";
            
            System.out.println(medal + " #" + (i + 1) + " " + r.algorithm + 
                " : " + String.format("%.4f", r.energy) + " kWh" +
                " (" + String.format("+%+.2f", improvement) + "% vs best)");
        }
        
        System.out.println();
        System.out.println("============================================================");
        System.out.println("ANALYSIS:");
        System.out.println("============================================================");
        System.out.println();
        
        // Find algorithm characteristics
        Result worst = results.get(results.size() - 1);
        
        System.out.println("BEST: " + best.algorithm + " (" + String.format("%.4f", best.energy) + " kWh)");
        System.out.println("WORST: " + worst.algorithm + " (" + String.format("%.4f", worst.energy) + " kWh)");
        System.out.println();
        
        // Algorithm characteristics
        System.out.println("Algorithm Characteristics:");
        System.out.println("- FFD+Power   : Greedy, fast, sorts VMs by size first");
        System.out.println("- PEAP        : Pure power minimization with Best Fit tie-break");
        System.out.println("- Tabu Search : Global search with memory (tabu list)");
        System.out.println("- ACO         : Swarm intelligence, pheromone-based");
        System.out.println("- PSO         : Swarm intelligence, velocity-based");
        System.out.println();
        
        // Recommendations
        System.out.println("Recommendations:");
        System.out.println("- For static workload: FFD+Power or PEAP (fast & efficient)");
        System.out.println("- For dynamic workload: Tabu, ACO, or PSO (global optimization)");
        System.out.println("- For energy-critical: Use the winner of this comparison");
        System.out.println();
        System.out.println("============================================================\n");
    }
    
    static class Result {
        String algorithm;
        double energy;
        int activeHosts;
        int vmsPlaced;
    }
}
