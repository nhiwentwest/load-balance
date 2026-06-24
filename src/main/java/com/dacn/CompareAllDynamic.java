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

/**
 * Compare ALL 5 Algorithms in DYNAMIC Scenario
 * 
 * Dynamic behavior: VMs arrive at different times
 * This tests rebalancing capability
 */
public class CompareAllDynamic {
    
    // Configuration
    private static final int NUM_HOSTS = 20;
    private static final int HOST_MIPS = 1000;
    private static final int HOST_PES = 4;
    private static final int HOST_RAM = 8192;
    private static final int HOST_BW = 100000;
    
    private static final int INITIAL_VMS = 8;
    private static final int ADD_VMS_1 = 7;   // Arrive at t=300
    private static final int ADD_VMS_2 = 5;    // Arrive at t=600
    private static final int TOTAL_VMS = INITIAL_VMS + ADD_VMS_1 + ADD_VMS_2;
    
    // Power Model
    private static final double HOST_MAX_POWER = 200;
    private static final double HOST_STATIC_POWER = 50;
    private static final double TIME_INTERVAL = 50.0;
    
    public static void main(String[] args) {
        System.out.println("============================================================");
        System.out.println("   DYNAMIC SCENARIO - ALL 5 ALGORITHMS COMPARISON");
        System.out.println("============================================================");
        System.out.println("Config: " + NUM_HOSTS + " Hosts");
        System.out.println("  - Initial VMs: " + INITIAL_VMS + " (t=0)");
        System.out.println("  - Add more VMs: " + ADD_VMS_1 + " (t=300)");
        System.out.println("  - Add more VMs: " + ADD_VMS_2 + " (t=600)");
        System.out.println("  - Total VMs: " + TOTAL_VMS);
        System.out.println("------------------------------------------------------------\n");
        
        // Run all algorithms
        Result resultFFD = runDynamic("FFD");
        Result resultPEAP = runDynamic("PEAP");
        Result resultTabu = runDynamic("Tabu");
        Result resultACO = runDynamic("ACO");
        Result resultPSO = runDynamic("PSO");
        
        // Print comparison
        printComparison(resultFFD, resultPEAP, resultTabu, resultACO, resultPSO);
        printRanking(resultFFD, resultPEAP, resultTabu, resultACO, resultPSO);
    }
    
    private static Result runDynamic(String algorithm) {
        System.out.println("\n>>> Running " + algorithm + "...");
        
        // Reset counters
        VmAllocationPolicyFfdPowerAware.resetCounters();
        VmAllocationPolicyPEAP.resetCounters();
        VmAllocationPolicyTabuSearch.resetCounters();
        VmAllocationPolicyACO.resetCounters();
        VmAllocationPolicyPSO.resetCounters();
        
        final double[] energyTracker = {0};
        final double[] previousTime = {0};
        
        CloudSimPlus simulation = new CloudSimPlus();
        Datacenter datacenter = createDatacenter(simulation, algorithm);
        DatacenterBroker broker = new DatacenterBrokerSimple(simulation);
        
        List<Vm> allVms = new ArrayList<>();
        final List<Double> powerSamples = new ArrayList<>();
        
        // Create initial VMs
        List<Vm> initialVms = createVms(INITIAL_VMS, 0);
        allVms.addAll(initialVms);
        
        // FFD: Sort
        if (algorithm.equals("FFD")) {
            initialVms.sort((v1, v2) -> Double.compare(v2.getMips(), v1.getMips()));
        }
        
        broker.submitVmList(new ArrayList<>(initialVms));
        List<Cloudlet> cloudlets = createCloudlets(INITIAL_VMS);
        broker.submitCloudletList(cloudlets);
        
        for (int i = 0; i < cloudlets.size(); i++) {
            cloudlets.get(i).setVm(initialVms.get(i % initialVms.size()));
        }
        
        // Dynamic VM arrival
        final int[] addedCount = {0};
        simulation.addOnClockTickListener(new EventListener<>() {
            @Override
            public void update(EventInfo info) {
                double t = info.getTime();
                
                // Add VMs at t=300
                if (Math.abs(t - 300) < 0.1 && addedCount[0] == 0) {
                    System.out.println("  *** [t=" + t + "] Adding " + ADD_VMS_1 + " VMs ***");
                    List<Vm> newVms = createVms(ADD_VMS_1, INITIAL_VMS);
                    allVms.addAll(newVms);
                    broker.submitVmList(newVms);
                    List<Cloudlet> newCloudlets = createCloudlets(ADD_VMS_1);
                    broker.submitCloudletList(newCloudlets);
                    for (int i = 0; i < newCloudlets.size(); i++) {
                        newCloudlets.get(i).setVm(newVms.get(i % newVms.size()));
                    }
                    addedCount[0]++;
                }
                
                // Add VMs at t=600
                if (Math.abs(t - 600) < 0.1 && addedCount[0] == 1) {
                    System.out.println("  *** [t=" + t + "] Adding " + ADD_VMS_2 + " VMs ***");
                    List<Vm> newVms = createVms(ADD_VMS_2, INITIAL_VMS + ADD_VMS_1);
                    allVms.addAll(newVms);
                    broker.submitVmList(newVms);
                    List<Cloudlet> newCloudlets = createCloudlets(ADD_VMS_2);
                    broker.submitCloudletList(newCloudlets);
                    for (int i = 0; i < newCloudlets.size(); i++) {
                        newCloudlets.get(i).setVm(newVms.get(i % newVms.size()));
                    }
                    addedCount[0]++;
                }
                
                // Track energy
                if (t - previousTime[0] >= TIME_INTERVAL) {
                    double dt = t - previousTime[0];
                    if (dt > 0) {
                        double power = calculateTotalPower(datacenter);
                        powerSamples.add(power);
                        energyTracker[0] += (power * (dt / 3600.0)) / 1000.0;
                        previousTime[0] = t;
                    }
                }
            }
        });
        
        simulation.start();
        
        double avgPower = powerSamples.stream().mapToDouble(Double::doubleValue).average().orElse(0);
        
        int activeHosts = 0;
        for (Host host : datacenter.getHostList()) {
            if (!host.getVmList().isEmpty()) activeHosts++;
        }
        
        Result result = new Result();
        result.algorithm = algorithm;
        result.energy = energyTracker[0];
        result.avgPower = avgPower;
        result.activeHosts = activeHosts;
        result.vmsPlaced = (int) allVms.stream().filter(vm -> vm.getHost() != null).count();
        
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
            case "FFD": return new DatacenterSimple(simulation, hostList, new VmAllocationPolicyFfdPowerAware());
            case "PEAP": return new DatacenterSimple(simulation, hostList, new VmAllocationPolicyPEAP());
            case "Tabu": return new DatacenterSimple(simulation, hostList, new VmAllocationPolicyTabuSearch());
            case "ACO": return new DatacenterSimple(simulation, hostList, new VmAllocationPolicyACO());
            case "PSO": return new DatacenterSimple(simulation, hostList, new VmAllocationPolicyPSO());
            default: return new DatacenterSimple(simulation, hostList);
        }
    }
    
    private static List<Vm> createVms(int count, int startId) {
        List<Vm> list = new ArrayList<>();
        int[] mips = {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000};
        for (int i = 0; i < count; i++) {
            Vm vm = new VmSimple(mips[(startId + i) % mips.length], 1);
            vm.setRam(512).setBw(1000).setSize(5000);
            vm.setCloudletScheduler(new CloudletSchedulerSpaceShared());
            list.add(vm);
        }
        return list;
    }
    
    private static List<Cloudlet> createCloudlets(int count) {
        List<Cloudlet> list = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            Cloudlet c = new CloudletSimple(5000 + i * 500, 1);
            c.setUtilizationModelCpu(new UtilizationModelDynamic(0.3 + Math.random() * 0.5));
            list.add(c);
        }
        return list;
    }
    
    private static double calculateTotalPower(Datacenter dc) {
        double total = 0;
        for (Host h : dc.getHostList()) {
            // FIX: Count ALL hosts - idle hosts still consume static power
            double totalMips = h.getPeList().stream().mapToDouble(Pe::getCapacity).sum();
            double usedMips = h.getVmList().stream().mapToDouble(vm -> vm.getMips()).sum();
            double util = Math.min(1.0, usedMips / totalMips);
            total += HOST_STATIC_POWER + (util * (HOST_MAX_POWER - HOST_STATIC_POWER));
        }
        return total;
    }
    
    private static void printComparison(Result ffd, Result peap, Result tabu, Result aco, Result pso) {
        System.out.println("\n");
        System.out.println("########################################################");
        System.out.println("     DYNAMIC SCENARIO - ALL ALGORITHMS COMPARISON        ");
        System.out.println("########################################################");
        System.out.println();
        System.out.println(String.format("%-15s %12s %12s", "Algorithm", "Energy(kWh)", "AvgPower(W)"));
        System.out.println(String.format("%-15s %12s %12s", "---------------", "------------", "------------"));
        System.out.println(String.format("%-15s %12.4f %12.2f", "FFD+Power", ffd.energy, ffd.avgPower));
        System.out.println(String.format("%-15s %12.4f %12.2f", "PEAP", peap.energy, peap.avgPower));
        System.out.println(String.format("%-15s %12.4f %12.2f", "Tabu Search", tabu.energy, tabu.avgPower));
        System.out.println(String.format("%-15s %12.4f %12.2f", "ACO", aco.energy, aco.avgPower));
        System.out.println(String.format("%-15s %12.4f %12.2f", "PSO", pso.energy, pso.avgPower));
        System.out.println("########################################################\n");
    }
    
    private static void printRanking(Result ffd, Result peap, Result tabu, Result aco, Result pso) {
        List<Result> results = new ArrayList<>();
        results.add(ffd); results.add(peap); results.add(tabu); results.add(aco); results.add(pso);
        results.sort((r1, r2) -> Double.compare(r1.energy, r2.energy));
        
        System.out.println("============================================================");
        System.out.println("       DYNAMIC SCENARIO - RANKING (by Energy)              ");
        System.out.println("============================================================");
        System.out.println();
        
        Result best = results.get(0);
        double baseline = best.energy;
        
        String[] medals = {"🥇", "🥈", "🥉", "  ", "  "};
        for (int i = 0; i < results.size(); i++) {
            Result r = results.get(i);
            double diff = ((r.energy - baseline) / baseline) * 100;
            String sign = diff >= 0 ? "+" : "";
            String rank = (diff == 0) ? "=" : String.valueOf(i+1);
            System.out.println(medals[i] + " #" + rank + " " + r.algorithm + 
                " : " + String.format("%.4f", r.energy) + " kWh" +
                " (" + sign + String.format("%.2f", diff) + "%)");
        }
        
        System.out.println();
        System.out.println("============================================================");
        System.out.println("ANALYSIS - Why Winner in Dynamic Scenario:");
        System.out.println("============================================================");
        System.out.println();
        System.out.println("Dynamic behavior (VMs arriving at different times):");
        System.out.println("  - t=0:   " + INITIAL_VMS + " VMs start");
        System.out.println("  - t=300: +" + ADD_VMS_1 + " VMs arrive");
        System.out.println("  - t=600: +" + ADD_VMS_2 + " VMs arrive");
        System.out.println();
        System.out.println("Key difference from static scenario:");
        System.out.println("  - Each new VM arrival requires re-optimization");
        System.out.println("  - Algorithms must adapt to changing workload");
        System.out.println("  - Global optimization becomes more important");
        System.out.println();
        
        if (best.algorithm.equals("PSO")) {
            System.out.println(">>> PSO wins because:");
            System.out.println("    - Velocity-based exploration finds better solutions");
            System.out.println("    - pBest + gBest guidance balances exploration/exploitation");
        } else if (best.algorithm.equals("Tabu")) {
            System.out.println(">>> Tabu Search wins because:");
            System.out.println("    - Tabu list prevents revisiting bad solutions");
            System.out.println("    - Global search adapts well to changes");
        } else if (best.algorithm.equals("ACO")) {
            System.out.println(">>> ACO wins because:");
            System.out.println("    - Pheromone积累了经验");
            System.out.println("    - Probabilistic selection handles uncertainty");
        } else if (best.algorithm.equals("PEAP")) {
            System.out.println(">>> PEAP wins because:");
            System.out.println("    - Simple greedy works well in dynamic too");
            System.out.println("    - Best Fit reduces fragmentation");
        } else {
            System.out.println(">>> FFD+Power wins because:");
            System.out.println("    - Sorting helps even with dynamic arrivals");
        }
        
        System.out.println();
        System.out.println("============================================================\n");
    }
    
    static class Result {
        String algorithm;
        double energy;
        double avgPower;
        int activeHosts;
        int vmsPlaced;
    }
}
