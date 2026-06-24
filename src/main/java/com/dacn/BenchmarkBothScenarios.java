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
 * Proper Dynamic VM Allocation Benchmark
 * 
 * Static: All VMs allocated at once, NO migration allowed
 * Dynamic: VMs arrive in waves over time, only initial placement (no migration)
 */
public class BenchmarkBothScenarios {
    
    // Configuration
    private static final int NUM_HOSTS = 50;
    private static final int HOST_MIPS = 1000;
    private static final int HOST_PES = 4;
    private static final int HOST_RAM = 8192;
    private static final int HOST_BW = 100000;
    
    private static final int NUM_VMS = 100;
    private static final int NUM_RUNS = 10;
    
    // Power Model
    private static final double HOST_MAX_POWER = 200;
    private static final double HOST_STATIC_POWER = 50;
    private static final double TIME_INTERVAL = 50.0;
    
    // Cloudlet length (simulation duration)
    private static final long CLOUDLET_LENGTH = 10000;
    
    public static void main(String[] args) {
        System.out.println("╔════════════════════════════════════════════════════════════════════════════╗");
        System.out.println("║     BENCHMARK: Static vs Dynamic VM Allocation                             ║");
        System.out.println("╠════════════════════════════════════════════════════════════════════════════╣");
        System.out.println("║ Static: All VMs at t=0                                                     ║");
        System.out.println("║ Dynamic: VMs arrive in waves (t=100, 300, 500)                            ║");
        System.out.println("╚════════════════════════════════════════════════════════════════════════════╝");
        System.out.println();
        
        String[] algorithms = {"NO_MIGRATION", "FFD", "PEAP", "Tabu", "ACO", "PSO"};
        
        // ========== STATIC SCENARIO ==========
        System.out.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        System.out.println("                    STATIC SCENARIO                                 ");
        System.out.println("          (All " + NUM_VMS + " VMs at t=0)                                  ");
        System.out.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        
        List<Double>[] staticResults = runStaticScenario(algorithms, NUM_RUNS);
        
        // ========== DYNAMIC SCENARIO ==========
        System.out.println();
        System.out.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        System.out.println("                    DYNAMIC SCENARIO                               ");
        System.out.println("      (VMs in waves: t=100, t=300, t=500)                         ");
        System.out.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        
        List<Double>[] dynamicResults = runDynamicScenario(algorithms, NUM_RUNS);
        
        // ========== COMPARISON ==========
        printComparison(algorithms, staticResults, dynamicResults);
    }
    
    @SuppressWarnings("unchecked")
    private static List<Double>[] runStaticScenario(String[] algorithms, int numRuns) {
        List<Double>[] results = new ArrayList[algorithms.length];
        for (int i = 0; i < algorithms.length; i++) {
            results[i] = new ArrayList<>();
        }
        
        for (int run = 1; run <= numRuns; run++) {
            System.out.print("Run " + run + "/" + numRuns + "...");
            for (int i = 0; i < algorithms.length; i++) {
                results[i].add(runStatic(algorithms[i]));
            }
            System.out.println(" Done");
        }
        
        printResults("Static", algorithms, results);
        return results;
    }
    
    @SuppressWarnings("unchecked")
    private static List<Double>[] runDynamicScenario(String[] algorithms, int numRuns) {
        List<Double>[] results = new ArrayList[algorithms.length];
        for (int i = 0; i < algorithms.length; i++) {
            results[i] = new ArrayList<>();
        }
        
        for (int run = 1; run <= numRuns; run++) {
            System.out.print("Run " + run + "/" + numRuns + "...");
            for (int i = 0; i < algorithms.length; i++) {
                results[i].add(runDynamic(algorithms[i]));
            }
            System.out.println(" Done");
        }
        
        printResults("Dynamic", algorithms, results);
        return results;
    }
    
    private static void printResults(String scenario, String[] algorithms, List<Double>[] results) {
        System.out.println();
        System.out.println("  Results for " + scenario + " scenario:");
        for (int i = 0; i < algorithms.length; i++) {
            double mean = results[i].stream().mapToDouble(Double::doubleValue).average().orElse(0);
            double stdDev = Math.sqrt(results[i].stream()
                .mapToDouble(r -> Math.pow(r - mean, 2))
                .sum() / results[i].size());
            System.out.printf("    %-15s: Mean=%.4f kWh, StdDev=%.4f%n", algorithms[i], mean, stdDev);
        }
    }
    
    // ===================== STATIC: All VMs at once =====================
    private static double runStatic(String algorithm) {
        resetCounters(algorithm);
        
        final double[] energy = {0};
        final double[] lastTime = {0};
        
        CloudSimPlus sim = new CloudSimPlus();
        Datacenter dc = createDatacenter(sim, algorithm);
        DatacenterBroker broker = new DatacenterBrokerSimple(sim);
        
        List<Vm> vmList = createVms(NUM_VMS);
        
        // FFD: sort VMs by MIPS descending
        if (algorithm.equals("FFD")) {
            vmList.sort((v1, v2) -> Double.compare(v2.getMips(), v1.getMips()));
        }
        
        List<Cloudlet> cloudlets = createCloudlets(NUM_VMS);
        
        // Energy tracking
        sim.addOnClockTickListener(evt -> {
            double t = evt.getTime();
            if (t - lastTime[0] >= TIME_INTERVAL) {
                double dt = t - lastTime[0];
                if (dt > 0) {
                    energy[0] += (calculatePower(dc) * dt) / 3600000.0;
                    lastTime[0] = t;
                }
            }
        });
        
        broker.submitVmList(vmList);
        broker.submitCloudletList(cloudlets);
        
        for (int i = 0; i < cloudlets.size(); i++) {
            cloudlets.get(i).setVm(vmList.get(i % vmList.size()));
        }
        
        sim.start();
        return energy[0];
    }
    
    // ===================== DYNAMIC: VMs in waves =====================
    private static double runDynamic(String algorithm) {
        resetCounters(algorithm);
        
        final double[] energy = {0};
        final double[] lastTime = {0};
        
        CloudSimPlus sim = new CloudSimPlus();
        Datacenter dc = createDatacenter(sim, algorithm);
        DatacenterBroker broker = new DatacenterBrokerSimple(sim);
        
        // Create all VMs upfront - with different start times
        List<Vm> allVms = createVms(NUM_VMS);
        
        // FFD: sort VMs by MIPS descending
        if (algorithm.equals("FFD")) {
            allVms.sort((v1, v2) -> Double.compare(v2.getMips(), v1.getMips()));
        }
        
        List<Cloudlet> allCloudlets = createCloudlets(NUM_VMS);
        
        // Wave 1: VMs 0-39 start at t=100
        // Wave 2: VMs 40-69 start at t=300
        // Wave 3: VMs 70-99 start at t=500
        for (int i = 0; i < 40; i++) {
            allVms.get(i).setStartTime(100.0);
            allCloudlets.get(i).setStartTime(100.0);
        }
        for (int i = 40; i < 70; i++) {
            allVms.get(i).setStartTime(300.0);
            allCloudlets.get(i).setStartTime(300.0);
        }
        for (int i = 70; i < NUM_VMS; i++) {
            allVms.get(i).setStartTime(500.0);
            allCloudlets.get(i).setStartTime(500.0);
        }
        
        // Submit all VMs at once - they will start at their designated times
        broker.submitVmList(allVms);
        broker.submitCloudletList(allCloudlets);
        
        for (int i = 0; i < allCloudlets.size(); i++) {
            allCloudlets.get(i).setVm(allVms.get(i % allVms.size()));
        }
        
        // Energy tracking
        sim.addOnClockTickListener(evt -> {
            double t = evt.getTime();
            if (t - lastTime[0] >= TIME_INTERVAL) {
                double dt = t - lastTime[0];
                if (dt > 0) {
                    energy[0] += (calculatePower(dc) * dt) / 3600000.0;
                    lastTime[0] = t;
                }
            }
        });
        
        sim.start();
        return energy[0];
    }
    
    private static void resetCounters(String algorithm) {
        switch (algorithm) {
            case "FFD": VmAllocationPolicyFfdPowerAware.resetCounters(); break;
            case "PEAP": VmAllocationPolicyPEAP.resetCounters(); break;
            case "Tabu": VmAllocationPolicyTabuSearch.resetCounters(); break;
            case "ACO": VmAllocationPolicyACO.resetCounters(); break;
            case "PSO": VmAllocationPolicyPSO.resetCounters(); break;
            case "NO_MIGRATION": VmAllocationPolicyNoMigration.resetCounters(); break;
        }
    }
    
    // ===================== HELPER METHODS =====================
    private static Datacenter createDatacenter(CloudSimPlus sim, String algorithm) {
        List<Host> hosts = new ArrayList<>();
        
        for (int i = 0; i < NUM_HOSTS; i++) {
            List<Pe> pes = new ArrayList<>();
            for (int j = 0; j < HOST_PES; j++) {
                pes.add(new PeSimple(HOST_MIPS));
            }
            Host host = new HostSimple(HOST_RAM, HOST_BW, 1000000, pes);
            host.setVmScheduler(new VmSchedulerTimeShared());
            hosts.add(host);
        }
        
        switch (algorithm) {
            case "NO_MIGRATION": return new DatacenterSimple(sim, hosts, new VmAllocationPolicyNoMigration());
            case "FFD": return new DatacenterSimple(sim, hosts, new VmAllocationPolicyFfdPowerAware());
            case "PEAP": return new DatacenterSimple(sim, hosts, new VmAllocationPolicyPEAP());
            case "Tabu": return new DatacenterSimple(sim, hosts, new VmAllocationPolicyTabuSearch());
            case "ACO": return new DatacenterSimple(sim, hosts, new VmAllocationPolicyACO());
            case "PSO": return new DatacenterSimple(sim, hosts, new VmAllocationPolicyPSO());
            default: return new DatacenterSimple(sim, hosts);
        }
    }
    
    private static List<Vm> createVms(int count) {
        List<Vm> list = new ArrayList<>();
        int[] mips = {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000};
        for (int i = 0; i < count; i++) {
            Vm vm = new VmSimple(mips[i % mips.length], 1);
            vm.setRam(512).setBw(1000).setSize(5000);
            vm.setCloudletScheduler(new CloudletSchedulerSpaceShared());
            list.add(vm);
        }
        return list;
    }
    
    private static List<Cloudlet> createCloudlets(int count) {
        List<Cloudlet> list = new ArrayList<>();
        Random rand = new Random();
        for (int i = 0; i < count; i++) {
            // Longer cloudlets to ensure simulation runs longer
            Cloudlet c = new CloudletSimple(CLOUDLET_LENGTH, 1);
            c.setUtilizationModelCpu(new UtilizationModelDynamic(0.3 + rand.nextDouble() * 0.5));
            list.add(c);
        }
        return list;
    }
    
    private static double calculatePower(Datacenter dc) {
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
    
    private static void printComparison(String[] algorithms, List<Double>[] staticResults, List<Double>[] dynamicResults) {
        double[] staticMeans = new double[algorithms.length];
        double[] dynamicMeans = new double[algorithms.length];
        
        for (int i = 0; i < algorithms.length; i++) {
            staticMeans[i] = staticResults[i].stream().mapToDouble(Double::doubleValue).average().orElse(0);
            dynamicMeans[i] = dynamicResults[i].stream().mapToDouble(Double::doubleValue).average().orElse(0);
        }
        
        int staticWinner = 0, dynamicWinner = 0;
        for (int i = 1; i < algorithms.length; i++) {
            if (staticMeans[i] < staticMeans[staticWinner]) staticWinner = i;
            if (dynamicMeans[i] < dynamicMeans[dynamicWinner]) dynamicWinner = i;
        }
        
        System.out.println();
        System.out.println("╔════════════════════════════════════════════════════════════════════════════════════╗");
        System.out.println("║                         FINAL COMPARISON                                          ║");
        System.out.println("╠════════════════════════════════════════════════════════════════════════════════════╣");
        
        System.out.print("║  Scenario  ");
        for (String alg : algorithms) System.out.printf("│ %8s ", alg);
        System.out.println("│  Winner   ║");
        System.out.println("╠════════════");
        for (int i = 0; i < algorithms.length; i++) System.out.print("╦══════════");
        System.out.println("╦═══════════╣");
        
        System.out.printf("║  Static   ");
        for (double m : staticMeans) System.out.printf("│ %8.4f ", m);
        System.out.printf("│ %-8s ║%n", algorithms[staticWinner]);
        
        System.out.printf("║  Dynamic  ");
        for (double m : dynamicMeans) System.out.printf("│ %8.4f ", m);
        System.out.printf("│ %-8s ║%n", algorithms[dynamicWinner]);
        
        System.out.println("╚════════════");
        for (int i = 0; i < algorithms.length; i++) System.out.print("╩══════════");
        System.out.println("╩═══════════╝");
        
        // Analysis
        System.out.println("\n📊 ANALYSIS:");
        System.out.printf("  • Static:   %s wins (%.1f%% better than NoMigration)%n",
            algorithms[staticWinner], 
            (staticMeans[0] - staticMeans[staticWinner]) / staticMeans[0] * 100);
        System.out.printf("  • Dynamic: %s wins (%.1f%% better than NoMigration)%n",
            algorithms[dynamicWinner],
            (dynamicMeans[0] - dynamicMeans[dynamicWinner]) / dynamicMeans[0] * 100);
    }
}
