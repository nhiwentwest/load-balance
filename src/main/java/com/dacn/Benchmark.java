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
 * Benchmark - Run multiple times to get STATISTICALLY MEANINGFUL results
 * 
 * Why multiple runs?
 * - Single run = may be lucky/unlucky
 * - Multiple runs = average performance, shows variance
 * - Standard deviation = shows stability
 */
public class Benchmark {
    
    // Configuration
    private static final int NUM_HOSTS = 50;
    private static final int HOST_MIPS = 1000;
    private static final int HOST_PES = 4;
    private static final int HOST_RAM = 8192;
    private static final int HOST_BW = 100000;
    
    private static final int NUM_VMS = 100;
    private static final int NUM_RUNS = 10;  // Run 10 times
    
    // Power Model
    private static final double HOST_MAX_POWER = 200;
    private static final double HOST_STATIC_POWER = 50;
    private static final double TIME_INTERVAL = 50.0;
    
    public static void main(String[] args) {
        System.out.println("╔═══════════════════════════════════════════════════════════╗");
        System.out.println("║           BENCHMARK - Multiple Runs Statistics            ║");
        System.out.println("╠═══════════════════════════════════════════════════════════╣");
        System.out.println("║ Running each algorithm " + NUM_RUNS + " times for statistical accuracy ║");
        System.out.println("╚═══════════════════════════════════════════════════════════╝");
        System.out.println();
        
        // Store results for all runs
        List<Double> ffdResults = new ArrayList<>();
        List<Double> peapResults = new ArrayList<>();
        List<Double> tabuResults = new ArrayList<>();
        List<Double> acoResults = new ArrayList<>();
        List<Double> psoResults = new ArrayList<>();
        
        // Run benchmark
        for (int run = 1; run <= NUM_RUNS; run++) {
            System.out.println("--- Run " + run + "/" + NUM_RUNS + " ---");
            
            ffdResults.add(runOnce("FFD"));
            peapResults.add(runOnce("PEAP"));
            tabuResults.add(runOnce("Tabu"));
            acoResults.add(runOnce("ACO"));
            psoResults.add(runOnce("PSO"));
        }
        
        // Print statistics
        printStatistics("FFD", ffdResults);
        printStatistics("PEAP", peapResults);
        printStatistics("Tabu", tabuResults);
        printStatistics("ACO", acoResults);
        printStatistics("PSO", psoResults);
        
        // Print final ranking
        printFinalRanking(ffdResults, peapResults, tabuResults, acoResults, psoResults);
    }
    
    private static double runOnce(String algorithm) {
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
        
        List<Vm> vmList = createVms(NUM_VMS);
        
        // FFD: Sort VMs
        if (algorithm.equals("FFD")) {
            vmList.sort((v1, v2) -> Double.compare(v2.getMips(), v1.getMips()));
        }
        
        List<Cloudlet> cloudletList = createCloudlets(NUM_VMS);
        
        // Energy tracking
        simulation.addOnClockTickListener(new EventListener<>() {
            @Override
            public void update(EventInfo info) {
                double t = info.getTime();
                if (t - previousTime[0] >= TIME_INTERVAL) {
                    double dt = t - previousTime[0];
                    if (dt > 0) {
                        double power = calculateTotalPower(datacenter);
                        energyTracker[0] += (power * (dt / 3600.0)) / 1000.0;
                        previousTime[0] = t;
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
        
        return energyTracker[0];
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
            Cloudlet c = new CloudletSimple(5000 + i * 500, 1);
            c.setUtilizationModelCpu(new UtilizationModelDynamic(0.3 + rand.nextDouble() * 0.5));
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
    
    private static void printStatistics(String name, List<Double> results) {
        double sum = results.stream().mapToDouble(Double::doubleValue).sum();
        double mean = sum / results.size();
        double min = results.stream().mapToDouble(Double::doubleValue).min().orElse(0);
        double max = results.stream().mapToDouble(Double::doubleValue).max().orElse(0);
        
        // Calculate standard deviation
        double variance = results.stream()
            .mapToDouble(r -> Math.pow(r - mean, 2))
            .sum() / results.size();
        double stdDev = Math.sqrt(variance);
        
        System.out.printf("%-10s: Mean=%.4f kWh, StdDev=%.4f, Min=%.4f, Max=%.4f%n", 
            name, mean, stdDev, min, max);
    }
    
    private static void printFinalRanking(List<Double> ffd, List<Double> peap, 
            List<Double> tabu, List<Double> aco, List<Double> pso) {
        
        System.out.println("\n╔═══════════════════════════════════════════════════════════╗");
        System.out.println("║              FINAL RANKING (by Mean Energy)               ║");
        System.out.println("╚═══════════════════════════════════════════════════════════╝");
        
        // Calculate means
        double ffdMean = ffd.stream().mapToDouble(Double::doubleValue).average().orElse(0);
        double peapMean = peap.stream().mapToDouble(Double::doubleValue).average().orElse(0);
        double tabuMean = tabu.stream().mapToDouble(Double::doubleValue).average().orElse(0);
        double acoMean = aco.stream().mapToDouble(Double::doubleValue).average().orElse(0);
        double psoMean = pso.stream().mapToDouble(Double::doubleValue).average().orElse(0);
        
        // Create sorted list
        List<StatResult> all = new ArrayList<>();
        all.add(new StatResult("FFD", ffdMean));
        all.add(new StatResult("PEAP", peapMean));
        all.add(new StatResult("Tabu", tabuMean));
        all.add(new StatResult("ACO", acoMean));
        all.add(new StatResult("PSO", psoMean));
        all.sort((a, b) -> Double.compare(a.mean, b.mean));
        
        double best = all.get(0).mean;
        
        String[] medals = {"🥇", "🥈", "🥉", "  ", "  "};
        for (int i = 0; i < all.size(); i++) {
            StatResult r = all.get(i);
            double diff = ((r.mean - best) / best) * 100;
            System.out.printf("%s #%d %-8s: %.4f kWh (%+.1f%%)%n", 
                medals[i], i+1, r.name, r.mean, diff);
        }
        
        System.out.println("\n💡 Interpretation:");
        System.out.println("   - Mean = average energy over " + NUM_RUNS + " runs");
        System.out.println("   - Lower is better");
        System.out.println("   - Winner is most energy-efficient ON AVERAGE");
    }
    
    static class StatResult {
        String name;
        double mean;
        StatResult(String name, double mean) {
            this.name = name;
            this.mean = mean;
        }
    }
}
