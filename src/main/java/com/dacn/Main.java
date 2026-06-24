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
import org.cloudsimplus.utilizationmodels.UtilizationModelPlanetLab;
import org.cloudsimplus.vms.Vm;
import org.cloudsimplus.vms.VmSimple;
import org.cloudsimplus.allocationpolicies.VmAllocationPolicySimple;
//import org.cloudsimplus.allocationpolicies.VmAllocationPolicyFfdPowerAware; // FFD + Power-Aware
import org.cloudsimplus.listeners.EventInfo;
import org.cloudsimplus.listeners.EventListener;

import java.io.File;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * FFD + Power-Aware VM Placement với PlanetLab Workload
 * 
 * Algorithm: First Fit Decreasing + Power-Aware
 * - Sort VMs by CPU utilization (descending)
 * - For each VM, select host with lowest power consumption
 */
public class Main {
    
    // Configuration
    private static final int NUM_HOSTS = 50;  // Giảm để dễ theo dõi
    private static final int HOST_MIPS = 1000; // MIPS per PE
    private static final int HOST_PES = 4;     // PEs per host
    private static final int HOST_RAM = 8192;  // MB
    private static final int HOST_BW = 100000; // Kbps
    
    private static final int NUM_VMS = 20;     // Giảm để dễ verify
    private static final int VM_MIPS = 250;     // MIPS per vCPU
    private static final int VM_PES = 1;        // vCPUs per VM
    private static final int VM_RAM = 512;      // MB
    private static final int VM_BW = 1000;      // Kbps
    
    private static final int NUM_CLOUDLETS = 20; // Number of cloudlets
    
    // Power Model (Watts) - typical server power consumption
    private static final double HOST_MAX_POWER = 200;  // Watts
    private static final double HOST_STATIC_POWER = 50; // Watts (idle)
    
    // PlanetLab data path
    private static final String PLANETLAB_PATH = "data/planetlab";
    
    // Energy tracking
    private static double totalEnergyConsumed = 0;
    private static double previousTime = 0;
    private static final double TIME_INTERVAL = 100.0; // Update every 100 seconds
    
    public static void main(String[] args) {
        System.out.println("=== Tabu Search VM Placement with PlanetLab ===");
        System.out.println("Hosts: " + NUM_HOSTS + ", VMs: " + NUM_VMS + ", Cloudlets: " + NUM_CLOUDLETS);
        
        // Create CloudSim Plus simulation
        CloudSimPlus simulation = new CloudSimPlus();
        
        // Create Datacenter
        Datacenter datacenter = createDatacenter(simulation);
        
        // Create Broker
        DatacenterBroker broker = new DatacenterBrokerSimple(simulation);
        
        // Create VMs
        List<Vm> vmList = createVms(NUM_VMS);
        
        // FFD: Sort VMs by MIPS (descending) before placement
        vmList.sort((v1, v2) -> Double.compare(v2.getMips(), v1.getMips()));
        System.out.println("\n=== FFD Sorting (by MIPS descending) ===");
        for (int i = 0; i < Math.min(5, vmList.size()); i++) {
            System.out.println("VM " + vmList.get(i).getId() + ": " + vmList.get(i).getMips() + " MIPS");
        }
        if (vmList.size() > 5) System.out.println("... and " + (vmList.size() - 5) + " more VMs");
        
        // Create Cloudlets với PlanetLab workload
        List<Cloudlet> cloudletList = createCloudletsWithPlanetLab(NUM_CLOUDLETS);
        
        // Add energy tracking listener - using correct API
        simulation.addOnClockTickListener(new EventListener<>() {
            @Override
            public void update(EventInfo info) {
                double currentTime = info.getTime();
                
                // Only update energy at fixed intervals to avoid excessive calculations
                if (currentTime - previousTime >= TIME_INTERVAL) {
                    double timeDelta = currentTime - previousTime;
                    
                    if (timeDelta > 0) {
                        // Calculate power for all hosts
                        double currentTotalPower = calculateTotalPower(datacenter);
                        
                    // Energy: Power (W) * time (hours) = Energy (Wh)
                    // Convert Wh to kWh by dividing by 1000
                    double energyDelta = (currentTotalPower * (timeDelta / 3600.0)) / 1000.0;
                    totalEnergyConsumed += energyDelta;
                    
                    previousTime = currentTime;
                    
                    System.out.printf("Time: %.0fs | Power: %.2f W | Energy: %.4f kWh%n", 
                        currentTime, currentTotalPower, totalEnergyConsumed);
                    }
                }
            }
        });
        
        // Submit to broker
        broker.submitVmList(vmList);
        broker.submitCloudletList(cloudletList);
        
        // Bind cloudlets to VMs (round-robin)
        for (int i = 0; i < cloudletList.size(); i++) {
            cloudletList.get(i).setVm(vmList.get(i % vmList.size()));
        }
        
        // Start simulation
        simulation.start();
        
        // Print results
        printResults(datacenter, vmList);
        
        // Print Tabu Search verification summary
        VmAllocationPolicyTabuSearch.printVerificationSummary();
    }
    
    /**
     * Calculate total power consumption of all hosts
     */
    private static double calculateTotalPower(Datacenter datacenter) {
        double totalPower = 0;
        
        for (Host host : datacenter.getHostList()) {
            if (!host.getVmList().isEmpty()) {
                // Calculate host utilization
                double totalMips = host.getPeList().stream()
                    .mapToDouble(Pe::getCapacity)
                    .sum();
                double allocatedMips = host.getVmList().stream()
                    .mapToDouble(vm -> vm.getMips())
                    .sum();
                double utilization = Math.min(1.0, allocatedMips / totalMips);
                
                // Linear power model: static + (max - static) * utilization
                double power = HOST_STATIC_POWER + (utilization * (HOST_MAX_POWER - HOST_STATIC_POWER));
                totalPower += power;
            }
        }
        
        return totalPower;
    }
    
    /**
     * Create Datacenter
     */
    private static Datacenter createDatacenter(CloudSimPlus simulation) {
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
        
        VmAllocationPolicyTabuSearch allocationPolicy = new VmAllocationPolicyTabuSearch();
        return new DatacenterSimple(simulation, hostList, allocationPolicy);
    }
    
    /**
     * Create VMs with varying MIPS to demonstrate FFD
     */
    private static List<Vm> createVms(int count) {
        List<Vm> vmList = new ArrayList<>();
        
        // Create VMs with varying MIPS to demonstrate FFD sorting
        int[] mipsValues = {100, 200, 300, 400, 500, 250, 350, 150, 450, 250};
        
        for (int i = 0; i < count; i++) {
            int mips = mipsValues[i % mipsValues.length];
            Vm vm = new VmSimple(mips, VM_PES);
            vm.setRam(VM_RAM).setBw(VM_BW).setSize(10000);
            vm.setCloudletScheduler(new CloudletSchedulerSpaceShared());
            vmList.add(vm);
        }
        
        return vmList;
    }
    
    /**
     * Create Cloudlets với PlanetLab workload
     */
    private static List<Cloudlet> createCloudletsWithPlanetLab(int count) {
        List<Cloudlet> cloudletList = new ArrayList<>();
        
        File planetlabDir = new File(PLANETLAB_PATH);
        
        // Find all planetlab date directories
        List<File> dateDirs = new ArrayList<>();
        if (planetlabDir.exists() && planetlabDir.isDirectory()) {
            File[] files = planetlabDir.listFiles();
            if (files != null) {
                for (File f : files) {
                    if (f.isDirectory() && f.getName().matches("2011\\d{4}")) {
                        dateDirs.add(f);
                    }
                }
            }
        }
        
        if (dateDirs.isEmpty()) {
            System.err.println("Warning: PlanetLab data not found, using synthetic workload");
            return createCloudletsSynthetic(count);
        }
        
        // Find all VM trace files
        List<File> vmFiles = new ArrayList<>();
        for (File dateDir : dateDirs) {
            File[] files = dateDir.listFiles();
            if (files != null) {
                for (File f : files) {
                    if (f.isFile() && !f.getName().startsWith(".")) {
                        vmFiles.add(f);
                    }
                }
            }
        }
        
        if (vmFiles.isEmpty()) {
            System.err.println("Warning: No PlanetLab trace files found, using synthetic workload");
            return createCloudletsSynthetic(count);
        }
        
        System.out.println("Found " + vmFiles.size() + " PlanetLab trace files");
        
        for (int i = 0; i < count; i++) {
            Cloudlet cloudlet = new CloudletSimple(1000, 1); // 1000 MI instead of 10000
            
            // Use PlanetLab utilization model
            String vmFile = vmFiles.get(i % vmFiles.size()).getAbsolutePath();
            UtilizationModelPlanetLab planetLab = new UtilizationModelPlanetLab(vmFile, 300);
            cloudlet.setUtilizationModelCpu(planetLab);
            
            cloudletList.add(cloudlet);
        }
        
        return cloudletList;
    }
    
    /**
     * Fallback: Create Cloudlets with synthetic workload
     */
    private static List<Cloudlet> createCloudletsSynthetic(int count) {
        List<Cloudlet> cloudletList = new ArrayList<>();
        
        for (int i = 0; i < count; i++) {
            Cloudlet cloudlet = new CloudletSimple(10000, 1);
            double randomUtil = 0.1 + Math.random() * 0.7;
            cloudlet.setUtilizationModelCpu(new UtilizationModelDynamic(randomUtil));
            cloudletList.add(cloudlet);
        }
        
        return cloudletList;
    }
    
    /**
     * Print simulation results
     */
    private static void printResults(Datacenter datacenter, List<Vm> vmList) {
        System.out.println("\n=== Simulation Results ===");
        
        int activeHosts = 0;
        double finalPower = calculateTotalPower(datacenter);
        
        for (Host host : datacenter.getHostList()) {
            if (!host.getVmList().isEmpty()) {
                activeHosts++;
            }
        }
        
        System.out.println("Total Energy Consumed: " + String.format("%.4f", totalEnergyConsumed) + " kWh");
        System.out.println("Final Power: " + String.format("%.2f", finalPower) + " W");
        System.out.println("Active Hosts: " + activeHosts + "/" + NUM_HOSTS);
        System.out.println("VMs Placed: " + vmList.stream().filter(vm -> vm.getHost() != null).count());
    }
}
