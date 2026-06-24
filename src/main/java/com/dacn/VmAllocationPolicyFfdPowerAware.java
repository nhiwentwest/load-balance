package com.dacn;

import org.cloudsimplus.hosts.Host;
import org.cloudsimplus.hosts.HostSuitability;
import org.cloudsimplus.vms.Vm;
import org.cloudsimplus.allocationpolicies.VmAllocationPolicyAbstract;
import org.cloudsimplus.resources.Pe;

import java.util.Comparator;
import java.util.List;
import java.util.Optional;

/**
 * FFD + Power-Aware VM Allocation Policy
 * 
 * Algorithm:
 * 1. Sort VMs by CPU MIPS (descending) - First Fit Decreasing
 * 2. For each VM, select host with lowest power consumption that can fit it
 * 
 * Verification:
 * - Logs each VM placement with power consumption details
 * - Shows host selection based on minimum power
 */
public class VmAllocationPolicyFfdPowerAware extends VmAllocationPolicyAbstract {

    // Power Model (Watts)
    private static final double HOST_MAX_POWER = 200;  // Watts
    private static final double HOST_STATIC_POWER = 50; // Watts (idle)

    // Verification counters
    private static int placementCount = 0;
    private static double totalPowerBefore = 0;
    private static double totalPowerAfter = 0;

    public VmAllocationPolicyFfdPowerAware() {
        super();
    }

    @Override
    public HostSuitability allocateHostForVm(Vm vm, Host host) {
        return host.createVm(vm);
    }

    /**
     * Main method - finds suitable host for VM using FFD + Power-Aware
     */
    @Override
    public Optional<Host> defaultFindHostForVm(Vm vm) {
        List<Host> hostList = getHostList();
        
        Host bestHost = null;
        double minPower = Double.MAX_VALUE;
        double currentSystemPower = 0;
        
        // Calculate current system power before placement
        for (Host h : hostList) {
            currentSystemPower += calculateHostPower(h);
        }
        
        placementCount++;
        System.out.println("\n=== FFD+Power-Aware Placement #" + placementCount + " ===");
        System.out.println("VM " + vm.getId() + " | MIPS: " + vm.getMips() + " | RAM: " + vm.getRam().getCapacity());
        
        // Find host with minimum power consumption
        for (Host host : hostList) {
            // Check if host can fit the VM
            if (host.isSuitableForVm(vm)) {
                // Calculate estimated power if VM placed here
                double currentLoad = getHostLoad(host);
                double vmMips = vm.getMips();
                double hostTotalMips = host.getPeList().stream()
                    .mapToDouble(Pe::getCapacity)
                    .sum();
                double newLoad = currentLoad + (vmMips / hostTotalMips);
                double estimatedPower = HOST_STATIC_POWER + 
                    (newLoad * (HOST_MAX_POWER - HOST_STATIC_POWER));
                
                System.out.printf("  Host %d: Load=%.2f%%, Est.Power=%.2fW | %s%n", 
                    host.getId(), 
                    currentLoad * 100, 
                    estimatedPower,
                    host.getVmList().isEmpty() ? "[EMPTY]" : "VMs=" + host.getVmList().size());
                
                if (estimatedPower < minPower) {
                    minPower = estimatedPower;
                    bestHost = host;
                }
            }
        }
        
        if (bestHost != null) {
            double powerBefore = calculateHostPower(bestHost);
            double newLoad = getHostLoad(bestHost) + (vm.getMips() / bestHost.getPeList().stream().mapToDouble(Pe::getCapacity).sum());
            double powerAfter = HOST_STATIC_POWER + (newLoad * (HOST_MAX_POWER - HOST_STATIC_POWER));
            
            System.out.println("  >>> SELECTED: Host " + bestHost.getId() + " | Power: " + 
                String.format("%.2f", powerBefore) + "W -> " + String.format("%.2f", powerAfter) + "W");
            
            totalPowerBefore += powerBefore;
            totalPowerAfter += powerAfter;
        } else {
            System.out.println("  >>> NO SUITABLE HOST FOUND!");
        }
        
        return Optional.ofNullable(bestHost);
    }

    /**
     * Get host CPU load (0.0 to 1.0)
     */
    private double getHostLoad(Host host) {
        if (host.getVmList().isEmpty()) {
            return 0.0;
        }
        
        // Calculate based on allocated MIPS vs total capacity
        double totalMips = host.getPeList().stream()
            .mapToDouble(Pe::getCapacity)
            .sum();
        double allocatedMips = host.getVmList().stream()
            .mapToDouble(vm -> vm.getMips())
            .sum();
        
        return Math.min(1.0, allocatedMips / totalMips);
    }

    /**
     * Calculate host power consumption
     */
    private double calculateHostPower(Host host) {
        double load = getHostLoad(host);
        return HOST_STATIC_POWER + (load * (HOST_MAX_POWER - HOST_STATIC_POWER));
    }

    /**
     * Print verification summary
     */
    public static void printVerificationSummary() {
        System.out.println("\n========== VERIFICATION SUMMARY ==========");
        System.out.println("Total VM Placements: " + placementCount);
        System.out.println("Total Power (before all placements): " + String.format("%.2f", totalPowerBefore) + "W");
        System.out.println("Total Power (after all placements): " + String.format("%.2f", totalPowerAfter) + "W");
        System.out.println("==========================================\n");
    }
    
    public static void resetCounters() {
        placementCount = 0;
        totalPowerBefore = 0;
        totalPowerAfter = 0;
    }
}
