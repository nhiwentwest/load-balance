package com.dacn;

import org.cloudsimplus.hosts.Host;
import org.cloudsimplus.hosts.HostSuitability;
import org.cloudsimplus.vms.Vm;
import org.cloudsimplus.allocationpolicies.VmAllocationPolicyAbstract;
import org.cloudsimplus.resources.Pe;

import java.util.*;

/**
 * PEAP - Power Efficient Allocation Policy
 * 
 * Reference: Beloglazov et al. 2012 - "Energy-aware resource allocation heuristics 
 * for efficient management of data centers for cloud computing"
 * 
 * Algorithm:
 * 1. Find all feasible hosts (hosts that can fit the VM)
 * 2. Calculate power consumption for each host after placing VM
 * 3. Select host with MINIMUM power consumption (like Power-Aware)
 * 4. If multiple hosts have same power, use Most Full (Best Fit) as tie-breaker
 * 
 * This is different from FFD+Power because:
 * - No pre-sorting of VMs
 * - Pure power minimization, not based on First Fit Decreasing
 * - Considers entire host utilization, not just MIPS
 */
public class VmAllocationPolicyPEAP extends VmAllocationPolicyAbstract {

    // Power Model (Watts)
    private static final double HOST_MAX_POWER = 200;
    private static final double HOST_STATIC_POWER = 50;

    // Verification counters
    private static int placementCount = 0;
    private static int findHostCalls = 0;

    public VmAllocationPolicyPEAP() {
        super();
    }

    @Override
    public HostSuitability allocateHostForVm(Vm vm, Host host) {
        return host.createVm(vm);
    }

    /**
     * Main method - finds suitable host for VM using Power Efficient approach
     */
    @Override
    public Optional<Host> defaultFindHostForVm(Vm vm) {
        List<Host> hostList = getHostList();
        
        findHostCalls++;
        placementCount++; // FIX: increment before debug check so modulo is correct

        boolean debug = (placementCount % 5 == 0);

        if (debug) {
            System.out.println("\n=== PEAP Placement #" + placementCount + " ===");
            System.out.println("VM " + vm.getId() + " | MIPS: " + vm.getMips() + " | RAM: " + vm.getRam().getCapacity());
        }

        // Find all feasible hosts
        List<Host> feasibleHosts = new ArrayList<>();
        for (Host host : hostList) {
            if (host.isSuitableForVm(vm)) {
                feasibleHosts.add(host);
            }
        }
        
        if (feasibleHosts.isEmpty()) {
            if (debug) System.out.println("  >>> NO SUITABLE HOST FOUND!");
            return Optional.empty();
        }
        
        // PEAP: Select host with MINIMUM power consumption
        Host bestHost = selectBestPowerHost(vm, feasibleHosts);
        
        if (bestHost != null) {
            double powerBefore = calculateHostPower(bestHost);
            double powerAfter = calculateHostPowerAfterPlacement(bestHost, vm);
            
            if (debug) {
                System.out.println("  >>> SELECTED: Host " + bestHost.getId() + 
                    " | Power: " + String.format("%.2f", powerBefore) + "W -> " + 
                    String.format("%.2f", powerAfter) + "W");
            }
        } else {
            if (debug) System.out.println("  >>> FALLBACK: Host " + feasibleHosts.get(0).getId());
            bestHost = feasibleHosts.get(0);
        }
        
        return Optional.of(bestHost);
    }

    /**
     * Select host with minimum power consumption after VM placement
     * Tie-breaker: Most Full (Best Fit)
     */
    private Host selectBestPowerHost(Vm vm, List<Host> feasibleHosts) {
        Host bestHost = null;
        double minPower = Double.MAX_VALUE;
        double maxUtilization = -1;

        for (Host host : feasibleHosts) {
            double power = calculateHostPowerAfterPlacement(host, vm);
            double utilization = getHostUtilization(host);

            // FIX: Prefer hosts already active to consolidate; if idle (util=0), penalise by
            // treating its effective power as if static cost is included.
            double effectivePower = host.getVmList().isEmpty()
                ? power + HOST_STATIC_POWER * 0.4  // add penalty for waking a new host
                : power;

            if (effectivePower < minPower ||
                    (effectivePower == minPower && utilization > maxUtilization)) {
                minPower = effectivePower;
                maxUtilization = utilization;
                bestHost = host;
            }
        }

        return bestHost;
    }

    /**
     * Calculate host power after placing VM
     */
    private double calculateHostPowerAfterPlacement(Host host, Vm vm) {
        double currentLoad = getHostUtilization(host);
        double vmMips = vm.getMips();
        double hostTotalMips = host.getPeList().stream()
            .mapToDouble(Pe::getCapacity)
            .sum();
        double newLoad = currentLoad + (vmMips / hostTotalMips);
        
        return HOST_STATIC_POWER + (newLoad * (HOST_MAX_POWER - HOST_STATIC_POWER));
    }

    /**
     * Get host CPU utilization (0.0 to 1.0)
     */
    private double getHostUtilization(Host host) {
        if (host.getVmList().isEmpty()) {
            return 0.0;
        }
        
        double totalMips = host.getPeList().stream()
            .mapToDouble(Pe::getCapacity)
            .sum();
        double allocatedMips = host.getVmList().stream()
            .mapToDouble(vm -> vm.getMips())
            .sum();
        
        return Math.min(1.0, allocatedMips / totalMips);
    }

    /**
     * Calculate current host power consumption
     */
    private double calculateHostPower(Host host) {
        double load = getHostUtilization(host);
        return HOST_STATIC_POWER + (load * (HOST_MAX_POWER - HOST_STATIC_POWER));
    }

    /**
     * Print verification summary
     */
    public static void printVerificationSummary() {
        System.out.println("\n========== PEAP VERIFICATION ==========");
        System.out.println("Total findHost Calls: " + findHostCalls);
        System.out.println("Total VM Placements: " + placementCount);
        System.out.println("======================================\n");
    }
    
    public static void resetCounters() {
        placementCount = 0;
        findHostCalls = 0;
    }
}
