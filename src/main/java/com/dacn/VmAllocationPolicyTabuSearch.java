package com.dacn;

import org.cloudsimplus.hosts.Host;
import org.cloudsimplus.hosts.HostSuitability;
import org.cloudsimplus.vms.Vm;
import org.cloudsimplus.allocationpolicies.VmAllocationPolicyAbstract;
import org.cloudsimplus.resources.Pe;

import java.util.*;

/**
 * Tabu Search VM Allocation Policy
 * 
 * Algorithm:
 * 1. Start with initial solution (FFD sorted VMs)
 * 2. Generate neighbor solutions by moving VMs to different hosts
 * 3. Select best non-tabu move (or satisfy aspiration criterion)
 * 4. Update tabu list
 * 5. Repeat until convergence
 * 
 * Reference: Glover 1989 + Koubaa 2024
 */
public class VmAllocationPolicyTabuSearch extends VmAllocationPolicyAbstract {

    // Tabu Search parameters
    private static final int TABU_TENURE = 5;         // How long a move stays tabu
    private static final int MAX_ITERATIONS = 50;    // Max iterations
    private static final int NEIGHBOR_COUNT = 10;     // Number of neighbors to evaluate
    
    // Power Model (Watts)
    private static final double HOST_MAX_POWER = 200;
    private static final double HOST_STATIC_POWER = 50;

    // Verification counters
    private static int placementCount = 0;
    private static int tabuSearchCalls = 0;

    public VmAllocationPolicyTabuSearch() {
        super();
    }

    @Override
    public HostSuitability allocateHostForVm(Vm vm, Host host) {
        return host.createVm(vm);
    }

    /**
     * Main method - finds suitable host for VM using Tabu Search
     */
    @Override
    public Optional<Host> defaultFindHostForVm(Vm vm) {
        List<Host> hostList = getHostList();
        
        tabuSearchCalls++;
        placementCount++;
        
        System.out.println("\n=== Tabu Search Placement #" + placementCount + " ===");
        System.out.println("VM " + vm.getId() + " | MIPS: " + vm.getMips() + " | RAM: " + vm.getRam().getCapacity());
        
        // Get all feasible hosts
        List<Host> feasibleHosts = new ArrayList<>();
        for (Host host : hostList) {
            if (host.isSuitableForVm(vm)) {
                feasibleHosts.add(host);
            }
        }
        
        if (feasibleHosts.isEmpty()) {
            System.out.println("  >>> NO SUITABLE HOST FOUND!");
            return Optional.empty();
        }
        
        // Tabu Search: select best host based on power optimization
        Host bestHost = tabuSearch(vm, feasibleHosts);
        
        if (bestHost != null) {
            double powerBefore = calculateHostPower(bestHost);
            double powerAfter = calculateHostPowerAfterPlacement(bestHost, vm);
            
            System.out.println("  >>> SELECTED: Host " + bestHost.getId() + 
                " | Power: " + String.format("%.2f", powerBefore) + "W -> " + 
                String.format("%.2f", powerAfter) + "W");
        } else {
            // Fallback to first feasible host
            bestHost = feasibleHosts.get(0);
            System.out.println("  >>> FALLBACK: Host " + bestHost.getId());
        }
        
        return Optional.of(bestHost);
    }

    /**
     * Tabu Search optimization for VM placement
     */
    private Host tabuSearch(Vm vm, List<Host> feasibleHosts) {
        // Initialize tabu list
        List<String> tabuList = new ArrayList<>();
        int iteration = 0;
        
        // Current best solution
        Host currentBest = null;
        double currentBestPower = Double.MAX_VALUE;
        
        // Evaluate all feasible hosts and find initial solution
        for (Host host : feasibleHosts) {
            double power = calculateHostPowerAfterPlacement(host, vm);
            if (power < currentBestPower) {
                currentBestPower = power;
                currentBest = host;
            }
        }
        
        if (currentBest == null) return null;
        
        Host bestSolution = currentBest;
        double bestSolutionPower = currentBestPower;
        
        System.out.println("  Initial solution: Host " + currentBest.getId() + 
            " (Power: " + String.format("%.2f", currentBestPower) + "W)");
        
        // Tabu Search iterations
        while (iteration < MAX_ITERATIONS) {
            iteration++;
            
            // Generate neighbors (random subset for efficiency)
            List<Host> neighbors = generateNeighbors(feasibleHosts, currentBest);
            
            Host bestNeighbor = null;
            double bestNeighborPower = Double.MAX_VALUE;
            
            // Evaluate neighbors
            for (Host neighbor : neighbors) {
                String moveKey = vm.getId() + "->" + neighbor.getId();
                
                // Check if move is tabu
                boolean isTabu = tabuList.contains(moveKey);
                
                // Calculate power for this placement
                double power = calculateHostPowerAfterPlacement(neighbor, vm);
                
                // Aspiration: accept tabu move if it's better than best known
                boolean acceptAsAspiration = !isTabu || (power < bestSolutionPower);
                
                if ((!isTabu || acceptAsAspiration) && power < bestNeighborPower) {
                    bestNeighborPower = power;
                    bestNeighbor = neighbor;
                }
            }
            
                // Update current solution
                if (bestNeighbor != null) {
                    currentBest = bestNeighbor;
                    currentBestPower = bestNeighborPower;

                    // Update global best solution
                    if (currentBestPower < bestSolutionPower) {
                        bestSolution = currentBest;
                        bestSolutionPower = currentBestPower;
                    }

                    // Add move to tabu list
                    tabuList.add(vm.getId() + "->" + bestNeighbor.getId());
                    if (tabuList.size() > TABU_TENURE) {
                        tabuList.remove(0);
                    }
                    // FIX: Only log every 10 iterations to reduce noise
                    if (iteration % 10 == 0) {
                        System.out.println("  Iter " + iteration + ": Host " + bestNeighbor.getId()
                            + " (Power: " + String.format("%.2f", bestNeighborPower) + "W)");
                    }
                }
        }
        
        System.out.println("  Final: Host " + bestSolution.getId() + 
            " (Best Power: " + String.format("%.2f", bestSolutionPower) + "W) " +
            "[Iterations: " + iteration + "]");
        
        return bestSolution;
    }

    /**
     * Generate neighbor solutions
     */
    private List<Host> generateNeighbors(List<Host> feasibleHosts, Host currentHost) {
        List<Host> neighbors = new ArrayList<>();
        neighbors.add(currentHost);

        // FIX: Shuffle a COPY, not the original list — original order must stay stable
        List<Host> shuffled = new ArrayList<>(feasibleHosts);
        Collections.shuffle(shuffled);
        int count = Math.min(NEIGHBOR_COUNT, shuffled.size());

        for (int i = 0; i < count; i++) {
            Host h = shuffled.get(i);
            if (!neighbors.contains(h)) {
                neighbors.add(h);
            }
        }

        return neighbors;
    }

    /**
     * Calculate host power after placing VM
     */
    private double calculateHostPowerAfterPlacement(Host host, Vm vm) {
        double currentLoad = getHostLoad(host);
        double vmMips = vm.getMips();
        double hostTotalMips = host.getPeList().stream()
            .mapToDouble(Pe::getCapacity)
            .sum();
        double newLoad = currentLoad + (vmMips / hostTotalMips);
        
        return HOST_STATIC_POWER + (newLoad * (HOST_MAX_POWER - HOST_STATIC_POWER));
    }

    /**
     * Get host CPU load (0.0 to 1.0)
     */
    private double getHostLoad(Host host) {
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
        double load = getHostLoad(host);
        return HOST_STATIC_POWER + (load * (HOST_MAX_POWER - HOST_STATIC_POWER));
    }

    /**
     * Print verification summary
     */
    public static void printVerificationSummary() {
        System.out.println("\n========== TABU SEARCH VERIFICATION ==========");
        System.out.println("Total Tabu Search Calls: " + tabuSearchCalls);
        System.out.println("Total VM Placements: " + placementCount);
        System.out.println("================================================\n");
    }
    
    public static void resetCounters() {
        placementCount = 0;
        tabuSearchCalls = 0;
    }
}
