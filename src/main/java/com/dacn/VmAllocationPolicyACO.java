package com.dacn;

import org.cloudsimplus.hosts.Host;
import org.cloudsimplus.hosts.HostSuitability;
import org.cloudsimplus.vms.Vm;
import org.cloudsimplus.allocationpolicies.VmAllocationPolicyAbstract;
import org.cloudsimplus.resources.Pe;

import java.util.*;

/**
 * ACO - Ant Colony Optimization for VM Placement
 * 
 * Reference: Dorigo 1992 - "Optimization, Learning and Natural Algorithms"
 *           & Wang et al. 2015 - "Ant colony optimization for virtual machine 
 *           placement in cloud computing"
 * 
 * Algorithm:
 * 1. Initialize pheromone trails on all host-VM pairs
 * 2. Multiple ants build solutions (each ant = one placement strategy)
 * 3. Evaluate solution quality (power consumption)
 * 4. Update pheromones: more pheromone on better solutions
 * 5. Repeat until convergence
 * 6. Use best solution found for actual VM placement
 * 
 * Parameters:
 * - NUM_ANTS: Number of ants (solutions per iteration)
 * - ITERATIONS: Number of ACO iterations
 * - ALPHA: Pheromone importance
 * - BETA: Heuristic importance (power efficiency)
 * - EVAPORATION: Pheromone evaporation rate
 */
public class VmAllocationPolicyACO extends VmAllocationPolicyAbstract {

    // ACO Parameters
    private static final int NUM_ANTS = 20;         // More ants → better exploration
    private static final int ITERATIONS = 30;       // More iterations
    private static final double ALPHA = 1.0;        // Pheromone importance
    private static final double BETA = 3.0;         // Heuristic importance (raised)
    private static final double EVAPORATION = 0.3; // Lower evaporation → info retained longer
    private static final double Q0 = 1.0;           // Pheromone deposit factor
    private static final double CONSOLIDATION_BONUS = 1.5; // Bonus multiplier for active hosts

    // Power Model (Watts)
    private static final double HOST_MAX_POWER = 200;
    private static final double HOST_STATIC_POWER = 50;

    // Verification counters
    private static int placementCount = 0;
    private static int acoSearchCalls = 0;

    public VmAllocationPolicyACO() {
        super();
    }

    @Override
    public HostSuitability allocateHostForVm(Vm vm, Host host) {
        return host.createVm(vm);
    }

    /**
     * Main method - finds suitable host using ACO
     */
    @Override
    public Optional<Host> defaultFindHostForVm(Vm vm) {
        List<Host> hostList = getHostList();
        
        acoSearchCalls++;
        placementCount++;
        
        // Debug output
        boolean debug = (placementCount % 5 == 0);
        
        if (debug) {
            System.out.println("\n=== ACO Placement #" + placementCount + " ===");
            System.out.println("VM " + vm.getId() + " | MIPS: " + vm.getMips() + " | RAM: " + vm.getRam().getCapacity());
        }
        
        // Get all feasible hosts
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
        
        // ACO optimization
        Host bestHost = acoSearch(vm, feasibleHosts, debug);
        
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
     * ACO optimization for VM placement
     */
    private Host acoSearch(Vm vm, List<Host> feasibleHosts, boolean debug) {
        int n = feasibleHosts.size();
        
        // Initialize pheromone matrix
        double[][] pheromone = new double[n][1];
        for (int i = 0; i < n; i++) {
            pheromone[i][0] = 1.0; // Initial pheromone
        }
        
        Host bestSolution = feasibleHosts.get(0);
        double bestPower = Double.MAX_VALUE;
        
        // ACO iterations
        for (int iter = 0; iter < ITERATIONS; iter++) {
            // Build solutions for each ant
            List<AntSolution> antSolutions = new ArrayList<>();
            
            for (int ant = 0; ant < NUM_ANTS; ant++) {
                AntSolution solution = buildAntSolution(vm, feasibleHosts, pheromone, n);
                antSolutions.add(solution);
            }
            
            // Evaluate solutions using consolidated fitness (power + consolidation)
            for (AntSolution sol : antSolutions) {
                double power = calculateHostPowerAfterPlacement(sol.host, vm);
                // Consolidation: prefer hosts already active (lower effective cost)
                double fitness = sol.host.getVmList().isEmpty()
                    ? power + HOST_STATIC_POWER * 0.5   // penalty for opening new host
                    : power;                             // no penalty if already active
                sol.power = fitness;

                if (fitness < bestPower) {
                    bestPower = fitness;
                    bestSolution = sol.host;
                }
            }
            
            // FIX: pass feasibleHosts so index mapping is correct
            updatePheromones(pheromone, antSolutions, feasibleHosts);
            
            if (debug && iter % 5 == 0) {
                System.out.println("  ACO Iter " + iter + ": Best Power = " + 
                    String.format("%.2f", bestPower) + "W");
            }
        }
        
        if (debug) {
            System.out.println("  Final ACO: Host " + bestSolution.getId() + 
                " (Power: " + String.format("%.2f", bestPower) + "W)");
        }
        
        return bestSolution;
    }

    /**
     * Build solution for one ant using probabilistic selection
     */
    private AntSolution buildAntSolution(Vm vm, List<Host> hosts, double[][] pheromone, int n) {
        double[] probabilities = new double[n];
        double totalProb = 0;

        for (int i = 0; i < n; i++) {
            Host host = hosts.get(i);
            double power = calculateHostPowerAfterPlacement(host, vm);

            // FIX: Heuristic favours already-active hosts (consolidation)
            double heuristic = 1.0 / (power + 1);
            if (!host.getVmList().isEmpty()) {
                heuristic *= CONSOLIDATION_BONUS; // boost probability of active hosts
            }

            probabilities[i] = Math.pow(pheromone[i][0], ALPHA) * Math.pow(heuristic, BETA);
            totalProb += probabilities[i];
        }

        // Roulette wheel selection
        double random = Math.random() * totalProb;
        double cumulative = 0;
        int selectedIndex = n - 1; // default to last if rounding error

        for (int i = 0; i < n; i++) {
            cumulative += probabilities[i];
            if (random <= cumulative) {
                selectedIndex = i;
                break;
            }
        }

        return new AntSolution(hosts.get(selectedIndex), 0);
    }

    /**
     * Update pheromone matrix based on solution quality
     */
    /**
     * FIX: Update pheromone using the HOST INDEX in feasibleHosts, not the ant solution list index.
     * Previously the code was using i from solutions.size() as the pheromone index,
     * which is completely wrong when solutions.size() != feasibleHosts.size().
     */
    private void updatePheromones(double[][] pheromone, List<AntSolution> solutions,
                                   List<Host> feasibleHosts) {
        int n = feasibleHosts.size();
        // Evaporation
        for (int i = 0; i < n; i++) {
            pheromone[i][0] *= (1 - EVAPORATION);
        }

        // Deposit pheromone on the correct host index
        for (AntSolution sol : solutions) {
            for (int i = 0; i < n; i++) {
                if (feasibleHosts.get(i).getId() == sol.host.getId()) {
                    double deposit = Q0 / (sol.power + 1);
                    pheromone[i][0] += deposit;
                    break;
                }
            }
        }
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
     * Get host CPU utilization
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
        System.out.println("\n========== ACO VERIFICATION ==========");
        System.out.println("Total ACO Search Calls: " + acoSearchCalls);
        System.out.println("Total VM Placements: " + placementCount);
        System.out.println("======================================\n");
    }
    
    public static void resetCounters() {
        placementCount = 0;
        acoSearchCalls = 0;
    }
    
    /**
     * Inner class for ant solution
     */
    private static class AntSolution {
        Host host;
        double power;
        
        AntSolution(Host host, double power) {
            this.host = host;
            this.power = power;
        }
    }
}
