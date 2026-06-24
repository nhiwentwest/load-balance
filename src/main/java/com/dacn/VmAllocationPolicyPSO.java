package com.dacn;

import org.cloudsimplus.hosts.Host;
import org.cloudsimplus.hosts.HostSuitability;
import org.cloudsimplus.vms.Vm;
import org.cloudsimplus.allocationpolicies.VmAllocationPolicyAbstract;
import org.cloudsimplus.resources.Pe;

import java.util.*;

/**
 * PSO - Particle Swarm Optimization for VM Placement
 * 
 * Reference: Kennedy & Eberhart 1995 - "Particle Swarm Optimization"
 *           & Liu et al. 2016 - "Virtual machine placement algorithm for 
 *           cloud computing using particle swarm optimization"
 * 
 * Algorithm:
 * 1. Initialize particles (each particle = one host assignment)
 * 2. Each particle has position and velocity
 * 3. Evaluate fitness (power consumption) for each particle
 * 4. Update personal best (pBest) for each particle
 * 5. Update global best (gBest) across all particles
 * 6. Update velocity and position based on pBest and gBest
 * 7. Repeat until convergence
 * 8. Use gBest for actual VM placement
 * 
 * Parameters:
 * - NUM_PARTICLES: Number of particles in swarm
 * - ITERATIONS: Number of PSO iterations
 * - W: Inertia weight (exploration vs exploitation)
 * - C1: Cognitive coefficient (personal best)
 * - C2: Social coefficient (global best)
 */
public class VmAllocationPolicyPSO extends VmAllocationPolicyAbstract {

    // PSO Parameters — tuned for energy-efficient consolidation
    private static final int NUM_PARTICLES = 30;    // More particles → better coverage
    private static final int ITERATIONS = 50;       // More iterations → better convergence
    private static final double W_MAX = 0.9;        // Inertia weight start (exploration)
    private static final double W_MIN = 0.4;        // Inertia weight end  (exploitation)
    private static final double C1 = 2.05;          // Cognitive (personal best pull)
    private static final double C2 = 2.05;          // Social    (global  best pull)
    private static final double CONSOLIDATION_PENALTY = 0.5; // Penalty per extra active host

    // Power Model (Watts)
    private static final double HOST_MAX_POWER = 200;
    private static final double HOST_STATIC_POWER = 50;

    // Verification counters
    private static int placementCount = 0;
    private static int psoSearchCalls = 0;

    public VmAllocationPolicyPSO() {
        super();
    }

    @Override
    public HostSuitability allocateHostForVm(Vm vm, Host host) {
        return host.createVm(vm);
    }

    /**
     * Main method - finds suitable host using PSO
     */
    @Override
    public Optional<Host> defaultFindHostForVm(Vm vm) {
        List<Host> hostList = getHostList();
        
        psoSearchCalls++;
        placementCount++;
        
        // Debug output
        boolean debug = (placementCount % 5 == 0);
        
        if (debug) {
            System.out.println("\n=== PSO Placement #" + placementCount + " ===");
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
        
        // PSO optimization
        Host bestHost = psoSearch(vm, feasibleHosts, debug);
        
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
     * PSO optimization for VM placement
     */
    private Host psoSearch(Vm vm, List<Host> feasibleHosts, boolean debug) {
        int n = feasibleHosts.size();
        int numParticles = Math.min(NUM_PARTICLES, n);
        Random rand = new Random();

        // FIX 1: Initialize particles with random spread across all feasible hosts
        List<Particle> particles = new ArrayList<>();
        for (int i = 0; i < numParticles; i++) {
            int idx = (n > numParticles) ? rand.nextInt(n) : (i % n);
            particles.add(new Particle(feasibleHosts.get(idx), idx));
        }

        // Global best
        int gBestIdx = 0;
        double gBestFitness = Double.MAX_VALUE;
        Host gBestHost = feasibleHosts.get(0);

        // PSO iterations with linear W decay
        for (int iter = 0; iter < ITERATIONS; iter++) {
            // FIX 2: W decays linearly from W_MAX → W_MIN over iterations
            double w = W_MAX - (W_MAX - W_MIN) * iter / ITERATIONS;

            // Evaluate fitness for each particle
            for (Particle p : particles) {
                // FIX 3: Multi-objective fitness = host power + consolidation bonus
                p.fitness = calculateFitness(p.position, vm, feasibleHosts);

                // Update personal best
                if (p.fitness < p.pBestFitness) {
                    p.pBestPositionIndex = p.positionIndex;
                    p.pBestFitness = p.fitness;
                }

                // Update global best
                if (p.fitness < gBestFitness) {
                    gBestFitness = p.fitness;
                    gBestIdx = p.positionIndex;
                    gBestHost = p.position;
                }
            }

            // Update velocity and position
            for (Particle p : particles) {
                double r1 = rand.nextDouble();
                double r2 = rand.nextDouble();

                // Standard PSO velocity update with decaying inertia
                p.velocity = w * p.velocity
                    + C1 * r1 * (p.pBestPositionIndex - p.positionIndex)
                    + C2 * r2 * (gBestIdx - p.positionIndex);

                // Clamp velocity to [-n/2, n/2] to avoid over-shooting
                double vMax = Math.max(1, n / 2.0);
                p.velocity = Math.max(-vMax, Math.min(vMax, p.velocity));

                // Update position
                p.positionIndex = (int) Math.round(p.positionIndex + p.velocity);
                p.positionIndex = Math.max(0, Math.min(n - 1, p.positionIndex));
                p.position = feasibleHosts.get(p.positionIndex);
            }

            if (debug && iter % 10 == 0) {
                System.out.println("  PSO Iter " + iter + " (w=" + String.format("%.2f", w)
                    + "): Best Fitness = " + String.format("%.2f", gBestFitness));
            }
        }

        if (debug) {
            System.out.println("  Final PSO: Host " + gBestHost.getId()
                + " (Fitness: " + String.format("%.2f", gBestFitness) + ")");
        }

        return gBestHost;
    }

    /**
     * FIX 3: Multi-objective fitness function.
     * Minimise: hostPowerAfterPlacement + penalty * (number of currently idle hosts opened)
     * This encourages packing VMs onto already-active hosts (consolidation) before
     * waking up new ones.
     */
    private double calculateFitness(Host host, Vm vm, List<Host> allFeasibleHosts) {
        double power = calculateHostPowerAfterPlacement(host, vm);

        // Count how many currently-idle hosts would be activated if we pick this host
        // (proxy: if this host has no VMs, placing here activates it → penalty)
        double activeHostPenalty = 0;
        if (host.getVmList().isEmpty()) {
            // Opening a new host wastes static power; penalise proportionally
            activeHostPenalty = HOST_STATIC_POWER * CONSOLIDATION_PENALTY;
        }

        return power + activeHostPenalty;
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
        System.out.println("\n========== PSO VERIFICATION ==========");
        System.out.println("Total PSO Search Calls: " + psoSearchCalls);
        System.out.println("Total VM Placements: " + placementCount);
        System.out.println("======================================\n");
    }
    
    public static void resetCounters() {
        placementCount = 0;
        psoSearchCalls = 0;
    }
    
    /**
     * Inner class for PSO particle
     */
    private static class Particle {
        Host position;
        int positionIndex;
        double velocity;
        double fitness;
        Host pBestPosition;    // Personal best position
        int pBestPositionIndex;
        double pBestFitness;   // Personal best fitness
        
        Particle(Host position, int index) {
            this.position = position;
            // FIX 4: Assign the real index so velocity arithmetic is correct
            this.positionIndex = index;
            this.velocity = (Math.random() * 2 - 1); // Small random initial velocity
            this.fitness = Double.MAX_VALUE;
            this.pBestPosition = position;
            // FIX 4: pBest starts at real index, not 0
            this.pBestPositionIndex = index;
            this.pBestFitness = Double.MAX_VALUE;
        }
    }
}
