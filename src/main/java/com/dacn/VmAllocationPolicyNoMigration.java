package com.dacn;

import org.cloudsimplus.hosts.Host;
import org.cloudsimplus.hosts.HostSuitability;
import org.cloudsimplus.vms.Vm;
import org.cloudsimplus.allocationpolicies.VmAllocationPolicyAbstract;

import java.util.Comparator;
import java.util.List;
import java.util.Optional;

/**
 * No Migration Policy - BASELINE
 * Uses First Fit (no optimization) - VMs placed wherever they fit first
 * This represents the "do nothing" approach to compare against
 */
public class VmAllocationPolicyNoMigration extends VmAllocationPolicyAbstract {
    
    public VmAllocationPolicyNoMigration() {
        super();
    }
    
    @Override
    public HostSuitability allocateHostForVm(Vm vm, Host host) {
        return host.createVm(vm);
    }
    
    /**
     * Default First Fit - just place VM anywhere it fits
     * No optimization at all - represents baseline "do nothing"
     */
    @Override
    public Optional<Host> defaultFindHostForVm(Vm vm) {
        List<Host> hostList = getHostList();
        
        // First Fit: find first host that can accommodate the VM
        return hostList.stream()
            .filter(host -> host.isSuitableForVm(vm))
            .min(Comparator.comparingLong(h -> h.getId()));  // First available (by ID)
    }
    
    public static void resetCounters() {
        // No counters to reset - this is a simple policy
    }
}
