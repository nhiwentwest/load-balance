package com.dacn.advanced;

import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.Logger;
import org.cloudsimplus.brokers.DatacenterBroker;
import org.cloudsimplus.brokers.DatacenterBrokerSimple;
import org.cloudsimplus.cloudlets.Cloudlet;
import org.cloudsimplus.cloudlets.CloudletSimple;
import org.cloudsimplus.core.CloudSimPlus;
import org.cloudsimplus.datacenters.Datacenter;
import org.cloudsimplus.datacenters.DatacenterSimple;
import org.cloudsimplus.hosts.Host;
import org.cloudsimplus.hosts.HostSimple;
import org.cloudsimplus.power.models.PowerModelHostSimple;
import org.cloudsimplus.resources.Pe;
import org.cloudsimplus.resources.PeSimple;
import org.cloudsimplus.schedulers.cloudlet.CloudletSchedulerTimeShared;
import org.cloudsimplus.schedulers.vm.VmSchedulerTimeShared;
import org.cloudsimplus.utilizationmodels.UtilizationModelDynamic;
import org.cloudsimplus.vms.Vm;
import org.cloudsimplus.vms.VmSimple;
import org.slf4j.LoggerFactory;
import py4j.GatewayServer;

import java.io.File;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Py4J Gateway cho Multi-Agent RL + Autoformer.
 * V3: Real migrations, SPECpower model, proper SLATAH/PDM, long-running cloudlets.
 */
public class Py4jBridge {

    // --- Simulation Config ---
    private static final double INTERVAL = 300.0;
    private static final double MAX_TIME = 86400.0;
    // Cluster size is configurable via NUM_HOSTS env var so the same bridge can
    // run the 20-host baseline or the 100-host scaled experiment. VM count and
    // all per-host arrays derive from this at construction time.
    private static final int NUM_HOSTS = readNumHosts();
    private static int readNumHosts() {
        String v = System.getenv("NUM_HOSTS");
        if (v != null && !v.trim().isEmpty()) {
            try { return Math.max(2, Integer.parseInt(v.trim())); }
            catch (NumberFormatException ignored) {}
        }
        return 20;
    }
    private static final int HOST_PES = 2;              // Beloglazov Table 5: HP G5
    private static final int HOST_MIPS = 2660;           // HP G5: 2660 MIPS/PE (Xeon 3075)
    private static final int HOST_RAM = 16384;           // 16GB
    private static final double HOST_GPU = 8.0;           // Logical GPU capacity per host
    private static final long HOST_BW = 1000;            // 1 Gbps = 1000 Mbit/s
    private static final long HOST_STORAGE = 1000000;
    // SPECpower-based: HP ProLiant ML110 G5 (Beloglazov 2012, Table 5)
    private static final double HOST_MAX_POWER = 188.0;  // Watts at 100% util
    private static final double HOST_IDLE_POWER = 93.7;   // Watts at 0% util
    private static final int TOP_K = 10;
    private static final int GLOBAL_STATE_DIM = 8;    // Expanded state
    private static final int VM_EXTRA_FEATURES = 8;    // VM/source features after candidate blocks
    private static final int VM_STATE_DIM = TOP_K * 3 + VM_EXTRA_FEATURES; // CPU, power, GPU + VM features
    private static final int DEFAULT_HISTORY_LEN = 20;
    private static final int[][] VM_TYPES = {
        {500,  613,  1},
        {1000, 1740, 1},
        {1500, 1740, 1},
        {2000, 1740, 1},
        {2500, 870,  1},
    };

    // --- Runtime ---
    private CloudSimPlus simulation;
    private Datacenter datacenter;
    private DatacenterBroker broker;
    private List<Host> hostList;
    private List<Vm> vmList;
    private Thread simThread;
    private Map<Long, Double> vmGpuRequests = new HashMap<>();

    // Pause/resume sync
    private volatile boolean simPaused = false;
    private volatile boolean simDone = false;
    private double nextPauseTime;

    // Metrics per interval
    private double prevEnergy = 0;
    private double currentEnergy = 0;
    private int intervalMigrations = 0;
    private List<String> lastStepMigrations = new ArrayList<>();
    private int totalMigrations = 0;
    private int invariantWarnings = 0;

    // History & SLA metrics tracking (Beloglazov-accurate)
    private double[][] hostCpuHistoryArr;
    private int historyLen = DEFAULT_HISTORY_LEN;
    private double totalActiveHostTime = 0;
    private double totalViolatedHostTime = 0;   // SLATAH: time where requested > allocated
    private double totalActiveVmTime = 0;
    private double totalMigrationDowntime = 0;  // PDM: actual downtime during migration
    private double currentSlatah = 0;
    private double currentPdm = 0;
    private int currentStep = 0;
    private int lastSkippedNoMovableSources = 0;
    private double adaptiveOverloadThreshold = 1.0;
    private double adaptiveCriticalThreshold = 1.0;
    private double adaptiveMeanUtilization = 0.65;
    private int lastInitialVmCount = 0;

    // Workload traces
    private List<String> traceFiles = new ArrayList<>();
    private boolean useAzureTraces = false;  // Flag for Azure vs Philly format
    // Array of UtilizationModels indexed by VM creation order
    private org.cloudsimplus.utilizationmodels.UtilizationModel[] vmUtilArray;

    public Py4jBridge() {
        configureCloudSimLogging();
        loadTraceFiles();
        System.out.println("[Bridge] Ready. Traces: " + traceFiles.size());
    }

    private static void configureCloudSimLogging() {
        Logger rootLogger = (Logger) LoggerFactory.getLogger(Logger.ROOT_LOGGER_NAME);
        rootLogger.setLevel(Level.ERROR);
    }

    private void loadTraceFiles() {
        String traceSource = Optional.ofNullable(System.getenv("TRACE_SOURCE"))
            .orElse("philly")
            .trim()
            .toLowerCase(Locale.ROOT);

        if (!"azure".equals(traceSource)) {
            if (loadPhillyTraces()) {
                return;
            }
        }

        // Azure test traces (pre-processed per-VM CSVs). Use TRACE_SOURCE=azure
        // for this path; the default follows the Philly EDA/data contract.
        File azureDir = new File("data/azure_test/traces");
        if (azureDir.exists()) {
            File[] files = azureDir.listFiles(f -> f.isFile() && f.getName().endsWith(".csv"));
            if (files != null && files.length > 0) {
                for (File f : files) traceFiles.add(f.getAbsolutePath());
                Collections.sort(traceFiles);
                useAzureTraces = true;
                System.out.println("[Bridge] Loaded Azure VM traces: " + traceFiles.size() + " files"
                    + " (TRACE_SOURCE=" + traceSource + ")");
                return;
            }
        }

        if ("azure".equals(traceSource) && loadPhillyTraces()) {
            return;
        }

        // Fallback: PlanetLab
        File dir = new File("data/planetlab");
        if (dir.exists()) {
            File[] dateDirs = dir.listFiles(File::isDirectory);
            if (dateDirs != null) {
                for (File dateDir : dateDirs) {
                    File[] files = dateDir.listFiles(f -> f.isFile() && !f.getName().startsWith("."));
                    if (files != null) {
                        for (File f : files) traceFiles.add(f.getAbsolutePath());
                    }
                }
            }
            Collections.sort(traceFiles);
        }
        System.out.println("[Bridge] Loaded PlanetLab: " + traceFiles.size() + " files");
    }

    private boolean loadPhillyTraces() {
        File dir = new File("data/Gen-Parallel-Workloads/Philly/training_data");
        if (dir.exists()) {
            File[] files = dir.listFiles(f -> f.isFile() && f.getName().endsWith(".csv"));
            if (files != null) {
                for (File f : files) traceFiles.add(f.getAbsolutePath());
            }
            Collections.sort(traceFiles);
            if (!traceFiles.isEmpty()) {
                useAzureTraces = false;
                System.out.println("[Bridge] Loaded Gen-Parallel-Workloads Philly traces: "
                    + traceFiles.size() + " files");
                return true;
            }
        }
        return false;
    }

    // ========== API cho Python ==========

    public double[] reset() {
        System.out.println("[Bridge] reset() called");
        simDone = false;
        simPaused = false;
        prevEnergy = 0;
        currentEnergy = 0;
        totalMigrations = 0;
        intervalMigrations = 0;
        totalActiveHostTime = 0;
        totalViolatedHostTime = 0;
        totalActiveVmTime = 0;
        totalMigrationDowntime = 0;
        currentSlatah = 0;
        currentPdm = 0;
        currentStep = 0;

        simulation = new CloudSimPlus();
        vmGpuRequests = new HashMap<>();
        hostList = createHosts();

        hostCpuHistoryArr = new double[NUM_HOSTS][historyLen];
        // Initialize to 0

        datacenter = new DatacenterSimple(simulation, hostList);
        datacenter.setSchedulingInterval(INTERVAL);
        broker = new DatacenterBrokerSimple(simulation);

        // Shuffle trace files for episode variation
        if (!traceFiles.isEmpty()) {
            Collections.shuffle(traceFiles);
        }

        // Offered load is derived from the observed process profile. This
        // preserves migration headroom instead of filling the cluster with a
        // hand-picked random VM-count range before scheduling begins.
        Random resetRng = new Random();
        int numVms = deriveInitialVmCount();
        lastInitialVmCount = numVms;
        vmList = createVms(numVms);
        Collections.shuffle(vmList, resetRng); // Random placement order
        broker.submitVmList(vmList);
        System.out.printf(
            "[Bridge] Process-derived offered load: mean=%.4f overload=%.4f initialVMs=%d%n",
            adaptiveMeanUtilization, adaptiveOverloadThreshold, numVms
        );

        List<Cloudlet> cloudlets = createCloudlets(numVms);
        broker.submitCloudletList(cloudlets);

        // Pause mechanism (energy now accumulated per-interval in step())
        simulation.addOnClockTickListener(info -> {
            double t = info.getTime();
            if (t >= nextPauseTime && !simPaused) {
                simPaused = true;
                simulation.pause();
            }
        });

        nextPauseTime = INTERVAL;
        simThread = new Thread(() -> {
            simulation.start();
            simDone = true;
            simPaused = true;
        });
        simThread.start();
        waitForPause();
        for (int hi = 0; hi < hostList.size(); hi++) {
            Arrays.fill(hostCpuHistoryArr[hi], getHostCpuUtil(hostList.get(hi)));
        }

        return collectGlobalState();
    }

    /**
     * step(overloadedHostIndices, underloadedHostIndices, selectionAction, placementActions)
     * Returns: double[GLOBAL_STATE_DIM + 12]
     *   [0..7]   = global_state (8 dim)
     *   [8]      = reward
     *   [9]      = done
     *   [10]     = intervalMigrations
     *   [11]     = currentStep
     *   [12..17] = extended metrics
     */
    public double[] step(int[] overloadedHostIndices, int[] underloadedHostIndices,
                         int selectionAction, int[] placementActions) {
        if (simDone) {
            double[] result = new double[GLOBAL_STATE_DIM + 4];
            result[GLOBAL_STATE_DIM + 1] = 1.0;
            return result;
        }

        currentStep++;

        Map<Vm, Host> vmSourceHost = new HashMap<>();
        List<Vm> vmsToMigrate = collectVmsToMigrate(
            overloadedHostIndices,
            underloadedHostIndices,
            selectionAction,
            vmSourceHost
        );

        // 4. Execute migrations (LOGICAL: direct VM reassignment)
        intervalMigrations = 0;
        double intervalSlaCost = 0.0;
        int attempted = 0;
        int skippedNoCandidates = lastSkippedNoMovableSources;
        int skippedSameHost = 0;
        int migFailed = 0;
        long activeHostsBefore = hostList.stream().filter(h -> !h.getVmList().isEmpty()).count();
        Map<Vm, Host> requestedTargets = new HashMap<>();
        Map<Vm, Double> requestedTargetUtils = new HashMap<>();
        Map<Vm, Double> requestedDowntimes = new HashMap<>();
        Map<Vm, Double> requestedSlaCosts = new HashMap<>();

        for (int i = 0; i < vmsToMigrate.size(); i++) {
            Vm vm = vmsToMigrate.get(i);
            List<Host> candidates = getTopKCandidates(vm);
            if (candidates.isEmpty()) {
                skippedNoCandidates++;
                continue;
            }

            int actionIdx = (placementActions != null && i < placementActions.length)
                ? placementActions[i] : 0;
            actionIdx = Math.max(0, Math.min(actionIdx, candidates.size() - 1));

            Host target = candidates.get(actionIdx);
            Host source = vmSourceHost.get(vm);

            if (target == null || source == null || target == source) {
                skippedSameHost++;
                continue;
            }
            if (!hasGpuCapacity(target, vm)) {
                migFailed++;
                continue;
            }

            attempted++;
            try {
                // Downtime formula (Beloglazov 2012): T_mig = V_ram / BW_link
                double linkBwMbps = Math.min(source.getBw().getCapacity(), target.getBw().getCapacity());
                double ramMb = vm.getRam().getCapacity() * 8.0;
                double downtime = ramMb / Math.max(1, linkBwMbps);

                // SLA tier cost
                double wTier = 0.5;
                if (vm.getRam().getCapacity() >= 4000) wTier = 3.0;
                else if (vm.getRam().getCapacity() >= 2000) wTier = 1.5;

                // Use CloudSim Plus' migration transaction. Metrics are
                // committed only after the simulation confirms the VM moved.
                datacenter.requestVmMigration(vm, target);
                requestedTargets.put(vm, target);
                requestedTargetUtils.put(vm, getHostCpuUtil(target));
                requestedDowntimes.put(vm, downtime);
                requestedSlaCosts.put(vm, wTier * downtime);
            } catch (Exception e) {
                migFailed++;
            }
        }

        // 5. Advance simulation 1 interval
        nextPauseTime += INTERVAL;
        if (nextPauseTime > MAX_TIME) {
            simDone = true;
        } else {
            simPaused = false;
            simulation.resume();
            waitForPause();
        }

        // Commit migration metrics only for VMs CloudSim actually moved.
        intervalMigrations = 0;
        lastStepMigrations.clear();
        double sumTargetUtil = 0.0;
        intervalSlaCost = 0.0;
        for (Map.Entry<Vm, Host> entry : requestedTargets.entrySet()) {
            Vm vm = entry.getKey();
            Host target = entry.getValue();
            if (vm.getHost().equals(target)) {
                intervalMigrations++;
                sumTargetUtil += requestedTargetUtils.getOrDefault(vm, 0.0);
                double downtime = requestedDowntimes.getOrDefault(vm, 0.0);
                totalMigrationDowntime += downtime;
                intervalSlaCost += requestedSlaCosts.getOrDefault(vm, 0.0);
                Host source = vmSourceHost.get(vm);
                int sourceId = hostList.indexOf(source);
                int targetId = hostList.indexOf(target);
                lastStepMigrations.add(vm.getId() + "," + sourceId + "," + targetId);
            } else {
                migFailed++;
            }
        }
        long activeHostsAfter = hostList.stream().filter(h -> !h.getVmList().isEmpty()).count();
        double avgTargetUtil = intervalMigrations > 0 ? sumTargetUtil / intervalMigrations : 0.0;
        double activeHostDelta = activeHostsAfter - activeHostsBefore;
        totalMigrations += intervalMigrations;
        long hostedVmCount = hostList.stream().mapToLong(h -> h.getVmList().size()).sum();
        long assignedVmCount = vmList.stream()
            .filter(v -> !v.getHost().equals(Host.NULL))
            .count();
        // A VM that is not migrating sits in exactly one host vmList AND has a non-null
        // host, so it contributes equally to both counts. Only VMs actively migrating can
        // differ between the two views (removed from a vmList mid-transit, or host pointer
        // not yet committed) -- each by at most one. The gap is therefore bounded by the
        // number of migrating VMs; anything larger is a genuine placement-accounting bug.
        // Count every VM whose host<->vmList accounting can legitimately be in flux at
        // this pause point. In an event-driven simulator a VM can be (a) flagged
        // isInMigration, (b) sitting in some host's migratingIn set, or (c) in a
        // migratingOut set -- and at the exact instant a migration finishes it may have
        // left the source vmList before its host pointer/target vmList membership is
        // committed. The union of these sets is the maximum number of VMs that can differ
        // between hostedVmCount and assignedVmCount without indicating a real leak.
        java.util.Set<Vm> inTransit = new java.util.HashSet<>();
        for (Vm v : vmList) {
            if (v.isInMigration()) inTransit.add(v);
        }
        for (Host h : hostList) {
            inTransit.addAll(h.getVmsMigratingIn());
            inTransit.addAll(h.getVmsMigratingOut());
        }
        long transitVmCount = inTransit.size();
        long invariantGap = Math.abs(hostedVmCount - assignedVmCount);
        if (invariantGap > transitVmCount) {
            // Previously this threw IllegalStateException and killed the entire
            // training run (e.g. the Ep25 crash with hosted=19 assigned=20).
            // An off-by-one host<->vmList accounting gap at a pause point is
            // transient in the event-driven simulator and self-heals on the
            // next interval, so we log and continue instead of aborting. A
            // genuinely large, persistent leak is still surfaced via the
            // running invariantWarnings counter exposed to the Python side.
            invariantWarnings++;
            System.err.printf(
                "[WARN] VM placement invariant gap: hosted=%d assigned=%d inTransit=%d (warning #%d)%n",
                hostedVmCount, assignedVmCount, transitVmCount, invariantWarnings
            );
        }

        // Diagnostic (first 3 steps per episode)
        if (currentStep <= 3) {
            System.out.printf("[DIAG] step=%d: OL=%d vmsToMig=%d attempted=%d " +
                "success=%d noCand=%d sameHost=%d failed=%d%n",
                currentStep, countValidHosts(overloadedHostIndices), vmsToMigrate.size(),
                attempted, intervalMigrations, skippedNoCandidates,
                skippedSameHost, migFailed);
        }

        // 6. Compute per-interval energy (Beloglazov 2012: E = Σ P(u) × Δt)
        double intervalEnergy = computeIntervalEnergy();
        currentEnergy += intervalEnergy;
        double deltaEnergy = intervalEnergy;

        // 7. Update History and SLA Metrics
        // Physical overload remains separate from the process-adaptive
        // operational and critical thresholds.
        double maxRawDemand = 0;
        int operationalOverloads = 0;
        int criticalOverloads = 0;
        int capacityOverloads = 0;
        for (int hi = 0; hi < hostList.size(); hi++) {
            Host h = hostList.get(hi);
            double util = getHostCpuUtil(h);         // capped [0,1] for obs/history
            double rawDemand = getRawHostCpuDemandRatio(h); // demand/capacity ratio
            if (rawDemand > maxRawDemand) maxRawDemand = rawDemand;
            if (rawDemand > adaptiveOverloadThreshold) operationalOverloads++;
            if (rawDemand > adaptiveCriticalThreshold) criticalOverloads++;
            if (rawDemand > 1.0) capacityOverloads++;
            // Shift history left and add new value
            System.arraycopy(hostCpuHistoryArr[hi], 1, hostCpuHistoryArr[hi], 0, historyLen - 1);
            hostCpuHistoryArr[hi][historyLen - 1] = util;
            if (!h.getVmList().isEmpty()) {
                totalActiveHostTime += INTERVAL;
                // SLA risk time uses the adaptive critical threshold. Actual
                // physical capacity violations remain in capacityOverloads.
                if (rawDemand > adaptiveCriticalThreshold) {
                    totalViolatedHostTime += INTERVAL;
                }
            }
        }
        // PDM denominator: total active VM-time
        long activeVmCount = vmList.stream()
            .filter(v -> !v.getHost().equals(Host.NULL)).count();
        totalActiveVmTime += activeVmCount * INTERVAL;

        currentSlatah = totalActiveHostTime > 0
            ? totalViolatedHostTime / totalActiveHostTime : 0.0;
        currentPdm = totalActiveVmTime > 0
            ? totalMigrationDowntime / totalActiveVmTime : 0.0;
        double currentSlav = currentSlatah * currentPdm;

        // 8. Reward: Energy + SLAV penalty + migration cost
        double reward = -(deltaEnergy / 10.0)           // normalize energy
                        - 1000.0 * currentSlav           // SLAV penalty
                        - 0.5 * intervalMigrations;       // migration cost

        // 8. Pack result (expanded: +attempted, +failed, +consolidation, +blocked)
        double[] globalState = collectGlobalState();
        double[] result = new double[GLOBAL_STATE_DIM + 12];
        System.arraycopy(globalState, 0, result, 0, GLOBAL_STATE_DIM);
        result[GLOBAL_STATE_DIM] = reward;
        result[GLOBAL_STATE_DIM + 1] = simDone ? 1.0 : 0.0;
        result[GLOBAL_STATE_DIM + 2] = intervalMigrations;
        result[GLOBAL_STATE_DIM + 3] = currentStep;
        result[GLOBAL_STATE_DIM + 4] = intervalSlaCost;
        result[GLOBAL_STATE_DIM + 5] = attempted;
        result[GLOBAL_STATE_DIM + 6] = migFailed;
        result[GLOBAL_STATE_DIM + 7] = avgTargetUtil;     // consolidation metric
        result[GLOBAL_STATE_DIM + 8] = activeHostDelta;    // +/- active hosts change
        long activeNow = hostList.stream().filter(h -> !h.getVmList().isEmpty()).count();
        result[GLOBAL_STATE_DIM + 9] = (double) activeNow / NUM_HOSTS;  // current active ratio
        result[GLOBAL_STATE_DIM + 10] = skippedNoCandidates;
        result[GLOBAL_STATE_DIM + 11] = skippedSameHost;

        System.out.printf("[Bridge] step=%d t=%.0f | sources=%d opOL=%d critOL=%d capOL=%d | sel=%d | mig=%d | " +
                "blocked=%d same=%d | SLATAH=%.6f PDM=%.10f | reward=%.3f | active=%d avgTgt=%.3f maxDemand=%.3f maxGpu=%.3f%n",
            currentStep, nextPauseTime - INTERVAL, countValidHosts(overloadedHostIndices),
            operationalOverloads, criticalOverloads, capacityOverloads,
            selectionAction, intervalMigrations,
            skippedNoCandidates, skippedSameHost,
            currentSlatah, currentPdm, reward,
            activeNow, avgTargetUtil, maxRawDemand, getMaxHostGpuUtil());

        return result;
    }

    // ========== API dimensions ==========
    public int getGlobalStateDim() { return GLOBAL_STATE_DIM; }
    public int getVmStateDim() { return VM_STATE_DIM; }
    public int getTopK() { return TOP_K; }
    public int getNumSelectionActions() { return 3; }
    public int getNumHosts() { return NUM_HOSTS; }
    public int getHistoryLen() { return historyLen; }
    public int getInvariantWarnings() { return invariantWarnings; }
    public void setHistoryLen(int requestedHistoryLen) {
        historyLen = Math.max(2, requestedHistoryLen);
    }

    /** Receive process-adaptive policy thresholds from Python before each step. */
    public void setAdaptiveThresholds(double overloadThreshold, double criticalThreshold) {
        adaptiveOverloadThreshold = Math.max(0.0, Math.min(1.0, overloadThreshold));
        adaptiveCriticalThreshold = Math.max(
            adaptiveOverloadThreshold,
            Math.min(1.0, criticalThreshold)
        );
    }

    public void setProcessProfile(double overloadThreshold, double criticalThreshold,
                                  double meanUtilization) {
        setAdaptiveThresholds(overloadThreshold, criticalThreshold);
        adaptiveMeanUtilization = Math.max(
            0.0,
            Math.min(adaptiveOverloadThreshold, meanUtilization)
        );
    }

    public int getInitialVmCount() { return lastInitialVmCount; }

    /** API: time-series history for Autoformer */
    public double[][] getHostHistory() {
        return hostCpuHistoryArr;
    }

    public int[] getHostVmCounts() {
        int[] counts = new int[NUM_HOSTS];
        for (int i = 0; i < NUM_HOSTS; i++) {
            counts[i] = hostList.get(i).getVmList().size();
        }
        return counts;
    }

    public double[] getHostRawDemandRatios() {
        double[] ratios = new double[NUM_HOSTS];
        for (int i = 0; i < NUM_HOSTS; i++) {
            ratios[i] = getRawHostCpuDemandRatio(hostList.get(i));
        }
        return ratios;
    }

    public int[] getMovableHostMask() {
        int[] mask = new int[NUM_HOSTS];
        for (int i = 0; i < NUM_HOSTS; i++) {
            mask[i] = hasMovableVm(hostList.get(i)) ? 1 : 0;
        }
        return mask;
    }

    /**
     * Explain why a host cannot be used as an overload-migration source.
     * 0=movable, 1=single/empty, 2=no CPU/RAM/BW candidate, 3=GPU blocked.
     */
    public int[] getHostMobilityReasonCodes() {
        int[] codes = new int[NUM_HOSTS];
        for (int i = 0; i < NUM_HOSTS; i++) {
            codes[i] = getHostMobilityReasonCode(hostList.get(i));
        }
        return codes;
    }

    /** API: SLAV metrics for CSV logging */
    public double[] getSlavMetrics() {
        return new double[]{currentSlatah, currentPdm, currentSlatah * currentPdm,
                            currentEnergy, totalMigrations, currentStep};
    }

    public List<String> getLastStepMigrations() {
        return lastStepMigrations;
    }

    public double[] getHostGpuUtils() {
        double[] utils = new double[NUM_HOSTS];
        for (int i = 0; i < NUM_HOSTS; i++) {
            utils[i] = getHostGpuUtil(hostList.get(i));
        }
        return utils;
    }

    /** Preview VM states for Agent 2 */
    public double[] previewVmStates(int[] overloadedHostIndices, int selectionAction) {
        return previewMigrationVmStates(overloadedHostIndices, null, selectionAction);
    }

    /** Preview VM states in the exact order consumed by step() placementActions. */
    public double[] previewMigrationVmStates(int[] overloadedHostIndices, int[] underloadedHostIndices,
                                             int selectionAction) {
        Map<Vm, Host> vmSourceHost = new HashMap<>();
        List<Vm> vmsToMigrate = collectVmsToMigrate(
            overloadedHostIndices,
            underloadedHostIndices,
            selectionAction,
            vmSourceHost
        );
        List<double[]> states = new ArrayList<>();
        for (Vm vm : vmsToMigrate) {
            Host source = vmSourceHost.get(vm);
            if (source != null) {
                states.add(buildVmState(vm, source));
            }
        }
        int numVms = states.size();
        double[] result = new double[1 + numVms * VM_STATE_DIM];
        result[0] = numVms;
        for (int i = 0; i < numVms; i++) {
            System.arraycopy(states.get(i), 0, result, 1 + i * VM_STATE_DIM, VM_STATE_DIM);
        }
        return result;
    }

    private List<Vm> collectVmsToMigrate(int[] overloadedHostIndices, int[] underloadedHostIndices,
                                         int selectionAction, Map<Vm, Host> vmSourceHost) {
        lastSkippedNoMovableSources = 0;
        List<Host> overloaded = new ArrayList<>();
        if (overloadedHostIndices != null) {
            for (int idx : overloadedHostIndices) {
                if (idx >= 0 && idx < hostList.size()) {
                    Host h = hostList.get(idx);
                    if (h.getVmList().size() > 1) overloaded.add(h);
                }
            }
        }

        List<Vm> vmsToMigrate = new ArrayList<>();
        for (Host h : overloaded) {
            List<Vm> hvms = new ArrayList<>(h.getVmList());
            Vm selected = selectVm(hvms, selectionAction);
            if (selected != null) {
                vmsToMigrate.add(selected);
                vmSourceHost.put(selected, h);
            } else {
                lastSkippedNoMovableSources++;
            }
        }

        int evacuatedCount = 0;
        int maxEvacuatePerStep = 2;
        if (underloadedHostIndices != null) {
            for (int idx : underloadedHostIndices) {
                if (evacuatedCount >= maxEvacuatePerStep) break;
                if (idx >= 0 && idx < hostList.size()) {
                    Host h = hostList.get(idx);
                    if (!h.getVmList().isEmpty()) {
                        for (Object obj : new ArrayList<>(h.getVmList())) {
                            Vm v = (Vm) obj;
                            if (!vmsToMigrate.contains(v) && hasPlacementCandidate(v)) {
                                vmsToMigrate.add(v);
                                vmSourceHost.put(v, h);
                            }
                        }
                        evacuatedCount++;
                    }
                }
            }
        }
        return vmsToMigrate;
    }

    private int countValidHosts(int[] hostIndices) {
        int count = 0;
        if (hostIndices != null) {
            for (int idx : hostIndices) {
                if (idx >= 0 && idx < hostList.size()) count++;
            }
        }
        return count;
    }

    // ========== Internal helpers ==========

    private void waitForPause() {
        while (!simPaused && !simDone) {
            try { Thread.sleep(5); } catch (InterruptedException e) { break; }
        }
    }

    private Vm selectVm(List<Vm> vms, int selectionAction) {
        if (vms.isEmpty()) return null;
        List<Vm> feasible = vms.stream()
            .filter(this::hasPlacementCandidate)
            .collect(Collectors.toList());
        if (feasible.isEmpty()) return null;
        List<Vm> pool = feasible;
        if (selectionAction == 0) { // MMT
            return pool.stream().min(Comparator.comparingLong(v -> v.getRam().getCapacity())).orElse(null);
        } else if (selectionAction == 1) { // Max Utilization (highest MIPS)
            return pool.stream().max(Comparator.comparingDouble(v -> v.getMips())).orElse(null);
        } else { // Random
            int idx = Math.floorMod(currentStep, pool.size());
            return pool.get(idx);
        }
    }

    private boolean hasMovableVm(Host host) {
        if (host == null || host.getVmList().size() <= 1) return false;
        for (Object obj : host.getVmList()) {
            Vm vm = (Vm) obj;
            if (hasPlacementCandidate(vm)) return true;
        }
        return false;
    }

    private int getHostMobilityReasonCode(Host host) {
        if (host == null || host.getVmList().size() <= 1) return 1;
        boolean hasBaseCandidate = false;
        for (Object obj : host.getVmList()) {
            Vm vm = (Vm) obj;
            for (Host target : hostList) {
                if (target.equals(host) || !target.isSuitableForVm(vm)) continue;
                hasBaseCandidate = true;
                if (hasGpuCapacity(target, vm)) return 0;
            }
        }
        return hasBaseCandidate ? 3 : 2;
    }

    private boolean hasPlacementCandidate(Vm vm) {
        return !getTopKCandidates(vm).isEmpty();
    }

    private List<Host> createHosts() {
        List<Host> hosts = new ArrayList<>();
        for (int i = 0; i < NUM_HOSTS; i++) {
            List<Pe> pes = new ArrayList<>();
            for (int j = 0; j < HOST_PES; j++) pes.add(new PeSimple(HOST_MIPS));
            Host h = new HostSimple(HOST_RAM, HOST_BW, HOST_STORAGE, pes);
            h.setVmScheduler(new VmSchedulerTimeShared());
            // Real SPECpower model (HP ProLiant ML110 G5)
            h.setPowerModel(new PowerModelHostSimple(HOST_MAX_POWER, HOST_IDLE_POWER));
            hosts.add(h);
        }
        return hosts;
    }

    private List<Vm> createVms(int count) {
        List<Vm> vms = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            int[] t = VM_TYPES[i % VM_TYPES.length];
            Vm vm = new VmSimple(i, t[0], t[2]);
            vm.setRam(t[1]).setBw(100).setSize(10000);  // VM BW = 100 Mbit/s
            vm.setCloudletScheduler(new CloudletSchedulerTimeShared());
            vmGpuRequests.put(vm.getId(), deriveGpuRequestForVm(i));
            vms.add(vm);
        }
        return vms;
    }

    private int deriveInitialVmCount() {
        double clusterMips = (double) NUM_HOSTS * HOST_PES * HOST_MIPS;
        // Center offered load between the observed normal-process mean and
        // the adaptive overload boundary. This creates learnable transitions
        // on both sides of the boundary while retaining migration headroom.
        double targetRatio = (adaptiveMeanUtilization + adaptiveOverloadThreshold) / 2.0;
        double targetMips = clusterMips * targetRatio;
        double allocatedMips = 0.0;
        int count = 0;
        while (count < NUM_HOSTS * TOP_K) {
            int[] type = VM_TYPES[count % VM_TYPES.length];
            double nextMips = type[0] * type[2];
            if (allocatedMips + nextMips > targetMips) break;
            allocatedMips += nextMips;
            count++;
        }
        return Math.max(NUM_HOSTS, count);
    }

    private List<Cloudlet> createCloudlets(int count) {
        vmUtilArray = new org.cloudsimplus.utilizationmodels.UtilizationModel[count];
        List<Cloudlet> cloudlets = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            // Long-running cloudlet: enough MI to survive 24h at moderate utilization
            CloudletSimple c = new CloudletSimple(300_000_000L, 1);
            org.cloudsimplus.utilizationmodels.UtilizationModel utilModel;
            if (!traceFiles.isEmpty()) {
                String file = traceFiles.get(i % traceFiles.size());
                try {
                    if (useAzureTraces) {
                        utilModel = new UtilizationModelAzure(file, INTERVAL);
                    } else {
                        utilModel = new UtilizationModelGenParallel(file, INTERVAL, i);
                    }
                } catch (Exception e) {
                    utilModel = createSyntheticModel(i);
                }
            } else {
                utilModel = createSyntheticModel(i);
            }
            c.setUtilizationModelCpu(utilModel);
            vmUtilArray[i] = utilModel;
            cloudlets.add(c);
        }
        return cloudlets;
    }

    private UtilizationModelDynamic createSyntheticModel(int seed) {
        // Varied base utilization to create interesting workload
        double base = 0.15 + (seed % 7) * 0.1;
        return new UtilizationModelDynamic(Math.min(0.9, base));
    }

    private double getHostCpuUtil(Host host) {
        if (host.getVmList().isEmpty()) return 0.0;
        double totalCap = host.getPeList().stream().mapToDouble(Pe::getCapacity).sum();
        double used = computeRawDemand(host);
        return Math.min(1.0, used / totalCap);
    }

    /**
     * Compute RAW CPU demand (uncapped) — can exceed 1.0 when oversubscribed.
     * Used for SLATAH: Beloglazov 2012 defines violation as demand > capacity.
     */
    private double getRawHostCpuDemandRatio(Host host) {
        if (host.getVmList().isEmpty()) return 0.0;
        double totalCap = host.getPeList().stream().mapToDouble(Pe::getCapacity).sum();
        double used = computeRawDemand(host);
        return used / totalCap;  // NOT capped — can be > 1.0
    }

    private double computeRawDemand(Host host) {
        double used = 0;
        double simTime = (simulation != null) ? simulation.clock() : 0;
        for (Object obj : host.getVmList()) {
            Vm vm = (Vm) obj;
            double vmAllocatedMips = vm.getMips() * vm.getPesNumber();
            double utilRatio = 0.5;
            long vmId = vm.getId();
            if (vmId >= 0 && vmId < vmUtilArray.length) {
                utilRatio = vmUtilArray[(int) vmId].getUtilization(simTime);
            }
            used += vmAllocatedMips * Math.max(0.01, Math.min(1.0, utilRatio));
        }
        return used;
    }

    /**
     * Compute energy consumed in ONE interval (Δt = 300s).
     * Beloglazov 2012: E = Σ P(u(t)) × Δt
     * P(u) = P_idle + (P_max - P_idle) × u  (linear SPECpower model)
     */
    private double computeIntervalEnergy() {
        double totalPowerW = 0;
        for (Host h : hostList) {
            double util = getHostCpuUtil(h);
            if (util <= 0 && h.getVmList().isEmpty()) continue; // host OFF = 0W
            // SPECpower linear: HP ProLiant ML110 G5
            totalPowerW += HOST_IDLE_POWER + util * (HOST_MAX_POWER - HOST_IDLE_POWER);
        }
        // Energy for 1 interval: Power(W) × time(s) → Joules → kWh
        return totalPowerW * INTERVAL / 3_600_000.0; // kWh
    }

    private double[] collectGlobalState() {
        double[] state = new double[GLOBAL_STATE_DIM];
        double[] utils = hostList.stream().mapToDouble(this::getHostCpuUtil).toArray();
        double avg = Arrays.stream(utils).average().orElse(0);
        state[0] = avg;                                                              // avg cpu
        state[1] = Math.sqrt(Arrays.stream(utils).map(u -> (u - avg) * (u - avg))
                    .average().orElse(0));                                            // std cpu
        long active = hostList.stream().filter(h -> !h.getVmList().isEmpty()).count();
        state[2] = (double) active / NUM_HOSTS;                                      // active ratio
        state[3] = currentEnergy - prevEnergy;                                       // delta energy
        prevEnergy = currentEnergy;
        state[4] = currentSlatah;                                                    // SLATAH
        state[5] = currentPdm;                                                       // PDM
        state[6] = (double) intervalMigrations / Math.max(1, NUM_HOSTS);             // norm migrations
        state[7] = (double) currentStep / (MAX_TIME / INTERVAL);                     // time progress
        return state;
    }

    private List<Host> getTopKCandidates(Vm vm) {
        return hostList.stream()
            .filter(h -> !h.equals(vm.getHost()))
            .filter(h -> h.isSuitableForVm(vm))
            .filter(h -> hasGpuCapacity(h, vm))
            .sorted(Comparator.comparingDouble(this::getHostCpuUtil))
            .limit(TOP_K)
            .collect(Collectors.toList());
    }

    private double[] buildVmState(Vm vm, Host sourceHost) {
        double[] s = new double[VM_STATE_DIM];
        List<Host> candidates = getTopKCandidates(vm);
        double vmGpuRatio = getVmGpuRequest(vm) / HOST_GPU;
        for (int i = 0; i < TOP_K; i++) {
            if (i < candidates.size()) {
                Host candidate = candidates.get(i);
                double candidateUtil = getHostCpuUtil(candidate);
                s[i] = candidateUtil;
                double power = HOST_IDLE_POWER + candidateUtil
                    * (HOST_MAX_POWER - HOST_IDLE_POWER);
                s[TOP_K + i] = power / HOST_MAX_POWER; // normalized power
                s[TOP_K * 2 + i] = getHostGpuFreeRatio(candidate, vm);
            } else {
                s[i] = -1;
                s[TOP_K + i] = -1;
                s[TOP_K * 2 + i] = -1;
            }
        }
        double totalMips = sourceHost.getPeList().stream().mapToDouble(Pe::getCapacity).sum();
        int base = TOP_K * 3;
        s[base]     = vm.getMips() / totalMips;                                   // vm cpu ratio
        s[base + 1] = vm.getRam().getCapacity() / (double) HOST_RAM;              // vm ram ratio
        s[base + 2] = vmGpuRatio;                                                 // vm gpu ratio
        s[base + 3] = getHostCpuUtil(sourceHost);                                 // source cpu util
        s[base + 4] = getHostGpuUtil(sourceHost);                                 // source gpu util
        long activeCount = hostList.stream().filter(h -> !h.getVmList().isEmpty()).count();
        s[base + 5] = (double) activeCount / NUM_HOSTS;                           // active ratio
        // Migration time estimate (seconds)
        double bwMBps = sourceHost.getBw().getCapacity() / 8.0;
        s[base + 6] = vm.getRam().getCapacity() / Math.max(1, bwMBps);            // est. downtime
        s[base + 7] = (double) sourceHost.getVmList().size() / (NUM_HOSTS);       // source vm density
        return s;
    }

    private double deriveGpuRequestForVm(int vmIndex) {
        double fallback = syntheticGpuRequest(vmIndex);
        if (traceFiles.isEmpty() || useAzureTraces) {
            return fallback;
        }
        String file = traceFiles.get(vmIndex % traceFiles.size());
        double traceGpu = readGpuRequestFromTrace(file, vmIndex);
        if (traceGpu < 0) {
            return fallback;
        }
        return Math.max(0.0, Math.min(HOST_GPU, traceGpu));
    }

    private double syntheticGpuRequest(int vmIndex) {
        double[] profile = {0.0, 0.0, 1.0, 1.0, 2.0};
        return profile[Math.floorMod(vmIndex, profile.length)];
    }

    private double readGpuRequestFromTrace(String path, int rowIndex) {
        try (java.io.BufferedReader br = new java.io.BufferedReader(new java.io.FileReader(path))) {
            String header = br.readLine();
            if (header == null) return -1.0;
            String[] columns = header.split(",");
            int gpuIdx = -1;
            int nodeIdx = -1;
            for (int i = 0; i < columns.length; i++) {
                String column = columns[i].trim();
                if ("gpu_num".equals(column)) {
                    gpuIdx = i;
                } else if ("node_num".equals(column)) {
                    nodeIdx = i;
                }
            }
            if (gpuIdx < 0) return -1.0;
            String line;
            int current = 0;
            int target = Math.max(0, rowIndex);
            while ((line = br.readLine()) != null) {
                if (current == target) {
                    String[] parts = line.split(",");
                    if (gpuIdx >= parts.length || parts[gpuIdx].isBlank()) return -1.0;
                    double gpu = Double.parseDouble(parts[gpuIdx]);
                    double nodes = 1.0;
                    if (nodeIdx >= 0 && nodeIdx < parts.length && !parts[nodeIdx].isBlank()) {
                        nodes = Math.max(1.0, Double.parseDouble(parts[nodeIdx]));
                    }
                    return gpu / nodes;
                }
                current++;
            }
        } catch (Exception ignored) {
            return -1.0;
        }
        return -1.0;
    }

    private double getVmGpuRequest(Vm vm) {
        if (vm == null) return 0.0;
        return Math.max(0.0, vmGpuRequests.getOrDefault(vm.getId(), 0.0));
    }

    private double getHostGpuUsed(Host host) {
        if (host == null || host.getVmList().isEmpty()) return 0.0;
        double used = 0.0;
        for (Object obj : host.getVmList()) {
            used += getVmGpuRequest((Vm) obj);
        }
        return used;
    }

    private double getHostGpuUtil(Host host) {
        return Math.min(1.0, getHostGpuUsed(host) / HOST_GPU);
    }

    private double getHostGpuFreeRatio(Host host, Vm incoming) {
        double reserved = Math.max(0.0, HOST_GPU - getHostGpuUsed(host) - getVmGpuRequest(incoming));
        return Math.max(0.0, reserved / HOST_GPU);
    }

    private boolean hasGpuCapacity(Host host, Vm vm) {
        return getHostGpuUsed(host) + getVmGpuRequest(vm) <= HOST_GPU + 1e-9;
    }

    private double getMaxHostGpuUtil() {
        return hostList.stream().mapToDouble(this::getHostGpuUtil).max().orElse(0.0);
    }

    // ========== Main ==========
    public static void main(String[] args) {
        // Port is configurable so a serving bridge can run alongside a training
        // bridge. Precedence: CLI arg > BRIDGE_PORT env > default 25333.
        int port = 25333;
        String envPort = System.getenv("BRIDGE_PORT");
        if (args.length > 0) {
            port = Integer.parseInt(args[0].trim());
        } else if (envPort != null && !envPort.trim().isEmpty()) {
            port = Integer.parseInt(envPort.trim());
        }
        Py4jBridge bridge = new Py4jBridge();
        GatewayServer server = new GatewayServer(bridge, port);
        server.start();
        System.out.println("[Bridge] Py4J Gateway started on port " + port + ". Waiting for Python...");
    }
}
