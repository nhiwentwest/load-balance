// --- Three.js & Dashboard Simulation Code ---

// State variables
let scene, camera, renderer;
let hostsGroup;
let hostMeshes = [];
let vmCubes = {}; // host_id -> array of VM meshes
let migrationParticles = [];
let currentStepIndex = 0;
let isPlaying = false;
let playbackTimer = null;
let speed = 2; // steps per second
let visData = null; // Full multi-day data
let currentDay = "1"; // Default day key
let currentSteps = []; // Steps for active day
let rewardHistory = [];

// DOM Elements
const btnPlay = document.getElementById('btn-play');
const btnPrev = document.getElementById('btn-prev');
const btnNext = document.getElementById('btn-next');
const btnReset = document.getElementById('btn-reset');
const speedSlider = document.getElementById('speed-slider');
const speedVal = document.getElementById('speed-val');
const timelineSlider = document.getElementById('timeline-slider');
const timelineStep = document.getElementById('timeline-step');
const logOutput = document.getElementById('log-output');
const loadingOverlay = document.getElementById('loading-overlay');
const btnClearLogs = document.getElementById('btn-clear-logs');
const daySelect = document.getElementById('day-select');

// Metrics DOM
const mEnergy = document.getElementById('metric-energy');
const mSaving = document.getElementById('metric-saving');
const mSlatah = document.getElementById('metric-slatah');
const mMigrations = document.getElementById('metric-migrations');
const mReward = document.getElementById('metric-reward');

// Agent DOM Cards
const agentA1 = document.getElementById('agent-a1');
const agentA2 = document.getElementById('agent-a2');
const agentA3 = document.getElementById('agent-a3');
const agentA4 = document.getElementById('agent-a4');
const a1Status = document.getElementById('a1-status');
const a2Status = document.getElementById('a2-status');
const a3Status = document.getElementById('a3-status');
const a4Status = document.getElementById('a4-status');

// HUD DOM
const hudUnder = document.getElementById('hud-th-under');
const hudOver = document.getElementById('hud-th-over');
const hudCrit = document.getElementById('hud-th-crit');

// Sparkline Canvas
const sparklineCanvas = document.getElementById('reward-sparkline');
const sparklineCtx = sparklineCanvas.getContext('2d');

// Grid layout parameters
const GRID_ROWS = 2;
const GRID_COLS = 10;
const SPACING_X = 3.2;
const SPACING_Z = 4.5;
const RACK_HEIGHT = 5.0;

// Host color definitions (Neon glow styles)
const COLORS = {
    inactive: 0x334155,
    underload: 0x06b6d4, // Cyan
    normal: 0x10b981,     // Emerald green
    overload: 0xf59e0b,   // Amber
    critical: 0xef4444,   // Red
};

// Check thresholds
const THRESHOLDS = {
    underload: 0.4716,
    overload: 0.9238,
    critical: 0.9392
};

// Initialize app when window loads
window.addEventListener('load', () => {
    initThree();
    checkDataLoaded();
});

// Init Three.js Environment
function initThree() {
    const container = document.getElementById('canvas-container');
    
    // Create Scene
    scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x06080e, 0.015);

    // Create Camera (Fixed Cinematic Perspective)
    camera = new THREE.PerspectiveCamera(40, container.clientWidth / container.clientHeight, 0.1, 100);
    camera.position.set(0, 16, 23);
    camera.lookAt(0, -1.0, 0);

    // Create Renderer
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.shadowMap.enabled = true;
    container.appendChild(renderer.domElement);

    // Lights
    const ambientLight = new THREE.AmbientLight(0x0f172a, 1.2);
    scene.add(ambientLight);

    const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight.position.set(5, 20, 10);
    dirLight.castShadow = true;
    scene.add(dirLight);

    // Accent point lights
    const blueLight = new THREE.PointLight(0x00f2fe, 1, 30);
    blueLight.position.set(-10, 5, 0);
    scene.add(blueLight);

    const redLight = new THREE.PointLight(0xff0844, 0.8, 30);
    redLight.position.set(10, 5, 0);
    scene.add(redLight);

    // Ground Grid
    const gridHelper = new THREE.GridHelper(40, 20, 0x1e293b, 0x0f172a);
    gridHelper.position.y = -RACK_HEIGHT/2 - 0.01;
    scene.add(gridHelper);

    hostsGroup = new THREE.Group();
    scene.add(hostsGroup);

    createHostRacks();
    window.addEventListener('resize', onWindowResize);
    animate();
}

// Create 20 Host meshes
function createHostRacks() {
    let hostIdx = 0;
    const startX = -((GRID_COLS - 1) * SPACING_X) / 2;
    const startZ = -((GRID_ROWS - 1) * SPACING_Z) / 2;

    for (let r = 0; r < GRID_ROWS; r++) {
        for (let c = 0; c < GRID_COLS; c++) {
            const x = startX + c * SPACING_X;
            const z = startZ + r * SPACING_Z;

            const rackGroup = new THREE.Group();
            rackGroup.position.set(x, 0, z);

            const rackGeom = new THREE.BoxGeometry(1.5, RACK_HEIGHT, 1.5);
            const rackMat = new THREE.MeshPhongMaterial({
                color: 0x1e293b,
                transparent: true,
                opacity: 0.15,
                shininess: 40,
                specular: 0x334155
            });
            const rackMesh = new THREE.Mesh(rackGeom, rackMat);
            rackMesh.receiveShadow = true;
            rackGroup.add(rackMesh);

            const edges = new THREE.EdgesGeometry(rackGeom);
            const lineMat = new THREE.LineBasicMaterial({
                color: COLORS.normal,
                transparent: true,
                opacity: 0.3
            });
            const line = new THREE.LineSegments(edges, lineMat);
            rackGroup.add(line);

            const ledGeom = new THREE.BoxGeometry(0.1, 0.1, 0.1);
            const ledMat = new THREE.MeshBasicMaterial({ color: 0x00ff00 });
            const led = new THREE.Mesh(ledGeom, ledMat);
            led.position.set(-0.5, RACK_HEIGHT/2 - 0.2, 0.76);
            rackGroup.add(led);

            const labelSprite = createLabelSprite(`H-${hostIdx.toString().padStart(2, '0')}`);
            labelSprite.position.set(0, RACK_HEIGHT/2 + 0.6, 0);
            rackGroup.add(labelSprite);

            hostMeshes.push({
                group: rackGroup,
                mesh: rackMesh,
                line: line,
                led: led,
                id: hostIdx,
                label: labelSprite,
                x: x,
                z: z
            });

            vmCubes[hostIdx] = [];
            hostsGroup.add(rackGroup);
            hostIdx++;
        }
    }
}

function createLabelSprite(text) {
    const canvas = document.createElement('canvas');
    canvas.width = 128;
    canvas.height = 64;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = 'rgba(0, 0, 0, 0)';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.font = 'Bold 28px "Outfit", sans-serif';
    ctx.fillStyle = '#94a3b8';
    ctx.textAlign = 'center';
    ctx.fillText(text, 64, 40);

    const texture = new THREE.CanvasTexture(canvas);
    const material = new THREE.SpriteMaterial({ map: texture, transparent: true });
    const sprite = new THREE.Sprite(material);
    sprite.scale.set(1.5, 0.75, 1);
    return sprite;
}

function updateLabelText(sprite, hostId, cpuVal, active) {
    const canvas = sprite.material.map.image;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.font = 'Bold 20px "Outfit", sans-serif';
    
    let color = '#94a3b8';
    if (!active) color = '#475569';
    else if (cpuVal > THRESHOLDS.critical) color = '#ef4444';
    else if (cpuVal > THRESHOLDS.overload) color = '#f59e0b';
    else if (cpuVal < THRESHOLDS.underload) color = '#06b6d4';
    else color = '#10b981';
    
    ctx.fillStyle = color;
    ctx.textAlign = 'center';
    
    const hText = `H-${hostId.toString().padStart(2, '0')}`;
    const cpuText = active ? `${Math.round(cpuVal * 100)}%` : 'OFF';
    
    ctx.fillText(hText, 64, 25);
    ctx.font = '16px "Fira Code", monospace';
    ctx.fillText(cpuText, 64, 48);
    
    sprite.material.map.needsUpdate = true;
}

function onWindowResize() {
    const container = document.getElementById('canvas-container');
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
}

function checkDataLoaded() {
    if (window.VIS_DATA) {
        visData = window.VIS_DATA;
        loadingOverlay.style.opacity = 0;
        setTimeout(() => loadingOverlay.style.display = 'none', 500);
        
        hudUnder.innerText = `${(THRESHOLDS.underload * 100).toFixed(1)}%`;
        hudOver.innerText = `${(THRESHOLDS.overload * 100).toFixed(1)}%`;
        hudCrit.innerText = `${(THRESHOLDS.critical * 100).toFixed(1)}%`;
        
        // Populate day options
        daySelect.innerHTML = "";
        Object.keys(visData.days).sort((a,b) => parseInt(a) - parseInt(b)).forEach(dKey => {
            const opt = document.createElement('option');
            opt.value = dKey;
            opt.innerText = `Simulation: Day ${dKey}`;
            daySelect.appendChild(opt);
        });

        // Set initial day steps
        currentDay = "1";
        currentSteps = visData.days[currentDay].steps;

        initSimulation();
    } else {
        setTimeout(checkDataLoaded, 200);
    }
}

function animate() {
    requestAnimationFrame(animate);
    const time = Date.now() * 0.003;
    hostMeshes.forEach(hm => {
        if (hm.line.material.color.getHex() === COLORS.critical) {
            hm.line.material.opacity = 0.3 + Math.sin(time * 3) * 0.4;
        } else if (hm.line.material.color.getHex() === COLORS.overload) {
            hm.line.material.opacity = 0.4 + Math.sin(time * 2) * 0.2;
        } else if (hm.line.material.color.getHex() === COLORS.inactive) {
            hm.line.material.opacity = 0.1;
        } else {
            hm.line.material.opacity = 0.3;
        }
    });
    updateParticles();
    renderer.render(scene, camera);
}

function initSimulation() {
    timelineSlider.max = currentSteps.length - 1;
    
    // Day Selection Event
    daySelect.addEventListener('change', (e) => {
        pauseSimulation();
        currentDay = e.target.value;
        currentSteps = visData.days[currentDay].steps;
        timelineSlider.max = currentSteps.length - 1;
        rewardHistory = [];
        logOutput.innerHTML = '';
        addLog(`Switched simulation context to Day ${currentDay}.`, 'system');
        setStep(0);
    });

    btnPlay.addEventListener('click', () => {
        if (isPlaying) pauseSimulation();
        else playSimulation();
    });

    btnNext.addEventListener('click', () => {
        pauseSimulation();
        stepForward();
    });

    btnPrev.addEventListener('click', () => {
        pauseSimulation();
        stepBackward();
    });

    btnReset.addEventListener('click', () => {
        pauseSimulation();
        setStep(0);
    });

    timelineSlider.addEventListener('input', (e) => {
        pauseSimulation();
        setStep(parseInt(e.target.value));
    });

    speedSlider.addEventListener('input', (e) => {
        speed = parseInt(e.target.value);
        speedVal.innerText = `${speed}x`;
        if (isPlaying) {
            pauseSimulation();
            playSimulation();
        }
    });

    btnClearLogs.addEventListener('click', () => {
        logOutput.innerHTML = '';
        addLog('Logs cleared.', 'system');
    });

    setStep(0);
}

function playSimulation() {
    isPlaying = true;
    btnPlay.innerHTML = '<i class="fa-solid fa-pause"></i>';
    btnPlay.classList.add('paused');
    
    const intervalTime = 1000 / speed;
    playbackTimer = setInterval(() => {
        if (currentStepIndex >= currentSteps.length - 1) {
            pauseSimulation();
            addLog("Simulation completed for current day.", "system");
        } else {
            stepForward();
        }
    }, intervalTime);
}

function pauseSimulation() {
    isPlaying = false;
    btnPlay.innerHTML = '<i class="fa-solid fa-play"></i>';
    btnPlay.classList.remove('paused');
    if (playbackTimer) {
        clearInterval(playbackTimer);
        playbackTimer = null;
    }
}

function stepForward() {
    if (currentStepIndex < currentSteps.length - 1) {
        setStep(currentStepIndex + 1);
    }
}

function stepBackward() {
    if (currentStepIndex > 0) {
        setStep(currentStepIndex - 1);
    }
}

function setStep(index) {
    currentStepIndex = index;
    timelineSlider.value = index;
    timelineStep.innerText = `Step ${index + 1} / ${currentSteps.length}`;
    
    const step = currentSteps[index];
    
    updateMetrics(step);
    updateAgentStatus(step.a);
    updateHostStates(step.h);

    if (step.m && step.m.length > 0) {
        step.m.forEach(m => {
            // m layout: [vm_id, source, target]
            triggerMigration(m[0], m[1], m[2]);
        });
    }

    logOutput.scrollTop = logOutput.scrollHeight;
}

function updateMetrics(step) {
    // met layout: [energy, slatah, pdm, slav, reward, r1, r2, r3, r4]
    const energyVal = step.met[0];
    const slatahVal = step.met[1];
    const rewardVal = step.met[4];

    mEnergy.innerHTML = `${energyVal.toFixed(2)} <span class="unit">kWh</span>`;
    mSlatah.innerText = `${(slatahVal * 100).toFixed(4)}%`;
    mReward.innerText = rewardVal.toFixed(2);
    
    if (rewardVal >= 0) mReward.className = "metric-value text-green font-mono";
    else mReward.className = "metric-value text-red font-mono";

    // Sum migrations
    let totalMigs = 0;
    for (let i = 0; i <= currentStepIndex; i++) {
        totalMigs += currentSteps[i].m ? currentSteps[i].m.length : 0;
    }
    mMigrations.innerText = totalMigs;

    const progress = (currentStepIndex + 1) / currentSteps.length;
    const baseline = 65.0 * progress;
    const current = energyVal;
    
    let savingPct = 0;
    if (baseline > 0) {
        savingPct = ((baseline - current) / baseline) * 100;
    }
    
    mSaving.innerText = `${savingPct > 0 ? '+' : ''}${savingPct.toFixed(1)}%`;
    if (savingPct >= 0) mSaving.className = "metric-value text-green font-mono";
    else mSaving.className = "metric-value text-red font-mono";

    rewardHistory.push(rewardVal);
    if (rewardHistory.length > 30) rewardHistory.shift();
    drawSparkline(rewardHistory);
}

function updateAgentStatus(a) {
    // a layout: [a1_act, a2_act, a3_act, a4_placements[], ul_idx[], ol_idx[], a1_tp, a1_fp, a1_fn, a2_tp, a2_fp, a2_fn]
    const a1_action = a[0];
    const a2_action = a[1];
    const a3_action = a[2];
    const a4_placements = a[3];
    const underload_indices = a[4];
    const overload_indices = a[5];
    const a1_tp = a[6] === 1;
    const a1_fp = a[7] === 1;
    const a1_fn = a[8] === 1;
    const a2_tp = a[9] === 1;
    const a2_fp = a[10] === 1;
    const a2_fn = a[11] === 1;

    // A1 Panel
    if (underload_indices && underload_indices.length > 0) {
        const uHost = underload_indices[0];
        agentA1.className = "agent-item active";
        a1Status.innerText = `Evacuated Host-${uHost.toString().padStart(2, '0')}`;
        if (a1_tp) addLog(`[A1 Decision] Underload detected on Host-${uHost} (TP)`, 'a1');
        else if (a1_fp) addLog(`[A1 Warning] False Positive underload on Host-${uHost}`, 'system');
    } else {
        agentA1.className = "agent-item";
        a1Status.innerText = "Monitoring idle hosts";
    }

    // A2 Panel
    if (overload_indices && overload_indices.length > 0) {
        const oHost = overload_indices[0];
        agentA2.className = "agent-item active-overload";
        a2Status.innerText = `Flagged Host-${oHost.toString().padStart(2, '0')}`;
        if (a2_tp) addLog(`[A2 Decision] Overload detected on Host-${oHost} (TP)`, 'a2');
        else if (a2_fn) addLog(`[A2 Warning] Missed Overload on Host-${oHost} (FN)`, 'system');
    } else {
        agentA2.className = "agent-item";
        a2Status.innerText = "Monitoring CPU utilization";
    }

    // A3 Panel
    const triggerSelection = (overload_indices && overload_indices.length > 0) || 
                            (underload_indices && underload_indices.length > 0);
    if (triggerSelection) {
        agentA3.className = "agent-item active";
        const policies = ["MMT (Min Migration Time)", "Max CPU Demand", "Random Selector"];
        a3Status.innerText = `Applied: ${policies[a3_action] || 'MMT'}`;
        addLog(`[A3 Heuristic] Selected VM selection policy: ${policies[a3_action]}`, 'a3');
    } else {
        agentA3.className = "agent-item";
        a3Status.innerText = "Waiting for trigger";
    }

    // A4 Panel
    if (a4_placements && a4_placements.length > 0) {
        agentA4.className = "agent-item active";
        a4Status.innerText = `Placed ${a4_placements.length} VMs on best hosts`;
        addLog(`[A4 Placer] Actionable placements determined: ${JSON.stringify(a4_placements)}`, 'a4');
    } else {
        agentA4.className = "agent-item";
        a4Status.innerText = "Waiting for placements";
    }
}

function updateHostStates(hostsData) {
    hostsData.forEach((host, hId) => {
        // host layout: [cpu, predicted_cpu, vm_count, gpu_util, active]
        const cpuVal = host[0];
        const predVal = host[1];
        const vm_count = host[2];
        const gpu_util = host[3];
        const active = host[4] === 1;

        const meshRef = hostMeshes[hId];
        
        let color = COLORS.normal;
        if (!active) color = COLORS.inactive;
        else if (cpuVal > THRESHOLDS.critical) color = COLORS.critical;
        else if (cpuVal > THRESHOLDS.overload) color = COLORS.overload;
        else if (cpuVal < THRESHOLDS.underload) color = COLORS.underload;

        meshRef.line.material.color.setHex(color);
        meshRef.line.material.opacity = active ? 0.3 : 0.05;

        if (!active) meshRef.led.material.color.setHex(0x334155);
        else if (cpuVal > THRESHOLDS.overload) meshRef.led.material.color.setHex(0xff0000);
        else meshRef.led.material.color.setHex(0x00ff00);

        meshRef.mesh.material.opacity = active ? 0.12 : 0.03;

        updateLabelText(meshRef.label, hId, cpuVal, active);
        updateVMCubes(hId, vm_count, cpuVal, active);
    });
}

function updateVMCubes(hostId, count, cpuVal, active) {
    const meshRef = hostMeshes[hostId];
    const existingCubes = vmCubes[hostId];

    while (existingCubes.length > count) {
        const cube = existingCubes.pop();
        meshRef.group.remove(cube);
    }

    while (existingCubes.length < count) {
        const vmHeight = 0.35;
        const vmGeom = new THREE.BoxGeometry(1.1, vmHeight, 1.1);
        const vmMat = new THREE.MeshPhongMaterial({
            color: 0x3b82f6,
            transparent: true,
            opacity: 0.5,
            shininess: 90,
            specular: 0xffffff
        });
        const cubeMesh = new THREE.Mesh(vmGeom, vmMat);
        
        const slotIndex = existingCubes.length;
        const posY = -RACK_HEIGHT/2 + 0.5 + slotIndex * 0.45;
        cubeMesh.position.set(0, posY, 0);

        meshRef.group.add(cubeMesh);
        existingCubes.push(cubeMesh);
    }

    existingCubes.forEach((cube, idx) => {
        let cubeColor = 0x60a5fa;
        if (!active) cubeColor = 0x475569;
        else if (cpuVal > THRESHOLDS.critical) cubeColor = 0xfc8181;
        else if (cpuVal > THRESHOLDS.overload) cubeColor = 0xfbd38d;
        else if (cpuVal < THRESHOLDS.underload) cubeColor = 0x76e4f7;
        
        cube.material.color.setHex(cubeColor);
        cube.material.opacity = active ? 0.6 : 0.1;
        
        const pulse = 1.0 + Math.sin(Date.now() * 0.005 + idx) * 0.03;
        cube.scale.set(pulse, 1.0, pulse);
    });
}

function triggerMigration(vmId, sourceId, targetId) {
    const srcHost = hostMeshes[sourceId];
    const tgtHost = hostMeshes[targetId];

    if (!srcHost || !tgtHost) return;

    const partGeom = new THREE.SphereGeometry(0.18, 16, 16);
    const partMat = new THREE.MeshBasicMaterial({
        color: 0x10b981,
        transparent: true,
        opacity: 0.9,
    });
    const particle = new THREE.Mesh(partGeom, partMat);
    scene.add(particle);

    const p1 = new THREE.Vector3(srcHost.x, RACK_HEIGHT/2 - 0.5, srcHost.z);
    const p2 = new THREE.Vector3(tgtHost.x, RACK_HEIGHT/2 - 0.5, tgtHost.z);

    const dist = p1.distanceTo(p2);
    const peakY = Math.max(p1.y, p2.y) + dist * 0.4 + 2.0;
    const midPoint = new THREE.Vector3().addVectors(p1, p2).multiplyScalar(0.5);
    midPoint.y = peakY;

    const curve = new THREE.QuadraticBezierCurve3(p1, midPoint, p2);
    
    const trailGeom = new THREE.BufferGeometry();
    const points = curve.getPoints(24);
    trailGeom.setFromPoints(points);
    const trailMat = new THREE.LineBasicMaterial({
        color: 0x10b981,
        transparent: true,
        opacity: 0.3
    });
    const trailLine = new THREE.Line(trailGeom, trailMat);
    scene.add(trailLine);

    const animData = {
        progress: 0,
        particle: particle,
        trail: trailLine,
        curve: curve,
        tgtHost: tgtHost,
        vmId: vmId,
        sourceId: sourceId,
        targetId: targetId
    };

    migrationParticles.push(animData);
    addLog(`Migration trace: VM-${vmId} relocated from Host-${sourceId} ➜ Host-${targetId}`, 'migration');

    const srcCubes = vmCubes[sourceId];
    if (srcCubes && srcCubes.length > 0) {
        gsap.to(srcCubes[srcCubes.length - 1].scale, {
            x: 0, y: 0, z: 0,
            duration: 0.5,
            ease: "power2.in"
        });
    }

    gsap.to(animData, {
        progress: 1.0,
        duration: 1.2,
        ease: "power1.inOut",
        onUpdate: () => {
            const pos = curve.getPointAt(animData.progress);
            particle.position.copy(pos);
            const scale = 1.0 + Math.sin(animData.progress * Math.PI) * 0.5;
            particle.scale.set(scale, scale, scale);
        },
        onComplete: () => {
            scene.remove(particle);
            scene.remove(trailLine);
            
            const idx = migrationParticles.indexOf(animData);
            if (idx > -1) migrationParticles.splice(idx, 1);

            gsap.fromTo(tgtHost.group.scale, 
                { x: 1.08, y: 1.08, z: 1.08 },
                { x: 1.0, y: 1.0, z: 1.0, duration: 0.4, ease: "bounce.out" }
            );

            createSparksExplosion(p2);
        }
    });
}

function createSparksExplosion(pos) {
    const sparkCount = 8;
    const sparkMeshes = [];
    const sparkGroup = new THREE.Group();
    sparkGroup.position.copy(pos);
    scene.add(sparkGroup);

    const geom = new THREE.BoxGeometry(0.06, 0.06, 0.06);
    const mat = new THREE.MeshBasicMaterial({ color: 0x10b981 });

    for (let i = 0; i < sparkCount; i++) {
        const spark = new THREE.Mesh(geom, mat);
        const theta = Math.random() * Math.PI * 2;
        const phi = Math.acos((Math.random() * 2) - 1);
        
        spark.userData = {
            vx: Math.sin(phi) * Math.cos(theta) * 0.08,
            vy: Math.abs(Math.sin(phi) * Math.sin(theta)) * 0.08,
            vz: Math.cos(phi) * 0.08,
            life: 1.0
        };

        sparkGroup.add(spark);
        sparkMeshes.push(spark);
    }

    const animateSparks = () => {
        let alive = false;
        sparkMeshes.forEach(s => {
            if (s.userData.life > 0) {
                s.position.x += s.userData.vx;
                s.position.y += s.userData.vy;
                s.position.z += s.userData.vz;
                s.userData.vy -= 0.003;
                s.userData.life -= 0.04;
                s.scale.set(s.userData.life, s.userData.life, s.userData.life);
                alive = true;
            }
        });

        if (alive) requestAnimationFrame(animateSparks);
        else scene.remove(sparkGroup);
    };

    animateSparks();
}

function updateParticles() {}

function addLog(text, className = '') {
    const div = document.createElement('div');
    div.className = `log-line ${className}`;
    
    const timeStr = new Date().toLocaleTimeString([], { hour12: false });
    div.innerHTML = `<span class="log-time">[${timeStr}]</span> ${text}`;
    logOutput.appendChild(div);
    
    while (logOutput.children.length > 50) {
        logOutput.removeChild(logOutput.firstChild);
    }
}

function drawSparkline(history) {
    const width = sparklineCanvas.width;
    const height = sparklineCanvas.height;
    sparklineCtx.clearRect(0, 0, width, height);

    if (history.length < 2) return;

    const gradient = sparklineCtx.createLinearGradient(0, 0, 0, height);
    gradient.addColorStop(0, 'rgba(0, 242, 254, 0.15)');
    gradient.addColorStop(1, 'rgba(0, 242, 254, 0.0)');
    
    const minVal = Math.min(...history);
    const maxVal = Math.max(...history);
    const range = maxVal - minVal || 1.0;

    const getX = (idx) => (idx / (history.length - 1)) * width;
    const getY = (val) => height - 3 - ((val - minVal) / range) * (height - 6);

    sparklineCtx.beginPath();
    sparklineCtx.moveTo(getX(0), height);
    for (let i = 0; i < history.length; i++) {
        sparklineCtx.lineTo(getX(i), getY(history[i]));
    }
    sparklineCtx.lineTo(getX(history.length - 1), height);
    sparklineCtx.closePath();
    sparklineCtx.fillStyle = gradient;
    sparklineCtx.fill();

    sparklineCtx.beginPath();
    sparklineCtx.moveTo(getX(0), getY(history[0]));
    for (let i = 1; i < history.length; i++) {
        sparklineCtx.lineTo(getX(i), getY(history[i]));
    }
    sparklineCtx.strokeStyle = 'rgba(0, 242, 254, 0.8)';
    sparklineCtx.lineWidth = 1.5;
    sparklineCtx.stroke();

    const headX = getX(history.length - 1);
    const headY = getY(history[history.length - 1]);
    sparklineCtx.beginPath();
    sparklineCtx.arc(headX, headY, 2.5, 0, Math.PI * 2);
    sparklineCtx.fillStyle = '#ffffff';
    sparklineCtx.shadowColor = '#00f2fe';
    sparklineCtx.shadowBlur = 4;
    sparklineCtx.fill();
    sparklineCtx.shadowBlur = 0;
}
