#!/usr/bin/env bash
# Train on Lightning.ai: N parallel CloudSim bridges + batched GPU + scaled hosts.
# Configurable via env: NUM_ENVS (default 4), NUM_HOSTS (default 100),
# NUM_EPISODES (default 100), BRIDGE_BASE_PORT (default 25333).
set -uo pipefail
cd "$(dirname "$0")"

PYBIN="${PYBIN:-python3}"
NUM_ENVS="${NUM_ENVS:-4}"
NUM_HOSTS="${NUM_HOSTS:-100}"
NUM_EPISODES="${NUM_EPISODES:-100}"
BASE_PORT="${BRIDGE_BASE_PORT:-25333}"
export NUM_ENVS NUM_HOSTS NUM_EPISODES
export BRIDGE_BASE_PORT="$BASE_PORT"

echo "[run] NUM_ENVS=$NUM_ENVS NUM_HOSTS=$NUM_HOSTS NUM_EPISODES=$NUM_EPISODES base_port=$BASE_PORT"

# Clean up any stale processes
pkill -f "com.dacn.advanced.Py4jBridge" 2>/dev/null || true
pkill -f "marl_v4_train" 2>/dev/null || true
pkill -f "resource_monitor" 2>/dev/null || true
sleep 2

# Launch one Java bridge per env, each on its own port, each with NUM_HOSTS.
BRIDGE_PIDS=()
for i in $(seq 0 $((NUM_ENVS - 1))); do
  PORT=$((BASE_PORT + i))
  echo "[run] starting bridge $i on port $PORT (NUM_HOSTS=$NUM_HOSTS)..."
  NUM_HOSTS="$NUM_HOSTS" java -cp "target/classes:target/dependency/*" \
    com.dacn.advanced.Py4jBridge "$PORT" > "bridge_${PORT}.log" 2>&1 < /dev/null &
  BRIDGE_PIDS+=($!)
done

# Wait for every bridge to report its port up
for i in $(seq 0 $((NUM_ENVS - 1))); do
  PORT=$((BASE_PORT + i))
  for t in $(seq 1 40); do
    grep -q "$PORT" "bridge_${PORT}.log" 2>/dev/null && break
    sleep 1
  done
  tail -n 1 "bridge_${PORT}.log" 2>/dev/null || true
done

echo "[run] starting resource monitor..."
$PYBIN python/resource_monitor.py \
  --out resource_usage.csv --summary resource_summary.json --interval 5 \
  > resource_monitor.log 2>&1 < /dev/null &
RESMON_PID=$!

echo "[run] starting training..."
export PYTHONPATH=python
export FRESH_TRAIN=1
$PYBIN python/marl_v4_train.py > train_100ep.log 2>&1
TRAIN_RC=$?

echo "[run] training rc=$TRAIN_RC — stopping monitor + bridges"
kill -TERM "$RESMON_PID" 2>/dev/null || true
sleep 6
for pid in "${BRIDGE_PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done

echo "[run] ===== RESOURCE SUMMARY ====="
cat resource_summary.json 2>/dev/null || echo "(no summary)"
echo "TRAINING_COMPLETE rc=$TRAIN_RC" | tee -a train_100ep.log
