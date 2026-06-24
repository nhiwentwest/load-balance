#!/usr/bin/env bash
set -euo pipefail
cd /home/zeus/content/dacn

# Kill existing
pkill -f "com.dacn.advanced.Py4jBridge" 2>/dev/null || true
pkill -f "marl_v4_train" 2>/dev/null || true
sleep 2

# Start Java bridge
echo "[run] Starting Java bridge..."
nohup java -cp "target/classes:target/dependency/*" com.dacn.advanced.Py4jBridge > bridge_train.log 2>&1 &
BRIDGE_PID=$!
echo "[run] Bridge PID=$BRIDGE_PID"
echo $BRIDGE_PID > /tmp/bridge_pid.txt

# Wait for bridge
for i in $(seq 1 30); do
  grep -q "25333" bridge_train.log 2>/dev/null && break
  sleep 1
done
tail -n 3 bridge_train.log

# Start training
echo "[run] Starting Python training (100 episodes)..."
export PYTHONPATH=python
export NUM_EPISODES=100
export FRESH_TRAIN=1
/home/zeus/miniconda3/bin/python python/marl_v4_train.py 2>&1 | tee train_100ep.log

# Cleanup
echo "[run] Training done, killing bridge..."
kill $BRIDGE_PID 2>/dev/null || true
echo "[run] TRAINING_COMPLETE"
