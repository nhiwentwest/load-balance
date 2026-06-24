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
sleep 5

# Start training
echo "[run] Starting Python training..."
export PYTHONPATH=python
export NUM_EPISODES=100
export FRESH_TRAIN=1
/home/zeus/miniconda3/bin/python python/marl_v4_train.py 2>&1 | tee train_fixed.log

# Cleanup
echo "[run] Training done, killing bridge..."
kill $BRIDGE_PID 2>/dev/null || true
echo "[run] Complete."
