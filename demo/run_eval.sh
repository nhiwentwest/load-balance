#!/usr/bin/env bash
# Demo: test 4 agent đã train (deterministic inference) trên CloudSim, LOCAL.
#
# Tự khởi động Java CloudSim bridge -> chạy eval_v7_demo.py -> tắt bridge.
# Lý do cần wrapper: eval gọi env.close() làm bridge tự thoát, nên mỗi lần
# chạy cần một bridge mới.
#
# Cách dùng:
#   bash demo/run_eval.sh            # 3 episodes (mặc định)
#   EVAL_EPISODES=5 bash demo/run_eval.sh
#
# Yêu cầu (đã cài sẵn local): java, torch, py4j, gymnasium, numpy.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

EPISODES="${EVAL_EPISODES:-3}"
BRIDGE_PORT="${BRIDGE_PORT:-25333}"
BRIDGE_LOG="/tmp/dacn_bridge_eval.log"

echo "=================================================================="
echo "DACN model eval demo — $EPISODES episode(s), bridge port $BRIDGE_PORT"
echo "=================================================================="

# Dọn bridge cũ nếu còn sót trên cổng này
if lsof -tiTCP:"$BRIDGE_PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "[demo] killing stale bridge on :$BRIDGE_PORT"
  lsof -tiTCP:"$BRIDGE_PORT" -sTCP:LISTEN | xargs kill 2>/dev/null || true
  sleep 2
fi

echo "[demo] starting Java CloudSim bridge..."
BRIDGE_PORT="$BRIDGE_PORT" java -cp "target/classes:target/dependency/*" \
  com.dacn.advanced.Py4jBridge "$BRIDGE_PORT" > "$BRIDGE_LOG" 2>&1 &
BRIDGE_PID=$!

cleanup() {
  if kill -0 "$BRIDGE_PID" 2>/dev/null; then
    kill "$BRIDGE_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

# Đợi bridge sẵn sàng (tối đa 30s)
for i in $(seq 1 30); do
  if grep -q "Waiting for Python" "$BRIDGE_LOG" 2>/dev/null; then
    echo "[demo] bridge ready."
    break
  fi
  sleep 1
done

echo "[demo] running eval_v7_demo.py..."
echo "------------------------------------------------------------------"
EVAL_EPISODES="$EPISODES" \
DATASET_LABEL="${DATASET_LABEL:-Philly local demo}" \
OUT_CSV="${OUT_CSV:-eval_local_demo.csv}" \
BRIDGE_PORT="$BRIDGE_PORT" \
python3 python/eval_v7_demo.py

echo "------------------------------------------------------------------"
echo "[demo] done. CSV: $ROOT/${OUT_CSV:-eval_local_demo.csv}"
