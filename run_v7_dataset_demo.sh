#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/zeus/content/dacn"
PY="/home/zeus/miniconda3/envs/cloudspace/bin/python"
JAVA_CP="target/classes:target/dependency/*"
EPISODES="${EVAL_EPISODES:-3}"

cd "$ROOT"

stop_bridge() {
  pkill -f "com.dacn.advanced.Py4jBridge" 2>/dev/null || true
  sleep 1
}

start_bridge() {
  local bridge_log="$1"
  nohup java -cp "$JAVA_CP" com.dacn.advanced.Py4jBridge > "$bridge_log" 2>&1 < /dev/null &
  echo "[demo] started Py4J bridge pid=$! log=$bridge_log"
  sleep 3
  if ! pgrep -f "com.dacn.advanced.Py4jBridge" >/dev/null; then
    echo "[demo] bridge failed to start; tailing log:"
    tail -80 "$bridge_log" || true
    exit 1
  fi
}

select_philly() {
  if [ -L data/azure_test ]; then
    unlink data/azure_test
  fi
  echo "[demo] dataset=Philly"
  echo "[demo] source=data/Gen-Parallel-Workloads/Philly/training_data/philly_data_training.csv"
  echo "[demo] Philly csv files=$(find data/Gen-Parallel-Workloads/Philly/training_data -type f -name '*.csv' | wc -l)"
}

select_azure() {
  if [ ! -e data/azure_test ]; then
    ln -s azure_test_backup data/azure_test
  fi
  echo "[demo] dataset=Azure"
  echo "[demo] source=data/azure_test_backup/traces/azure_vm_*.csv"
  echo "[demo] Azure trace files=$(find data/azure_test_backup/traces -type f -name '*.csv' | wc -l)"
}

run_eval() {
  local label="$1"
  local out_csv="$2"
  local out_log="$3"
  local bridge_log="$4"

  stop_bridge
  start_bridge "$bridge_log"
  DATASET_LABEL="$label" OUT_CSV="$out_csv" EVAL_EPISODES="$EPISODES" \
    "$PY" python/eval_v7_demo.py | tee "$out_log"
  stop_bridge
}

select_philly
run_eval "Philly Gen-Parallel-Workloads trace (stress/bursty)" \
  "eval_v7_philly_demo.csv" \
  "eval_v7_philly_demo.log" \
  "bridge_eval_v7_philly.log"

select_azure
run_eval "Azure unseen VM traces (generalization)" \
  "eval_v7_azure_demo.csv" \
  "eval_v7_azure_demo.log" \
  "bridge_eval_v7_azure.log"

"$PY" - <<'PY'
import csv
from pathlib import Path

root = Path("/home/zeus/content/dacn")
inputs = [
    ("Philly", root / "eval_v7_philly_demo.csv"),
    ("Azure", root / "eval_v7_azure_demo.csv"),
]
rows = []
for dataset, path in inputs:
    with path.open() as f:
        data = list(csv.DictReader(f))
    def avg(key):
        return sum(float(r[key]) for r in data) / max(1, len(data))
    rows.append({
        "dataset": dataset,
        "episodes": len(data),
        "avg_reward": f"{avg('total_reward'):.4f}",
        "avg_energy_kwh": f"{avg('energy_kwh'):.4f}",
        "avg_slatah": f"{avg('slatah'):.8f}",
        "avg_migrations": f"{avg('migrations'):.4f}",
        "total_failures": str(sum(int(float(r["failures"])) for r in data)),
        "avg_overloads": f"{avg('overloads'):.4f}",
        "avg_underloads": f"{avg('underloads'):.4f}",
    })

out = root / "eval_v7_dataset_comparison.csv"
with out.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print("=" * 80)
print("DATASET COMPARISON")
for row in rows:
    print(row)
print(f"comparison_csv={out}")
print("=" * 80)
PY
