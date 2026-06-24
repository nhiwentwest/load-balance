#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/zeus/content/dacn}"
PYTHON_BIN="${PYTHON_BIN:-/home/zeus/miniconda3/envs/cloudspace/bin/python}"
STEPS="${STEPS:-30}"
OUT_CSV="${OUT_CSV:-eval_v7_azure_30step.csv}"
BRIDGE_LOG="${BRIDGE_LOG:-bridge_demo_azure_30step.log}"

cd "$ROOT"

echo "== V7 Azure short demo =="
echo "Project : $(pwd)"
echo "Steps   : $STEPS CloudSim steps"
echo "Dataset : Azure unseen VM traces"
echo

mkdir -p data
if [ ! -e data/azure_test ]; then
  ln -s azure_test_backup data/azure_test
fi

if [ ! -d data/azure_test/traces ]; then
  echo "ERROR: Missing Azure traces at $ROOT/data/azure_test/traces" >&2
  echo "Check that $ROOT/data/azure_test_backup/traces exists." >&2
  exit 1
fi

if [ ! -f target/classes/com/dacn/advanced/Py4jBridge.class ]; then
  echo "ERROR: Missing Java class target/classes/com/dacn/advanced/Py4jBridge.class" >&2
  echo "Run: mvn -q package dependency:copy-dependencies" >&2
  exit 1
fi

echo "Azure trace sample:"
ls data/azure_test/traces | head -5
echo

echo "Restarting Java Py4J bridge..."
ps -eo pid,cmd | awk '/com[.]dacn[.]advanced[.]Py4jBridge/ {print $1}' | xargs -r kill
sleep 1

nohup java -cp "target/classes:target/dependency/*" com.dacn.advanced.Py4jBridge \
  > "$BRIDGE_LOG" 2>&1 < /dev/null &

sleep 3
if ! ps -eo pid,cmd | awk '/com[.]dacn[.]advanced[.]Py4jBridge/ {found=1} END {exit !found}'; then
  echo "ERROR: Java bridge did not start. Log tail:" >&2
  tail -40 "$BRIDGE_LOG" >&2 || true
  exit 1
fi

tail -5 "$BRIDGE_LOG"
echo

echo "Running V7 checkpoint inference..."
DATASET_LABEL="Azure unseen VM traces ${STEPS}-step demo" \
OUT_CSV="$OUT_CSV" \
EVAL_EPISODES=1 \
EVAL_MAX_STEPS="$STEPS" \
"$PYTHON_BIN" -u - <<'PY'
import os
import sys

ROOT = os.environ.get("ROOT", "/home/zeus/content/dacn")
PY_DIR = os.path.join(ROOT, "python")
if PY_DIR not in sys.path:
    sys.path.insert(0, PY_DIR)

import eval_v7_demo

max_steps = int(os.environ.get("EVAL_MAX_STEPS", "30"))
orig_reset = eval_v7_demo.CloudSimEnv.reset
orig_step = eval_v7_demo.CloudSimEnv.step

def reset_limited(self, *args, **kwargs):
    self._demo_step_count = 0
    return orig_reset(self, *args, **kwargs)

def step_limited(self, *args, **kwargs):
    next_global, rewards, done, info = orig_step(self, *args, **kwargs)
    self._demo_step_count = getattr(self, "_demo_step_count", 0) + 1
    if self._demo_step_count >= max_steps:
        self.done = True
        done = True
        info = dict(info)
        info["truncated_after_steps"] = self._demo_step_count
    return next_global, rewards, done, info

eval_v7_demo.CloudSimEnv.reset = reset_limited
eval_v7_demo.CloudSimEnv.step = step_limited

print(f"[short-demo] Truncating after {max_steps} CloudSim steps.")
eval_v7_demo.run_eval(
    num_episodes=int(os.environ.get("EVAL_EPISODES", "1")),
    out_csv=os.environ.get("OUT_CSV", "eval_v7_azure_30step.csv"),
)
PY

echo
echo "CSV result:"
cat "$OUT_CSV"
echo
echo "Done. To change length, run for example: STEPS=60 ./demo_azure_30step.sh"
