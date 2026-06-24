#!/usr/bin/env python3
"""
Demo: expose the DACN hyperparameter Config state as a live Prometheus
/metrics endpoint — fully local, no Java bridge / torch required.

It loads the EDA prior (eda_data/eda_results.json) and the ORA-lite job
history (data/.../philly_data_training.csv), then serves the same
get_prometheus_metrics() text that lambda_runner exposes in production.

Run:
    python3 demo/serve_metrics.py            # serves on :8000/metrics
    PROMETHEUS_PORT=8000 python3 demo/serve_metrics.py
Then:
    curl http://127.0.0.1:8000/metrics
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY_DIR = os.path.join(ROOT, "python")
if PY_DIR not in sys.path:
    sys.path.insert(0, PY_DIR)

from config import Config
from monitoring import MonitoringSystem


def main():
    port = int(os.environ.get("PROMETHEUS_PORT", "8000"))

    # 1. EDA prior -> Config thresholds (SOURCE becomes *_EDA_PRIOR on success)
    loaded = Config.load_eda_prior()

    print("=" * 70)
    print("DACN hyperparameter evidence — Prometheus demo")
    print("=" * 70)
    print(f"  SOURCE                 = {Config.SOURCE}")
    print(f"  EDA prior loaded       = {loaded}")
    print(f"  underload  (Q25)       = {Config.UNDERLOAD_THRESHOLD:.4f}")
    print(f"  overload   (Q90)       = {Config.OVERLOAD_THRESHOLD:.4f}")
    print(f"  critical   (Q95)       = {Config.CRITICAL_OVERLOAD_THRESHOLD:.4f}")
    print(f"  autoformer seq_len     = {Config.AF_SEQ_LEN}")
    print(f"  ORA-lite history loaded= {Config.ORA_LITE_HISTORY_LOADED} "
          f"({Config.ORA_LITE_HISTORY_SAMPLE_COUNT} samples)")
    print("=" * 70)

    # 2. Live HTTP endpoint
    monitor = MonitoringSystem()
    monitor.start_http_endpoint(port)
    print(f"\nScrape me:  curl http://127.0.0.1:{port}/metrics")
    print("Ctrl-C to stop.\n")

    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        monitor.stop_http_endpoint()
        print("\nstopped.")


if __name__ == "__main__":
    main()
