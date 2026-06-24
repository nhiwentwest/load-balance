#!/usr/bin/env python3
"""
resource_monitor.py — lightweight CPU/GPU/RAM sampler for the training run.

Runs as a standalone background process during MARL training and records
per-sample system resource usage to a CSV, then prints a peak/average summary
that can be quoted directly in the report.

GPU stats come from `nvidia-smi` (always present on a CUDA box, no pip dep).
CPU/RAM come from psutil if available, otherwise from /proc.

Usage:
    python3 python/resource_monitor.py --out resource_usage.csv --interval 5
    # stop with Ctrl-C or SIGTERM; summary is written to resource_summary.json

Columns:
    ts, cpu_pct, ram_used_mb, ram_total_mb, ram_pct,
    gpu_idx, gpu_name, gpu_util_pct, gpu_mem_used_mb, gpu_mem_total_mb
"""
import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time

try:
    import psutil
except Exception:
    psutil = None


_RUNNING = True


def _stop(signum, frame):
    global _RUNNING
    _RUNNING = False


def read_cpu_ram():
    if psutil is not None:
        vm = psutil.virtual_memory()
        return (
            float(psutil.cpu_percent(interval=None)),
            vm.used / 1e6,
            vm.total / 1e6,
            float(vm.percent),
        )
    # /proc fallback (no per-call cpu%, report load-based estimate)
    used_mb = total_mb = pct = 0.0
    try:
        with open("/proc/meminfo") as fh:
            mem = {}
            for line in fh:
                k, _, v = line.partition(":")
                mem[k.strip()] = float(v.strip().split()[0]) / 1024.0
        total_mb = mem.get("MemTotal", 0.0)
        avail_mb = mem.get("MemAvailable", 0.0)
        used_mb = total_mb - avail_mb
        pct = 100.0 * used_mb / total_mb if total_mb else 0.0
    except Exception:
        pass
    cpu = 0.0
    try:
        cpu = 100.0 * os.getloadavg()[0] / (os.cpu_count() or 1)
    except Exception:
        pass
    return cpu, used_mb, total_mb, pct


def read_gpus():
    """Return list of dicts via nvidia-smi; empty list if no GPU."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode != 0:
            return []
        gpus = []
        for line in out.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 5:
                continue
            gpus.append({
                "idx": int(parts[0]),
                "name": parts[1],
                "util": float(parts[2]),
                "mem_used": float(parts[3]),
                "mem_total": float(parts[4]),
            })
        return gpus
    except Exception:
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="resource_usage.csv")
    ap.add_argument("--summary", default="resource_summary.json")
    ap.add_argument("--interval", type=float, default=5.0)
    args = ap.parse_args()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    if psutil is not None:
        psutil.cpu_percent(interval=None)  # prime the counter

    gpus0 = read_gpus()
    gpu_names = sorted({g["name"] for g in gpus0})
    gpu_count = len(gpus0)
    print(f"[resmon] GPUs detected: {gpu_count} {gpu_names}")
    print(f"[resmon] CPU cores: {os.cpu_count()}  psutil={'yes' if psutil else 'no'}")
    print(f"[resmon] sampling every {args.interval}s -> {args.out}")

    agg = {
        "cpu_pct": [], "ram_used_mb": [], "ram_pct": [],
        "gpu_util_pct": [], "gpu_mem_used_mb": [],
    }
    ram_total_mb = 0.0
    gpu_mem_total_mb = 0.0
    n_samples = 0

    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "ts", "cpu_pct", "ram_used_mb", "ram_total_mb", "ram_pct",
            "gpu_idx", "gpu_name", "gpu_util_pct",
            "gpu_mem_used_mb", "gpu_mem_total_mb",
        ])
        while _RUNNING:
            ts = time.time()
            cpu, ram_used, ram_total, ram_pct = read_cpu_ram()
            ram_total_mb = ram_total
            gpus = read_gpus()
            agg["cpu_pct"].append(cpu)
            agg["ram_used_mb"].append(ram_used)
            agg["ram_pct"].append(ram_pct)
            if gpus:
                for g in gpus:
                    gpu_mem_total_mb = max(gpu_mem_total_mb, g["mem_total"])
                    agg["gpu_util_pct"].append(g["util"])
                    agg["gpu_mem_used_mb"].append(g["mem_used"])
                    w.writerow([
                        f"{ts:.0f}", f"{cpu:.1f}", f"{ram_used:.0f}",
                        f"{ram_total:.0f}", f"{ram_pct:.1f}",
                        g["idx"], g["name"], f"{g['util']:.0f}",
                        f"{g['mem_used']:.0f}", f"{g['mem_total']:.0f}",
                    ])
            else:
                w.writerow([
                    f"{ts:.0f}", f"{cpu:.1f}", f"{ram_used:.0f}",
                    f"{ram_total:.0f}", f"{ram_pct:.1f}",
                    -1, "none", 0, 0, 0,
                ])
            fh.flush()
            n_samples += 1
            time.sleep(args.interval)

    def stats(key):
        v = agg[key]
        if not v:
            return {"avg": 0.0, "peak": 0.0}
        return {"avg": round(sum(v) / len(v), 2), "peak": round(max(v), 2)}

    summary = {
        "samples": n_samples,
        "interval_sec": args.interval,
        "cpu_cores": os.cpu_count(),
        "ram_total_mb": round(ram_total_mb, 0),
        "gpu_count": gpu_count,
        "gpu_names": gpu_names,
        "gpu_mem_total_mb": round(gpu_mem_total_mb, 0),
        "cpu_pct": stats("cpu_pct"),
        "ram_used_mb": stats("ram_used_mb"),
        "ram_pct": stats("ram_pct"),
        "gpu_util_pct": stats("gpu_util_pct"),
        "gpu_mem_used_mb": stats("gpu_mem_used_mb"),
    }
    with open(args.summary, "w") as fh:
        json.dump(summary, fh, indent=2)
    print("\n[resmon] ===== RESOURCE SUMMARY =====")
    print(json.dumps(summary, indent=2))
    print(f"[resmon] CSV: {args.out}  JSON: {args.summary}")


if __name__ == "__main__":
    main()
