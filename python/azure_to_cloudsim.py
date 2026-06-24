"""
Convert Azure Public Dataset V2 (vm_cpu_readings) into CloudSim-compatible
per-VM CSV trace files.

Azure V2 schema (no header):
  col 0: timestamp (every 300s, starting 0)
  col 1: encrypted VM id
  col 2: min CPU utilization (%)
  col 3: max CPU utilization (%)
  col 4: avg CPU utilization (%)

Output: one CSV per VM with exactly 288 rows (1 day at 300s intervals),
each row = avg CPU utilization as fraction [0, 1].
Format compatible with UtilizationModelAzure.java.
"""
import csv
import os
import sys
from collections import defaultdict

def convert_azure_to_cloudsim(input_csv, output_dir, num_vms=80, min_readings=2):
    """
    Read Azure V2 CPU readings, pick VMs with enough data,
    and write per-VM CSV traces.
    
    Note: Azure V2 files are sharded by TIME, not by VM.
    Each file contains ALL VMs but only a few timestamps.
    So most VMs will have 5-10 readings per file.
    We cycle/repeat to fill 288 intervals.
    
    Args:
        input_csv: Path to raw Azure vm_cpu_readings CSV
        output_dir: Directory to write per-VM traces
        num_vms: How many VMs to extract (we need 80 for our 20-host sim)
        min_readings: Minimum readings a VM must have to be included
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Phase 1: Collect per-VM timeseries
    print(f"[Azure→CloudSim] Reading {input_csv}...")
    vm_data = defaultdict(list)  # vm_id → [(timestamp, avg_cpu), ...]
    
    with open(input_csv, 'r') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if len(row) < 5:
                continue
            try:
                ts = int(row[0])
                vm_id = row[1]
                avg_cpu = float(row[4])  # Use avg CPU utilization
                vm_data[vm_id].append((ts, avg_cpu))
            except (ValueError, IndexError):
                continue
            
            if (i + 1) % 2_000_000 == 0:
                print(f"  ... processed {i+1:,} rows, {len(vm_data)} unique VMs")
    
    print(f"[Azure→CloudSim] Total: {len(vm_data)} unique VMs found")
    
    # Phase 2: Filter VMs with enough readings and sort by most data
    qualified = {k: v for k, v in vm_data.items() if len(v) >= min_readings}
    print(f"[Azure→CloudSim] {len(qualified)} VMs have >= {min_readings} readings")
    
    # Sort by number of readings (descending), pick top N
    sorted_vms = sorted(qualified.items(), key=lambda x: len(x[1]), reverse=True)
    selected = sorted_vms[:num_vms]
    
    print(f"[Azure→CloudSim] Selected {len(selected)} VMs")
    
    # Phase 3: Write per-VM CSV files
    # Each file: exactly 288 rows, each row = avg_cpu as fraction [0, 1]
    TARGET_INTERVALS = 288
    
    for idx, (vm_id, readings) in enumerate(selected):
        # Sort by timestamp
        readings.sort(key=lambda x: x[0])
        
        # Extract CPU values and normalize to [0, 1]
        cpu_values = [r[1] / 100.0 for r in readings]  # Azure uses %, we need fraction
        
        # Take first 288 values, or pad/cycle if less
        trace = []
        for i in range(TARGET_INTERVALS):
            trace.append(cpu_values[i % len(cpu_values)])
        
        # Clamp to [0.01, 0.99]
        trace = [max(0.01, min(0.99, v)) for v in trace]
        
        # Write CSV: one column (avg_cpu), no header
        out_path = os.path.join(output_dir, f"azure_vm_{idx:03d}.csv")
        with open(out_path, 'w', newline='') as f:
            writer = csv.writer(f)
            for val in trace:
                writer.writerow([f"{val:.6f}"])
        
        if idx < 3:
            avg = sum(trace) / len(trace)
            print(f"  VM {idx}: {len(readings)} readings, avg={avg:.3f}, "
                  f"min={min(trace):.3f}, max={max(trace):.3f}")
    
    print(f"\n[Azure→CloudSim] Done! Wrote {len(selected)} VM traces to {output_dir}")
    print(f"[Azure→CloudSim] Each file: {TARGET_INTERVALS} rows × 1 column (avg CPU fraction)")
    return len(selected)


if __name__ == "__main__":
    input_csv = sys.argv[1] if len(sys.argv) > 1 else "data/azure_test/vm_cpu_readings-1.csv"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "data/azure_test/traces"
    
    n = convert_azure_to_cloudsim(input_csv, output_dir, num_vms=80, min_readings=2)
    if n < 80:
        print(f"\nWARNING: Only {n} VMs found. Need 80 for full simulation.")
