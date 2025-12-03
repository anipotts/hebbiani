"""
Analyze calibration (ECE) from experiment logs.
"""

import argparse
import json
import os
import glob
from collections import defaultdict
from typing import List, Dict, Any

from consensus_core import EvalRecord
from metrics import calculate_ece, calibration_bins


def load_records(paths: List[str]) -> List[EvalRecord]:
    records = []
    for path in paths:
        if not os.path.exists(path):
            continue
        print(f"Loading {path}...")
        with open(path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    # Ensure new fields exist if loading older logs
                    data.setdefault("confidence_score", None)
                    data.setdefault("calibrated_correct", None)
                    data.setdefault("raw_reasoning", None)
                    
                    # Reconstruct EvalRecord
                    # Filter out extra keys not in dataclass if any
                    valid_keys = EvalRecord.__annotations__.keys()
                    filtered_data = {k: v for k, v in data.items() if k in valid_keys}
                    
                    records.append(EvalRecord(**filtered_data))
                except Exception as e:
                    print(f"Skipping bad line in {path}: {e}")
    return records


def analyze_calibration(records: List[EvalRecord], output_dir: str):
    # Group by agent
    by_agent = defaultdict(list)
    for r in records:
        by_agent[r.agent_name].append(r)

    ece_summary = []
    
    print("\n=== CALIBRATION ANALYSIS (ECE) ===")
    
    for agent_name, agent_records in by_agent.items():
        y_true = []
        y_prob = []
        
        for r in agent_records:
            if r.confidence_score is not None:
                # y_true is 1 if normalized_match is True (approx correct), else 0
                # Note: normalized_match handles approximate string matching against ground truth
                is_correct = 1 if r.normalized_match else 0
                y_true.append(is_correct)
                y_prob.append(r.confidence_score)
        
        if not y_true:
            print(f"Agent {agent_name}: No confidence scores found.")
            continue
            
        ece = calculate_ece(y_true, y_prob)
        print(f"Agent {agent_name}: ECE = {ece:.4f} (n={len(y_true)})")
        
        ece_summary.append({
            "agent_name": agent_name,
            "ece": ece,
            "num_points": len(y_true),
            "n_bins": 10
        })
        
        # Calculate bins for plotting
        bins_data = calibration_bins(y_true, y_prob, n_bins=10)
        
        # Save bins
        bin_file = os.path.join(output_dir, f"calibration_bins_{agent_name}.json")
        with open(bin_file, "w") as f:
            json.dump(bins_data, f, indent=2)

    # Save ECE summary
    summary_file = os.path.join(output_dir, "calibration_summary.json")
    with open(summary_file, "w") as f:
        json.dump(ece_summary, f, indent=2)
    print(f"\nCalibration data saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Compute ECE and calibration stats.")
    parser.add_argument(
        "--logs", 
        type=str, 
        nargs="+", 
        default=["experiment_logs/run_records_rabbitmq.jsonl"],
        help="Paths to run_records.jsonl files (globs allowed)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiment_logs",
        help="Output directory for calibration stats"
    )
    args = parser.parse_args()

    # Expand globs
    expanded_paths = []
    for p in args.logs:
        expanded_paths.extend(glob.glob(p))
    
    if not expanded_paths:
        # Fallback to default locations if nothing found
        defaults = [
            "experiment_logs/run_records_rabbitmq.jsonl",
            "experiment_logs/sync_run_records.jsonl",
            "experiment_logs/async_run_records.jsonl"
        ]
        expanded_paths = [p for p in defaults if os.path.exists(p)]

    if not expanded_paths:
        print("No log files found to analyze.")
        return

    records = load_records(expanded_paths)
    print(f"Loaded {len(records)} records.")
    
    analyze_calibration(records, args.output)


if __name__ == "__main__":
    main()



