#!/usr/bin/env python3
"""
Live Progress Tracker for IPCSP Single-IP vs. Decomposition Benchmarks.

Reads live_status.json / progress_tracking.csv from runs/extreme_cases_benchmark/
and displays a unified table showing:
  1) MIPGap (%): relative difference between BestObj (incumbent) and BestBd
  2) Unexp (Unexplored B&B nodes): current waiting tree nodes, peak, rate (/s),
     and trajectory (GROWING vs. SHRINKING toward zero)
  3) Explored nodes, Work units, and Sub-IP task completion.

Usage:
  uv run python track_progress.py
  uv run python track_progress.py --watch 15
  uv run python track_progress.py --history runs/extreme_cases_benchmark/p64_decomp/SrTiO3_3x3x3_P1
"""

import argparse
import glob
import json
import os
import time
import pandas as pd
from tabulate import tabulate


def load_live_statuses(runs_dir: str):
    pattern = os.path.join(runs_dir, "*", "*", "live_status.json")
    files = sorted(glob.glob(pattern))
    rows = []
    for path in files:
        try:
            with open(path, "r") as f:
                d = json.load(f)
            rel_dir = os.path.relpath(os.path.dirname(path), runs_dir)
            unexp = d.get("unexplored_nodes", 0) or 0
            peak_u = d.get("peak_unexplored_nodes", 0) or 0
            rate = d.get("unexp_rate_per_s", 0.0) or 0.0
            trend = d.get("unexp_trend", "")
            pct_peak = (100.0 * unexp / peak_u) if peak_u > 0 else 0.0

            b_obj = d.get("best_obj")
            b_bnd = d.get("best_bound")
            gap = d.get("mip_gap_pct")
            avg_sub_gap = d.get("avg_subip_gap_pct")

            gap_str = f"{gap:.2f}%" if gap is not None else "N/A"
            if avg_sub_gap is not None:
                gap_str += f" (sub:{avg_sub_gap:.2f}%)"

            rows.append({
                "Run Dir": rel_dir,
                "Test": d.get("test", ""),
                "Mode": "Decomp" if d.get("mode") == "decomposition" else "Single-IP",
                "p": d.get("p", ""),
                "Event": d.get("event", ""),
                "Wall (s)": f"{d.get('elapsed_wall_s', 0.0):.1f}",
                "BestObj": f"{b_obj:.4f}" if b_obj is not None else "None",
                "BestBd": f"{b_bnd:.4f}" if b_bnd is not None else "RootLP",
                "MIPGap (%)": gap_str,
                "Unexp Nodes": f"{int(unexp):,}",
                "Peak Unexp": f"{int(peak_u):,}",
                "Unexp Trend": f"{trend} ({rate:+.1f}/s, {pct_peak:.1f}% pk)",
                "Expl Nodes": f"{int(d.get('explored_nodes', 0) or 0):,}",
                "Tasks": f"{d.get('tasks_done', 0)}/{d.get('tasks_total', 1)} ({d.get('pruned_tasks', 0)} prn)",
                "Procs": d.get("active_workers", 0),
                "Updated (UTC)": d.get("timestamp_utc", ""),
            })
        except Exception:
            pass
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", default="runs/extreme_cases_benchmark",
                    help="Root directory containing benchmark runs")
    ap.add_argument("--history", default=None,
                    help="Optional path to a specific run directory to print its full progress_tracking.csv history")
    ap.add_argument("--watch", type=float, default=0.0,
                    help="Refresh interval in seconds (0 = print once and exit)")
    args = ap.parse_args()

    if args.history:
        csv_path = os.path.join(args.history, "progress_tracking.csv") if os.path.isdir(args.history) else args.history
        if not os.path.exists(csv_path):
            print(f"No progress_tracking.csv found at {csv_path}")
            return
        df = pd.read_csv(csv_path)
        print(tabulate(df.tail(50), headers="keys", tablefmt="github", showindex=False))
        return

    while True:
        rows = load_live_statuses(args.runs_dir)
        if args.watch > 0:
            print("\033[2J\033[H", end="")
        print(f"=== IPCSP Live Solver Progress Tracker ({time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}) ===")
        if not rows:
            print(f"No live_status.json files found yet under {args.runs_dir}.")
        else:
            print(tabulate(rows, headers="keys", tablefmt="github", showindex=False))
        if args.watch <= 0:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
