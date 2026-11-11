#!/usr/bin/env python3
"""
Merges individual p-task benchmark results (from runs/decomposition_benchmark/p*)
into unified benchmark CSV files and calculates speedups / parallel efficiencies.
Supports partial CSVs if some runs are still in progress.
"""

import argparse
import glob
import os
import pandas as pd
from tabulate import tabulate


def merge_results(
    runs_dir: str = "runs/table1_benchmark",
    output_dir: str = "results/table1_benchmark",
    ignore_p: list[int] | None = None,
):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Resolve target run directory
    target_dir = os.path.abspath(runs_dir)
    if not os.path.isdir(target_dir):
        alt = os.path.join(base_dir, runs_dir)
        if os.path.isdir(alt):
            target_dir = alt
        else:
            print(f"Error: run directory '{runs_dir}' not found.")
            return

    p_dirs = sorted(
        [
            d
            for d in glob.glob(os.path.join(target_dir, "*"))
            if os.path.isdir(d) and ("p" in os.path.basename(d).lower() or "decomp" in os.path.basename(d).lower())
        ],
        key=lambda d: (
            int("".join(filter(str.isdigit, os.path.basename(d))))
            if any(c.isdigit() for c in os.path.basename(d))
            else 999999
        ),
    )

    if ignore_p:
        ignore_set = set(ignore_p)
        p_dirs = [
            d
            for d in p_dirs
            if not (
                os.path.basename(d).startswith("p")
                and os.path.basename(d)[1:].isdigit()
                and int(os.path.basename(d)[1:]) in ignore_set
            )
        ]

    if not p_dirs:
        print(f"No p-directories found in: {target_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)

    summary_list = []
    solve_only_list = []
    procs_list = []
    tasks_list = []
    single_ip_list = []

    for d in p_dirs:
        # Support both direct p{P}/*.csv and per-structure p{P}/{structure}/*.csv
        candidate_dirs = [d] + sorted(
            [sd for sd in glob.glob(os.path.join(d, "*")) if os.path.isdir(sd)]
        )
        for cd in candidate_dirs:
            sum_f = os.path.join(cd, "summary.csv")
            solve_f = os.path.join(cd, "solve_only_comparison.csv")
            procs_f = os.path.join(cd, "processors.csv")
            tasks_f = os.path.join(cd, "tasks.csv")
            single_f = os.path.join(cd, "original_single_ip.csv")

            if os.path.exists(sum_f) and os.path.getsize(sum_f) > 10:
                try:
                    summary_list.append(pd.read_csv(sum_f))
                except Exception as e:
                    print(f"Warning: could not read {sum_f}: {e}")
            if os.path.exists(solve_f) and os.path.getsize(solve_f) > 10:
                try:
                    solve_only_list.append(pd.read_csv(solve_f))
                except Exception as e:
                    print(f"Warning: could not read {solve_f}: {e}")
            if os.path.exists(procs_f) and os.path.getsize(procs_f) > 10:
                try:
                    procs_list.append(pd.read_csv(procs_f))
                except Exception as e:
                    print(f"Warning: could not read {procs_f}: {e}")
            if os.path.exists(tasks_f) and os.path.getsize(tasks_f) > 10:
                try:
                    tasks_list.append(pd.read_csv(tasks_f))
                except Exception as e:
                    print(f"Warning: could not read {tasks_f}: {e}")
            if os.path.exists(single_f) and os.path.getsize(single_f) > 10:
                try:
                    single_ip_list.append(pd.read_csv(single_f))
                except Exception as e:
                    print(f"Warning: could not read {single_f}: {e}")

    if not summary_list:
        print(
            f"No summary CSVs found yet across {len(p_dirs)} directories in {runs_dir}."
        )
        return

    df_summary = pd.concat(summary_list, ignore_index=True)
    df_summary = df_summary.drop_duplicates(subset=["test", "p"], keep="last")
    df_summary["p"] = pd.to_numeric(df_summary["p"])
    df_summary = df_summary.sort_values(by=["test", "p"]).reset_index(drop=True)

    # Compute speedup vs p=1 baseline and vs single Gurobi instance with p threads
    p1_rows = df_summary[df_summary["p"] == 1].set_index("test")
    speedups_vs_single_p_threads = []
    speedups_vs_single_p_threads_wall = []
    speedups_vs_p1 = []
    efficiencies_pct = []

    for _, row in df_summary.iterrows():
        t = row["test"]
        p = int(row["p"])
        par_time = float(row.get("parallel_time_s", float("nan")))
        par_wall = float(row.get("parallel_wall_s", float("nan")))
        single_solve = float(row.get("original_single_ip_solve_time_s", float("nan")))
        single_wall = float(row.get("original_single_ip_total_wall_s", float("nan")))

        sp_single = (single_solve / par_time) if par_time > 0 else float("nan")
        sp_single_wall = (single_wall / par_wall) if par_wall > 0 else float("nan")
        speedups_vs_single_p_threads.append(sp_single)
        speedups_vs_single_p_threads_wall.append(sp_single_wall)

        if t in p1_rows.index:
            p1_time = float(p1_rows.loc[t].get("parallel_time_s", float("nan")))
            sp = (p1_time / par_time) if par_time > 0 else float("nan")
            eff = (sp / p) * 100.0 if p > 0 and not pd.isna(sp) else float("nan")
        else:
            sp = float("nan")
            eff = float("nan")
        speedups_vs_p1.append(sp)
        efficiencies_pct.append(eff)

    df_summary["speedup_vs_single_gurobi_p_threads"] = speedups_vs_single_p_threads
    df_summary["speedup_vs_single_gurobi_p_threads_wall"] = speedups_vs_single_p_threads_wall
    df_summary["speedup_vs_p1"] = speedups_vs_p1
    df_summary["parallel_efficiency_pct"] = efficiencies_pct

    # Time in seconds to first reach optimality
    if "decomp_solve_to_opt_s" in df_summary.columns:
        df_summary["Time in seconds to first reach optimality"] = df_summary["decomp_solve_to_opt_s"]
    elif "decomp_wall_to_opt_s" in df_summary.columns:
        df_summary["Time in seconds to first reach optimality"] = df_summary["decomp_wall_to_opt_s"]
    else:
        df_summary["Time in seconds to first reach optimality"] = df_summary.get("parallel_time_s", float("nan"))
    df_summary["time_to_first_opt_s"] = df_summary["Time in seconds to first reach optimality"]

    # Save merged files
    merged_sum_path = os.path.join(output_dir, "summary.csv")
    df_summary.to_csv(merged_sum_path, index=False)

    if solve_only_list:
        df_solve = pd.concat(solve_only_list, ignore_index=True).drop_duplicates(
            subset=["test", "p"], keep="last"
        )
        df_solve["p"] = pd.to_numeric(df_solve["p"])
        df_solve = df_solve.sort_values(by=["test", "p"]).reset_index(drop=True)
        df_solve["speedup_vs_single_gurobi_p_threads"] = speedups_vs_single_p_threads
        df_solve["speedup_vs_p1"] = speedups_vs_p1
        df_solve["parallel_efficiency_pct"] = efficiencies_pct
        if "decomp_solve_to_opt_s" in df_solve.columns:
            df_solve["Time in seconds to first reach optimality"] = df_solve["decomp_solve_to_opt_s"]
        elif "decomp_wall_to_opt_s" in df_solve.columns:
            df_solve["Time in seconds to first reach optimality"] = df_solve["decomp_wall_to_opt_s"]
        else:
            df_solve["Time in seconds to first reach optimality"] = df_solve.get("parallel_makespan_solve_s", float("nan"))
        df_solve["time_to_first_opt_s"] = df_solve["Time in seconds to first reach optimality"]
        df_solve.to_csv(
            os.path.join(output_dir, "solve_only_comparison.csv"), index=False
        )

    if procs_list:
        df_procs = pd.concat(procs_list, ignore_index=True)
        df_procs["p"] = pd.to_numeric(df_procs["p"])
        df_procs = df_procs.sort_values(by=["test", "p", "processor"]).reset_index(
            drop=True
        )
        df_procs.to_csv(os.path.join(output_dir, "processors.csv"), index=False)

    if tasks_list:
        df_tasks = pd.concat(tasks_list, ignore_index=True)
        df_tasks["p"] = pd.to_numeric(df_tasks["p"])
        df_tasks = df_tasks.sort_values(by=["test", "p", "task_id"]).reset_index(
            drop=True
        )
        df_tasks.to_csv(os.path.join(output_dir, "tasks.csv"), index=False)

    if single_ip_list:
        df_single = pd.concat(single_ip_list, ignore_index=True).drop_duplicates(
            subset=["test", "p"], keep="last"
        )
        df_single["p"] = pd.to_numeric(df_single["p"])
        df_single = df_single.sort_values(by=["test", "p"]).reset_index(drop=True)
        df_single.to_csv(
            os.path.join(output_dir, "original_single_ip.csv"), index=False
        )

    print("=" * 110)
    print("  CONSOLIDATED IPCSP DECOMPOSITION BENCHMARK RESULTS")
    print("=" * 110)

    display_cols = [
        "test",
        "p",
        "original_single_ip_solve_time_s",
        "num_subIPs",
        "avg_subIP_solve_s",
        "max_subIP_solve_s",
        "parallel_time_s",
        "Time in seconds to first reach optimality",
        "speedup_vs_single_gurobi_p_threads",
        "speedup_vs_p1",
        "parallel_efficiency_pct",
    ]
    cols = [c for c in display_cols if c in df_summary.columns]
    print(
        tabulate(
            df_summary[cols],
            headers="keys",
            tablefmt="github",
            showindex=False,
            floatfmt=".4f",
        )
    )
    print("\nMerged CSVs written to:", output_dir)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--runs-dir",
        default="runs/table1_benchmark",
        help="Directory containing per-p runs (default: runs/table1_benchmark)",
    )
    ap.add_argument(
        "--output-dir",
        default="results/table1_benchmark",
        help="Directory to save consolidated CSVs (default: results/table1_benchmark)",
    )
    ap.add_argument(
        "--ignore-p",
        type=int,
        nargs="+",
        default=None,
        help="List of p values to ignore (e.g. --ignore-p 1)",
    )
    args = ap.parse_args()
    merge_results(args.runs_dir, args.output_dir, ignore_p=args.ignore_p)


if __name__ == "__main__":
    main()
