"""
Benchmark: single large integer program (Gurobi single instance with p threads)
vs. Parikh decomposition into sub-IPs scheduled with LPT over p processors
(each running an independent Gurobi instance with 1 thread).

For every structure it reports
  * baseline: build time, Gurobi solve time (with p threads) and size of the single IP
  * decomposition: number of sub-IPs, average solve time per sub-IP,
    parallel time (= makespan, the most loaded processor's solve time),
    and how the sub-IPs / configurations are split over the p processors.

CSV files are written to results/decomposition_benchmark/:
  summary.csv     one row per (structure, p)
  processors.csv  one row per processor
  tasks.csv       one row per sub-IP

Usage examples:
  .venv/bin/python benchmark_decomposition.py
  .venv/bin/python benchmark_decomposition.py --tests Y2O3_1 MgAl2O4_1 --p 2 4 8
  .venv/bin/python benchmark_decomposition.py --tests SrTiO3_1 --time-limit 60
"""

import argparse
import math
import os
import time

import pandas as pd
from tabulate import tabulate

from ipcsp.integer_program import Allocate
from ipcsp.matrix_generator import Phase
from ipcsp.parallel_search import run_parallel_search

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "results", "decomposition_benchmark"
)

# Same structures / parameters as test_parallel.py (Table 1 of the paper)
TEST_CASES = {
    "SrTiO3_1": dict(phase="SrTiO", ions={"O": 3, "Sr": 1, "Ti": 1},
                     grid=4, cell=3.9, group="1"),
    "Y2O3_1": dict(phase="YSrTiO", ions={"O": 48, "Y": 32},
                   grid=8, cell=10.7, group="206"),
    "Y2Ti2O7_1": dict(phase="YSrTiO", ions={"O": 56, "Y": 16, "Ti": 16},
                      grid=8, cell=10.2, group="227"),
    "MgAl2O4_1": dict(phase="LiMgAlPO", ions={"O": 32, "Mg": 8, "Al": 16},
                      grid=8, cell=8.2, group="227"),
    "Ca3Al2Si3O12_2": dict(phase="CaAlSiO",
                           ions={"Ca": 24, "Al": 16, "Si": 24, "O": 96},
                           grid=8, cell=11.9, group="206"),
}


def sci(n):
    """Format a (possibly huge) integer in scientific notation."""
    if n == 0:
        return "0"
    e = int(math.floor(math.log10(n)))
    mant = n / 10**e if e < 300 else int(str(n)[:4]) / 1000
    return f"{mant:.3f}e{e}"


def ratio(a, b):
    """a/b for arbitrarily large ints."""
    return a / b if b else float("nan")


def run_baseline(name, tc, time_limit, threads):
    phase = Phase(tc["phase"])
    alloc = Allocate(tc["ions"], tc["grid"], tc["cell"], phase)
    t0 = time.perf_counter()
    res = alloc.optimize_cube_symmetry_ase(
        group=tc["group"], PoolSolutions=1, TimeLimit=time_limit,
        verbose=False, threads=threads, write_lp=False,
    )
    wall = time.perf_counter() - t0
    st = alloc.last_stats
    return {
        "energy": res[2] if res else None,
        "build_time": st["build_time"],
        "solve_time": st["solve_time"],
        "wall_time": wall,
        "num_vars": st["num_vars"],
        "num_constrs": st["num_constrs"],
        "num_qterms": st["num_qterms"],
        "status": st["status"],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tests", nargs="+", default=list(TEST_CASES),
                    choices=list(TEST_CASES))
    ap.add_argument("--p", nargs="+", type=int, default=[4],
                    help="number(s) of processors")
    ap.add_argument("--R-target", type=float, default=1.5)
    ap.add_argument("--max-tasks", type=int, default=None,
                    help="cap on generated sub-IPs (default: max(p, 2*#ion types))")
    ap.add_argument("--time-limit", type=float, default=0,
                    help="Gurobi TimeLimit per IP in seconds (0 = none)")
    ap.add_argument("--baseline-threads", type=int, default=None,
                    help="Gurobi threads for the single IP (default: matches p, "
                         "so single IP with p threads vs. p independent Gurobis)")
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary_rows, proc_rows, task_rows, single_ip_rows = [], [], [], []
    baseline_cache = {}

    def save_csvs():
        df_sum = pd.DataFrame(summary_rows)
        df_procs = pd.DataFrame(proc_rows)
        df_tasks = pd.DataFrame(task_rows)
        df_single = pd.DataFrame(single_ip_rows)

        df_sum.to_csv(os.path.join(RESULTS_DIR, "summary.csv"), index=False)
        df_procs.to_csv(os.path.join(RESULTS_DIR, "processors.csv"), index=False)
        df_tasks.to_csv(os.path.join(RESULTS_DIR, "tasks.csv"), index=False)
        df_single.to_csv(os.path.join(RESULTS_DIR, "original_single_ip.csv"), index=False)

        # Solve-only views (pure solver time with build phase excluded)
        if not df_sum.empty:
            solve_cols = [
                "test", "grid", "group", "p", "baseline_threads",
                "original_single_ip_solve_time_s",
                "original_single_ip_vars", "original_single_ip_constrs", "original_single_ip_energy",
                "total_configurations_sci", "num_subIPs",
                "avg_subIP_solve_s", "max_subIP_solve_s",
                "parallel_time_s", "sum_subIP_solve_s",
                "speedup_solve", "decomp_energy", "energy_match",
                "subIP_vars", "subIP_constrs_avg", "load_imbalance_max_over_avg"
            ]
            solve_only_df = df_sum[[c for c in solve_cols if c in df_sum.columns]].rename(
                columns={
                    "parallel_time_s": "parallel_makespan_solve_s",
                    "avg_subIP_solve_s": "avg_individual_subIP_solve_s",
                    "max_subIP_solve_s": "max_individual_subIP_solve_s",
                    "sum_subIP_solve_s": "sum_all_subIP_solve_s",
                    "speedup_solve": "solve_speedup"
                }
            )
            solve_only_df.to_csv(os.path.join(RESULTS_DIR, "solve_only_comparison.csv"), index=False)

        if not df_procs.empty:
            proc_cols = ["test", "p", "processor", "num_IPs", "IP_ratio", "configurations_sci", "config_ratio", "solve_time_s", "is_bottleneck"]
            df_procs[[c for c in proc_cols if c in df_procs.columns]].to_csv(
                os.path.join(RESULTS_DIR, "solve_only_processors.csv"), index=False
            )

        if not df_tasks.empty:
            task_cols = ["test", "p", "task_id", "processor", "prefix", "configurations_sci", "config_ratio", "solve_time_s", "status", "energy", "num_vars", "num_constrs", "nodes"]
            df_tasks[[c for c in task_cols if c in df_tasks.columns]].to_csv(
                os.path.join(RESULTS_DIR, "solve_only_tasks.csv"), index=False
            )

    for name in args.tests:
        tc = TEST_CASES[name]
        print(f"\n{'=' * 30} {name} {'=' * 30}")

        for p in args.p:
            threads = args.baseline_threads if args.baseline_threads is not None else p
            print(f"\n--- p={p} (Baseline: single IP with {threads} threads | Decomposition: {p} independent Gurobis) ---")

            if (name, threads) not in baseline_cache:
                print(f">>> Baseline: single large IP ({threads} threads)")
                base = run_baseline(name, tc, args.time_limit, threads)
                baseline_cache[(name, threads)] = base
                single_ip_rows.append({
                    "test": name,
                    "grid": tc["grid"],
                    "group": tc["group"],
                    "p": p,
                    "baseline_threads": threads,
                    "original_single_ip_solve_time_s": base["solve_time"],
                    "original_single_ip_build_time_s": base["build_time"],
                    "original_single_ip_total_wall_s": base["wall_time"],
                    "original_single_ip_energy": base["energy"],
                    "original_single_ip_vars": base["num_vars"],
                    "original_single_ip_constrs": base["num_constrs"],
                    "original_single_ip_qterms": base["num_qterms"],
                })
                save_csvs()
            else:
                base = baseline_cache[(name, threads)]
                print(f">>> Baseline: cached run with {threads} threads (solve time: {base['solve_time']:.4f}s)")

            print(f">>> Decomposition over p={p} processors ({p} independent Gurobis, 1 thread each)")
            phase = Phase(tc["phase"])
            best_ip, _, par_time, st = run_parallel_search(
                ions_count=tc["ions"], grid_size=tc["grid"], cell_size=tc["cell"],
                phase=phase, p=p, R_target=args.R_target, group=tc["group"],
                top=1, verbose=False, max_tasks=args.max_tasks,
                TimeLimit=args.time_limit, return_stats=True,
            )

            tasks = st["tasks"]
            total_cfg = st["total_configurations"]
            n_tasks = st["num_tasks"]
            solve_times = [t.get("solve_time", float("nan")) for t in tasks]
            build_times = [t.get("build_time", float("nan")) for t in tasks]
            n_feasible = sum(1 for t in tasks if t.get("obj") is not None)

            avg_solve = sum(solve_times) / len(solve_times) if solve_times else float("nan")
            avg_build = sum(build_times) / len(build_times) if build_times else float("nan")
            proc_loads = [ps["configurations"] for ps in st["processors"]]
            avg_load = total_cfg / p

            summary_rows.append({
                "test": name,
                "grid": tc["grid"],
                "group": tc["group"],
                "p": p,
                "baseline_threads": threads,
                # Original single large IP metrics
                "original_single_ip_solve_time_s": base["solve_time"],
                "original_single_ip_build_time_s": base["build_time"],
                "original_single_ip_total_wall_s": base["wall_time"],
                "original_single_ip_energy": base["energy"],
                "original_single_ip_vars": base["num_vars"],
                "original_single_ip_constrs": base["num_constrs"],
                "original_single_ip_qterms": base["num_qterms"],
                # Backward-compatibility alias keys
                "baseline_solve_s": base["solve_time"],
                "baseline_build_s": base["build_time"],
                "baseline_wall_s": base["wall_time"],
                "baseline_energy": base["energy"],
                "baseline_vars": base["num_vars"],
                "baseline_constrs": base["num_constrs"],
                "baseline_qterms": base["num_qterms"],
                "algorithm": st.get("algorithm", "Algorithm 1"),
                # Decomposition metrics
                "decomp_energy": best_ip if best_ip != float("inf") else None,
                "energy_match": (base["energy"] is not None and best_ip != float("inf")
                                 and abs(base["energy"] - best_ip) < 1e-4),
                "num_subIPs": n_tasks,
                "num_feasible_subIPs": n_feasible,
                "total_configurations": str(total_cfg),
                "total_configurations_sci": sci(total_cfg),
                "avg_subIP_solve_s": avg_solve,
                "max_subIP_solve_s": max(solve_times, default=float("nan")),
                "sum_subIP_solve_s": sum(solve_times),
                "avg_subIP_build_s": avg_build,
                "parallel_time_s": par_time,  # makespan (solve only)
                "parallel_wall_s": st["parallel_wall_time"],  # incl. building, spawn
                "decomposition_s": st["decomposition_time"],
                "speedup_solve": (base["solve_time"] / par_time) if par_time > 0 else float("nan"),
                "speedup_wall": (base["wall_time"] / st["parallel_wall_time"])
                                if st["parallel_wall_time"] > 0 else float("nan"),
                "load_imbalance_max_over_avg": ratio(max(proc_loads), avg_load) if avg_load else float("nan"),
                # sub-IP model size (same variables for all; constraints grow with prefix)
                "subIP_vars": tasks[0].get("num_vars") if tasks else None,
                "subIP_constrs_avg": (sum(t.get("num_constrs", 0) for t in tasks) / len(tasks))
                                     if tasks else None,
            })

            for ps in st["processors"]:
                proc_rows.append({
                    "test": name,
                    "p": p,
                    "processor": ps["processor"],
                    "num_IPs": ps["num_ips"],
                    "IP_ratio": ratio(ps["num_ips"], n_tasks),
                    "configurations": str(ps["configurations"]),
                    "configurations_sci": sci(ps["configurations"]),
                    "config_ratio": ratio(ps["configurations"], total_cfg),
                    "solve_time_s": ps["solve_time"],
                    "build_time_s": ps["build_time"],
                    "wall_time_s": ps["wall_time"],
                    "best_energy": ps["best_energy"] if ps["best_energy"] != float("inf") else None,
                    "is_bottleneck": ps["solve_time"] == par_time,
                })

            for t in tasks:
                task_rows.append({
                    "test": name,
                    "p": p,
                    "task_id": t["task_id"],
                    "processor": t["processor"],
                    "prefix": t["prefix"],
                    "configurations": str(t["configurations"]),
                    "configurations_sci": sci(t["configurations"]),
                    "config_ratio": ratio(t["configurations"], total_cfg),
                    "build_time_s": t.get("build_time"),
                    "solve_time_s": t.get("solve_time"),
                    "wall_time_s": t.get("wall_time"),
                    "status": t.get("status"),
                    "pruned_by_peer": t.get("pruned_by_peer", False),
                    "energy": t.get("obj"),
                    "num_vars": t.get("num_vars"),
                    "num_constrs": t.get("num_constrs"),
                    "num_qterms": t.get("num_qterms"),
                    "nodes": t.get("nodes"),
                })

            save_csvs()  # keep partial results if the run is interrupted

    summary = pd.DataFrame(summary_rows)
    procs = pd.DataFrame(proc_rows)
    tasks_df = pd.DataFrame(task_rows)
    save_csvs()

    # ---------- Console report ----------
    print("\n\n" + "=" * 110)
    print("  SINGLE IP vs DECOMPOSITION")
    print("=" * 110)
    view = summary[[
        "test", "p", "baseline_threads", "baseline_energy", "decomp_energy", "energy_match",
        "baseline_vars", "baseline_constrs", "baseline_qterms",
        "baseline_build_s", "baseline_solve_s",
        "num_subIPs", "num_feasible_subIPs", "avg_subIP_build_s",
        "avg_subIP_solve_s", "max_subIP_solve_s", "parallel_time_s",
        "parallel_wall_s", "speedup_solve",
    ]]
    print(tabulate(view, headers="keys", tablefmt="github", showindex=False,
                   floatfmt=".4f"))

    print("\n" + "=" * 110)
    print("  SPLIT OVER PROCESSORS")
    print("=" * 110)
    pv = procs.copy()
    pv["IPs"] = pv.apply(lambda r: f"{r.num_IPs} ({100 * r.IP_ratio:.1f}%)", axis=1)
    pv["configs"] = pv.apply(
        lambda r: f"{r.configurations_sci} ({100 * r.config_ratio:.2f}%)", axis=1)
    print(tabulate(pv[["test", "p", "processor", "IPs", "configs", "solve_time_s",
                       "build_time_s", "wall_time_s", "is_bottleneck"]],
                   headers="keys", tablefmt="github", showindex=False, floatfmt=".4f"))

    print(f"\nCSV files written to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
