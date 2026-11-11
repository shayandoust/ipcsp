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
import sys
import time

if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(100000)

import pandas as pd
from tabulate import tabulate

from ipcsp.integer_program import Allocate
from ipcsp.matrix_generator import Phase
from ipcsp.parallel_search import run_parallel_search

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "results", "decomposition_benchmark"
)

# Benchmark structures from Table 1 of the paper (all 16 benchmark instances)
TEST_CASES = {
    # 1. SrTiO3, Z = 1: 5 ions, cell 3.9 A, grid 4, group 1 (64 orbits)
    "SrTiO3_1": dict(
        phase="SrTiO", ions={"O": 3, "Sr": 1, "Ti": 1}, grid=4, cell=3.9, group="1"
    ),
    # 2. SrTiO3, Z = 8: 40 ions, cell 7.8 A, grid 8, group 195 (56 orbits)
    "SrTiO3_2": dict(
        phase="SrTiO", ions={"O": 24, "Sr": 8, "Ti": 8}, grid=8, cell=7.8, group="195"
    ),
    # 3. SrTiO3, Z = 27: 135 ions, cell 11.7 A, grid 6, group 221 (20 orbits)
    "SrTiO3_3": dict(
        phase="SrTiO", ions={"O": 81, "Sr": 27, "Ti": 27}, grid=6, cell=11.7, group="221"
    ),
    # 4. SrTiO3, Z = 27: 135 ions, cell 11.7 A, grid 6, group 200 (24 orbits)
    "SrTiO3_4": dict(
        phase="SrTiO", ions={"O": 81, "Sr": 27, "Ti": 27}, grid=6, cell=11.7, group="200"
    ),
    # 5. SrTiO3, Z = 27: 135 ions, cell 11.7 A, grid 6, group 195 (28 orbits)
    "SrTiO3_5": dict(
        phase="SrTiO", ions={"O": 81, "Sr": 27, "Ti": 27}, grid=6, cell=11.7, group="195"
    ),
    # 6. Y2O3: 80 ions, cell 10.7 A, grid 8, group 206 (17 orbits)
    "Y2O3_1": dict(
        phase="YSrTiO", ions={"O": 48, "Y": 32}, grid=8, cell=10.7, group="206"
    ),
    # 7. Y2O3: 80 ions, cell 10.7 A, grid 8, group 199 (40 orbits)
    "Y2O3_2": dict(
        phase="YSrTiO", ions={"O": 48, "Y": 32}, grid=8, cell=10.7, group="199"
    ),
    # 8. Y2O3: 80 ions, cell 10.7 A, grid 16, group 206 (124 orbits)
    "Y2O3_3": dict(
        phase="YSrTiO", ions={"O": 48, "Y": 32}, grid=16, cell=10.7, group="206"
    ),
    # 9. Y2Ti2O7: 88 ions, cell 10.2 A, grid 8, group 227 (11 orbits)
    "Y2Ti2O7_1": dict(
        phase="YSrTiO", ions={"O": 56, "Y": 16, "Ti": 16}, grid=8, cell=10.2, group="227"
    ),
    # 10. Y2Ti2O7: 88 ions, cell 10.2 A, grid 16, group 227 (51 orbits)
    "Y2Ti2O7_2": dict(
        phase="YSrTiO", ions={"O": 56, "Y": 16, "Ti": 16}, grid=16, cell=10.2, group="227"
    ),
    # 11. MgAl2O4: 56 ions, cell 8.2 A, grid 8, group 227 (11 orbits)
    "MgAl2O4_1": dict(
        phase="LiMgAlPO", ions={"O": 32, "Mg": 8, "Al": 16}, grid=8, cell=8.2, group="227"
    ),
    # 12. MgAl2O4: 56 ions, cell 8.2 A, grid 16, group 227 (51 orbits)
    "MgAl2O4_2": dict(
        phase="LiMgAlPO", ions={"O": 32, "Mg": 8, "Al": 16}, grid=16, cell=8.2, group="227"
    ),
    # 13. MgAl2O4: 56 ions, cell 8.2 A, grid 8, group 196 (22 orbits)
    "MgAl2O4_3": dict(
        phase="LiMgAlPO", ions={"O": 32, "Mg": 8, "Al": 16}, grid=8, cell=8.2, group="196"
    ),
    # 14. MgAl2O4: 56 ions, cell 8.2 A, grid 8, group 195 (56 orbits)
    "MgAl2O4_4": dict(
        phase="LiMgAlPO", ions={"O": 32, "Mg": 8, "Al": 16}, grid=8, cell=8.2, group="195"
    ),
    # 15. Ca3Al2Si3O12: 160 ions, cell 11.9 A, grid 8, group 206 (17 orbits)
    "Ca3Al2Si3O12_2": dict(
        phase="CaAlSiO", ions={"Ca": 24, "Al": 16, "Si": 24, "O": 96}, grid=8, cell=11.9, group="206"
    ),
    # 16. Ca3Al2Si3O12: 160 ions, cell 11.9 A, grid 16, group 230 (62 orbits)
    "Ca3Al2Si3O12_1": dict(
        phase="CaAlSiO", ions={"Ca": 24, "Al": 16, "Si": 24, "O": 96}, grid=16, cell=11.9, group="230"
    ),
    # =========================================================================
    # EXTREME CASES BENCHMARK (P1 Space Group / Large Supercells)
    # =========================================================================
    # 1. Ca3Al2Si3O12, Garnet (Cubic), 160 ions (Z=8), 11.9 A, g=8 (n=512 in P1), Omega ~ 10^210
    "Ca3Al2Si3O12_P1": dict(
        phase="CaAlSiO", ions={"Ca": 24, "Al": 16, "Si": 24, "O": 96}, grid=8, cell=11.9, group="1"
    ),
    # 2. SrTiO3 (Supercell), Perovskite (3x3x3), 135 ions (Z=27), 11.7 A, g=8 (n=512 in P1), Omega ~ 10^180
    "SrTiO3_3x3x3_P1": dict(
        phase="SrTiO", ions={"O": 81, "Sr": 27, "Ti": 27}, grid=8, cell=11.7, group="1"
    ),
    # 3. Y2Ti2O7, Pyrochlore (Cubic), 88 ions (Z=8), 10.2 A, g=8 (n=512 in P1), Omega ~ 10^133
    "Y2Ti2O7_P1": dict(
        phase="YSrTiO", ions={"O": 56, "Y": 16, "Ti": 16}, grid=8, cell=10.2, group="1"
    ),
    # 4. Y2O3, Bixbyite (Cubic), 80 ions (Z=16), 10.7 A, g=8 (n=512 in P1), Omega ~ 10^97
    "Y2O3_P1": dict(
        phase="YSrTiO", ions={"O": 48, "Y": 32}, grid=8, cell=10.7, group="1"
    ),
    # 5. MgAl2O4, Spinel (Cubic), 56 ions (Z=8), 8.2 A, g=8 (n=512 in P1), Omega ~ 10^97
    "MgAl2O4_P1": dict(
        phase="LiMgAlPO", ions={"O": 32, "Mg": 8, "Al": 16}, grid=8, cell=8.2, group="1"
    ),
    # 6. SrTiO3 (Supercell), Perovskite (2x2x2), 40 ions (Z=8), 7.8 A, g=8 (n=512 in P1), Omega ~ 10^74
    "SrTiO3_2x2x2_P1": dict(
        phase="SrTiO", ions={"O": 24, "Sr": 8, "Ti": 8}, grid=8, cell=7.8, group="1"
    ),
    # 7. Au0.5Cu0.5, FCC L10 Alloy Supercell (8x8x8 FCC cells), 2,048 atoms, 30.4 A, Omega ~ 2^2048
    "AuCu_8x8x8": dict(
        phase="AuCu", ions={"Au": 1024, "Cu": 1024}, grid=8, cell=30.4, group="1"
    ),
    # 8. Au0.5Cu0.5, FCC L10 Alloy Supercell (16x16x16 FCC cells), 16,384 atoms, 60.8 A, Omega ~ 2^16384
    "AuCu_16x16x16": dict(
        phase="AuCu", ions={"Au": 8192, "Cu": 8192}, grid=16, cell=60.8, group="1"
    ),
}

EXTREME_TESTS = [
    "Ca3Al2Si3O12_P1",
    "SrTiO3_3x3x3_P1",
    "Y2Ti2O7_P1",
    "Y2O3_P1",
    "MgAl2O4_P1",
    "SrTiO3_2x2x2_P1",
    "AuCu_8x8x8",
    "AuCu_16x16x16",
]

DEFAULT_TESTS = [k for k in TEST_CASES if k.startswith("SrTiO3") and not k.endswith("_P1")]


def sci(n):
    """Format a (possibly huge) integer in scientific notation."""
    if n == 0:
        return "0"
    log10_n = math.log10(n)
    e = int(math.floor(log10_n))
    mant = (n / 10**e) if e < 300 else (10.0 ** (log10_n - e))
    return f"{mant:.3f}e{e}"


def ratio(a, b):
    """a/b for arbitrarily large ints."""
    return a / b if b else float("nan")


def run_baseline(name, tc, time_limit, threads, progress_interval=30.0, progress_dir=None, p_val=None):
    phase = Phase(tc["phase"])
    alloc = Allocate(tc["ions"], tc["grid"], tc["cell"], phase)
    t0 = time.perf_counter()
    res = alloc.optimize_cube_symmetry_ase(
        group=tc["group"], PoolSolutions=1, TimeLimit=time_limit,
        verbose=False, threads=threads, write_lp=False,
        track_progress=True,
        progress_interval=progress_interval,
        progress_dir=progress_dir,
        progress_meta={"test": name, "p": p_val or threads, "mode": "single_ip_baseline", "threads": threads},
    )
    wall = time.perf_counter() - t0
    st = alloc.last_stats
    if getattr(alloc, "model", None) is not None:
        try:
            alloc.model.dispose()
            import gurobipy as _gb
            _gb.disposeDefaultEnv()
        except Exception:
            pass
    return {
        "energy": res[2] if res else None,
        "best_bound": st.get("best_bound"),
        "mip_gap_pct": st.get("mip_gap_pct"),
        "unexplored_nodes": st.get("unexplored_nodes", 0),
        "peak_unexplored_nodes": st.get("peak_unexplored_nodes", 0),
        "build_time": st["build_time"],
        "solve_time": st["solve_time"],
        "wall_time": wall,
        "num_vars": st["num_vars"],
        "num_constrs": st["num_constrs"],
        "num_qterms": st["num_qterms"],
        "status": st["status"],
        "nodes": st.get("nodes", 0.0),
        "work": st.get("work", 0.0),
        "iter_count": st.get("iter_count", 0.0),
        "time_to_opt_s": st.get("time_to_opt_s", st["solve_time"]),
        "wall_to_opt_s": st.get("wall_to_opt_s", wall),
        "nodes_to_opt": st.get("nodes_to_opt", st.get("nodes", 0.0)),
        "work_to_opt": st.get("work_to_opt", st.get("work", 0.0)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tests", nargs="+", default=DEFAULT_TESTS,
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
    ap.add_argument("--subip-procs", type=int, default=None,
                    help="number of Gurobi worker processes for decomposition (default: p)")
    ap.add_argument("--subip-threads", type=int, default=None,
                    help="Gurobi threads per sub-IP worker process (default: 1)")
    ap.add_argument("--skip-baseline", "--decomp-only", action="store_true",
                    help="Skip baseline single IP and run decomposition search directly")
    ap.add_argument("--baseline-only", "--skip-decomp", action="store_true",
                    help="Run single large IP baseline only and skip decomposition search")
    ap.add_argument("--progress-interval", type=float, default=30.0,
                    help="Interval in seconds for periodic MIPGap & Unexp progress logs (default: 30s)")
    ap.add_argument("--output-dir", type=str, default=RESULTS_DIR,
                    help="directory to save benchmark CSV results")
    args = ap.parse_args()

    results_dir = args.output_dir
    os.makedirs(results_dir, exist_ok=True)
    summary_rows, proc_rows, task_rows, single_ip_rows = [], [], [], []
    baseline_cache = {}

    def save_csvs():
        df_sum = pd.DataFrame(summary_rows)
        df_procs = pd.DataFrame(proc_rows)
        df_tasks = pd.DataFrame(task_rows)
        df_single = pd.DataFrame(single_ip_rows)

        df_sum.to_csv(os.path.join(results_dir, "summary.csv"), index=False)
        df_procs.to_csv(os.path.join(results_dir, "processors.csv"), index=False)
        df_tasks.to_csv(os.path.join(results_dir, "tasks.csv"), index=False)
        df_single.to_csv(os.path.join(results_dir, "original_single_ip.csv"), index=False)

        # Solve-only views (pure solver time with build phase excluded)
        if not df_sum.empty:
            solve_cols = [
                "test", "grid", "group", "p", "baseline_threads",
                "decomp_procs", "decomp_threads",
                "original_single_ip_solve_time_s",
                "original_single_ip_solve_time_to_opt_s",
                "original_single_ip_vars", "original_single_ip_constrs", "original_single_ip_energy",
                "total_configurations_sci", "num_subIPs",
                "avg_subIP_solve_s", "max_subIP_solve_s",
                "parallel_time_s", "sum_subIP_solve_s",
                "speedup_vs_single_gurobi_p_threads",
                "speedup_solve",
                "decomp_solve_to_opt_s",
                "speedup_find_opt_solve",
                "first_opt_processor",
                "first_opt_task_id",
                "work_reduction_ratio_solve",
                "decomp_energy", "energy_match",
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
            solve_only_df.to_csv(os.path.join(results_dir, "solve_only_comparison.csv"), index=False)

        if not df_procs.empty:
            proc_cols = [
                "test", "p", "decomp_procs", "decomp_threads", "processor", "num_IPs", "IP_ratio",
                "configurations_sci", "config_ratio", "solve_time_s",
                "nodes", "work", "iter_count", "found_global_opt",
                "solve_time_to_opt_s", "is_bottleneck"
            ]
            df_procs[[c for c in proc_cols if c in df_procs.columns]].to_csv(
                os.path.join(results_dir, "solve_only_processors.csv"), index=False
            )

        if not df_tasks.empty:
            task_cols = [
                "test", "p", "decomp_procs", "decomp_threads", "task_id", "processor", "prefix",
                "configurations_sci", "config_ratio", "solve_time_s",
                "status", "energy", "found_global_opt", "solve_time_to_opt_s",
                "nodes", "work", "iter_count", "num_vars", "num_constrs"
            ]
            df_tasks[[c for c in task_cols if c in df_tasks.columns]].to_csv(
                os.path.join(results_dir, "solve_only_tasks.csv"), index=False
            )

    for name in args.tests:
        tc = TEST_CASES[name]
        print(f"\n{'=' * 30} {name} {'=' * 30}")

        for p in args.p:
            threads = args.baseline_threads if args.baseline_threads is not None else p
            print(f"\n--- p={p} (Baseline: single IP with {threads} threads | Decomposition: {p} independent Gurobis) ---")

            if args.skip_baseline:
                print(">>> Baseline: skipped (--skip-baseline requested)")
                base = {
                    "energy": None,
                    "build_time": None,
                    "solve_time": None,
                    "wall_time": None,
                    "num_vars": None,
                    "num_constrs": None,
                    "num_qterms": None,
                    "status": None,
                    "nodes": 0.0,
                    "work": 0.0,
                    "iter_count": 0.0,
                    "time_to_opt_s": None,
                    "wall_to_opt_s": None,
                    "nodes_to_opt": 0.0,
                    "work_to_opt": 0.0,
                }
                for cand_dir in [
                    results_dir,
                    os.path.join(os.path.dirname(results_dir), f"p{p}", name),
                    os.path.join(os.path.dirname(os.path.dirname(results_dir)), f"p{p}", name),
                ]:
                    cand_single = os.path.join(cand_dir, "original_single_ip.csv")
                    if os.path.exists(cand_single):
                        try:
                            df_ex = pd.read_csv(cand_single)
                            m_row = df_ex[(df_ex["test"] == name) & (df_ex["p"] == p)]
                            if not m_row.empty:
                                r0 = m_row.iloc[0].to_dict()
                                base = {
                                    "energy": r0.get("original_single_ip_energy"),
                                    "build_time": r0.get("original_single_ip_build_time_s", 0.0),
                                    "solve_time": r0.get("original_single_ip_solve_time_s", 0.0),
                                    "wall_time": r0.get("original_single_ip_total_wall_s", 0.0),
                                    "num_vars": r0.get("original_single_ip_vars", 0),
                                    "num_constrs": r0.get("original_single_ip_constrs", 0),
                                    "num_qterms": r0.get("original_single_ip_qterms", 0),
                                    "status": r0.get("status"),
                                    "nodes": r0.get("original_single_ip_nodes", 0.0),
                                    "work": r0.get("original_single_ip_work", 0.0),
                                    "iter_count": r0.get("original_single_ip_iters", 0.0),
                                    "time_to_opt_s": r0.get("original_single_ip_solve_time_to_opt_s"),
                                    "wall_to_opt_s": r0.get("original_single_ip_wall_time_to_opt_s"),
                                    "nodes_to_opt": r0.get("original_single_ip_nodes_to_opt"),
                                    "work_to_opt": r0.get("original_single_ip_work_to_opt"),
                                }
                                print(f">>> Baseline: loaded existing baseline results from {cand_single}")
                                break
                        except Exception:
                            pass
            elif (name, threads) not in baseline_cache:
                print(f">>> Baseline: single large IP ({threads} threads)")
                base = run_baseline(
                    name,
                    tc,
                    args.time_limit,
                    threads,
                    progress_interval=args.progress_interval,
                    progress_dir=results_dir,
                    p_val=p,
                )
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
                    "original_single_ip_best_bound": base.get("best_bound"),
                    "original_single_ip_mip_gap_pct": base.get("mip_gap_pct"),
                    "original_single_ip_unexplored_nodes": base.get("unexplored_nodes"),
                    "original_single_ip_peak_unexplored_nodes": base.get("peak_unexplored_nodes"),
                    "original_single_ip_vars": base["num_vars"],
                    "original_single_ip_constrs": base["num_constrs"],
                    "original_single_ip_qterms": base["num_qterms"],
                    "original_single_ip_solve_time_to_opt_s": base.get("time_to_opt_s"),
                    "original_single_ip_wall_time_to_opt_s": base.get("wall_to_opt_s"),
                    "original_single_ip_nodes_to_opt": base.get("nodes_to_opt"),
                    "original_single_ip_work_to_opt": base.get("work_to_opt"),
                    "original_single_ip_nodes": base.get("nodes"),
                    "original_single_ip_work": base.get("work"),
                    "original_single_ip_iters": base.get("iter_count"),
                })
                save_csvs()
            else:
                base = baseline_cache[(name, threads)]
                print(f">>> Baseline: cached run with {threads} threads (solve time: {base['solve_time']:.4f}s)")

            if args.baseline_only:
                print(">>> Decomposition: skipped (--baseline-only requested)")
                continue

            subip_procs = args.subip_procs if args.subip_procs is not None else p
            subip_threads = args.subip_threads if args.subip_threads is not None else 1
            print(f">>> Decomposition over p={p} (workers: {subip_procs}, Gurobi threads per worker: {subip_threads})")
            phase = Phase(tc["phase"])
            best_ip, _, par_time, st = run_parallel_search(
                ions_count=tc["ions"], grid_size=tc["grid"], cell_size=tc["cell"],
                phase=phase, p=p, R_target=args.R_target, group=tc["group"],
                top=1, verbose=False, max_tasks=args.max_tasks,
                TimeLimit=args.time_limit, return_stats=True,
                num_procs=subip_procs, threads_per_worker=subip_threads,
                progress_interval=args.progress_interval,
                progress_dir=results_dir,
                progress_meta={"test": name, "p": p, "mode": "decomposition"},
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
            load_imbalance = (
                ratio(max(proc_loads) * subip_procs, total_cfg)
                if total_cfg and proc_loads
                else float("nan")
            )

            summary_rows.append({
                "test": name,
                "grid": tc["grid"],
                "group": tc["group"],
                "p": p,
                "baseline_threads": threads,
                "decomp_procs": subip_procs,
                "decomp_threads": subip_threads,
                # Original single large IP metrics
                "original_single_ip_solve_time_s": base.get("solve_time"),
                "original_single_ip_build_time_s": base.get("build_time"),
                "original_single_ip_total_wall_s": base.get("wall_time"),
                "original_single_ip_energy": base.get("energy"),
                "original_single_ip_vars": base.get("num_vars"),
                "original_single_ip_constrs": base.get("num_constrs"),
                "original_single_ip_qterms": base.get("num_qterms"),
                "original_single_ip_solve_time_to_opt_s": base.get("time_to_opt_s"),
                "original_single_ip_wall_time_to_opt_s": base.get("wall_to_opt_s"),
                "original_single_ip_nodes_to_opt": base.get("nodes_to_opt"),
                "original_single_ip_work_to_opt": base.get("work_to_opt"),
                # Backward-compatibility alias keys
                "baseline_solve_s": base.get("solve_time"),
                "baseline_build_s": base.get("build_time"),
                "baseline_wall_s": base.get("wall_time"),
                "baseline_energy": base.get("energy"),
                "baseline_vars": base.get("num_vars"),
                "baseline_constrs": base.get("num_constrs"),
                "baseline_qterms": base.get("num_qterms"),
                "baseline_solve_to_opt_s": base.get("time_to_opt_s"),
                "baseline_wall_to_opt_s": base.get("wall_to_opt_s"),
                "baseline_nodes_to_opt": base.get("nodes_to_opt"),
                "baseline_work_to_opt": base.get("work_to_opt"),
                "baseline_nodes": base.get("nodes"),
                "baseline_work": base.get("work"),
                "baseline_iters": base.get("iter_count"),
                "algorithm": st.get("algorithm", "Algorithm 1"),
                # Decomposition metrics
                "decomp_energy": best_ip if best_ip != float("inf") else None,
                "energy_match": (base.get("energy") is not None and best_ip != float("inf")
                                 and abs(base["energy"] - best_ip) < 1e-4) if base.get("energy") is not None else None,
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
                "speedup_vs_single_gurobi_p_threads": (base["solve_time"] / par_time)
                                if (base.get("solve_time") and par_time > 0) else float("nan"),
                "speedup_vs_single_gurobi_p_threads_wall": (base["wall_time"] / st["parallel_wall_time"])
                                if (base.get("wall_time") and st["parallel_wall_time"] > 0) else float("nan"),
                "speedup_solve": (base["solve_time"] / par_time)
                                if (base.get("solve_time") and par_time > 0) else float("nan"),
                "speedup_wall": (base["wall_time"] / st["parallel_wall_time"])
                                if (base.get("wall_time") and st["parallel_wall_time"] > 0) else float("nan"),
                # Time / effort to FIRST HIT OPT (Discovery phase)
                "decomp_solve_to_opt_s": st.get("decomp_solve_time_to_opt_s"),
                "decomp_wall_to_opt_s": st.get("decomp_wall_time_to_opt_s"),
                "decomp_nodes_to_opt": st.get("decomp_nodes_to_opt"),
                "decomp_work_to_opt": st.get("decomp_work_to_opt"),
                "first_opt_task_id": st.get("first_opt_task_id"),
                "first_opt_processor": st.get("first_opt_processor"),
                "speedup_find_opt_solve": ratio(base.get("time_to_opt_s", 0.0), st.get("decomp_solve_time_to_opt_s")),
                "speedup_find_opt_wall": ratio(base.get("wall_to_opt_s", 0.0), st.get("decomp_wall_time_to_opt_s")),
                # Total effort / work reduction across all sub-IPs
                "sum_subIP_nodes": st.get("sum_subIP_nodes"),
                "sum_subIP_work": st.get("sum_subIP_work"),
                "sum_subIP_iters": st.get("sum_subIP_iters"),
                "work_reduction_ratio_solve": ratio(base.get("solve_time", 0.0), sum(solve_times)),
                "work_reduction_ratio_work": ratio(base.get("work", 0.0), st.get("sum_subIP_work")),
                "load_imbalance_max_over_avg": load_imbalance,
                # sub-IP model size
                "subIP_vars": tasks[0].get("num_vars") if tasks else None,
                "subIP_constrs_avg": (sum(t.get("num_constrs", 0) for t in tasks) / len(tasks))
                                     if tasks else None,
            })

            for ps in st["processors"]:
                proc_rows.append({
                    "test": name,
                    "p": p,
                    "decomp_procs": subip_procs,
                    "decomp_threads": subip_threads,
                    "processor": ps["processor"],
                    "num_IPs": ps["num_ips"],
                    "IP_ratio": ratio(ps["num_ips"], n_tasks),
                    "configurations": str(ps["configurations"]),
                    "configurations_sci": sci(ps["configurations"]),
                    "config_ratio": ratio(ps["configurations"], total_cfg),
                    "solve_time_s": ps["solve_time"],
                    "build_time_s": ps["build_time"],
                    "wall_time_s": ps["wall_time"],
                    "nodes": ps.get("nodes"),
                    "work": ps.get("work"),
                    "iter_count": ps.get("iter_count"),
                    "best_energy": ps["best_energy"] if ps["best_energy"] != float("inf") else None,
                    "found_global_opt": ps.get("found_global_opt", False),
                    "solve_time_to_opt_s": ps.get("solve_time_to_best_s"),
                    "nodes_to_opt": ps.get("nodes_to_best"),
                    "work_to_opt": ps.get("work_to_best"),
                    "is_bottleneck": ps["solve_time"] == par_time,
                })

            for t in tasks:
                task_rows.append({
                    "test": name,
                    "p": p,
                    "decomp_procs": subip_procs,
                    "decomp_threads": subip_threads,
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
                    "found_global_opt": t.get("found_global_opt", False),
                    "solve_time_to_opt_s": t.get("time_to_opt_s"),
                    "nodes_to_opt": t.get("nodes_to_opt"),
                    "work_to_opt": t.get("work_to_opt"),
                    "num_vars": t.get("num_vars"),
                    "num_constrs": t.get("num_constrs"),
                    "num_qterms": t.get("num_qterms"),
                    "nodes": t.get("nodes"),
                    "work": t.get("work"),
                    "iter_count": t.get("iter_count"),
                })

            save_csvs()  # keep partial results if the run is interrupted

    summary = pd.DataFrame(summary_rows)
    procs = pd.DataFrame(proc_rows)
    tasks_df = pd.DataFrame(task_rows)
    single_df = pd.DataFrame(single_ip_rows)
    save_csvs()

    # ---------- Console report ----------
    if not single_df.empty:
        print("\n\n" + "=" * 115)
        print("  SINGLE LARGE IP BASELINE RESULTS")
        print("=" * 115)
        b_cols = [
            "test", "grid", "group", "p", "baseline_threads",
            "original_single_ip_energy", "original_single_ip_solve_time_s",
            "original_single_ip_total_wall_s", "original_single_ip_solve_time_to_opt_s",
            "original_single_ip_nodes", "original_single_ip_work",
            "original_single_ip_vars", "original_single_ip_constrs", "original_single_ip_qterms"
        ]
        print(tabulate(single_df[[c for c in b_cols if c in single_df.columns]], headers="keys", tablefmt="github", showindex=False, floatfmt=".4f"))

    if not summary.empty:
        print("\n\n" + "=" * 115)
        print("  1. SINGLE IP vs DECOMPOSITION (Proof of Optimality / Makespan)")
        print("=" * 115)
        view1 = summary[[
            "test", "p", "baseline_threads", "decomp_procs", "decomp_threads",
            "baseline_energy", "decomp_energy", "energy_match",
            "baseline_solve_s", "avg_subIP_solve_s", "max_subIP_solve_s",
            "parallel_time_s", "speedup_vs_single_gurobi_p_threads", "work_reduction_ratio_solve",
        ]]
        print(tabulate(view1, headers="keys", tablefmt="github", showindex=False,
                       floatfmt=".4f"))

        print("\n\n" + "=" * 115)
        print("  2. TIME & EFFORT TO FIRST HIT OPT (Discovery Phase: Comparison to MC-EMMA)")
        print("=" * 115)
        view2 = summary[[
            "test", "p", "baseline_solve_to_opt_s", "decomp_solve_to_opt_s",
            "speedup_find_opt_solve", "first_opt_processor", "first_opt_task_id",
            "baseline_nodes_to_opt", "decomp_nodes_to_opt",
            "baseline_work_to_opt", "decomp_work_to_opt",
        ]]
        print(tabulate(view2, headers="keys", tablefmt="github", showindex=False,
                       floatfmt=".4f"))

    if not procs.empty:
        print("\n\n" + "=" * 115)
        print("  3. SPLIT OVER PROCESSORS")
        print("=" * 115)
        pv = procs.copy()
        pv["IPs"] = pv.apply(lambda r: f"{r.num_IPs} ({100 * r.IP_ratio:.1f}%)", axis=1)
        pv["configs"] = pv.apply(
            lambda r: f"{r.configurations_sci} ({100 * r.config_ratio:.2f}%)", axis=1)
        print(tabulate(pv[["test", "p", "processor", "IPs", "configs", "solve_time_s",
                           "build_time_s", "wall_time_s", "found_global_opt", "solve_time_to_opt_s", "is_bottleneck"]],
                       headers="keys", tablefmt="github", showindex=False, floatfmt=".4f"))

    print(f"\nCSV files written to: {results_dir}")


if __name__ == "__main__":
    main()
