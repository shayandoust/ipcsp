import ctypes
import math
import multiprocessing
import time
import gurobipy as gb

from ipcsp.integer_program import Allocate
from ipcsp.decomposition import (
    algo1_decomposition,
    algo2_decomposition,
    decompose,
    is_balanced,
    lpt_schedule,
)
from ipcsp.matrix_generator import Phase


class SharedIncumbentTracker:
    """
    Process-safe shared-memory coordinator that allows p independent Gurobi processes
    to broadcast and consume the best global objective value (cutoff) in real time.
    """

    def __init__(self, initial_cutoff: float = math.inf, tolerance: float = 1e-5):
        self._best_obj = multiprocessing.Value(ctypes.c_double, initial_cutoff)
        self._best_task_id = multiprocessing.Value(ctypes.c_int, -1)
        self._lock = multiprocessing.Lock()
        self.tolerance = tolerance

    def try_update(self, candidate_obj: float, task_id: int = -1) -> bool:
        """
        Atomically updates the global best objective if candidate_obj is strictly better.
        Returns True if a new global best was registered.
        """
        if candidate_obj >= self._best_obj.value - self.tolerance:
            return False

        with self._lock:
            if candidate_obj < self._best_obj.value - self.tolerance:
                self._best_obj.value = candidate_obj
                self._best_task_id.value = task_id
                return True
        return False

    def get_cutoff(self):
        """Returns (best_energy, task_id)."""
        return self._best_obj.value, self._best_task_id.value


def make_gurobi_sync_callback(
    tracker: SharedIncumbentTracker,
    task_id: int,
    prefix_str: str = "",
    check_interval: int = 50,
):
    """
    Creates a Gurobi callback that:
      1. Publishes newly found integer solutions (MIPSOL) to the shared tracker.
      2. Periodically checks if the subtree's lower bound (OBJBND) exceeds the global
         cutoff found by any peer, terminating the search early if it cannot beat it.
    """
    last_node = [-1]
    state = {"pruned_by_peer": False, "peer_cutoff": math.inf}

    def _callback(model, where):
        try:
            if where == gb.GRB.Callback.MIPSOL:
                obj = model.cbGet(gb.GRB.Callback.MIPSOL_OBJ)
                is_new = tracker.try_update(obj, task_id)
                if is_new:
                    print(
                        f"  [Task {task_id:02d} | {prefix_str}] -> New GLOBAL BEST energy: {obj:.4f}"
                    )

            elif where == gb.GRB.Callback.MIP:
                node_cnt = int(model.cbGet(gb.GRB.Callback.MIP_NODCNT))
                if node_cnt - last_node[0] < check_interval:
                    return
                last_node[0] = node_cnt

                best_local = model.cbGet(gb.GRB.Callback.MIP_OBJBST)
                if best_local < gb.GRB.INFINITY:
                    tracker.try_update(best_local, task_id)

                global_best, src_id = tracker.get_cutoff()
                if global_best < math.inf:
                    lower_bound = model.cbGet(gb.GRB.Callback.MIP_OBJBND)
                    if lower_bound >= global_best - tracker.tolerance:
                        state["pruned_by_peer"] = True
                        state["peer_cutoff"] = global_best
                        model.terminate()
        except Exception:
            pass

    return _callback, state


def process_worker(
    tasks,
    allocate_args,
    allocate_kwargs,
    worker_id,
    return_dict,
    tracker=None,
    share_cutoffs=True,
):
    """
    Solves every sub-IP (task) assigned to this processor sequentially.
    Stores (best_energy, best_solutions, worker_stats) in return_dict[worker_id].
    Shares cutoffs and incumbents in real time via tracker if enabled.
    """
    t_worker_start = time.perf_counter()

    best_energy = float("inf")
    best_res = None
    task_stats = []

    allocate = Allocate(*allocate_args)

    for task_id, task_size, prefix in tasks:
        prefix_str = "".join(f"{s} " for s in prefix).strip()
        cb = None
        cb_state = {"pruned_by_peer": False}
        cutoff_val = None

        if tracker is not None and share_cutoffs:
            curr_best, _ = tracker.get_cutoff()
            if curr_best < math.inf:
                cutoff_val = curr_best + 1e-5
            cb, cb_state = make_gurobi_sync_callback(
                tracker, task_id, prefix_str=prefix_str
            )

        t0 = time.perf_counter()
        res = allocate.optimize_cube_symmetry_ase(
            group=allocate_kwargs.get("group", "1"),
            PoolSolutions=allocate_kwargs.get("top", 1),
            TimeLimit=allocate_kwargs.get("TimeLimit", 0),
            verbose=allocate_kwargs.get("verbose", False),
            prefix=prefix,
            threads=1,
            write_lp=False,
            Cutoff=cutoff_val,
            callback=cb,
        )
        wall = time.perf_counter() - t0

        # last_stats is filled even when the sub-IP is infeasible (res is None)
        st = dict(allocate.last_stats or {})
        st.update(
            {
                "task_id": task_id,
                "processor": worker_id,
                "configurations": task_size,
                "prefix": prefix_str,
                "wall_time": wall,
                "pruned_by_peer": cb_state.get("pruned_by_peer", False),
            }
        )
        task_stats.append(st)

        if res is not None:
            solutions, runtime, objVal = res
            if tracker is not None and share_cutoffs:
                tracker.try_update(objVal, task_id)
            if objVal < best_energy:
                best_energy = objVal
                best_res = solutions

    worker_stats = {
        "tasks": task_stats,
        "wall_time": time.perf_counter() - t_worker_start,
    }
    return_dict[worker_id] = (best_energy, best_res, worker_stats)



def run_parallel_search(
    ions_count,
    grid_size,
    cell_size,
    phase,
    p=4,
    R_target=1.5,
    group="1",
    top=1,
    verbose=False,
    max_tasks=None,
    TimeLimit=0,
    return_stats=False,
    share_cutoffs=True,
):
    """
    Decomposes the configuration space using Algorithm 2 for balanced Parikh vectors
    (in the form of (n, ..., n)) and Algorithm 1 for all other cases, schedules the sub-IPs
    over p processors with LPT and solves them in parallel.
    Shares incumbent energies and cutoffs across processes in real time if share_cutoffs=True.

    Returns (best_energy, best_solutions, parallel_time), where parallel_time is the
    makespan: the summed Gurobi solve time of the most loaded processor.
    With return_stats=True a fourth element with detailed per-task/per-processor
    statistics is returned.
    """
    N = grid_size**3
    ion_types = list(ions_count.keys())
    total_ions = sum(ions_count.values())

    symbols = list(ion_types)
    counts = [ions_count[s] for s in symbols]

    if total_ions < N:
        symbols.append("V")
        counts.append(N - total_ions)

    balanced = is_balanced(counts)
    algo_name = "Algorithm 2" if balanced else "Algorithm 1"
    print(
        f"Decomposing search space using {algo_name}. Alphabet: {symbols}, counts: {counts}"
    )

    t_decomp_start = time.perf_counter()

    # In CSP formulation, the first orbit (containing site 0) must contain an ion (cannot be vacancy).
    if max_tasks is None:
        max_tasks = max(p, len(ion_types) * 2)

    all_tasks = []
    if balanced and total_ions == N:
        # Balanced Parikh vector with full occupancy (no vacancies):
        # Algorithm 2 can decompose the entire search space directly.
        all_tasks = decompose(
            symbols,
            counts,
            p,
            R_target=R_target,
            max_tasks=max_tasks,
        )
    else:
        # Branch on site 0 = ion for each ion type (cannot be vacancy in CSP formulation).
        for ion in ion_types:
            if ions_count[ion] > 0:
                b_counts = list(counts)
                b_counts[symbols.index(ion)] -= 1
                branch_tasks = decompose(
                    symbols,
                    b_counts,
                    p,
                    R_target=R_target,
                    max_tasks=max_tasks // len(ion_types),
                )
                for sz, pref in branch_tasks:
                    all_tasks.append((sz, [ion] + pref))

    if not all_tasks:
        all_tasks = [(1, [])]

    print(f"Generated {len(all_tasks)} independent search tasks.")

    processors, loads = lpt_schedule(all_tasks, p)
    decomposition_time = time.perf_counter() - t_decomp_start
    print(f"Scheduled tasks across {p} processors using LPT. Machine loads: {loads}")

    # Give every task a global id so it can be traced in the reports
    task_counter = 0
    processors_with_ids = []
    for proc_tasks in processors:
        tagged = []
        for size, prefix in proc_tasks:
            tagged.append((task_counter, size, prefix))
            task_counter += 1
        processors_with_ids.append(tagged)

    allocate_args = (ions_count, grid_size, cell_size, phase)
    allocate_kwargs = {
        "group": str(group),
        "top": top,
        "verbose": verbose,
        "TimeLimit": TimeLimit,
    }

    tracker = SharedIncumbentTracker() if share_cutoffs else None

    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    t_parallel_start = time.perf_counter()
    processes = []
    for i in range(p):
        proc = multiprocessing.Process(
            target=process_worker,
            args=(
                processors_with_ids[i],
                allocate_args,
                allocate_kwargs,
                i,
                return_dict,
                tracker,
                share_cutoffs,
            ),
        )
        processes.append(proc)
        proc.start()

    for proc in processes:
        proc.join()
    parallel_wall_time = time.perf_counter() - t_parallel_start

    overall_best_energy = float("inf")
    overall_best_res = []
    processor_stats = []
    task_stats = []

    for i in range(p):
        if i in return_dict:
            best_energy, best_res, wstats = return_dict[i]
            solve_sum = sum(t.get("solve_time", 0.0) for t in wstats["tasks"])
            build_sum = sum(t.get("build_time", 0.0) for t in wstats["tasks"])
            print(
                f"Processor {i} finished. Best IP energy: {best_energy:.4f}, "
                f"Solve time: {solve_sum:.2f}s, Build time: {build_sum:.2f}s, "
                f"Wall time: {wstats['wall_time']:.2f}s"
            )
            processor_stats.append(
                {
                    "processor": i,
                    "num_ips": len(wstats["tasks"]),
                    "configurations": loads[i],
                    "solve_time": solve_sum,
                    "build_time": build_sum,
                    "wall_time": wstats["wall_time"],
                    "best_energy": best_energy,
                }
            )
            task_stats.extend(wstats["tasks"])
            if best_energy < overall_best_energy:
                overall_best_energy = best_energy
                overall_best_res = best_res
        else:
            # Worker crashed / produced nothing
            processor_stats.append(
                {
                    "processor": i,
                    "num_ips": len(processors_with_ids[i]),
                    "configurations": loads[i],
                    "solve_time": float("nan"),
                    "build_time": float("nan"),
                    "wall_time": float("nan"),
                    "best_energy": float("inf"),
                }
            )

    # Parallel time = makespan over processors (solve time only)
    parallel_time = max((ps["solve_time"] for ps in processor_stats), default=0.0)

    print(
        f"\nOverall minimum IP energy across all processors: {overall_best_energy:.4f}"
    )
    print(f"Parallel (makespan) solve time: {parallel_time:.4f}s")

    if not return_stats:
        return overall_best_energy, overall_best_res, parallel_time

    stats = {
        "p": p,
        "algorithm": algo_name,
        "share_cutoffs": share_cutoffs,
        "symbols": symbols,
        "counts": counts,
        "num_tasks": len(all_tasks),
        "total_configurations": sum(loads),
        "decomposition_time": decomposition_time,
        "parallel_wall_time": parallel_wall_time,
        "parallel_time": parallel_time,
        "processors": processor_stats,
        "tasks": sorted(task_stats, key=lambda t: t["task_id"]),
    }
    return overall_best_energy, overall_best_res, parallel_time, stats
