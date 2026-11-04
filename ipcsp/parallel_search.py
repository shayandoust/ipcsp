import ctypes
import json
import math
import multiprocessing
import time
import gurobipy as gb

from ipcsp import root_dir
from ipcsp.integer_program import Allocate
from ipcsp.decomposition import (
    algo1_decomposition,
    algo2_decomposition,
    decompose,
    is_balanced,
    lpt_schedule,
    build_orbit_dp,
    decompose_orbits_balanced,
)
from ipcsp.matrix_generator import Phase, get_Ewald, get_Buck, get_garnet


class SharedIncumbentTracker:
    """
    Process-safe shared-memory coordinator that allows p independent Gurobi processes
    to broadcast and consume the best global objective value (cutoff) in real time,
    while capturing the exact time and solver effort to first discover optimal solutions.
    """

    def __init__(self, initial_cutoff: float = math.inf, tolerance: float = 1e-5):
        self._best_obj = multiprocessing.Value(ctypes.c_double, initial_cutoff)
        self._best_task_id = multiprocessing.Value(ctypes.c_int, -1)
        self._best_proc_id = multiprocessing.Value(ctypes.c_int, -1)
        self._wall_time_to_opt = multiprocessing.Value(ctypes.c_double, math.inf)
        self._proc_solve_time_to_opt = multiprocessing.Value(ctypes.c_double, math.inf)
        self._task_solve_time_to_opt = multiprocessing.Value(ctypes.c_double, math.inf)
        self._nodes_to_opt = multiprocessing.Value(ctypes.c_double, math.inf)
        self._work_to_opt = multiprocessing.Value(ctypes.c_double, math.inf)
        self._start_time = multiprocessing.Value(ctypes.c_double, 0.0)
        self._lock = multiprocessing.Lock()
        self.tolerance = tolerance

    def set_start_time(self, t0: float):
        with self._lock:
            self._start_time.value = t0

    def try_update(
        self,
        candidate_obj: float,
        task_id: int = -1,
        processor_id: int = -1,
        task_solve_time: float = 0.0,
        task_nodes: float = 0.0,
        task_work: float = 0.0,
        proc_solve_time: float = 0.0,
        proc_nodes: float = 0.0,
        proc_work: float = 0.0,
    ) -> bool:
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
                self._best_proc_id.value = processor_id
                t_now = time.perf_counter()
                t0 = self._start_time.value
                self._wall_time_to_opt.value = (t_now - t0) if t0 > 0 else 0.0
                self._task_solve_time_to_opt.value = task_solve_time
                self._proc_solve_time_to_opt.value = proc_solve_time
                self._nodes_to_opt.value = proc_nodes
                self._work_to_opt.value = proc_work
                return True
        return False

    def get_cutoff(self):
        """Returns (best_energy, task_id)."""
        return self._best_obj.value, self._best_task_id.value

    def get_opt_stats(self):
        """Returns a snapshot dictionary of the best incumbent discovery stats."""
        with self._lock:
            return {
                "best_obj": self._best_obj.value if self._best_obj.value < math.inf else None,
                "task_id": self._best_task_id.value,
                "processor_id": self._best_proc_id.value,
                "wall_time_to_opt_s": self._wall_time_to_opt.value if self._wall_time_to_opt.value < math.inf else None,
                "proc_solve_time_to_opt_s": self._proc_solve_time_to_opt.value if self._proc_solve_time_to_opt.value < math.inf else None,
                "task_solve_time_to_opt_s": self._task_solve_time_to_opt.value if self._task_solve_time_to_opt.value < math.inf else None,
                "nodes_to_opt": self._nodes_to_opt.value if self._nodes_to_opt.value < math.inf else None,
                "work_to_opt": self._work_to_opt.value if self._work_to_opt.value < math.inf else None,
            }


def make_gurobi_sync_callback(
    tracker: SharedIncumbentTracker,
    task_id: int,
    worker_id: int = -1,
    prior_proc_solve_time: float = 0.0,
    prior_proc_nodes: float = 0.0,
    prior_proc_work: float = 0.0,
    prefix_str: str = "",
    check_interval: int = 50,
    share_cutoffs: bool = True,
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
                rt = model.cbGet(gb.GRB.Callback.RUNTIME)
                nodes = model.cbGet(gb.GRB.Callback.MIPSOL_NODCNT)
                try:
                    work = model.cbGet(gb.GRB.Callback.WORK)
                except Exception:
                    work = 0.0

                is_new = tracker.try_update(
                    obj,
                    task_id=task_id,
                    processor_id=worker_id,
                    task_solve_time=rt,
                    task_nodes=nodes,
                    task_work=work,
                    proc_solve_time=prior_proc_solve_time + rt,
                    proc_nodes=prior_proc_nodes + nodes,
                    proc_work=prior_proc_work + work,
                )
                if is_new:
                    print(
                        f"  [Task {task_id:02d} | Proc {worker_id} | {prefix_str}] -> New GLOBAL BEST energy: {obj:.4f}"
                    )

            elif where == gb.GRB.Callback.MIP:
                node_cnt = int(model.cbGet(gb.GRB.Callback.MIP_NODCNT))
                if node_cnt - last_node[0] < check_interval:
                    return
                last_node[0] = node_cnt

                best_local = model.cbGet(gb.GRB.Callback.MIP_OBJBST)
                if best_local < gb.GRB.INFINITY:
                    try:
                        work = model.cbGet(gb.GRB.Callback.WORK)
                    except Exception:
                        work = 0.0
                    rt = model.cbGet(gb.GRB.Callback.RUNTIME)
                    tracker.try_update(
                        best_local,
                        task_id=task_id,
                        processor_id=worker_id,
                        task_solve_time=rt,
                        task_nodes=node_cnt,
                        task_work=work,
                        proc_solve_time=prior_proc_solve_time + rt,
                        proc_nodes=prior_proc_nodes + node_cnt,
                        proc_work=prior_proc_work + work,
                    )

                if share_cutoffs:
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


def get_orbit_sizes(grid_size, group):
    """Loads orbit multiplicity sizes for a given cubic grid and space group."""
    grid_path = root_dir / "data/grids" / f"CO{grid_size}G{group}.json"
    with open(grid_path, "r") as f:
        orbits = json.load(f)
    orb_key = list(orbits.keys())
    return [len(orbits[k]) + 1 for k in orb_key]


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
    Builds the base Gurobi model ONCE on this processor, then solves all assigned
    prefix tasks in a loop by mutating variable bounds (LB/UB) + model.reset(0).
    Stores (best_energy, best_solutions, worker_stats) in return_dict[worker_id].
    Shares cutoffs and incumbents in real time via tracker if enabled.
    """
    t_worker_start = time.perf_counter()

    best_energy = float("inf")
    best_res = None
    task_stats = []

    if not tasks:
        worker_stats = {
            "tasks": [],
            "wall_time": time.perf_counter() - t_worker_start,
            "solve_time": 0.0,
            "nodes": 0.0,
            "work": 0.0,
            "iter_count": 0.0,
            "best_energy": float("inf"),
            "solve_time_to_best_s": None,
            "nodes_to_best": None,
            "work_to_best": None,
        }
        return_dict[worker_id] = (best_energy, best_res, worker_stats)
        return

    group = allocate_kwargs.get("group", "1")
    top = allocate_kwargs.get("top", 1)
    verbose = allocate_kwargs.get("verbose", False)
    time_limit = allocate_kwargs.get("TimeLimit", 0)

    # 1. Build base model ONCE on this processor (without solving yet)
    t0_base = time.perf_counter()
    allocate = Allocate(*allocate_args)
    allocate.build_base_model(group=group, write_lp=False, verbose=verbose)
    base_build_time = time.perf_counter() - t0_base

    cum_proc_solve_time = 0.0
    cum_proc_nodes = 0.0
    cum_proc_work = 0.0
    cum_proc_iters = 0.0

    best_proc_energy = float("inf")
    proc_solve_time_to_best = None
    proc_nodes_to_best = None
    proc_work_to_best = None

    for idx, (task_id, task_size, prefix) in enumerate(tasks):
        prefix_str = " ".join(str(s) for s in prefix).strip()
        cb = None
        cb_state = {"pruned_by_peer": False}
        cutoff_val = None

        if tracker is not None:
            if share_cutoffs:
                curr_best, _ = tracker.get_cutoff()
                if curr_best < math.inf:
                    cutoff_val = curr_best - 1e-6
            cb, cb_state = make_gurobi_sync_callback(
                tracker,
                task_id,
                worker_id=worker_id,
                prior_proc_solve_time=cum_proc_solve_time,
                prior_proc_nodes=cum_proc_nodes,
                prior_proc_work=cum_proc_work,
                prefix_str=prefix_str,
                share_cutoffs=share_cutoffs,
            )

        t0_task = time.perf_counter()
        res = allocate.optimize_cube_symmetry_ase(
            group=group,
            PoolSolutions=top,
            TimeLimit=time_limit,
            verbose=verbose,
            prefix=prefix,
            threads=1,
            write_lp=False,
            Cutoff=cutoff_val,
            callback=cb,
            reuse_model=True,
        )
        wall = time.perf_counter() - t0_task

        st = dict(allocate.last_stats or {})
        # The base model build time is attributed to the first task on this processor
        st["build_time"] = (base_build_time if idx == 0 else 0.0) + st.get(
            "build_time", 0.0
        )

        task_solve = st.get("solve_time", 0.0)
        task_nodes = st.get("nodes", 0.0)
        task_work = st.get("work", 0.0)
        task_iters = st.get("iter_count", 0.0)

        task_obj = st.get("obj")
        task_time_to_opt = st.get("time_to_opt_s", task_solve)
        task_nodes_to_opt = st.get("nodes_to_opt", task_nodes)
        task_work_to_opt = st.get("work_to_opt", task_work)

        if task_obj is not None and task_obj < float("inf"):
            if tracker is not None:
                tracker.try_update(
                    task_obj,
                    task_id=task_id,
                    processor_id=worker_id,
                    task_solve_time=task_time_to_opt,
                    task_nodes=task_nodes_to_opt,
                    task_work=task_work_to_opt,
                    proc_solve_time=cum_proc_solve_time + task_time_to_opt,
                    proc_nodes=cum_proc_nodes + task_nodes_to_opt,
                    proc_work=cum_proc_work + task_work_to_opt,
                )
            if task_obj < best_proc_energy:
                best_proc_energy = task_obj
                proc_solve_time_to_best = cum_proc_solve_time + task_time_to_opt
                proc_nodes_to_best = cum_proc_nodes + task_nodes_to_opt
                proc_work_to_best = cum_proc_work + task_work_to_opt

        cum_proc_solve_time += task_solve
        cum_proc_nodes += task_nodes
        cum_proc_work += task_work
        cum_proc_iters += task_iters

        st.update(
            {
                "task_id": task_id,
                "processor": worker_id,
                "configurations": task_size,
                "prefix": prefix_str,
                "wall_time": wall,
                "pruned_by_peer": cb_state.get("pruned_by_peer", False),
                "work": task_work,
                "iter_count": task_iters,
                "time_to_opt_s": task_time_to_opt if task_obj is not None else None,
                "nodes_to_opt": task_nodes_to_opt if task_obj is not None else None,
                "work_to_opt": task_work_to_opt if task_obj is not None else None,
                "proc_solve_time_so_far": cum_proc_solve_time,
            }
        )
        task_stats.append(st)

        if res is not None:
            solutions, runtime, objVal = res
            if objVal < best_energy:
                best_energy = objVal
                best_res = solutions

    worker_stats = {
        "tasks": task_stats,
        "wall_time": time.perf_counter() - t_worker_start,
        "solve_time": cum_proc_solve_time,
        "nodes": cum_proc_nodes,
        "work": cum_proc_work,
        "iter_count": cum_proc_iters,
        "best_energy": best_proc_energy,
        "solve_time_to_best_s": proc_solve_time_to_best,
        "nodes_to_best": proc_nodes_to_best,
        "work_to_best": proc_work_to_best,
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
    oversample_factor=4,
):
    """
    Decomposes the search space using orbit-aware exact DP with max-heap balanced splitting,
    schedules the sub-IPs over p processors with LPT, and solves them in parallel using
    model reuse (in-place bound fixing) across tasks on each processor.
    Shares incumbent energies and cutoffs across processes in real time if share_cutoffs=True.

    Returns (best_energy, best_solutions, parallel_time), where parallel_time is the
    makespan: the summed Gurobi solve time of the most loaded processor.
    With return_stats=True a fourth element with detailed per-task/per-processor
    statistics is returned.
    """
    # Pre-generate / load Ewald and Buckingham/garnet matrices ONCE in main process
    # to eliminate file-system race conditions across child processes
    get_Ewald(grid_size, cell_size)
    types = list(ions_count.keys())
    if getattr(phase, "garnet", False):
        if hasattr(phase, "gar_param"):
            for ion_pair in phase.gar_param:
                if ion_pair[0] in types and ion_pair[1] in types:
                    get_garnet(ion_pair, grid_size, cell_size, phase)
    elif getattr(phase, "buck", None):
        for ion_pair in phase.buck:
            if ion_pair[0] in types and ion_pair[1] in types:
                get_Buck(ion_pair, grid_size, cell_size, phase)

    orb_sizes = get_orbit_sizes(grid_size, group)
    algo_name = "Orbit-Aware Balanced DP"
    print(
        f"Decomposing search space using {algo_name}. Grid: {grid_size}, Group: {group}, "
        f"Orbits: {len(orb_sizes)}, Ions: {ions_count}"
    )

    t_decomp_start = time.perf_counter()
    orbit_tasks, total_configs = decompose_orbits_balanced(
        orb_sizes,
        ions_count,
        p,
        oversample_factor=oversample_factor,
        max_tasks=max_tasks,
    )

    all_tasks = [(cnt, list(prefix)) for prefix, cnt in orbit_tasks]
    if not all_tasks:
        all_tasks = [(1, [])]

    print(
        f"Generated {len(all_tasks)} independent search tasks (total configurations: {total_configs})."
    )

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

    tracker = SharedIncumbentTracker()

    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    t_parallel_start = time.perf_counter()
    tracker.set_start_time(t_parallel_start)
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
                    "nodes": wstats.get("nodes", 0.0),
                    "work": wstats.get("work", 0.0),
                    "iter_count": wstats.get("iter_count", 0.0),
                    "solve_time_to_best_s": wstats.get("solve_time_to_best_s"),
                    "nodes_to_best": wstats.get("nodes_to_best"),
                    "work_to_best": wstats.get("work_to_best"),
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
                    "nodes": float("nan"),
                    "work": float("nan"),
                    "iter_count": float("nan"),
                    "solve_time_to_best_s": float("nan"),
                    "nodes_to_best": float("nan"),
                    "work_to_best": float("nan"),
                }
            )

    # Parallel time = makespan over processors (solve time only)
    parallel_time = max((ps["solve_time"] for ps in processor_stats), default=0.0)

    # Tag each task and processor with whether it hit the global optimum
    for t in task_stats:
        t["found_global_opt"] = (
            t.get("obj") is not None
            and abs(t.get("obj") - overall_best_energy) <= 1e-4
        )
    for ps in processor_stats:
        ps["found_global_opt"] = (
            ps.get("best_energy") is not None
            and abs(ps.get("best_energy") - overall_best_energy) <= 1e-4
        )

    opt_stats = tracker.get_opt_stats()
    decomp_wall_time_to_opt_s = opt_stats.get("wall_time_to_opt_s")
    decomp_solve_time_to_opt_s = opt_stats.get("proc_solve_time_to_opt_s")
    decomp_nodes_to_opt = opt_stats.get("nodes_to_opt")
    decomp_work_to_opt = opt_stats.get("work_to_opt")
    first_opt_task_id = opt_stats.get("task_id")
    first_opt_processor = opt_stats.get("processor_id")

    # Fallback if tracker didn't catch an incumbent
    if (
        decomp_solve_time_to_opt_s is None
        or decomp_solve_time_to_opt_s == float("inf")
    ) and overall_best_energy < float("inf"):
        winning_procs = [ps for ps in processor_stats if ps.get("found_global_opt")]
        if winning_procs:
            best_p = min(
                winning_procs,
                key=lambda x: x.get("solve_time_to_best_s") or float("inf"),
            )
            first_opt_processor = best_p["processor"]
            decomp_solve_time_to_opt_s = best_p.get("solve_time_to_best_s")
            decomp_nodes_to_opt = best_p.get("nodes_to_best")
            decomp_work_to_opt = best_p.get("work_to_best")

    print(
        f"\nOverall minimum IP energy across all processors: {overall_best_energy:.4f}"
    )
    print(f"Parallel (makespan) solve time: {parallel_time:.4f}s")
    if decomp_solve_time_to_opt_s is not None and decomp_solve_time_to_opt_s < float("inf"):
        print(
            f"First hit OPT on Processor {first_opt_processor} (Task {first_opt_task_id}) "
            f"at solve time: {decomp_solve_time_to_opt_s:.4f}s (wall clock: {decomp_wall_time_to_opt_s:.4f}s, "
            f"nodes: {decomp_nodes_to_opt}, work: {decomp_work_to_opt:.4f})"
        )

    if not return_stats:
        return overall_best_energy, overall_best_res, parallel_time

    symbols = list(ions_count.keys())
    counts = [ions_count[s] for s in symbols]

    stats = {
        "p": p,
        "algorithm": algo_name,
        "share_cutoffs": share_cutoffs,
        "symbols": symbols,
        "counts": counts,
        "num_tasks": len(all_tasks),
        "total_configurations": total_configs,
        "decomposition_time": decomposition_time,
        "parallel_wall_time": parallel_wall_time,
        "parallel_time": parallel_time,
        "first_opt_task_id": first_opt_task_id,
        "first_opt_processor": first_opt_processor,
        "decomp_wall_time_to_opt_s": decomp_wall_time_to_opt_s,
        "decomp_solve_time_to_opt_s": decomp_solve_time_to_opt_s,
        "decomp_nodes_to_opt": decomp_nodes_to_opt,
        "decomp_work_to_opt": decomp_work_to_opt,
        "sum_subIP_nodes": sum(t.get("nodes", 0.0) for t in task_stats),
        "sum_subIP_work": sum(t.get("work", 0.0) for t in task_stats),
        "sum_subIP_iters": sum(t.get("iter_count", 0.0) for t in task_stats),
        "processors": processor_stats,
        "tasks": sorted(task_stats, key=lambda t: t["task_id"]),
    }
    return overall_best_energy, overall_best_res, parallel_time, stats
