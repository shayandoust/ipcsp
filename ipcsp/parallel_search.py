import ctypes
import json
import math
import multiprocessing
import time
import gurobipy as gb

from ipcsp import root_dir
from ipcsp.integer_program import (
    Allocate,
    append_progress_record,
    classify_unexp_trend,
    compute_mip_gap_pct,
)
from ipcsp.decomposition import (
    algo1_decomposition,
    algo2_decomposition,
    decompose,
    is_balanced,
    lpt_schedule,
    build_orbit_dp,
    decompose_orbits_balanced,
)
from ipcsp.matrix_generator import Phase, get_Ewald, get_Buck, get_garnet, get_fcc_alloy


class SharedIncumbentTracker:
    """
    Process-safe shared-memory coordinator that allows p independent Gurobi processes
    to broadcast and consume the best global objective value (cutoff) in real time,
    while capturing the exact time and solver effort to first discover optimal solutions
    and tracking live MIPGap (%) and Unexp (unexplored B&B nodes) across all workers.
    """

    def __init__(
        self,
        initial_cutoff: float = math.inf,
        tolerance: float = 1e-5,
        num_procs: int = 1,
        total_tasks: int = 0,
    ):
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
        self.num_procs = max(int(num_procs), 1)
        self.total_tasks = int(total_tasks)

        n = self.num_procs
        self._worker_active = multiprocessing.Array(ctypes.c_int, n)
        self._worker_task_id = multiprocessing.Array(ctypes.c_int, [-1] * n)
        self._worker_tasks_done = multiprocessing.Array(ctypes.c_int, n)
        self._worker_tasks_total = multiprocessing.Array(ctypes.c_int, n)
        self._worker_pruned = multiprocessing.Array(ctypes.c_int, n)
        self._worker_objbst = multiprocessing.Array(ctypes.c_double, [math.inf] * n)
        self._worker_objbnd = multiprocessing.Array(ctypes.c_double, [-math.inf] * n)
        self._worker_mipgap = multiprocessing.Array(ctypes.c_double, [math.inf] * n)
        self._worker_unexp = multiprocessing.Array(ctypes.c_double, n)
        self._worker_peak_unexp = multiprocessing.Array(ctypes.c_double, n)
        self._worker_expl = multiprocessing.Array(ctypes.c_double, n)
        self._worker_cum_nodes = multiprocessing.Array(ctypes.c_double, n)
        self._worker_cum_work = multiprocessing.Array(ctypes.c_double, n)
        self._worker_cur_work = multiprocessing.Array(ctypes.c_double, n)
        self._worker_cum_iters = multiprocessing.Array(ctypes.c_double, n)
        self._worker_cur_iters = multiprocessing.Array(ctypes.c_double, n)

    def set_start_time(self, t0: float):
        with self._lock:
            self._start_time.value = t0

    def init_worker(self, worker_id: int, num_assigned_tasks: int):
        if 0 <= worker_id < self.num_procs:
            self._worker_active[worker_id] = 1
            self._worker_tasks_total[worker_id] = int(num_assigned_tasks)
            self._worker_tasks_done[worker_id] = 0
            self._worker_pruned[worker_id] = 0

    def start_worker_task(self, worker_id: int, task_id: int):
        if 0 <= worker_id < self.num_procs:
            self._worker_task_id[worker_id] = int(task_id)
            self._worker_objbst[worker_id] = math.inf
            self._worker_objbnd[worker_id] = -math.inf
            self._worker_mipgap[worker_id] = math.inf
            self._worker_unexp[worker_id] = 0.0
            self._worker_expl[worker_id] = 0.0
            self._worker_cur_work[worker_id] = 0.0
            self._worker_cur_iters[worker_id] = 0.0

    def update_worker_progress(
        self,
        worker_id: int,
        best_local: float,
        best_bound: float,
        unexp_nodes: float,
        expl_nodes: float,
        work_units: float,
        iters: float = 0.0,
    ):
        if not (0 <= worker_id < self.num_procs):
            return
        if abs(best_local) < 1e90:
            self._worker_objbst[worker_id] = best_local
        if abs(best_bound) < 1e90:
            self._worker_objbnd[worker_id] = best_bound

        global_best = self._best_obj.value
        ref_obj = min(
            best_local if abs(best_local) < 1e90 else math.inf,
            global_best if abs(global_best) < 1e90 else math.inf,
        )
        if ref_obj < 1e90 and abs(best_bound) < 1e90:
            gap_pct = max(0.0, 100.0 * (ref_obj - best_bound) / max(abs(ref_obj), 1e-10))
            self._worker_mipgap[worker_id] = gap_pct
        else:
            self._worker_mipgap[worker_id] = math.inf

        self._worker_unexp[worker_id] = float(unexp_nodes)
        if unexp_nodes > self._worker_peak_unexp[worker_id]:
            self._worker_peak_unexp[worker_id] = float(unexp_nodes)
        self._worker_expl[worker_id] = float(expl_nodes)
        self._worker_cur_work[worker_id] = float(work_units)
        self._worker_cur_iters[worker_id] = float(iters)

    def finish_worker_task(
        self,
        worker_id: int,
        task_nodes: float,
        task_work: float,
        task_iters: float,
        pruned: bool = False,
    ):
        if 0 <= worker_id < self.num_procs:
            self._worker_tasks_done[worker_id] += 1
            if pruned:
                self._worker_pruned[worker_id] += 1
            self._worker_cum_nodes[worker_id] += float(task_nodes)
            self._worker_cum_work[worker_id] += float(task_work)
            self._worker_cum_iters[worker_id] += float(task_iters)
            self._worker_unexp[worker_id] = 0.0
            self._worker_expl[worker_id] = 0.0
            self._worker_cur_work[worker_id] = 0.0
            self._worker_cur_iters[worker_id] = 0.0
            self._worker_mipgap[worker_id] = 0.0

    def finish_worker(self, worker_id: int):
        if 0 <= worker_id < self.num_procs:
            self._worker_active[worker_id] = 0
            self._worker_task_id[worker_id] = -1
            self._worker_unexp[worker_id] = 0.0
            self._worker_expl[worker_id] = 0.0
            self._worker_cur_work[worker_id] = 0.0
            self._worker_cur_iters[worker_id] = 0.0

    def get_global_progress_snapshot(self):
        """
        Aggregates live progress across all p independent Gurobi workers.
        """
        t_now = time.perf_counter()
        t0 = self._start_time.value
        wall_s = (t_now - t0) if t0 > 0 else 0.0
        g_best = self._best_obj.value if self._best_obj.value < 1e90 else None

        active_procs = 0
        tasks_done = 0
        pruned_tasks = 0
        total_unexp = 0
        max_proc_unexp = 0
        sum_peak_unexp = 0
        total_expl = 0
        total_work = 0.0
        total_iters = 0.0
        active_bounds = []
        active_gaps = []

        for w in range(self.num_procs):
            is_act = self._worker_active[w] == 1
            tasks_done += self._worker_tasks_done[w]
            pruned_tasks += self._worker_pruned[w]
            sum_peak_unexp += int(self._worker_peak_unexp[w])
            total_expl += int(self._worker_cum_nodes[w] + (self._worker_expl[w] if is_act else 0.0))
            total_work += float(self._worker_cum_work[w] + (self._worker_cur_work[w] if is_act else 0.0))
            total_iters += float(self._worker_cum_iters[w] + (self._worker_cur_iters[w] if is_act else 0.0))

            if is_act:
                active_procs += 1
                u_w = int(self._worker_unexp[w])
                total_unexp += u_w
                if u_w > max_proc_unexp:
                    max_proc_unexp = u_w
                bnd_w = self._worker_objbnd[w]
                if abs(bnd_w) < 1e90:
                    active_bounds.append(bnd_w)
                gap_w = self._worker_mipgap[w]
                if gap_w < 1e90:
                    active_gaps.append(gap_w)

        worst_active_bd = min(active_bounds) if active_bounds else None
        best_active_bd = max(active_bounds) if active_bounds else None
        avg_subip_gap = (sum(active_gaps) / len(active_gaps)) if active_gaps else None
        min_subip_gap = min(active_gaps) if active_gaps else None
        max_subip_gap = max(active_gaps) if active_gaps else None

        if active_procs == 0 and g_best is not None:
            global_bd = g_best
            global_gap_pct = 0.0
        elif g_best is not None and worst_active_bd is not None:
            global_bd = min(g_best, worst_active_bd)
            global_gap_pct = max(0.0, 100.0 * (g_best - global_bd) / max(abs(g_best), 1e-10))
        else:
            global_bd = worst_active_bd
            global_gap_pct = None

        return {
            "elapsed_wall_s": wall_s,
            "best_obj": g_best,
            "best_bound": global_bd,
            "worst_active_bound": worst_active_bd,
            "best_active_bound": best_active_bd,
            "mip_gap_pct": global_gap_pct,
            "avg_subip_gap_pct": avg_subip_gap,
            "min_subip_gap_pct": min_subip_gap,
            "max_subip_gap_pct": max_subip_gap,
            "workers_with_bound": len(active_bounds),
            "unexplored_nodes": total_unexp,
            "max_worker_unexplored_nodes": max_proc_unexp,
            "sum_worker_peak_unexplored": sum_peak_unexp,
            "explored_nodes": total_expl,
            "work_units": total_work,
            "iter_count": int(total_iters),
            "tasks_done": tasks_done,
            "tasks_total": self.total_tasks,
            "active_workers": active_procs,
            "pruned_tasks": pruned_tasks,
        }

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
      1. Publishes newly found integer solutions (MIPSOL) to the shared tracker
         with immediate MIPGap (%) and Unexp (unexplored B&B nodes) reporting.
      2. Periodically updates shared worker progress (MIPGap, BestBd, Unexp, Expl)
         and checks if the subtree's lower bound (OBJBND) exceeds the global
         cutoff found by any peer, terminating the search early if it cannot beat it.
    """
    last_node = [-1]
    last_rt = [-10.0]
    local_cache = {
        "best_obj": math.inf,
        "best_bnd": -math.inf,
        "unexp": 0,
        "expl": 0,
        "work": 0.0,
        "iters": 0.0,
    }
    state = {"pruned_by_peer": False, "peer_cutoff": math.inf}

    def _callback(model, where):
        if where not in (
            gb.GRB.Callback.MIP,
            gb.GRB.Callback.MIPSOL,
            gb.GRB.Callback.MIPNODE,
        ):
            return
        try:
            if where == gb.GRB.Callback.MIPNODE:
                bnd = model.cbGet(gb.GRB.Callback.MIPNODE_OBJBND)
                bst = model.cbGet(gb.GRB.Callback.MIPNODE_OBJBST)
                if abs(bnd) < 1e90 and bnd > local_cache["best_bnd"]:
                    local_cache["best_bnd"] = bnd
                if abs(bst) < 1e90 and bst < local_cache["best_obj"]:
                    local_cache["best_obj"] = bst

            elif where == gb.GRB.Callback.MIPSOL:
                obj = model.cbGet(gb.GRB.Callback.MIPSOL_OBJ)
                bst = model.cbGet(gb.GRB.Callback.MIPSOL_OBJBST)
                bnd = model.cbGet(gb.GRB.Callback.MIPSOL_OBJBND)
                rt = model.cbGet(gb.GRB.Callback.RUNTIME)
                nodes = int(model.cbGet(gb.GRB.Callback.MIPSOL_NODCNT))
                try:
                    work = model.cbGet(gb.GRB.Callback.WORK)
                except Exception:
                    work = 0.0

                if obj < local_cache["best_obj"]:
                    local_cache["best_obj"] = obj
                if abs(bst) < 1e90 and bst < local_cache["best_obj"]:
                    local_cache["best_obj"] = bst
                if abs(bnd) < 1e90 and bnd > local_cache["best_bnd"]:
                    local_cache["best_bnd"] = bnd
                local_cache["expl"] = max(local_cache["expl"], nodes)
                local_cache["work"] = max(local_cache["work"], work)

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
                tracker.update_worker_progress(
                    worker_id=worker_id,
                    best_local=local_cache["best_obj"],
                    best_bound=local_cache["best_bnd"],
                    unexp_nodes=local_cache["unexp"],
                    expl_nodes=local_cache["expl"],
                    work_units=local_cache["work"],
                    iters=local_cache["iters"],
                )
                if is_new:
                    snap = tracker.get_global_progress_snapshot()
                    sub_bnd = local_cache["best_bnd"] if abs(local_cache["best_bnd"]) < 1e90 else None
                    sub_gap = compute_mip_gap_pct(obj, sub_bnd)
                    bnd_s = f"{sub_bnd:.4f}" if sub_bnd is not None else "RootLP"
                    gap_s = f"{sub_gap:.2f}%" if sub_gap is not None else "N/A"
                    print(
                        f"  [DECOMP INCUMBENT | t={snap['elapsed_wall_s']:8.1f}s | Task {task_id:03d} | Proc {worker_id:03d} | {prefix_str}] "
                        f"New Global BestObj: {obj:.4f} | SubIP BestBd: {bnd_s} | SubIP Gap: {gap_s} | "
                        f"SubIP Unexp: {local_cache['unexp']:,} | Total Unexp: {snap['unexplored_nodes']:,} | "
                        f"Expl: {snap['explored_nodes']:,} | Tasks: {snap['tasks_done']}/{snap['tasks_total']}",
                        flush=True,
                    )

            elif where == gb.GRB.Callback.MIP:
                node_cnt = int(model.cbGet(gb.GRB.Callback.MIP_NODCNT))
                unexp = int(model.cbGet(gb.GRB.Callback.MIP_NODLFT))
                rt = model.cbGet(gb.GRB.Callback.RUNTIME)
                local_cache["expl"] = node_cnt
                local_cache["unexp"] = unexp

                if (node_cnt - last_node[0] < check_interval) and (rt - last_rt[0] < 2.0):
                    return
                last_node[0] = node_cnt
                last_rt[0] = rt

                best_local = model.cbGet(gb.GRB.Callback.MIP_OBJBST)
                lower_bound = model.cbGet(gb.GRB.Callback.MIP_OBJBND)
                itc = model.cbGet(gb.GRB.Callback.MIP_ITRCNT)
                try:
                    work = model.cbGet(gb.GRB.Callback.WORK)
                except Exception:
                    work = 0.0

                if abs(best_local) < 1e90 and best_local < local_cache["best_obj"]:
                    local_cache["best_obj"] = best_local
                if abs(lower_bound) < 1e90 and lower_bound > local_cache["best_bnd"]:
                    local_cache["best_bnd"] = lower_bound
                local_cache["work"] = work
                local_cache["iters"] = itc

                if best_local < gb.GRB.INFINITY:
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

                tracker.update_worker_progress(
                    worker_id=worker_id,
                    best_local=local_cache["best_obj"],
                    best_bound=local_cache["best_bnd"],
                    unexp_nodes=unexp,
                    expl_nodes=node_cnt,
                    work_units=work,
                    iters=itc,
                )

                if share_cutoffs:
                    global_best, src_id = tracker.get_cutoff()
                    if global_best < math.inf:
                        if lower_bound >= global_best - tracker.tolerance:
                            state["pruned_by_peer"] = True
                            state["peer_cutoff"] = global_best
                            model.terminate()
        except Exception:
            pass

    return _callback, state


def get_orbit_sizes(grid_size, group, phase=None):
    """Loads orbit multiplicity sizes for a given cubic/FCC grid and space group."""
    if getattr(phase, "fcc_alloy", False) and str(group) == "1":
        return [1] * (4 * grid_size**3)
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
    threads_per_worker=None,
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
        if tracker is not None:
            tracker.init_worker(worker_id, 0)
            tracker.finish_worker(worker_id)
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

    if tracker is not None:
        tracker.init_worker(worker_id, len(tasks))

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
            tracker.start_worker_task(worker_id, task_id)
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
            threads=threads_per_worker,
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
                is_new = tracker.try_update(
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
                if is_new:
                    snap = tracker.get_global_progress_snapshot()
                    print(
                        f"  [DECOMP INCUMBENT | t={snap['elapsed_wall_s']:8.1f}s | Task {task_id:03d} | Proc {worker_id:03d} | {prefix_str}] "
                        f"New Global BestObj: {task_obj:.4f} | SubIP Gap: 0.00% | "
                        f"Total Unexp: {snap['unexplored_nodes']:,} | Expl: {snap['explored_nodes']:,} | "
                        f"Tasks: {snap['tasks_done']}/{snap['tasks_total']}",
                        flush=True,
                    )
            if task_obj < best_proc_energy:
                best_proc_energy = task_obj
                proc_solve_time_to_best = cum_proc_solve_time + task_time_to_opt
                proc_nodes_to_best = cum_proc_nodes + task_nodes_to_opt
                proc_work_to_best = cum_proc_work + task_work_to_opt

        was_pruned = bool(
            cb_state.get("pruned_by_peer", False) or st.get("status") == gb.GRB.CUTOFF
        )
        if tracker is not None:
            tracker.finish_worker_task(
                worker_id,
                task_nodes=task_nodes,
                task_work=task_work,
                task_iters=task_iters,
                pruned=was_pruned,
            )

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
                "pruned_by_peer": was_pruned,
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

    if tracker is not None:
        tracker.finish_worker(worker_id)

    if getattr(allocate, "model", None) is not None:
        try:
            allocate.model.dispose()
            gb.disposeDefaultEnv()
        except Exception:
            pass

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
    num_procs=None,
    threads_per_worker=None,
    progress_interval=30.0,
    progress_dir=None,
    progress_meta=None,
):
    """
    Decomposes the search space using orbit-aware exact DP with max-heap balanced splitting,
    schedules the sub-IPs over worker processes with LPT, and solves them in parallel using
    model reuse (in-place bound fixing) across tasks on each processor.
    Defaults to num_procs = p workers and threads_per_worker = 1 threads.
    Shares incumbent energies and cutoffs across processes in real time if share_cutoffs=True.

    Returns (best_energy, best_solutions, parallel_time), where parallel_time is the
    makespan: the summed Gurobi solve time of the most loaded processor.
    With return_stats=True a fourth element with detailed per-task/per-processor
    statistics is returned.
    """
    if num_procs is None:
        num_procs = p
    if threads_per_worker is None:
        threads_per_worker = 1
    # Pre-generate / load Ewald and Buckingham/garnet/FCC matrices ONCE in main process
    # to eliminate file-system race conditions across child processes
    types = list(ions_count.keys())
    if getattr(phase, "fcc_alloy", False):
        get_fcc_alloy(grid_size, cell_size, phase)
    else:
        get_Ewald(grid_size, cell_size)
        if getattr(phase, "garnet", False):
            if hasattr(phase, "gar_param"):
                for ion_pair in phase.gar_param:
                    if ion_pair[0] in types and ion_pair[1] in types:
                        get_garnet(ion_pair, grid_size, cell_size, phase)
        elif getattr(phase, "buck", None):
            for ion_pair in phase.buck:
                if ion_pair[0] in types and ion_pair[1] in types:
                    get_Buck(ion_pair, grid_size, cell_size, phase)

    orb_sizes = get_orbit_sizes(grid_size, group, phase=phase)
    algo_name = "Orbit-Aware Balanced DP"
    print(
        f"Decomposing search space using {algo_name}. Grid: {grid_size}, Group: {group}, "
        f"Orbits: {len(orb_sizes)}, Ions: {ions_count}"
    )

    t_decomp_start = time.perf_counter()
    orbit_tasks, total_configs = decompose_orbits_balanced(
        orb_sizes,
        ions_count,
        num_procs,
        oversample_factor=oversample_factor,
        max_tasks=max_tasks,
    )

    all_tasks = [(cnt, list(prefix)) for prefix, cnt in orbit_tasks]
    if not all_tasks:
        all_tasks = [(1, [])]

    print(
        f"Generated {len(all_tasks)} independent search tasks (total configurations: {total_configs})."
    )

    processors, loads = lpt_schedule(all_tasks, num_procs)
    decomposition_time = time.perf_counter() - t_decomp_start
    print(f"Scheduled tasks across {num_procs} processors using LPT. Machine loads: {loads}")

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

    tracker = SharedIncumbentTracker(
        num_procs=num_procs,
        total_tasks=len(all_tasks),
    )

    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    t_parallel_start = time.perf_counter()
    tracker.set_start_time(t_parallel_start)
    processes = []
    for i in range(num_procs):
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
                threads_per_worker,
            ),
        )
        processes.append(proc)
        proc.start()

    meta = dict(progress_meta or {})
    prog_hist = {
        "last_wall": 0.0,
        "last_unexp": 0,
        "peak_total_unexp": 0,
        "last_best_obj": math.inf,
    }

    def _emit_decomp_progress(event_type="PROGRESS"):
        snap = tracker.get_global_progress_snapshot()
        wall_s = snap["elapsed_wall_s"]
        dt = max(wall_s - prog_hist["last_wall"], 1e-3) if prog_hist["last_wall"] > 0 else max(wall_s, 1e-3)
        unexp = snap["unexplored_nodes"]
        if unexp > prog_hist["peak_total_unexp"]:
            prog_hist["peak_total_unexp"] = unexp
        peak_u = max(prog_hist["peak_total_unexp"], snap["sum_worker_peak_unexplored"])
        expl = snap["explored_nodes"]
        rate, trend_lbl, unexp_str = classify_unexp_trend(
            unexp, prog_hist["last_unexp"], peak_u, dt, expl
        )

        b_obj = snap["best_obj"]
        b_bnd = snap["best_bound"]
        g_gap = snap["mip_gap_pct"]
        avg_gap = snap["avg_subip_gap_pct"]

        obj_s = f"{b_obj:.4f}" if b_obj is not None else "None"
        bnd_s = (
            f"{b_bnd:.4f}"
            if b_bnd is not None
            else f"RootLP ({snap['workers_with_bound']}/{snap['active_workers']} rdy)"
        )
        gap_s = f"{g_gap:.2f}%" if g_gap is not None else "N/A"
        avg_gap_s = f" (AvgSubIP: {avg_gap:.2f}%)" if avg_gap is not None else ""
        pct_tasks = (100.0 * snap["tasks_done"] / snap["tasks_total"]) if snap["tasks_total"] > 0 else 100.0

        print(
            f"  [DECOMP {event_type:<9} | t={wall_s:8.1f}s] "
            f"BestObj: {obj_s:>10} | WorstBd: {bnd_s:>14} | MIPGap: {gap_s:>7}{avg_gap_s} | "
            f"Total Unexp: {unexp_str} (MaxProc: {snap['max_worker_unexplored_nodes']:,}) | "
            f"Expl: {expl:,} | Work: {snap['work_units']:.1f} | "
            f"Tasks: {snap['tasks_done']}/{snap['tasks_total']} ({pct_tasks:.1f}%, {snap['pruned_tasks']} pruned) | "
            f"ActiveProcs: {snap['active_workers']}/{num_procs}",
            flush=True,
        )

        row = {
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "test": meta.get("test", getattr(phase, "name", "")),
            "mode": meta.get("mode", "decomposition"),
            "p": p,
            "threads": threads_per_worker,
            "event": event_type,
            "phase": "FINISHED" if event_type == "FINISHED" else "DECOMP_B&B",
            "elapsed_wall_s": round(wall_s, 3),
            "solver_runtime_s": round(wall_s, 3),
            "best_obj": round(b_obj, 6) if b_obj is not None else None,
            "best_bound": round(b_bnd, 6) if b_bnd is not None else None,
            "mip_gap_pct": round(g_gap, 4) if g_gap is not None else None,
            "avg_subip_gap_pct": round(avg_gap, 4) if avg_gap is not None else None,
            "unexplored_nodes": int(unexp),
            "peak_unexplored_nodes": int(peak_u),
            "max_worker_unexplored_nodes": int(snap["max_worker_unexplored_nodes"]),
            "unexp_rate_per_s": round(rate, 2),
            "unexp_trend": trend_lbl,
            "explored_nodes": int(expl),
            "work_units": round(snap["work_units"], 3),
            "iter_count": int(snap["iter_count"]),
            "tasks_done": int(snap["tasks_done"]),
            "tasks_total": int(snap["tasks_total"]),
            "active_workers": int(snap["active_workers"]),
            "pruned_tasks": int(snap["pruned_tasks"]),
        }
        append_progress_record(progress_dir, row)
        prog_hist["last_wall"] = wall_s
        prog_hist["last_unexp"] = unexp

    last_hb = time.perf_counter()
    while any(proc.is_alive() for proc in processes):
        time.sleep(1.0)
        now = time.perf_counter()
        curr_best, _ = tracker.get_cutoff()
        if curr_best < prog_hist["last_best_obj"] - 1e-5:
            prog_hist["last_best_obj"] = curr_best
            if progress_dir:
                # Record incumbent snapshot to CSV/JSON immediately
                snap = tracker.get_global_progress_snapshot()
                wall_s = snap["elapsed_wall_s"]
                dt = max(wall_s - prog_hist["last_wall"], 1e-3)
                unexp = snap["unexplored_nodes"]
                if unexp > prog_hist["peak_total_unexp"]:
                    prog_hist["peak_total_unexp"] = unexp
                peak_u = max(prog_hist["peak_total_unexp"], snap["sum_worker_peak_unexplored"])
                rate, trend_lbl, _ = classify_unexp_trend(
                    unexp, prog_hist["last_unexp"], peak_u, dt, snap["explored_nodes"]
                )
                append_progress_record(
                    progress_dir,
                    {
                        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "test": meta.get("test", getattr(phase, "name", "")),
                        "mode": meta.get("mode", "decomposition"),
                        "p": p,
                        "threads": threads_per_worker,
                        "event": "INCUMBENT",
                        "phase": "DECOMP_B&B",
                        "elapsed_wall_s": round(wall_s, 3),
                        "solver_runtime_s": round(wall_s, 3),
                        "best_obj": round(curr_best, 6),
                        "best_bound": round(snap["best_bound"], 6) if snap["best_bound"] is not None else None,
                        "mip_gap_pct": round(snap["mip_gap_pct"], 4) if snap["mip_gap_pct"] is not None else None,
                        "avg_subip_gap_pct": round(snap["avg_subip_gap_pct"], 4) if snap["avg_subip_gap_pct"] is not None else None,
                        "unexplored_nodes": int(unexp),
                        "peak_unexplored_nodes": int(peak_u),
                        "max_worker_unexplored_nodes": int(snap["max_worker_unexplored_nodes"]),
                        "unexp_rate_per_s": round(rate, 2),
                        "unexp_trend": trend_lbl,
                        "explored_nodes": int(snap["explored_nodes"]),
                        "work_units": round(snap["work_units"], 3),
                        "iter_count": int(snap["iter_count"]),
                        "tasks_done": int(snap["tasks_done"]),
                        "tasks_total": int(snap["tasks_total"]),
                        "active_workers": int(snap["active_workers"]),
                        "pruned_tasks": int(snap["pruned_tasks"]),
                    },
                )
        if progress_interval > 0 and (now - last_hb >= progress_interval):
            _emit_decomp_progress("PROGRESS")
            last_hb = now

    for proc in processes:
        proc.join()
    parallel_wall_time = time.perf_counter() - t_parallel_start
    if progress_interval > 0:
        _emit_decomp_progress("FINISHED")

    overall_best_energy = float("inf")
    overall_best_res = []
    processor_stats = []
    task_stats = []

    for i in range(num_procs):
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
        "num_procs": num_procs,
        "threads_per_worker": threads_per_worker,
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
