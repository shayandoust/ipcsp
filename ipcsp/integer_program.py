import csv
import gurobipy as gb
import math
import numpy as np
import os
import time as _time

from ipcsp import root_dir
from ipcsp.matrix_generator import Phase, get_Ewald, get_Buck, get_garnet, get_fcc_alloy
import ase
from ipcsp.grids_and_symmetry import cubic, fcc
import json
from ipcsp.lp_to_bqm import BQM

griddir = root_dir / "data/grids/"


def compute_mip_gap_pct(best_obj, best_bound):
    """
    Computes the relative MIPGap (%) between the best feasible solution (incumbent)
    and the best theoretical bound (BestBd):
      100 * |BestBd - Incumbent| / max(|Incumbent|, 1e-10)
    Returns None if either value is unavailable or infinite.
    """
    if (
        best_obj is None
        or best_bound is None
        or not math.isfinite(best_obj)
        or not math.isfinite(best_bound)
        or abs(best_obj) >= 1e90
        or abs(best_bound) >= 1e90
    ):
        return None
    denom = max(abs(best_obj), 1e-10)
    gap = abs(best_bound - best_obj) / denom
    return max(0.0, 100.0 * gap)


def classify_unexp_trend(unexp, last_unexp, peak_unexp, dt, expl):
    """
    Classifies the branch-and-bound tree unexplored-node trajectory:
      - ROOT_OR_INIT: branching has not started yet (expl == 0 and unexp == 0)
      - GROWING: unexplored nodes are increasing (far from completion)
      - SHRINKING: unexplored nodes are decreasing toward zero (converging)
      - STEADY: unexplored node count is unchanged over the interval
    Returns (rate_per_s, trend_label, formatted_str).
    """
    dt_safe = max(dt, 1e-6)
    delta = unexp - last_unexp
    rate = delta / dt_safe
    pct_peak = (100.0 * unexp / peak_unexp) if peak_unexp > 0 else 0.0
    if unexp == 0 and expl == 0:
        return 0.0, "ROOT_NODE", "0 (Root LP / Cuts)"
    if unexp == 0 and expl > 0:
        return rate, "DRAINED", f"0 (Drained from peak {peak_unexp:,})"
    if delta > 0:
        return rate, "GROWING", f"{unexp:,} (+{rate:.1f}/s | GROWING, peak {peak_unexp:,})"
    if delta < 0:
        return (
            rate,
            "SHRINKING",
            f"{unexp:,} ({rate:.1f}/s | SHRINKING, {pct_peak:.1f}% of peak {peak_unexp:,})",
        )
    return (
        0.0,
        "STEADY",
        f"{unexp:,} (0.0/s | STEADY, {pct_peak:.1f}% of peak {peak_unexp:,})",
    )


PROGRESS_CSV_COLUMNS = [
    "timestamp_utc",
    "test",
    "mode",
    "p",
    "threads",
    "event",
    "phase",
    "elapsed_wall_s",
    "solver_runtime_s",
    "best_obj",
    "best_bound",
    "mip_gap_pct",
    "avg_subip_gap_pct",
    "unexplored_nodes",
    "peak_unexplored_nodes",
    "max_worker_unexplored_nodes",
    "unexp_rate_per_s",
    "unexp_trend",
    "explored_nodes",
    "work_units",
    "iter_count",
    "tasks_done",
    "tasks_total",
    "active_workers",
    "pruned_tasks",
]


def append_progress_record(progress_dir, row_dict):
    """
    Appends a progress row to <progress_dir>/progress_tracking.csv and atomically
    updates <progress_dir>/live_status.json for easy external monitoring.
    """
    if not progress_dir:
        return
    try:
        os.makedirs(progress_dir, exist_ok=True)
        csv_path = os.path.join(progress_dir, "progress_tracking.csv")
        file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
        normalized = {col: row_dict.get(col) for col in PROGRESS_CSV_COLUMNS}
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=PROGRESS_CSV_COLUMNS)
            if not file_exists:
                writer.writeheader()
            writer.writerow(normalized)

        json_tmp = os.path.join(progress_dir, f".live_status.{os.getpid()}.tmp")
        json_path = os.path.join(progress_dir, "live_status.json")
        with open(json_tmp, "w") as jf:
            json.dump(normalized, jf, indent=2)
        os.replace(json_tmp, json_path)
    except Exception:
        pass



class Allocate:

    def __init__(self, ions_count, grid_size, cell_size, phase):

        self.ions = ions_count

        self.phase = phase

        self.grid = grid_size

        self.cell = cell_size

        self.model = None

        # Timing / size statistics of the last optimize_cube_symmetry_ase call
        self.last_stats = None

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

    def solution_to_Atoms(self, i, orbits):

        self.model.params.SolutionNumber = i

        if getattr(self.phase, "fcc_alloy", False):
            grid_positions = fcc(self.grid)
        else:
            grid_positions = cubic(self.grid)
        # ions = list(self.ions.keys())

        # Atoms
        symbols = ""
        positions = []

        for v in self.model.getVars():
            if v.Xn > 0.5:
                # print(v.varName, end=' ')
                t, o = v.varName.split(sep="_")

                # The element itself
                positions.append(grid_positions[int(o)])
                symbols += t

                # And its orbit
                for pos in orbits[o]:
                    positions.append(grid_positions[pos])
                    symbols += t

        # print(symbols, positions)
        return ase.Atoms(
            symbols=symbols,
            scaled_positions=positions,
            cell=[self.cell, self.cell, self.cell],
            pbc=True,
        )

    def build_base_model(self, group="1", write_lp=False, verbose=False):
        """
        Builds the base Gurobi model (variables, stoichiometry constraints, and quadratic objective)
        once without optimizing. Caches the model and variable structures on self.
        """
        t_build_start = _time.perf_counter()

        is_fcc = getattr(self.phase, "fcc_alloy", False)
        N = (4 * self.grid**3) if is_fcc else (self.grid**3)
        T = len(self.ions)  # different types

        if is_fcc and str(group) == "1":
            orbits = {str(i): [] for i in range(N)}
            orb_key = [str(i) for i in range(N)]
            orb_size = [1] * N
            o_pos = list(range(N))
        else:
            # PATH hack
            with open(
                os.path.join(
                    ".",
                    griddir / "CO{grid}G{group}.json".format(grid=self.grid, group=group),
                ),
                "r",
            ) as f:
                orbits = json.load(f)

            orb_key = list(orbits.keys())
            orb_size = [len(orbits[k]) + 1 for k in orb_key]

            # o_pos[position] = number_of_the_orbit
            o_pos = [0] * N
            for o_idx, k in enumerate(orb_key):
                o_pos[int(k)] = o_idx
                for pos in orbits[k]:
                    o_pos[pos] = o_idx

        if verbose:
            print("Generating integer program\n")

        O = len(orbits)
        types = list(self.ions.keys())  # ordered list of elements
        counts = [self.ions[t] for t in types]
        fully_occupied = sum(counts) == N

        model_name = "Ion allocation in {name} with symmetry {group}".format(
            name=self.phase.name, group=group
        )
        last_wls_warn = 0.0
        while True:
            try:
                m = gb.Model(model_name)
                break
            except gb.GurobiError as e:
                msg = str(e)
                if any(k in msg for k in ("Too many sessions", "Network error", "Overage for too long", "license", "session")):
                    now_t = _time.perf_counter()
                    if now_t - last_wls_warn >= 60.0:
                        print(f"WLS license busy ({e}), waiting ~15s to retry...", flush=True)
                        last_wls_warn = now_t
                    _time.sleep(15.0 + (os.getpid() % 17) * 0.35)
                else:
                    raise
        Vars = [[] for _ in range(T)]

        # Create variables
        for i in range(O):  # iterate over all orbitals
            tmp_var = []
            for j in range(T):
                v = m.addVar(vtype=gb.GRB.BINARY, name=str(types[j]) + "_" + orb_key[i])
                Vars[j].append(v)
                tmp_var.append(v)
            if i == 0:
                m.addConstr(gb.LinExpr([1.0] * T, tmp_var) == 1, "first_orbit_has_ion")
            elif fully_occupied:
                m.addConstr(gb.LinExpr([1.0] * T, tmp_var) == 1, f"one_per_orbit_{i}")
            else:
                m.addConstr(gb.LinExpr([1.0] * T, tmp_var) <= 1, f"one_per_orbit_{i}")

        for j in range(T):
            tmp = gb.LinExpr()
            for i in range(O):
                tmp.add(Vars[j][i], orb_size[i])
            m.addConstr(tmp == counts[j], f"number_of_ions_type_{j}")

        if verbose:
            print("Variables and constraints were generated")

        if is_fcc:
            fcc_data = get_fcc_alloy(self.grid, self.cell, self.phase)
            rows = fcc_data["rows"]
            cols = fcc_data["cols"]
            t_A, t_B = types[0], types[1]
            pair_ab = (min(t_A, t_B), max(t_A, t_B))
            v_aa = fcc_data[f"V_{t_A}_{t_A}"]
            v_bb = fcc_data[f"V_{t_B}_{t_B}"]
            v_ab = fcc_data[f"V_{pair_ab[0]}_{pair_ab[1]}"]

            const_offset = float(np.sum(v_bb))
            lin_coeffs = np.zeros(O, dtype=np.float64)
            np.add.at(lin_coeffs, rows, v_ab - v_bb)
            np.add.at(lin_coeffs, cols, v_ab - v_bb)
            q_coeffs = v_aa + v_bb - 2.0 * v_ab

            energy = gb.QuadExpr(const_offset)
            energy.addTerms(lin_coeffs.tolist(), Vars[0])
            v1 = [Vars[0][r] for r in rows]
            v2 = [Vars[0][c] for c in cols]
            energy.addTerms(q_coeffs.tolist(), v1, v2)
        else:
            # Vectorized orbit-reduced quadratic objective construction (P^T U P)
            P = np.zeros((N, O), dtype=np.float64)
            P[np.arange(N), o_pos] = 1.0

            dist = get_Ewald(self.grid, self.cell)
            dist_triu = np.triu(dist, 1) * 2.0
            dist_u = dist_triu + np.diag(np.diag(dist))
            W_ewald = P.T @ dist_u @ P
            W_ewald_sym = P.T @ (dist_triu + dist_triu.T) @ P
            del dist, dist_triu, dist_u

            if verbose:
                print("Ewald sum contribution was added to the objective function")

            Q = np.zeros((T * O, T * O), dtype=np.float64)
            for j1 in range(T):
                q1 = self.phase.charge[types[j1]]
                Q[j1 * O : (j1 + 1) * O, j1 * O : (j1 + 1) * O] += (q1 * q1) * W_ewald
                for j2 in range(j1 + 1, T):
                    q2 = self.phase.charge[types[j2]]
                    Q[j1 * O : (j1 + 1) * O, j2 * O : (j2 + 1) * O] += (
                        q1 * q2
                    ) * W_ewald_sym
            del W_ewald, W_ewald_sym

            if self.phase.garnet:
                if verbose:
                    print("Generating Morse and Leonard parts")
                for ion_pair in self.phase.gar_param:
                    if not ((ion_pair[0] in types) and (ion_pair[1] in types)):
                        continue
                    buck = get_garnet(ion_pair, self.grid, self.cell, self.phase)
                    if ion_pair[0] == ion_pair[1]:
                        j1 = types.index(ion_pair[0])
                        Q[j1 * O : (j1 + 1) * O, j1 * O : (j1 + 1) * O] += (
                            P.T @ buck @ P
                        )
                    else:
                        j1 = types.index(ion_pair[0])
                        j2 = types.index(ion_pair[1])
                        Q[j1 * O : (j1 + 1) * O, j2 * O : (j2 + 1) * O] += (
                            P.T @ (buck + buck.T) @ P
                        )
                    del buck
            else:
                # Buckingham Part
                for ion_pair in self.phase.buck:
                    if not ((ion_pair[0] in types) and (ion_pair[1] in types)):
                        continue
                    buck = get_Buck(ion_pair, self.grid, self.cell, self.phase)
                    if ion_pair[0] == ion_pair[1]:
                        j1 = types.index(ion_pair[0])
                        Q[j1 * O : (j1 + 1) * O, j1 * O : (j1 + 1) * O] += (
                            P.T @ buck @ P
                        )
                    else:
                        j1 = types.index(ion_pair[0])
                        j2 = types.index(ion_pair[1])
                        Q[j1 * O : (j1 + 1) * O, j2 * O : (j2 + 1) * O] += (
                            P.T @ (buck + buck.T) @ P
                        )
                    del buck

            Q_triu = np.triu(Q, 1) + np.tril(Q, -1).T + np.diag(np.diag(Q))
            del Q
            rows, cols = np.nonzero(Q_triu)
            vals = Q_triu[rows, cols]
            del Q_triu
            flat_vars = [Vars[j][i] for j in range(T) for i in range(O)]
            v1 = [flat_vars[r] for r in rows]
            v2 = [flat_vars[c] for c in cols]
            energy = gb.QuadExpr()
            energy.addTerms(vals.tolist(), v1, v2)
            del rows, cols, vals, v1, v2

        if verbose:
            print("Objective function was generated")

        m.setObjective(energy, gb.GRB.MINIMIZE)
        m.update()
        if write_lp:
            if verbose:
                print("Writing model file")
            m.write("model.lp")

        build_time = _time.perf_counter() - t_build_start

        self.model = m
        self._Vars = Vars
        self._types = types
        self._orb_key = orb_key
        self._orb_size = orb_size
        self._orbits = orbits
        self._o_pos = o_pos
        self._built_group = str(group)
        self._base_build_time = build_time
        return m, Vars, types, orb_key

    def optimize_cube_symmetry_ase(
        self,
        group="1",
        PoolSolutions=1,
        TimeLimit=0,
        verbose=True,
        prefix=None,
        threads=None,
        write_lp=False,
        Cutoff=None,
        callback=None,
        reuse_model=True,
        track_progress=False,
        progress_interval=30.0,
        progress_dir=None,
        progress_meta=None,
    ):
        """
        The function to generate an integer program and solve allocation problem using Gurobi.
        We rely on atomic simulation environment to handle allocations afterwards.

        write_lp -- write the model to model.lp (needed by the quantum/QUBO pipeline).
          Default: False to avoid concurrent disk writes during parallel benchmarks.
        reuse_model -- reuse previously built base model on self and solve via LB/UB bound mutation.
        track_progress -- emit real-time incumbent & periodic progress logs with MIPGap (%) and
          Unexp (unexplored B&B nodes) + write progress_tracking.csv / live_status.json.
        After the call, self.last_stats holds build/solve times and model sizes.
        """
        first_build = False
        if (
            not reuse_model
            or self.model is None
            or getattr(self, "_built_group", None) != str(group)
        ):
            self.build_base_model(group=group, write_lp=write_lp, verbose=verbose)
            first_build = True

        t_task_start = _time.perf_counter()
        m = self.model
        Vars = self._Vars
        types = self._types
        orbits = self._orbits
        orb_key = self._orb_key
        T = len(types)
        O = len(orb_key)

        # Reset solver state from previous task while keeping quadratic matrix in memory
        m.reset(0)

        # Reset all variable bounds to binary [0, 1]
        for j in range(T):
            for i in range(O):
                Vars[j][i].LB = 0.0
                Vars[j][i].UB = 1.0

        # Apply prefix via variable bounds (no extra linear constraints added!)
        if prefix:
            for orb_i, symbol in enumerate(prefix):
                if orb_i >= O:
                    break
                if symbol == "V":
                    for j in range(T):
                        Vars[j][orb_i].UB = 0.0  # Fix all ions at orbit orb_i to 0
                else:
                    for j, t_name in enumerate(types):
                        if t_name == symbol:
                            Vars[j][orb_i].LB = 1.0  # Fix chosen ion at orb_i to 1
                        else:
                            Vars[j][orb_i].UB = 0.0  # Fix other ions at orb_i to 0

        if TimeLimit > 0:
            m.params.TimeLimit = TimeLimit
        else:
            m.params.TimeLimit = gb.GRB.INFINITY

        m.params.OutputFlag = 1 if verbose else 0

        if PoolSolutions > 1:
            m.params.PoolSolutions = PoolSolutions
            m.params.PoolSearchMode = 2
        else:
            m.params.PoolSolutions = 1
            m.params.PoolSearchMode = 0

        if threads is not None:
            m.params.Threads = threads

        if Cutoff is not None:
            m.params.Cutoff = Cutoff
        else:
            m.params.Cutoff = gb.GRB.INFINITY

        m.Params.NodefileStart = 1

        m.update()
        if write_lp:
            if verbose:
                print("Writing model file")
            m.write("model.lp")
        task_prep_time = _time.perf_counter() - t_task_start
        build_time = (self._base_build_time if first_build else 0.0) + task_prep_time

        incumbent_records = []
        t0_solve = _time.perf_counter()

        prog_state = {
            "best_obj": math.inf,
            "best_bound": -math.inf,
            "unexp": 0,
            "peak_unexp": 0,
            "expl": 0,
            "work": 0.0,
            "iters": 0.0,
            "phase": "PRESOLVE",
            "last_log_wall": 0.0,
            "last_log_unexp": 0,
        }
        meta = dict(progress_meta or {})

        def _emit_single_ip_progress(event_type, wall_s, rt_s):
            dt = max(wall_s - prog_state["last_log_wall"], 1e-3) if prog_state["last_log_wall"] > 0 else max(wall_s, 1e-3)
            unexp = prog_state["unexp"]
            peak_u = prog_state["peak_unexp"]
            expl = prog_state["expl"]
            rate, trend_lbl, unexp_str = classify_unexp_trend(
                unexp, prog_state["last_log_unexp"], peak_u, dt, expl
            )
            b_obj = prog_state["best_obj"] if prog_state["best_obj"] < 1e90 else None
            b_bnd = prog_state["best_bound"] if abs(prog_state["best_bound"]) < 1e90 else None
            gap_pct = compute_mip_gap_pct(b_obj, b_bnd)

            obj_s = f"{b_obj:.4f}" if b_obj is not None else "None"
            bnd_s = f"{b_bnd:.4f}" if b_bnd is not None else f"None ({prog_state['phase']})"
            gap_s = f"{gap_pct:.2f}%" if gap_pct is not None else "N/A"

            print(
                f"  [SINGLE-IP {event_type:<9} | t={wall_s:8.1f}s] "
                f"BestObj: {obj_s:>11} | BestBd: {bnd_s:>14} | MIPGap: {gap_s:>8} | "
                f"Unexp: {unexp_str} | Expl: {expl:,} | Work: {prog_state['work']:.1f}",
                flush=True,
            )
            row = {
                "timestamp_utc": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
                "test": meta.get("test", self.phase.name),
                "mode": meta.get("mode", "single_ip_baseline"),
                "p": meta.get("p", threads or 1),
                "threads": threads or 1,
                "event": event_type,
                "phase": prog_state["phase"],
                "elapsed_wall_s": round(wall_s, 3),
                "solver_runtime_s": round(rt_s, 3),
                "best_obj": round(b_obj, 6) if b_obj is not None else None,
                "best_bound": round(b_bnd, 6) if b_bnd is not None else None,
                "mip_gap_pct": round(gap_pct, 4) if gap_pct is not None else None,
                "unexplored_nodes": int(unexp),
                "peak_unexplored_nodes": int(peak_u),
                "unexp_rate_per_s": round(rate, 2),
                "unexp_trend": trend_lbl,
                "explored_nodes": int(expl),
                "work_units": round(prog_state["work"], 3),
                "iter_count": int(prog_state["iters"]),
                "tasks_done": 0,
                "tasks_total": 1,
                "active_workers": 1,
                "pruned_tasks": 0,
            }
            append_progress_record(progress_dir, row)
            if event_type == "PROGRESS":
                prog_state["last_log_wall"] = wall_s
                prog_state["last_log_unexp"] = unexp

        def _internal_cb(model, where):
            if not track_progress:
                if where == gb.GRB.Callback.MIPSOL:
                    try:
                        obj = model.cbGet(gb.GRB.Callback.MIPSOL_OBJ)
                        rt = model.cbGet(gb.GRB.Callback.RUNTIME)
                        nodes = int(model.cbGet(gb.GRB.Callback.MIPSOL_NODCNT))
                        try:
                            work = model.cbGet(gb.GRB.Callback.WORK)
                        except Exception:
                            work = 0.0
                        wall = _time.perf_counter() - t0_solve
                        incumbent_records.append({
                            "obj": obj,
                            "runtime": rt,
                            "nodes": nodes,
                            "work": work,
                            "wall": wall,
                        })
                    except Exception:
                        pass
                if callback is not None:
                    callback(model, where)
                return

            if where not in (
                gb.GRB.Callback.MIP,
                gb.GRB.Callback.MIPSOL,
                gb.GRB.Callback.MIPNODE,
                gb.GRB.Callback.PRESOLVE,
            ):
                if callback is not None:
                    callback(model, where)
                return

            try:
                if where == gb.GRB.Callback.PRESOLVE:
                    prog_state["phase"] = "PRESOLVE"
                elif where == gb.GRB.Callback.MIPNODE:
                    prog_state["phase"] = "B&B_SEARCH"
                    bnd = model.cbGet(gb.GRB.Callback.MIPNODE_OBJBND)
                    bst = model.cbGet(gb.GRB.Callback.MIPNODE_OBJBST)
                    if abs(bnd) < 1e90 and bnd > prog_state["best_bound"]:
                        prog_state["best_bound"] = bnd
                    if abs(bst) < 1e90 and bst < prog_state["best_obj"]:
                        prog_state["best_obj"] = bst
                elif where == gb.GRB.Callback.MIP:
                    prog_state["phase"] = "B&B_SEARCH"
                    bst = model.cbGet(gb.GRB.Callback.MIP_OBJBST)
                    bnd = model.cbGet(gb.GRB.Callback.MIP_OBJBND)
                    expl = int(model.cbGet(gb.GRB.Callback.MIP_NODCNT))
                    unexp = int(model.cbGet(gb.GRB.Callback.MIP_NODLFT))
                    itc = model.cbGet(gb.GRB.Callback.MIP_ITRCNT)
                    if abs(bst) < 1e90 and bst < prog_state["best_obj"]:
                        prog_state["best_obj"] = bst
                    if abs(bnd) < 1e90 and bnd > prog_state["best_bound"]:
                        prog_state["best_bound"] = bnd
                    prog_state["expl"] = expl
                    prog_state["unexp"] = unexp
                    if unexp > prog_state["peak_unexp"]:
                        prog_state["peak_unexp"] = unexp
                    prog_state["iters"] = itc
                elif where == gb.GRB.Callback.MIPSOL:
                    prog_state["phase"] = "B&B_SEARCH"
                    obj = model.cbGet(gb.GRB.Callback.MIPSOL_OBJ)
                    bst = model.cbGet(gb.GRB.Callback.MIPSOL_OBJBST)
                    bnd = model.cbGet(gb.GRB.Callback.MIPSOL_OBJBND)
                    rt = model.cbGet(gb.GRB.Callback.RUNTIME)
                    nodes = int(model.cbGet(gb.GRB.Callback.MIPSOL_NODCNT))
                    try:
                        work = model.cbGet(gb.GRB.Callback.WORK)
                    except Exception:
                        work = 0.0
                    wall = _time.perf_counter() - t0_solve

                    is_new_best = obj < prog_state["best_obj"] - 1e-5
                    if obj < prog_state["best_obj"]:
                        prog_state["best_obj"] = obj
                    if abs(bst) < 1e90 and bst < prog_state["best_obj"]:
                        prog_state["best_obj"] = bst
                    if abs(bnd) < 1e90 and bnd > prog_state["best_bound"]:
                        prog_state["best_bound"] = bnd
                    prog_state["expl"] = max(prog_state["expl"], nodes)
                    prog_state["work"] = max(prog_state["work"], work)

                    b_bnd = prog_state["best_bound"] if abs(prog_state["best_bound"]) < 1e90 else None
                    gap_pct = compute_mip_gap_pct(prog_state["best_obj"], b_bnd)

                    incumbent_records.append({
                        "obj": obj,
                        "best_bound": b_bnd,
                        "mip_gap_pct": gap_pct,
                        "unexplored_nodes": prog_state["unexp"],
                        "peak_unexplored_nodes": prog_state["peak_unexp"],
                        "runtime": rt,
                        "nodes": nodes,
                        "work": work,
                        "wall": wall,
                    })
                    if track_progress and is_new_best:
                        _emit_single_ip_progress("INCUMBENT", wall, rt)

                if track_progress and progress_interval > 0:
                    wall_now = _time.perf_counter() - t0_solve
                    if wall_now - prog_state["last_log_wall"] >= progress_interval:
                        try:
                            rt_now = model.cbGet(gb.GRB.Callback.RUNTIME)
                            prog_state["work"] = model.cbGet(gb.GRB.Callback.WORK)
                        except Exception:
                            rt_now = wall_now
                        _emit_single_ip_progress("PROGRESS", wall_now, rt_now)
            except Exception:
                pass
            if callback is not None:
                callback(model, where)

        if track_progress:
            print(
                f"  [SINGLE-IP START     | t=     0.0s] Model built ({m.NumVars:,} vars, {m.NumConstrs:,} constrs, {m.NumQNZs:,} qterms) | "
                f"Starting Gurobi with {threads or 1} threads (progress interval: {progress_interval:.0f}s)...",
                flush=True,
            )

        m.optimize(_internal_cb)
        runtime = m.Runtime
        wall_total = _time.perf_counter() - t0_solve
        node_count = getattr(m, "NodeCount", 0.0)
        work_units = getattr(m, "Work", 0.0)
        iter_count = getattr(m, "IterCount", 0.0)

        try:
            final_bound = m.ObjBound
            if abs(final_bound) < 1e90:
                prog_state["best_bound"] = final_bound
        except Exception:
            pass

        if m.SolCount > 0:
            prog_state["best_obj"] = m.objVal
        if m.status in (gb.GRB.OPTIMAL, gb.GRB.CUTOFF, gb.GRB.INFEASIBLE):
            prog_state["unexp"] = 0
        prog_state["expl"] = int(node_count)
        prog_state["work"] = work_units
        prog_state["iters"] = iter_count
        prog_state["phase"] = "FINISHED"

        final_b_obj = prog_state["best_obj"] if prog_state["best_obj"] < 1e90 else None
        final_b_bnd = prog_state["best_bound"] if abs(prog_state["best_bound"]) < 1e90 else None
        final_gap_pct = (
            0.0
            if m.status == gb.GRB.OPTIMAL
            else compute_mip_gap_pct(final_b_obj, final_b_bnd)
        )

        if track_progress:
            _emit_single_ip_progress("FINISHED", wall_total, runtime)

        time_to_opt = None
        wall_to_opt = None
        nodes_to_opt = None
        work_to_opt = None

        if m.SolCount > 0:
            final_obj = m.objVal
            matching_inc = next(
                (r for r in incumbent_records if abs(r["obj"] - final_obj) <= 1e-4 or r["obj"] <= final_obj + 1e-4),
                None
            )
            if matching_inc is not None:
                time_to_opt = matching_inc["runtime"]
                wall_to_opt = matching_inc["wall"]
                nodes_to_opt = matching_inc["nodes"]
                work_to_opt = matching_inc["work"]
            else:
                time_to_opt = runtime
                wall_to_opt = wall_total
                nodes_to_opt = node_count
                work_to_opt = work_units

        self.last_stats = {
            "build_time": build_time,
            "solve_time": runtime,
            "status": m.status,
            "num_vars": m.NumVars,
            "num_constrs": m.NumConstrs,
            "num_qterms": m.NumQNZs,
            "sol_count": m.SolCount,
            "obj": m.objVal if m.SolCount > 0 else None,
            "best_bound": final_b_bnd,
            "mip_gap_pct": final_gap_pct,
            "unexplored_nodes": int(prog_state["unexp"]),
            "peak_unexplored_nodes": int(prog_state["peak_unexp"]),
            "nodes": node_count,
            "work": work_units,
            "iter_count": iter_count,
            "time_to_opt_s": time_to_opt,
            "wall_to_opt_s": wall_to_opt,
            "nodes_to_opt": nodes_to_opt,
            "work_to_opt": work_to_opt,
            "incumbents": incumbent_records,
        }

        if m.status == gb.GRB.CUTOFF:
            if verbose:
                print("Cutoff! No solution beating cutoff.")
            return None

        if (
            m.status == gb.GRB.OPTIMAL
            or m.status == gb.GRB.TIME_LIMIT
            or m.status == gb.GRB.INTERRUPTED
        ):
            if verbose:
                print("There are", m.SolCount, "solutions")
            if m.SolCount == 0:
                return None
            res = []
            for i in range(m.SolCount):
                res.append(self.solution_to_Atoms(i, orbits))

            if verbose:
                print("\nThe optimal assignment is as follows:")
                for v in m.getVars():
                    try:
                        if v.Xn == 1:
                            print(v.varName, end=" ")
                    except AttributeError:
                        pass
                print()
                print("Minimal energy via optimizer: %g" % m.objVal)

            if PoolSolutions > 1:
                return res, runtime, m.objVal
            else:
                return res[0:1], runtime, m.objVal

        return None

    def optimize_qubo(
        self,
        group="1",
        at_dwave=False,
        num_reads=10,
        infinity_placement=100,
        infinity_orbit=100,
        annealing_time=200,
    ):
        """
        The function to optimise the structure on the quantum annealer
        infinity_orbit is the penalty for putting two ions on the orbit
        infinity_placement is the penalty for placing the incorrect number of ions into the structure
        """
        import neal
        import dimod
        import dwave.embedding
        import dwave.system
        from dwave.system.samplers import DWaveSampler
        from dwave.system.composites import EmbeddingComposite

        print(
            "Running integer programming optimisation to generate a model file with the required coefficients and "
            "obtain the ground truth for the lowest energy allocation."
        )

        _, _, target_energy = self.optimize_cube_symmetry_ase(
            group=group, verbose=False
        )

        print("Generating quadratic unconstrained binary problem from model.lp")

        bqm_model = BQM()
        bqm_model.parse_lp("model.lp")
        bqm_model.parse_constraints()
        bqm_model.max_bound()
        bqm_model.qubofy(infinity_placement, infinity_orbit)

        np.set_printoptions(suppress=True)
        print(
            "Five number summary of the interaction coefficients of the Ising hamiltonian:",
            np.percentile(
                np.array(list(bqm_model.quadratic.values())),
                [0, 25, 50, 75, 100],
                method="midpoint",
            ),
        )

        # print(bqm_model.linear, bqm_model.quadratic, bqm_model.offset)
        print("The offset is equal to", bqm_model.offset)

        # solver = neal.SimulatedAnnealingSampler()
        bqm = dimod.BinaryQuadraticModel(
            bqm_model.linear, bqm_model.quadratic, bqm_model.offset, dimod.BINARY
        )

        # bqm.fix_variable(('Sr', 7), 1)
        # bqm.fix_variable(('Ti', 0), 1)
        # print(list(bqm.variables))
        print("There are ", len(bqm.variables), "variables in the program")
        print("Running the Annealer")
        print(
            "A series of readouts with the energy, allocation to the lattice positions:"
        )

        def stoic(datum):
            # counts = {'O': 0, 'Sr': 0, 'Ti': 0}
            counts = {}
            for k, v in datum[0].items():
                # counts[k[0]] += v
                if k[0] in counts:
                    counts[k[0]] += v
                else:
                    counts[k[0]] = v
            # print('===', datum[0])
            return "Counts: " + str(counts)

        def simplify(datum):
            sample = {"energy": 0, "sample": [], "num_occurrences": 0}
            sample["energy"] = datum[1]
            sample["num_occurrences"] = int(datum[2])

            for k, v in datum[0].items():
                if v == 1:
                    sample["sample"].append(k)

            # print('<', sample, '>')
            # print('===', datum[0])
            return sample

        if at_dwave:
            embedding = dwave.embedding.chimera.find_clique_embedding(
                len(bqm.variables), 16
            )
            print(
                "The number of qubits: ",
                sum(len(chain) for chain in embedding.values()),
            )
            print(
                "The longest chain: ", max(len(chain) for chain in embedding.values())
            )
            exit()
            sampler = EmbeddingComposite(DWaveSampler())
            response = sampler.sample(
                bqm, num_reads=num_reads, annealing_time=annealing_time
            )
            min_energy = 1000000
            sol = None
            json_result = []

            i = 1
            for datum in response.data(["sample", "energy", "num_occurrences"]):
                # stoic only makes sense for P1 space group
                # print(f"Readout {i}:", simplify(datum), stoic(datum))
                print(f"Readout {i}:", simplify(datum))
                i += 1
                json_result.append(simplify(datum))

                if datum.energy < min_energy:
                    sol = datum
                    min_energy = datum.energy
            # print(type(sol))

            with open("last_dwave.json", "w") as f:
                json.dump(json_result, f, indent=2)

            print("The best found allocation:\n(atom specie, position on a lattice)")
            for i in sol.sample.keys():
                if sol.sample[i] == 1:
                    print(i)
            print(
                "The lowest found energy: ",
                sol.energy,
                "Occurrences: ",
                sol.num_occurrences,
            )
            return sol.energy, target_energy
        else:
            solver = neal.SimulatedAnnealingSampler()
            response = solver.sample(bqm, num_reads=num_reads)
            min_energy = 10000
            sample = 0

            i = 1
            for datum in response.data(["sample", "energy", "num_occurrences"]):
                # stoic only makes sense for P1 space group
                # print(f"Readout {i}:", simplify(datum), stoic(datum))
                print(f"Readout {i}:", simplify(datum))
                i += 1

                if datum.energy < min_energy:
                    sample = datum
                    min_energy = datum.energy

            # ok, sample contains the best found allocation and we want to print it
            print("The best found allocation: (atom specie, position on a lattice)")
            for i in sample.sample.keys():
                if sample.sample[i] == 1:
                    print(i)
            print(
                "The lowest found energy: ",
                sample.energy,
                "Occurrences: ",
                sample.num_occurrences,
            )
            return sample.energy, target_energy
