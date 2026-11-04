import gurobipy as gb
import numpy as np
import os
import time as _time

from ipcsp import root_dir
from ipcsp.matrix_generator import Phase, get_Ewald, get_Buck, get_garnet
import ase
from ipcsp.grids_and_symmetry import cubic
import json
from ipcsp.lp_to_bqm import BQM

griddir = root_dir / "data/grids/"


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

        grid_positions = cubic(self.grid)
        # ions = list(self.ions.keys())

        # Atoms
        symbols = ""
        positions = []

        for v in self.model.getVars():
            if v.Xn == 1:
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

        N = self.grid**3  # number
        T = len(self.ions)  # different types

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

        if verbose:
            print("Generating integer program\n")

        orb_size = [len(orbits[k]) + 1 for k in orb_key]

        # o_pos[position] = number_of_the_orbit
        o_pos = []
        for i in range(N):
            for orb, pos in orbits.items():
                if int(orb) == i or i in pos:
                    o_pos.append(orb_key.index(orb))
                    break

        O = len(orbits)
        types = list(self.ions.keys())  # ordered list of elements
        counts = [self.ions[t] for t in types]

        m = gb.Model(
            "Ion allocation in {name} with symmetry {group}".format(
                name=self.phase.name, group=group
            )
        )
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
            else:
                m.addConstr(gb.LinExpr([1.0] * T, tmp_var) <= 1, f"one_per_orbit_{i}")

        for j in range(T):
            tmp = gb.LinExpr()
            for i in range(O):
                tmp.add(Vars[j][i], orb_size[i])
            m.addConstr(tmp == counts[j], f"number_of_ions_type_{j}")

        if verbose:
            print("Variables and constraints were generated")
        energy = gb.QuadExpr()

        # Coulomb interaction
        dist = get_Ewald(self.grid, self.cell)

        for i1 in range(N):
            for j1 in range(T):  # self-interaction
                energy.add(
                    Vars[j1][o_pos[i1]]
                    * Vars[j1][o_pos[i1]]
                    * dist[i1, i1]
                    * self.phase.charge[types[j1]] ** 2
                )

            for i2 in range(i1 + 1, N):
                for j1 in range(T):  # pairwise Coulumb
                    energy.add(
                        Vars[j1][o_pos[i1]]
                        * Vars[j1][o_pos[i2]]
                        * 2
                        * dist[i1, i2]
                        * self.phase.charge[types[j1]] ** 2
                    )

                    for j2 in range(j1 + 1, T):
                        energy.add(
                            Vars[j1][o_pos[i1]]
                            * Vars[j2][o_pos[i2]]
                            * 2
                            * dist[i1, i2]
                            * self.phase.charge[types[j1]]
                            * self.phase.charge[types[j2]]
                        )
                        energy.add(
                            Vars[j2][o_pos[i1]]
                            * Vars[j1][o_pos[i2]]
                            * 2
                            * dist[i1, i2]
                            * self.phase.charge[types[j1]]
                            * self.phase.charge[types[j2]]
                        )
        del dist
        if verbose:
            print("Ewald sum contribution was added to the objective function")

        # HACK for the garnet
        if self.phase.garnet:
            if verbose:
                print("Generating Morse and Leonard parts")
            for ion_pair in self.phase.gar_param:
                if not ((ion_pair[0] in types) and (ion_pair[1] in types)):
                    continue
                buck = get_garnet(ion_pair, self.grid, self.cell, self.phase)
                if ion_pair[0] == ion_pair[1]:
                    j1 = types.index(ion_pair[0])
                    for i1 in range(N):
                        for i2 in range(i1, N):
                            energy.add(
                                Vars[j1][o_pos[i1]] * Vars[j1][o_pos[i2]] * buck[i1, i2]
                            )
                else:
                    j1 = types.index(ion_pair[0])
                    j2 = types.index(ion_pair[1])
                    for i1 in range(N):
                        for i2 in range(i1 + 1, N):
                            energy.add(
                                Vars[j1][o_pos[i1]] * Vars[j2][o_pos[i2]] * buck[i1, i2]
                            )
                            energy.add(
                                Vars[j2][o_pos[i1]] * Vars[j1][o_pos[i2]] * buck[i1, i2]
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
                    for i1 in range(N):
                        for i2 in range(i1, N):
                            energy.add(
                                Vars[j1][o_pos[i1]] * Vars[j1][o_pos[i2]] * buck[i1, i2]
                            )
                else:
                    j1 = types.index(ion_pair[0])
                    j2 = types.index(ion_pair[1])
                    for i1 in range(N):
                        for i2 in range(i1 + 1, N):
                            energy.add(
                                Vars[j1][o_pos[i1]] * Vars[j2][o_pos[i2]] * buck[i1, i2]
                            )
                            energy.add(
                                Vars[j2][o_pos[i1]] * Vars[j1][o_pos[i2]] * buck[i1, i2]
                            )
                del buck

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
    ):
        """
        The function to generate an integer program and solve allocation problem using Gurobi.
        We rely on atomic simulation environment to handle allocations afterwards.

        write_lp -- write the model to model.lp (needed by the quantum/QUBO pipeline).
          Default: False to avoid concurrent disk writes during parallel benchmarks.
        reuse_model -- reuse previously built base model on self and solve via LB/UB bound mutation.
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

        def _internal_cb(model, where):
            if where == gb.GRB.Callback.MIPSOL:
                try:
                    obj = model.cbGet(gb.GRB.Callback.MIPSOL_OBJ)
                    rt = model.cbGet(gb.GRB.Callback.RUNTIME)
                    nodes = model.cbGet(gb.GRB.Callback.MIPSOL_NODCNT)
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

        m.optimize(_internal_cb)
        runtime = m.Runtime
        node_count = getattr(m, "NodeCount", 0.0)
        work_units = getattr(m, "Work", 0.0)
        iter_count = getattr(m, "IterCount", 0.0)

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
                wall_to_opt = _time.perf_counter() - t0_solve
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
