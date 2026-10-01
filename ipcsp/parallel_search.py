import multiprocessing
from ipcsp.integer_program import Allocate
from ipcsp.decomposition import algo1_decomposition, lpt_schedule
from ipcsp.matrix_generator import Phase


def process_worker(tasks, allocate_args, allocate_kwargs, worker_id, return_dict):
    best_energy = float("inf")
    best_res = None
    best_runtime = 0

    allocate = Allocate(*allocate_args)

    for task_size, prefix in tasks:
        res = allocate.optimize_cube_symmetry_ase(
            group=allocate_kwargs.get("group", "1"),
            PoolSolutions=allocate_kwargs.get("top", 1),
            TimeLimit=allocate_kwargs.get("TimeLimit", 0),
            verbose=allocate_kwargs.get("verbose", False),
            prefix=prefix,
            threads=1,
        )
        if res is not None:
            solutions, runtime, objVal = res
            best_runtime += runtime
            if objVal < best_energy:
                best_energy = objVal
                best_res = solutions

    return_dict[worker_id] = (best_energy, best_res, best_runtime)


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
):
    N = grid_size**3
    ion_types = list(ions_count.keys())
    total_ions = sum(ions_count.values())

    symbols = list(ion_types)
    counts = [ions_count[s] for s in symbols]

    if total_ions < N:
        symbols.append("V")
        counts.append(N - total_ions)

    print(
        f"Decomposing search space using Algorithm 1. Alphabet: {symbols}, counts: {counts}"
    )

    # In CSP formulation, the first orbit (containing site 0) must contain an ion (cannot be vacancy).
    # We branch on site 0 = ion for each ion type.
    if max_tasks is None:
        max_tasks = max(p, len(ion_types) * 2)

    all_tasks = []
    for ion in ion_types:
        if ions_count[ion] > 0:
            b_counts = list(counts)
            b_counts[symbols.index(ion)] -= 1
            branch_tasks = algo1_decomposition(
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
    print(f"Scheduled tasks across {p} processors using LPT. Machine loads: {loads}")

    allocate_args = (ions_count, grid_size, cell_size, phase)
    allocate_kwargs = {"group": str(group), "top": top, "verbose": verbose}

    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    processes = []
    for i in range(p):
        proc = multiprocessing.Process(
            target=process_worker,
            args=(processors[i], allocate_args, allocate_kwargs, i, return_dict),
        )
        processes.append(proc)
        proc.start()

    for proc in processes:
        proc.join()

    overall_best_energy = float("inf")
    overall_best_res = []
    total_runtime = 0

    for i in range(p):
        if i in return_dict:
            best_energy, best_res, best_runtime = return_dict[i]
            total_runtime += best_runtime
            print(
                f"Processor {i} finished. Best IP energy: {best_energy:.4f}, Runtime: {best_runtime:.2f}s"
            )
            if best_energy < overall_best_energy:
                overall_best_energy = best_energy
                overall_best_res = best_res

    print(
        f"\nOverall minimum IP energy across all processors: {overall_best_energy:.4f}"
    )
    return overall_best_energy, overall_best_res, total_runtime
