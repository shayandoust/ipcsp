import os
import shutil
import pandas as pd
from tabulate import tabulate

from ipcsp.parallel_search import run_parallel_search
from ipcsp.matrix_generator import Phase
from ipcsp.__main__ import process_results, get_cif_energies, RESULTS_DIR, TYPICAL_DIR


def parse_typical_energies(test_name):
    """Parse initial and final energies from typical_results/<test_name>/energies.txt."""
    path = os.path.join(TYPICAL_DIR, test_name, "energies.txt")
    if not os.path.exists(path):
        return None, None
    init_e, final_e = None, None
    with open(path, "r") as f:
        for line in f:
            if "Energy initial:" in line and "final:" in line:
                parts = line.strip().split()
                try:
                    init_idx = parts.index("initial:") + 1
                    final_idx = parts.index("final:") + 1
                    init_e = float(parts[init_idx])
                    final_e = float(parts[final_idx])
                    break
                except (ValueError, IndexError):
                    pass
    return init_e, final_e


def test_parallel_benchmarks():
    """
    Evaluates benchmark structures with full-scale unit cells matching typical_results,
    running independent Gurobi IP instances on p processors via Algorithm 1 decomposition
    and LPT scheduling, and compares the energies found to typical_results.
    """
    print("\n" + "=" * 90)
    print(
        "  RUNNING PARALLEL GUROBI IP EVALUATION ON BENCHMARK STRUCTURES (MATCHING TYPICAL_RESULTS)"
    )
    print("=" * 90 + "\n")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Benchmark test cases matching Table 1 and typical_results
    test_cases = [
        {
            "name": "SrTiO3_1",
            "test_folder": "SrTiO_1",
            "formula": "SrTiO3",
            "ions_count": {"O": 3, "Sr": 1, "Ti": 1},
            "grid": 4,
            "group": "1",
            "cell_size": 3.9,
            "phase_name": "SrTiO",
            "lib": "SrTiO/buck.lib",
            "cif": "SrTiO3.cif",
            "top": 1,
        },
        {
            "name": "Y2O3_1",
            "test_folder": "Y2O3_1",
            "formula": "Y2O3",
            "ions_count": {"O": 48, "Y": 32},
            "grid": 8,
            "group": "206",
            "cell_size": 10.7,
            "phase_name": "YSrTiO",
            "lib": "YSrTiO/buck.lib",
            "cif": "Y2O3.cif",
            "top": 1,
        },
        {
            "name": "Y2Ti2O7_1",
            "test_folder": "Y2Ti2O7_1",
            "formula": "Y2Ti2O7",
            "ions_count": {"O": 56, "Y": 16, "Ti": 16},
            "grid": 8,
            "group": "227",
            "cell_size": 10.2,
            "phase_name": "YSrTiO",
            "lib": "YSrTiO/buck.lib",
            "cif": "Y2Ti2O7.cif",
            "top": 2,
        },
        {
            "name": "MgAl2O4_1",
            "test_folder": "MgAl2O4_1",
            "formula": "MgAl2O4",
            "ions_count": {"O": 32, "Mg": 8, "Al": 16},
            "grid": 8,
            "group": "227",
            "cell_size": 8.2,
            "phase_name": "LiMgAlPO",
            "lib": "LiMgAlPO/buck.lib",
            "cif": "MgAl2O4.cif",
            "top": 1,
        },
        {
            "name": "Ca3Al2Si3O12_2",
            "test_folder": "Ca3Al2Si3O12_2",
            "formula": "Ca3Al2Si3O12",
            "ions_count": {"Ca": 24, "Al": 16, "Si": 24, "O": 96},
            "grid": 8,
            "group": "206",
            "cell_size": 11.9,
            "phase_name": "CaAlSiO",
            "lib": "CaAlSiO/pedone.lib",
            "cif": "Ca3Al2Si3O12.cif",
            "top": 1,
        },
    ]

    p = 4
    comparison_data = []

    for tc in test_cases:
        print(f"\n{'='*30} Evaluating {tc['name']} ({tc['formula']}) {'='*30}")
        print(
            f"Parameters: grid={tc['grid']}, group={tc['group']}, cell={tc['cell_size']} A, ions={tc['ions_count']}"
        )

        phase = Phase(tc["phase_name"])

        best_ip, results, runtime = run_parallel_search(
            ions_count=tc["ions_count"],
            grid_size=tc["grid"],
            cell_size=tc["cell_size"],
            phase=phase,
            p=p,
            R_target=1.5,
            group=tc["group"],
            top=tc["top"],
            verbose=False,
        )

        # Process with GULP relaxation and save to results/
        best_relaxed = process_results(
            lib=phase.filedir / tc["lib"],
            results=results,
            ions_count=tc["ions_count"],
            test_name=tc["test_folder"],
        )

        cif_energy = get_cif_energies(
            filename=tc["cif"], library=phase.filedir / tc["lib"]
        )

        typ_init, typ_final = parse_typical_energies(tc["test_folder"])

        diff = (
            (best_relaxed - typ_final)
            if (typ_final is not None and best_relaxed is not None)
            else None
        )
        diff_str = f"{diff:+.4f}" if diff is not None else "N/A"
        status = (
            "EXACT MATCH"
            if (diff is not None and abs(diff) < 0.01)
            else ("CLOSE" if diff is not None and abs(diff) < 1.0 else "DIFFER")
        )

        comparison_data.append(
            {
                "Test Name": tc["name"],
                "Formula": tc["formula"],
                "Grid g": tc["grid"],
                "Group": tc["group"],
                "Parallel IP Energy (eV)": f"{best_ip:.4f}",
                "Found Energy (eV)": f"{best_relaxed:.4f}",
                "Typical Energy (eV)": (
                    f"{typ_final:.4f}" if typ_final is not None else "N/A"
                ),
                "Diff (eV)": diff_str,
                "Status": status,
                "Parallel Time (s)": f"{runtime:.2f}",
            }
        )

    # Output comparative results table
    comp_df = pd.DataFrame(comparison_data)
    print("\n\n" + "=" * 105)
    print("                      COMPARISON: PARALLEL SEARCH vs TYPICAL RESULTS")
    print("=" * 105)
    print(tabulate(comp_df, headers="keys", tablefmt="github", showindex=False))
    print("=" * 105 + "\n")

    # Save to results/parallel_comparison.txt
    comp_file = os.path.join(RESULTS_DIR, "parallel_comparison.txt")
    with open(comp_file, "w+") as f:
        print("Comparison: Parallel Search vs Typical Results\n", file=f)
        print(
            tabulate(comp_df, headers="keys", tablefmt="github", showindex=False),
            file=f,
        )
    print(f"Comparison report saved to: {comp_file}")


if __name__ == "__main__":
    test_parallel_benchmarks()
