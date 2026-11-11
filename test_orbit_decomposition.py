import json
import os
import unittest
from ipcsp import root_dir
from ipcsp.decomposition import (
    build_orbit_dp,
    decompose_orbits_balanced,
    lpt_schedule,
)
from ipcsp.integer_program import Allocate
from ipcsp.matrix_generator import Phase, get_Ewald


def load_orb_sizes(grid, group):
    p = root_dir / f"data/grids/CO{grid}G{group}.json"
    with open(p, "r") as f:
        orbits = json.load(f)
    orb_key = list(orbits.keys())
    return [len(orbits[k]) + 1 for k in orb_key]


class TestOrbitDecomposition(unittest.TestCase):
    def test_known_exact_counts(self):
        """Verify exact total valid orbit completions matching analytical Table 1 values."""
        cases = [
            ("SrTiO3_1", 4, 1, {"O": 3, "Sr": 1, "Ti": 1}, 11913300),
            ("Y2O3_1", 8, 206, {"O": 48, "Y": 32}, 82),
            ("Y2Ti2O7_1", 8, 227, {"O": 56, "Y": 16, "Ti": 16}, 6),
            ("MgAl2O4_1", 8, 227, {"O": 32, "Mg": 8, "Al": 16}, 2),
            ("Ca3Al2Si3O12_2", 8, 206, {"Ca": 24, "Al": 16, "Si": 24, "O": 96}, 432),
            ("SrTiO3_2", 8, 195, {"O": 24, "Sr": 8, "Ti": 8}, 2996040),
            ("SrTiO3_4", 6, 200, {"O": 81, "Sr": 27, "Ti": 27}, 503695360),
            ("SrTiO3_5", 6, 195, {"O": 81, "Sr": 27, "Ti": 27}, 16952664960),
            ("MgAl2O4_4", 8, 195, {"O": 32, "Mg": 8, "Al": 16}, 794586540),
        ]
        for name, grid, grp, ions, expected in cases:
            with self.subTest(case=name):
                orbs = load_orb_sizes(grid, grp)
                types, count_fn = build_orbit_dp(orbs, ions)
                init_rem = tuple(ions[t] for t in types)
                total = count_fn(0, init_rem)
                self.assertEqual(total, expected)

    def test_orbit_dp_edge_cases(self):
        """Test boundary conditions for build_orbit_dp."""
        # 1. Zero ions remaining when starting with empty counts
        types, count_fn = build_orbit_dp([1, 1], {"A": 0})
        # Empty counts: orbit 0 cannot be vacancy, but A count is 0 => 0 completions
        self.assertEqual(count_fn(0, (0,)), 0)

        # 2. Insufficient total capacity
        orbs = [2, 2]  # total capacity = 4
        ions = {"A": 6}
        types, count_fn = build_orbit_dp(orbs, ions)
        self.assertEqual(count_fn(0, (6,)), 0)

        # 3. Wyckoff multiplicity exceeding ion count
        orbs = [8, 48]
        ions = {"A": 8, "B": 10}  # B count is 10, cannot fit on size 48 orbit
        types, count_fn = build_orbit_dp(orbs, ions)
        # B on orbit 1 is impossible (48 > 10)
        self.assertEqual(count_fn(0, (8, 10)), 0)

    def test_decompose_balanced_conservation(self):
        """Ensure sum of generated task sizes equals the exact configuration total for various p."""
        orbs = load_orb_sizes(8, 206)
        ions = {"Ca": 24, "Al": 16, "Si": 24, "O": 96}
        for p in [1, 2, 4, 8, 16, 32, 64]:
            tasks, total = decompose_orbits_balanced(orbs, ions, p)
            self.assertEqual(total, 432)
            self.assertGreaterEqual(len(tasks), 1)
            sum_cnt = sum(cnt for _, cnt in tasks)
            self.assertEqual(sum_cnt, total)

    def test_decompose_p1_returns_single_task(self):
        """When p=1, decompose_orbits_balanced returns a single unconstrained task."""
        orbs = load_orb_sizes(4, 1)
        ions = {"O": 3, "Sr": 1, "Ti": 1}
        tasks, total = decompose_orbits_balanced(orbs, ions, 1)
        self.assertEqual(len(tasks), 1)
        pref, cnt = tasks[0]
        self.assertEqual(pref, ())
        self.assertEqual(cnt, total)

    def test_decompose_fewer_configs_than_processors(self):
        """When total configurations < p (e.g. MgAl2O4_1 has total=2), handles gracefully."""
        orbs = load_orb_sizes(8, 227)
        ions = {"O": 32, "Mg": 8, "Al": 16}
        tasks, total = decompose_orbits_balanced(orbs, ions, 16)
        self.assertEqual(total, 2)
        self.assertEqual(len(tasks), 2)
        lpt_tasks = [(cnt, prefix) for prefix, cnt in tasks]
        procs, loads = lpt_schedule(lpt_tasks, 16)
        self.assertEqual(len(procs), 16)
        self.assertEqual(sum(loads), 2)
        # First 2 processors get 1 task each, others get 0
        self.assertEqual(loads[:2], [1, 1])
        self.assertEqual(loads[2:], [0] * 14)

    def test_bound_fixing_model_reuse(self):
        """Verifies that model reuse via in-place bound fixing produces correct solutions and resets cleanly."""
        phase = Phase("SrTiO")
        alloc = Allocate({"O": 3, "Sr": 1, "Ti": 1}, 4, 3.9, phase)
        # Pre-build base model
        m, Vars, types, orb_key = alloc.build_base_model(group="1", write_lp=False)
        self.assertIsNotNone(m)
        self.assertEqual(len(Vars), 3)  # O, Sr, Ti
        self.assertEqual(len(orb_key), 64)

        # Task 1: prefix with Orbit 0 = Ti
        res1 = alloc.optimize_cube_symmetry_ase(
            group="1",
            PoolSolutions=1,
            TimeLimit=5,
            verbose=False,
            prefix=("Ti",),
            threads=1,
            write_lp=False,
            reuse_model=True,
        )
        self.assertIsNotNone(res1)
        energy1 = res1[2]
        self.assertAlmostEqual(energy1, -158.7578, places=3)
        self.assertLess(alloc.last_stats["build_time"], 0.05)  # Microsecond reuse

        # Task 2: prefix with Orbit 0 = O
        res2 = alloc.optimize_cube_symmetry_ase(
            group="1",
            PoolSolutions=1,
            TimeLimit=5,
            verbose=False,
            prefix=("O",),
            threads=1,
            write_lp=False,
            reuse_model=True,
        )
        self.assertIsNotNone(res2)
        energy2 = res2[2]
        self.assertAlmostEqual(energy2, -158.7578, places=3)
        self.assertLess(alloc.last_stats["build_time"], 0.05)

    def test_load_balance_across_all_p(self):
        """Verify load imbalance max_load / avg_load <= 1.05 for SrTiO3_1 across all p."""
        orbs = load_orb_sizes(4, 1)
        ions = {"O": 3, "Sr": 1, "Ti": 1}
        for p in [1, 2, 4, 8, 16, 32, 64]:
            tasks, total = decompose_orbits_balanced(orbs, ions, p)
            procs, loads = lpt_schedule([(cnt, pref) for pref, cnt in tasks], p)
            max_load = max(loads)
            avg_load = total / p
            ratio = max_load / avg_load
            self.assertLessEqual(
                ratio, 1.05, f"Load imbalance {ratio:.3f} exceeded 1.05 at p={p}"
            )

    def test_garnet_matrix_pregeneration(self):
        """Verify that Garnet phase matrix pre-generation properly detects gar_param and loads matrices."""
        phase = Phase("CaAlSiO")
        self.assertTrue(getattr(phase, "garnet", False))
        self.assertTrue(hasattr(phase, "gar_param"))
        types = ["Ca", "Al", "Si", "O"]
        pregen_count = 0
        if getattr(phase, "garnet", False):
            if hasattr(phase, "gar_param"):
                for ion_pair in phase.gar_param:
                    if ion_pair[0] in types and ion_pair[1] in types:
                        m = get_Ewald(8, 11.9)
                        self.assertIsNotNone(m)
                        pregen_count += 1
        self.assertEqual(pregen_count, 4)

    def test_str_output_directory_generate_matrices(self):
        """Verify generate_Ewald and matrix generators accept string output_directory without TypeError."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            from ipcsp.matrix_generator import generate_Ewald

            # Pass str rather than Path
            generate_Ewald(2, str(tmpdir))
            npy_file = os.path.join(tmpdir, "C2.npy")
            self.assertTrue(os.path.exists(npy_file))


if __name__ == "__main__":
    unittest.main()
