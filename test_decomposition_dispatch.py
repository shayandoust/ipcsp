import unittest
from ipcsp.decomposition import (
    is_balanced,
    is_balanced_parikh_vector,
    decompose,
    algo1_decomposition,
    algo2_decomposition,
    multinomial,
    lpt_schedule,
)


class TestDecompositionDispatch(unittest.TestCase):
    def test_is_balanced(self):
        # Balanced Parikh vectors in the form of (n, ..., n) with n > 0
        self.assertTrue(is_balanced([4, 4]))
        self.assertTrue(is_balanced([32, 32, 32, 32, 32, 32]))
        self.assertTrue(is_balanced((3, 3, 3)))
        self.assertTrue(is_balanced([1]))
        self.assertTrue(is_balanced({"A": 10, "B": 10, "C": 10}))
        self.assertTrue(is_balanced_parikh_vector([5, 5, 5]))

        # Unbalanced Parikh vectors
        self.assertFalse(is_balanced([3, 1, 1, 59]))
        self.assertFalse(is_balanced([256, 128, 64, 32, 16]))
        self.assertFalse(is_balanced([4, 4, 0]))
        self.assertFalse(is_balanced([0, 0, 0]))
        self.assertFalse(is_balanced([]))
        self.assertFalse(is_balanced({"A": 2, "B": 1}))

    def test_decompose_balanced_routes_to_algo2(self):
        symbols = ["A", "B"]
        counts = [4, 4]
        p = 2

        # Should use Algorithm 2
        tasks = decompose(symbols, counts, p=p, k=1)
        total_words = sum(sz for sz, _ in tasks)
        self.assertEqual(total_words, multinomial(counts))
        self.assertEqual(len(tasks), 2)  # 2^1 = 2 tasks

    def test_decompose_balanced_larger(self):
        symbols = ["A", "B", "C"]
        counts = [2, 2, 2]
        p = 3

        tasks = decompose(symbols, counts, p=p, k=1)
        total_words = sum(sz for sz, _ in tasks)
        self.assertEqual(total_words, multinomial(counts))
        self.assertEqual(len(tasks), 3)
        sizes = [sz for sz, _ in tasks]
        self.assertEqual(max(sizes), min(sizes))  # completely uniform for symmetric tree depth 1

    def test_decompose_unbalanced_routes_to_algo1(self):
        symbols = ["O", "Sr", "Ti", "V"]
        counts = [3, 1, 1, 59]
        p = 4

        # Unbalanced: should use Algorithm 1
        tasks = decompose(symbols, counts, p=p, R_target=1.1, max_tasks=20)
        self.assertGreater(len(tasks), 1)
        total_words = sum(sz for sz, _ in tasks)
        self.assertEqual(total_words, multinomial(counts))

    def test_lpt_schedule_on_decomposed_tasks(self):
        symbols = ["A", "B"]
        counts = [5, 5]
        p = 4

        tasks = decompose(symbols, counts, p=p)
        procs, loads = lpt_schedule(tasks, p)
        self.assertEqual(len(procs), p)
        self.assertEqual(sum(loads), multinomial(counts))

    def test_parallel_search_dispatch(self):
        from ipcsp.parallel_search import is_balanced
        # Unbalanced
        unbalanced_ions = {"O": 3, "Sr": 1, "Ti": 1}
        N_unbalanced = 4**3
        unbalanced_counts = list(unbalanced_ions.values()) + [N_unbalanced - sum(unbalanced_ions.values())]
        self.assertFalse(is_balanced(unbalanced_counts))

        # Balanced (no vacancies)
        balanced_ions = {"A": 4, "B": 4}
        balanced_counts = list(balanced_ions.values())
        self.assertTrue(is_balanced(balanced_counts))


if __name__ == "__main__":
    unittest.main()

