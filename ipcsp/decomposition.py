import math
from collections import defaultdict
import numpy as np


def multinomial(counts):
    counts = [c for c in counts if c > 0]
    if not counts:
        return 1
    total = sum(counts)
    res = math.factorial(total)
    for c in counts:
        res //= math.factorial(c)
    return res


def lpt_schedule(tasks, p):
    # tasks is a list of (task_size, prefix)
    # returns a list of p lists, each containing the prefixes for that processor
    tasks.sort(key=lambda x: x[0], reverse=True)
    processors = [[] for _ in range(p)]
    processor_loads = [0] * p

    for size, prefix in tasks:
        # Find the processor with minimum load
        min_p = min(range(p), key=lambda i: processor_loads[i])
        processors[min_p].append((size, prefix))
        processor_loads[min_p] += size

    return processors, processor_loads


def algo1_decomposition(symbols, initial_counts, p, R_target, max_tasks=None):
    """
    symbols: list of symbol names, e.g. ['O', 'Sr', 'Ti', 'V']
    initial_counts: list of initial counts corresponding to symbols
    p: number of processors
    R_target: target approximation ratio (e.g. 1.1)
    max_tasks: optional maximum number of tasks to generate
    """

    # Psi represents the remaining counts as a tuple
    Psi_init = tuple(initial_counts)
    V = multinomial(Psi_init)

    # D[Psi] = list of prefixes
    D = {}
    D[Psi_init] = [[]]  # initial prefix is empty list

    # M_dict caches the size of each Psi
    M_dict = {}
    M_dict[Psi_init] = V

    M = V
    R_current = 1 + (p - 1) * M / V

    while R_current > R_target:
        if max_tasks is not None and sum(len(pref) for pref in D.values()) >= max_tasks:
            break
        # Get the suffix Parikh vector with the largest task size
        # We find the Psi in D with the maximum M_dict[Psi]
        Psi_current = max(D.keys(), key=lambda psi: M_dict[psi])

        M = M_dict[Psi_current]
        prefixes = D.pop(Psi_current)

        if M == 1:
            break

        for i, a in enumerate(symbols):
            if Psi_current[i] > 0:
                # Create the new Parikh vector
                Psi_new = list(Psi_current)
                Psi_new[i] -= 1
                Psi_new = tuple(Psi_new)

                # The new prefixes have 'a' appended
                new_prefixes_for_a = [prefix + [a] for prefix in prefixes]

                if Psi_new in D:
                    D[Psi_new].extend(new_prefixes_for_a)
                else:
                    D[Psi_new] = new_prefixes_for_a
                    M_dict[Psi_new] = multinomial(Psi_new)

        # Update max task size
        if D:
            M = max(M_dict[psi] for psi in D.keys())
        else:
            M = 0

        R_current = 1 + (p - 1) * M / V

    # Reconstruct the multiset of tasks
    tasks = []
    for psi, prefixes in D.items():
        size = M_dict[psi]
        for prefix in prefixes:
            tasks.append((size, prefix))

    return tasks


def algo2_decomposition(symbols, initial_counts, p, k=1, c_factor=2.0):
    """
    Algorithm 2: Bounded-Spread Decomposition Strategy.
    Generates p^k tasks.
    """
    Psi_init = tuple(initial_counts)
    memo_decompose = {}
    memo_split = {}

    def Decompose(Psi_s, n):
        state = (Psi_s, n)
        if state in memo_decompose:
            return memo_decompose[state]

        if n == 1 or sum(Psi_s) == 0:
            res = [(multinomial(Psi_s), [])]
            memo_decompose[state] = res
            return res

        # Children
        C = []
        for i, a in enumerate(symbols):
            if Psi_s[i] > 0:
                Psi_new = list(Psi_s)
                Psi_new[i] -= 1
                C.append((i, tuple(Psi_new)))

        if n < len(C):
            res = [(multinomial(Psi_s), [])]
            memo_decompose[state] = res
            return res

        def Split(i, t):
            split_state = (Psi_s, n, i, t)
            if split_state in memo_split:
                return memo_split[split_state]

            best = None
            if i == len(C) and t == 0:
                return []
            if i == len(C) and t > 0:
                return None
            if t == 0:
                return None

            for j in range(1, t - (len(C) - i - 1) + 1):
                L1 = Decompose(C[i][1], j)
                L2 = Split(i + 1, t - j)

                if L1 is not None and L2 is not None:
                    sym = symbols[C[i][0]]
                    L1_updated = []
                    for size, prefix in L1:
                        # Append the current symbol to the beginning of the prefix
                        L1_updated.append((size, [sym] + prefix))

                    candidate = L1_updated + L2

                    sizes = [s for s, _ in candidate]
                    if sizes:
                        if min(sizes) > 0 and max(sizes) / min(sizes) <= c_factor:
                            if best is None or max(sizes) < max([s for s, _ in best]):
                                best = candidate

            memo_split[split_state] = best
            return best

        res = Split(0, n)
        if res is None:
            # Fallback
            res = [(multinomial(Psi_s), [])]
        memo_decompose[state] = res
        return res

    result = Decompose(Psi_init, p**k)
    return result



def is_balanced(counts):
    """
    Check if a Parikh vector is balanced, i.e., in the form of (n, ..., n) with n > 0.
    counts can be a list, tuple, dict of counts, or numpy array.
    """
    if isinstance(counts, dict):
        counts = list(counts.values())
    if not counts:
        return False
    first = counts[0]
    return first > 0 and all(c == first for c in counts)


is_balanced_parikh_vector = is_balanced


def decompose(
    symbols,
    initial_counts,
    p,
    R_target=1.5,
    max_tasks=None,
    k=None,
    c_factor=2.0,
):
    """
    Decompose a Parikh world:
    Uses Algorithm 2 for balanced Parikh vectors (i.e., in the form of (n, ..., n)),
    and Algorithm 1 for all other cases.
    """
    if is_balanced(initial_counts):
        if k is None:
            non_zero = sum(1 for c in initial_counts if c > 0)
            if max_tasks is not None and p > 1 and max_tasks >= p:
                k = max(1, int(math.floor(math.log(max_tasks) / math.log(p))))
            elif p > 1 and non_zero > p:
                k = max(1, int(math.ceil(math.log(non_zero) / math.log(p))))
            else:
                k = 1
        tasks = algo2_decomposition(symbols, initial_counts, p, k=k, c_factor=c_factor)
        # If Algo 2 returned fallback without splitting (1 task) and more tasks were requested/possible:
        if len(tasks) <= 1 and multinomial(initial_counts) > 1 and p > 1:
            tasks = algo1_decomposition(
                symbols,
                initial_counts,
                p,
                R_target=R_target,
                max_tasks=max_tasks,
            )
        return tasks
    else:
        return algo1_decomposition(
            symbols,
            initial_counts,
            p,
            R_target=R_target,
            max_tasks=max_tasks,
        )


if __name__ == "__main__":
    # Unbalanced Parikh vector: uses Algorithm 1
    unbalanced_counts = [3, 1, 1, 59]
    print(f"Counts {unbalanced_counts} balanced? {is_balanced(unbalanced_counts)}")
    tasks1 = decompose(["O", "Sr", "Ti", "V"], unbalanced_counts, 4, R_target=1.1)
    print("Number of tasks (Algorithm 1):", len(tasks1))
    processors1, loads1 = lpt_schedule(tasks1, 4)
    print("Loads:", loads1)

    # Balanced Parikh vector: uses Algorithm 2
    balanced_counts = [4, 4]
    print(f"Counts {balanced_counts} balanced? {is_balanced(balanced_counts)}")
    tasks2 = decompose(["A", "B"], balanced_counts, 4)
    print("Number of tasks (Algorithm 2):", len(tasks2))
    processors2, loads2 = lpt_schedule(tasks2, 4)
    print("Loads:", loads2)

