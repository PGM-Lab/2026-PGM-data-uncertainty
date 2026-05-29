from typing import Set, Tuple, List, Generator
import numpy as np
from itertools import combinations, permutations, product, combinations_with_replacement, chain
from more_itertools import distinct_permutations, distinct_combinations, roundrobin


def value_to_representation(exo_values, n_child_states, n_parent_states):
    if isinstance(exo_values, (int, np.integer)):
        digits = []
        for _ in range(n_parent_states):
            digits.append(exo_values % n_child_states)
            exo_values = exo_values // n_child_states

        digits.reverse()
        return digits
    elif isinstance(exo_values, (list, tuple)) and len(exo_values) > 0 and \
            isinstance(exo_values[0], (int, np.integer)):
        return [value_to_representation(i, n_child_states, n_parent_states) for i in exo_values]
    else:
        raise TypeError

def test_solution_satisfiability(cnf_solution, n_child_states, n_parent_states):

    if isinstance(cnf_solution[0], (int, np.integer)):
        cnf_solution = value_to_representation(cnf_solution, n_child_states, n_parent_states)

    for d in range(n_parent_states):
        digit_subset = set()
        for e in cnf_solution:
            digit_subset = digit_subset.union({e[d]})
            if len(digit_subset) >= n_child_states:
                break
        if len(digit_subset) < n_child_states:
            return False
    return True


def index_cps(digit_list):
    idx = dict()
    for i in range(len(digit_list)):
        if digit_list[i] not in idx.keys():
            idx[digit_list[i]] = []
        idx[digit_list[i]].append(i)

    idx_cps = [tuple(idx[i]) for i in idx.keys() if len(idx[i]) > 1]
    return idx_cps


def common_cps(list_1, list_2):
    common_cps = []
    for tw_1 in list_1:
        for tw_2 in list_2:
            commons = tuple(set(tw_1).intersection(tw_2))

            if len(commons) > 1:
                common_cps.append(commons)
    return common_cps


def twice_present(comb, row):
    for c in comb:
        n = 0
        for e in row:
            if e == c:
                n+=1
            if n == 2:
                break
        if n < 2:
            return False
    return True


def probabilities_index_sort(probabilities):
    ## output indexes sorted from low to high probability
    indexes = []
    for i in range(len(probabilities)):
        indexes.append((probabilities[i], i))

    indexes = sorted(indexes)

    return [i[1] for i in indexes]


def get_unique_splits(n_cols: int, n_rows: int, base: int, first=True) -> Set[Tuple[int]]:

    if n_cols == base and first:
        return {tuple([base]*n_rows)}

    min_uniques = n_cols // (base-1)

    if n_cols == n_rows * min_uniques:

        mins = tuple([min_uniques]*n_rows)
        maxs = tuple([base-1]*int(n_cols//base))
        print(n_cols/base)

        return {tuple([min_uniques]*n_rows)}

    else:
        possibilities = []
        for s in get_unique_splits(n_cols-1, n_rows, base, False):
            for i in range(len(s)):
                e = list(s).copy()
                e[i] +=1
                if e[i] <= base-1:
                    possibilities.append(tuple(e))

        return set(possibilities)

#print(get_unique_splits(8,3,4))


def matrix_to_set(matrix):

    sol = []
    for col in range(matrix.shape[-1]):
        exo_value = 0
        for p in range(matrix.shape[0]):
            exo_value += matrix[:,col][-p - 1] * ((np.max(matrix)+1) ** p)
        sol.append(exo_value)
    return sol



def test_irreducible(cnf_solution, base, n_digits):

    repr = []

    for n in cnf_solution:
        digits = []
        exo_values = n
        for _ in range(n_digits):
            digits.append(exo_values % base)
            exo_values = exo_values // base

        digits.reverse()
        repr.append(digits)

    def _test_sat(c_sol):
        for d in range(n_digits):
            digit_subset = set()
            for e in c_sol:
                digit_subset = digit_subset.union({e[d]})
                if len(digit_subset) >= base:
                    break
            if len(digit_subset) < base:
                return False
        return True

    if not _test_sat(repr):
        return False
    for it in repr:
        new_sol = [j for j in repr if j != it]
        if _test_sat(new_sol):
            return False
    return True



def all_irreducibles(size, child_states, parent_states):

    n_us = child_states**parent_states

    gen = combinations([i for i in range(n_us)], size)

    for g in gen:
        if test_irreducible(g, child_states, parent_states):
            yield g


def count_irreducibles(size, child_states, parent_states):
    c = 0
    for _ in all_irreducibles(size, child_states, parent_states):
        c += 1
    return c



def distinct_combinations_(idx_set, size, idx_tw):

    idx_set = idx_set.copy()

    tws_used = []
    for it in idx_tw:
        if len(set(it).intersection(idx_set)) == len(it):
            for et in it:
                idx_set.remove(et)
            idx_set += [it[0]] * len(it)
            tws_used.append(it)

    generator = distinct_combinations(idx_set, size)

    for e in generator:
        e = list(e)

        for tw in tws_used:

            elt = tw[0]
            c = 0
            for i in range(len(e)):
                if e[i] == elt:
                    e[i] = tw[c]
                    c += 1

        yield e



def distinct_permutations_(idx_set, idx_tw):
    tws_used =[]
    for it in idx_tw:
        its_in_idx_set = list(set(it).intersection(idx_set))
        if len(its_in_idx_set) > 1:
            for i in range(len(its_in_idx_set)):
                idx_set.remove(its_in_idx_set[i])
            idx_set += [its_in_idx_set[0]] * len(its_in_idx_set)
            tws_used.append(its_in_idx_set)

    generator = distinct_permutations(idx_set, len(idx_set))

    for e in generator:

        e = list(e)

        for tw in tws_used:

            elt = tw[0]
            c=0
            for i in range(len(e)):
                if e[i] == elt:
                    e[i] = tw[c]
                    c+=1

        yield e



def prod_twin_(comb, repeat, idx_set, idx_tw):


    if len(idx_tw) == 0:
        non_twin_gen = product(comb, repeat=repeat)
        for g in non_twin_gen:

            yield g, idx_set.copy()

    else:
        idx_set = idx_set.copy()
        it = idx_tw.pop(-1)

        its_in_idx_set = list(set(it).intersection(idx_set))
        if len(its_in_idx_set) > 1:

            for et in its_in_idx_set:
                idx_set.remove(et)

            twin_gen = combinations_with_replacement(comb, len(its_in_idx_set))
            for twin in twin_gen:
                rest = prod_twin_(comb, repeat-len(its_in_idx_set), idx_set, idx_tw)

                for r, idxs in rest:

                    yield twin + r, list(its_in_idx_set) + idxs
        else:
            rest = prod_twin_(comb, repeat, idx_set, idx_tw)
            for r in rest:
                yield r


def distinct_permutations_non_unique_generator(comb, repeat, idx_set, idx_tw):

    gen = prod_twin_(comb, repeat, idx_set, idx_tw)

    for ds, idx in gen:

        if twice_present(comb, ds):
            yield ds, idx


def distinct_permutations_unique_idx_generator(n_uniques, unique_col_idx, non_unique_col_idx, n_uq_idx_cov, idx_tw):

    unique_idx_generator = distinct_combinations_(non_unique_col_idx, n_uq_idx_cov, idx_tw.copy())

    for idx_set_1 in unique_idx_generator:

        # Generate possibilities for indexes to put remaining
        unique_idx_generator_2 = combinations(unique_col_idx, n_uniques - n_uq_idx_cov)
        for idx_set_2 in unique_idx_generator_2:
            idx_set = list(idx_set_1) + list(idx_set_2)

            perms = distinct_permutations_(idx_set, idx_tw.copy())

            for p_idx_set in perms:

                yield p_idx_set

def random_probabilities(n_child_states, n_parent_states, seed):
    np.random.seed(seed=seed)

    probs = []
    for _ in range(n_parent_states):
        dist_c = np.random.rand(n_child_states)
        dist_c = dist_c / sum(dist_c)
        probs.append(list(dist_c))
    return probs


def restrict_specs(n, probabilities, n_parent_states):
    probabilities = np.array(probabilities).flatten()
    n_lowest_idx = np.argsort(probabilities)[:n]
    n_child_states = len(probabilities) // n_parent_states
    n_restricts = [0]*n_parent_states
    for n_p in range(n_parent_states):
        for n_c in range(n_child_states):
            if n_p*n_child_states + n_c in n_lowest_idx:
                n_restricts[n_p] += 1
    return n_restricts




def chain_generators(generators: List[Generator[Tuple, None ,None]],
                     interleave: bool = True,
                     ) -> Generator[Tuple[int], None, None] | Generator[Tuple[int], None, bool]:
    """
    Combine list of generators into single generator

    :param generators: list of generators to chain
    :param interleave: If True, input generators are interleaved, if False, input generators are exhausted consecutively
    :return: New generator
    """

    if interleave:
        interleaved = roundrobin(*generators)
        for e in interleaved:
            yield e

    else:
        chain_ = chain(generators)
        for c in chain_:
            for e in c:
                yield e


