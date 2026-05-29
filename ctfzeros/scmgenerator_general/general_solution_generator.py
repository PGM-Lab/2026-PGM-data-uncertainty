import math
from .general_generator_utils import *
from typing import Generator, List, Iterable, Collection, Tuple, Set, Union, Container
import itertools
import random


def first_row_generator(base: int,
                        n_cols: int,
                        n_rows: int,
                        probabilities: List | Tuple | np.ndarray = (),
                        n_restrict: int = 1
                        ) -> Generator[Tuple, None, None]:
    """
    Generating first rows of solution matrices, unique combinations of entries possible
    (permutations of first row would give duplicate solutions)

    :param base: The base of the number system, corresponds to number of child states in model
    :param n_cols: The number of columns in the solution matrix, corresponds to size of the solution
    :param n_rows: The number of rows in the full solutions for which first row is to be generated
    :param probabilities: List with probabilities to restrict solutions search by
    :param n_restrict: Number of probabilities to restrict by
    :return: Generator that generates unique solution first rows
    """

    # If no probabilities provided, no digits restricted by occurence in solution
    restricted_digits = []

    if len(probabilities) > 0:
        # restrict solutions to those containing unique digits corresponding to n_restrict lowest probabilities
        indexes = probabilities_index_sort(probabilities)
        restricted_digits = indexes[:n_restrict]

    # Calculate how many digits required unique in the row to ensure all columns will contain 1 unique digit minimum
    min_uq = 0 if math.ceil(n_cols / (float(base) - 1)) < n_rows else math.floor(n_cols / float(n_rows))

    digit_list = tuple([i for i in range(base)])
    n_remaining_cols = n_cols - len(digit_list)

    # Generate rows with different number of unique digits
    for n_unique in range(max(min_uq, base-n_remaining_cols, len(restricted_digits)), base):

        # Generator for the remainder of the row after every digit is added once
        # If digit is in restricted_digits, this should never be considered for additional occurrence
        non_unique_generator = combinations([i for i in digit_list if i not in restricted_digits], base-n_unique)

        # Consider each possible non-unique digit combination
        for digit_comb in non_unique_generator:

            first_part_row = digit_list + digit_comb

            n_remainder = n_cols - len(first_part_row)

            unique_idx = [i for i in digit_list if i not in digit_comb]

            remainder_row_generator = combinations_with_replacement(digit_comb, n_remainder)

            for remainder in remainder_row_generator:
                full_row = first_part_row + remainder
                # generate the full row, as well as information about which row idx correspond to unique digits
                # and the groups of indexes that share a digit
                yield full_row, unique_idx, index_cps(full_row)


def remaining_rows_generator(base: int,
                             n_cols: int,
                             n_remaining_rows: int,
                             unique_col_idx: List | Tuple,
                             idx_copies: List | Tuple,
                             probabilities: List | Tuple | np.ndarray = (),
                             n_restricts: List | Tuple | np.ndarray = (),
                             c: int = 0
                             ) -> Generator[List, None, None]:
    """
    For the information about unique and duplicate digits of a first row (unique_col_idx and idx_tw),
    generates complete solutions

    :param base: The base of the number system, corresponds to number of child states in model
    :param n_cols: The number of columns in the solution matrix, corresponds to size of the solution
    :param n_remaining_rows: The rows remaining to generate
    :param unique_col_idx: List of the columns which so far are unique
    :param idx_copies: Contains tuples of column indexes that are duplicates
    :param probabilities: List of row probability lists to restrict solution by. If not provided, no restriction.
    :param n_restricts: List of number of restrictions per row
    :param c: To keep track of depth
    :return: Generates complete solutions for first rows
    """

    restricted_digits = []
    if len(probabilities) > 0 and len(probabilities[0]) > 0:
        indexes = probabilities_index_sort(probabilities[c])
        restricted_digits = indexes[:n_restricts[c]]

    digit_list = [i for i in range(base)]

    # Keep track of the columns not yet containing row-unique digits
    non_unique_col_idx = [i for i in range(n_cols) if i not in unique_col_idx]
    n_remaining_cols = len(non_unique_col_idx)

    # Calculate the minimum number of unique digits required in the current row
    if n_cols == base:
        min_row_uq = base
    elif math.ceil(n_remaining_cols / (float(base) - 1)) < n_remaining_rows:
        min_row_uq = 0
    else:
        min_row_uq = math.floor(n_remaining_cols / float(n_remaining_rows))

    # Loop over every legal number of unique digits in row
    for n_unique in range(max(min_row_uq, base - (n_cols-base), len(restricted_digits)), base):

        non_unique_generator = combinations([i for i in digit_list if i not in restricted_digits], base - n_unique)

        # Every non-unique digit combination
        for digit_comb in non_unique_generator:

            uniques = [i for i in digit_list if i not in digit_comb]

            # For this set of unique digits, consider different number of new cols to be unique
            for n_uq_idx_cov in range(min_row_uq, min(len(uniques)+1, n_remaining_cols+1)):

                # Generate distinct permutations of row indexed to place unique digits
                # The non-unique columns to be made unique are currently not unique and combinations are
                # generated according to the information about duplicates in idx_tw
                unique_idx_generator = distinct_permutations_unique_idx_generator(len(uniques), unique_col_idx,
                                                                                  non_unique_col_idx, n_uq_idx_cov,
                                                                                  idx_copies)

                for u_idx_set in unique_idx_generator:

                    # remaining indexed to fill with non-unique digits
                    remaining_idx = [i for i in range(n_cols) if i not in u_idx_set]

                    # possible digit configurations for the non-unique part of the row
                    remainder_row_generator = distinct_permutations_non_unique_generator(digit_comb,
                                                                                         n_cols - len(uniques),
                                                                                         remaining_idx,
                                                                                         idx_copies.copy())

                    for complete_row, non_u_idx_set in remainder_row_generator:

                        # Create row
                        row = [0]*n_cols
                        for i, idx in enumerate(u_idx_set):
                            row[idx] = uniques[i]
                        for i, idx in enumerate(non_u_idx_set):
                            row[idx] = complete_row[i]

                        """row = np.zeros((1,n_cols), dtype=np.int32)
                        row[:,u_idx_set] = np.array(uniques)
                        row[:,non_u_idx_set] = np.array(complete_row)
                        row = list(row[0])"""

                        if n_remaining_rows == 1:
                            # this is the final row
                            yield [row]

                        else:

                            # Update index info
                            uniqe_idx_new = list(set(u_idx_set).union(set(unique_col_idx)))
                            idx_tw_new = common_cps(index_cps(row), idx_copies)

                            # complete the solution by generating remaining rows
                            remaining_rows = remaining_rows_generator(base, n_cols, n_remaining_rows-1,
                                                                      uniqe_idx_new, idx_tw_new,
                                                                      probabilities, n_restricts, c+1)

                            for sub_solution in remaining_rows:
                                yield [row]+sub_solution


def solution_matrix_generator(n_child_states, n_parent_states, size, probabilities = (), n_restricts=()):

    base = n_child_states
    n_cols = size
    n_rows = n_parent_states

    probabilities = [list(p) for p in probabilities]
    first_row_prob = () if len(probabilities) == 0 else probabilities.pop(0)
    first_row_restrict = () if len(n_restricts) == 0 else n_restricts.pop(0)

    for first_row, unique_idx, idx_copies in first_row_generator(base=base, n_cols=n_cols, n_rows=n_rows,
                                                             probabilities=first_row_prob,
                                                             n_restrict=first_row_restrict):

        base_solution = np.zeros((n_rows, n_cols), dtype=np.int32)
        base_solution[0] = np.array(first_row)

        remainder = remaining_rows_generator(base, n_cols, n_rows-1, unique_idx, idx_copies,
                                             probabilities=probabilities,
                                             n_restricts=n_restricts)
        c=0
        for sub_solution in remainder:
            solution = base_solution.copy()
            solution[1:] = np.array(sub_solution)
            c+=1
            yield tuple(sorted(matrix_to_set(solution)))


def expanded_irreducible_generator(n_child_states,
                                   n_parent_states,
                                   size,
                                   exclude_us=(),
                                   random_samples = True,
                                   max_expansions=int(1e6),
                                   seed = 0,
                                   probabilities = (),
                                   n_restricts = ()
                                   ):

    random.seed(seed)
    base_generator = solution_matrix_generator(n_child_states, n_parent_states,size, probabilities, n_restricts)

    expansion_set = [i for i in range(n_child_states**n_parent_states) if i not in exclude_us]

    for base in base_generator:

        base_set = set(base)

        # compare with exclude set
        base_exclude_overlap = len(base_set.intersection(exclude_us))
        if base_exclude_overlap > 0:
            # skip base inconsistent with exclude set
            continue

        expansion_set_current = list(set(expansion_set) - base_set)

        expansion_size = int((n_child_states-1)*n_parent_states + 1 - len(base))

        # iterators for expansion
        if not random_samples or math.comb(len(expansion_set_current), expansion_size) <= max_expansions:
            expansion_iterator = itertools.combinations(expansion_set_current, expansion_size)
        else:
            expansion_iterator = range(max_expansions)

        break_count = 0
        # generate expanded solutions
        for exp in expansion_iterator:

            if random_samples and math.comb(len(expansion_set_current), expansion_size) > max_expansions:
                # exp = tuple(np.random.choice(expansion_set_current, size=expansion_size, replace=False))
                exp = tuple(random.sample(expansion_set_current, expansion_size))

            yield tuple(sorted(base + exp))
            break_count += 1
            if break_count == max_expansions:
                break


def scm_general_solution_generator(n_child_states, n_parent_states, child_dist=None, n_restricts=(), exclude_us=(),
                             exhaustive=False, random=False, complete_dist=False, seed=0, linalg_solve = True):

    if child_dist is None:
        child_dist = np.array([1 / n_child_states] * (n_child_states * n_parent_states))

    probabilities = np.array(child_dist).reshape((n_parent_states, n_child_states))

    solution_size = (n_child_states-1)*n_parent_states+1

    if n_child_states**n_parent_states < 20:
        exhaustive = True

    if exhaustive and not random:
        generator = exhaustive_cnf_generator(n_child_states, n_parent_states, exclude_us)
    elif exhaustive and random:
        generator = random_cnf_generator(n_child_states, n_parent_states, exclude_us, seed=seed)
    else:

        sizes = [i for i in range(solution_size-1, n_child_states, -1)][1:4]
        first = solution_size-5 if n_parent_states == 2 else solution_size-2
        restrict_n = [first, max(1, first-3), 0]

        #print(sizes, restrict_n)
        generators_r = []
        for r in restrict_n:
            generators_s = []
            for s in sizes:
                restrict_s = restrict_specs(r, probabilities.copy(), n_parent_states)
                g = expanded_irreducible_generator(n_child_states, n_parent_states, exclude_us=exclude_us,
                                           size=s, probabilities=probabilities.copy(), n_restricts=restrict_s, seed=seed)
                generators_s.append(g)
            generators_r.append(chain_generators(generators_s, interleave=True))
        generator = chain_generators(generators_r, interleave=False)

    probabilities = np.array(probabilities).flatten()[[i for i in range(n_child_states*n_parent_states) if i % n_child_states != n_child_states-1]]
    probabilities = np.append(probabilities, 1.0)


    for subset in generator:
        representations = value_to_representation(sorted(subset), n_child_states, n_parent_states)
        matrix_f = np.zeros((n_child_states*n_parent_states, solution_size), dtype=int)
        for i, r in enumerate(representations):
            r = np.add(r, [n_child_states*p for p in range(n_parent_states)])
            matrix_f[r, i] = 1

        matrix_s = matrix_f[[i for i in range(n_parent_states*n_child_states) if i % n_child_states!= n_child_states-1]]
        matrix_s = np.vstack((matrix_s, np.array([1]*solution_size)))

        if linalg_solve:

            try:
                inverse = np.linalg.inv(matrix_s)
                thetas = np.dot(inverse, probabilities)
                if (thetas.min() >= 0 and np.isclose(thetas.sum(), 1.0)
                        and np.allclose(np.dot(matrix_s, thetas), probabilities)):

                    if complete_dist:
                        complete_solution = [0.0] * solution_size
                        for i in subset:
                            complete_solution[i] = round(thetas[subset.index(i)], 4)
                        yield tuple(complete_solution)
                    else:
                        yield subset, thetas

            except (np.linalg.LinAlgError, ValueError):
                pass

        else:
            yield subset, None



def random_cnf_generator(n_child_states, n_parent_states, exclude_us = (), seed=0):
    random.seed(seed)
    exo_values = [i for i in range(n_child_states ** n_parent_states) if i not in exclude_us]
    solution_size = (n_child_states - 1)*n_parent_states + 1

    while True:
        suggestion = tuple(sorted(random.sample(exo_values, solution_size)))
        if test_solution_satisfiability(list(suggestion), n_child_states, n_parent_states):
            yield suggestion


def exhaustive_cnf_generator(n_child_states, n_parent_states, exclude_us = ()):

    exo_values = [i for i in range(n_child_states ** n_parent_states) if i not in exclude_us]
    solution_size = (n_child_states - 1)*n_parent_states + 1

    subset_generator = itertools.combinations(exo_values, solution_size)
    for subset in subset_generator:
        if test_solution_satisfiability(list(subset), n_child_states, n_parent_states):
            yield subset


if __name__ == "__main__":

    # Model is endogenous parent -> child <- exogenousU, parent and child of arbitrary domain sizes
    child_domain_size = 4
    parent_domain_size = 3

    # List of distributions P(child|parent)
    #probabilities = [[0.45, 0.3, 0.15, 0.1], [0.25, 0.05, 0.4, 0.3], [0.4, 0.25, 0.2, 0.15], [0.5, 0.35, 0.05, 0.1]]

    # generate random distribution
    probabilities = random_probabilities(child_domain_size, parent_domain_size, seed=0)
    #probabilities = np.array(probabilities)
    #probabilities = probabilities.reshape((child_domain_size*parent_domain_size,))

    # U values to exclude, will always have probability 0
    exclude_us = (0,)

    scm_generator_general = scm_general_solution_generator(child_domain_size, parent_domain_size,
                                                     child_dist=probabilities, exclude_us=exclude_us, seed=0)

    for u_domain, u_values in scm_generator_general:
        print(u_domain, u_values)







