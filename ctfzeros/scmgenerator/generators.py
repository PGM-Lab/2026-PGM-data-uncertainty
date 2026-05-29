import itertools
import numpy as np
import math
import collections
import random
from typing import List, Tuple, Generator
from more_itertools import roundrobin
from pysat import solvers
from ctfzeros.scmgenerator.generator_utils import binary_neighbours, get_summand_sets, irreducible_check, binary_map, u_id_map, \
    binary_from_one_indexes, shift_binary_set, pattern_match, get_candidate_assumptions, \
    probability_guided_variable_selector, check_solution



"""
Model representation is Us in {1,2,...,2**2**n}
1-indexed U_m is found in clauses specified by binary number m-1, where 0 is left clause and 1 is right clause

Bitstring "1010" to represent right-clause, left-clause, right-clause, left-clause i.e. U_11 for n=2

"""


# TODO len 2 irreducibles does not work
# TODO 0-index U


def all_solutions(n_parents: int, solution_size:int, max_one_clauses: Tuple = (),
                  ) -> Generator[Tuple[int], None, None]:
    """
    Generate all irreducible solutions as tuples of variable id
    :param n_parents: number of parents
    :param solution_size: size of (irreducible) solutions to generate
    :param max_one_clauses: list of clauses restricted to one active variable
    :return: solution generator, solution format like (2,3,5,9) for solution size 4
    """

    if len(max_one_clauses) == 0:
        bitstring_solutions = all_bitstring_solutions(n_parents, solution_size)

        for solution in bitstring_solutions:
            for shift_id in range(2**2**n_parents):
                bin_shift = binary_map(shift_id, 2**n_parents, one_indexed=False)
                shifted_sol = shift_binary_set(solution, bin_shift)

                yield tuple(sorted([u_id_map(e) for e in shifted_sol]))

    else:

        max_one_clauses_pattern = ""
        max_one_clauses_std_format = ()
        for i in range(1, 2 * (2 ** n_parents), 2):
            if i in max_one_clauses:
                max_one_clauses_pattern += "0"
                max_one_clauses_std_format += (int((i - 1) // 2),)
            elif i - 1 in max_one_clauses:
                max_one_clauses_pattern += "1"
                max_one_clauses_std_format += (int((i - 1) // 2),)
            else:
                max_one_clauses_pattern += "*"
        for shift_id in range(2 ** 2 ** n_parents):

            bin_shift = binary_map(shift_id, 2 ** n_parents, one_indexed=False)
            if pattern_match(bin_shift, max_one_clauses_pattern):
                bitstring_generator = all_bitstring_solutions(n_parents, solution_size, max_one_clauses_std_format)
                for solution in bitstring_generator:
                    shifted_sol = shift_binary_set(solution, bin_shift)
                    yield tuple(sorted([u_id_map(e) for e in shifted_sol]))


def all_bitstring_solutions(n_parents: int, solution_size: int,
                            max_one_clauses: Tuple = (),
                            ) -> Generator[Tuple[str], None, None]:
    """
    For the n-parent problem, generate irreducible bitstring solutions of given size
    :param n_parents: number of parents
    :param solution_size: size of solution
    :param max_one_clauses: clauses to choose single-occurence U from
    :return: bitstring solution generator
    """

    all_clauses = [i for i in range(2**n_parents)]  # clauses ided by number

    summand_sets = get_summand_sets(2**n_parents, solution_size)
    #summand_sets = random.sample(sorted(summand_sets), len(summand_sets))

    generators = []

    for solution_specs in summand_sets:
        if len(max_one_clauses) <= len([i for i in solution_specs if i == 1]):
            std_solutions = generate_std_solutions(solution_specs, all_clauses, 2**n_parents, max_one_clauses)
            generators.append(std_solutions)
            #for solution in std_solutions:
            #    yield solution

    chained_generator = chain_generators(generators, interleave=True)
    for solution in chained_generator:
        yield solution


def generate_std_solutions(solution_specs: Tuple[int], clauses: List[int], total_clauses: int,
                           max_one_clauses: Tuple = (),
                           ) -> Generator[Tuple[str], None, None]:
    """

    :param solution_specs: Specifies the size of each clause-group of the solution, such that Us are chosen
                            that only appear in exactly the clauses of a group
    :param clauses: Clauses to build solutions over
    :param total_clauses: Total number of clauses/size of solution
    :param max_one_clauses: Clauses to choose single-occurence Us from
    :return: Bitstring solution generator over standard "0000" clause config
    """

    size_counter = collections.Counter(solution_specs)

    base_solution = tuple()
    if len(max_one_clauses) > 1:
        base_solution = tuple([binary_from_one_indexes((i,), total_clauses) for i in max_one_clauses])
        size_counter[1] = size_counter[1] - len(max_one_clauses)
        solution_specs = tuple([1]*size_counter[1] + [e for e in solution_specs if e != 1])
        clauses = [i for i in clauses if i not in max_one_clauses]

    if len(clauses) == 0:
        yield base_solution
    else:
        current_size = sorted(size_counter.keys(), reverse=True)[0]

        current_size_subsolutions = generate_same_size_subsolutions(clauses,
                                                                    total_clauses,
                                                                    current_size,
                                                                    size_counter[current_size])

        for subsolution in current_size_subsolutions:

            solution = base_solution[:]
            remaining_clauses = clauses.copy()

            for bitstring_id in subsolution:

                solution += (bitstring_id, )
                for idx in range(len(bitstring_id)):
                    if bitstring_id[idx] =="1":
                        remaining_clauses.remove(idx)

            if len(size_counter) == 1:
                yield solution

            else:
                remaining_specs = tuple([e for e in solution_specs if e != current_size])

                recursive_generator = generate_std_solutions(remaining_specs,
                                                             remaining_clauses,
                                                             total_clauses)

                for recursive_solution in recursive_generator:
                    solution = subsolution[:] + recursive_solution[:]
                    yield solution


def generate_same_size_subsolutions(clauses: List[int], total_clauses: int, size: int, num: int,
                                    break_count: int = -1
                                    ) -> Generator[Tuple[str], None, None]:
    """
    For a set of clauses, choose number of subsets of given size such that no clause is found in multiple subsets

    :param clauses: Clauses to build solution for, eg. [0,1,2,3] indicates first 4 clauses to be
    included (from right side clause set)
    :param total_clauses: Total number of clauses
    :param size: The size of the subsolutions to generate
    :param num: Number of subsolutions
    :param break_count: Break counter to avoid generating duplicate subsolutions
    :return: Tuple of binary solution identifiers
    """


    if size == 1 and num == len(clauses):
        # If looking for irreducible solution of max size, ie the set of U's present in exactly one of the clauses
        yield tuple([binary_from_one_indexes((i,), total_clauses) for i in clauses])

    else:
        subsolution_generator = itertools.combinations(clauses, size) # iterator over subsolutions

        if num == 1:
            for subsolution in subsolution_generator:

                if break_count < subsolution[0]:
                    yield tuple([binary_from_one_indexes(subsolution, total_clauses)])
        else:

            for subsolution in subsolution_generator:

                if break_count > subsolution[0]:
                    # skip already generated solutions
                    continue

                remaining_clauses = clauses.copy()
                for clause_id in subsolution:
                    remaining_clauses.remove(clause_id)

                recursive_gen = generate_same_size_subsolutions(remaining_clauses, total_clauses, size, num-1, subsolution[0])

                for recursive_solution in recursive_gen:
                    bitstring_solution = binary_from_one_indexes(subsolution, total_clauses)
                    yield tuple(sorted((bitstring_solution,) + recursive_solution, reverse=True))


def probability_guided_solutions(n_parents: int, probabilities: np.ndarray, solution_size: int,
                                 max_one_clauses: Tuple = ()
                                 ) -> Generator[Tuple[int], None, None]:
    """
    Solution generation guided by a problem specific probability array
    :param n_parents: Number of parents
    :param probabilities: All y probabilities, 2*2**n_parents length array
    :param solution_size: Size of irreducible solutions to generate
    :param max_one_clauses: Tuple of clause ids for clauses with max one active, clause id in (0, ..., 2*2**n_parents)
    :return: solution generator, solutions format like (2,3,5,9)
    """

    probability_bitstring = ""
    max_one_clauses_std_format = ()
    for i in range(1, 2*(2**n_parents), 2):
        if probabilities[i-1] >= probabilities[i]:
            probability_bitstring += "0"
            if i in max_one_clauses:
                max_one_clauses_std_format += (int((i-1)//2),)
        else:
            probability_bitstring += "1"
            if i-1 in max_one_clauses:
                max_one_clauses_std_format += (int((i-1)//2),)

    # standard format solution generator
    bitstring_generator = all_bitstring_solutions(n_parents, solution_size, max_one_clauses_std_format)
    for solution in bitstring_generator:
        # shift the solution to match probability guided selection
        shifted_sol = shift_binary_set(solution, probability_bitstring)
        yield tuple(sorted([u_id_map(e) for e in shifted_sol]))


def full_length_solutions(base_generator: Generator, new_size: int,
                                   expansion_set: List = (),
                                   assumptions: Tuple = (),
                                   exclude: Tuple = (),
                                   random_samples: bool = True,
                                   max_expansions: int = int(1e6),
                                   seed: int = 0):
    """

    :param base_generator: Irreducible solution generator
    :param new_size: expansion size
    :param expansion_set: set of variables to add to solution from
    :param assumptions: variables to always include
    :param exclude: variables to always exclude
    :param random_samples: if True, choose expansion variables at random
    :param max_expansions: number of expansions to check for each solution base
    :return: Generator for extended solutions of given length
    """
    #np.random.seed(seed=seed)
    random.seed(seed)

    for base in base_generator:

        base_set = set(base)

        # compare with exclude set
        base_exclude_overlap = len(base_set.intersection(exclude))
        if base_exclude_overlap > 0:
            # skip base inconsistent with exclude set
            continue

        # expansion set
        expansion_set_current = expansion_set.copy()
        for v in base:
            if v in expansion_set_current:
                expansion_set_current.remove(v)

        # force include assumption Us
        base = tuple(base_set.union(assumptions))
        expansion_size = int(new_size - len(base))

        # iterators for expansion
        if not random_samples or math.comb(len(expansion_set_current), expansion_size) <= max_expansions:
            expansion_iterator = itertools.combinations(expansion_set_current, expansion_size)
        else:
            expansion_iterator = range(max_expansions)

        break_count = 0
        # generate expanded solutions
        for exp in expansion_iterator:

            if random_samples and math.comb(len(expansion_set_current), expansion_size) > max_expansions:
                #exp = tuple(np.random.choice(expansion_set_current, size=expansion_size, replace=False))
                exp = tuple(random.sample(expansion_set_current, expansion_size))

            yield tuple(sorted(base + exp))
            break_count += 1
            if break_count == max_expansions:
                break


def build_solution_generator(n_parents: int,
                             n_max_one_clauses: int = 0,
                             irreducible_solution_lengths: Tuple[int] = (),
                             probability_guided_irreducibles: bool = True,
                             y_distribution: np.ndarray = (),
                             max_expansions: int = int(1e4),
                             random_expansions: bool = False,
                             probability_guided_expansion: bool = False,
                             exclude_us = (),
                             n_assumptions: int = 0,
                             size_assumptions: int = 1,
                             include_no_assumption_generator: bool = False,
                             solver: bool = False,
                             seed: int = 0) -> Generator[Tuple[int], None, None]:
    """
    Return a solution generator given requirements
    :param n_parents: # parent variables in problem to be solved
    :param n_max_one_clauses: # clauses to restrict to one variable, will be chosen according to the distribution
    :param irreducible_solution_lengths: The irreducible solutions sizes to generate full solutions from
    :param probability_guided_irreducibles: True if irreducibles are to be build guided by the distribution
    :param y_distribution: The distribution over the y variable given the parents
    :param max_expansions: Number of solutions to generate for each irreducible base solution
    :param random_expansions: True if irreducible solutions are to be extended by random selection of variables
    :param probability_guided_expansion: True if the variables added to the irreducible solution are chosen
    guided by the y-distribution
    :param n_assumptions: Number of assumptions to include in generator
    :param size_assumptions: Number of variables to include in each assumption
    :param include_no_assumption_generator: True if a generator with no assumptions is to be included and chained
    with the assumption generators
    :param solver: If True, a solver-based generator is returned, disregarding all requirements except exclude list
    :return: The solution generator
    """

    solution_size = 2**n_parents + 1

    # If solver is True, ignore all other requirements and return a solver-based solution generator
    if solver:
        return solver_based_solution_generator(n_parents, solution_size, exclude_us=exclude_us)

    # Assumptions
    assumptions = get_candidate_assumptions(y_distribution, no_assumptions=n_assumptions,
                                            size_assumptions=size_assumptions,
                                            include_empty=include_no_assumption_generator)
    #print("assumptions:", assumptions)

    # Zero detection
    idx_zeros = np.where(y_distribution == 0.0)[0]
    #print("zero clauses:", idx_zeros)

    # If n_max_one_clauses > zero probability clauses, restrict lowest probability clauses to contain max one variable
    guide = y_distribution.copy().reshape(2*(2**n_parents), )
    if len(idx_zeros) < n_max_one_clauses:
        for _ in range(n_max_one_clauses-len(idx_zeros)):
            guide[idx_zeros] = 1.0
            idx_lowest_prob = np.where(guide == np.min(guide))[0][0]
            idx_zeros = np.append(idx_zeros, idx_lowest_prob)
    #print("max one variable clauses (including zero clauses): ", idx_zeros)
    restricted_clauses = tuple(sorted(idx_zeros))

    # Irreducible length settings
    generator_settings = [i for i in range(solution_size-max(1,len(assumptions[0])), max(n_max_one_clauses,2), -1)]
    if len(irreducible_solution_lengths) > 0:
        generator_settings = [i for i in generator_settings if i in irreducible_solution_lengths]
        if 2 in irreducible_solution_lengths:
            generator_settings.append(2)
    #print("irreducible length settings:", generator_settings)

    #generator_settings = random.sample(generator_settings, len(generator_settings))

    # Expansion set

    # pattern to match to avoid max one clause variables
    expand_str = ""
    for i in range(1, 2 * (solution_size - 1), 2):
        if i in restricted_clauses:
            expand_str += "0"
        elif i - 1 in restricted_clauses:
            expand_str += "1"
        else:
            expand_str += "*"

    # all remaining variables to be considered
    if probability_guided_expansion:
        n_us = min((2 ** 2 ** n_parents) // 2, 1e4)
        expansion_set = probability_guided_variable_selector(y_distribution,
                                                             pattern_mask=expand_str,
                                                             n_us_considered=int(n_us))
    else:
        n_us = 2 ** 2 ** n_parents
        expansion_set = [i for i in range(1, n_us + 1)
                         if pattern_match(binary_map(i, one_indexed=True), expand_str)]
    expansion_set = list(set(expansion_set) - set(exclude_us))
    #print("Size of expansion set:", len(expansion_set))


    assumption_specific_generators = []
    for assumptions_list in assumptions:
        base_generators = []
        for irreducible_len in generator_settings:
            if probability_guided_irreducibles:
                base_generator = probability_guided_solutions(n_parents, y_distribution, irreducible_len,
                                                        max_one_clauses=restricted_clauses)
            else:
                base_generator = all_solutions(n_parents,
                                         solution_size=irreducible_len,
                                         max_one_clauses=restricted_clauses)

            base_generators.append(base_generator)

        chained_base_generator = chain_generators(base_generators, interleave=True)
        candidates = full_length_solutions(chained_base_generator, solution_size,
                                                 exclude=exclude_us,
                                                 expansion_set=expansion_set,
                                                 assumptions=assumptions_list,
                                                 max_expansions=int(max_expansions),
                                                 random_samples=random_expansions,
                                                 seed=seed)

        assumption_specific_generators.append(candidates)

    generator = chain_generators(assumption_specific_generators, interleave=True)
    return generator


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
        chain = itertools.chain(generators)
        for c in chain:
            for e in c:
                yield e


def similar_solution_generator(solution: Tuple[int], n_parents: int) -> Generator[Tuple[int], None, None]:
    """
    Given a solution, generate new neighbour solutions, i.e. replace a variable by one of its binary neighbours
    :param solution: The solution for which neighbours are to be generated
    :param n_parents: The current parent setting, required to find neighbours
    :return: Solution neighbour generator
    """

    for i in range(len(solution)):
        if True:
            for new_num in binary_neighbours(solution[i], n_parents):
                if new_num not in solution:
                    inc_solution = list(solution).copy()
                    inc_solution[i] = new_num
                    yield tuple(sorted(inc_solution))


def neighbourhood_generator(viable_solution: Tuple[int], n_parents: int, y_dist, track_uniques = True):
    explored_sols = []
    to_explore_sols = [viable_solution]
    c = 0
    for viable_solution in to_explore_sols:

        if track_uniques:
            explored_sols.append(viable_solution)

        for similar_solution in similar_solution_generator(viable_solution, n_parents):
            c += 1
            is_viable, _ = check_solution(similar_solution, y_dist)

            if is_viable and similar_solution not in explored_sols and similar_solution not in to_explore_sols:

                to_explore_sols.append(similar_solution)

                yield similar_solution, c
                c = 0


def scm_solution_generator(n_parents: int,
                              y_dist: np.ndarray,
                              exclude_us: Tuple = (),
                              probability_guided_irreducibles: bool = True,
                              probability_guided_expansion: bool = True,
                              max_expansions: int = 1000,
                              random_expansions: bool = True,
                              solver: bool = False,
                              neighbour_limit: int = 0,
                              seed: int = 0,
                              ) -> Generator[Tuple[Tuple, Tuple, int], None, None]:

    exclude_us_1_idx = [i+1 for i in exclude_us]

    generator = build_solution_generator(n_parents=n_parents,
                                         exclude_us=exclude_us_1_idx,
                                         y_distribution=y_dist,
                                         probability_guided_irreducibles=probability_guided_irreducibles,
                                         probability_guided_expansion=probability_guided_expansion,
                                         max_expansions=max_expansions,
                                         random_expansions=random_expansions,
                                         solver=solver,
                                         seed=seed)

    if not solver and probability_guided_irreducibles:
        generator_2 = build_solution_generator(n_parents=n_parents,
                                             exclude_us=exclude_us_1_idx,
                                             y_distribution=y_dist,
                                             probability_guided_irreducibles=False,
                                             probability_guided_expansion=False,
                                             max_expansions=max_expansions,
                                             random_expansions=random_expansions,
                                             seed=seed
                                             )
        generator = chain_generators([generator, generator_2], interleave=False)


    # TODO simplify, neighbourhood generator always returns viable solutions
    c = 0
    for solution in generator:
        c += 1
        is_viable, thetas = check_solution(solution, y_dist)
        if is_viable:
            solution_0_idx = tuple([i - 1 for i in solution])
            yield solution_0_idx, thetas, c
            c = 0

            if neighbour_limit > 0:

                g = neighbourhood_generator(solution, n_parents=int(math.log2(len(y_dist)//2)), y_dist=y_dist,
                                            track_uniques=True)
                nl = 0
                for new_point in g:
                    nl += 1
                    new_solution = new_point[0]
                    c += new_point[1]
                    is_viable, thetas = check_solution(new_solution, y_dist)
                    if is_viable:
                        new_solution_0_idx = tuple([i-1 for i in new_solution])
                        yield new_solution_0_idx, thetas, c
                        c = 0

                    if nl >= neighbour_limit:
                        break


def solver_based_solution_generator(n_parents: int, solution_length: int, irreducible_only: bool = False,
                                    exclude_us=()
                                    ) -> Generator[Tuple[int], None, None]:
    """
    Solver based generator
    :param n_parents: Number of parents
    :param solution_length: Length of solution
    :param irreducible_only: True generates only irreducible solutions of requested length
    :return: Solution generator
    """

    n_latent_values = 2**(2**n_parents)

    clauses = []
    for idx in range(1, 2**n_parents+1):
        block_size = (n_latent_values // 2**idx)

        pattern = [1] * block_size + [0] * block_size
        this_row = pattern * (2 ** (idx - 1))

        clauses.append([i for i in range(1, n_latent_values+1) if this_row[i-1] == 1])
        clauses.append([i for i in range(1, n_latent_values+1) if this_row[i-1] == 0])

    solver = solvers.Solver("minisat22")
    for clause in clauses:
        solver.add_clause(clause)

    var_list = [i for i in range(1, n_latent_values + 1)]
    for var in exclude_us:
        var_list.remove(var)

    for var_set in itertools.combinations(var_list, solution_length):

        solution = [-s for s in range(1, n_latent_values + 1)]

        for var in var_set:
            solution[var - 1] = int(var)

        works = solver.solve(solution)
        if works:
            if not irreducible_only:
                yield tuple([s for s in solution if s > 0])
            else:
                if irreducible_check(n_latent_values, solution, solver):
                    yield tuple([s for s in solution if s > 0])



if __name__ == "__main__":

    y_dist = np.array(
                        [
                            0.3,
                            0.7,
                            1.0,
                            0.0,
                            0.47368,
                            0.52632,
                            0.83333,
                            0.16667,
                            #0.2316,
                            #0.7684,
                            #0.99167,
                            #0.00833,
                            #0.25177,
                            #0.74823,
                            #0.09589,
                            #0.90411

                        ]
                    ).reshape((8, 1))

    exclude_us = (0,1)

    scm_generator = scm_solution_generator(
                                            n_parents=2,
                                            y_dist=y_dist,
                                            exclude_us=exclude_us,
                                            solver=False,
                                            seed=4
                                           )

    for u_domain, u_values, _ in scm_generator:
        print(u_domain, u_values)



