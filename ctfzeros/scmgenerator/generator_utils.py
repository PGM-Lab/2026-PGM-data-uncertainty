import numpy as np
import itertools
from typing import Set, Tuple, List, Iterable


def get_summand_sets(number: int, n_summands: int) -> Set[Tuple[int]]:
    """
    For number, find all possibilities of fixed total of n summands summing to number

    :param number: the total sum
    :param n_summands: number of summands
    :return: Set of tuples of sorted summands
    """
    if n_summands == number:
        return {tuple([1]*number)}

    else:
        possibilities = []
        for s in get_summand_sets(number-1, n_summands):
            for i in range(len(s)):
                e = list(s).copy()
                e[i] +=1
                possibilities.append(tuple(sorted(e)))

        return set(possibilities)


def irreducible_check(n_latent_values: int, solution: List[int], solver) -> bool:
    """
    Check if a solution is irreducible by removing elements and consulting solver
    :param n_latent_values: total variable count
    :param solution: solution to check
    :param solver: solver to verify reducibility of solution
    :return: True for irreducible, False if not
    """

    if len(solution) < n_latent_values:
        sol = [-s for s in range(1, n_latent_values + 1)]
        for s in solution:
            sol[s - 1] = s
        solution = sol

    works = solver.solve(solution)
    solution_len = len([i for i in solution if i > 0])
    if works:
        count = 0
        for k in solution:
            if k > 0:
                copy = solution.copy()
                copy[k - 1] = -k
                is_solution = solver.solve(copy)
                if not is_solution:
                    count += 1
        if count == solution_len:
            return True
    return False


def binary_neighbours(num: int, n_parents: int) -> List[int]:
    """
    Get numbers that differ in one digit only from the input number
    :param num: U number id
    :param n_parents: number of parents in problem
    :return: List of binary neighbours
    """

    bnum = bin(num-1)[2:]
    bnum = [int(it) for it in "0" * (2**n_parents//2 - len(bnum)) + bnum]

    neighbours = []
    for i in range(len(bnum)):
        new_bnum = bnum.copy()
        new_bnum[i] = (new_bnum[i]+1) % 2
        new_bnum = ''.join(str(e) for e in new_bnum)
        new_num = int(new_bnum, 2)+1
        neighbours.append(new_num)

    return neighbours


def binary_map(number: int, n_digits: int = 0, one_indexed: bool = False) -> str:
    """
    Map integer number to a binary representation
    :param number: Number to encode
    :param n_digits: Number of digits in encoding
    :param one_indexed: If True, subtract 1 from number before encoding
    :return: Binary representation string
    """

    if one_indexed:
        number -= 1


    binary_num = bin(int(number))[2:]

    if not n_digits:
        n_digits = len(binary_num)
    n_pad = n_digits - len(binary_num) if n_digits > len(binary_num) else 0

    # return [int(it) for it in "0" * n_pad + binary_num]
    return "0" * n_pad + binary_num


def u_id_map(binary_num: str, one_indexed: bool = True) -> int:
    """
    For string binary number, return the U-value id number
    :param binary_num: Binary number
    :param one_indexed: If True, U id is one-indexed, so add 1 to returned value
    :return: U id
    """
    # binary_number = ''.join(str(e) for e in binary_map)
    number = int(binary_num, 2)

    return number + 1 if one_indexed else number


def binary_from_one_indexes(one_indexes: Iterable[int], n_digits: int) -> str:
    """
    Map list of integers to binary number

    :param one_indexes: list of the indexes of 1 in binary string, others to be 0
    :param total_digits: total digits in binary string
    :return: binary string
    """

    return "".join(["1" if i in one_indexes else "0" for i in range(n_digits)])




def probability_array_to_bitstring(probabilities: np.ndarray) -> str:
    """
    :param probabilities: Numpy array with y-distribution, length 2*2**n_parents
    :return: bitstring identifying >0.5 probabilities
    """
    probability_bitstring = ""
    for i in range(1, probabilities.shape[0], 2):
        if probabilities[i - 1] >= probabilities[i]:
            probability_bitstring += "0"
        else:
            probability_bitstring += "1"
    return probability_bitstring


def shift_binary_set(binary_set: Tuple[str], shift_str: str = "0"*32) -> List[str]:
    """
    For a set of strings, shift each element by digitwise adding the shift string % 2
    :param binary_set: Set of binary identifiers to shift
    :param shift_str: Shift specification
    :return: List of shifted bitstrings
    """
    shifted_binary_set = []
    for bnum in binary_set:
        shifted_bnum = ""
        for i in range(len(bnum)):
            shifted_bnum += str((int(bnum[i])+int(shift_str[i])) % 2)
        shifted_binary_set.append(shifted_bnum)
    return shifted_binary_set


def pattern_match(bitstring: str, mask: str):
    """
    :param bitstring: Bitstring to be matched
    :param mask: Mask string to match
    :return: True if a match, False otherwise
    """
    for i in range(len(bitstring)):
        if bitstring[i] != mask[i] and mask[i] != "*":
            return False
    return True


def get_candidate_assumptions(y_probabilities: np.ndarray, no_assumptions: int = 1,
                              size_assumptions: int = 1, include_empty: bool = False
                              ) -> Tuple[Tuple[int]]:
    """

    :param y_probabilities: Y distribution from which assumptions are selected
    :param no_assumptions: number of assumptions to select
    :param size_assumptions: size of the assumptions selected
    :param include_empty: True if empty assumption is to be included in the returned assumption tuple
    :return: A tuple of assumption tuples, an assumption with format (,), (1,), (1,2) etc.
    """

    assumption_list = probability_guided_variable_selector(y_probabilities, no_assumptions+size_assumptions)

    assumptions_it = itertools.combinations(assumption_list, size_assumptions)

    assumptions = []
    for i in range(no_assumptions):
        assumptions.append(next(assumptions_it))

    if include_empty or len(assumptions) == 0:
        assumptions = [tuple()] + assumptions

    return tuple(assumptions)


def probability_guided_variable_selector(y_probabilities: np.ndarray,
                                            n_us_considered: int = 1e5,
                                            pattern_mask: str = "",
                                            exclude_set: Tuple = (),
                                            ) -> List[int]:
    """
    Select variables of decreasing degree of consistency with probability vector
    :param y_probabilities: Y distribution from which assumptions are selected
    :param n_us_considered: maximum total us considered
    :param pattern_mask: str eg "0*1*", exclude Us that does not match the pattern
    :param exclude_set: tuple of Us, exclude Us in the tuple
    :return: list of consistent Us
    """

    n_parents = int(np.log2(y_probabilities.shape[0] // 2))

    if len(pattern_mask) == 0:
        pattern_mask = "*"*(2*2**n_parents)

    high_probabilities = []
    probability_bits = []
    for i in range(1, 2 * (2 ** n_parents), 2):
        if y_probabilities[i - 1] >= y_probabilities[i]:
            probability_bits += [0]
            high_probabilities.append((y_probabilities[i - 1], (i - 1) // 2))
        else:
            probability_bits += [1]
            high_probabilities.append((y_probabilities[i], (i - 1) // 2))
    high_probabilities = sorted(high_probabilities, reverse=True)

    top_assumption = u_id_map("".join(str(i) for i in probability_bits), one_indexed=True)
    assumption_list = [top_assumption]

    lowest_high_prob_idx = high_probabilities.pop()[1]
    next_assumption_bits = probability_bits.copy()
    next_assumption_bits[lowest_high_prob_idx] = (next_assumption_bits[lowest_high_prob_idx] + 1) % 2
    next_assumption = u_id_map("".join(str(i) for i in next_assumption_bits), one_indexed=True)
    assumption_list.append(next_assumption)

    current = [probability_bits, next_assumption_bits]
    while len(assumption_list) < n_us_considered and len(high_probabilities) > 0:
        lowest_high_prob_idx = high_probabilities.pop()[1]
        new = []
        for e in current:
            next_assumption_bits = e.copy()
            next_assumption_bits[lowest_high_prob_idx] = (next_assumption_bits[lowest_high_prob_idx] + 1) % 2
            next_assumption = u_id_map("".join(str(i) for i in next_assumption_bits), one_indexed=True)
            assumption_list.append(next_assumption)
            new.append(next_assumption_bits)
        current += new

    assumption_list = [i for i in assumption_list if
                       pattern_match(binary_map(i, one_indexed=True), pattern_mask) and i not in exclude_set]

    return assumption_list


def solution_to_matrix(solution: Tuple[int]) -> np.ndarray:
    """
    From a solution, build the part of the full U matrix corresponding to the solution, i.e. keep only
    non-zero columns
    :param solution: Tuple of U id's
    :return: Full solution matrix, number of colomns = size of solution and number of rows = 2*n_parents
    """
    n_cols = len(solution)

    active_matrix_full = np.zeros(((n_cols-1)*2, n_cols))
    col_idx = 0

    for n in solution:

        col_n = binary_map(n, n_cols-1, one_indexed=True)+"1"  # add one for last row all sum to 1
        col_n = np.asarray(list(col_n), int)

        full_n = np.reshape(
            np.vstack([(col_n[:-1] +1) % 2, col_n[:-1]]),
            newshape=((n_cols-1)*2, ),
            order="F",
        )

        active_matrix_full[:, col_idx] = full_n
        col_idx += 1

    return active_matrix_full


def check_solution(solution, y_dist):
    """
    check if a solution is viable
    :param candidate_solution:
    :param y_probabilities:
    :param parent_probabilities:
    :param query:
    :return: is viable (True/False), solution p(u)
    """

    active_matrix_full = solution_to_matrix(solution)  # full matrix cols corresponding to solution variables

    # square matrix A
    active_matrix_right = active_matrix_full[[i for i in range(1, active_matrix_full.shape[0]+1, 2)]]
    equation_set_matrix = np.vstack((active_matrix_right, np.ones_like(active_matrix_right[0])))

    # equation set probabilities b
    right_prob_idx = [i+1 for i in range(0,y_dist.shape[0], 2)]
    probabilities_right = np.append(y_dist[right_prob_idx], 1.0)

    thetas = np.zeros_like(probabilities_right)

    try:
        # Solve A*thetas = b
        inverse = np.linalg.inv(equation_set_matrix)
        thetas = np.dot(inverse, probabilities_right)

        if (thetas.min() >= 0 and np.isclose(thetas.sum(), 1.0)
                and np.allclose(np.dot(equation_set_matrix, thetas), probabilities_right)):

            result = True

        else:
            result = False

    except (np.linalg.LinAlgError, ValueError):
        result = False


    return result, thetas