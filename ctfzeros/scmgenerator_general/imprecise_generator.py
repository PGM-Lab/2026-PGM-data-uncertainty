import numpy as np

from ctfzeros.scmgenerator_general.general_generator_utils import value_to_representation
from ctfzeros.scmgenerator_general.general_solution_generator import scm_general_solution_generator


def coefficient_matrix(subset, n_child_states, n_parent_states, norm_constraint=False, remove_last=False):
    #solution_size = (n_child_states - 1) * n_parent_states + 1
    solution_size = len(subset)
    representations = value_to_representation(sorted(subset), n_child_states, n_parent_states)
    matrix_f = np.zeros((n_child_states * n_parent_states, solution_size), dtype=int)
    for i, r in enumerate(representations):
        r = np.add(r, [n_child_states * p for p in range(n_parent_states)])
        matrix_f[r, i] = 1


    if remove_last:
        matrix_f = matrix_f[[i for i in range(n_parent_states * n_child_states) if i % n_child_states != n_child_states - 1]]

    # normalization constraint
    if norm_constraint:
        matrix_f = np.vstack((matrix_f, np.array([1] * solution_size)))

    return matrix_f





def independent_term(probabilities, n_child_states, n_parent_states, norm_constraint=False, remove_last=False):
    probabilities = np.array(probabilities)

    if remove_last:
        probabilities = probabilities.flatten()[
            [i for i in range(n_child_states * n_parent_states) if i % n_child_states != n_child_states - 1]]

    if norm_constraint:
        probabilities = np.append(probabilities, 1.0)
    return probabilities


def get_extreme_points(subset, child_domain_size, parent_domain_size, plow, pupp):
    import polytope as pc


    Usize = len(subset)

    coeff = coefficient_matrix(subset, child_domain_size, parent_domain_size, remove_last=True)
    plow = independent_term(plow, child_domain_size, parent_domain_size, remove_last=True)
    pupp = independent_term(pupp, child_domain_size, parent_domain_size, remove_last=True)

    # 2. Convert ranges to H-representation (Ax <= b)
    # Ax <= pupp  AND  -Ax <= -plow
    A = np.concatenate([coeff, -coeff], axis=0)
    b = np.concatenate([pupp, -plow], axis=0)

    # Normalization constraint

    eps = 0.001
    A = np.concatenate([A, np.array([[1]*Usize, [-1]*Usize])], axis=0)
    b = np.concatenate([b, np.array([1 + eps, -1])], axis=0)

    # 3. Add Non-negativity constraints: x >= 0  => -I*x <= 0
    A_pos = -np.eye(Usize)
    b_pos = np.zeros(Usize)

    A_final = np.vstack([A, A_pos])
    b_final = np.append(b, b_pos)

    # 4. Create Polytope and find Extreme Points
    p = pc.Polytope(A_final, b_final)
    p = pc.reduce(p)
    vertices = pc.extreme(p)
    tol = 1e-7

    if vertices is not None:
        vertices = np.abs(vertices[np.sum(vertices, axis=1) <= 1 + tol])

    if vertices is None:
        vertices = []

    return vertices


def exact_imprecise_empirical(child_domain_size, parent_domain_size,plow, pupp, exclude_us=set(),seed=0):

    scm_generator = scm_general_solution_generator(child_domain_size, parent_domain_size, child_dist=None, exclude_us=exclude_us, seed=seed, linalg_solve=False)

    for subset, _ in scm_generator:
        ext = get_extreme_points(subset,child_domain_size, parent_domain_size, plow, pupp)
        if len(ext) > 0:
            for p in ext:
                yield subset, p







