from bcause.factors.imprecise import IntervalProbFactor

import numpy as np

def perturbate_prob(factor, eps, reachability=True):
    assert len(factor.left_vars) == 1
    Y = factor.left_vars[0]
    X = factor.right_vars
    assert len(factor.domain[Y]) == 2

    factor = factor.reorder(*X, Y)

    probs = factor.values
    plow = [max(p - eps, 0) for p in probs]
    pupp = [min(p + eps, 1) for p in probs]

    if reachability:
        assert len(factor.domain[Y]) == 2   # implemented for binary children (it could be generalized)
        for i in range(1, len(probs) + 1, 2):
            plow[i], pupp[i] = 1 - pupp[i - 1], 1 - plow[i - 1]

    return IntervalProbFactor(factor.domain, values_low=plow, values_up=pupp, left_vars=factor.left_vars)


def full_U_probs(subset, ext, Usize):
    p = np.zeros(Usize)
    ext = list(ext)
    for j, i in enumerate(subset): p[i] = ext[j]
    return p



def map_probs(dom1, dom2, p1):
    assert set(dom1).issubset(set(dom2))
    assert len(p1) == len(dom1)

    p2 = np.zeros(len(dom2))
    p1 = list(p1)

    for pos_old, state in list(enumerate(dom1)):
        prob = p1[pos_old]
        pos_new = dom2.index(state)
        p2[pos_new] = prob

    return p2