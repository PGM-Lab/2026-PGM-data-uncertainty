import numpy as np
from bcause import MultinomialFactor
from bcause.factors.mulitnomial import canonical_multinomial
from bcause.inference.causal.elimination import CausalVariableElimination
from bcause.models.cmodel import StructuralCausalModel
import bcause.util.domainutils as dutils
from bcause.util.domainutils import subdomain

def update_exo_probs(model, new_domains, new_probs):
    # check input are exogenous and for the same variables
    assert new_domains.keys() == new_probs.keys()
    assert set(new_probs.keys()).issubset(set(model.exogenous))

    target_vars = new_probs.keys()

    #update the exogenous factors
    new_factors = {}
    for v in model.variables:
        if v in target_vars:

            dom = new_domains[v]
            if type(dom) is not dict: dom = {v:new_domains[v]}
            f = MultinomialFactor({v:new_domains[v]}, values = new_probs[v])
        elif not target_vars.isdisjoint(model.get_parents(v)):
            f = model.factors[v].restrict(**new_domains)
        else:
            f = model.factors[v]
        new_factors[v] = f

    # Return the new model
    return StructuralCausalModel(model.graph, new_factors)




def missing_exo_state(f, exovar):
    endoVars = [x for x in f.variables if x != exovar]
    endoDom = dutils.subdomain(f.domain, *endoVars)

    canonical_eq = canonical_multinomial(endoDom, exovar, [x for x in f.right_vars if x != exovar]).reorder(
        *f.variables)


    def is_present(u):
        values1 = canonical_eq.R(**{exovar:u}).values
        for v in f.domain[exovar]:
            values2 = f.R(**{exovar:u}).values
            if values1 == values2:
                return True
        return False

    return [u for u in canonical_eq.domain[exovar] if not is_present(u)]


def get_missing_states(f,exovar):
    leftvar = f.left_vars[0]
    endoVars = [x for x in f.variables if x != exovar]
    endoPa = [x for x in endoVars if x != leftvar]
    ycard = len(f.domain[leftvar])

    exoCard = ycard ** np.prod([len(f.domain[v]) for v in endoPa])
    #if len(f.domain[leftvar]) != 2: raise ValueError("Non binary variable")

    return [u for u in range(exoCard) if u not in f.domain[exovar]]
def get_state_mapping(f,exovar):
    leftvar = f.left_vars[0]
    endoVars = [x for x in f.variables if x != exovar]
    endoPa = [x for x in endoVars if x != leftvar]

    ycard = len(f.domain[leftvar])

    exoCard = ycard ** np.prod([len(f.domain[v]) for v in endoPa])

    #if len(f.domain[leftvar]) != 2: raise ValueError("Non binary variable")



    def correct_state(v):
        f.R(**{exovar: v})
        values = "".join([str(int(x)) for x in f.R(**{exovar: v}).to_deterministic().values])
        return int(values, ycard)

    map = {correct_state(v): v for v in f.domain[exovar]}
    return {u: map[u] if u in map else None for u in range(exoCard)}


# def get_state_mapping(f, exovar):
#     endoVars = [x for x in f.variables if x != exovar]
#     endoDom = dutils.subdomain(f.domain, *endoVars)
#
#     canonical_eq = canonical_multinomial(endoDom, exovar, [x for x in f.right_vars if x != exovar]).reorder(
#         *f.variables)
#
#     def get_mapped_state(u):
#         values1 = canonical_eq.R(**{exovar:u}).values
#         for v in f.domain[exovar]:
#             values2 = f.R(**{exovar:v}).values
#             if values1 == values2:
#                 return v
#         return None
#
#     return {u:get_mapped_state(u) for u in canonical_eq.domain[exovar]}


def update_domains(self, **domains):
    new_factors = self.factors
    for v,d in domains.items():
        for k in new_factors.keys():
            f = new_factors[k]
            if v in f.variables:
                f = f.change_domains(**{v:d})
            new_factors[k] = f
    return self.builder(dag=self.graph, factors=new_factors)

def lazy_cartesian_product(gen1, gen2):
    gen2_list = list(gen2)
    for x in gen1:
        for y in gen2_list:
            yield (x, y)



def get_sm_model():
    dag = "[V][U][X|V][Y1|X:U][Y2|Y1:U]"
    model = StructuralCausalModel(dag)
    f1_vals = np.array([
        [[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]],
        [[1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]]
    ])

    f2_vals = np.array([
        [[1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
         [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]],
        [[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
         [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]],
    ])

    domains = dict(X=[0,1],Y1=[0,1],Y2=[0,1],V=[0,1])
    domains["U"] = [i for i in range(0,16)]

    domf1 = subdomain(domains, "X", "Y1", "U")
    f1 = MultinomialFactor(domf1, f1_vals, left_vars="Y1")

    domf2 = subdomain(domains, "Y1", "Y2", "U")
    f2 = MultinomialFactor(domf2, f2_vals, left_vars="Y2")
    domfx = subdomain(domains, "X", "V")
    fx = MultinomialFactor(domfx, [1,0,0,1], left_vars="X")

    model.set_factor("X",fx)
    model.set_factor("Y1",f1)
    model.set_factor("Y2",f2)

    model.fill_random_marginals(subdomain(domains, "U", "V"))
    return model
