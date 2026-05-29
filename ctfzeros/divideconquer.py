from itertools import product
from typing import Callable

import networkx as nx
import numpy as np
from bcause.inference.causal import CausalObservationalInference
from bcause.inference.causal.multi import CausalMultiInference
from bcause.inference.inference import Inference
from bcause.inference.probabilistic.datainference import LaplaceInference
from bcause.models.cmodel import StructuralCausalModel
from bcause.util.domainutils import subdomain
from bcause.util.runningutils import get_logger
from bcause.util.watch import Watch

from ctfzeros.model_utils import update_exo_probs, get_missing_states, lazy_cartesian_product
from ctfzeros.scmgenerator.generators import scm_solution_generator
from ctfzeros.scmgenerator_general.general_solution_generator import scm_general_solution_generator
from semimarkovian.semimarkovian_example import exhaustive_solution_generator_M1


# Define the logger
#log_format = '%(asctime)s|%(levelname)s|%(filename)s: %(message)s'
#log = get_logger(__name__, fmt=log_format)
#logging.getLogger("bcause").setLevel(logging.INFO)

#log.propagate = 0

class DCCC_inverted_tree(CausalMultiInference, CausalObservationalInference):
    def __init__(self, model:StructuralCausalModel, data, causal_inf_fn: Callable = None, interval_result=True, num_runs=None, new_method=True):
        self._data = data
        self._prior_model = model
        self._num_runs = num_runs or float("inf")
        self._model = model
        self.num_generated = 0
        self.new_method = new_method

        Y = [v for v in model.endogenous if len(model.get_edogenous_parents(v)) > 0][0]
        X = sorted([v for v in model.endogenous if len(model.get_edogenous_parents(v)) == 0])

        # Directly calculate Ux from the data (which will not change in this topology)
        infdata = LaplaceInference(data, model.domains)
        self.new_doms = {u: model.domains[u] for u in [model.get_exogenous_parents(x)[0] for x in X]}
        self.new_probs = {model.get_exogenous_parents(x)[0]: infdata.query(x).values for x in X}
        # m = update_exo_probs(model, new_doms, new_probs)

        # Calculate the distribution P(Y|X1,...,Xm)
        y_dist = infdata.query(Y, conditioning=X).reorder(*X,Y).values
        y_dist = np.array(y_dist).reshape((len(y_dist), 1))


        self.Uy = self.model.get_exogenous_parents(Y)[0]

        if not new_method:
            self.scm_generator = scm_solution_generator(
               n_parents=len(X),
               y_dist=y_dist,
               exclude_us=set(get_missing_states(model.factors[Y], self.Uy)),
               solver= len(X) < 3
            )

        else:


            child_domain_size = len(model.domains[Y])
            probabilities = y_dist
            parent_domain_size = np.prod([len(model.domains[x]) for x in X])
            exclude_us = set(get_missing_states(model.factors[Y], self.Uy))

            self.scm_generator = scm_general_solution_generator(child_domain_size, parent_domain_size,
                                                                   child_dist=probabilities, exclude_us=exclude_us, seed=0)



        super().__init__([], causal_inf_fn=causal_inf_fn, interval_result=interval_result, outliers_removal=False)

    def compile(self, *args, **kwargs) -> Inference:
        models = []
        for domUy, pUy, _ in self.scm_generator:
            self.new_doms[self.Uy] = list(domUy)
            self.new_probs[self.Uy] = list(pUy)
            models.append(update_exo_probs(self._prior_model, self.new_doms, self.new_probs))
            if len(models)>=self._num_runs: break
        self.add_models(models)
        return super().compile()



    def compile_incremental(self, step_runs=1, *args, **kwargs) -> Inference:

       if self.new_method:

            models = []
            for domUy, pUy in self.scm_generator:
                #self.num_generated += num_generated

                domUy = [int(s) for s in domUy] #{v:[int(s) for s in d] for v,d in domUy.items()}
                self.num_generated += 1
                self.new_doms[self.Uy] = list(domUy)
                self.new_probs[self.Uy] = list(pUy)
                models.append(update_exo_probs(self._prior_model, self.new_doms, self.new_probs))
                if len(models)>=step_runs:
                    self.add_models(models)
                    models = []
                    if len(self._models)>=self._num_runs:
                        return super().compile()
                    yield super().compile()

            self.add_models(models)
            models = []
            yield super().compile()

       else:

        #def compile_incremental_old(self, step_runs=1, *args, **kwargs) -> Inference:
            models = []
            for domUy, pUy, num_generated in self.scm_generator:
                self.num_generated += num_generated
                self.new_doms[self.Uy] = list(domUy)
                self.new_probs[self.Uy] = list(pUy)
                models.append(update_exo_probs(self._prior_model, self.new_doms, self.new_probs))
                if len(models)>=step_runs:
                    self.add_models(models)
                    models = []
                    if len(self._models)>=self._num_runs:
                        return super().compile()
                    yield super().compile()

            self.add_models(models)
            models = []
            yield super().compile()





class DCCC_semimarkovian(CausalMultiInference, CausalObservationalInference):
    def __init__(self, model:StructuralCausalModel, data, data_inter = None, causal_inf_fn: Callable = None, interval_result=True, num_runs=None, new_method=True):
        self._data = data
        self._data_inter = data_inter

        self._prior_model = model
        self._num_runs = num_runs or float("inf")
        self._model = model
        self.num_generated = 0
        self.new_method = new_method

        infdata = LaplaceInference(data[model.endogenous], domains=subdomain(model.domains, *model.endogenous))


        # Directly calculate Ux from the data (which will not change in this topology)
        X = [v for v in model.endogenous if len(model.get_edogenous_parents(v)) == 0][0]
        Y1 = model.get_edogenous_children(X)[0]
        Y2 = model.get_edogenous_children(Y1)[0]
        self.Uy = model.get_exogenous_parents(Y1)[0]

        self.new_doms = {u: model.domains[u] for u in [model.get_exogenous_parents(X)[0] for x in [X]]}
        self.new_probs = {model.get_exogenous_parents(x)[0]: infdata.query(x).values for x in [X]}

        # Calculate the empirical
        pemp = infdata.query([Y1, Y2], conditioning=X)
        y_dist = pemp.values_array([X, Y1, Y2]).flatten()


        if data_inter is not None:
            infdata_inter = LaplaceInference(data_inter[model.endogenous],
                                             domains=subdomain(model.domains, *model.endogenous))
            # Calculate the empirical interventional
            pinter = infdata_inter.query(Y2, conditioning=Y1)
            y2_do_y1 = pinter.values_array([Y1, Y2]).flatten()
            #print(f"dist_y1y2_given_x: {y_dist}")
            #print(f"dist_y2_do_y1{y2_do_y1}")
            self.scm_generator = exhaustive_solution_generator_M1(y_dist, y2_do_y1)
        else:
            self.scm_generator = exhaustive_solution_generator_M1(y_dist)

        super().__init__([], causal_inf_fn=causal_inf_fn, interval_result=interval_result, outliers_removal=False)

    def compile(self, *args, **kwargs) -> Inference:

        models = []
        for domUy, pUy in self.scm_generator:
            self.new_doms[self.Uy] = list(domUy)
            self.new_probs[self.Uy] = list(pUy)
            #print(domUy, pUy)
            models.append(update_exo_probs(self._prior_model, self.new_doms, self.new_probs))
            if len(models)>=self._num_runs: break
        self.add_models(models)
        return super().compile()




    def compile_incremental(self, step_runs=1, *args, **kwargs) -> Inference:

        models = []
        for domUy, pUy in self.scm_generator:

            self.num_generated += 1

            self.new_doms[self.Uy] = list(domUy)
            self.new_probs[self.Uy] = list(pUy)
            # print(domUy, pUy)
            models.append(update_exo_probs(self._prior_model, self.new_doms, self.new_probs))


            if len(models)>=step_runs:
                self.add_models(models)
                models = []
                out = super().compile()
                if len(self._models)>=self._num_runs:
                    return out
                yield out

        self.add_models(models)
        models = []
        yield super().compile()


    # def compile_incremental(self, step_runs=1, *args, **kwargs) -> Inference:
    #
    #     tlearn = 0
    #
    #     models = []
    #     #for domUy, pUy in self.scm_generator:
    #     #self.scm_generator.ne
    #     while True:
    #         try:
    #             t1 = Watch.get_time()
    #             domUy, pUy = next(self.scm_generator)
    #
    #             self.num_generated += 1
    #
    #             self.new_doms[self.Uy] = list(domUy)
    #             self.new_probs[self.Uy] = list(pUy)
    #             # print(domUy, pUy)
    #             models.append(update_exo_probs(self._prior_model, self.new_doms, self.new_probs))
    #             t2 = Watch.get_time()
    #
    #             tlearn = tlearn + t2-t1
    #
    #             print(tlearn)
    #
    #             if len(models) >= step_runs:
    #                 self.add_models(models)
    #                 models = []
    #                 if len(self._models) >= self._num_runs:
    #                     return super().compile()
    #                 yield super().compile()
    #
    #
    #         except StopIteration:
    #             break
    #
    #     self.add_models(models)
    #     models = []
    #     yield super().compile()

class DCCC_semimarkovian_experimental(CausalMultiInference, CausalObservationalInference):
    def __init__(self, model:StructuralCausalModel, dataX, data_interX, data_interY1, causal_inf_fn: Callable = None, interval_result=True, num_runs=None, new_method=True):

        self._prior_model = model
        self._num_runs = num_runs or float("inf")
        self._model = model
        self.num_generated = 0
        self.new_method = new_method

        # Directly calculate Ux from the data (which will not change in this topology)
        X = [v for v in model.endogenous if len(model.get_edogenous_parents(v)) == 0][0]
        Y1 = model.get_edogenous_children(X)[0]
        Y2 = model.get_edogenous_children(Y1)[0]

        infdata = LaplaceInference(dataX, domains=subdomain(model.domains, X))

        self.new_doms = {u: model.domains[u] for u in [model.get_exogenous_parents(X)[0] for x in [X]]}
        self.new_probs = {model.get_exogenous_parents(x)[0]: infdata.query(x).values for x in [X]}

        infdata_interX = LaplaceInference(data_interX, domains=subdomain(model.domains, Y1, X))
        infdata_interY1 = LaplaceInference(data_interY1, domains=subdomain(model.domains, Y2, Y1))

        pinter = infdata_interX.query(Y1, conditioning=X)
        dist_y1_given_x = pinter.values_array([X, Y1]).flatten()

        pinter = infdata_interY1.query(Y2, conditioning=Y1)
        dist_y2_do_y1 = pinter.values_array([Y1, Y2]).flatten()

        self.Uy = model.get_exogenous_parents(Y1)[0]

        exp_dist = list(dist_y1_given_x) + list(dist_y2_do_y1)
        self.scm_generator = scm_general_solution_generator(2, 4, child_dist=exp_dist, exhaustive=True)

        super().__init__([], causal_inf_fn=causal_inf_fn, interval_result=interval_result, outliers_removal=False)

    def compile(self, *args, **kwargs) -> Inference:

        models = []
        for domUy, pUy in self.scm_generator:
            self.new_doms[self.Uy] = list(domUy)
            self.new_probs[self.Uy] = list(pUy)
            #print(domUy, pUy)
            models.append(update_exo_probs(self._prior_model, self.new_doms, self.new_probs))
            if len(models)>=self._num_runs: break
        self.add_models(models)
        return super().compile()

    def compile_incremental(self, step_runs=1, *args, **kwargs) -> Inference:

        models = []
        for domUy, pUy in self.scm_generator:

            self.num_generated += 1

            self.new_doms[self.Uy] = list(domUy)
            self.new_probs[self.Uy] = list(pUy)
            # print(domUy, pUy)
            models.append(update_exo_probs(self._prior_model, self.new_doms, self.new_probs))

            if len(models)>=step_runs:
                self.add_models(models)
                models = []
                if len(self._models)>=self._num_runs:
                    return super().compile()
                yield super().compile()

        self.add_models(models)
        models = []
        yield super().compile()

class DCCC_markovapprox_endomerge(CausalMultiInference, CausalObservationalInference):
    def __init__(self, model:StructuralCausalModel, data, causal_inf_fn: Callable = None, interval_result=True, num_runs=None):


        self.input_model = model
        self.X = [v for v in model.endogenous if len(model.get_edogenous_parents(v)) == 0][0]
        self.Y1 = model.get_edogenous_children(self.X)[0]
        self.Y2 = model.get_edogenous_children(self.Y1)[0]


        # merge the model
        model, data = self.get_merged_markovian(model, data)

        self._prior_model = model
        self._num_runs = num_runs or float("inf")
        self._model = model
        self.num_generated = 0

        for v in model.endogenous:
            assert len(model.get_exogenous_parents(v)) == 1
            assert len(model.get_edogenous_parents(v)) <= 1
        assert len(model.endogenous) == 2

        infdata = LaplaceInference(data[model.endogenous], domains=subdomain(model.domains, *model.endogenous))

        X = [v for v in model.endogenous if len(model.get_edogenous_parents(v)) == 0][0]
        Y = [v for v in model.endogenous if len(model.get_edogenous_parents(v)) == 1][0]

        # Directly calculate Ux from the data (which will not change in this topology)
        self.new_doms = {u: model.domains[u] for u in [model.get_exogenous_parents(X)[0] for x in [X]]}
        self.new_probs = {model.get_exogenous_parents(x)[0]: infdata.query(x).values for x in [X]}

        # Calculate the empirical
        pemp = infdata.query(Y, conditioning=X)
        y_dist = pemp.values_array([X, Y]).flatten()

        self.Uy = model.get_exogenous_parents(Y)[0]

        self.scm_generator = exhaustive_solution_generator_M1(y_dist)

        super().__init__([], causal_inf_fn=causal_inf_fn, interval_result=interval_result, outliers_removal=False)

    def compile(self, *args, **kwargs) -> Inference:


        models = []
        for domUy, pUy in self.scm_generator:
            self.new_doms[self.Uy] = list(domUy)
            self.new_probs[self.Uy] = list(pUy)
            models.append(update_exo_probs(self._prior_model, self.new_doms, self.new_probs))

        self._model = models[0]
        self.add_models(models)

        return super().compile()

    def prob_sufficiency(self, cause, effect):

        if effect == self.Y1:
            restrict_values = ["10", "11"]  # [1*]
        elif effect == self.Y2:
            restrict_values = ["01", "11"]  # [*1]
        else:
            raise ValueError("Wrong effecet variable")

        aux_ires = self._interval_result
        self.set_interval_result(False)
        res = self.counterfactual_query("Y1Y2", do=dict(X=1), evidence=dict(X=0))
        res = [p.R(Y1Y2_1=restrict_values).marginalize("Y1Y2_1").values[0] for p in res]
        self.set_interval_result(aux_ires)


        return [min(res), max(res)] if self._interval_result else res



    @staticmethod
    def get_merged_markovian(model, data):
        endo_graph = nx.DiGraph()
        dom = dict()

        c = sorted(model.endo_ccomponents[1])

        for c in model.endo_ccomponents:
            c = sorted(c)
            node = "".join(c)

            pa = [x for x in model.get_parents(*c) if model.is_endogenous(x) and x not in c]

            # This code is for the models in which the confounder is at the end
            if len(c) > 1:
                ch = [x for x in model.get_children(*c) if model.is_endogenous(x) and x not in c]
                assert len(ch) == 0
            if len(pa) > 0 and (pa[0], node) not in endo_graph.edges:
                endo_graph.add_edge(pa[0], node)

            if len(c) == 1:
                dom[node] = model.domains[node]
            else:
                new_vals = list(product(*[model.domains[v] for v in c]))
                # map = dict(zip(new_vals, range(0,len(new_vals))))

                map = {val: "".join([str(v) for v in val]) for val in new_vals}

                data[node] = data.apply(lambda t: map[tuple([t[v] for v in c])], axis=1)
                data = data.drop(columns=c)

                # dom[node] = list(range(np.prod([len(model.domains[v]) for v in c])))
                dom[node] = list(map.values())

        return StructuralCausalModel.markovian_model(endo_graph, dom), data


class DCCC_markovapprox_exosplit(CausalMultiInference, CausalObservationalInference):
    def __init__(self, model:StructuralCausalModel, data, data_inter = None, causal_inf_fn: Callable = None, interval_result=True, num_runs=None, new_method=True):


        self._data = data
        self._data_inter = data_inter
        model = model.to_markovian()
        self._prior_model = model
        self._num_runs = num_runs or float("inf")
        self._model = model
        self.num_generated = 0
        self.new_method = new_method

        for v in model.endogenous:
            assert len(model.get_exogenous_parents(v)) == 1
            assert len(model.get_edogenous_parents(v)) <= 1

        infdata = LaplaceInference(data[model.endogenous], domains=subdomain(model.domains, *model.endogenous))

        X = [v for v in model.endogenous if len(model.get_edogenous_parents(v)) == 0][0]
        Y = [v for v in model.endogenous if len(model.get_edogenous_parents(v)) == 1]

        # Directly calculate Ux from the data (which will not change in this topology)
        self.new_doms = {u: model.domains[u] for u in [model.get_exogenous_parents(X)[0] for x in [X]]}
        self.new_probs = {model.get_exogenous_parents(x)[0]: infdata.query(x).values for x in [X]}

        self.U_target = [model.get_exogenous_parents(y)[0] for y in Y]

        self.scm_generators = dict()

        # Calculate the empirical

        for u in self.U_target:
            y = model.get_edogenous_children(u)[0]
            PAy = model.get_edogenous_parents(y)[0]

            pemp = infdata.query(y, conditioning=PAy)
            probabilities = pemp.values_array([PAy, y]).flatten()

            child_domain_size = len(model.domains[y])
            parent_domain_size = len(model.domains[PAy])
            exclude_us = set(get_missing_states(model.factors[y], u))

            self.scm_generators[u] = scm_general_solution_generator(child_domain_size, parent_domain_size,
                                                               child_dist=probabilities, exclude_us=exclude_us, seed=0)

        super().__init__([], causal_inf_fn=causal_inf_fn, interval_result=interval_result, outliers_removal=False)

    def compile(self, *args, **kwargs) -> Inference:

        assert len(self.U_target) == 2
        u0, u1 = self.U_target

        models = []

        for t in lazy_cartesian_product(self.scm_generators[u0], self.scm_generators[u1]):
            self.new_doms[u0] = list(t[0][0])
            self.new_probs[u0] = list(t[0][1])
            self.new_doms[u1] = list(t[1][0])
            self.new_probs[u1] = list(t[1][1])
            models.append(update_exo_probs(self._prior_model, self.new_doms, self.new_probs))

        self.add_models(models)
        return super().compile()









'''
    def compile_incremental(self, step_runs=1, *args, **kwargs) -> Inference:

       if self.new_method:

            models = []
            for domUy, pUy in self.scm_generator:
                #self.num_generated += num_generated

                domUy = [int(s) for s in domUy] #{v:[int(s) for s in d] for v,d in domUy.items()}
                self.num_generated += 1
                self.new_doms[self.Uy] = list(domUy)
                self.new_probs[self.Uy] = list(pUy)
                models.append(update_exo_probs(self._prior_model, self.new_doms, self.new_probs))
                if len(models)>=step_runs:
                    self.add_models(models)
                    models = []
                    if len(self._models)>=self._num_runs:
                        return super().compile()
                    yield super().compile()

            self.add_models(models)
            models = []
            yield super().compile()

       else:

        #def compile_incremental_old(self, step_runs=1, *args, **kwargs) -> Inference:
            models = []
            for domUy, pUy, num_generated in self.scm_generator:
                self.num_generated += num_generated
                self.new_doms[self.Uy] = list(domUy)
                self.new_probs[self.Uy] = list(pUy)
                models.append(update_exo_probs(self._prior_model, self.new_doms, self.new_probs))
                if len(models)>=step_runs:
                    self.add_models(models)
                    models = []
                    if len(self._models)>=self._num_runs:
                        return super().compile()
                    yield super().compile()

            self.add_models(models)
            models = []
            yield super().compile()

'''