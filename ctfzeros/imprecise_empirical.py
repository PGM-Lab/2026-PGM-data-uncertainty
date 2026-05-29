from typing import Callable

import numpy as np
from bcause.inference.causal import CausalObservationalInference
from bcause.inference.causal.multi import CausalMultiInference
from bcause.inference.inference import Inference
from bcause.inference.probabilistic.datainference import LaplaceInference
from bcause.models.cmodel import StructuralCausalModel
from bcause.util.watch import Watch

from ctfzeros.ie_utils import perturbate_prob
from ctfzeros.model_utils import update_exo_probs, get_missing_states
from ctfzeros.scmgenerator_general.imprecise_generator import get_extreme_points, exact_imprecise_empirical



class LPCC_imprecise_empirical(CausalMultiInference, CausalObservationalInference):
    def __init__(self, model:StructuralCausalModel, data, causal_inf_fn: Callable = None, interval_result=True, num_runs=None, perturbation=0.01):
        self._data = data
        self._prior_model = model
        self._num_runs = num_runs or float("inf")
        self._model = model
        self.num_generated = 0

        Y = [v for v in model.endogenous if len(model.get_edogenous_parents(v)) > 0][0]
        X = sorted([v for v in model.endogenous if len(model.get_edogenous_parents(v)) == 0])
        self.Uy = model.get_exogenous_parents(Y)[0]

        # Only for binary child (so far)
        assert len(model.domains[Y]) == 2

        # Directly calculate Ux from the data (which will not change in this topology)
        infdata = LaplaceInference(data, model.domains)
        self.new_doms = {u: model.domains[u] for u in [model.get_exogenous_parents(x)[0] for x in X]}
        self.new_probs = {model.get_exogenous_parents(x)[0]: infdata.query(x).values for x in X}
        # m = update_exo_probs(model, new_doms, new_probs)

        # Calculate the distribution P(Y|X1,...,Xm)
        y_dist = infdata.query(Y, conditioning=X).reorder(*X, Y)
        y_dist = perturbate_prob(y_dist, perturbation)

        self.plow = y_dist.store_low.values_list
        self.pupp = y_dist.store_up.values_list

        self.Ysize = len(model.domains[Y])
        self.Xsize = np.prod([len(model.domains[x]) for x in X])
        self.Udom = model.domains[self.Uy]
        self.Usize = self.Ysize ** (self.Xsize)

        super().__init__([], causal_inf_fn=causal_inf_fn, interval_result=interval_result, outliers_removal=False)

    def compile(self, *args, **kwargs) -> Inference:

        ext_points = get_extreme_points(self.Udom, self.Ysize, self.Xsize, self.plow, self.pupp)

        models = []

        for pUy in ext_points:
            self.new_doms[self.Uy] = self._prior_model.domains[self.Uy]
            self.new_probs[self.Uy] = list(pUy)
            models.append(update_exo_probs(self._prior_model, self.new_doms, self.new_probs))

            if len(models)>=self._num_runs: break
        self.add_models(models)
        return super().compile()


    def compile_incremental(self, step_runs=1, *args, **kwargs) -> Inference:
        ext_points = get_extreme_points(self.Udom, self.Ysize, self.Xsize, self.plow, self.pupp)
        models = []
        for pUy in ext_points:
            self.new_doms[self.Uy] = self._prior_model.domains[self.Uy]
            self.new_probs[self.Uy] = list(pUy)
            models.append(update_exo_probs(self._prior_model, self.new_doms, self.new_probs))
            if len(models) >= step_runs:
                self.add_models(models)
                models = []
                if len(self._models) >= self._num_runs:
                    return super().compile()
                yield super().compile()

        self.add_models(models)
        models = []
        yield super().compile()





class DCCC_imprecise_empirical(CausalMultiInference, CausalObservationalInference):
    def __init__(self, model:StructuralCausalModel, data, causal_inf_fn: Callable = None, interval_result=True, num_runs=None, perturbation=0.01):
        self._data = data
        self._prior_model = model
        self._num_runs = num_runs or float("inf")
        self._model = model
        self.num_generated = 0
        self._causal_inf = []

        Y = [v for v in model.endogenous if len(model.get_edogenous_parents(v)) > 0][0]
        X = sorted([v for v in model.endogenous if len(model.get_edogenous_parents(v)) == 0])
        self.Uy = model.get_exogenous_parents(Y)[0]

        # Only for binary child (so far)
        assert len(model.domains[Y]) == 2

        # Directly calculate Ux from the data (which will not change in this topology)
        infdata = LaplaceInference(data, model.domains)
        self.new_doms = {u: model.domains[u] for u in [model.get_exogenous_parents(x)[0] for x in X]}
        self.new_probs = {model.get_exogenous_parents(x)[0]: infdata.query(x).values for x in X}
        # m = update_exo_probs(model, new_doms, new_probs)

        # Calculate the distribution P(Y|X1,...,Xm)
        y_dist = infdata.query(Y, conditioning=X).reorder(*X, Y)
        y_dist = perturbate_prob(y_dist, perturbation)

        self.plow = y_dist.store_low.values_list
        self.pupp = y_dist.store_up.values_list

        self.Ysize = len(model.domains[Y])
        self.Xsize = np.prod([len(model.domains[x]) for x in X])
        self.Udom = model.domains[self.Uy]
        self.Usize = self.Ysize ** (self.Xsize)

        exclude_us = set(get_missing_states(model.factors[Y], self.Uy))

        self.gen = exact_imprecise_empirical(self.Ysize, self.Xsize, self.plow, self.pupp, exclude_us)

        super().__init__([], causal_inf_fn=causal_inf_fn, interval_result=interval_result, outliers_removal=False)

    def compile(self, *args, **kwargs) -> Inference:

        models = []

        for domUy, pUy in self.gen:
            self.new_doms[self.Uy] = list(domUy)
            self.new_probs[self.Uy] = list(pUy)
            models.append(update_exo_probs(self._prior_model, self.new_doms, self.new_probs))

            if len(models)>=self._num_runs: break
        self.add_models(models)
        return super().compile()


    def compile_incremental(self, step_runs=1, *args, **kwargs) -> Inference:

        models = []

        for domUy, pUy in self.gen:
            self.new_doms[self.Uy] = list(domUy)
            self.new_probs[self.Uy] = list(pUy)
            models.append(update_exo_probs(self._prior_model, self.new_doms, self.new_probs))

            if len(models) >= step_runs:
                self.add_models(models)
                models = []
                if len(self._models) >= self._num_runs:
                    return self._compile()
                yield self._compile()
                #yield None
        self.add_models(models)
        models = []
        yield self._compile()
        #yield None




    def _compile(self, *args, **kwargs) -> Inference:
        if len(self._models)<1: raise ValueError("Required at least 1 precise model")
        t1 = Watch.time_absolut()

        self._model = self._models[0]

        i = len(self._causal_inf)
        N = len(self.models)


        new_inf = [self._causal_inf_fn(m) for m in self._models[i:N]]
        self._causal_inf = self._causal_inf + new_inf
        #self._causal_inf = [self._causal_inf_fn(m) for m in self._models]

        self._compiled = True

        t2 = Watch.time_absolut()
        self._tcompile += t2-t1
        return self

    #
    # def fit_incremental(self, step_runs=1, *args, **kwargs) -> Inference:
    #
    #     models = []
    #
    #     for domUy, pUy in self.gen:
    #         self.new_doms[self.Uy] = list(domUy)
    #         self.new_probs[self.Uy] = list(pUy)
    #         models.append(update_exo_probs(self._prior_model, self.new_doms, self.new_probs))
    #
    #         if len(models) >= step_runs:
    #             self.add_models(models)
    #             models = []
    #             if len(self._models) >= self._num_runs:
    #                 return
    #             #yield None
    #     self.add_models(models)
    #     models = []
    #     yield super().compile()
    #     #yield None
