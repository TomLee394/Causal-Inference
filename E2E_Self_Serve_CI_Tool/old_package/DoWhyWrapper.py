import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import dowhy
from dowhy import CausalModel

class DoWhyWrapper:
    """ Causal inference pipeline using DoWhy following 4 steps:
    
    Step 1: Model the problem as a causal graph
    Step 2: Identify causal effect using properties of the causal graph
    Step 3: Estimate the causal effect
    Step 4: Refute the estimate
    
    """
    def __init__(self, df, treatment, outcome, graph, common_causes=None, instruments=None, verbose=True):
        if df is None:
            print("Input dataframe is None!")
            raise ValueError
            
        self._verbose = verbose
        self._identified_estimand = None
        self._estimator_dict = {}
        self._refuter_dict = {}
        
        # step 1
        self._model = CausalModel(data=df, treatment=treatment, outcome=outcome, graph=graph, common_causes=common_causes, instruments=instruments)
        # step 2
        self._identified_estimand = self._model.identify_effect(proceed_when_unidentifiable=True)
        if self._verbose:
            print("------Step 2: Identifying causal effect------")
            print(self._identified_estimand)


    def estimate_effect(self, method_name="backdoor.linear_regression", **kwargs):  
        # step 3 - estimate the causal effect
       
        estimate = self._model.estimate_effect(self._identified_estimand, method_name=method_name, **kwargs)
        if self._verbose:
            print("------Step 3: Estimating causal effect------")
            print("Using method: " + method_name)
            print("DoWhy Causal Estimate is " + str(estimate.value))
        
        # save the estimator
        self._estimator_dict[method_name] = estimate
        
        return estimate
    
    def refute_estimate(self, estimator_name, method_name, **kwargs):
        # step 4- refuting the estimate
        if estimator_name not in self._estimator_dict:
            print("Estimate method name is not valid!")
            raise ValueError
            
        estimate = self._estimator_dict[estimator_name]
        ref_est = self._model.refute_estimate(self._identified_estimand, estimate, method_name=method_name, **kwargs)
        if self._verbose:
            print("------Step 4: Refuting the estimate------")
            print(ref_est)
            
        # save the refuters
        if estimator_name not in self._refuter_dict:
            self._refuter_dict[estimator_name] = {}
            
        self._refuter_dict[estimator_name][method_name] = ref_est
        
        return ref_est
        
    def get_model(self):
        return self._model
        
    def get_identified_estimand(self):
        return self._identified_estimand
    
    def get_estimator_dict(self):
        return self._estimator_dict
    
    def get_refuter_dict(self):
        return self._refuter_dict