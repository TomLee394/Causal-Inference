
import pandas as pd
import numpy as np
import graphviz as gr
from matplotlib import style
import seaborn as sns
from matplotlib import pyplot as plt
from dowhy import CausalModel

class CausalIdentifcation():

    """
    Main class for causal identification. Instances of this class will traverse the input DAG and identify valid adjustment sets (set of confounders to control for) that will create conditional independence between T and Y. 

    Parameters
    ----------
    data : pandas.DataFrame
        The input data as a pandas dataframe containing X, T, and Y.
    treatment : str
        The name of the treatment variable.
    outcome : str
        The name of the outcome variable.
    dag : str
        The learned DAG in DOT format.
      """


    def __init__(self, data, treatment: str, outcome: str, dag, model=None, identified_estimand=None):
        
        self.data = data
        self.treatment = treatment
        self.outcome = outcome
        self.dag = dag
        self.model, self.identified_estimand = self._identify_effect()

    def _identify_effect(self):
        """
        Builds the causal model and attempts to identify valid adjustment sets through graph traversal using do-calculus from Pearl's (2012) framework of causal inference. Returns the identified estimand and stores it as an instance variable.
        """

        model = CausalModel(
            data=self.data,
            treatment=self.treatment,
            outcome=self.outcome,
            proceed_when_unidentifiable=True,
            graph=self.dag.source  # Pass the graph in DOT format
        )
        identified_estimand = model.identify_effect()
        return model, identified_estimand
    
    def view_model(self):
        """
        Displays the causal model.
        """

        self.model.view_model()

    def get_adj_set(self, criterion='backdoor'):
        """
        Returns the valid adjustment sets.

        Parameters
        ----------
        criterion : str, optional
            The criterion to use for identifying valid adjustment sets. Options are 'backdoor', 'instrumental' and 'frontdoor'. Default is 'backdoor'.
        """
        
        adj_set = getattr(self.identified_estimand, f"{criterion}_variables")
        return adj_set