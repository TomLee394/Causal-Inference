#this module will contain the main classes for causal discovery and DAG visualization

import pandas as pd
import numpy as np
import torch
from dagma import utils
# from dagma.linear import DagmaLinear
from dagma.nonlinear import DagmaMLP, DagmaNonlinear
from dagma.linear import DagmaLinear
import graphviz as gr

class NonLinear_CausalDiscovery():

    """
    Main class for causal discovery (Non-linear). Instances of this class will learn a DAG from input data using a specified method. Instantiating this class will automatically learn a DAG from the input data using the provided method.

    Linear class to be added later.

    Parameters
    ----------
    data : np.ndarray
        The input data containing X, T and Y as a NumPy array.
    method: str, optional
        Method used for causal discovery. Currently only supports the default algorithm ("dagma")
    col_names: list
        List of column names for the input data.
    mlp_dims: list
        Dimensions of the MLP used for structural equations. Default is [d, int(d/2), 1] where d is the number of columns in the input data.
    """


    def __init__(self, data: np.ndarray, col_names=None ,method: str='dagma', dag=None, pruned_dag=None, mlp_dims=None):
        self.data = data
        self.col_names = col_names
        if mlp_dims is None:
            self.mlp_dims = [self.data.shape[1], int(self.data.shape[1]/2), 1]
        else:
            self.mlp_dims=mlp_dims
        self.method = method
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._validate_data()
        self.dag = self.learn_dag()
        self.pruned_dag = None

    def _validate_data(self):
        if not isinstance(self.data, np.ndarray):
            raise ValueError('Input data must be a NumPy ndarray.')
        if not np.issubdtype(self.data.dtype, np.number):
            raise ValueError('Input data must be numeric.')
        if self.col_names is None:
            raise ValueError('List of column names not provided.')
        if len(self.col_names) != self.data.shape[1]:
            raise ValueError('Number of column names does not match number of columns in data.')

    def learn_dag(self):
        if self.method == 'dagma':
            # define MLP for structural equations with given dimensions(default [d, int(d/2), 1]) and associated model
            d = self.data.shape[1]
            eq_model = DagmaMLP(dims=self.mlp_dims, bias=True, dtype=torch.double).to(self.device)
            model = DagmaNonlinear(eq_model, dtype=torch.double)

            #Move data to GPU if available, otherwise use CPU
            data_tensor = torch.tensor(self.data, dtype=torch.double).to(self.device)

            # fit the model with default L1 and L2 regularization and no default minimum weight threshold
            W_est = model.fit(data_tensor, lambda1=0, lambda2=0, w_threshold=0)
            return W_est  # Move the result back to CPU and convert to NumPy array
        else:
            raise ValueError('Invalid method. Currently only supports "dagma".')
        
    def prune_dag(self, prune_method='percent', prune_threshold=0.02):
        if prune_method == 'percent':
            pruned_dag = np.where(self.dag <= self.dag.sum()*prune_threshold, 0, self.dag)
        if prune_method == 'absolute':
            pruned_dag = np.where(self.dag <= prune_threshold, 0, self.dag)
        return pruned_dag
    
    def display_dag(self, prune_method='percent', prune_threshold=0.02, return_pruned_W=False):
        if prune_method in ['percent', 'absolute']:
            adj_matrix = self.prune_dag(prune_method=prune_method, prune_threshold=prune_threshold)
        else:
            raise ValueError('Invalid pruning method. Supports either "percent" or "absolute".')

        dot = gr.Digraph()

        for i in range(adj_matrix.shape[0]):
            for j in range(adj_matrix.shape[1]):
                if adj_matrix[i,j] != 0:
                    weight = adj_matrix[i, j]
                    dot.edge(self.col_names[i], self.col_names[j], label=f'{weight:.3f}')

        if return_pruned_W:
            self.pruned_dag = adj_matrix
            return dot
        else:
            return dot

class Linear_CausalDiscovery():

    """
    Main class for causal discovery (Linear). Instances of this class will learn a DAG from input data using a specified method. Instantiating this class will automatically learn a DAG from the input data using the provided method.

    Parameters
    ----------
    data : np.ndarray
        The input data containing X, T and Y as a NumPy array.
    method: str, optional
        Method used for causal discovery. Currently only supports the default algorithm ("dagma")
    col_names: list
        List of column names for the input data.
    """

    def __init__(self, data: np.ndarray, col_names=None ,method: str='dagma', loss_type='l2', dag=None, pruned_dag=None):
        self.data = data
        self.col_names = col_names
        self.method = method
        self.loss_type = loss_type
        self._validate_data()
        self.dag = self.learn_dag()
        self.pruned_dag = None 

    def _validate_data(self):
        if not isinstance(self.data, np.ndarray):
            raise ValueError('Input data must be a NumPy ndarray.')
        if not np.issubdtype(self.data.dtype, np.number):
            raise ValueError('Input data must be numeric.')
        if self.col_names is None:
            raise ValueError('List of column names not provided.')
        if len(self.col_names) != self.data.shape[1]:
            raise ValueError('Number of column names does not match number of columns in data.')
    
    def learn_dag(self):
        if self.method == 'dagma':
            # Define linear model for structural equations
            model = DagmaLinear(loss_type=self.loss_type)
            W_est = model.fit(self.data, lambda1=0, w_threshold=0)
            return W_est  # Move the result back to CPU and convert to NumPy array
        else:
            raise ValueError('Invalid method. Currently only supports "dagma".')
    
    def prune_dag(self, prune_method='percent', prune_threshold=0.02):
        if prune_method == 'percent':
            pruned_dag = np.where(self.dag <= self.dag.sum()*prune_threshold, 0, self.dag)
        if prune_method == 'absolute':
            pruned_dag = np.where(self.dag <= prune_threshold, 0, self.dag)
        return pruned_dag
    
    def display_dag(self, prune_method='percent', prune_threshold=0.02, return_pruned_W=False):
        if prune_method in ['percent', 'absolute']:
            adj_matrix = self.prune_dag(prune_method=prune_method, prune_threshold=prune_threshold)
        else:
            raise ValueError('Invalid pruning method. Supports either "percent" or "absolute".')

        dot = gr.Digraph()

        for i in range(adj_matrix.shape[0]):
            for j in range(adj_matrix.shape[1]):
                if adj_matrix[i,j] != 0:
                    weight = adj_matrix[i, j]
                    dot.edge(self.col_names[i], self.col_names[j], label=f'{weight:.3f}')

        if return_pruned_W:
            self.pruned_dag = adj_matrix
            return dot
        else:
            return dot

def manual_display_dag(W, col_names):

    """
    Helper function for manually displaying a DAG. This function takes an adjacency matrix and column names as input and returns a graphviz object for visualization. One intended use is for displaying sample-split-and-averaged (cross-fitted) DAGs.

    Parameters
    ----------

    W : np.ndarray
        Adjacency matrix (i,j) representing the DAG, where W[i,j] is the weight of the edge from node i to node j.
    
    col_names : list
        List of column names for the input data.
    """

    dot = gr.Digraph(engine='dot')

    for name in col_names:
        dot.node(name)

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            if W[i,j] != 0:
                weight = W[i, j]
                dot.edge(col_names[i], col_names[j], label=f'{weight:.3f}')

    return dot
    