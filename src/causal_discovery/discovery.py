import pandas as pd
import numpy as np
from causallearn.search.ScoreBased.GES import ges
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.PermutationBased.GRaSP import grasp
import io
from contextlib import redirect_stdout

def run_ges(data):
    """
    Run GES algorithm to learn causal graph
    """
    numpy_data = data.values.astype(float)
    
    # To avoid singular covariance issues
    numpy_data += np.random.normal(loc=0.0, scale=1e-6, size=numpy_data.shape)

    cg = ges(numpy_data)

    adj_matrix = cg['G'].graph.T.astype(int)
    return pd.DataFrame(
        adj_matrix,
        columns=data.columns,
        index=data.columns
    )
    
def run_grasp(data, depth=1, verbose=False):
    """
    Run GRaSP algorithm for causal discovery.
    """
    numpy_data = data.values.astype(float)
    
    # Avoid singular covariance issues
    numpy_data += np.random.normal(loc=0.0, scale=1e-6, size=numpy_data.shape)
    
    if not verbose:
        buf = io.StringIO()
        with redirect_stdout(buf):
            G = grasp(numpy_data)
    else:
        G = grasp(numpy_data, depth=depth)

    # Convert to adjacency matrix with correct format
    adj_matrix = [[1 if G.graph[i,j] == -1 else 0 for j in range(len(G.graph[0]))] for i in range(len(G.graph))]
    

    return (pd.DataFrame(adj_matrix, columns=data.columns, index=data.columns).abs() > 0.5).astype(int)

def run_pc(data, verbose=False):
    """
    Run PC algorithm to learn causal graph
    """  
    numpy_data = data.values.astype(float)
    
    # Run PC
    cg = pc(numpy_data, verbose=verbose, show_progress=False)
    
    # Convert to adjacency matrix
    adj_matrix = cg.G.graph.T.astype(int)
    return pd.DataFrame(adj_matrix, columns=data.columns, index=data.columns)
