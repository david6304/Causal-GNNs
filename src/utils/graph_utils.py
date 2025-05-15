import networkx as nx
import random
import pickle as pkl
from pathlib import Path
from src.utils.config_loader import load_config


def topological_sort(adj_matrix):
    G = nx.DiGraph(adj_matrix)
    return list(nx.topological_sort(G))

def remove_bidirectional_edges(graph):
    """
    Randomly remove one direction of bidirectional edges in the graph. Needed for grasp
    """
    graph = graph.copy()
    bidir_pairs = set()
    
    # Find all bidirectional edges
    for src in graph.columns:
        for tgt in graph.index:
            if graph.loc[src, tgt] != 0 and graph.loc[tgt, src] != 0 and (tgt, src) not in bidir_pairs:
                bidir_pairs.add((src, tgt))
    
    # Remove one direction
    for src, tgt in bidir_pairs:
        if random.random() < 0.5:
            graph.loc[src, tgt] = 0
        else:
            graph.loc[tgt, src] = 0
    
    return graph

def get_dag(dataset):
    """
    Load the ground truth DAG from its pickle file.
    """
    cfg = load_config()
    gt_folder = Path(cfg.get('ground_truth', 'src/ground_truth_graphs'))

    file_path = gt_folder / f"{dataset}.pkl"

    with open(file_path, 'rb') as f:
        obj = pkl.load(f)
        
    model = obj['model']
    graph = obj['adjmat']

    adj_mat = graph.astype(int)
    return model, adj_mat
