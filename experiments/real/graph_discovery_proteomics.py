import os
import numpy as np
import pandas as pd
from pathlib import Path
from src.causal_discovery.discovery import run_grasp
import pickle

def get_clustered_data(task="metab", all_organs=False):
    """
    Get all clustered files from the processed directory.
    """
    df_dict = {}
    folder = Path(f'data/real/{task}/processed/imputed')
    files = sorted(folder.glob("*.csv"))
    for file in files:
        if file and file.exists() and not all_organs and file.stem != "all_organs_imputed":
            df_dict[file.stem] = pd.read_csv(file)
        elif file and file.exists() and all_organs and file.stem == "all_organs_imputed":
            df_dict[file.stem] = pd.read_csv(file)
    return df_dict

def get_adj_mats(df_dict, task="metab"):
    """
    Gets the adjacency matrices for each organ pair, running grasp if they don't exist already.
    """
    adj_mats = {}
    for key, df in df_dict.items():
        save_path = Path("src/learned_graphs/real")
        save_path.mkdir(parents=True, exist_ok=True)
        organs = key.split("_")[:2]
        save_name = f"{task}_{organs[0]}_{organs[1]}"
        
        if os.path.isfile(save_path / f"{save_name}.pkl"):
            print(f"{save_name} already exists; skipping GRASP.")
            with open(save_path / f"{save_name}.pkl", "rb") as f:
                adj_mat = pickle.load(f)
        else:
            print(f"Running GRASP for {save_name}...")
            df = df.set_index(df.columns[0])
            adj_mat = run_grasp(df, verbose=True)
            
            with open(save_path / f"{save_name}.pkl", "wb") as f:
                pickle.dump(adj_mat, f)
            
        adj_mats[save_name] = adj_mat
    return adj_mats

def combine_adj_mats(adj_mats, task="metab"):
    """
    Combine adjacency matrices by unioning their edges
    """
    all_nodes = set()
    for adj_mat in adj_mats.values():
        all_nodes.update(adj_mat.columns)
    all_nodes = list(all_nodes)
    
    # Reindex each adjacency matrix to include all nodes
    for key, adj_mat in adj_mats.items():
        adj_mat = adj_mat.reindex(index=all_nodes, columns=all_nodes, fill_value=0)
        adj_mats[key] = adj_mat
    
    # Combine by taking max across all adjacency matrices
    combined_adj_mat = pd.DataFrame(0, index=all_nodes, columns=all_nodes)
    for adj_mat in adj_mats.values():
        combined_adj_mat = combined_adj_mat.combine(adj_mat, np.maximum)
    
    # Save combined adjacency matrix
    with open(f"src/learned_graphs/real/{task}_combined.pkl", "wb") as f:
        pickle.dump(combined_adj_mat, f)
    print(f"Combined adjacency matrix saved to src/learned_graphs/real/{task}_combined.pkl")
    return combined_adj_mat

def get_combined_adj_mat(task="metab"):
    """
    Get the combined adjacency matrix from the saved file.
    """
    if not os.path.isfile(f"src/learned_graphs/real/{task}_combined.pkl"):
        print("Combined adjacency matrix not found. Running GRASP...")
        df_dict = get_clustered_data(task=task)
        adj_mats = get_adj_mats(df_dict, task=task)
        combined_adj_mat = combine_adj_mats(adj_mats, task=task)
    else:
        with open(f"src/learned_graphs/real/{task}_combined.pkl", "rb") as f:
            combined_adj_mat = pickle.load(f)
    
    if not os.path.isfile("src/learned_graphs/real/all_organs.pkl"):
        print("All organs graph not found. Running GRASP...")
        df_dict = get_clustered_data(all_organs=True)
        adj_mats = get_adj_mats(df_dict)
        all_organs_graph = adj_mats["all_organs"]
        with open("src/learned_graphs/real/all_organs.pkl", "wb") as f:
            pickle.dump(all_organs_graph, f)
    else:
        with open("src/learned_graphs/real/all_organs.pkl", "rb") as f:
            all_organs_graph = pickle.load(f)
    return combined_adj_mat, all_organs_graph
