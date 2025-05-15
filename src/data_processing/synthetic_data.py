import os
import pickle
import random
import numpy as np
import networkx as nx
import bnlearn as bn


def generate_random_dag(num_nodes, edge_prob=0.3, seed=None):
    """
    Create a random DAG.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    labels = [f"X{i+1}" for i in range(num_nodes)]
    G = nx.DiGraph()
    G.add_nodes_from(labels)

    for i, u in enumerate(labels):
        for v in labels[i+1:]:
            if random.random() < edge_prob:
                G.add_edge(u, v)

    return G


def inject_missing(df, missing_rate, seed=None):
    """
    Randomly replace values in the df with NaN at a given missing rate.
    """
    # cast to float to allow NaN
    df = df.astype(float).copy()
    rs = np.random.RandomState(seed)
    mask = rs.rand(*df.shape) < missing_rate
    df.values[mask] = np.nan
    return df


def inject_noise(df, noise_level, seed=None):
    """
    Randomly flip values in the df at a given noise level.
    """
    rs = np.random.RandomState(seed)
    df = df.copy()
    for col in df.columns:
        classes = df[col].dropna().unique()
        if len(classes) < 2:
            continue
        
        mask = rs.rand(len(df)) < noise_level
        indices = np.where(mask)[0]
        for i in indices:
            col_index = df.columns.get_loc(col)
            current = df.iat[i, col_index]
            choices = [c for c in classes if c != current]
            if choices:
                df.iat[i, col_index] = rs.choice(choices)
    return df


def generate_synthetic_data(dataset_name, n_samples=1000, noise_levels=None, missing_rates=None):
    """
    Generate synthetic datasets using bnlearn bayesian networks.
    """
    # Load and pickle the model
    model = bn.import_DAG(f"data/synthetic/bifs/{dataset_name}.bif")
    os.makedirs("src/ground_truth_graphs", exist_ok=True)
    with open(f"src/ground_truth_graphs/{dataset_name}.pkl", "wb") as f:
        pickle.dump(model, f)
        
    df = bn.sampling(model, n=n_samples)
    root_dir = os.path.join("data", "synthetic", dataset_name)

    # Save
    base_dir = os.path.join(root_dir, "base")
    os.makedirs(base_dir, exist_ok=True)
    df.to_csv(os.path.join(base_dir, "base.csv"), index=False)

    if noise_levels:
        noise_dir = os.path.join(root_dir, "noise")
        os.makedirs(noise_dir, exist_ok=True)
        for level in noise_levels:
            df_noise = inject_noise(df, level)
            df_noise.to_csv(os.path.join(noise_dir, f"noise_{level}.csv"), index=False)

    if missing_rates:
        missing_dir = os.path.join(root_dir, "missing")
        os.makedirs(missing_dir, exist_ok=True)
        for rate in missing_rates:
            df_missing = inject_missing(df, rate)
            df_missing.to_csv(os.path.join(missing_dir, f"missing_{rate}.csv"), index=False)


 
noise_levels = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
missing_rates = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
n_samples = 3000

# Output folders
root_data = "data/synthetic/"
gt_folder = "src/ground_truth_graphs"
os.makedirs(root_data, exist_ok=True)
os.makedirs(gt_folder, exist_ok=True)

datasets = [
    "asia",
    "alarm",
    "sachs",
    "insurance",
    "child"
]

for dataset in datasets:
    generate_synthetic_data(
        dataset,
        n_samples=n_samples,
        noise_levels=noise_levels,
        missing_rates=missing_rates
    )
