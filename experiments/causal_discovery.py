import time
import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from src.data_processing.data_loader import load_synthetic_data
from src.utils.graph_utils import get_dag
from src.causal_discovery.discovery import run_ges, run_grasp, run_pc
from src.evaluation.metrics import structural_hamming_distance
from sklearn.model_selection import KFold


def evaluate_methods():
    results = []
    best_shd = {ds: float('inf') for ds in datasets}
    best_graphs = {}
    for ds in datasets:
        print(f"Dataset: {ds}")
        _, adj_gt = get_dag(dataset=ds)
        for variant in variants:
            print(f"Variant: {variant}")
            if variant == 'base':
                df = load_synthetic_data(dataset=ds, variant=variant)
                df_dict = {0: df}
            else:
                df_dict = load_synthetic_data(dataset=ds, variant=variant)

            order = sorted(df_dict.keys())
            for lvl in order:
                df = df_dict[lvl]
                print(f"Level: {lvl}")

                for method in methods:
                    print(f"Running {method.upper()} on {n_splits} random splits")
                    directed_list = []
                    extra_edges_list = []
                    missing_edges_list = []
                    time_list = []
                    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
                    for i_split, (i_train, _) in enumerate(kf.split(df), start=1):
                        print(f"Split {i_split}/{n_splits}")
                        split_df = df.iloc[i_train]
                        
                        if variant == 'missing':
                            # Fill na values with mode
                            split_df = split_df.fillna(split_df.mode().iloc[0])
                            
                        start = time.time()
                        if method == 'pc':
                            learned_adj = run_pc(split_df)
                        elif method == 'grasp':
                            learned_adj = run_grasp(split_df)
                        elif method == 'ges':
                            learned_adj = run_ges(split_df)
                            
                        elapsed = time.time() - start
                        shd_dict = structural_hamming_distance(adj_gt, learned_adj)
                        directed = shd_dict['directed_shd']
                        extra = shd_dict['extra_edges']
                        missing = shd_dict['missing_edges']
                        
                        # Store best result from grasp
                        if variant == 'base' and method == 'grasp' and directed < best_shd[ds]:
                            best_shd[ds] = directed
                            best_graphs[ds] = learned_adj
                            
                        print(f"Directed SHD={directed}, Extra={extra}, Missing={missing}, time={elapsed:.2f}s")
                        directed_list.append(directed)
                        extra_edges_list.append(extra)
                        missing_edges_list.append(missing)
                        time_list.append(elapsed)
                        
                    avg_directed = np.mean(directed_list)
                    avg_extra = np.mean(extra_edges_list)
                    avg_missing = np.mean(missing_edges_list)
                    avg_time = np.mean(time_list)
                    std_directed = np.std(directed_list)
                    std_extra = np.std(extra_edges_list)
                    std_missing = np.std(missing_edges_list)
                    std_time = np.std(time_list)
                    
                    print(f"Avg Directed SHD={avg_directed:.2f}, Avg Extra={avg_extra:.2f}, Avg Missing={avg_missing:.2f}, Avg time={avg_time:.2f}s")
                    print(f"Std Directed SHD={std_directed:.2f}, Std Extra={std_extra:.2f}, Std Missing={std_missing:.2f}, Std time={std_time:.2f}s")
                    results.append({
                        'dataset': ds,
                        'variant': variant,
                        'level': lvl,
                        'method': method,
                        'directed_shd': avg_directed,
                        'extra_edges': avg_extra,
                        'missing_edges': avg_missing,
                        'time_s': avg_time,
                        'directed_shd_std': std_directed,
                        'extra_edges_std': std_extra,
                        'missing_edges_std': std_missing,
                        'time_s_std': std_time,
                    })
                    
                    
    # Save best grasp graphs for each dataset
    graph_dir = os.path.join("src", "learned_graphs")
    os.makedirs(graph_dir, exist_ok=True)
    for ds, graph in best_graphs.items():
        path = os.path.join(graph_dir, f"{ds}.pkl")
        with open(path, 'wb') as f:
            pickle.dump(graph, f)
        print(f"Saved best GRaSP graph for {ds} to {path}")

    return pd.DataFrame(results)

def plot_mean_std_trends():
    output_dir = "results/causal_discovery"
    variants = ['noise', 'missing']
    datasets = ['asia', 'alarm']
    
    # Prepare subplot grid: rows=datasets, cols=variants
    fig, axes = plt.subplots(nrows=len(datasets), ncols=len(variants),
                             figsize=(10, 4*len(datasets)), sharex='col', sharey='row')
    
    for i, ds in enumerate(datasets):
        for j, variant in enumerate(variants):
            file_path = os.path.join(output_dir, f"{variant}_results.csv")
            if not os.path.exists(file_path):
                continue
            
            ax = axes[i][j] if len(datasets) > 1 else axes[j]
            
            df = pd.read_csv(file_path)
            df = df[df['dataset'] == ds]
            
            for method in df['method'].unique():
                df_m = df[df['method'] == method]
                x = df_m['replicate']
                y = df_m['directed_shd']
                yerr = df_m['directed_shd_std']
                ax.errorbar(x, y, yerr=yerr, label=method.upper(), marker='o', capsize=3)
                
            ax.set_title(f"{ds} - {variant}")
            if j == 0:
                ax.set_ylabel('Directed SHD')
            if i == 0:
                ax.legend(title='Method', loc='upper right')
            if i == len(datasets) - 1:
                ax.set_xlabel('Level')
                
    plt.tight_layout()
    plt.savefig("figures/causal_discovery/mean_std_trends.png")
    plt.show()


def base_summary_to_latex():
    """
    Read the base variant results and convert into a latex table. 
    """
    
    file_path = "results/causal_discovery/base_results.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        metrics = ['directed_shd', 'extra_edges', 'missing_edges', 'time_s']
        df = df.set_index(['dataset', 'method'])
        summary_df = pd.DataFrame(index=df.index)
        for m in metrics:
            summary_df[m] = df[m].round(2).astype(str) + " ± " + df[f"{m}_std"].round(2).astype(str) 
            
        print("Base results summary:")
        print(summary_df)

        # Export base summary to LaTeX
        latex_path = "results/causal_discovery/base_summary.tex"
        
        with open(latex_path, "w") as f:
            f.write(summary_df.to_latex(
                caption="Base variant performance (mean $\\pm$ std)",
                label="tab:base_results",
            ))
        print(f"Latex file saved to {latex_path}")

def run_discovery():
    output_dir = os.path.join("results", "causal_discovery")
    
    df_results = evaluate_methods()
    print("Summary of all methods:")
    print(df_results)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    df_base = df_results[df_results["variant"] == "base"]
    df_base.to_csv(os.path.join(output_dir, "base_results.csv"), index=False)
    print("Saved base results to base_results.csv")

    df_noise = df_results[df_results["variant"] == "noise"]
    df_noise.to_csv(os.path.join(output_dir, "noise_results.csv"), index=False)
    print("Saved noise results to noise_results.csv")
    
    df_missing = df_results[df_results["variant"] == "missing"]
    df_missing.to_csv(os.path.join(output_dir, "missing_results.csv"), index=False)
    print("Saved missing results to missing_results.csv")
        

n_splits = 5
seed = 0

datasets = [
    'asia', 
    'sachs',
    'insurance',
    'child',
    'alarm'
]
variants = [
    'base',
    'noise', 
    'missing'
]
methods = [
    'pc',
    'grasp',
    # 'ges'
]

run_discovery()
base_summary_to_latex()
plot_mean_std_trends()
