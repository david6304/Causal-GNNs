import matplotlib.pyplot as plt
import os
import pickle
import pandas as pd
import numpy as np
from src.data_processing.data_loader import load_synthetic_data
from src.utils.graph_utils import get_dag
from src.evaluation.metrics import calculate_ace
from src.data_processing.preprocessor import preprocess_data
from src.models.causal_gnns.gcn import GCN
from src.models.causal_gnns.gat import GAT
from src.models.causal_gnns.transformer_gnn import TransformerGNN
from src.models.causal_gnns.wrapper import GNNWrapper
from src.utils.hyperparam_tuning import load_hyperparameters


def run_ace_eval():
    """
    Calculate average ace values for each model / graph combo over 5 data splits.
    """

    df = load_synthetic_data(dataset=dataset, variant='base')
    _, adj_gt = get_dag(dataset=dataset, domain='discrete')

    # Build fully-connected graph 
    nodes = list(adj_gt.index)
    n = len(nodes)
    adj_full = pd.DataFrame(np.ones((n, n), dtype=int), index=nodes, columns=nodes)
    np.fill_diagonal(adj_full.values, 0)

    path = f"src/learned_graphs/{dataset}.pkl"
    with open(path, "rb") as f:
        learned_adj = pickle.load(f)

    # Pre-generate 5 splits and their random graphs
    nodes = list(adj_gt.index)
    splits = []
    for i in range(5):
        print(f"Preparing split {i+1}/5")
        # Generate a random directed graph for this split
        orig_edges = int(adj_gt.values.sum())
        max_edges = len(nodes) * (len(nodes) - 1)
        p = orig_edges / max_edges if max_edges > 0 else 0
        adj_rand = pd.DataFrame(
            np.random.binomial(1, p, size=(len(nodes), len(nodes))),
            index=nodes,
            columns=nodes
        )
        np.fill_diagonal(adj_rand.values, 0)

        train_df, val_df, test_df = preprocess_data(df, random_state=i)
        node_classes = train_df.nunique(axis=0).astype(int).tolist()
        
        splits.append({
            'adj_rand': adj_rand,
            'train_df': train_df,
            'val_df': val_df,
            'test_df': test_df,
            'node_classes': node_classes,
        })

    output_dir = f"results/understanding/{dataset}"
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "cace_summary.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(columns=["model","graph","source","target","mean","std"])

    # Evaluate each model over all splits and update CSV
    for model_name, model_class in model_classes.items():
        print(f"Evaluating model: {model_name}")
        params = load_hyperparameters(model_key=model_name, dataset_key=dataset)

        scores = {}
        for i, split in enumerate(splits):
            graph_list = {
                'true': adj_gt,
                'learned': learned_adj,
                'random': split['adj_rand'],
                'full': adj_full
            }
            
            for graph_name, graph in graph_list.items():
                if model_name != 'gcn' and graph_name == 'full':
                    continue
                
                print(f"Evaluating {model_name} on {graph_name} graph, split {i+1}/5")

                model = GNNWrapper(
                    graph=graph,
                    node_classes=split['node_classes'],
                    model_class=model_classes[model_name],
                    max_epochs=50,
                    patience=10,
                    verbose=True,
                    **params
                )
                model.train(
                    split['train_df'], split['train_df'],
                    split['val_df'],   split['val_df']
                )
                for source, target in node_pairs:
                    ace = calculate_ace(
                        model, split['test_df'],
                        source, target
                    )
                    scores.setdefault((graph_name, source, target), []).append(ace)
                    
        # Compute and save mean/std for this model
        for (graph_name, source, target_pair), vals in scores.items():
            mean_score = np.mean(vals)
            std_score  = np.std(vals)
            
            # Mask any existing rows for this model/graph/source/target
            mask = (
                (df['model']==model_name) &
                (df['graph']==graph_name) &
                (df['source']==source) &
                (df['target']==target_pair)
            )
            df = df[~mask]
            df.loc[len(df)] = [
                model_name, graph_name, source, target_pair,
                mean_score, std_score
            ]
        df.to_csv(csv_path, index=False)
        print(f"Updated ace values for {model_name}")

    return df


def plot_ace_barcharts(
    base_csv = "results/understanding",
    base_out = "figures/ace_plots"
):
    """Create bar charts of CACE for each node-pair for each dataset in separate folders."""
    models = ['standard_gnn', 'gcn', 'gat', 'transformer']
    graph_types = ['true', 'learned', 'random']
    bar_width = 0.8 / len(graph_types)
    x = np.arange(len(models))

    # Define consistent colours for each graph type
    colours = ['tab:blue', 'tab:orange', 'tab:green']
    graph_colours = {g: colours[i] for i, g in enumerate(graph_types)}

    csv_path = os.path.join(base_csv, dataset, "cace_summary.csv")
    out_dir = os.path.join(base_out, dataset)
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    df.loc[(df['model'] == 'gcn') & (df['graph'] == 'full'), 'model'] = 'standard_gnn'

    for source, target in df[['source', 'target']].drop_duplicates().values:
        subset_all = df[(df['source'] == source) & (df['target'] == target)]
        fig, ax = plt.subplots(figsize=(6, 4))
        legend_plotted = set()

        for i, m in enumerate(models):
            subset_model = subset_all[subset_all['model'] == m]

            if m == 'standard_gnn':
                entry = subset_model[subset_model['graph'] == 'full']
                mean = entry['mean'].iloc[0] if not entry.empty else 0
                std  = entry['std'].iloc[0] if not entry.empty else 0
                ax.bar(x[i], mean, bar_width, yerr=std, color='grey')
            else:
                for j, g in enumerate(graph_types):
                    entry = subset_model[subset_model['graph'] == g]
                    mean = entry['mean'].iloc[0] if not entry.empty else 0
                    std  = entry['std'].iloc[0] if not entry.empty else 0
                    offset = (j - (len(graph_types) - 1) / 2) * bar_width
                    label = g.capitalize() if g not in legend_plotted else None
                    ax.bar(x[i] + offset, mean, bar_width, yerr=std,
                            label=label, color=graph_colours[g])
                    legend_plotted.add(g)

        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in models], rotation=45)
        ax.set_ylabel("Average Causal Effect")
        ax.set_title(f"{dataset.upper()} ACE: {source} → {target}")
        ax.legend(title="Graph Type")
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        fig.tight_layout()

        out_file = os.path.join(out_dir, f"ace_{source}_{target}.png")
        fig.savefig(out_file)
        plt.close(fig)

node_pairs = [
    ('BP', 'TPR'), 
    ('LVFAILURE', 'LVEDVOLUME'),
    ('HR', 'CO'), 
    ('STROKEVOLUME', 'CO'), 
]

model_classes = {
    'gcn': GCN,
    'gat': GAT,
    'transformer': TransformerGNN
}

dataset = 'alarm'
    
run_ace_eval()
plot_ace_barcharts()
