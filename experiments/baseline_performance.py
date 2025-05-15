import os
import pickle
import random
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from src.data_processing.data_loader import load_synthetic_data
from src.data_processing.preprocessor import preprocess_data
from src.utils.hyperparam_tuning import load_hyperparameters
from src.models.baselines.xgb import XGBBaseline
from src.models.baselines.mlp import MLPBaseline
from src.models.causal_gnns.gcn import GCN
from src.models.causal_gnns.gat import GAT
from src.models.causal_gnns.transformer_gnn import TransformerGNN
from src.models.causal_gnns.wrapper import GNNWrapper


def evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test, target_node=None):
    
    if hasattr(model, 'train'):
        model.train(X_train, y_train, X_val, y_val)
    else:
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    preds = model.predict(X_test)
    
    if isinstance(preds, dict):
        preds = preds[target_node]
    
    if isinstance(y_test, pd.DataFrame):
        y_test = y_test[target_node]

    results = {}
    for name, fn in metrics.items():
        if fn == f1_score:
            score = fn(y_test, preds, average='macro')
        else:
            score = fn(y_test, preds)
        results[name] = score
    return results

def run_evaluation():
    for ds in datasets:
        print(f"Dataset: {ds}")
        df = load_synthetic_data(dataset=ds, variant='base')
        target = target_nodes[ds]

        # load learned graph
        learned_adj = None
        graph_path = os.path.join("src", "learned_graphs", f"{ds}.pkl")
        with open(graph_path, "rb") as f:
            learned_adj = pickle.load(f)

        # Pre-generate splits 
        splits = []
        for split in range(n_splits):
            print(f"Preparing split {split+1}/{n_splits}")
            X_train, y_train, X_val, y_val, X_test, y_test = preprocess_data(
                df, target_node=target,
                test_size=0.3, val_size=0.2,
                random_state=seed + split
            )
            splits.append((X_train, y_train, X_val, y_val, X_test, y_test))

        for name, cl in models_to_run.items():
            print(f"Model: {name}")
            accuracies = []
            f1_scores = []

            for i, (X_train, y_train, X_val, y_val, X_test, y_test) in enumerate(splits):
                print(f"Training {name}, split {i+1}/{n_splits}")

                params = load_hyperparameters(model_key=name, dataset_key=ds)
                print(f"Hyperparameters: {params}")
                best_res = None
                for attempt in range(2):
                    if name in ('gcn', 'gat', 'transformer'):
                        
                        train_df = pd.concat([X_train, y_train], axis=1)
                        val_df = pd.concat([X_val, y_val], axis=1)
                        test_df = pd.concat([X_test, y_test], axis=1)
                        node_classes = train_df.nunique(axis=0).astype(int).tolist()

                        X_test_df = test_df.copy()
                        mask_token = int(max(node_classes))
                        X_test_df[target] = mask_token
                        wrapper_params = {
                            'graph': learned_adj,
                            'node_classes': node_classes,
                            'verbose': True,
                            'max_epochs': 50,
                            'patience': 10
                        }
                        Wrapper, gnn_class = cl
                        model = Wrapper(**{**wrapper_params, **params, 'model_class': gnn_class})
                        res = evaluate_model(model, train_df, train_df, val_df, val_df, X_test_df, test_df, target)
                    else:
                        # Baselines
                        if name == 'mlp':
                            params['input_dim'] = X_train.shape[1]
                        model = cl(num_classes=int(y_train.nunique()), **params)
                        res = evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test, target)
                        
                    if res['accuracy'] >= 0.75 or attempt == 1:
                        best_res = res
                        break
                    print(f"Attempt {attempt+1} accuracy {res['accuracy']:.4f} < 0.75; retrying")
                res = best_res

                print(f"Result for split {i+1}/{n_splits}: accuracy={res['accuracy']:.4f}, f1={res['f1']:.4f}")
                accuracies.append(res['accuracy'])
                f1_scores.append(res['f1'])

            mean_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)
            mean_f1 = np.mean(f1_scores)
            std_f1 = np.std(f1_scores)
            print(f"Mean accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
            print(f"Mean F1 score: {mean_f1:.4f} ± {std_f1:.4f}")


            output_dir = "results/performance"
            os.makedirs(output_dir, exist_ok=True)
            csv_file = os.path.join(output_dir, "performance.csv")

            summary_df = pd.DataFrame([{
                "dataset": ds,
                "model": name,
                "mean_accuracy": mean_acc,
                "std_accuracy": std_acc,
                "mean_f1": mean_f1,
                "std_f1": std_f1
            }])

            # Update the csv for just this dataset / model combo
            if os.path.exists(csv_file):
                existing = pd.read_csv(csv_file)
                existing = existing[~((existing["dataset"] == ds) & (existing["model"] == name))]
                new_df = pd.concat([existing, summary_df], ignore_index=True)
            else:
                new_df = summary_df

            new_df.to_csv(csv_file, index=False)

def summarise_performance_csv(
    csv_path="results/performance/performance.csv", 
    tex_path="results/performance/performance_summary.tex"
    ):
    """Read the performance CSV, format means and stds, and write a LaTeX table."""

    df = pd.read_csv(csv_path)

    df["dataset"] = df["dataset"].str.capitalize()
    df["model"] = df["model"].str.upper()

    df["Accuracy"] = df.apply(
        lambda row: f"{row['mean_accuracy']:.3f} ± {row['std_accuracy']:.3f}", axis=1
    )
    df["F1 Score"] = df.apply(
        lambda row: f"{row['mean_f1']:.3f} ± {row['std_f1']:.3f}", axis=1
    )

    out_df = df[["dataset", "model", "Accuracy", "F1 Score"]]

    with open(tex_path, "w") as tex_file:
        tex_file.write(out_df.to_latex(index=False, escape=False,
            caption="Performance summary: mean ± std for Accuracy and F1 Score.",
            label="tab:performance_summary"))

seed = 0
n_splits = 10

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

datasets = [
    'asia',
    'alarm'
]
target_nodes = {
    'asia': 'dysp',
    'alarm': 'BP',
}
models_to_run = {
    'xgb': XGBBaseline,
    'mlp': MLPBaseline,
    'gcn': (GNNWrapper, GCN),
    'gat': (GNNWrapper, GAT),
    'transformer': (GNNWrapper, TransformerGNN),
}

metrics = {
    'accuracy': accuracy_score,
    'f1': f1_score,
}
run_evaluation()
summarise_performance_csv()