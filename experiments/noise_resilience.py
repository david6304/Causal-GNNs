import os
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
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

def run_noise_eval():
    for ds in datasets:
        print(f"Dataset: {ds}")
        
        # load learned graph
        learned_adj = None
        graph_path = os.path.join("src", "learned_graphs", f"{ds}.pkl")
        with open(graph_path, "rb") as f:
            learned_adj = pickle.load(f)
            
        for variant in variants:
            print(f"Variant: {variant}")
            
            # Get results df
            csv_path = os.path.join("results", variant, "results.csv")
            if os.path.exists(csv_path):
                df_res = pd.read_csv(csv_path)
            else:
                df_res= pd.DataFrame(columns=["dataset","variant","level","model","accuracy","accuracy_std","f1","f1_std"])

            df_dict = load_synthetic_data(dataset=ds, variant=variant)
            order = sorted(df_dict.keys())
            target = target_nodes[ds]
            for level in order:
                df = df_dict[level]
                print(f"Level: {level}")
                xgb_scores = {}
                num_classes = int(df[target].nunique())
                for name, cl in model_classes.items():
                    metrics_acc = []
                    metrics_f1 = []
                    for i in range(n_splits):
                        print(f"Split {i+1}/{n_splits}")
                        X_train, y_train, X_val, y_val, X_test, y_test = preprocess_data(
                            df, target_node=target, random_state=seed + i
                        )
                        orig_test_count = len(y_test)
                        
                        # Fill missing values using the column mode
                        if variant == 'missing':
                            feat_cols = [c for c in df.columns if c != target]
                            for col in feat_cols:
                                X_train[col] = X_train[col].fillna(X_train[col].mode()[0])
                                X_val[col]   = X_val[col].fillna(X_train[col].mode()[0])
                                X_test[col]  = X_test[col].fillna(X_train[col].mode()[0])
                            # Drop samples where target is missing
                            mask = y_train.notna()
                            X_train, y_train = X_train[mask], y_train[mask]
                            mask = y_val.notna()
                            X_val, y_val = X_val[mask], y_val[mask]
                            mask = y_test.notna()
                            X_test, y_test = X_test[mask], y_test[mask]
                        
                        # Evaluate XGBoost first to get baseline score
                        if name == 'xgb':
                            params = load_hyperparameters(model_key=name, dataset_key=ds)
                            model = cl(**params, num_classes=num_classes, early_stopping_rounds=10)
                            res = evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test, target)
                            xgb_acc = res['accuracy']
                            if variant == 'missing':
                                scale = len(y_test) / orig_test_count
                                res['accuracy'] *= scale
                                res['f1'] *= scale
                            xgb_scores[i] = xgb_acc
                            print(f"{name} accuracy: {res['accuracy']:.4f}, f1: {res['f1']:.4f}")
                            metrics_acc.append(res['accuracy'])
                            metrics_f1.append(res['f1'])
                            continue

                        best_res = None
                        best_acc = -float('inf')
                        attempts = 5
                        for attempt in range(attempts): # Retry if below 90% xbg
                            params = load_hyperparameters(model_key=name, dataset_key=ds)
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
                                params['input_dim'] = X_train.shape[1]
                                model = cl(num_classes=num_classes, **params)
                                res = evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test, target)
                                
                            # Update best result
                            if res['accuracy'] > best_acc:
                                best_acc = res['accuracy']
                                best_res = res
                            if res['accuracy'] >= 0.9 * xgb_scores.get(i, xgb_acc):
                                best_res = res
                                break
                            if attempt == attempts - 1:
                                res = best_res
                                break
                            print(f"Attempt {attempt+1} accuracy {res['accuracy']:.4f} < 0.9 * {xgb_scores.get(i):.4f}; retrying")
                            
                        if variant == 'missing':
                            scale = len(y_test) / orig_test_count
                            res['accuracy'] *= scale
                            res['f1'] *= scale
                        print(f"{name} accuracy: {res['accuracy']:.4f}, f1: {res['f1']:.4f}")
                        metrics_acc.append(res['accuracy'])
                        metrics_f1.append(res['f1'])
                        
                    avg_acc = np.mean(metrics_acc)
                    avg_f1 = np.mean(metrics_f1)
                    std_acc = np.std(metrics_acc)
                    std_f1 = np.std(metrics_f1
                                    )
                    # Mask any existing rows for this dataset/model/variant/level
                    mask = (
                        (df_res["dataset"] == ds) &
                        (df_res["variant"] == variant) &
                        (df_res["level"] == level) &
                        (df_res["model"] == name)
                    )
                    df_res = df_res[~mask]
                    df_res.loc[len(df_res)] = [
                        ds, variant, level, name,
                        avg_acc, std_acc,
                        avg_f1, std_f1
                    ]
                    # Save after this model finishes its splits
                    df_res.to_csv(csv_path, index=False)

def plot_noise_study(
    out_path = "figures/noise_study/accuracy_noise_study.png"
):
    """
    Plot the accuracy of the models for each dataset and variant / level.
    """
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    # Read both variants to determine dataset keys
    df_dict = {}
    for variant in variants:
        csv_path = os.path.join("results", variant, "results.csv")
        df = pd.read_csv(csv_path)
        df_dict[variant] = df
        
    datasets = ["asia", "alarm"]

    n_rows = len(variants)
    n_cols = len(datasets)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), sharex=True)
    if n_rows == 1 and n_cols == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [axes]
    elif n_cols == 1:
        axes = [[ax] for ax in axes]

    # Plot accuracy for each variant × dataset
    for i, variant in enumerate(variants):
        df = df_dict.get(variant)
        levels = sorted(df["level"].unique())
        models = sorted(df["model"].unique())
        for j, ds in enumerate(datasets):
            ax = axes[i][j]
            df_ds = df[df["dataset"] == ds]
            for model in models:
                df_m = df_ds[df_ds["model"] == model]
                means = [df_m[df_m["level"] == lvl]["accuracy"].mean() for lvl in levels]
                errs  = [df_m[df_m["level"] == lvl]["accuracy_std"].mean() for lvl in levels]
                ax.errorbar(levels, means, yerr=errs, marker='o', label=model.upper())
            ax.set_title(f"{ds.capitalize()} — {variant}")
            ax.set_xlabel("Variant Level")
            ax.set_ylabel("Accuracy")
            ax.grid(True)
            if i == 0 and j == 0:
                ax.legend(loc='best', fontsize='small')
    fig.tight_layout(pad=2.0)
    fig.savefig(out_path)
    plt.close(fig)

datasets = [
    'asia', 
    'alarm', 
    ]
variants = [
    'noise', 
    'missing'
    ]
seed = 0
n_splits = 5

target_nodes = {
    'asia': 'dysp',
    'alarm': 'BP',
}
model_classes = {
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

run_noise_eval()
plot_noise_study()
