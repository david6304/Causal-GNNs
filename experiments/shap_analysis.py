import pickle
import pandas as pd
import numpy as np
import shap
import os
import matplotlib.pyplot as plt
from src.data_processing.data_loader import load_synthetic_data
from src.data_processing.preprocessor import preprocess_data
from src.models.baselines.xgb import XGBBaseline
from src.models.causal_gnns.wrapper import GNNWrapper
from src.models.causal_gnns.gat import GAT
from src.utils.hyperparam_tuning import load_hyperparameters

def explain_with_kernel_shap(name, model, X_train, X_test):

    background = X_train.sample(n=min(50, len(X_train)), random_state=seed)
    X_sample  = X_test.sample(n=min(50, len(X_test )), random_state=seed)

    print(f"Computing SHAP values for {name} on {dataset}")

    def f(x):
        df = pd.DataFrame(x, columns=X_train.columns)
        if name != 'xgb':
            df[target_node] = model.mask_token
        preds = model.predict(df)
        if isinstance(preds, dict):
            preds = preds[target_node]
        preds = np.array(preds)
        return preds

    explainer = shap.KernelExplainer(f, background.values)
    sv_list   = explainer.shap_values(X_sample.values, nsamples=50)

    # Get average across all output classes
    all_sv = np.stack(sv_list)
    sv = np.reshape(all_sv, (-1, all_sv.shape[-1]))

    vals = np.abs(sv).mean(axis=0)
    vals = pd.Series(vals, index=X_train.columns, name=name).sort_values(ascending=False)

    return vals

def compute_shap():
    df = load_synthetic_data(dataset=dataset)
    X_train, y_train, X_val, y_val, X_test, y_test = preprocess_data(df, target_node=target_node, random_state=seed)
    train_df = pd.concat([X_train, y_train], axis=1)
    val_df = pd.concat([X_val, y_val], axis=1)
    node_classes = train_df.nunique(axis=0).astype(int).tolist()
    num_classes = int(y_train.nunique())
    
    # load learned graph 
    path = f"src/learned_graphs/{dataset}.pkl"
    with open(path, "rb") as f:
        learned_adj = pickle.load(f)

    all_results = {}
    for name in models:        
        print(f"Computing SHAP for {name} on {dataset}")
        params = load_hyperparameters(model_key=name, dataset_key=dataset)
        if name == 'xgb':
            model = XGBBaseline(**params, num_classes=num_classes)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        else:
            model = GNNWrapper(
                graph=learned_adj,
                node_classes=node_classes,
                model_class=GAT,
                max_epochs=50,
                patience=10,
                verbose=True,
                **params
            )
            model.train(train_df, train_df, val_df, val_df)
        
        shap_res = explain_with_kernel_shap(name, model, X_train, X_test)
        all_results[name] = shap_res
    pd.DataFrame(all_results).to_csv("results/explainability/combined_shap_importances.csv", header=True)
    print(f"Saved SHAP importances to results/explainability/combined_shap_importances.csv")
        
def plot_shap_importances(csv_path, out_dir):
    """Load combined SHAP importances CSV and create top-10 bar charts."""

    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_path, index_col=0)

    for model in df.columns:
        series = df[model]
        top10 = series.abs().sort_values(ascending=False).head(10)[::-1]
        plt.figure(figsize=(6,4))
        top10.plot.barh()
        plt.xlabel("Mean |SHAP value|")
        plt.title(f"Top 10 SHAP importances: {model.upper()}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"top10_{model}.png"))
        plt.close()
    print(f"Saved SHAP top-10 plots to {out_dir}")
    
dataset = 'alarm'
target_node = 'BP'
models = [
    'xgb',
    'gat'
]
seed = 0

compute_shap()
plot_shap_importances(
    csv_path="results/explainability/combined_shap_importances.csv",
    out_dir=f"figures/shap"
)
