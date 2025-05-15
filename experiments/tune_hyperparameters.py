import os
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score
from src.data_processing.data_loader import load_synthetic_data
from src.data_processing.preprocessor import preprocess_data
from src.models.baselines.xgb import XGBBaseline
from src.models.baselines.mlp import MLPBaseline
from src.models.causal_gnns.gcn import GCN
from src.models.causal_gnns.gat import GAT
from src.models.causal_gnns.transformer_gnn import TransformerGNN
from src.models.causal_gnns.wrapper import GNNWrapper
from src.utils.hyperparam_tuning import tune_hyperparameters, update_hyperparameters

def tune_and_eval(model_name, model_class, base_params, X_train, y_train, X_val, y_val, X_test, y_test, target_node=None):
    if isinstance(model_class, tuple):
        model_class, gnn_class = model_class
        base_params['model_class'] = gnn_class
    params, _ = tune_hyperparameters(
        metric_fn=accuracy_score,
        model_class=model_class,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        base_params=base_params,
        random_state=seed,
        n_trials=n_trials,
        target_node=target_node
    )
    update_hyperparameters(model_name, ds, params)
    model = model_class(**base_params, **params)
    if hasattr(model, 'train'):
        model.train(X_train, y_train, X_val, y_val)
    else:
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    y_pred = model.predict(X_test)
    if isinstance(y_pred, dict):
        y_pred = y_pred[target_node]
    print(f"{model_class.__name__} Results: {accuracy_score(y_pred, y_test)}")
    
datasets = [
    'asia',
    'alarm'
    ]
target_nodes = {
    'asia': 'dysp',
    'alarm': 'BP'
    }
seed = 0
n_trials = 20
models = {
    'xgb': XGBBaseline,
    'mlp': MLPBaseline,
    'gcn': (GNNWrapper, GCN),
    'gat': (GNNWrapper, GAT),
    'transformer': (GNNWrapper, TransformerGNN),
}

for ds in datasets:
    print(f"Tuning for dataset: {ds}")

    df = load_synthetic_data(dataset=ds)
    target_node = target_nodes[ds]
    X_train, y_train, X_val, y_val, X_test, y_test = preprocess_data(
        df,
        target_node=target_node,
        random_state=seed
    )
    
    # Load learned graph
    graph_path = os.path.join("src", "learned_graphs", f"{ds}.pkl")
    with open(graph_path, "rb") as f:
        learned_adj = pickle.load(f)
        
    num_classes = y_train.nunique()

    train_df = pd.concat([X_train, y_train], axis=1)
    val_df = pd.concat([X_val, y_val], axis=1)
    
    node_classes = train_df.nunique(axis=0).astype(int).tolist()
    mask_token = int(max(node_classes))
    test_df = pd.concat([X_test, y_test], axis=1)
    test_df[target_node] = mask_token

    wrapper_base = {
        'graph': learned_adj,
        'node_classes': node_classes,
        'max_epochs': 20,
        'patience': 5,
        'verbose': False,
    }
    
    for name, model_class in models.items():
        print(f"Tuning {name}...")
        if name in ('gcn', 'gat', 'transformer'):
            tune_and_eval(
                model_name=name,
                model_class=model_class,
                base_params=wrapper_base,
                X_train=train_df, y_train=train_df,
                X_val=val_df, y_val=val_df,
                X_test=test_df, y_test=y_test,
                target_node=target_node
            )
        else:
            if name == 'xgb':
                base_params = {'num_classes': num_classes, 'early_stopping_rounds': 10}
            else:
                base_params = {
                    'num_classes': num_classes,
                    'input_dim': X_train.shape[1],
                    'max_epochs': 80,
                    'patience': 15
                }
            tune_and_eval(
                model_name=name,
                model_class=model_class,
                base_params=base_params,
                X_train=X_train, y_train=y_train,
                X_val=X_val, y_val=y_val,
                X_test=X_test, y_test=y_test,
                target_node=None
            )
    print(f"Finished tuning for dataset: {ds}")
