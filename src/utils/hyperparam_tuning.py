import optuna 
import inspect
import pandas as pd
import os
from sklearn.metrics import f1_score
import yaml
from src.models.causal_gnns.wrapper import GNNWrapper
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
import numpy as np

import time

def tune_hyperparameters(
    metric_fn,
    model_class,
    X_train,
    y_train,
    X_val,
    y_val,
    target_node=None,
    base_params=None,
    random_state=None,
    n_trials=100,
    cv_folds=3,
    verbose=False,
    direction="maximize",
):
    """
    Tune hyperparameters for baselines and GNNs using Optunas bayesian optimisation.
    """
    init_params = inspect.signature(model_class.__init__).parameters

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction=direction, sampler=sampler)

    # Concat for cross-val
    X = pd.concat([X_train, X_val], axis=0)
    y = pd.concat([y_train, y_val], axis=0)

    def objective(trial):
        start_time = time.time()
        params = {}
        
        if "lr" in init_params:
            params["lr"] = trial.suggest_categorical("lr", [1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
        if "n_estimators" in init_params:
            params["n_estimators"] = trial.suggest_categorical("n_estimators", [50, 100, 200, 500])
        if "max_depth" in init_params:
            params["max_depth"] = trial.suggest_categorical("max_depth", [3, 5, 7, 10])
            
        if "dropout" in init_params:
            params["dropout"] = trial.suggest_categorical("dropout", [0.0, 0.1, 0.2, 0.3])
        if "hidden_dim" in init_params:
            params["hidden_dim"] = trial.suggest_categorical("hidden_dim", [16, 32, 64])
        if "heads" in init_params:
            params["heads"] = trial.suggest_categorical("heads", [1, 4, 8])
        if "embedding_dim" in init_params:
            params["embedding_dim"] = trial.suggest_categorical("embedding_dim", [8, 16, 32])
        if "num_layers" in init_params:
            params["num_layers"] = trial.suggest_categorical("num_layers", [1, 2, 3])

        if base_params:
            params.update(base_params)

        if isinstance(y, pd.DataFrame) and target_node:
            labels = y[target_node]
        elif not isinstance(y, pd.DataFrame):
            labels = y
        else:
            labels = None

        if labels is not None:
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            splits = cv.split(X, labels)
        else:
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            splits = cv.split(X)

        scores = []
        for train, val in splits:
            X_train, X_val = X.iloc[train], X.iloc[val]
            y_train, y_val = y.iloc[train], y.iloc[val]
            model = model_class(**params)
            if hasattr(model, 'fit'):
                if model.task == 'regression':
                    model.fit(X_train, y_train)
                else:
                    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
            elif hasattr(model, 'train'):
                model.train(X_train, y_train, X_val, y_val)
            
            if model_class == GNNWrapper:
                mask_token = int(max(model.node_classes))
                X_val_masked = X_val.copy()
                X_val_masked[target_node] = mask_token
                preds = model.predict(X_val_masked)
            else:
                preds = model.predict(X_val)
            
            
            if isinstance(preds, dict):
                preds = preds[target_node]
                y_true = y_val[target_node]
            else:
                y_true = y_val
                
            score = metric_fn(y_true, preds) if metric_fn != f1_score else metric_fn(y_true, preds, average='macro')
            scores.append(score)
        
        if verbose:
            print(f"Trial {trial.number} finished in {time.time() - start_time:.2f} seconds.")
            
        lowest_score = float(np.min(scores))
            
        return lowest_score

    study.optimize(objective, n_trials=n_trials, n_jobs=1)

    return study.best_params, study.best_value

def update_hyperparameters(model_key, dataset_key, hyperparameters, file_path="config/hyperparameters.yaml"):
    """
    Update the hyperparameter config file.
    """
    # Load existing data
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            try:
                data = yaml.safe_load(f)
                if data is None:
                    data = {}
            except yaml.YAMLError:
                data = {}
    else:
        data = {}

    if model_key not in data:
        data[model_key] = {}
    data[model_key][dataset_key] = hyperparameters

    # Save the updated hyperparameters
    with open(file_path, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False)
    print(f"Updated {file_path} with {model_key} hyperparameters for {dataset_key}")

def load_hyperparameters(model_key, dataset_key, file_path="config/hyperparameters.yaml"):
    """
    Load hyperparameters for a specific model and dataset from config file.
    """
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            try:
                data = yaml.safe_load(f)
                if data is None:
                    data = {}
            except yaml.YAMLError:
                data = {}
    else:
        data = {}
        
    return data.get(model_key, {}).get(dataset_key, {})
