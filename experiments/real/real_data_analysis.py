import os
import numpy as np
import pandas as pd
import torch
from src.models.causal_gnns.gat_real import GATWrapper
from experiments.real.graph_discovery_proteomics import get_combined_adj_mat
from src.utils.hyperparam_tuning import tune_hyperparameters, load_hyperparameters, update_hyperparameters
from src.data_processing.preprocessor import preprocess_omics_data, preprocess_genotype_data
from src.data_processing.data_loader import load_real_data, load_genotype_data
from sklearn.metrics import f1_score, root_mean_squared_error, accuracy_score
import pickle
from src.models.baselines.xgb import XGBBaseline
from bioservices import KEGG
from mygene import MyGeneInfo
from src.data_processing.proteomics import get_cluster_to_proteins


def tune_gnn(task="metab", genotype=None):
    """
    Tune the model using cross validation.
    """
    combined_graph, all_organs_graph = get_combined_adj_mat(task=task)
    save_name = task
    if genotype is not None:
        print(f"Genotype: {genotype}")
        save_name = f"{task}_{genotype}"
    
    if task == "metab":
        p_df, m_df = load_real_data()
        X_train, y_train, X_val, y_val, X_test, y_test = preprocess_omics_data(p_df, m_df, random_state=0)
        output_dim = y_train.shape[1]
    else:
        df = load_genotype_data(target=task, genotype=genotype)
        X_train, y_train, X_val, y_val, X_test, y_test, label_mapping = preprocess_genotype_data(df, target=task, random_state=0)
        output_dim = y_train.nunique()
    
    objective = "regression" if task == "metab" else "classification"
    direction = "minimize" if task == "metab" else "maximize"
    metric_fn = root_mean_squared_error if task == "metab" else accuracy_score
    
    base_params = {
        "output_dim": output_dim,
        "max_epochs": 500,
        "patience": 20,
        "graph": combined_graph,
        "verbose": False,
        "task": objective,
    }
    
    params, value = tune_hyperparameters(
        metric_fn=metric_fn,
        model_class=GATWrapper,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        target_node=None,
        base_params=base_params,
        random_state=0,
        n_trials=5,
        cv_folds=3,
        verbose=True,
        direction=direction,
    )
    
    # save the best parameters
    update_hyperparameters(
        model_key="GAT_real",
        dataset_key=save_name,
        hyperparameters=params,
    )
    
    # Train and evaluate the model with the best parameters
    model = GATWrapper(**params, **base_params)
    
    model.train(X_train, y_train, X_val, y_val)
    preds = model.predict(X_test)
    if task == "metab":
        rmse = root_mean_squared_error(y_test, preds)
        print(f"Test RMSE: {rmse:.4f}")
    else:
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average='weighted')
        print(f"Test Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")

def tune_xgboost(task="metab", genotype=None):
    """
    Tune the model using cross validation.
    """
    save_name = task
    if genotype is not None:
        print(f"Genotype: {genotype}")
        save_name = f"{task}_{genotype}"
        
    if task == "metab":
        p_df, m_df = load_real_data()
        X_train, y_train, X_val, y_val, X_test, y_test = preprocess_omics_data(p_df, m_df, random_state=0)
    else:
        df = load_genotype_data(target=task, genotype=genotype)
        X_train, y_train, X_val, y_val, X_test, y_test, label_mapping = preprocess_genotype_data(df, target=task, random_state=0)
    
    objective = "regression" if task == "metab" else "classification"
    direction = "minimize" if task == "metab" else "maximize"
    metric_fn = root_mean_squared_error if task == "metab" else accuracy_score
    base_params = {
        "early_stopping_rounds": 0,
        "task": objective,
    }
    params, value = tune_hyperparameters(
        metric_fn=metric_fn,
        model_class=XGBBaseline,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        target_node=None,
        base_params=base_params,
        random_state=0,
        n_trials=8,
        cv_folds=3 if task == "metab" else 5,
        verbose=False,
        direction=direction,
    )
    
    # save the best parameters
    update_hyperparameters(
        model_key="xgb",
        dataset_key=save_name,
        hyperparameters=params,
    )
    
    # Train and evaluate the model with the best parameters
    model = XGBBaseline(**params, **base_params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    if task == "metab":
        rmse = root_mean_squared_error(y_test, preds)
        print(f"Test RMSE: {rmse:.4f}")
    else:
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average='weighted')
        print(f"Test Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
    

def eval_models(task="metab", genotype=None, splits=10, top_n=50):
    # Load the graph
    combined_graph, all_organs_graph = get_combined_adj_mat(task=task)
    
    results = {}
    save_name = task
    if genotype is not None:
        print(f"Genotype: {genotype}")
        save_name = f"{task}_{genotype}"
    
    for split in range(splits):
        results[split] = {}
        # Load the data
        if task == "metab":
            p_df, m_df = load_real_data()
            X_train, y_train, X_val, y_val, X_test, y_test = preprocess_omics_data(p_df, m_df, random_state=split)
            output_dim = y_train.shape[1]
        else:
            df = load_genotype_data(target=task, genotype=genotype)          
            X_train, y_train, X_val, y_val, X_test, y_test, label_mapping = preprocess_genotype_data(df, target=task, random_state=split)
            output_dim = y_train.nunique()
        
        if task == "metab":
            # Compare with just predicting the mean
            train_mean = y_train.mean(axis=0)
            mean_preds_full = pd.DataFrame(np.repeat([train_mean.values], len(y_test), axis=0), columns=y_test.columns, index=y_test.index)
            rmse = root_mean_squared_error(y_test, mean_preds_full)
            print(f"Test RMSE for mean prediction: {rmse:.4f}")
            
            results[split]["mean"] = rmse
            
            # Compare with just predicting the mean for each metabolite
            rmse_results = rmse_per_metabolite(y_test, mean_preds_full)
            
            # Store the mean of the RMSE of the top n metabolites
            results[split]["mean_top_n"] = rmse_results.head(top_n)["rmse"].mean()
            print(f"Mean RMSE for top {top_n} metabolites: {results[split]['mean_top_n']:.4f}")

        objective = "regression" if task == "metab" else "classification"
        params = load_hyperparameters(
            model_key="GAT_real",
            dataset_key=save_name,
        )
        base_params = {
            "output_dim": output_dim,
            "max_epochs": 500,
            "patience": 10,
            "graph": combined_graph,
            "verbose": True,
            "task": objective,
        }
        model = GATWrapper(**params, **base_params)
        model.train(X_train, y_train, X_val, y_val)
 
        preds = model.predict(X_test)
        if task == "metab":
            rmse = root_mean_squared_error(y_test, preds)
            print(f"Test RMSE for GNN: {rmse:.4f}")
            results[split]["gnn"] = rmse
            rmse_results = rmse_per_metabolite(y_test, preds)
            # Store the mean of the RMSE of the top n metabolites
            results[split]["gnn_top_n"] = rmse_results.head(top_n)["rmse"].mean()
            print(f"Mean RMSE for top {top_n} metabolites: {results[split]['gnn_top_n']:.4f}")
        else:
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average='weighted')
            print(f"Test Accuracy for GNN: {acc:.4f}, F1 Score: {f1:.4f}")
            results[split]["gnn"] = acc
            results[split]["gnn_f1"] = f1
            
            
        params = load_hyperparameters(
            model_key="xgb",
            dataset_key=save_name,
        )
        base_params = {
            "early_stopping_rounds": 0,
            "task": objective,
        }
        xgb = XGBBaseline(**params, **base_params)
        xgb.fit(X_train, y_train)
        preds = xgb.predict(X_test)
        
        if task == "metab":
            rmse = root_mean_squared_error(y_test, preds)
            print(f"Test RMSE for XGBoost: {rmse}")
            results[split]["xgb"] = rmse
            rmse_results = rmse_per_metabolite(y_test, preds)
            # Store the mean of the RMSE of the top n metabolites
            results[split]["xgb_top_n"] = rmse_results.head(top_n)["rmse"].mean()
            print(f"Mean RMSE for top {top_n} metabolites: {results[split]['xgb_top_n']:.4f}")
    
        else:
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average='weighted')
            print(f"Test Accuracy for XGBoost: {acc:.4f}, F1 Score: {f1:.4f}")
            results[split]["xgb"] = acc
            results[split]["xgb_f1"] = f1
    # Get mean and std of results across splits 
    df = pd.DataFrame.from_dict(results, orient="index")
    means = df.mean()
    stds = df.std()
    
    summary = pd.concat([means, stds], axis=1)
    if task == "metab":
        summary.columns = ["mean_rmse", "std_rmse"]
    else:
        summary = pd.DataFrame({
            "mean_accuracy": [means["gnn"], means["xgb"]],
            "std_accuracy": [stds["gnn"], stds["xgb"]],
            "mean_f1": [means["gnn_f1"], means["xgb_f1"]],
            "std_f1": [stds["gnn_f1"], stds["xgb_f1"]],
        }, index=["gnn", "xgb"])
    
    print(summary)
    summary.to_csv(f"results/real/{save_name}_results.csv")
    df.to_csv(f"results/real/splits_{save_name}_results.csv")
    

def get_pathway_to_metabolite(metabolite_ids):
    """
    Get the mapping of pathway ids to metabolite ids from the KEGG database.
    """
    if os.path.isfile("data/real/pathway_to_metabolites.pkl"):
        with open("data/real/pathway_to_metabolites.pkl", "rb") as f:
            pathway_to_metabolites = pickle.load(f)
        return pathway_to_metabolites
    
    k = KEGG()
    # Create a dict of pathways to metabolites
    pathway_to_metabolites = {}
    links = k.link("pathway", "compound").splitlines()
    for line in links:
        line = line.split("\t")
        metabolite_id = line[0].split(":")[1]
        pathway_id = line[1].split(":")[1]
        pathway_id = pathway_id.replace("map", "hsa")
        if pathway_id not in pathway_to_metabolites:
            pathway_to_metabolites[pathway_id] = []
        if metabolite_id not in metabolite_ids:
            continue
        pathway_to_metabolites[pathway_id].append(metabolite_id)
    
    # Save the mapping to a file
    with open("data/real/pathway_to_metabolites.pkl", "wb") as f:
        pickle.dump(pathway_to_metabolites, f)

    return pathway_to_metabolites
    
def get_disease_to_pathways():
    """
    Get the mapping of diseases to pathways from the KEGG database.
    """
    if os.path.isfile("data/real/disease_to_pathways.pkl"):
        with open("data/real/disease_to_pathways.pkl", "rb") as f:
            disease_to_pathways = pickle.load(f)
        return disease_to_pathways

    k = KEGG()
    disease_names = k.list("disease").splitlines()
    id_to_name = {}
    for line in disease_names:
        line = line.split("\t")
        if len(line) == 2:
            id_to_name[line[0]] = line[1]
    
    pathway_scores = pd.read_csv("results/real/pathway_scores.csv", index_col=0)
    # Get the pathways from the pathway scores
    pathways = pathway_scores.index.tolist()
    disease_to_pathways = {}
    links = k.link("disease", "pathway").splitlines()
    
    # Convert pathways from map to hsa
    pathways = [pathway.replace("map", "hsa") for pathway in pathways]
    
    for line in links:
        line = line.split("\t")
        pathway_id = line[0].split(":")[1]
        disease_id = line[1].split(":")[1]
        if pathway_id not in pathways:
            continue
        disease = id_to_name.get(disease_id, disease_id)
        if disease not in disease_to_pathways:
            disease_to_pathways[disease] = []
        disease_to_pathways[disease].append(pathway_id)
    
    # Save the mapping to a file
    with open("data/real/disease_to_pathways.pkl", "wb") as f:
        pickle.dump(disease_to_pathways, f)
    
    return disease_to_pathways

        
def score_pathways(metabolite_preds):
    """
    Use the predicted metabolite values to score kegg pathways
    """
    
    kegg_mapping = pd.read_csv("data/real/kegg.csv")
    metabolite_to_kegg = {metabolite: kegg for metabolite, kegg in zip(kegg_mapping["ID"], kegg_mapping["ID2"])}
    
    # Rename metabolites with kegg ids in predictions - remove any with kegg id of NA
    metabolite_preds.rename(columns=metabolite_to_kegg, inplace=True)
    keep_cols = [col for col in metabolite_preds.columns.dropna() if col != "NA"]
    metabolite_preds = metabolite_preds.loc[:, keep_cols]
    
    cols = metabolite_preds.columns.to_list()
    pathway_to_metabolites = get_pathway_to_metabolite(cols)
    
    pathway_scores = {}
    for pathway, metabolites in pathway_to_metabolites.items():
        # Get the metabolites in the pathway that are also in the predictions
        common_metabolites = [metab for metab in metabolites if metab in metabolite_preds.columns]
        
        # Calculate the mean of the predicted values for these metabolites
        if common_metabolites:
            pathway_scores[pathway] = metabolite_preds[common_metabolites].mean(axis=1)
        else:
            pathway_scores[pathway] = pd.Series([None] * metabolite_preds.shape[0], index=metabolite_preds.index)
    
    return pathway_scores

def pred_pathways():
    """
    Calculate pathway scores for the gnn and save the results.
    """
    combined, all_organs = get_combined_adj_mat()
    p_df, m_df = load_real_data()
    X_train, y_train, X_val, y_val, X_test, y_test = preprocess_omics_data(
        p_df, m_df, test_size=0.1, val_size=0.2, random_state=0
    )
    output_dim = y_train.shape[1]
    params = load_hyperparameters(
        model_key="GAT_real",
        dataset_key="Real",
    )
    base_params = {
        "max_epochs": 200,
        "patience": 10,
        "graph": combined,
        "verbose": False,
        "output_dim": output_dim,
    }
    model = GATWrapper(**params, **base_params)
    model.train(X_train, y_train, X_val, y_val)

    preds = model.predict(X_test)
    preds = pd.DataFrame(preds, columns=y_test.columns, index=y_test.index)
    
    pathway_scores = score_pathways(preds)
    
    # Only use the first sample for each metabolite
    for pathway, scores in pathway_scores.items():
        pathway_scores[pathway] = scores.iloc[0]
    
    df = pd.DataFrame.from_dict(pathway_scores, orient="index", columns=["score"])
    df = df.sort_values(by="score", ascending=False)
    df.to_csv("results/real/pathway_scores.csv")
    
def pred_diseases():
    """
    Predict diseases based on pathway scores.
    """
    pathway_scores = pd.read_csv("results/real/pathway_scores.csv", index_col=0)
    disease_to_pathways = get_disease_to_pathways()
    
    pathway_ids = pathway_scores.index.tolist()
    
    disease_scores = {}
    for disease, pathways in disease_to_pathways.items():
        if disease not in disease_scores:
            disease_scores[disease] = []
        common_pathways = [pathway for pathway in pathways if pathway in pathway_ids]
        
        if common_pathways:
            disease_scores[disease] = pathway_scores.loc[common_pathways, "score"].mean()
        else:
            disease_scores[disease] = 0
    
    df = pd.DataFrame.from_dict(disease_scores, orient="index", columns=["score"])
    df = df.sort_values(by="score", ascending=False)
    df.to_csv("results/real/disease_scores.csv")
        
def rmse_per_metabolite(y_test, preds):
    """
    Calculate RMSE for each metabolite then sort by variance.
    """
    results = []
    if not isinstance(preds, pd.DataFrame):
        preds = pd.DataFrame(preds, columns=y_test.columns, index=y_test.index)
    for col in y_test.columns:
        variance = y_test[col].var()
        rmse = root_mean_squared_error(y_test[col], preds[col])
        results.append((col, rmse, variance))
    results = pd.DataFrame(results, columns=["metabolite", "rmse", "variance"])
    results = results.sort_values(by="variance", ascending=False)
    return results

def get_protein_to_pathway_dict(protein_ids):
    if os.path.exists("data/real/mappings/protein_to_entrez.pkl"):
        mapping = pickle.load(open("data/real/mappings/protein_to_entrez.pkl", "rb"))
    else:
        mg = MyGeneInfo()
        # First get entrez IDs from ensembl IDs
        out = mg.querymany(protein_ids, scopes="ensembl.gene", fields="entrezgene", species="mouse")
        mapping = {item["query"]: str(item["entrezgene"]) for item in out if "entrezgene" in item}
        with open("data/real/mappings/protein_to_entrez.pkl", "wb") as f:
            pickle.dump(mapping, f)
        
    if os.path.exists("data/real/mappings/protein_to_pathway.pkl"):
        return pickle.load(open("data/real/mappings/protein_to_pathway.pkl", "rb"))
    
    k = KEGG()
    # Get pathways for each entrez ID
    pathway_dict = {}
    for protein_id, entrez_id in mapping.items():
        print(f"Getting pathways for {protein_id} ({entrez_id})")
        links = k.link("pathway", f"mmu:{entrez_id}")
        pathways = [line.split("\t")[1].split(":")[1] for line in links.split("\n") if line]
        print(f"Found {len(pathways)} pathways for {protein_id}")
        pathway_dict[protein_id] = pathways
    
    with open("data/real/mappings/protein_to_pathway.pkl", "wb") as f:
        pickle.dump(pathway_dict, f)
    return pathway_dict

def get_cluster_to_pathway_dict(cluster_ids):
    if os.path.exists("data/real/mappings/cluster_to_pathway.pkl"):
        return pickle.load(open("data/real/mappings/cluster_to_pathway.pkl", "rb"))
    
    cluster_to_proteins = get_cluster_to_proteins("data/real/clusters", cluster_ids)
    proteins_ids = set()
    for cluster_id, proteins in cluster_to_proteins.items():
        proteins_ids.update(proteins)
    
    protein_to_pathway_dict = get_protein_to_pathway_dict(proteins_ids)
        
    cluster_to_pathway = {}
    for cluster_id, proteins in cluster_to_proteins.items():
        pathways = set()
        for protein in proteins:
            if protein in protein_to_pathway_dict:
                pathways.update(protein_to_pathway_dict[protein])
        cluster_to_pathway[cluster_id] = list(pathways)
    with open("data/real/mappings/cluster_to_pathway.pkl", "wb") as f:
        pickle.dump(cluster_to_pathway, f)
    return cluster_to_pathway

def get_pathway_id_to_names(pathway_ids):
    """
    Get the mapping of pathway ids to names from the KEGG database.
    """
    if os.path.exists("data/real/mappings/pathway_id_to_name.pkl"):
        return pickle.load(open("data/real/mappings/pathway_id_to_name.pkl", "rb"))
    
    k = KEGG()
    pathway_names = k.list("pathway", "mmu").splitlines()
    id_to_name = {}
    for line in pathway_names:
        line = line.split("\t")
        if len(line) == 2:
            id_to_name[line[0]] = line[1]
    
    pathway_id_to_name = {pathway: id_to_name[pathway] for pathway in pathway_ids if pathway in id_to_name}
    
    with open("data/real/mappings/pathway_id_to_name.pkl", "wb") as f:
        pickle.dump(pathway_id_to_name, f)
    
    return pathway_id_to_name

def score_genotype_pathways(cluster_scores):
    """
    Use the predicted cluster scores to score kegg pathways
    """
    
    # Get the mapping of clusters to pathways
    cluster_to_pathway = get_cluster_to_pathway_dict(cluster_scores.index.tolist())
    
    pathway_scores = {}
    for cluster, score in cluster_scores.items():
        pathways = cluster_to_pathway.get(cluster, [])
        for pathway in pathways:
            if pathway not in pathway_scores:
                pathway_scores[pathway] = []
            pathway_scores[pathway].append(score)
    
    pathway_id_to_name = get_pathway_id_to_names(pathway_scores.keys())
    pathway_means = {}
    # Calculate the mean of the scores for each pathway
    for pathway, scores in pathway_scores.items():
        name = pathway_id_to_name.get(pathway, pathway)
        parts = name.split("-")
        name = parts[0]
        pathway_means[name] = np.mean(scores)
    
    return pd.Series(pathway_means).sort_values(ascending=False)

def pred_genotype_pathways(combined=False):
    """
    Calculate pathway scores for the gnn and save the results.
    """
    combined_graph, all_organs = get_combined_adj_mat(task="genotype")
    df = load_genotype_data()
    results = {}
    for split in range(10):
        print(f"Split {split+1}")
        X_train, y_train, X_val, y_val, X_test, y_test, label_mapping = preprocess_genotype_data(df, random_state=split)
        
        output_dim = y_train.nunique()
        params = load_hyperparameters(
            model_key="GAT_real",
            dataset_key="genotype",
        )
        base_params = {
            "max_epochs": 500,
            "patience": 20,
            "graph": combined_graph,
            "verbose": False,
            "output_dim": output_dim,
            "task": "classification",
        }
        model = GATWrapper(**params, **base_params)
        model.train(X_train, y_train, X_val, y_val)
        
        if not combined:
            ds_masks, wt_masks = model.explain(X_test, y_test)    
            ds_pathway_scores = score_genotype_pathways(ds_masks)
            wt_pathway_scores = score_genotype_pathways(wt_masks)
            results[split] = {
                "ds_pathway_scores": ds_pathway_scores,
                "wt_pathway_scores": wt_pathway_scores,
            }
        
        else:
            masks = model.explain(X_test, y_test, combined)
            masks.sort_values(ascending=False, inplace=True)
            pathway_scores = score_genotype_pathways(masks)
            results[split] = {
                "masks": masks,
                "pathway_scores": pathway_scores,
            }
    
    if not combined:
        ds_pathway_scores = pd.DataFrame([res["ds_pathway_scores"] for res in results.values()])
        ds_pathway_scores = ds_pathway_scores.agg(["mean", "std"]).T.sort_values(by="mean", ascending=False)
        ds_pathway_scores.to_csv("results/real/ds_pathway_scores.csv")
        
        wt_pathway_scores = pd.DataFrame([res["wt_pathway_scores"] for res in results.values()])
        wt_pathway_scores = wt_pathway_scores.agg(["mean", "std"]).T.sort_values(by="mean", ascending=False)
        wt_pathway_scores.to_csv("results/real/wt_pathway_scores.csv")
    else:
        masks = pd.DataFrame([res["masks"] for res in results.values()])
        masks = masks.agg(["mean", "std"]).T.sort_values(by="mean", ascending=False)
        masks.to_csv("results/real/genotype_pathway_masks.csv")
        
        pathway_scores = pd.DataFrame([res["pathway_scores"] for res in results.values()])
        pathway_scores = pathway_scores.agg(["mean", "std"]).T.sort_values(by="mean", ascending=False)
        pathway_scores.to_csv("results/real/genotype_pathway_scores.csv")
        
def create_latex_tables():
    """
    Create latex tables for the metabolite predictions and disease scores.
    """
    
    m_df = pd.read_csv("results/real/metabolomic_results.csv", index_col=0)
    m_latex = m_df.to_latex(
        float_format="%.3f",
        index=True,
        label="tab:metabolomic_results",
        caption="Metabolomic prediction results.",
    )
    
    with open("results/real/metabolomic_results.tex", "w") as f:
        f.write(m_latex)
    
    d_df = pd.read_csv("results/real/disease_scores.csv", index_col=0)
    d_df = d_df.head(10)
    d_latex = d_df.to_latex(
        float_format="%.3f",
        index=True,
        label="tab:disease_scores",
        caption="Disease prediction results.",
    )
    with open("results/real/disease_scores.tex", "w") as f:
        f.write(d_latex)
        
    ds_pathway_scores = pd.read_csv("results/real/ds_pathway_scores.csv", index_col=0)
    ds_pathway_scores = ds_pathway_scores.head(10)
    ds_latex = ds_pathway_scores.to_latex(
        float_format="%.3f",
        index=True,
        label="tab:ds_pathway_scores",
        caption="Down syndrome pathway scores.",
    )
    with open("results/real/ds_pathway_scores.tex", "w") as f:
        f.write(ds_latex)
    
    wt_pathway_scores = pd.read_csv("results/real/wt_pathway_scores.csv", index_col=0)
    wt_pathway_scores = wt_pathway_scores.head(10)
    wt_latex = wt_pathway_scores.to_latex(
        float_format="%.3f",
        index=True,
        label="tab:wt_pathway_scores",
        caption="Wild type pathway scores.",
    )
    with open("results/real/wt_pathway_scores.tex", "w") as f:
        f.write(wt_latex)
        
    genotype_results = pd.read_csv("results/real/genotype_results.csv", index_col=0)
    genotype_latex = genotype_results.to_latex(
        float_format="%.3f",
        index=True,
        label="tab:genotype_results",
        caption="Genotype prediction results.",
    )
    with open("results/real/genotype_results.tex", "w") as f:
        f.write(genotype_latex)
        
    genotype_pathway_scores = pd.read_csv("results/real/genotype_pathway_scores.csv", index_col=0)
    genotype_pathway_scores = genotype_pathway_scores.head(10)
    genotype_pathway_latex = genotype_pathway_scores.to_latex(
        float_format="%.3f",
        index=True,
        label="tab:genotype_pathway_scores",
        caption="Genotype pathway scores.",
    )
    with open("results/real/genotype_pathway_scores.tex", "w") as f:
        f.write(genotype_pathway_latex)


# Set seeds 
np.random.seed(0)
torch.manual_seed(0)
# tune_gnn()
# tune_xgboost()
# eval_models()
# pred_pathways()
# pred_diseases()
# create_latex_tables()

# tune_gnn("sex", genotype="D")
# tune_gnn("sex", genotype="W")
# tune_xgboost("sex", genotype="D")
# tune_xgboost("sex", genotype="W")
# eval_models("sex", genotype="D")
# eval_models("sex", genotype="W")

# tune_xgboost(task="genotype")
# eval_models(task="genotype")
pred_genotype_pathways(combined=True)
create_latex_tables()