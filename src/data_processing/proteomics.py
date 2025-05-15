import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

def process_organ(csv_path, organ_suffix, output_csv=None, ds_only=True):
    """
    Loads proteomics data for a given organ, filters to p<0.01 and then takes top 3000 by FC. 
    3000 is limit for DAVID clustering.
    """
    
    id_col = "Ensembl_gene_id"

    df = pd.read_csv(csv_path, low_memory=False)

    cols = df.columns.tolist()
    group_str = "Dp116Yey_all vs WT_all" if organ_suffix.upper() == "BAT" else "DP1Y_all vs WT_all"
    group_i = next((i for i, c in enumerate(cols) if group_str.lower() in c.lower()), None)
    next_cols = cols[group_i+1:] if group_i is not None else cols
    
    # Find p<0.01 column
    p_col = next((c for c in next_cols if "p<0.01" in c.lower()), None)
    if not p_col:
        raise KeyError(f"p<0.01 column not found for organ {organ_suffix}")
    
    df[p_col] = pd.to_numeric(df[p_col], errors="coerce")
    top_3000 = df[p_col].abs().nlargest(3000).index
    df = df.loc[top_3000]
    
    start = cols.index("Linear normalized counts") + 1
    end = cols.index("Linear normalized counts + 0.01")
    sample_cols = cols[start:end]
    
    # Only include DS mice
    if ds_only:
        sample_cols = [c for c in sample_cols if c.startswith("D")]
    id_column = next((c for c in cols if id_col in c), None)
    
    columns = [id_column] + sample_cols
    samples = df[columns].copy()
    samples = samples.set_index(id_column)[sample_cols].T
    
    if output_csv:
        out_dir = os.path.dirname(output_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        samples.to_csv(output_csv)
    return samples

def generate_gene_list(organ, input_csv, output_dir):
    """
    Generate a txt file of the emsemble ids to paste into DAVID
    """
    out_file = os.path.join(output_dir, f"{organ}.txt")
    if os.path.isfile(out_file):
        print(f"{out_file} already exists; skipping gene list generation.")
        return
    
    df = pd.read_csv(input_csv, index_col=0)

    ids = [col.split("_")[0] for col in df.columns]
    unique_ids = list(dict.fromkeys(ids))
    os.makedirs(output_dir, exist_ok=True)
    with open(out_file, "w") as f:
        for gid in unique_ids:
            f.write(f"{gid}\n")
    print(f"Generated gene list for {organ}: {out_file}")

def aggregate_clusters(organ, input_csv, cluster_dir, output_csv):
    """
    Group proteins into DAVID clusters
    """
    if os.path.isfile(output_csv):
        print(f"{output_csv} already exists; skipping cluster aggregation for {organ}.")
        return

    files = os.listdir(cluster_dir)
    match = next((f for f in files if f.lower() == f"{organ.lower()}_clusters.txt"), None)
    
    gene_to_cluster = {}
    with open(os.path.join(cluster_dir, match), "r") as f:
        curr_cluster = "Unclustered"
        for line in f:
            line = line.strip()
            if line.startswith("Gene Group"):
                curr_cluster = line.split(" ")[2].split("\t")[0]
            elif line and not line.startswith("ENSEMBL_GENE_ID"):
                ensembl_id = line.split("\t")[0]
                gene_to_cluster[ensembl_id] = curr_cluster
    
    df = pd.read_csv(input_csv, index_col=0)
    
    # Normalise before aggregation
    df = np.log2(df + 1e-6)
    medians = df.median(axis=1)
    df = df.sub(medians, axis=0)
    df = df.T.groupby(gene_to_cluster).mean().T
    
    
    out_dir = os.path.dirname(output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(output_csv)
    

def combine_clustered_matrices(organs, processed_dir, output_filename, n_clusters=30):
    """
    Combine all organs' clustered CSVs into a single table with clusters as rows and samples as columns.
    """
    combined_dfs = []
    for organ in organs:
        file_path = os.path.join(processed_dir, f"clustered/{organ}_clustered.csv")
        if not os.path.isfile(file_path):
            print(f"Missing clustered file for {organ}: {file_path}, skipping.")
            continue
        df = pd.read_csv(file_path, index_col=0)
        
        # Filter to top n_clusters columns by variance
        mask = df.var().nlargest(n_clusters).index
        df = df[mask]
        
        # Make BAT DS sample name consistent with others
        df.index = [idx.replace("Dp161Yey", "DP1Y") for idx in df.index]
        
        # Remove organ code from index
        new_index = []
        for i in df.index:
            parts = i.split("_")
            if len(parts) > 1 and parts[1] not in ("F", "M"):
                parts.pop(1)
            new_index.append("_".join(parts))
        df.index = new_index
        
        # Add organ name to cluster numbers
        df.columns = [f"{col}_{organ}" for col in df.columns]
        combined_dfs.append(df)
        
    if not combined_dfs:
        print("No clustered dataframes to concatenate.")
        return
    df = pd.concat(combined_dfs, axis=1, join='outer')
    
    out_path = os.path.join(processed_dir, output_filename)
    if os.path.isfile(out_path):
        print(f"{out_path} already exists; skipping concatenation.")
    else:
        df.to_csv(out_path)
        print(f"Saved combined clustered CSV to {out_path}")

def impute_data(processed_dir):
    """
    Impute missing data in the combined clustered CSV.
    """
    folder = Path(processed_dir, "combined")
    files = sorted(folder.glob("*_combined.csv"))
    df_dict = {}
    for file in files:
        if file and file.exists():
            df_dict[file.stem] = pd.read_csv(file)
    
    for key, df in df_dict.items():
        if key == "all_organs_combined":
            out_path= os.path.join(processed_dir, "imputed/all_organs_imputed.csv")
        else:
            organs = key.split("_")[0:2]
            out_path = os.path.join(processed_dir, f"imputed/{organs[0]}_{organs[1]}_imputed.csv")
        if os.path.isfile(out_path):
            print(f"{out_path} already exists; skipping imputation.")
        else:
            df = df.set_index(df.columns[0])
            print(f"Imputing data for {key}...")
            imputer = IterativeImputer(max_iter=25, random_state=0, verbose=2)
            imputed_data = imputer.fit_transform(df)
            
            # Save imputed data
            imputed_df = pd.DataFrame(imputed_data, columns=df.columns, index=df.index)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            imputed_df.to_csv(out_path)
            print(f"Saved imputed data to {out_path}")

def get_proteins_in_cluster(cluster_id, cluster_dir):
    """
    Get the proteins in a given cluster.
    """
    cluster_no, organ = cluster_id.split("_")
    cluster_file = os.path.join(cluster_dir, f"{organ}_clusters.txt")
    proteins = set()
    with open(cluster_file, "r") as f:
        curr_cluster = None
        for line in f:
            if line.startswith("Gene Group"):
                if proteins:
                    break
                curr_cluster = line.split(" ")[2].split("\t")[0]
            elif not line.startswith("ENSEMBL_GENE_ID") and curr_cluster == cluster_no:
                proteins.add(line.split("\t")[0])
    return proteins

def get_cluster_to_proteins(cluster_dir, cluster_ids):
    """
    Get a dictionary of cluster IDs to proteins in that cluster.
    """
    cluster_to_proteins = {}
    for cluster_id in cluster_ids:
        proteins = get_proteins_in_cluster(cluster_id, cluster_dir)
        if proteins:
            cluster_to_proteins[cluster_id] = proteins
    return cluster_to_proteins

def generate_csvs(task="metab", ds_only=True):

    organs = [
        ("liver", "LIV", "data/real/proteomics/liver.csv"),
        ("BAT",   "BAT", "data/real/proteomics/BAT.csv"),
        ("blood", "BLOOD", "data/real/proteomics/blood.csv"),
    ]
    processed_dir = f"data/real/{task}/processed"
    gene_list_dir = "data/real/gene_ids"
    cluster_dir = "data/real/clusters"

    # Process organs to get top 3000 proteins
    for organ, suffix, path in organs:
        output_csv = os.path.join(processed_dir, f"{organ}_processed.csv")
        if os.path.isfile(output_csv):
            print(f"{output_csv} already exists; skipping processing for {output_csv}.")
        else:
            process_organ(path, organ_suffix=suffix, output_csv=output_csv, ds_only=ds_only)

    # Generate DAVID gene lists from processed csvs
    for organ, suffix, _ in organs:
        input_csv = os.path.join(processed_dir, f"{organ}.csv")
        generate_gene_list(organ, input_csv, gene_list_dir)

    # Aggregate by DAVID clusters
    for organ, suffix, _ in organs:
        input_csv = os.path.join(processed_dir, f"{organ}_processed.csv")
        output_csv = os.path.join(processed_dir, f"clustered/{organ}_clustered.csv")
        aggregate_clusters(organ, input_csv, cluster_dir, output_csv)

    organs = ['liver', 'BAT', 'blood']
    combine_clustered_matrices(organs, processed_dir, "combined/all_organs_combined.csv")

    # Create csv for each organ pairs
    for i, organ1 in enumerate(organs):
        for j, organ2 in enumerate(organs):
            if i >= j:
                continue
            combine_clustered_matrices([organ1, organ2], processed_dir, f"combined/{organ1}_{organ2}_combined.csv")

    impute_data(processed_dir)

# generate_csvs()
generate_csvs(task="genotype", ds_only=False)