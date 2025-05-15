from pathlib import Path
import pandas as pd
from src.utils.config_loader import load_config


def load_synthetic_data(dataset="asia", variant="base", level=None):
    """
    Load synthetic data for a given dataset and variant
    """
    cfg = load_config()
    synth_cfg = cfg['data']['synthetic']

    files = []

    folder = Path(synth_cfg[dataset]) / variant
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    files = sorted(folder.glob("*.csv"))

    # If noise or missing variant without level then return dict with all levels
    if variant in ('noise', 'missing') and level is None:
        dfs_dict = {}
        for file in files:
            if file and file.exists():
                # Get level from filename
                stem = file.stem 
                lvl = float(stem.split('_')[1])
                dfs_dict[lvl] = pd.read_csv(file)
        return dfs_dict


    dfs = []
    for file in files:
        if file and file.exists():
            dfs.append(pd.read_csv(file))

    return dfs[0] if len(dfs) == 1 else dfs

def load_real_data():
    p_df = pd.read_csv("data/real/metab/processed/imputed/all_organs_imputed.csv", index_col=0)
    m_df = pd.read_csv("data/real/metabolomics.csv", index_col=0).T

    map_df = pd.read_csv("data/real/sample_mapping.csv", index_col=0)
    
    metabolomics_ids = list(m_df.index)
    proteomics_ids = list(p_df.index)
    
    mapping = {}
    
    for m_id in metabolomics_ids:
        id_no = m_id.split("_")[-1]
        if id_no not in map_df.index:
            continue

        cols = map_df.columns.tolist()
        start = cols.index("NAS (NASH score)") + 1
        next_cols = cols[start : start + 3]

        p_id = None
        for col in next_cols:
            vals = map_df.loc[id_no, col]
            if isinstance(vals, pd.Series):
                # Get the first non-null value
                cleaned_vals = vals.dropna().astype(str).str.strip().loc[lambda x: x != ""]
                if cleaned_vals.empty:
                    continue
                val = cleaned_vals.iloc[0]
            else:
                val = vals
                if pd.isna(val) or not str(val).strip():
                    continue
                
            parts = str(val).split("_")
            if not parts[0].startswith("D"):
                continue

            parts[0] = "DP1Y"
            p_id = "_".join([parts[0]] + parts[2:])
            break

        if p_id and p_id in proteomics_ids:
            mapping[m_id] = p_id

    # Remove metabolomics rows without mapping then rename the rows using mapping
    m_df = m_df.loc[list(mapping.keys())]
    m_df.rename(index=mapping, inplace=True)
    m_df = m_df.reindex(p_df.index).dropna()
    
    # Remove rows from proteomics that are not in metabolomics
    p_df = p_df.loc[m_df.index]
    p_df = p_df.reindex(m_df.index).dropna()
    
    return p_df, m_df

def load_genotype_data(target='genotype', genotype='D'):
    """
    Load the proteomics data with the genotypes as labels.
    """
    df = pd.read_csv("data/real/genotype/processed/imputed/all_organs_imputed.csv", index_col=0)
    df['genotype'] = df.index.str[0]
    if target == 'sex':
        df['sex'] = df.index.str.split("_").str[1]
        df = df[df['genotype'] == genotype]
        df.drop(columns=['genotype'], inplace=True)
    elif target == 'diet':
        df['diet'] = df.index.str.split("_").str[2]
        df = df[df['genotype'] == genotype]
        df.drop(columns=['genotype'], inplace=True)
    
    return df