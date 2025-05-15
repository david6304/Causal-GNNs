from sklearn.model_selection import train_test_split


def preprocess_data(df, target_node=None, test_size=0.3, val_size=0.2, random_state=0):
    """
    Split data into train/val/test.
    If target_node is provided, split each into X and y.
    """
    df_copy = df.copy()

    # Determine splits
    if target_node:
        X = df_copy.drop(columns=[target_node])
        y = df_copy[target_node]


        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            test_size=test_size + val_size,
            stratify=y,
            random_state=random_state
        )
  
        val_ratio = val_size / (test_size + val_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=1-val_ratio,
            stratify=y_temp,
            random_state=random_state
        )
        return X_train, y_train, X_val, y_val, X_test, y_test

    # No target_node, split full DataFrame
    train, temp = train_test_split(
        df_copy,
        test_size=test_size + val_size,
        random_state=random_state
    )
    val_ratio = val_size / (test_size + val_size)
    val, test = train_test_split(
        temp,
        test_size=1-val_ratio,
        random_state=random_state
    )
    return train, val, test

def preprocess_omics_data(X, y, test_size=0.3, val_size=0.2, random_state=0):
    """
    Split omics data into train/val/test. Can be used to just split into train/val for cv.
    """
    indices = X.index.values
    train_indices, temp_indices = train_test_split(
        indices, test_size=test_size + val_size, random_state=random_state
    )
    val_indices, test_indices = train_test_split(
        temp_indices, test_size=val_size / (test_size + val_size), random_state=random_state
    )
    X_train, y_train = X.loc[train_indices], y.loc[train_indices]
    X_val, y_val = X.loc[val_indices], y.loc[val_indices]
    X_test, y_test = X.loc[test_indices], y.loc[test_indices]
    return X_train, y_train, X_val, y_val, X_test, y_test

def preprocess_genotype_data(df, target='genotype', test_size=0.3, val_size=0.2, random_state=0):
    """
    Split genotype data into train/val/test.
    """
    df_copy = df.copy()
    X, y = df_copy.drop(columns=[target]), df_copy[target]
    label_mapping = {label: i for i, label in enumerate(y.unique())}
    y = y.map(label_mapping)
    inv_mapping = {i: label for label, i in label_mapping.items()}
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=test_size + val_size,
        stratify=y,
        random_state=random_state
    )
    val_ratio = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=1-val_ratio,
        stratify=y_temp,
        random_state=random_state
    )
    return X_train, y_train, X_val, y_val, X_test, y_test, inv_mapping
