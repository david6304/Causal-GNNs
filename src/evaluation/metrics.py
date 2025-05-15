import numpy as np
   
def structural_hamming_distance(true_adj, learned_adj):
    """Calculate directed SHD, extra edges, and missing edges"""

    nodes = true_adj.columns.tolist()
    true_adj = true_adj.loc[nodes, nodes].astype(int)
    learned_adj = learned_adj.loc[nodes, nodes].astype(int)

    extra = int(((learned_adj != 0) & (true_adj == 0)).sum().sum())
    missing = int(((true_adj == 1) & (learned_adj != 1)).sum().sum())

 
    directed_shd = int((true_adj != learned_adj).sum().sum())

    return {
        "directed_shd": directed_shd,
        "extra_edges": extra,
        "missing_edges": missing
    }
    
def intervene_and_resample_model(data, model, intervention_node, intervention_value):
    """
    Predict all descendants of the intervention node using the given model.
    """
    data[intervention_node] = intervention_value

    preds = model.predict(data, starting_node=intervention_node)
    
    for node, pred in preds.items():
        data[node] = pred
    return data
    
def calculate_ace(model, data, intervention_node, effect_node):
    """
    Calculate the average causal effect (ace) of an intervention on a node using a given model.
    """
    classes = data[intervention_node].unique()
    classes.sort()

    do_interventions = []
    for c in classes:
        intervened_data = intervene_and_resample_model(data.copy(), model, intervention_node, c)
        do_interventions.append(intervened_data)

    do_0 = do_interventions[0]
    do_interventions = do_interventions[1:]

    ace = np.mean([do_interventions[i][effect_node] - do_0[effect_node] for i in range(len(do_interventions))])
    return ace
