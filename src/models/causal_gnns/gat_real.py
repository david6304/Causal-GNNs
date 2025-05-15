import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.explain import GNNExplainer, Explainer

class GAT(nn.Module):
    def __init__(
        self,
        output_dim,
        hidden_dim=32,
        dropout=0.0,
        num_layers=2,
        heads=8,
    ):

        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_layers = num_layers

        self.in_layer = nn.Linear(1, hidden_dim)

        self.gat_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.residuals = nn.ModuleList()

        self.gat_layers.append(
            GATConv(hidden_dim, hidden_dim, heads=heads, dropout=dropout, concat=True)
        )
        self.norm_layers.append(nn.LayerNorm(hidden_dim * heads))
        self.residuals.append(nn.Linear(hidden_dim, hidden_dim * heads))

        prev_dim = hidden_dim * heads
        for _ in range(1, num_layers):
            self.gat_layers.append(
                GATConv(prev_dim, hidden_dim, heads=1, concat=False, dropout=dropout)
            )
            self.norm_layers.append(nn.LayerNorm(hidden_dim))
            self.residuals.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim

 
        final_dim = hidden_dim * heads if num_layers == 1 else hidden_dim
        self.pool = global_mean_pool
        self.out_layer = nn.Linear(final_dim, output_dim)

    def forward(self, x, edge_index, batch=None):
        # Explainer cannot pass batch but will be a single graph
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long)
            
        x = self.in_layer(x)

        for i, conv in enumerate(self.gat_layers):
            residual = self.residuals[i](x)
            x = conv(x, edge_index)
            if i < self.num_layers - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.norm_layers[i](x + residual)    
            
        x = self.pool(x, batch=batch)
        return self.out_layer(x)

    def __repr__(self):
        return "GAT"

class GATWrapper(nn.Module):
    def __init__(
        self,
        output_dim,
        lr,
        graph,
        max_epochs=100,
        patience=10,
        hidden_dim=32,
        dropout=0.0,
        num_layers=2,
        heads=8,
        verbose=False,
        task="regression"
    ):
        super().__init__()
        
        graph_tensor = torch.tensor(graph.values, dtype=torch.float32)
        self.edge_index = graph_tensor.nonzero(as_tuple=False).t()
        
        self.model = GAT(
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_layers=num_layers,
            heads=heads
        )
    
        self.task = task
        if task == "regression":
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)
        self.max_epochs = max_epochs
        self.patience = patience
        self.verbose = verbose
        
    
    def train(self, X_train, y_train, X_val, y_val):
        """
        Takes train and val seperately for compatibility with hyperparameter tuning.
        """
        best_val_loss, best_state = float("inf"), None
        p_counter = 0
        
        X_train = torch.tensor(X_train.values, dtype=torch.float32)
        y_train = torch.tensor(y_train.values, dtype=torch.float32)
        X_val = torch.tensor(X_val.values, dtype=torch.float32)
        y_val = torch.tensor(y_val.values, dtype=torch.float32)
        
        if self.task == "classification":
            y_train = y_train.long()
            y_val = y_val.long()
        
        train_list = []
        for x, y in zip(X_train, y_train):
            data = Data(x=x.unsqueeze(-1), edge_index=self.edge_index, y=y)
            train_list.append(data)
        train_loader = DataLoader(train_list, batch_size=4, shuffle=True)
        
        val_list = []
        for x, y in zip(X_val, y_val):
            data = Data(x=x.unsqueeze(-1), edge_index=self.edge_index, y=y)
            val_list.append(data)
        val_loader = DataLoader(val_list, batch_size=4, shuffle=False)
        
        for epoch in range(self.max_epochs):                
            self.model.train()
            train_loss = 0
            for batch in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch.x, batch.edge_index, batch.batch)
                if self.task == "classification":
                    y = batch.y.view(-1)
                else:
                    y = batch.y.view(batch.batch_size, -1)
                loss = self.criterion(outputs, y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * batch.num_graphs
            train_loss /= len(train_loader.dataset)
            
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    outputs = self.model(batch.x, batch.edge_index, batch.batch)
                    if self.task == "classification":
                        y = batch.y.view(-1)
                    else:
                        y = batch.y.view(batch.batch_size, -1)
                    loss = self.criterion(outputs, y)
                    val_loss += loss.item() * batch.num_graphs
            val_loss /= len(val_loader.dataset)
            
            if self.verbose:
                print(f"Epoch {epoch+1}/{self.max_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = self.model.state_dict()
                p_counter = 0
            else:
                p_counter += 1
                if p_counter >= self.patience:
                    if self.verbose:
                        print(f"Early stopping at epoch {epoch} with val loss {best_val_loss}")
                    break
        
        self.model.load_state_dict(best_state)
                
    def predict(self, X):
        X = torch.tensor(X.values, dtype=torch.float32)
        data_list = []
        for x in X:
            data = Data(x=x.unsqueeze(-1), edge_index=self.edge_index)
            data_list.append(data)
        batch = Batch.from_data_list(data_list)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(batch.x, batch.edge_index, batch.batch)
        if self.task == "classification":
            outputs = torch.argmax(outputs, dim=1)
        return outputs
    
    def explain(self, X, y, combined=False):
        """
        Explain the model's predictions using GNNExplainer.
        """
        self.model.eval()
        explainer = Explainer(
            model=self.model,
            algorithm=GNNExplainer(epochs=200),
            explanation_type='model',
            node_mask_type='attributes',
            edge_mask_type='object',
            model_config=dict(
                mode='multiclass_classification',
                task_level='node',
                return_type='log_probs',
            ),
        )
        cols = X.columns
        X = torch.tensor(X.values, dtype=torch.float32)
        y = torch.tensor(y.values, dtype=torch.long)
        
        if not combined:
            ds_masks, wt_masks = [], []
            for x, y in zip(X, y):
                data = Data(x=x.unsqueeze(-1), edge_index=self.edge_index, y=y)
                explanation = explainer(data.x, data.edge_index)
                node_mask = explanation.node_mask.squeeze()
                if y:
                    ds_masks.append(node_mask)
                else:
                    wt_masks.append(node_mask)
            
            ds_masks = torch.stack(ds_masks).mean(dim=0).numpy()
            wt_masks = torch.stack(wt_masks).mean(dim=0).numpy()
            
            ds_masks = pd.Series(ds_masks, index=cols, name="Down Syndrome Importance")
            wt_masks = pd.Series(wt_masks, index=cols, name="Wild Type Importance")
            
            return ds_masks, wt_masks
        else:
            masks = []
            for x, y in zip(X, y):
                data = Data(x=x.unsqueeze(-1), edge_index=self.edge_index, y=y)
                explanation = explainer(data.x, data.edge_index)
                node_mask = explanation.node_mask.squeeze()
                masks.append(node_mask)
            
            masks = torch.stack(masks).mean(dim=0).numpy()
            masks = pd.Series(masks, index=cols, name="Combined Importance")
            return masks
            