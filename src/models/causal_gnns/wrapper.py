import torch
import networkx as nx
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
import random
from src.models.causal_gnns.gcn import GCN

class GNNWrapper:
    def __init__(
        self,
        model_class,
        graph,
        node_classes,
        embedding_dim,
        hidden_dim,
        lr,
        batch_size=4,
        max_epochs=50,
        patience=10,
        num_layers=2,
        dropout=0.0,
        heads=4,
        verbose=False
    ):
        """
        Wrapper class to train and evaluate the GNN models. 
        """
        self.verbose = verbose
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        
        # Convert df graph into edge_index for the GNNs
        graph_tensor = torch.tensor(graph.values, dtype=torch.float32)
        self.edge_index = graph_tensor.nonzero(as_tuple=False).t()
        
        self.node_classes = node_classes
        self.num_nodes = len(node_classes)
        self.mask_token = int(max(node_classes))
        
        self.graph = nx.from_pandas_adjacency(graph, create_using=nx.DiGraph)
        
        if model_class != GCN:
            extra_params = {"heads": heads}
        else:   
            extra_params = {}
        
        self.model = model_class(
            node_classes=node_classes,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            **extra_params
        )

    def __repr__(self):
        return f"GNNWrapper({self.model.__class__.__name__})"
    
    def sim_interventions(self, X):
        """ 
        Simulates a do-intervention for each row the graph by randomly selecting a node and masking its descendants.
        """
        X = X.copy()
        # If graph is fully connected then mask random subset of nodes
        if len(self.graph.edges) == self.num_nodes * (self.num_nodes - 1):
            for row in range(X.shape[0]):
                num_masked = random.randint(1, self.num_nodes)
                nodes = random.sample(range(self.num_nodes), num_masked)
                for node in nodes:
                    col = X.columns[node]
                    pos = X.columns.get_loc(col)
                    X.iat[row, pos] = self.mask_token
        else:           
            for row in range(X.shape[0]):
                node = random.randint(0, self.num_nodes - 1)
                col = X.columns[node]
                for desc in nx.descendants(self.graph, col):
                    pos = X.columns.get_loc(desc)
                    X.iat[row, pos] = self.mask_token 
        return X
        
    
    def train(self, X_train, y_train, X_val, y_val):
        """ Trains a GNN model using the validation set for early stopping."""
        
        self.columns = X_train.columns.tolist()
    
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = torch.nn.CrossEntropyLoss()
        
        # Simulate interventions
        X_train = self.sim_interventions(X_train)
        X_val = self.sim_interventions(X_val)
        
        X_train = torch.tensor(X_train.values, dtype=torch.long)
        y_train = torch.tensor(y_train.values, dtype=torch.long)
        X_val = torch.tensor(X_val.values, dtype=torch.long)
        y_val = torch.tensor(y_val.values, dtype=torch.long)
        
        # For batching
        train_list = []
        for x, y in zip(X_train, y_train):
            data = Data(x=x.unsqueeze(-1), edge_index=self.edge_index, y=y)
            train_list.append(data)
        train_loader = DataLoader(train_list, batch_size=self.batch_size, shuffle=True, num_workers=0)
        val_list = []
        for x, y in zip(X_val, y_val):
            data = Data(x=x.unsqueeze(-1), edge_index=self.edge_index, y=y)
            val_list.append(data)
        val_loader = DataLoader(val_list, batch_size=self.batch_size, shuffle=False, num_workers=0)        
        best_val_loss, best_state = float("inf"), None
        p_counter = 0 
        
        for epoch in range(self.max_epochs):
            self.model.train()
            train_loss = 0
            total_nodes = 0
            for batch in train_loader:
                logits_list = self.model(batch.x, batch.edge_index)
                batch_size = batch.x.size(0) // self.num_nodes
                y = batch.y.view(batch_size, self.num_nodes)
                losses = []
                for i in range(self.num_nodes):
                    loss = self.criterion(logits_list[i], y[:, i])
                    losses.append(loss)
                loss = torch.stack(losses).mean()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * batch.x.size(0)
                total_nodes += batch.x.size(0)
            train_loss /= total_nodes
                
            self.model.eval()
            val_loss = 0
            total_nodes = 0
            with torch.no_grad():
                for batch in val_loader:
                    logits_list = self.model(batch.x, batch.edge_index)
                    batch_size = batch.x.size(0) // self.num_nodes
                    y = batch.y.view(batch_size, self.num_nodes)
                    losses = []
                    for i in range(self.num_nodes):
                        loss = self.criterion(logits_list[i], y[:, i])
                        losses.append(loss)
                    loss = torch.stack(losses).mean()
                    val_loss += loss.item() * batch.x.size(0)
                    total_nodes += batch.x.size(0)
            val_loss /= total_nodes
                    
            
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
                        print(f"Early stopping at epoch {epoch+1}")
                    break
        
        self.model.load_state_dict(best_state)

    def predict(self, X, starting_node=None):
        # Fill missing values with mask token
        X = X.copy()
        X = X.fillna(self.mask_token)
        X_t = torch.tensor(X.values, dtype=torch.long)
        
        if starting_node is not None:
            # Mask descendant of starting nodes
            for desc in nx.descendants(self.graph, starting_node):
                pos = X.columns.get_loc(desc)
                X_t[:, pos] = self.mask_token
            # Remove incoming edges from graph
            starting_node = X.columns.get_loc(starting_node)
            edge_index = self.edge_index[:, self.edge_index[1] != starting_node]
        else:
            edge_index = self.edge_index
        
        data_list = [
            Data(x=x.unsqueeze(-1), edge_index=edge_index)
            for x in X_t
        ]
        batch = Batch.from_data_list(data_list)
        
        self.model.eval()
        with torch.no_grad():
            logits_list = self.model(batch.x, batch.edge_index)
        
        return {col: logits_list[i].argmax(dim=1) for i, col in enumerate(self.columns)}
