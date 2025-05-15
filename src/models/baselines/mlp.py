import torch
import torch.nn as nn
class MLP(nn.Module):
    def __init__(self, input_dim: int, num_layers: int, hidden_dim: int, dropout: float, output_dim: int):
        super(MLP, self).__init__()
        layers = []
        prev_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class MLPBaseline:
    def __init__(
        self,
        input_dim,
        num_layers=2,
        hidden_dim=64,
        dropout=0.2,
        lr=0.01,
        max_epochs=20,
        patience=15,
        batch_size=32,
        verbose=False,
        num_classes=2
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.verbose = verbose
        self.num_classes = num_classes
    
        self.output_dim = self.num_classes
        self.criterion = nn.CrossEntropyLoss()
        
        self.model = MLP(
            input_dim=self.input_dim,
            num_layers=self.num_layers,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
            output_dim=self.output_dim
        )
        
    def __repr__(self):
        return "MLPBaseline"

    def train(self, X_train, y_train, X_val, y_val):
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Create torch tensors.
        X_train_tensor = torch.tensor(X_train.values, dtype=torch.float)
        X_val_tensor = torch.tensor(X_val.values, dtype=torch.float)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
        y_val_tensor = torch.tensor(y_val.values, dtype=torch.long)

        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

        best_val_loss = float('inf')
        p_counter = 0
  
        for epoch in range(self.max_epochs):
            self.model.train()
            train_loss = 0.0
            for X, y in train_dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(X)
                loss = self.criterion(outputs, y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * X.size(0)
            train_loss /= len(train_dataset)
    
        
            self.model.eval()
            with torch.no_grad():
                val_loss = 0.0
                for X_val, y_val in val_dataloader:
                    outputs = self.model(X_val)
                    loss = self.criterion(outputs, y_val)
                    val_loss += loss.item() * X_val.size(0)
            val_loss /= len(val_dataset)
            
            if self.verbose:
                print(f"Epoch {epoch+1}/{self.max_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.best_model = self.model.state_dict()
                p_counter = 0
            else:
                p_counter += 1
                if p_counter >= self.patience:
                    if self.verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break
        
        self.model.load_state_dict(self.best_model)


    def predict(self, X):
        X = X.copy()
        X = torch.tensor(X.values, dtype=torch.float)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X)
            
        predictions = torch.argmax(outputs, dim=1)
        return predictions
