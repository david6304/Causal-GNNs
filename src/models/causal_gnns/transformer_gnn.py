import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv

class TransformerGNN(nn.Module):
    def __init__(
        self,
        node_classes,
        embedding_dim=16,
        hidden_dim=32,
        dropout=0.0,
        num_layers=2,
        heads=8
    ):

        super().__init__()
        self.num_nodes = len(node_classes)
        self.node_classes = node_classes
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_layers = num_layers
        self.heads_count = heads

        self.node_embeddings = nn.ModuleList([
            nn.Embedding(c + 1, embedding_dim)
            for c in node_classes
        ])

        # Transformer blocks
        self.attention_layers = nn.ModuleList()
        self.attention_norms = nn.ModuleList()
        self.residuals = nn.ModuleList()
        self.ffns = nn.ModuleList()
        self.ffn_norms = nn.ModuleList()
        
        prev_dim = embedding_dim
        for _ in range(num_layers):
            self.attention_layers.append(
                TransformerConv(prev_dim, hidden_dim, heads=heads, dropout=dropout)
            )
            self.attention_norms.append(nn.LayerNorm(hidden_dim * heads))
            self.residuals.append(nn.Linear(prev_dim, hidden_dim * heads))

            self.ffns.append(
                nn.Sequential(
                    nn.Linear(hidden_dim * heads, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim * heads)
                )
            )
            self.ffn_norms.append(nn.LayerNorm(hidden_dim * heads))
            prev_dim = hidden_dim * heads
        
        self.final_conv = TransformerConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout)
        self.final_norm = nn.LayerNorm(hidden_dim * heads)
        
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim * heads, c)
            for c in node_classes
        ])

    def forward(self, x, edge_index):
        """
        x has shape (batch_size * num_nodes, 1)
        """
        total = x.size(0)
        batch_size = total // self.num_nodes
        x = x.view(batch_size, self.num_nodes).squeeze(-1)

        embeddings = []
        for i, emb in enumerate(self.node_embeddings):
            obs = x[:, i] 
            # Mask token needs to be clamped to be within embedding range for some nodes
            obs = obs.clamp(min=0, max=self.node_classes[i])
            emb_i = emb(obs)
            embeddings.append(emb_i.unsqueeze(1))
        x = torch.cat(embeddings, dim=1)
        x = x.view(batch_size * self.num_nodes, -1)

        for i in range(self.num_layers):
            # Multi head attention
            residual = self.residuals[i](x)
            x = self.attention_layers[i](x, edge_index)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.attention_norms[i](x + residual)

            residual = x
            x = self.ffns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.ffn_norms[i](x + residual)

        x = self.final_conv(x, edge_index)
        x = self.final_norm(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = x.view(batch_size, self.num_nodes, -1)

        logits = []
        for i, head in enumerate(self.heads):
            logits.append(head(x[:, i, :])) 

        return logits

    def __repr__(self):
        return f"TransformerGNN(num_layers={self.num_layers})"