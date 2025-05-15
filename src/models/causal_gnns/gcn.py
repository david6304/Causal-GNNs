import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    def __init__(
        self,
        node_classes,
        embedding_dim=16,
        hidden_dim=32,
        dropout=0.0,
        num_layers=2
    ):
        super().__init__()
        self.num_nodes = len(node_classes)
        self.node_classes = node_classes
        self.dropout = dropout
        self.num_layers = num_layers

        self.node_embeddings = nn.ModuleList([
            nn.Embedding(c + 1, embedding_dim)
            for c in node_classes
        ])  # +1 for mask token

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(embedding_dim, hidden_dim))
        for _ in range(1, num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, c)
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

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        x = x.view(batch_size, self.num_nodes, -1)

        logits = []
        for i, head in enumerate(self.heads):
            logits.append(head(x[:, i, :]))
        
        return logits
    
    def __repr__(self):
        return f"GCN"