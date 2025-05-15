import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GAT(nn.Module):
    def __init__(
        self,
        node_classes,
        embedding_dim=16,
        hidden_dim=32,
        dropout=0.0,
        num_layers=2,
        heads=8,
    ):

        super().__init__()
        self.num_nodes = len(node_classes)
        self.node_classes = node_classes
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_layers = num_layers

        self.node_embeddings = nn.ModuleList([
            nn.Embedding(c + 1, embedding_dim)
            for c in node_classes
        ])

        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(
            GATConv(embedding_dim, hidden_dim, heads=heads, dropout=dropout, concat=True)
        )

        prev_dim = hidden_dim * heads
        for _ in range(1, num_layers):
            self.gat_layers.append(
                GATConv(prev_dim, hidden_dim, heads=1, concat=False, dropout=dropout)
            )
            prev_dim = hidden_dim

 
        final_dim = hidden_dim * heads if num_layers == 1 else hidden_dim

        self.heads = nn.ModuleList([
            nn.Linear(final_dim, c)
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

        for i, conv in enumerate(self.gat_layers):
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
        return "GAT"
