from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool, BatchNorm

@dataclass
class GNNConfig:
    node_input_dim: int = 5    # from GraphBuilder (we use 5 node features)
    hidden_dim: int = 64
    num_layers: int = 3
    output_dim: int = 1
    dropout: float = 0.1

class GINRegressor(nn.Module):
    def __init__(self, cfg: GNNConfig):
        super().__init__()
        self.cfg = cfg

        self.node_embed = nn.Linear(cfg.node_input_dim, cfg.hidden_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(cfg.num_layers):
            mlp = nn.Sequential(
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
                nn.ReLU(),
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            )
            self.convs.append(GINConv(mlp))
            self.norms.append(BatchNorm(cfg.hidden_dim))

        self.readout = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.output_dim),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.node_embed(x)
        x = F.relu(x)

        for conv, bn in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.cfg.dropout, training=self.training)

        x = global_mean_pool(x, batch)
        out = self.readout(x)
        return out.view(-1)
    
    


    

    def encode(self, data):
        """Return graph-level embedding without final linear output."""
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.node_embed(x)
        x = F.relu(x)

        for conv, bn in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)

        x = global_mean_pool(x, batch)
        return x  # [batch_size, hidden_dim]




if __name__ == "__main__":
    cfg = GNNConfig()
    model = GINRegressor(cfg)
    print(model)