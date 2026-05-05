from dataclasses import dataclass
from typing import List


from src.models.gnn_models import GINRegressor, GNNConfig



import numpy as np
import torch
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

from configs.base_config import BaseConfig
from src.data.loader import ResearchDataLoader
from src.data.graphs import GraphBuilder
from src.models.gnn_models import GINRegressor, GNNConfig
from src.utils.seed import set_seed
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

@dataclass
class GNNTrainConfig:
    batch_size: int = 64
    max_epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-4

def build_dataset(cfg: BaseConfig):
    loader = ResearchDataLoader(cfg)
    df, _ = loader.load_and_validate()

    gb = GraphBuilder()
    data_list = []
    targets = []

    for _, row in df.iterrows():
        g = gb.smiles_to_graph(row["SMILES"])
        if g is None:
            continue
        g.y = torch.tensor([float(row[cfg.training.target_column])], dtype=torch.float)
        data_list.append(g)
        targets.append(float(row[cfg.training.target_column]))

    print(f"📊 Built graph dataset: {len(data_list)} graphs")
    return data_list

def train_gnn():
    base_cfg = BaseConfig()
    base_cfg.training.target_column = "CO2_Uptake_mmolg"  # ensure target
    set_seed(base_cfg.training.random_seed)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")

    data_list = build_dataset(base_cfg)

    train_cfg = GNNTrainConfig()

    # train/val/test split (80/10/10)
    n_total = len(data_list)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    n_test = n_total - n_train - n_val

    train_data, val_data, test_data = random_split(
        data_list,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(base_cfg.training.random_seed),
    )

    train_loader = DataLoader(train_data, batch_size=train_cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=train_cfg.batch_size)
    test_loader = DataLoader(test_data, batch_size=train_cfg.batch_size)

    gnn_cfg = GNNConfig()
    model = GINRegressor(gnn_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(train_cfg.max_epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = torch.nn.functional.mse_loss(out, batch.y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs

        avg_loss = total_loss / len(train_data)

        # validation
        model.eval()
        val_losses: List[float] = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch)
                loss = torch.nn.functional.mse_loss(out, batch.y.view(-1))
                val_losses.append(loss.item())

        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()

        print(f"Epoch {epoch:03d} | Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f}")

    # test evaluation with best model
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch)
            y_true.extend(batch.y.view(-1).cpu().numpy())
            y_pred.extend(out.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    r2 = r2_score(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))

    print("📊 GNN Results on test set:")
    print(f"   R²   : {r2:.4f}")
    print(f"   RMSE : {rmse:.4f}")
    print(f"   MAE  : {mae:.4f}")

if __name__ == "__main__":
    train_gnn()