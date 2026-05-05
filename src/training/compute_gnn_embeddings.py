import numpy as np
import torch
from torch_geometric.loader import DataLoader
from pathlib import Path

from configs.base_config import BaseConfig
from src.data.loader import ResearchDataLoader
from src.data.graphs import GraphBuilder
from src.models.gnn_models import GINRegressor, GNNConfig
from src.utils.seed import set_seed

def main():
    cfg = BaseConfig()
    set_seed(cfg.training.random_seed)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🖥️ Using device for embeddings: {device}")

    # 1. Load cleaned data
    loader = ResearchDataLoader(cfg)
    df, _ = loader.load_and_validate()

    # 2. Build graphs
    gb = GraphBuilder()
    data_list = []
    idx_list = []

    for idx, row in df.iterrows():
        g = gb.smiles_to_graph(row["SMILES"])
        if g is None:
            continue
        g.idx = idx  # keep index to align later
        data_list.append(g)
        idx_list.append(idx)

    print(f"📊 Graphs for embeddings: {len(data_list)}")

    # 3. Initialize GNN encoder (untrained for now; or you can load trained weights)
    gnn_cfg = GNNConfig()
    model = GINRegressor(gnn_cfg).to(device)
    model.eval()

    loader = DataLoader(data_list, batch_size=64, shuffle=False)

    all_embeddings = []
    all_indices = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            emb = model.encode(batch)  # [batch_size, hidden_dim]
            all_embeddings.append(emb.cpu().numpy())
            all_indices.extend(batch.idx.cpu().numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)

    # 4. Save embeddings aligned with dataframe index order
    # Create a matrix of shape [len(df), hidden_dim], fill with NaN, then fill rows we have
    hidden_dim = embeddings.shape[1]
    emb_full = np.full((len(df), hidden_dim), np.nan, dtype=np.float32)

    for i_row, idx in enumerate(all_indices):
        emb_full[idx] = embeddings[i_row]

    out_path = Path("data/processed/gnn_embeddings.npy")
    np.save(out_path, emb_full)
    print(f"💾 Saved GNN embeddings to {out_path} with shape {embeddings.shape}")

if __name__ == "__main__":
    main()