from pathlib import Path
import json

from configs.base_config import BaseConfig
from src.data.loader import ResearchDataLoader
from src.data.features import FeatureEngineer
from src.models.baselines import RandomForestBaseline
from src.utils.seed import set_seed

def main():
    cfg = BaseConfig()
    cfg.ensure_dirs()
    set_seed(cfg.training.random_seed)

    # 1. Load data
    loader = ResearchDataLoader(cfg)
    df, _ = loader.load_and_validate()

    # 2. Features
    fe = FeatureEngineer(cfg)
    X, y = fe.create_features(df)

    # 3. Append GNN embeddings
    X_multi = fe.append_gnn_embeddings(X, emb_path="data/processed/gnn_embeddings.npy")

    # 4. Train RF on multi-modal features
    rf = RandomForestBaseline(cfg)
    results = rf.train_and_evaluate(X_multi, y)

    # 5. Save metrics
    metrics_path = cfg.paths.metrics_dir / "rf_multimodal_co2_uptake.json"
    metrics = {
        "target": cfg.training.target_column,
        "model": "RandomForest + GNN embeddings",
        "r2": results.r2,
        "rmse": results.rmse,
        "mae": results.mae,
        "n_features": X_multi.shape[1],
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"💾 Saved multi-modal metrics to {metrics_path}")

if __name__ == "__main__":
    main()