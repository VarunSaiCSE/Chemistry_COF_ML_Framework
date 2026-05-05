from pathlib import Path
import json
from src.models.baselines import RandomForestBaseline


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
    df, report = loader.load_and_validate()

    # 2. Features
    fe = FeatureEngineer(cfg)
    X, y = fe.create_features(df)

    # 3. Train baseline model
    rf = RandomForestBaseline(cfg)
    results = rf.train_and_evaluate(X, y)

    # 4. Save metrics
    metrics_path = cfg.paths.metrics_dir / "rf_co2_uptake.json"
    metrics = {
        "target": cfg.training.target_column,
        "model": "RandomForest",
        "r2": results.r2,
        "rmse": results.rmse,
        "mae": results.mae,
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"💾 Saved metrics to {metrics_path}")

if __name__ == "__main__":
    main()