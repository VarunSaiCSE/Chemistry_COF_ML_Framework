from pathlib import Path
import json

from configs.base_config import BaseConfig
from src.data.loader import ResearchDataLoader
from src.data.features import FeatureEngineer
from src.models.baselines import RandomForestBaseline, XGBoostBaseline, CatBoostBaseline
from src.utils.seed import set_seed

def save_metrics(path: Path, name: str, target: str, results):
    metrics = {
        "target": target,
        "model": name,
        "r2": results.r2,
        "rmse": results.rmse,
        "mae": results.mae,
    }
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"💾 Saved metrics to {path}")

def main():
    cfg = BaseConfig()
    cfg.ensure_dirs()
    set_seed(cfg.training.random_seed)

    # 1. Load data
    loader = ResearchDataLoader(cfg)
    df, _ = loader.load_and_validate()

    # 2. Features (tabular only)
    fe = FeatureEngineer(cfg)
    X, y = fe.create_features(df)

    # 3. RandomForest
    rf = RandomForestBaseline(cfg)
    rf_res = rf.train_and_evaluate(X, y)
    save_metrics(cfg.paths.metrics_dir / "rf_co2_uptake.json", "RandomForest", cfg.training.target_column, rf_res)

    # 4. XGBoost
    xgb = XGBoostBaseline(cfg)
    xgb_res = xgb.train_and_evaluate(X, y)
    save_metrics(cfg.paths.metrics_dir / "xgboost_co2_uptake.json", "XGBoost", cfg.training.target_column, xgb_res)

    # 5. CatBoost
    cb = CatBoostBaseline(cfg)
    cb_res = cb.train_and_evaluate(X, y)
    save_metrics(cfg.paths.metrics_dir / "catboost_co2_uptake.json", "CatBoost", cfg.training.target_column, cb_res)

if __name__ == "__main__":
    main()