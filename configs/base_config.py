from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class DataConfig:
    raw_data_path: Path = Path("data/raw/cof_dataset.csv")
    processed_data_path: Path = Path("data/processed/cof_dataset_processed.csv")

@dataclass
class TrainingConfig:
    random_seed: int = 42
    test_size: float = 0.2
    val_size: float = 0.1
    target_column: str = "CO2_Uptake_mmolg"

@dataclass
class ModelConfig:
    n_estimators: int = 500
    max_depth: int = 10
    learning_rate: float = 0.1

@dataclass
class PathsConfig:
    results_dir: Path = Path("results")
    models_dir: Path = Path("results/models")
    metrics_dir: Path = Path("results/metrics")
    figures_dir: Path = Path("results/figures")

@dataclass
class BaseConfig:
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    def ensure_dirs(self):
        self.paths.results_dir.mkdir(parents=True, exist_ok=True)
        self.paths.models_dir.mkdir(parents=True, exist_ok=True)
        self.paths.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.paths.figures_dir.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    cfg = BaseConfig()
    cfg.ensure_dirs()
    print("✅ Config initialized. Results directories created:")
    print(f" - {cfg.paths.results_dir}")
    print(f" - {cfg.paths.models_dir}")
    print(f" - {cfg.paths.metrics_dir}")
    print(f" - {cfg.paths.figures_dir}")