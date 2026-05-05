import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Dict, Any

from configs.base_config import BaseConfig

@dataclass
class DataQualityReport:
    n_rows: int
    n_cols: int
    missing_by_column: Dict[str, int]
    numeric_columns: int
    categorical_columns: int

class ResearchDataLoader:
    def __init__(self, config: BaseConfig | None = None):
        self.config = config or BaseConfig()
        self.raw_path: Path = self.config.data.raw_data_path

    def load_raw(self) -> pd.DataFrame:
        if not self.raw_path.exists():
            raise FileNotFoundError(f"Raw data file not found at {self.raw_path}")
        df = pd.read_csv(self.raw_path)
        print(f"📥 Loaded raw dataset from {self.raw_path}")
        print(f"   Shape: {df.shape[0]} rows x {df.shape[1]} columns")
        return df

    def basic_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        # Strip column names
        df = df.rename(columns={c: c.strip() for c in df.columns})

        # Drop exact duplicate rows
        before = len(df)
        df = df.drop_duplicates()
        after = len(df)
        if after < before:
            print(f"🧹 Removed {before - after} duplicate rows")

        # Simple type coercion for numeric fields
        numeric_cols = [
            "Surface_Area_m2g",
            "Pore_Volume_cm3g",
            "Density_gcm3",
            "Pore_Size_Angstrom",
            "Porosity_fraction",
            "Thermal_Stability_C",
            "Bandgap_eV",
            "Charge_Mobility_cm2Vs",
            "CO2_Uptake_mmolg",
            "CH4_Uptake_mmolg",
            "H2_Uptake_mmolg",
            "CO2_CH4_Selectivity",
            "CO2_N2_Selectivity",
            "SOx_Stability_Score",
            "NOx_Stability_Score",
            "H2O_Stability_Score",
            "C_Content_percent",
            "H_Content_percent",
            "N_Content_percent",
            "O_Content_percent",
            "Other_Content_percent",
            "Novelty_Score",
            "Discovery_Year",
        ]

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Very simple missing handling for now: drop rows missing the target
        target_col = self.config.training.target_column
        if target_col in df.columns:
            before = len(df)
            df = df.dropna(subset=[target_col])
            after = len(df)
            if after < before:
                print(f"🧹 Dropped {before - after} rows with missing target '{target_col}'")

        return df

    def assess_quality(self, df: pd.DataFrame) -> DataQualityReport:
        missing = df.isnull().sum().to_dict()
        numeric = df.select_dtypes(include=[np.number]).shape[1]
        categorical = df.select_dtypes(exclude=[np.number]).shape[1]

        report = DataQualityReport(
            n_rows=df.shape[0],
            n_cols=df.shape[1],
            missing_by_column=missing,
            numeric_columns=numeric,
            categorical_columns=categorical,
        )

        print(f"📊 Data quality report:")
        print(f"   Rows: {report.n_rows}, Columns: {report.n_cols}")
        print(f"   Numeric columns: {report.numeric_columns}, Categorical: {report.categorical_columns}")
        n_missing_cols = sum(1 for v in missing.values() if v > 0)
        print(f"   Columns with missing values: {n_missing_cols}")

        return report

    def load_and_validate(self) -> Tuple[pd.DataFrame, DataQualityReport]:
        df = self.load_raw()
        df = self.basic_cleaning(df)
        report = self.assess_quality(df)

        # Optionally save a cleaned version
        processed_path = self.config.data.processed_data_path
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(processed_path, index=False)
        print(f"💾 Saved cleaned dataset to {processed_path}")

        return df, report

if __name__ == "__main__":
    cfg = BaseConfig()
    loader = ResearchDataLoader(cfg)
    df, report = loader.load_and_validate()
    print("✅ Data loading & validation complete.")