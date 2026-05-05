import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List

from configs.base_config import BaseConfig

@dataclass
class FeatureEngineer:
    config: BaseConfig

    def _encode_functional_groups(self, df: pd.DataFrame) -> pd.DataFrame:
        # Extract unique functional groups
        all_groups: List[str] = []
        for s in df["Functional_Groups"].fillna(""):
            all_groups.extend([g.strip() for g in s.split(";") if g.strip()])

        unique_groups = sorted(set(all_groups))

        for g in unique_groups:
            col_name = f"FG_{g}"
            df[col_name] = df["Functional_Groups"].fillna("").apply(
                lambda x, g=g: int(g in [t.strip() for t in x.split(";") if t.strip()])
            )

        return df

    def create_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        cfg = self.config
        target_col = cfg.training.target_column

        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataframe")

        # Base numeric columns
        numeric_cols = [
            "Surface_Area_m2g",
            "Pore_Volume_cm3g",
            "Density_gcm3",
            "Pore_Size_Angstrom",
            "Porosity_fraction",
            "Thermal_Stability_C",
            "Bandgap_eV",
            "Charge_Mobility_cm2Vs",
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

        available_numeric = [c for c in numeric_cols if c in df.columns]
        feat_df = df[available_numeric].copy()

        # Derived features
        if {"Surface_Area_m2g", "Pore_Volume_cm3g"}.issubset(df.columns):
            feat_df["SA_to_PV_ratio"] = df["Surface_Area_m2g"] / (df["Pore_Volume_cm3g"] + 1e-6)

        if {"CO2_Uptake_mmolg", "Surface_Area_m2g"}.issubset(df.columns):
            feat_df["CO2_per_SA"] = df["CO2_Uptake_mmolg"] / (df["Surface_Area_m2g"] + 1e-6)

        if {"CH4_Uptake_mmolg", "Surface_Area_m2g"}.issubset(df.columns):
            feat_df["CH4_per_SA"] = df["CH4_Uptake_mmolg"] / (df["Surface_Area_m2g"] + 1e-6)

        if {"H2_Uptake_mmolg", "Surface_Area_m2g"}.issubset(df.columns):
            feat_df["H2_per_SA"] = df["H2_Uptake_mmolg"] / (df["Surface_Area_m2g"] + 1e-6)

        if {"C_Content_percent", "N_Content_percent"}.issubset(df.columns):
            feat_df["C_to_N_ratio"] = df["C_Content_percent"] / (df["N_Content_percent"] + 1e-6)

        if {"O_Content_percent", "C_Content_percent"}.issubset(df.columns):
            feat_df["O_to_C_ratio"] = df["O_Content_percent"] / (df["C_Content_percent"] + 1e-6)

        # One-hot encode topology & dimensionality
        if "Topology" in df.columns:
            topo_dummies = pd.get_dummies(df["Topology"], prefix="Topo")
            feat_df = pd.concat([feat_df, topo_dummies], axis=1)

        if "Dimensionality" in df.columns:
            dim_dummies = pd.get_dummies(df["Dimensionality"], prefix="Dim")
            feat_df = pd.concat([feat_df, dim_dummies], axis=1)

        # Functional group binary flags
        if "Functional_Groups" in df.columns:
            feat_df = pd.concat(
                [feat_df, self._encode_functional_groups(df.copy())[[c for c in df.columns if c.startswith("FG_")]]],
                axis=1,
            )

        # Drop any columns that accidentally still contain NaNs
        feat_df = feat_df.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
        feat_df = feat_df.fillna(0.0)

        y = df[target_col].astype(float)

        print(f"✅ Feature matrix created: X shape = {feat_df.shape}, y length = {len(y)}")
        return feat_df, y
    




    def append_gnn_embeddings(
        self,
        feat_df: pd.DataFrame,
        emb_path: str = "data/processed/gnn_embeddings.npy",
    ) -> pd.DataFrame:
        import numpy as np
        from pathlib import Path

        path = Path(emb_path)
        if not path.exists():
            print(f"⚠️ GNN embeddings file not found at {path}, returning original features.")
            return feat_df

        emb = np.load(path)  # shape [n_rows, hidden_dim]
        if emb.shape[0] != len(feat_df):
            print("⚠️ Embedding row count mismatch, skipping embeddings.")
            return feat_df

        # Some rows may be NaN if graphs were missing
        mask_valid = ~np.isnan(emb).any(axis=1)
        if not mask_valid.all():
            print(f"⚠️ {np.sum(~mask_valid)} rows have NaN embeddings; setting them to 0.")
        emb[np.isnan(emb)] = 0.0

        emb_cols = [f"GNN_emb_{i}" for i in range(emb.shape[1])]
        emb_df = pd.DataFrame(emb, columns=emb_cols, index=feat_df.index)

        feat_df_with_emb = pd.concat([feat_df, emb_df], axis=1)
        print(f"✅ Appended GNN embeddings: new X shape = {feat_df_with_emb.shape}")
        return feat_df_with_emb
    




if __name__ == "__main__":
    from src.data.loader import ResearchDataLoader
    cfg = BaseConfig()
    loader = ResearchDataLoader(cfg)
    df, _ = loader.load_and_validate()

    fe = FeatureEngineer(cfg)
    X, y = fe.create_features(df)
    print("Feature engineering done.")