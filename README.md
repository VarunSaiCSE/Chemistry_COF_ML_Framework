# Machine Learning & GNN Pipeline for COF Property Prediction

This is an end‑to‑end machine learning pipeline for Covalent Organic Framework (COF) property prediction.  
It combines descriptor‑based tree models (RandomForest, XGBoost, CatBoost) with graph neural networks (GNNs) built from linker SMILES, and supports both tabular and multi‑modal modeling.

The current version runs on a synthetic COF‑like dataset to validate the full stack (data → features → models → metrics) and to prepare for integration with real experimental COF data.

---

## 1. Motivation

Covalent Organic Frameworks are porous, crystalline materials with tunable structure and properties.  
They are promising for gas storage and separation, catalysis, electronics, and sensing.  
Most existing COF work uses either:

- **Descriptor‑based models** (surface area, pore volume, composition, etc.), or  
- **Structure‑based models** (e.g., GNNs on molecular graphs).

This project is designed to:

1. Provide a **clean, reproducible ML pipeline** that a chemistry group can reuse.  
2. Benchmark **tree ensembles** vs **GNNs** vs **multi‑modal models** for COF property prediction.  
3. Serve as a **ready‑to‑plug‑in framework** once real COF datasets are available.

---

## 2. Repository Structure

```text
Chemistry_COF_ML_Framework/
  configs/
    base_config.py        # Global configuration (paths, training, model params)
    __init__.py
  data/
    raw/
      cof_dataset.csv     # Synthetic COF dataset (6000 rows x 35 columns)
    processed/
      cof_dataset_processed.csv  # Cleaned dataset
      gnn_embeddings.npy         # Precomputed GNN graph embeddings
  results/
    models/               # Saved models (optional)
    metrics/              # JSON metrics files for each experiment
    figures/              # Plots (future work)
  src/
    data/
      loader.py           # Data loading + validation
      features.py         # Feature engineering + multi-modal fusion helpers
      graphs.py           # SMILES → RDKit → PyTorch Geometric graphs
      __init__.py
    models/
      baselines.py        # RF, XGBoost, CatBoost baselines
      gnn_models.py       # GIN-based GNN encoder and regressor
      __init__.py
    training/
      train_baselines.py              # Legacy RF baseline (tabular)
      train_tree_ensemble_baselines.py# RF/XGBoost/CatBoost comparison
      train_gnn.py                    # Structure-only GNN training
      compute_gnn_embeddings.py       # Precompute graph embeddings
      train_baselines_multimodal.py   # RF on tabular + GNN embeddings
      __init__.py
    utils/
      seed.py              # Reproducible random seeds
      __init__.py
  README.md
```

---

## 3. Synthetic COF Dataset

The synthetic dataset (`data/raw/cof_dataset.csv`) contains 6000 COF‑like entries with the following information:

- **Identifiers / metadata**
  - `COF_ID`  
  - `Discovery_Year`  

- **Structural descriptors**
  - `SMILES` (linker SMILES string)  
  - `Topology` (e.g., hcb, dia, etc.)  
  - `Dimensionality` (2D / 3D)  
  - `Functional_Groups` (semicolon‑separated list of linker functional groups)  

- **Textural / physical properties**
  - `Surface_Area_m2g`  
  - `Pore_Volume_cm3g`  
  - `Pore_Size_Angstrom`  
  - `Density_gcm3`  
  - `Porosity_fraction`  

- **Stability / selectivity metrics**
  - `Thermal_Stability_C`  
  - `SOx_Stability_Score`  
  - `NOx_Stability_Score`  
  - `H2O_Stability_Score`  
  - `CO2_CH4_Selectivity`  
  - `CO2_N2_Selectivity`  

- **Elemental composition**
  - `C_Content_percent`  
  - `H_Content_percent`  
  - `N_Content_percent`  
  - `O_Content_percent`  
  - `Other_Content_percent`  

- **Electronic / application‑related descriptors**
  - `Bandgap_eV`  
  - `Charge_Mobility_cm2Vs`  
  - `Novelty_Score`  

- **Target properties**
  - `CO2_Uptake_mmolg`  
  - `CH4_Uptake_mmolg`  
  - `H2_Uptake_mmolg`  

These targets are generated from the descriptors using prescribed functional relationships plus noise (e.g., CO₂ uptake increasing with surface area, pore volume, and selected functional groups).  
The dataset is **COF‑inspired but synthetic** and is intended only for pipeline development and debugging, not for drawing scientific conclusions.

---

## 4. Models Implemented

### 4.1 Descriptor‑based tree ensembles (tabular)

Implemented in `src/models/baselines.py`:

- **RandomForestRegressor**
  - Strong baseline for tabular COF descriptors.
  - Uses surface area, pore volume, composition, stability scores, topology one‑hots, etc.

- **XGBoostRegressor**
  - Gradient‑boosted tree ensemble.
  - Often competitive or superior to RF on real‑world tabular problems.

- **CatBoostRegressor**
  - Gradient boosting with efficient handling of categorical features.
  - Useful when categorical descriptors are prominent.

### 4.2 Structure‑based graph neural network (GNN)

Implemented in `src/models/gnn_models.py` and `src/data/graphs.py`:

- **GraphBuilder**
  - Converts linker SMILES to RDKit molecules.
  - Builds PyTorch Geometric `Data` objects with:
    - Node features: atomic number, degree, formal charge, aromatic flag, ring flag.
    - Edge features: bond type, conjugation, ring membership.

- **GINRegressor**
  - Node embedding layer.
  - Several GINConv + BatchNorm layers with ReLU and dropout.
  - Global mean pooling over nodes to obtain graph‑level embeddings.
  - MLP head for scalar regression (e.g., CO₂ uptake).
  - `encode()` method exposes graph embeddings for multi‑modal fusion.

### 4.3 Multi‑modal model (tabular + graph)

- Precompute graph embeddings using `compute_gnn_embeddings.py`.  
- Concatenate embeddings with tabular features inside `FeatureEngineer.append_gnn_embeddings()`.  
- Train a RandomForest on the fused feature space via `train_baselines_multimodal.py`.

This architecture mimics realistic “structure + descriptors” setups in materials ML.

---

## 5. Environment Setup

The project assumes a conda environment with PyTorch, PyTorch Geometric, RDKit, and tree‑ensemble libraries.

```bash
# Create and activate environment
conda create -n chemenv python=3.11 -y
conda activate chemenv

# Core packages
conda install -y pandas numpy scikit-learn matplotlib seaborn

# PyTorch (CPU or MPS; choose appropriate command from pytorch.org)
conda install -y pytorch torchvision torchaudio -c pytorch

# RDKit
conda install -y -c conda-forge rdkit

# PyTorch Geometric (CPU)
pip install torch-geometric

# Tree ensembles
pip install xgboost catboost
```

---

## 6. Running the Pipeline

All commands are run from the project root (`Chemistry_COF_ML_Framework/`) with the `chemenv` environment activated.

### 6.1 Initialize configuration and directories

```bash
python -m configs.base_config
```

Creates the `results/` directory structure and validates that configuration is importable.

### 6.2 Train descriptor‑based tree ensemble baselines (CO₂ uptake)

```bash
python -m src.training.train_tree_ensemble_baselines
```

This performs:

1. Load and validate dataset (`ResearchDataLoader`).  
2. Build tabular feature matrix (`FeatureEngineer.create_features`).  
3. Train and evaluate:
   - RandomForest
   - XGBoost
   - CatBoost  
4. Save metrics as JSON under `results/metrics/`.

### 6.3 Train structure‑only GNN

```bash
python -m src.training.train_gnn
```

This performs:

1. Build graph dataset from SMILES (`GraphBuilder`).  
2. Train a GIN‑based GNN on CO₂ uptake.  
3. Report test performance (R², RMSE, MAE).

### 6.4 Precompute GNN graph embeddings

```bash
python -m src.training.compute_gnn_embeddings
```

This:

1. Rebuilds graphs for each COF.  
2. Passes them through `GINRegressor.encode()`.  
3. Saves embeddings to `data/processed/gnn_embeddings.npy`.

### 6.5 Train multi‑modal baseline (tabular + GNN embedding)

```bash
python -m src.training.train_baselines_multimodal
```

This:

1. Creates tabular features.  
2. Appends GNN embeddings as additional columns.  
3. Trains a RandomForest on the fused feature matrix.  
4. Saves metrics to `results/metrics/rf_multimodal_co2_uptake.json`.

---

## 7. Results on Synthetic CO₂ Uptake

On the synthetic dataset (6000 rows, 35 columns), using CO₂ uptake as the target:

| Model                               | Input Features                     | R²      | RMSE   | MAE   |
|-------------------------------------|------------------------------------|---------|--------|-------|
| RandomForest                        | Tabular descriptors                | 0.9971  | 0.2889 | 0.1464 |
| XGBoost                             | Tabular descriptors                | 0.9976  | 0.2666 | 0.1685 |
| CatBoost                            | Tabular descriptors                | 0.9955  | 0.3609 | 0.2312 |
| GIN GNN                             | SMILES graph only                  | ≈ 0.00  | ≈ 5.6  | ≈ 4.8 |
| RandomForest + GNN embeddings       | Tabular + GNN graph embeddings     | 0.9971  | 0.2929 | 0.1484 |

Key observations:

- **Descriptor‑based models (RF, XGBoost, CatBoost)** achieve **very high R²** on this dataset.  
  This is expected because the target was synthetically constructed to depend strongly on the numeric descriptors (surface area, pore volume, stability scores, etc.).  
- **Structure‑only GNN** has R² ≈ 0, meaning that with only linker SMILES graphs it cannot recover the synthetic relationship; the structural information alone is insufficient in this setup.  
- **Multi‑modal RF** (tabular + GNN embeddings) performs similarly to the descriptor‑only RF.  
  On this synthetic dataset, graph embeddings do not add significant new signal beyond the hand‑crafted descriptors.

These results validate that the pipeline is functioning correctly and highlight the limitations of relying solely on structural information when the synthetic targets are primarily descriptor‑driven.

---

## 8. Conclusions and Intended Use

- The current ChemOps implementation is a **methodological and engineering framework**, not a final scientific study.  
- It demonstrates:
  - robust data loading and validation,  
  - feature engineering from COF‑like descriptors,  
  - descriptor‑based tree ensembles (RF/XGBoost/CatBoost),  
  - a GIN‑based GNN for linker‑level graphs,  
  - and a simple multi‑modal fusion model.  
- On synthetic data, tree ensembles nearly saturate performance, while the GNN alone cannot learn the descriptor‑defined relationships.  
  This behavior is consistent with how the synthetic dataset was constructed.

The framework is intended to be reused with **real COF datasets** (experimental or high‑fidelity simulated):

- Replace `data/raw/cof_dataset.csv` with real measurements (keeping or adapting the schema).  
- Define tasks that are scientifically relevant (e.g., CO₂ uptake at specific T/P, water stability, bandgap).  
- Re‑run the existing training scripts to benchmark RF/XGBoost/CatBoost vs GNN vs multi‑modal models.  
- Extend the repository with additional targets, uncertainty estimation, scaffold‑based splits, and interpretability analyses (feature importance, saliency maps, etc.).

---

## 9. Future Work

Planned extensions once real COF data are available:

1. **Integration with experimental datasets**
   - Accept linker SMILES, topologies, and measured properties from ongoing COF projects.
2. **More rigorous evaluation**
   - Scaffold‑based splits by linker structure.
   - Cross‑validation and uncertainty quantification.
3. **Richer multi‑modal models**
   - Fusion of GNN embeddings, tabular descriptors, and possibly spectroscopic data (IR/Raman/PXRD).
4. **Interpretability**
   - Feature importance (RF/XGBoost/CatBoost).  
   - GNN attribution methods to identify key atoms, bonds, and functional groups.
5. **Publication‑grade experiments**
   - Comprehensive comparison across multiple COF properties and model families.
   - Joint analysis with domain experts to derive chemically meaningful insights.

---

## 10. Contact Info

Mail: andra.varunsai@gmail.com
Time: IST
