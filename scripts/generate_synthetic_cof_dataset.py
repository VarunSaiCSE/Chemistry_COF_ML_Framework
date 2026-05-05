import numpy as np
import pandas as pd
from pathlib import Path
import random

# Reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

N_SAMPLES = 6000

# Some reasonable options for categorical fields
TOPOLOGIES = ["hcb", "sql", "dia", "ftw", "pts", "srs", "bor"]
DIMENSIONALITY = ["2D", "3D"]
FUNCTIONAL_GROUPS = ["NH2", "NO2", "CF3", "OH", "CN", "SO3H", "COOH", "F"]

def random_functional_group_string():
    k = np.random.randint(1, 4)  # 1 to 3 groups
    groups = np.random.choice(FUNCTIONAL_GROUPS, size=k, replace=False)
    return ";".join(groups)

# Very simplified "SMILES" building blocks – not chemically perfect, but enough for RDKit later
BASE_SMILES = [
    "c1ccccc1",          # benzene
    "c1ccncc1",          # pyridine-like
    "c1ccc(cc1)N",       # aniline-like
    "c1ccc(cc1)O",       # phenol-like
    "c1ccc(cc1)C#N",     # benzonitrile-like
    "c1ccc(cc1)C(F)(F)F",# CF3-phenyl
    "c1ccc2ccccc2c1",    # naphthalene-like
    "c1cc2ccccc2n1"      # fused N-heterocycle
]

def sample_smiles():
    return random.choice(BASE_SMILES)

def main():
    rows = []

    for i in range(N_SAMPLES):
        cof_id = f"COF_{i:04d}"

        # Basic structure / categories
        smiles = sample_smiles()
        topology = random.choice(TOPOLOGIES)
        dimensionality = random.choice(DIMENSIONALITY)
        func_groups = random_functional_group_string()

        # Structural properties
        surface_area = np.random.uniform(150, 4500)  # m2/g
        pore_volume = np.random.uniform(0.1, 3.5)    # cm3/g
        density = np.random.uniform(0.2, 1.3)        # g/cm3
        pore_size = np.random.uniform(5, 45)         # Angstrom
        porosity = np.clip(np.random.normal(0.6, 0.15), 0.2, 0.9)

        # Thermal stability
        thermal_stability = np.random.uniform(220, 550)

        # Electronic properties
        # Rough trend: more electron-withdrawing groups -> lower bandgap
        n_no2 = func_groups.count("NO2")
        n_cf3 = func_groups.count("CF3")
        n_cn  = func_groups.count("CN")

        base_gap = np.random.uniform(1.5, 3.5)
        bandgap = base_gap - 0.3 * n_no2 - 0.2 * n_cf3 - 0.2 * n_cn
        bandgap = float(np.clip(bandgap, 0.5, 4.5))

        charge_mobility = np.random.lognormal(mean=0.5, sigma=0.6)  #  ~0.5–20

        # Gas uptakes: correlate with surface area and pore volume + noise
        co2_uptake = 0.004 * surface_area + 1.5 * pore_volume + np.random.normal(0, 1.5)
        ch4_uptake = 0.0015 * surface_area + 0.8 * pore_volume + np.random.normal(0, 0.8)
        h2_uptake  = 0.001 * surface_area + 0.5 * pore_volume + np.random.normal(0, 0.6)

        co2_uptake = float(np.clip(co2_uptake, 0.5, 25.0))
        ch4_uptake = float(np.clip(ch4_uptake, 0.2, 8.0))
        h2_uptake  = float(np.clip(h2_uptake, 0.1, 6.0))

        co2_ch4_sel = (co2_uptake / (ch4_uptake + 1e-3)) + np.random.normal(0, 0.5)
        co2_n2_sel  = np.random.uniform(5, 50)

        co2_ch4_sel = float(np.clip(co2_ch4_sel, 1.0, 80.0))

        # Stability scores – depend weakly on functional groups and topology
        # Just synthetic, but gives some structure
        base_stability = np.random.uniform(0.3, 0.8)
        penalty_water = 0.15 if "NH2" in func_groups or "OH" in func_groups else 0.0
        penalty_sox = 0.1 if "SO3H" in func_groups else 0.0
        penalty_nox = 0.05 if "NO2" in func_groups else 0.0

        h2o_stability = float(np.clip(base_stability - penalty_water + np.random.normal(0, 0.05), 0.0, 1.0))
        sox_stability = float(np.clip(base_stability - penalty_sox + np.random.normal(0, 0.05), 0.0, 1.0))
        nox_stability = float(np.clip(base_stability - penalty_nox + np.random.normal(0, 0.05), 0.0, 1.0))

        # Composition – rough trends based on functional groups
        c_pct = np.random.uniform(55, 75)
        h_pct = np.random.uniform(2, 7)
        n_pct = np.random.uniform(5, 20)
        o_pct = np.random.uniform(2, 15)
        others = max(0.0, 100.0 - (c_pct + h_pct + n_pct + o_pct))

        # IR / Raman / PXRD peaks – synthetic but with some structure
        def random_peaks(n_peaks, low, high):
            positions = np.sort(np.random.uniform(low, high, size=n_peaks))
            intensities = np.random.uniform(0.2, 1.0, size=n_peaks)
            return positions, intensities

        n_ir = np.random.randint(3, 8)
        ir_pos, ir_int = random_peaks(n_ir, 800, 3500)

        n_raman = np.random.randint(3, 10)
        raman_pos, raman_int = random_peaks(n_raman, 200, 2000)

        n_pxrd = np.random.randint(3, 12)
        pxrd_pos, pxrd_int = random_peaks(n_pxrd, 2, 40)

        ir_peaks_str = ",".join(f"{p:.1f}" for p in ir_pos)
        ir_int_str = ",".join(f"{x:.2f}" for x in ir_int)

        raman_peaks_str = ",".join(f"{p:.1f}" for p in raman_pos)
        raman_int_str = ",".join(f"{x:.2f}" for x in raman_int)

        pxrd_peaks_str = ",".join(f"{p:.2f}" for p in pxrd_pos)
        pxrd_int_str = ",".join(f"{x:.2f}" for x in pxrd_int)

        # Meta
        year = np.random.randint(2005, 2026)
        patent_status = np.random.choice(["Yes", "No"], p=[0.35, 0.65])
        novelty_score = float(np.clip(np.random.normal(0.6, 0.15), 0.0, 1.0))

        row = {
            "COF_ID": cof_id,
            "SMILES": smiles,
            "Topology": topology,
            "Dimensionality": dimensionality,
            "Functional_Groups": func_groups,
            "Surface_Area_m2g": surface_area,
            "Pore_Volume_cm3g": pore_volume,
            "Density_gcm3": density,
            "Pore_Size_Angstrom": pore_size,
            "Porosity_fraction": porosity,
            "Thermal_Stability_C": thermal_stability,
            "Bandgap_eV": bandgap,
            "Charge_Mobility_cm2Vs": charge_mobility,
            "CO2_Uptake_mmolg": co2_uptake,
            "CH4_Uptake_mmolg": ch4_uptake,
            "H2_Uptake_mmolg": h2_uptake,
            "CO2_CH4_Selectivity": co2_ch4_sel,
            "CO2_N2_Selectivity": co2_n2_sel,
            "SOx_Stability_Score": sox_stability,
            "NOx_Stability_Score": nox_stability,
            "H2O_Stability_Score": h2o_stability,
            "C_Content_percent": c_pct,
            "H_Content_percent": h_pct,
            "N_Content_percent": n_pct,
            "O_Content_percent": o_pct,
            "Other_Content_percent": others,
            "IR_Peaks_cm1": ir_peaks_str,
            "IR_Intensities": ir_int_str,
            "Raman_Peaks_cm1": raman_peaks_str,
            "Raman_Intensities": raman_int_str,
            "PXRD_Peaks_2theta": pxrd_peaks_str,
            "PXRD_Intensities": pxrd_int_str,
            "Discovery_Year": year,
            "Patent_Status": patent_status,
            "Novelty_Score": novelty_score
        }

        rows.append(row)

    df = pd.DataFrame(rows)

    out_path = Path("data/raw/cof_dataset.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"✅ Generated synthetic COF dataset: {out_path} with {len(df)} rows")

if __name__ == "__main__":
    main()