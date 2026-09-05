import os
import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from pyod.models.ecod import ECOD
from pyod.models.copod import COPOD
from pyod.models.hbos import HBOS


# ============================================================
# FINAL MODEL FEATURES
# ============================================================

MODEL_FEATURES = [
    "Temperature_C",
    "Humidity_Percent",
    "Pressure_hPa",

    "Temperature_Diff",
    "Humidity_Diff",
    "Pressure_Diff",

    "Temperature_Deviation",
    "Humidity_Deviation",
    "Pressure_Deviation",

    "Temperature_LocalZ",
    "Humidity_LocalZ",
    "Pressure_LocalZ",

    "Pressure_Missing"
]


# ============================================================
# SCORE NORMALIZATION
# ============================================================

def normalize_scores(scores):

    scores = np.asarray(scores)

    min_score = np.min(scores)
    max_score = np.max(scores)

    if max_score - min_score < 1e-12:
        return np.zeros_like(scores)

    return (
        (scores - min_score)
        / (max_score - min_score)
    )


# ============================================================
# TRAIN MODELS
# ============================================================

def train_models(
    df,
    contamination=0.02
):

    print("\n")
    print("=" * 60)
    print("TRAINING SKYGUARD AI MODELS")
    print("=" * 60)

    df = df.copy()

    df = df.sort_values(
        ["DateTime", "Location"]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Check required columns
    # --------------------------------------------------------

    missing = [
        col
        for col in MODEL_FEATURES
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing model features: {missing}"
        )

    X = df[MODEL_FEATURES].copy()

    print(f"\nTraining observations: {len(X):,}")
    print(f"Training features: {len(MODEL_FEATURES)}")

    # --------------------------------------------------------
    # Scaling
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # ========================================================
    # 1. ISOLATION FOREST
    # ========================================================

    print("\nTraining Isolation Forest...")

    iforest = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )

    iforest.fit(X_scaled)

    # ========================================================
    # 2. ECOD
    # ========================================================

    print("Training ECOD...")

    ecod = ECOD(
        contamination=contamination
    )

    ecod.fit(X_scaled)

    # ========================================================
    # 3. COPOD
    # ========================================================

    print("Training COPOD...")

    copod = COPOD(
        contamination=contamination
    )

    copod.fit(X_scaled)

    # ========================================================
    # 4. HBOS
    # ========================================================

    print("Training HBOS...")

    hbos = HBOS(
        contamination=contamination
    )

    hbos.fit(X_scaled)

    # ========================================================
    # SAVE MODELS
    # ========================================================

    os.makedirs("models", exist_ok=True)

    joblib.dump(
        scaler,
        "models/scaler.pkl"
    )

    joblib.dump(
        iforest,
        "models/isolation_forest.pkl"
    )

    joblib.dump(
        ecod,
        "models/ecod.pkl"
    )

    joblib.dump(
        copod,
        "models/copod.pkl"
    )

    joblib.dump(
        hbos,
        "models/hbos.pkl"
    )

    print("\nModels successfully saved:")

    print("  models/scaler.pkl")
    print("  models/isolation_forest.pkl")
    print("  models/ecod.pkl")
    print("  models/copod.pkl")
    print("  models/hbos.pkl")

    print("\nTraining complete.")

    return {
        "scaler": scaler,
        "iforest": iforest,
        "ecod": ecod,
        "copod": copod,
        "hbos": hbos
    }


# ============================================================
# LOAD SAVED MODELS
# ============================================================

def load_models():

    required_files = [
        "models/scaler.pkl",
        "models/isolation_forest.pkl",
        "models/ecod.pkl",
        "models/copod.pkl",
        "models/hbos.pkl"
    ]

    for file_path in required_files:

        if not os.path.exists(file_path):

            raise FileNotFoundError(
                f"Required model file not found: {file_path}"
            )

    return {
        "scaler": joblib.load(
            "models/scaler.pkl"
        ),

        "iforest": joblib.load(
            "models/isolation_forest.pkl"
        ),

        "ecod": joblib.load(
            "models/ecod.pkl"
        ),

        "copod": joblib.load(
            "models/copod.pkl"
        ),

        "hbos": joblib.load(
            "models/hbos.pkl"
        )
    }


# ============================================================
# LIVE INFERENCE
# ============================================================

def run_live_inference(df):

    df = df.copy()

    if df.empty:
        return df

    df = df.sort_values(
        ["DateTime", "Location"]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Check model features
    # --------------------------------------------------------

    missing = [
        col
        for col in MODEL_FEATURES
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing model features: {missing}"
        )

    # --------------------------------------------------------
    # Load trained models
    # --------------------------------------------------------

    models = load_models()

    scaler = models["scaler"]
    iforest = models["iforest"]
    ecod = models["ecod"]
    copod = models["copod"]
    hbos = models["hbos"]

    # --------------------------------------------------------
    # Prepare X
    # --------------------------------------------------------

    X = df[MODEL_FEATURES].copy()

    X_scaled = scaler.transform(X)

    # ========================================================
    # ISOLATION FOREST
    # ========================================================

    df["IF_Anomaly"] = (
        iforest.predict(X_scaled) == -1
    ).astype(int)

    df["IF_Score_Raw"] = (
        -iforest.decision_function(X_scaled)
    )

    # ========================================================
    # ECOD
    # ========================================================

    df["ECOD_Anomaly"] = (
        ecod.predict(X_scaled)
    ).astype(int)

    df["ECOD_Score"] = (
        ecod.decision_function(X_scaled)
    )

    # ========================================================
    # COPOD
    # ========================================================

    df["COPOD_Anomaly"] = (
        copod.predict(X_scaled)
    ).astype(int)

    df["COPOD_Score"] = (
        copod.decision_function(X_scaled)
    )

    # ========================================================
    # HBOS
    # ========================================================

    df["HBOS_Anomaly"] = (
        hbos.predict(X_scaled)
    ).astype(int)

    df["HBOS_Score"] = (
        hbos.decision_function(X_scaled)
    )

    # ========================================================
    # MODEL AGREEMENT
    # ========================================================

    df["Model_Agreement"] = (
        df["IF_Anomaly"]
        + df["ECOD_Anomaly"]
        + df["COPOD_Anomaly"]
        + df["HBOS_Anomaly"]
    )

    # ========================================================
    # ENSEMBLE DECISION
    # ========================================================

    df["Ensemble_Anomaly"] = (
        df["Model_Agreement"] >= 3
    ).astype(int)

    # ========================================================
    # SIMPLE LIVE SEVERITY
    # ========================================================

    df["Anomaly_Severity"] = "Normal"

    df.loc[
        df["Model_Agreement"] == 1,
        "Anomaly_Severity"
    ] = "Low"

    df.loc[
        df["Model_Agreement"] == 2,
        "Anomaly_Severity"
    ] = "Medium"

    df.loc[
        df["Model_Agreement"] >= 3,
        "Anomaly_Severity"
    ] = "High"

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    df = df.sort_values(
        ["DateTime", "Location"]
    ).reset_index(drop=True)

    return df