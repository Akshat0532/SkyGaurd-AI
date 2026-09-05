import os
import time
import pandas as pd

from src.data_preprocessing import (
    load_and_preprocess_raw_data
)

from src.feature_engineering import (
    create_features
)

from src.anomaly_detection import (
    train_models,
    run_live_inference
)

# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------

HISTORICAL_DATA = (
    "data/processed/SkyGuard_clean_3hourly.csv"
)

HISTORICAL_FEATURES = (
    "data/processed/SkyGuard_features.csv"
)

LIVE_OUTPUT = (
    "outputs/live_anomalies.csv"
)

UPDATE_INTERVAL = 10

CONTAMINATION = 0.02

# ------------------------------------------------------------
# IMPORTANT
# ------------------------------------------------------------
# Set this to True the FIRST time you run the new system.
#
# After successful training, change it to False.
# ------------------------------------------------------------

TRAIN_FROM_HISTORICAL = True


# ============================================================
# TRAINING
# ============================================================

def train_from_historical():

    print("\n")
    print("=" * 70)
    print("HISTORICAL MODEL TRAINING")
    print("=" * 70)

    if not os.path.exists(HISTORICAL_DATA):

        raise FileNotFoundError(
            f"Historical dataset not found:\n"
            f"{HISTORICAL_DATA}"
        )

    # --------------------------------------------------------
    # Load historical 3-hourly data
    # --------------------------------------------------------

    print(
        f"\nLoading historical dataset:\n"
        f"{HISTORICAL_DATA}"
    )

    historical_df = pd.read_csv(
        HISTORICAL_DATA
    )

    historical_df["DateTime"] = pd.to_datetime(
        historical_df["DateTime"]
    )

    print(
        f"Historical observations: "
        f"{len(historical_df):,}"
    )

    print(
        f"Stations: "
        f"{historical_df['Location'].nunique()}"
    )

    # --------------------------------------------------------
    # Feature engineering
    # --------------------------------------------------------

    print("\nCreating historical features...")

    historical_features = create_features(
        historical_df
    )

    if historical_features.empty:

        raise RuntimeError(
            "Historical feature engineering produced "
            "zero rows."
        )

    print(
        f"Feature rows generated: "
        f"{len(historical_features):,}"
    )

    # --------------------------------------------------------
    # Save features
    # --------------------------------------------------------

    os.makedirs(
        "data/processed",
        exist_ok=True
    )

    historical_features.to_csv(
        HISTORICAL_FEATURES,
        index=False
    )

    print(
        f"\nHistorical features saved to:\n"
        f"{HISTORICAL_FEATURES}"
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    train_models(
        historical_features,
        contamination=CONTAMINATION
    )


# ============================================================
# FIND LIVE STATION FILES
# ============================================================

def get_live_station_files():

    weather_folder = "weather_data"

    station_files = [
        "Chennai_432790-99999.csv",
        "Mumbai_430030-99999.csv",
        "New_Delhi_421820-99999.csv",
        "Kolkata_428090-99999.csv",
        "Hyderabad_431280-99999.csv"
    ]

    available_files = []

    print("\nChecking live station files...")

    for filename in station_files:

        filepath = os.path.join(
            weather_folder,
            filename
        )

        if os.path.exists(filepath):

            try:

                df = pd.read_csv(filepath)

                print(
                    f"{filename:<35} : "
                    f"{len(df):,} observations"
                )

                available_files.append(
                    filepath
                )

            except Exception as e:

                print(
                    f"{filename:<35} : "
                    f"ERROR - {e}"
                )

        else:

            print(
                f"{filename:<35} : "
                f"NOT FOUND"
            )

    return available_files


# ============================================================
# BUILD LIVE DATA WITH HISTORICAL CONTEXT
# ============================================================

def build_live_dataset():

    print("\n")
    print("=" * 70)
    print("BUILDING LIVE DATASET")
    print("=" * 70)

    # --------------------------------------------------------
    # Historical data
    # --------------------------------------------------------

    historical = pd.read_csv(
        HISTORICAL_DATA
    )

    historical["DateTime"] = pd.to_datetime(
        historical["DateTime"]
    )

    # --------------------------------------------------------
    # Find live files
    # --------------------------------------------------------

    live_files = get_live_station_files()

    if not live_files:

        print("\nNo live station files found.")

        return pd.DataFrame()

    # --------------------------------------------------------
    # Preprocess live data
    # --------------------------------------------------------

    print("\nPreprocessing live station data...")

    live = load_and_preprocess_raw_data(
        live_files
    )

    if live.empty:

        print("\nNo live observations available.")

        return pd.DataFrame()

    # --------------------------------------------------------
    # We only need the recent historical tail.
    #
    # Feature engineering requires four previous
    # observations for every station.
    #
    # Keep 4 historical observations per station.
    # --------------------------------------------------------

    historical_tail = (
        historical
        .sort_values(
            ["Location", "DateTime"]
        )
        .groupby("Location")
        .tail(4)
    )

    # --------------------------------------------------------
    # Combine historical context + live data
    # --------------------------------------------------------

    combined = pd.concat(
        [
            historical_tail,
            live
        ],
        ignore_index=True
    )

    # --------------------------------------------------------
    # Remove duplicate observations
    # --------------------------------------------------------

    combined = combined.drop_duplicates(
        subset=[
            "Location",
            "DateTime"
        ],
        keep="last"
    )

    combined = combined.sort_values(
        [
            "Location",
            "DateTime"
        ]
    ).reset_index(drop=True)

    print(
        f"\nCombined observations: "
        f"{len(combined):,}"
    )

    # --------------------------------------------------------
    # Feature engineering
    # --------------------------------------------------------

    print("\nRunning feature engineering...")

    features = create_features(
        combined
    )

    if features.empty:

        print(
            "\nNo feature rows generated."
        )

        print(
            "Check that the live station data "
            "contains valid sensor readings."
        )

        return pd.DataFrame()

    print(
        f"Feature rows generated: "
        f"{len(features):,}"
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Keep only observations originating from LIVE data.
    #
    # Historical rows are present only to provide the
    # rolling/difference context.
    # --------------------------------------------------------

    live_times = live[
        [
            "Location",
            "DateTime"
        ]
    ].drop_duplicates()

    features = features.merge(
        live_times,
        on=[
            "Location",
            "DateTime"
        ],
        how="inner"
    )

    print(
        f"Live feature rows: "
        f"{len(features):,}"
    )

    return features


# ============================================================
# LIVE ANOMALY DETECTION
# ============================================================

def run_live_cycle():

    print("\n")
    print("=" * 70)
    print("STARTING LIVE PIPELINE CYCLE")
    print("=" * 70)

    # --------------------------------------------------------
    # Build live feature dataset
    # --------------------------------------------------------

    features = build_live_dataset()

    if features.empty:

        print(
            "\nNo live feature rows available."
        )

        return

    # --------------------------------------------------------
    # Load saved models and infer
    # --------------------------------------------------------

    print(
        "\nRunning anomaly detection "
        "using SAVED models..."
    )

    results = run_live_inference(
        features
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    os.makedirs(
        "outputs",
        exist_ok=True
    )

    results.to_csv(
        LIVE_OUTPUT,
        index=False
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    anomalies = int(
        results["Ensemble_Anomaly"].sum()
    )

    total = len(results)

    percentage = (
        anomalies / total * 100
        if total > 0
        else 0
    )

    print("\n")
    print("=" * 70)
    print("LIVE ANOMALY RESULTS")
    print("=" * 70)

    print(
        f"\nLive observations analysed: {total:,}"
    )

    print(
        f"Anomalies detected: {anomalies:,}"
    )

    print(
        f"Anomaly percentage: "
        f"{percentage:.2f}%"
    )

    print("\nModel agreement:")

    print(
        results[
            "Model_Agreement"
        ]
        .value_counts()
        .sort_index()
    )

    print(
        f"\nResults saved to:\n"
        f"{LIVE_OUTPUT}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("SKYGUARD AI")
    print("LIVE WEATHER SENSOR ANOMALY DETECTION")
    print("=" * 70)

    os.makedirs(
        "models",
        exist_ok=True
    )

    os.makedirs(
        "outputs",
        exist_ok=True
    )

    # ========================================================
    # ONE-TIME HISTORICAL TRAINING
    # ========================================================

    if TRAIN_FROM_HISTORICAL:

        train_from_historical()

    else:

        print(
            "\nSkipping historical training."
        )

        print(
            "Using existing models from models/."
        )

    # ========================================================
    # LIVE LOOP
    # ========================================================

    while True:

        try:

            run_live_cycle()

        except Exception as e:

            print("\n")
            print("=" * 70)
            print("ERROR IN LIVE PIPELINE")
            print("=" * 70)

            print(
                f"\n{type(e).__name__}: {e}"
            )

        print("\n")
        print("=" * 70)
        print(
            "Pipeline cycle complete."
        )

        print(
            "Next cycle in 15 minutes..."
        )

        print("=" * 70)

        time.sleep(
            UPDATE_INTERVAL
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()