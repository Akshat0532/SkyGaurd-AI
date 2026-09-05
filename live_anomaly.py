import time
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

from src.feature_engineering import create_features


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

WEATHER_FOLDER = PROJECT_ROOT / "weather_data"
MODEL_FOLDER = PROJECT_ROOT / "models"
OUTPUT_FOLDER = PROJECT_ROOT / "outputs"

OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = OUTPUT_FOLDER / "live_anomalies.csv"


# ============================================================
# MODEL FEATURES
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
# LOAD TRAINED MODELS
# ============================================================

print("=" * 60)
print("LOADING TRAINED MODELS")
print("=" * 60)

try:

    scaler = joblib.load(
        MODEL_FOLDER / "scaler.pkl"
    )

    isolation_forest = joblib.load(
        MODEL_FOLDER / "isolation_forest.pkl"
    )

    ecod = joblib.load(
        MODEL_FOLDER / "ecod.pkl"
    )

    copod = joblib.load(
        MODEL_FOLDER / "copod.pkl"
    )

    hbos = joblib.load(
        MODEL_FOLDER / "hbos.pkl"
    )

    print("All models loaded successfully.")

except Exception as e:

    print("ERROR loading models:")
    print(e)

    raise


# ============================================================
# LOAD LIVE WEATHER DATA
# ============================================================

def load_weather_data():

    files = list(
        WEATHER_FOLDER.glob("*.csv")
    )

    if not files:

        print("No weather CSV files found.")

        return pd.DataFrame()

    dataframes = []

    for file in files:

        try:

            df = pd.read_csv(file)

            # ------------------------------------------------
            # Check required columns
            # ------------------------------------------------

            required_columns = [
                "Location",
                "Time",
                "Temperature_C",
                "Humidity_Percent",
                "Pressure_hPa"
            ]

            missing_columns = [
                column
                for column in required_columns
                if column not in df.columns
            ]

            if missing_columns:

                print(
                    f"Skipping {file.name}."
                )

                print(
                    "Missing columns:",
                    missing_columns
                )

                continue

            # ------------------------------------------------
            # Convert Time -> DateTime
            #
            # The CSV format remains unchanged.
            # DateTime is only used internally.
            # ------------------------------------------------

            df["DateTime"] = pd.to_datetime(
                df["Time"],
                errors="coerce"
            )

            df = df.dropna(
                subset=["DateTime"]
            )

            dataframes.append(df)

        except Exception as e:

            print(
                f"Error reading {file.name}:"
            )

            print(e)

    if not dataframes:

        return pd.DataFrame()

    # ========================================================
    # COMBINE STATIONS
    # ========================================================

    data = pd.concat(
        dataframes,
        ignore_index=True
    )

    # ========================================================
    # REMOVE DUPLICATES
    # ========================================================

    data = data.drop_duplicates(
        subset=[
            "Location",
            "DateTime"
        ],
        keep="last"
    )

    # ========================================================
    # SORT
    # ========================================================

    data = data.sort_values(
        [
            "Location",
            "DateTime"
        ]
    ).reset_index(drop=True)

    return data


# ============================================================
# PREPARE LIVE DATA
# ============================================================

def prepare_live_data(df):

    if df.empty:

        return pd.DataFrame()

    df = df.copy()

    # ========================================================
    # ENSURE NUMERIC WEATHER VALUES
    # ========================================================

    numeric_columns = [
        "Temperature_C",
        "Humidity_Percent",
        "Pressure_hPa"
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # ========================================================
    # PRESSURE MISSING INDICATOR
    # ========================================================

    df["Pressure_Missing"] = (
        df["Pressure_hPa"]
        .isna()
        .astype(int)
    )

    processed = []

    # ========================================================
    # PROCESS EACH STATION SEPARATELY
    # ========================================================

    for location, station_df in df.groupby(
        "Location"
    ):

        station_df = station_df.copy()

        station_df = station_df.sort_values(
            "DateTime"
        )

        station_df = station_df.set_index(
            "DateTime"
        )

        # ----------------------------------------------------
        # Convert 15-minute observations to 3-hour intervals.
        #
        # This matches the time resolution used by the
        # original SkyGaurd-AI feature engineering pipeline.
        # ----------------------------------------------------

        station_df = station_df.resample(
            "3h"
        ).first()

        station_df["Location"] = location

        # ----------------------------------------------------
        # Interpolate pressure
        # ----------------------------------------------------

        station_df["Pressure_hPa"] = (
            station_df["Pressure_hPa"]
            .interpolate(
                method="time"
            )
        )

        station_df = station_df.reset_index()

        processed.append(
            station_df
        )

    if not processed:

        return pd.DataFrame()

    result = pd.concat(
        processed,
        ignore_index=True
    )

    result = result.sort_values(
        [
            "Location",
            "DateTime"
        ]
    ).reset_index(drop=True)

    return result


# ============================================================
# RUN LIVE ANOMALY DETECTION
# ============================================================

def detect_anomalies():

    print("\n")
    print("=" * 60)
    print("RUNNING LIVE ANOMALY DETECTION")
    print("=" * 60)

    # ========================================================
    # LOAD WEATHER DATA
    # ========================================================

    raw_data = load_weather_data()

    if raw_data.empty:

        print("No weather data available.")

        return

    print(
        "Raw observations:",
        len(raw_data)
    )

    # ========================================================
    # PREPARE DATA
    # ========================================================

    prepared_data = prepare_live_data(
        raw_data
    )

    if prepared_data.empty:

        print(
            "No prepared weather data available."
        )

        return

    print(
        "Prepared observations:",
        len(prepared_data)
    )

    # ========================================================
    # FEATURE ENGINEERING
    # ========================================================

    try:

        features = create_features(
            prepared_data
        )

    except Exception as e:

        print(
            "Feature engineering failed:"
        )

        print(e)

        return

    if features.empty:

        print(
            "Not enough historical observations "
            "to calculate features yet."
        )

        return

    print(
        "Feature rows:",
        len(features)
    )

    # ========================================================
    # CHECK MODEL FEATURES
    # ========================================================

    missing_features = [
        feature
        for feature in MODEL_FEATURES
        if feature not in features.columns
    ]

    if missing_features:

        print(
            "ERROR: Missing model features:"
        )

        for feature in missing_features:

            print(
                " -",
                feature
            )

        return

    # ========================================================
    # GET LATEST OBSERVATION FOR EACH STATION
    # ========================================================

    latest = (
        features
        .sort_values(
            [
                "Location",
                "DateTime"
            ]
        )
        .groupby(
            "Location"
        )
        .tail(1)
        .copy()
    )

    if latest.empty:

        print(
            "No latest observations available."
        )

        return

    # ========================================================
    # PREPARE MODEL INPUT
    # ========================================================

    X = latest[
        MODEL_FEATURES
    ].copy()

    # --------------------------------------------------------
    # Replace infinite values
    # --------------------------------------------------------

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # --------------------------------------------------------
    # Fill remaining missing values
    # --------------------------------------------------------

    X = X.fillna(0)

    # ========================================================
    # APPLY SAVED SCALER
    # ========================================================

    try:

        X_scaled = scaler.transform(
            X
        )

    except Exception as e:

        print(
            "ERROR applying scaler:"
        )

        print(e)

        return

    # ========================================================
    # MODEL PREDICTIONS
    # ========================================================

    try:

        # ----------------------------------------------------
        # Isolation Forest
        #
        # -1 = anomaly
        #  1 = normal
        # ----------------------------------------------------

        isolation_prediction = (
            isolation_forest.predict(
                X_scaled
            )
        )

        # ----------------------------------------------------
        # PyOD models
        #
        # 1 = anomaly
        # 0 = normal
        # ----------------------------------------------------

        ecod_prediction = ecod.predict(
            X_scaled
        )

        copod_prediction = copod.predict(
            X_scaled
        )

        hbos_prediction = hbos.predict(
            X_scaled
        )

    except Exception as e:

        print(
            "ERROR during model prediction:"
        )

        print(e)

        return

    # ========================================================
    # CONVERT MODEL PREDICTIONS TO 0 / 1
    # ========================================================

    latest["IsolationForest_Anomaly"] = (
        isolation_prediction == -1
    ).astype(int)

    latest["ECOD_Anomaly"] = (
        ecod_prediction == 1
    ).astype(int)

    latest["COPOD_Anomaly"] = (
        copod_prediction == 1
    ).astype(int)

    latest["HBOS_Anomaly"] = (
        hbos_prediction == 1
    ).astype(int)

    # ========================================================
    # MODEL AGREEMENT
    # ========================================================

    latest["Model_Agreement"] = (
        latest["IsolationForest_Anomaly"]
        + latest["ECOD_Anomaly"]
        + latest["COPOD_Anomaly"]
        + latest["HBOS_Anomaly"]
    )

    # ========================================================
    # ENSEMBLE ANOMALY
    #
    # At least 3 out of 4 models must agree.
    # ========================================================

    latest["Ensemble_Anomaly"] = (
        latest["Model_Agreement"] >= 3
    ).astype(int)

    # ========================================================
    # ANOMALY SEVERITY
    # ========================================================

    def get_severity(agreement):

        if agreement >= 4:

            return "High"

        elif agreement == 3:

            return "Medium"

        elif agreement == 2:

            return "Low"

        else:

            return "Normal"

    latest["Anomaly_Severity"] = (
        latest["Model_Agreement"]
        .apply(get_severity)
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    output_columns = [
        "Location",
        "DateTime",

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

        "Pressure_Missing",

        "IsolationForest_Anomaly",
        "ECOD_Anomaly",
        "COPOD_Anomaly",
        "HBOS_Anomaly",

        "Model_Agreement",
        "Ensemble_Anomaly",
        "Anomaly_Severity"
    ]

    # Only select columns that actually exist
    output_columns = [
        column
        for column in output_columns
        if column in latest.columns
    ]

    latest = latest[
        output_columns
    ]

    latest.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    print("\n")
    print("=" * 60)
    print("LIVE RESULTS")
    print("=" * 60)

    for _, row in latest.iterrows():

        if row["Ensemble_Anomaly"] == 1:

            status = "ANOMALY"

        else:

            status = "NORMAL"

        print(
            f"{row['Location']:15} | "
            f"{row['DateTime']} | "
            f"{status:7} | "
            f"Agreement: "
            f"{row['Model_Agreement']}/4 | "
            f"Severity: "
            f"{row['Anomaly_Severity']}"
        )

    print("\nResults saved to:")
    print(OUTPUT_FILE)


# ============================================================
# CONTINUOUS LIVE LOOP
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 60)
    print("SKYGUARD-AI LIVE ANOMALY DETECTION")
    print("=" * 60)

    print(
        "\nWeather source: Open-Meteo"
    )

    print(
        "Weather update interval: 15 minutes"
    )

    print(
        "Anomaly check interval: 15 minutes"
    )

    print(
        "\nPress CTRL+C to stop."
    )

    while True:

        try:

            detect_anomalies()

            print(
                "\nWaiting for next update..."
            )

            time.sleep(
                15 * 60
            )

        except KeyboardInterrupt:

            print(
                "\n\nLive anomaly detection stopped."
            )

            break

        except Exception as e:

            print(
                "\nUnexpected error:"
            )

            print(e)

            print(
                "\nRetrying in 60 seconds..."
            )

            time.sleep(60)