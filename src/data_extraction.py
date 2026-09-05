
import requests
import pandas as pd
import time
from pathlib import Path


# ============================================================
# SETTINGS
# ============================================================

# Station names and their original NOAA IDs are kept so that
# the CSV filenames remain compatible with the old system.

STATIONS = {
    "Chennai": {
        "id": "432790-99999",
        "latitude": 13.0827,
        "longitude": 80.2707
    },

    "Mumbai": {
        "id": "430030-99999",
        "latitude": 19.0760,
        "longitude": 72.8777
    },

    "New_Delhi": {
        "id": "421820-99999",
        "latitude": 28.6139,
        "longitude": 77.2090
    },

    "Kolkata": {
        "id": "428090-99999",
        "latitude": 22.5726,
        "longitude": 88.3639
    },

    "Hyderabad": {
        "id": "431280-99999",
        "latitude": 17.3850,
        "longitude": 78.4867
    }
}


# Collect weather every 15 minutes
UPDATE_INTERVAL = 15 * 60


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_FOLDER = PROJECT_ROOT / "weather_data"

OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# OPEN-METEO API
# ============================================================

API_URL = "https://api.open-meteo.com/v1/forecast"


# ============================================================
# GET CURRENT WEATHER
# ============================================================

def get_weather(station_name, station_info):

    latitude = station_info["latitude"]
    longitude = station_info["longitude"]

    params = {
        "latitude": latitude,
        "longitude": longitude,

        # Current weather variables
        "current": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "surface_pressure"
        ),

        # Use the local timezone of the coordinates
        "timezone": "auto"
    }

    print("\n" + "=" * 60)
    print("Station:", station_name)
    print("Latitude:", latitude)
    print("Longitude:", longitude)
    print("=" * 60)

    try:

        response = requests.get(
            API_URL,
            params=params,
            timeout=30
        )

        response.raise_for_status()

        data = response.json()

    except requests.RequestException as e:

        print("ERROR connecting to Open-Meteo:")
        print(e)

        return None

    except ValueError as e:

        print("ERROR reading API response:")
        print(e)

        return None


    # ========================================================
    # CHECK RESPONSE
    # ========================================================

    if "current" not in data:

        print("ERROR: Current weather data not found.")

        return None


    current = data["current"]


    # ========================================================
    # EXTRACT VALUES
    # ========================================================

    timestamp = current.get("time")

    temperature = current.get(
        "temperature_2m"
    )

    humidity = current.get(
        "relative_humidity_2m"
    )

    pressure = current.get(
        "surface_pressure"
    )


    # ========================================================
    # VALIDATE VALUES
    # ========================================================

    if timestamp is None:

        print("ERROR: Timestamp missing.")

        return None


    if temperature is None:

        print("ERROR: Temperature missing.")

        return None


    if humidity is None:

        print("ERROR: Humidity missing.")

        return None


    if pressure is None:

        print("ERROR: Pressure missing.")

        return None


    # ========================================================
    # CREATE DATAFRAME
    # ========================================================

    df = pd.DataFrame(
        [
            {
                "Location": station_name,
                "Time": pd.to_datetime(timestamp),
                "Temperature_C": float(temperature),
                "Humidity_Percent": float(humidity),
                "Pressure_hPa": float(pressure)
            }
        ]
    )


    # ========================================================
    # DISPLAY CURRENT READING
    # ========================================================

    print("Time:", timestamp)
    print(
        "Temperature:",
        temperature,
        "°C"
    )
    print(
        "Humidity:",
        humidity,
        "%"
    )
    print(
        "Pressure:",
        pressure,
        "hPa"
    )

    return df


# ============================================================
# UPDATE STATION CSV
# ============================================================

def update_station_file(
    station_name,
    station_info,
    new_data
):

    station_id = station_info["id"]

    filename = (
        f"{station_name}_{station_id}.csv"
    )

    filepath = OUTPUT_FOLDER / filename


    # ========================================================
    # LOAD EXISTING DATA
    # ========================================================

    if filepath.exists():

        try:

            old_data = pd.read_csv(
                filepath
            )

            old_data["Time"] = pd.to_datetime(
                old_data["Time"],
                errors="coerce"
            )

        except Exception as e:

            print(
                "WARNING: Could not read existing CSV:"
            )

            print(e)

            old_data = pd.DataFrame()

    else:

        old_data = pd.DataFrame()


    # ========================================================
    # COMBINE OLD + NEW DATA
    # ========================================================

    if not old_data.empty:

        combined = pd.concat(
            [
                old_data,
                new_data
            ],
            ignore_index=True
        )

    else:

        combined = new_data.copy()


    # ========================================================
    # REMOVE INVALID TIMESTAMPS
    # ========================================================

    combined = combined.dropna(
        subset=["Time"]
    )


    # ========================================================
    # REMOVE DUPLICATES
    # ========================================================

    combined = combined.drop_duplicates(
        subset=[
            "Location",
            "Time"
        ],
        keep="last"
    )


    # ========================================================
    # SORT
    # ========================================================

    combined = combined.sort_values(
        "Time"
    ).reset_index(drop=True)


    # ========================================================
    # SAVE
    # ========================================================

    combined.to_csv(
        filepath,
        index=False
    )


    print("\nCSV UPDATED")
    print("-" * 40)
    print("File:", filepath)
    print("Total rows:", len(combined))
    print(
        "Latest reading:",
        combined.iloc[-1]["Time"]
    )


# ============================================================
# UPDATE ALL STATIONS
# ============================================================

def update_all_stations():

    print("\n")
    print("=" * 60)
    print("SKYGUARD-AI WEATHER DATA UPDATE")
    print("=" * 60)


    for station_name, station_info in STATIONS.items():

        try:

            new_data = get_weather(
                station_name,
                station_info
            )

            if new_data is not None:

                update_station_file(
                    station_name,
                    station_info,
                    new_data
                )

        except Exception as e:

            print(
                f"ERROR processing {station_name}:"
            )

            print(e)


    print("\n")
    print("=" * 60)
    print("UPDATE CYCLE COMPLETE")
    print("=" * 60)


# ============================================================
# CONTINUOUS LIVE LOOP
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("SkyGaurd-AI Live Weather Collector")
    print("=" * 60)

    print("\nData source: Open-Meteo")

    print(
        "Update interval:",
        UPDATE_INTERVAL // 60,
        "minutes"
    )

    print(
        "Output folder:",
        OUTPUT_FOLDER
    )

    print("\nPress CTRL+C to stop.\n")


    while True:

        try:

            update_all_stations()

            print(
                f"\nWaiting {UPDATE_INTERVAL // 60} "
                "minutes for next update..."
            )

            time.sleep(
                UPDATE_INTERVAL
            )

        except KeyboardInterrupt:

            print(
                "\n\nWeather collector stopped."
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