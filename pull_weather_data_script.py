import requests
import pandas as pd
import numpy as np


# -------------------------------------------------
# 1️⃣ FETCH WEATHER DATA
# -------------------------------------------------
def fetch_open_meteo_weather(lat, lon, start_date, end_date):

    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "temperature_2m",
            "dewpoint_2m",
            "relative_humidity_2m",
            "precipitation",
            "wind_speed_10m",
            "wind_direction_10m",
            "visibility",
            "surface_pressure"
        ],
        "timezone": "auto"
    }

    response = requests.get(url, params=params)
    data = response.json()

    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"])

    return df


# -------------------------------------------------
# 2️⃣ CLEAN DATA
# -------------------------------------------------
def clean_data(df):

    df["visibility"] = df["visibility"].fillna(10)
    df = df.ffill()

    return df


# -------------------------------------------------
# 3️⃣ FEATURE ENGINEERING
# -------------------------------------------------
def engineer_features(df):

    df = df.rename(columns={
        "temperature_2m": "Temp (C)",
        "dewpoint_2m": "Dew Point Temp (C)",
        "relative_humidity_2m": "Rel Hum (%)",
        "precipitation": "Precip. Amount (mm)",
        "wind_speed_10m": "Wind Spd (km/h)",
        "visibility": "Visibility (km)",
        "surface_pressure": "Stn Press (kPa)"
    })

    # Convert pressure from hPa → kPa
    df["Stn Press (kPa)"] = df["Stn Press (kPa)"] / 10

    # Time cyclical features
    df["hour"] = df["time"].dt.hour
    df["month"] = df["time"].dt.month

    df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24)
    df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24)

    df["month_sin"] = np.sin(2*np.pi*df["month"]/12)
    df["month_cos"] = np.cos(2*np.pi*df["month"]/12)

    # Wind direction encoding
    df["wind_dir_sin"] = np.sin(np.deg2rad(df["wind_direction_10m"]))
    df["wind_dir_cos"] = np.cos(np.deg2rad(df["wind_direction_10m"]))

    # Wind cubic power
    df["wind_power_potential"] = df["Wind Spd (km/h)"] ** 3

    return df


# -------------------------------------------------
# 4️⃣ ADD MODEL-SPECIFIC FEATURES
# -------------------------------------------------
def add_model_specific_features(df,
                                capacity_mw=120,
                                prev_output_value=60):

    # Lag feature (replace with real previous output later)
    df["output_mw_lag_1h"] = prev_output_value

    # Capacity
    df["available_capacity_mw"] = capacity_mw

    # Weather quality rule
    df["weather_quality_good"] = np.where(
        (df["Wind Spd (km/h)"] > 15) &
        (df["Precip. Amount (mm)"] < 2),
        1, 0
    )

    df["weather_quality_bad"] = 1 - df["weather_quality_good"]

    return df


# -------------------------------------------------
# 5️⃣ SELECT FINAL MODEL FEATURES (18 FEATURES)
# -------------------------------------------------
def get_model_ready_dataframe(df):

    model_features = [
        'Temp (C)',
        'Dew Point Temp (C)',
        'Rel Hum (%)',
        'Precip. Amount (mm)',
        'Wind Spd (km/h)',
        'Visibility (km)',
        'Stn Press (kPa)',
        'hour_sin',
        'hour_cos',
        'month_sin',
        'month_cos',
        'wind_dir_sin',
        'wind_dir_cos',
        'wind_power_potential',
        'output_mw_lag_1h',
        'available_capacity_mw',
        'weather_quality_bad',
        'weather_quality_good',
        'output_mw'
    ]

    return df[model_features]

# -------------------------------------------------
# 🔁 PIPELINE FUNCTION FOR INFERENCE
# -------------------------------------------------
def build_model_ready_weather_dataframe(
        lat,
        lon,
        start_date,
        end_date,
        capacity_mw=120,
        prev_output_value=60):

    df = fetch_open_meteo_weather(lat, lon, start_date, end_date)
    df = clean_data(df)
    df = engineer_features(df)
    df = add_model_specific_features(
        df,
        capacity_mw=capacity_mw,
        prev_output_value=prev_output_value
    )

    df_final = get_model_ready_dataframe(df)

    return df_final



# -------------------------------------------------
# 🚀 MAIN EXECUTION
# -------------------------------------------------
if __name__ == "__main__":

    print("Fetching weather data...")

    df = fetch_open_meteo_weather(
        lat=43.65,
        lon=-79.38,
        start_date="2024-01-01",
        end_date="2024-01-05"
    )

    print("Cleaning data...")
    df = clean_data(df)

    print("Engineering features...")
    df = engineer_features(df)

    print("Adding model-specific features...")
    df = add_model_specific_features(
        df,
        capacity_mw=120,
        prev_output_value=60
    )

    df_final = get_model_ready_dataframe(df)

    print("\n✅ FINAL MODEL-READY DATA")
    print(df_final.head())
    print("\nShape:", df_final.shape)

    df_final.to_csv("model_ready_weather.csv", index=False)
    print("\nSaved as model_ready_weather.csv")

