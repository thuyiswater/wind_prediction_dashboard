import numpy as np
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
import joblib
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

from pull_weather_data_script import (
    fetch_open_meteo_weather,
    clean_data,
    engineer_features,
    add_model_specific_features,
    get_model_ready_dataframe
)

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
LATITUDE = 42.31   # Chatham-Kent
LONGITUDE = -82.08
HOURS_REQUIRED = 24

MODEL_T1_PATH = r"models/best_cnn_lstm_T1.keras"
MODEL_T3_PATH = r"models/best_cnn_lstm_T3.keras"
ANN_MODEL_T1_PATH = r"models/best_ann_model_T1.pkl"
ANN_MODEL_T3_PATH = r"models/best_ann_model_T3.pkl"
SCALER_PATH = r"scripts/label_encoders_scaler.pkl"

# -------------------------------------------------
# THE MONKEY PATCH: Force Keras to ignore the keyword
# -------------------------------------------------
original_dense_init = Dense.__init__

def patched_dense_init(self, *args, **kwargs):
    # Rip the bad argument out of the dictionary if it exists
    kwargs.pop('quantization_config', None)
    # Pass everything else to the normal Keras Dense layer
    original_dense_init(self, *args, **kwargs)

# Overwrite the built-in Keras Dense layer behavior globally
Dense.__init__ = patched_dense_init

# -------------------------------------------------
# LOAD MODELS ONCE
# -------------------------------------------------
print("Loading models...")
# Now load them normally without custom_objects!
model_T1 = load_model(MODEL_T1_PATH, compile=False)
model_T3 = load_model(MODEL_T3_PATH, compile=False)
ann_model_T1 = joblib.load(ANN_MODEL_T1_PATH)
ann_model_T3 = joblib.load(ANN_MODEL_T3_PATH)
scaler_dict = joblib.load(SCALER_PATH)
scaler = scaler_dict['scaler']
print("The scaler is looking for these exact columns:", scaler.feature_names_in_)

def predict_ensemble(input_data):
    pred_T1 = model_T1.predict(input_data, verbose=0)[0][0]
    pred_T3 = model_T3.predict(input_data, verbose=0)[0][0]
    return (pred_T1 + pred_T3) / 2

def predict_ann_ensemble(input_df):
    """Predicts the error using the Scikit-Learn MLPRegressors (ANN)"""

    ann_input = input_df.drop(columns=['output_mw'], errors='ignore')
    pred_T1 = ann_model_T1.predict(ann_input)[0]
    pred_T3 = ann_model_T3.predict(ann_input)[0]

    return (pred_T1 + pred_T3) / 2

# BUILD BASE WEATHER DATA (WITHOUT LAG)
def build_base_weather_dataframe():

    end_date = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=2)).strftime("%Y-%m-%d")

    df = fetch_open_meteo_weather(
        lat=LATITUDE,
        lon=LONGITUDE,
        start_date=start_date,
        end_date=end_date
    )

    df = clean_data(df)
    df = engineer_features(df)
    df = add_model_specific_features(df, capacity_mw=120)

    df = df.tail(HOURS_REQUIRED)

    if len(df) < HOURS_REQUIRED:
        raise ValueError("Not enough hourly data fetched.")

    return df


# -------------------------------------------------
# RECURSIVE 24-HOUR PREDICTION
# -------------------------------------------------
import numpy as np

def recursive_24h_forecast(base_df):
    predictions = []
    corrected_predictions = []
    prev_output = 60  # Dummy value for first hour only (or use actual last known value)

    # Inject the missing 'output_mw' column
    if 'output_mw' not in base_df.columns:
        base_df['output_mw'] = prev_output

    # Format the data
    df_model = get_model_ready_dataframe(base_df)
    rolling_buffer = df_model.tail(24).copy()
    
    for i in range(HOURS_REQUIRED):
        # Grab the current 24-hour block
        data = rolling_buffer.values
        # Identify the indices of the columns that need scaling
        scale_cols = [
            'output_mw', 'available_capacity_mw', 'output_mw', 'Temp (C)',
            'Dew Point Temp (C)', 'Rel Hum (%)', 'Precip. Amount (mm)',
            'Wind Spd (km/h)', 'Visibility (km)', 'Stn Press (kPa)'
        ]
        
        # Expand dims to shape (1, 24, 19)
        model_input = np.expand_dims(data, axis=0)
        pred = predict_ensemble(model_input)
        predictions.append(pred)

        # Correction Model (ANN) predicts the Error
        current_hour_df = rolling_buffer.iloc[[-1]].copy()
        ann_error_pred = predict_ann_ensemble(current_hour_df)
        # 3. Calculate Final Corrected Prediction
        corrected_pred = pred + ann_error_pred
        corrected_predictions.append(corrected_pred)

        # Slide Window
        new_row = rolling_buffer.iloc[-1].copy()
        new_row["output_mw_lag_1h"] = new_row["output_mw"]
        new_row["output_mw"] = corrected_pred
    
        rolling_buffer = pd.concat([rolling_buffer.iloc[1:], pd.DataFrame([new_row])], ignore_index=True)

    return predictions, corrected_predictions

def run_test_benchmarks():
    """Runs the ANN ensemble on the T1 and T3 test datasets and returns the metrics."""
    # Attempt to load the test datasets. Update these paths if your data folder is located elsewhere!
    try:
        test_df_T1 = pd.read_csv('../data/processed/error_data/wind_power_error_test_1.csv')
        test_df_T3 = pd.read_csv('../data/processed/error_data/wind_power_error_test_2.csv')
    except FileNotFoundError:
        # Fallback path just in case the app is run from a different directory
        test_df_T1 = pd.read_csv('data/processed/error_data/wind_power_error_test_1.csv')
        test_df_T3 = pd.read_csv('data/processed/error_data/wind_power_error_test_2.csv')

    # Prepare data
    y_test_T1 = test_df_T1['error']
    X_test_T1 = test_df_T1.drop(columns=['error'])

    y_test_T3 = test_df_T3['error']
    X_test_T3 = test_df_T3.drop(columns=['error'])

    # 1. Ensemble evaluated on T1 Test
    preds_T1_on_T1 = ann_model_T1.predict(X_test_T1)
    preds_T3_on_T1 = ann_model_T3.predict(X_test_T1)
    ensemble_preds_T1 = (preds_T1_on_T1 + preds_T3_on_T1) / 2
    
    rmse_T1 = np.sqrt(mean_squared_error(y_test_T1, ensemble_preds_T1))
    r2_T1 = r2_score(y_test_T1, ensemble_preds_T1)

    # 2. Ensemble evaluated on T3 Test
    preds_T1_on_T3 = ann_model_T1.predict(X_test_T3)
    preds_T3_on_T3 = ann_model_T3.predict(X_test_T3)
    ensemble_preds_T3 = (preds_T1_on_T3 + preds_T3_on_T3) / 2
    
    rmse_T3 = np.sqrt(mean_squared_error(y_test_T3, ensemble_preds_T3))
    r2_T3 = r2_score(y_test_T3, ensemble_preds_T3)

    return rmse_T1, r2_T1, rmse_T3, r2_T3

# -------------------------------------------------
# MAIN
# -------------------------------------------------
if __name__ == "__main__":

    print("Building weather features...")
    base_weather_df = build_base_weather_dataframe()

    print("Running recursive 24-hour inference...")
    base_preds, corrected_preds = recursive_24h_forecast(base_weather_df)

    print("\n-----------------------------------------")
    for i, (base, corr) in enumerate(zip(base_preds, corrected_preds), 1):
        print(f"Hour {i}: Base = {base:.2f} MW | Corrected = {corr:.2f} MW")
    print("-----------------------------------------")
    print(f"Final Hour Forecast -> Base: {base_preds[-1]:.2f} MW | Corrected: {corrected_preds[-1]:.2f} MW")
    print("-----------------------------------------\n")

