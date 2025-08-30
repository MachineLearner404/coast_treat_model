from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
import joblib
import pandas as pd
import requests
from datetime import datetime, timedelta

app = FastAPI(title="Coastal Alert ML Backend")

# Load model and scaler
model = joblib.load("backend/knn_model.pkl")
scaler = joblib.load("backend/scaler.pkl")

@app.get("/")
def home():
    return {"message": "Coastal Alert FastAPI Backend Running!"}

@app.get("/predict_from_coords/")
def predict_from_coords(
    latitude: float = Query(..., description="Latitude of location"),
    longitude: float = Query(..., description="Longitude of location")
):
    # Current UTC time rounded to nearest hour
    now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    one_hour_later = now + timedelta(hours=1)

    start_hour = now.isoformat() + "Z"
    end_hour = one_hour_later.isoformat() + "Z"

    # --- Marine API ---
    marine_url = (
        f"https://marine-api.open-meteo.com/v1/marine?"
        f"latitude={latitude}&longitude={longitude}"
        f"&hourly=wave_height,wave_direction,wave_period,sea_level_height_msl,"
        f"sea_surface_temperature,ocean_current_direction,ocean_current_velocity,"
        f"swell_wave_direction,swell_wave_period"
        f"&start_hour={start_hour}&end_hour={end_hour}"
    )

    marine_resp = requests.get(marine_url)
    if marine_resp.status_code != 200:
        raise HTTPException(status_code=marine_resp.status_code, detail="Failed to fetch marine data")
    marine_data = marine_resp.json()["hourly"]

    # --- Weather API ---
    weather_url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={latitude}&longitude={longitude}"
        f"&hourly=temperature_2m,relative_humidity_2m,precipitation,weather_code,"
        f"pressure_msl,surface_pressure,wind_speed_10m,wind_direction_10m,wind_direction_100m"
        f"&start={start_hour}&end={end_hour}"
    )

    weather_resp = requests.get(weather_url)
    if weather_resp.status_code != 200:
        raise HTTPException(status_code=weather_resp.status_code, detail="Failed to fetch weather data")
    weather_data = weather_resp.json()["hourly"]

    # Merge features into single record
    merged_data = {
        "wave_height": marine_data["wave_height"][0],
        "wave_direction": marine_data["wave_direction"][0],
        "wave_period": marine_data["wave_period"][0],
        "sea_level_height_msl": marine_data["sea_level_height_msl"][0],
        "sea_surface_temperature": marine_data["sea_surface_temperature"][0],
        "ocean_current_direction": marine_data["ocean_current_direction"][0],
        "ocean_current_velocity": marine_data["ocean_current_velocity"][0],
        "swell_wave_direction": marine_data["swell_wave_direction"][0],
        "swell_wave_period": marine_data["swell_wave_period"][0],
        "temperature_2m": weather_data["temperature_2m"][0],
        "relative_humidity_2m": weather_data["relative_humidity_2m"][0],
        "precipitation": weather_data["precipitation"][0],
        "weather_code": weather_data["weather_code"][0],
        "pressure_msl": weather_data["pressure_msl"][0],
        "surface_pressure": weather_data["surface_pressure"][0],
        "wind_speed_10m": weather_data["wind_speed_10m"][0],
        "wind_direction_10m": weather_data["wind_direction_10m"][0],
        "wind_direction_100m": weather_data["wind_direction_100m"][0]
    }

    # Convert to DataFrame and scale
    df = pd.DataFrame([merged_data])
    scaled_data = scaler.transform(df)

    # Predict
    prediction = model.predict(scaled_data)[0]
    merged_data["prediction"] = str(prediction)
    return JSONResponse(content=merged_data)