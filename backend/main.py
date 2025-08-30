from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import joblib
import numpy as np
import pandas as pd
import json

# Initialize app
app = FastAPI(title="Coastal Alert ML Backend")

# Load model and scaler
model = joblib.load("backend/knn_model.pkl")
scaler = joblib.load("backend/scaler.pkl")

# Define expected input schema for single record
class Features(BaseModel):
    wave_height: float
    wave_direction: float
    wave_period: float
    sea_level_height_msl: float
    sea_surface_temperature: float
    ocean_current_direction: float
    ocean_current_velocity: float
    swell_wave_direction: float
    swell_wave_period: float
    temperature_2m: float
    relative_humidity_2m: float
    precipitation: float
    weather_code: float
    pressure_msl: float
    surface_pressure: float
    wind_speed_10m: float
    wind_direction_10m: float
    wind_direction_100m: float

@app.get("/")
def home():
    return {"message": "Coastal Alert FastAPI Backend Running!"}

# Endpoint: Single record prediction using Pydantic
@app.post("/predict")
def predict(data: Features):
    # Convert input to numpy array
    input_data = np.array([[
        data.wave_height,
        data.wave_direction,
        data.wave_period,
        data.sea_level_height_msl,
        data.sea_surface_temperature,
        data.ocean_current_direction,
        data.ocean_current_velocity,
        data.swell_wave_direction,
        data.swell_wave_period,
        data.temperature_2m,
        data.relative_humidity_2m,
        data.precipitation,
        data.weather_code,
        data.pressure_msl,
        data.surface_pressure,
        data.wind_speed_10m,
        data.wind_direction_10m,
        data.wind_direction_100m
    ]])

    # Scale input
    scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(scaled)[0]

    return {"prediction": str(prediction)}

# Endpoint: JSON file upload for batch or single record prediction
@app.post("/predict_file/")
async def predict_file(file: UploadFile = File(...)):
    # Read uploaded file
    contents = await file.read()
    try:
        data_json = json.loads(contents)
    except Exception as e:
        return JSONResponse(content={"error": "Invalid JSON file", "details": str(e)}, status_code=400)

    # Convert to DataFrame (supports single record or list of records)
    if isinstance(data_json, dict):
        df = pd.DataFrame([data_json])
    elif isinstance(data_json, list):
        df = pd.DataFrame(data_json)
    else:
        return JSONResponse(content={"error": "JSON must be dict or list of dicts"}, status_code=400)

    # Scale features
    scaled_data = scaler.transform(df)

    # Predict
    predictions = model.predict(scaled_data)

    # Return predictions as JSON
    result = {"predictions": predictions.tolist()}
    return JSONResponse(content=result)