import streamlit as st
import requests
import json
import pandas as pd

st.set_page_config(page_title="Coastal Alert ML", layout="wide")
st.title("Coastal Alert Prediction")

# Backend URL
API_URL = "http://127.0.0.1:8000"  # Update if hosted elsewhere

# Sidebar: Choose input method
input_method = st.sidebar.radio("Select Input Method", ["Manual Input", "Upload JSON File"])

if input_method == "Manual Input":
    st.header("Manual Input for Single Prediction")
    
    # Take user input
    wave_height = st.number_input("Wave Height")
    wave_direction = st.number_input("Wave Direction")
    wave_period = st.number_input("Wave Period")
    sea_level_height_msl = st.number_input("Sea Level Height MSL")
    sea_surface_temperature = st.number_input("Sea Surface Temperature")
    ocean_current_direction = st.number_input("Ocean Current Direction")
    ocean_current_velocity = st.number_input("Ocean Current Velocity")
    swell_wave_direction = st.number_input("Swell Wave Direction")
    swell_wave_period = st.number_input("Swell Wave Period")
    temperature_2m = st.number_input("Temperature 2m")
    relative_humidity_2m = st.number_input("Relative Humidity 2m")
    precipitation = st.number_input("Precipitation")
    weather_code = st.number_input("Weather Code")
    pressure_msl = st.number_input("Pressure MSL")
    surface_pressure = st.number_input("Surface Pressure")
    wind_speed_10m = st.number_input("Wind Speed 10m")
    wind_direction_10m = st.number_input("Wind Direction 10m")
    wind_direction_100m = st.number_input("Wind Direction 100m")

    if st.button("Predict"):
        # Prepare JSON payload
        payload = {
            "wave_height": wave_height,
            "wave_direction": wave_direction,
            "wave_period": wave_period,
            "sea_level_height_msl": sea_level_height_msl,
            "sea_surface_temperature": sea_surface_temperature,
            "ocean_current_direction": ocean_current_direction,
            "ocean_current_velocity": ocean_current_velocity,
            "swell_wave_direction": swell_wave_direction,
            "swell_wave_period": swell_wave_period,
            "temperature_2m": temperature_2m,
            "relative_humidity_2m": relative_humidity_2m,
            "precipitation": precipitation,
            "weather_code": weather_code,
            "pressure_msl": pressure_msl,
            "surface_pressure": surface_pressure,
            "wind_speed_10m": wind_speed_10m,
            "wind_direction_10m": wind_direction_10m,
            "wind_direction_100m": wind_direction_100m
        }
        try:
            response = requests.post(f"{API_URL}/predict", json=payload)
            result = response.json()
            st.success(f"✅ Prediction: {result['prediction']}")
        except Exception as e:
            st.error(f"Error: {e}")

elif input_method == "Upload JSON File":
    st.header("Upload JSON File for Batch Prediction")
    uploaded_file = st.file_uploader("Choose a JSON file", type="json")

    if uploaded_file is not None:
        try:
            data_json = json.load(uploaded_file)
            st.write("Preview of uploaded data:")
            if isinstance(data_json, dict):
                st.json(data_json)
            elif isinstance(data_json, list):
                st.dataframe(pd.DataFrame(data_json))
            
            if st.button("Predict from File"):
                files = {"file": (uploaded_file.name, uploaded_file, "application/json")}
                response = requests.post(f"{API_URL}/predict_file/", files=files)
                predictions = response.json().get("predictions", [])
                
                st.success("✅ Predictions:")
                st.write(predictions)
        except Exception as e:
            st.error(f"Error reading JSON: {e}")