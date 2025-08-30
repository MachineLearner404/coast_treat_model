import streamlit as st
import requests
import json
import pandas as pd
from datetime import datetime, timedelta

st.set_page_config(page_title="Coastal Alert ML", layout="wide")
st.title("Coastal Alert Prediction")

API_URL = "http://127.0.0.1:8000"  # Update if hosted elsewhere

# Sidebar: Choose input method
input_method = st.sidebar.radio("Select Input Method", ["Manual Input", "Upload JSON File"])

# Helper function to get API data
def get_merged_api_data(lat, lon):
    # Calculate time range: 1 hour before now to now
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(hours=1)
    start_str = start_time.strftime("%Y-%m-%dT%H:00")
    end_str = end_time.strftime("%Y-%m-%dT%H:00")
    
    # Marine API
    marine_url = (
        f"https://marine-api.open-meteo.com/v1/marine?latitude={lat}&longitude={lon}"
        f"&hourly=wave_height,wave_direction,wave_period,sea_level_height_msl,"
        f"sea_surface_temperature,ocean_current_direction,ocean_current_velocity,"
        f"swell_wave_direction,swell_wave_period&start_hour={start_str}&end_hour={end_str}"
    )
    marine_data = requests.get(marine_url).json()
    
    # Weather API
    weather_url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
        f"&hourly=temperature_2m,relative_humidity_2m,precipitation,weather_code,"
        f"pressure_msl,surface_pressure,wind_speed_10m,wind_direction_10m,wind_direction_100m"
        f"&start={start_str}&end={end_str}"
    )
    weather_data = requests.get(weather_url).json()
    
    # Merge hourly data into one dict (assumes only 1-hour range)
    merged_data = {}
    for key in marine_data['hourly']:
        merged_data[key] = marine_data['hourly'][key][0]  # take first value
    for key in weather_data['hourly']:
        merged_data[key] = weather_data['hourly'][key][0]
    
    return merged_data

if input_method == "Manual Input":
    st.header("Manual Input for Single Prediction")
    lat = st.number_input("Latitude", value=17.68)
    lon = st.number_input("Longitude", value=83.20)

    if st.button("Predict"):
        try:
            payload = get_merged_api_data(lat, lon)
            st.json(payload)
            
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
            st.json(data_json) if isinstance(data_json, dict) else st.dataframe(pd.DataFrame(data_json))
            
            if st.button("Predict from File"):
                predictions = []
                if isinstance(data_json, list):
                    for item in data_json:
                        lat, lon = item['latitude'], item['longitude']
                        payload = get_merged_api_data(lat, lon)
                        response = requests.post(f"{API_URL}/predict", json=payload)
                        predictions.append(response.json())
                elif isinstance(data_json, dict):
                    lat, lon = data_json['latitude'], data_json['longitude']
                    payload = get_merged_api_data(lat, lon)
                    response = requests.post(f"{API_URL}/predict", json=payload)
                    predictions.append(response.json())
                
                st.success("✅ Predictions:")
                st.write(predictions)
        except Exception as e:
            st.error(f"Error reading JSON: {e}")