import streamlit as st
import requests
import json
import pandas as pd

st.set_page_config(page_title="Coastal Alert ML", layout="wide")
st.title("Coastal Alert Prediction")

# Change this to your deployed backend URL
API_URL = "https://coast-treat-model.onrender.com"

# Sidebar: Choose input method
input_method = st.sidebar.radio("Select Input Method", ["Manual Input", "Upload JSON File"])

# --- Helper function to call FastAPI backend ---
def get_prediction(lat, lon):
    url = f"{API_URL}/predict_from_coords/"
    payload = {"latitude": lat, "longitude": lon}
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Backend error {response.status_code}", "details": response.text}
    except Exception as e:
        return {"error": str(e)}

# --- Manual Input ---
if input_method == "Manual Input":
    st.header("Manual Input for Single Prediction")
    lat = st.number_input("Latitude", value=17.68)
    lon = st.number_input("Longitude", value=83.20)

    if st.button("Predict"):
        result = get_prediction(lat, lon)
        if "error" in result:
            st.error(result)
        else:
            st.json(result)  # Show full merged data
            st.success(f"✅ Prediction: {result['prediction']}")

# --- Upload JSON ---
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
                if isinstance(data_json, list):  # Multiple coords
                    for item in data_json:
                        lat, lon = item['latitude'], item['longitude']
                        result = get_prediction(lat, lon)
                        predictions.append(result)
                elif isinstance(data_json, dict):  # Single coord
                    lat, lon = data_json['latitude'], data_json['longitude']
                    result = get_prediction(lat, lon)
                    predictions.append(result)

                st.success("✅ Predictions:")
                st.write(predictions)
        except Exception as e:
            st.error(f"Error reading JSON: {e}")