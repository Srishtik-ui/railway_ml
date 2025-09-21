# streamlit_app.py
import streamlit as st
import joblib, os, json, requests
import numpy as np
from datetime import datetime, time

# Path to model artifact inside the repo (models/railway_classifier.joblib)
MODEL_PATH = "./models/railway_classifier.joblib"

st.set_page_config(page_title="Railway On-Time Predictor", layout="centered")

st.title("Railway On-Time Predictor")
st.write("Predict whether a train will be delayed (>5 min) or on-time. You may run prediction locally (load model file) or call a deployed API.")

# Load artifacts if available
model_loaded = False
model_tuple = None
metrics = {}
if os.path.exists(MODEL_PATH):
    try:
        artifacts = joblib.load(MODEL_PATH)
        model_tuple = artifacts.get("model_tuple")
        metrics = artifacts.get("metrics", {})
        model_loaded = True
    except Exception as e:
        st.warning(f"Could not load model artifact at {MODEL_PATH}: {e}")

# Sidebar: choose mode
st.sidebar.header("Mode / Settings")
use_api = st.sidebar.checkbox("Call remote prediction API instead of local model", value=False)
predict_api_url = st.sidebar.text_input("Prediction API URL (e.g. https://myapi.example.com/predict)", value=os.environ.get("PREDICT_API_URL",""))
api_token = st.sidebar.text_input("API token (if using remote API)", value=os.environ.get("API_TOKEN",""), type="password")

st.sidebar.markdown("**Model info**")
if model_loaded:
    st.sidebar.write("Model loaded from:", MODEL_PATH)
    st.sidebar.write("Saved metrics:", metrics)
else:
    st.sidebar.write("No local model found. Use API mode or upload model to `models/railway_classifier.joblib`.")

# Input form
with st.form("predict_form"):
    st.subheader("Input")
    train_number = st.text_input("Train number", value="12345")
    station_code = st.text_input("Station code", value="NDLS")
    col1, col2 = st.columns(2)
    with col1:
        scheduled_date = st.date_input("Scheduled arrival date", value=datetime.utcnow().date())
    with col2:
        scheduled_time = st.time_input("Scheduled arrival time", value=time(8,0))
    scheduled_dt = datetime.combine(scheduled_date, scheduled_time)

    speed_kmph = st.number_input("Speed (km/h)", min_value=0.0, value=40.0, step=1.0)
    rolling_stock_health = st.number_input("Rolling stock health (0.0 - 1.0)", min_value=0.0, max_value=1.0, value=0.9, step=0.01)

    st.markdown("You can leave token blank for local model. If using remote API, supply token in sidebar.")
    submitted = st.form_submit_button("Predict")

if submitted:
    # Build feature vector matching serve.py: minute_of_day, day_of_week, is_weekend, speed, health
    minute_of_day = scheduled_dt.hour * 60 + scheduled_dt.minute
    day_of_week = scheduled_dt.weekday()
    is_weekend = 1 if day_of_week in (5,6) else 0

    # Attempt to fetch online features (optional) - only for explanation
    prev_delay, avg_prev3 = 0.0, 0.0
    # if you have a Redis or online store, you can pull values here (not implemented)
    # For example, pull from a small endpoint if you have one:
    if predict_api_url and use_api:
        # Call remote API
        payload = {
            "train_number": train_number,
            "station_code": station_code,
            "scheduled_arrival": scheduled_dt.isoformat(),
            "speed_kmph": speed_kmph,
            "rolling_stock_health": rolling_stock_health,
            "token": api_token
        }
        try:
            resp = requests.post(predict_api_url, json=payload, timeout=10)
            resp.raise_for_status()
            out = resp.json()
            st.success("Prediction from remote API:")
            st.json(out)
        except Exception as e:
            st.error(f"Remote API call failed: {e}")
    else:
        # Local model path
        if not model_loaded:
            st.error("No local model available. Either upload model to models/railway_classifier.joblib or enable API mode.")
        else:
            kind, mdl = model_tuple
            vec = np.array([[minute_of_day, day_of_week, is_weekend, float(speed_kmph), float(rolling_stock_health)]])
            try:
                if kind == "lgb":
                    proba = mdl.predict(vec, num_iteration=mdl.best_iteration)[0]
                else:
                    # sklearn model
                    if hasattr(mdl, "predict_proba"):
                        proba = mdl.predict_proba(vec)[0][1]
                    else:
                        proba = float(mdl.predict(vec)[0])
                threshold = metrics.get("threshold", 0.5)
                delayed = bool(proba > threshold)
                st.markdown("**Local prediction**")
                st.write(f"Predicted delayed probability: **{proba:.3f}**")
                st.write(f"Threshold used: **{threshold:.2f}** → Delayed: **{delayed}**")
                st.write("Input feature values:")
                st.json({
                    "minute_of_day": minute_of_day,
                    "day_of_week": day_of_week,
                    "is_weekend": is_weekend,
                    "speed_kmph": speed_kmph,
                    "rolling_stock_health": rolling_stock_health
                })
            except Exception as e:
                st.error(f"Prediction failed: {e}")

st.info("To deploy: push this repo to GitHub and connect it in Streamlit Cloud (instructions below).")
