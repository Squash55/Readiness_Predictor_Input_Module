
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Readiness Predictor (Interactive)", layout="wide")

# Load model
model = joblib.load("optimized_rf_model.joblib")

# Feature input interface
st.title("🔧 Interactive Readiness Predictor (Artificial data)")
st.markdown("Adjust the sliders below to simulate different readiness driver profiles.")

inputs = {}
inputs['Mission Complexity'] = st.slider("Mission Complexity", 0, 100, 50)
inputs['Maintenance Burden'] = st.slider("Maintenance Burden", 0, 100, 50)
inputs['Personnel Gaps'] = st.slider("Personnel Gaps", 0, 100, 50)
inputs['Logistics Readiness'] = st.slider("Logistics Readiness", 0, 100, 50)
inputs['Equipment Availability'] = st.slider("Equipment Availability", 0, 100, 50)
inputs['Cyber Resilience'] = st.slider("Cyber Resilience", 0, 100, 50)
inputs['Fuel Supply Score'] = st.slider("Fuel Supply Score", 0, 100, 50)
inputs['Flight Ops Readiness'] = st.slider("Flight Ops Readiness", 0, 100, 50)
inputs['Medical Support Score'] = st.slider("Medical Support Score", 0, 100, 50)
inputs['Training Level'] = st.slider("Training Level", 0, 100, 50)

input_df = pd.DataFrame([inputs])
pred = model.predict(input_df)[0]

st.subheader("📈 Predicted Readiness Score")
st.metric(label="Predicted Readiness", value=f"{pred:.1f}")

# Feedback
st.subheader("🧠 Smart Feedback")
if pred > 85:
    st.success("Excellent! This base shows high readiness potential.")
elif pred > 70:
    st.info("Moderate readiness. Consider boosting Cyber Resilience or Maintenance.")
elif pred > 60:
    st.warning("Low readiness. Focus on Equipment and Personnel Gaps.")
else:
    st.error("Critical readiness concern. Immediate intervention needed.")

st.markdown("---")
