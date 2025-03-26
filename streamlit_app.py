
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Readiness Predictor (Inline Model)", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("Expanded_Readiness_Spreadsheet.csv")
    features = [
        "Mission Complexity", "Maintenance Burden", "Personnel Gaps", "Logistics Readiness",
        "Equipment Availability", "Cyber Resilience", "Fuel Supply Score", "Flight Ops Readiness",
        "Medical Support Score", "Training Level"
    ]
    for col in features:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=features)
    return df, features

df, features = load_data()

# Simulate target and train model inline
X = df[features]
y = np.random.normal(75, 10, len(X))

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# User input interface
st.title("🔧 Interactive Readiness Predictor (Artificial data)")
st.markdown("Adjust the sliders below to simulate different readiness driver profiles.")

input_values = {}
for feature in features:
    input_values[feature] = st.slider(feature, 0, 100, 50)

input_df = pd.DataFrame([input_values])
pred = model.predict(input_df)[0]

# Prediction Output
st.subheader("📈 Predicted Readiness Score")
st.metric(label="Predicted Readiness", value=f"{pred:.1f}")

# Smart Feedback
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
