
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Interactive Readiness Predictor (with Live Pareto)", layout="wide")

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

# Train Random Forest inline
X = df[features]
y = np.random.normal(75, 10, len(X))
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# UI for inputs
st.title("🔧 Interactive Readiness Predictor + Live Pareto (Artificial data)")
st.markdown("Adjust the sliders to simulate base performance and view real-time predictions.")

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

# Live Pareto Chart
st.subheader("📊 Feature Importance (Live from Model)")

importances = model.feature_importances_
importance_df = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values(by="Importance", ascending=False)

fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(importance_df["Feature"], importance_df["Importance"], color="skyblue")
ax.invert_yaxis()
ax.set_title("Top Factors Affecting Readiness")
ax.set_xlabel("Relative Importance")
st.pyplot(fig)

st.markdown("---")
