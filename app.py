import streamlit as st
import pandas as pd
import joblib
from datetime import date

# Load model
model = joblib.load("model/traffic_model.pkl")

st.title("🚦 Traffic Congestion Prediction & Route Recommendation")

# ---------------- USER INPUTS ----------------
junction = st.selectbox("Select Current Junction", [1, 2, 3, 4])
selected_date = st.date_input("Select Date", date.today())
hour = st.slider("Hour of Day", 0, 23, 9)

# ---------------- FEATURE FUNCTIONS ----------------
def time_block(hour):
    if 6 <= hour <= 9:
        return 1
    elif 10 <= hour <= 16:
        return 2
    elif 17 <= hour <= 20:
        return 3
    else:
        return 0

def season(month):
    if month in [3, 4, 5, 6]:
        return 0
    elif month in [7, 8, 9]:
        return 1
    else:
        return 2

def time_penalty(hour):
    if 8 <= hour <= 10 or 17 <= hour <= 19:
        return 30
    elif 11 <= hour <= 16:
        return 10
    elif 0 <= hour <= 5:
        return -15
    else:
        return 0

def delay_estimate(score):
    if score < 40:
        return "0–2 minutes"
    elif score < 80:
        return "5–10 minutes"
    else:
        return "15+ minutes"

# ---------------- DERIVED FEATURES ----------------
is_weekend = 1 if selected_date.weekday() >= 5 else 0
is_peak = 1 if (8 <= hour <= 10 or 17 <= hour <= 19) else 0
season_val = season(selected_date.month)
hour_squared = hour ** 2

junction_complexity = {
    1: 18,
    2: 12,
    3: 6,
    4: 25
}

# ---------------- PREDICTION ----------------
if st.button("Predict"):

    congestion_scores = {}
    ml_predictions = {}

    for j in [1, 2, 3, 4]:
        input_df = pd.DataFrame({
            "Junction": [j],
            "Hour": [hour],
            "Hour_Squared": [hour_squared],
            "TimeBlock": [time_block(hour)],
            "IsPeakHour": [is_peak],
            "IsWeekend": [is_weekend],
            "Season": [season_val]
        })

        ml_value = model.predict(input_df)[0]
        ml_predictions[j] = ml_value

        final_score = (
            ml_value
            + time_penalty(hour)
            + junction_complexity[j]
        )

        congestion_scores[j] = final_score

    # ---------------- CURRENT ROUTE ----------------
    st.subheader("🚘 Current Route Analysis")

    ml_only = ml_predictions[junction]
    final_score = congestion_scores[junction]

    st.write(f"**ML Predicted Vehicles:** {int(ml_only)}")
    st.write(f"**Rule-based Adjustment Applied**")
    st.write(f"**Final Congestion Score:** {int(final_score)}")
    st.write(f"**Estimated Delay:** {delay_estimate(final_score)}")

    if final_score < 40:
        st.success("Congestion Level: Low 🟢")
    elif final_score < 80:
        st.warning("Congestion Level: Medium 🟡")
    else:
        st.error("Congestion Level: High 🔴")

    # ---------------- ROUTE COMPARISON ----------------
    st.subheader("🧠 Route Comparison")
    for j in congestion_scores:
        st.write(f"Junction {j}: Score = {int(congestion_scores[j])}")

    # ---------------- OPTIMAL ROUTE ----------------
    optimal_junction = min(congestion_scores, key=congestion_scores.get)

    st.subheader("🛣 Recommended Route")
    if optimal_junction != junction:
        st.success(f"Choose **Junction {optimal_junction}** (Lowest congestion)")
    else:
        st.success("Your current route is already optimal 🎉")
