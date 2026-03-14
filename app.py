import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# LOAD MODEL
# =========================
model = pickle.load(open("model.pkl", "rb"))

st.title("⚡ PJM Energy Consumption Forecast (Next 30 Days)")

st.write("Enter latest energy consumption values to forecast next 30 days.")

# =========================
# USER INPUT
# =========================

lag24 = st.number_input("Energy Consumption 24 hours ago (MW)", value=30000)

lag168 = st.number_input("Energy Consumption 168 hours ago (MW)", value=30000)

month = st.selectbox("Current Month", list(range(1,13)))

dayofweek = st.selectbox(
    "Current Day",
    [0,1,2,3,4,5,6],
    format_func=lambda x: ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"][x]
)

hour = st.slider("Current Hour", 0, 23, 12)

# =========================
# FORECAST BUTTON
# =========================

if st.button("Forecast Next 30 Days"):

    future_hours = 24*30
    predictions = []

    current_hour = hour
    current_day = dayofweek
    current_month = month

    for i in range(future_hours):

        input_data = np.array([[current_hour, current_day, current_month, lag24, lag168]])

        pred = model.predict(input_data)[0]

        predictions.append(pred)

        # update lag values
        lag168 = lag24
        lag24 = pred

        # update hour
        current_hour += 1
        if current_hour > 23:
            current_hour = 0
            current_day = (current_day + 1) % 7

    # =========================
    # CREATE FORECAST DATAFRAME
    # =========================

    future_dates = pd.date_range(
        start=pd.Timestamp.now(),
        periods=future_hours,
        freq="H"
    )

    forecast_df = pd.DataFrame({
        "Datetime": future_dates,
        "Predicted_MW": predictions
    })

    st.subheader("📊 30 Days Forecast Data")
    st.dataframe(forecast_df)

    # =========================
    # PLOT FORECAST
    # =========================

    fig, ax = plt.subplots()

    ax.plot(forecast_df["Datetime"], forecast_df["Predicted_MW"])

    ax.set_title("Next 30 Days Energy Forecast")
    ax.set_xlabel("Datetime")
    ax.set_ylabel("Energy Consumption (MW)")

    st.pyplot(fig)