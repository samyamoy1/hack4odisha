import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import requests
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier

# ====== Sample Weather Data (Training) ======
data = {
    "day": [1, 2, 3, 4, 5, 6, 7],
    "temp": [31, 32, 30, 29, 30, 31, 30],
    "humidity": [80, 78, 83, 88, 85, 80, 82],
    "wind": [10, 11, 9, 14, 8, 12, 10],
    "rain_mm": [1.2, 0.0, 0.0, 6.5, 4.0, 2.0, 5.0]
}

df = pd.DataFrame(data)

# ====== Train Temperature Model ======
X_temp = df[["day", "humidity", "wind", "rain_mm"]]
y_temp = df["temp"]

temp_model = LinearRegression()
temp_model.fit(X_temp, y_temp)

# ====== Train Rain Model ======
df["rain_today"] = df["rain_mm"].apply(lambda x: 1 if x > 0 else 0)
X_rain = df[["temp", "humidity", "wind"]]
y_rain = df["rain_today"]

rain_model = RandomForestClassifier(n_estimators=100, random_state=42)
rain_model.fit(X_rain, y_rain)

# ====== Streamlit UI ======
st.title("🌦 Weather & Rain Prediction App (Hackathon Edition)")
st.write("Enter your location to get today's weather and tomorrow's prediction!")

# ====== User Location Input ======
location = st.text_input("Enter your city:", "Kolkata")

if location:
    try:
        # ====== Get Today’s Weather (from OpenWeatherMap) ======
        api_key = "332c7aeda1d896aa5c4ce26b89c28096"  # replace with your key
        url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}&units=metric"
        response = requests.get(url).json()

        if response.get("main"):
            today_temp = response["main"]["temp"]
            today_humidity = response["main"]["humidity"]
            today_wind = response["wind"]["speed"]

            st.subheader(f"📍 Weather in {location} Today")
            st.write(f"🌡 Temperature: {today_temp}°C")
            st.write(f"💧 Humidity: {today_humidity}%")
            st.write(f"🌬 Wind: {today_wind} km/h")

            # Show weather icon
            if "weather" in response:
                icon_code = response["weather"][0]["icon"]
                desc = response["weather"][0]["description"].title()
                icon_url = f"http://openweathermap.org/img/wn/{icon_code}@2x.png"
                st.image(icon_url, caption=desc)

            # ====== Predict Tomorrow ======
            today_day = pd.Timestamp.now().day
            tomorrow_day = today_day + 1

            tomorrow_features = pd.DataFrame([[tomorrow_day, today_humidity, today_wind, 2.0]],
                                             columns=["day", "humidity", "wind", "rain_mm"])
            predicted_temp = temp_model.predict(tomorrow_features)[0]
            temp_lower = predicted_temp - 1
            temp_upper = predicted_temp + 1

            rain_features = pd.DataFrame([[predicted_temp, today_humidity, today_wind]],
                                          columns=["temp", "humidity", "wind"])
            rain_prediction = rain_model.predict(rain_features)[0]
            rain_proba = rain_model.predict_proba(rain_features)[0][1] * 100

            st.subheader("📅 Tomorrow's Forecast")
            st.write(f"🌡 Temperature: {predicted_temp:.2f}°C "
                     f"(Range: {temp_lower:.2f}°C - {temp_upper:.2f}°C)")
            st.write(f"🌧 Rain Prediction: {'Yes' if rain_prediction == 1 else 'No'} "
                     f"({rain_proba:.2f}% chance)")

            # ====== Umbrella Advice ======
            if rain_proba > 50:
                st.success("🌂 Carry an umbrella tomorrow!")
            else:
                st.info("😎 No umbrella needed, enjoy your day!")

            # ====== Fun Extra Advice ======
            if today_temp > 35:
                st.warning("🔥 Stay hydrated, it’s hot outside!")
            elif today_temp < 10:
                st.warning("🧥 Wear a jacket, it’s chilly!")
            if rain_proba > 70:
                st.warning("⚡ Thunderstorms likely, avoid outdoor plans.")

        else:
            st.error("❌ Could not fetch weather data. Check the city name.")

    except Exception as e:
        st.error(f"Error: {e}")
