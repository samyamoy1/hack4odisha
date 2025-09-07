import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import requests
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier

# ---------------------------
# Training Sample Data
# ---------------------------
data = {
    "day": [1, 2, 3, 4, 5, 6, 7],
    "temp": [31, 32, 30, 29, 30, 31, 30],
    "humidity": [80, 78, 83, 88, 85, 80, 82],
    "wind": [10, 11, 9, 14, 8, 12, 10],
    "rain_mm": [1.2, 0.0, 0.0, 6.5, 4.0, 2.0, 5.0],
    "aqi": [60, 75, 90, 140, 120, 80, 100]
}
df = pd.DataFrame(data)

# Train temperature model
X_temp = df[["day", "humidity", "wind", "rain_mm"]]
y_temp = df["temp"]
temp_model = LinearRegression()
temp_model.fit(X_temp, y_temp)

# Train AQI classification model
df["aqi_category"] = df["aqi"].apply(
    lambda x: 0 if x <= 100 else (1 if x <= 200 else 2)
)
X_aqi = df[["temp", "humidity", "wind"]]
y_aqi = df["aqi_category"]
aqi_model = RandomForestClassifier(n_estimators=100, random_state=42)
aqi_model.fit(X_aqi, y_aqi)

# ---------------------------
# Helper Functions
# ---------------------------
def get_air_quality_category(aqi_value):
    if aqi_value <= 50:
        return "Good", "🟢 Green"
    elif aqi_value <= 100:
        return "Moderate", "🟡 Yellow"
    elif aqi_value <= 150:
        return "Unhealthy for Sensitive Groups", "🟠 Orange"
    elif aqi_value <= 200:
        return "Unhealthy", "🔴 Red"
    elif aqi_value <= 300:
        return "Very Unhealthy", "🟣 Purple"
    else:
        return "Hazardous", "⚫ Black"

def fetch_weather(city, api_key):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    return requests.get(url).json()

def fetch_air_quality(lat, lon, api_key):
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}"
    data = requests.get(url).json()
    if "list" in data and data["list"]:
        aqi_code = data["list"][0]["main"]["aqi"]
        mapping = {1: 30, 2: 80, 3: 120, 4: 180, 5: 300}
        return mapping.get(aqi_code, 0)
    return None

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🌦 Weather + AQI Forecast App ")

city = st.text_input("Enter your city:", "Kolkata")
api_key = "332c7aeda1d896aa5c4ce26b89c28096"  # replace with your OpenWeatherMap API key

if city:
    try:
        weather_data = fetch_weather(city, api_key)
        if weather_data.get("main"):
            today_temp = weather_data["main"]["temp"]
            today_humidity = weather_data["main"]["humidity"]
            today_wind = weather_data["wind"]["speed"]
            lat, lon = weather_data["coord"]["lat"], weather_data["coord"]["lon"]

            # Show today's weather
            st.subheader(f"📍 Weather in {city} Today")
            st.write(f"🌡 Temperature: {today_temp}°C")
            st.write(f"💧 Humidity: {today_humidity}%")
            st.write(f"🌬 Wind: {today_wind} km/h")

            # Get today's AQI
            aqi_value = fetch_air_quality(lat, lon, api_key)
            if aqi_value is not None:
                category, color = get_air_quality_category(aqi_value)
                st.subheader("🌍 Air Quality Today")
                st.write(f"**Status:** {category} ({color})")
                st.write(f"**AQI Value:** {aqi_value}/500")

                # ---------------------------
                # AQI Legend Table
                # ---------------------------
                st.subheader("📝 AQI Legend")
                aqi_legend = {
                    "Good": "🟢 Green (0-50)",
                    "Moderate": "🟡 Yellow (51-100)",
                    "Unhealthy for Sensitive Groups": "🟠 Orange (101-150)",
                    "Unhealthy": "🔴 Red (151-200)",
                    "Very Unhealthy": "🟣 Purple (201-300)",
                    "Hazardous": "⚫ Black (301+)"
                }
                legend_df = pd.DataFrame(list(aqi_legend.items()), columns=["Category", "Indicator"])
                st.table(legend_df)

                # ---------------------------
                # Tomorrow Prediction (ML)
                # ---------------------------
                tomorrow_day = pd.Timestamp.now().day + 1

                # Predict temperature
                tomorrow_features = pd.DataFrame(
                    [[tomorrow_day, today_humidity, today_wind, 2.0]],
                    columns=["day", "humidity", "wind", "rain_mm"]
                )
                predicted_temp = temp_model.predict(tomorrow_features)[0]

                # Predict AQI category
                aqi_features = pd.DataFrame(
                    [[predicted_temp, today_humidity, today_wind]],
                    columns=["temp", "humidity", "wind"]
                )
                predicted_aqi_cat = aqi_model.predict(aqi_features)[0]
                predicted_aqi_value = [80, 150, 250][predicted_aqi_cat]

                pred_category, pred_color = get_air_quality_category(predicted_aqi_value)

                st.subheader("📅 Tomorrow's Forecast (ML)")
                st.write(f"🌡 Predicted Temperature: {predicted_temp:.2f}°C")
                st.write(f"🌍 Predicted AQI: {pred_category} ({pred_color}) — {predicted_aqi_value}/500")

        else:
            st.error("❌ Could not fetch weather data. Check the city name.")
    except Exception as e:
        st.error(f"Error: {e}")

