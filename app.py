import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import requests
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier

# Sample weather data (training)

data = {
    "day": [1, 2, 3, 4, 5, 6, 7],
    "temp": [31, 32, 30, 29, 30, 31, 30],
    "humidity": [80, 78, 83, 88, 85, 80, 82],
    "wind": [10, 11, 9, 14, 8, 12, 10],
    "rain_mm": [1.2, 0.0, 0.0, 6.5, 4.0, 2.0, 5.0]
}

df = pd.DataFrame(data)

# Train temperature model
X_temp = df[["day", "humidity", "wind", "rain_mm"]]
y_temp = df["temp"]
temp_model = LinearRegression().fit(X_temp, y_temp)

# Train rain model
df["rain_today"] = df["rain_mm"].apply(lambda x: 1 if x > 0 else 0)
X_rain = df[["temp", "humidity", "wind"]]
y_rain = df["rain_today"]
rain_model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_rain, y_rain)

# Helper functions

def get_weather(city, api_key):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    return requests.get(url).json()

def get_air_quality(lat, lon, api_key):
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}"
    return requests.get(url).json()

# Streamlit UI

st.title("Weather & Air Quality App")
st.write("Check today's weather and a simple forecast for tomorrow.")

city = st.text_input("Enter your city:", "Kolkata")
expected_rain = st.slider("Expected rainfall tomorrow (mm):", 0.0, 10.0, 2.0)

api_key = "332c7aeda1d896aa5c4ce26b89c28096"  # put your own key

if city:
    try:
        weather = get_weather(city, api_key)

        if weather.get("main"):
            temp_now = weather["main"]["temp"]
            humidity_now = weather["main"]["humidity"]
            wind_now = weather["wind"]["speed"]
            lat, lon = weather["coord"]["lat"], weather["coord"]["lon"]

            st.subheader(f"Weather in {city} today")
            st.write(f"Temperature: {temp_now} °C")
            st.write(f"Humidity: {humidity_now}%")
            st.write(f"Wind Speed: {wind_now} km/h")

            # Air Quality
            aqi_data = get_air_quality(lat, lon, api_key)
            if "list" in aqi_data:
                aqi = aqi_data["list"][0]["main"]["aqi"]
                components = aqi_data["list"][0]["components"]

                st.subheader("Air Quality Index")
                st.write(f"AQI Level: {aqi}")
                st.write(f"PM2.5: {components['pm2_5']} µg/m³")
                st.write(f"PM10: {components['pm10']} µg/m³")
                st.write(f"NO₂: {components['no2']} µg/m³")
                st.write(f"CO: {components['co']} µg/m³")

            # Tomorrow's prediction
            today_day = pd.Timestamp.now().day
            tomorrow_day = today_day + 1

            tomorrow_features = pd.DataFrame(
                [[tomorrow_day, humidity_now, wind_now, expected_rain]],
                columns=["day", "humidity", "wind", "rain_mm"]
            )
            predicted_temp = temp_model.predict(tomorrow_features)[0]

            rain_features = pd.DataFrame(
                [[predicted_temp, humidity_now, wind_now]],
                columns=["temp", "humidity", "wind"]
            )
            rain_prediction = rain_model.predict(rain_features)[0]
            rain_proba = rain_model.predict_proba(rain_features)[0][1] * 100

            st.subheader("Tomorrow's Forecast (estimate)")
            st.write(f"Temperature: {predicted_temp:.1f} °C")
            st.write(f"Chance of Rain: {rain_proba:.1f}%")
            if rain_prediction == 1:
                st.write("You may need an umbrella.")
            else:
                st.write("No umbrella needed.")

        else:
            st.error("Could not fetch weather data. Please check the city name.")

    except Exception as e:
        st.error(f"Error: {e}")
