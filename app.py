import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import streamlit as st
import datetime
import altair as alt
import google.generativeai as genai
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
import json

# -------------------- Gemini API Setup --------------------
GEMINI_API_KEY = "AIzaSyB7zaBH4aRkIsB4mt3iHvsfELvwI1Eh-xQ"  # 🔑 Replace this
genai.configure(api_key=GEMINI_API_KEY)

# ✅ Create model once (not inside function)
model = genai.GenerativeModel("models/gemini-1.5-flash")

# -------------------- Generate synthetic data --------------------
np.random.seed(42)
days, hours_per_day = 30, 24
total_rows = days * hours_per_day

dates = [datetime.datetime.now() - datetime.timedelta(days=i//24, hours=i%24) for i in range(total_rows)]
dates = [d.strftime("%Y-%m-%d %H:%M:%S") for d in dates]

temp = np.random.normal(30, 3, total_rows)
humidity = np.random.normal(80, 5, total_rows)
wind = np.random.normal(10, 2, total_rows)
rain_mm = np.random.choice([0, 0.5, 1, 2, 5, 10], total_rows, p=[0.5, 0.2, 0.1, 0.1, 0.05, 0.05])

aqi = []
for t, h, w, r in zip(temp, humidity, wind, rain_mm):
    base = 50 + (t - 25) * 2 + (h - 70) * 0.5 - w * 1.5 + r * 2
    noise = np.random.normal(0, 10)
    aqi.append(min(max(int(base + noise), 10), 300))

df = pd.DataFrame({
    "date": dates,
    "temp": temp.round(1),
    "humidity": humidity.round(1),
    "wind": wind.round(1),
    "rain_mm": rain_mm,
    "aqi": aqi
})
df['day'] = pd.to_datetime(df['date']).dt.day
df['aqi_category'] = df['aqi'].apply(lambda x: 0 if x <= 100 else (1 if x <= 200 else 2))

# -------------------- Train ML models --------------------
temp_model = LinearRegression()
temp_model.fit(df[['day', 'humidity', 'wind', 'rain_mm']], df['temp'])

aqi_model = RandomForestClassifier(n_estimators=200, random_state=42)
aqi_model.fit(df[['temp', 'humidity', 'wind']], df['aqi_category'])

# -------------------- Helper functions --------------------
def get_air_quality_category(aqi_value):
    if aqi_value <= 50: return "Good", "🟢"
    elif aqi_value <= 100: return "Moderate", "🟡"
    elif aqi_value <= 150: return "Unhealthy for Sensitive Groups", "🟠"
    elif aqi_value <= 200: return "Unhealthy", "🔴"
    elif aqi_value <= 300: return "Very Unhealthy", "🟣"
    else: return "Hazardous", "⚫"

def activity_advice(temp, rain_mm, aqi_value, activity):
    advice = []
    if activity != "None":
        if rain_mm > 1: advice.append("🌂 Carry umbrella, might rain.")
        if temp < 18: advice.append("🧥 Wear warm clothes.")
        if temp > 32: advice.append("🧢 Stay hydrated & wear light clothes.")
        if aqi_value > 150: advice.append("😷 High pollution, limit outdoor activity.")
        if activity in ["Jogging", "Cycling", "Walking"]:
            if rain_mm > 1: advice.append("⚠️ Be careful on wet surfaces.")
            if temp < 10: advice.append("❄️ Consider shorter duration.")
    if not advice: advice.append("✅ Weather looks good for your activity!")
    return " ".join(advice)

def fetch_weather_from_gemini(city):
    prompt = f"""
    Give me the current weather for {city} in India in JSON format with:
    {{
      "temp": <temperature in °C>,
      "humidity": <humidity in %>,
      "wind": <wind speed in km/h>,
      "rain_probability": <chance of rain in %>
    }}
    Respond with JSON only.
    """
    response = model.generate_content(prompt)
    text = response.text.strip()
    try:
        data = json.loads(text[text.find("{"):text.rfind("}")+1])
        return data
    except:
        st.error("⚠️ Could not parse Gemini weather data. Try again.")
        return None

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="Weather + AQI + Gemini", layout="wide")
st.title("🌦 Weather + AQI Forecast (via Gemini API)")

cities = ["Mumbai","Delhi","Bangalore","Kolkata","Chennai","Hyderabad","Pune","Ahmedabad"]
with st.form("weather_form"):
    city = st.selectbox("Select your city:", cities)
    activity = st.selectbox("Select your planned activity:", ["None","Jogging","Cycling","Walking","Outdoor Work","Picnic"])
    submitted = st.form_submit_button("Get Forecast & Advice")

if submitted:
    weather_data = fetch_weather_from_gemini(city)
    if weather_data:
        today_temp = weather_data["temp"]
        today_humidity = weather_data["humidity"]
        today_wind = weather_data["wind"]
        rain_prob = weather_data["rain_probability"]

        today_aqi_cat = aqi_model.predict([[today_temp, today_humidity, today_wind]])[0]
        today_aqi_value = [80,150,250][today_aqi_cat]
        category, badge = get_air_quality_category(today_aqi_value)

        tab1, tab2, tab3 = st.tabs(["📍 Today", "📅 Tomorrow (ML)", "📈 5-Day Forecast"])

        with tab1:
            st.info(f"🌡 Temp: {today_temp}°C | 💧 Humidity: {today_humidity}% | 🌬 Wind: {today_wind} km/h")
            st.success(f"🌍 AQI: {badge} {category} — {today_aqi_value}/500")
            st.progress(int(rain_prob))
            st.caption(f"🌧 Rain Probability: {rain_prob}%")

        tomorrow_day = pd.Timestamp.now().day + 1
        predicted_rain_mm = df['rain_mm'].mean()
        tomorrow_features = pd.DataFrame([[tomorrow_day, today_humidity, today_wind, predicted_rain_mm]],
                                         columns=['day','humidity','wind','rain_mm'])
        predicted_temp = temp_model.predict(tomorrow_features)[0]
        aqi_features = pd.DataFrame([[predicted_temp, today_humidity, today_wind]],
                                    columns=['temp','humidity','wind'])
        predicted_aqi_cat = aqi_model.predict(aqi_features)[0]
        predicted_aqi_value = [80,150,250][predicted_aqi_cat]
        pred_category, pred_badge = get_air_quality_category(predicted_aqi_value)

        with tab2:
            st.info(f"🌡 Predicted Temp: {predicted_temp:.2f}°C")
            st.success(f"🌍 Predicted AQI: {pred_badge} {pred_category} — {predicted_aqi_value}/500")
            st.warning(f"🌧 Predicted Rain: {predicted_rain_mm:.1f} mm")
            st.info(activity_advice(predicted_temp, predicted_rain_mm, predicted_aqi_value, activity))

        forecast_days = 5
        forecast_dates = [datetime.date.today() + datetime.timedelta(days=i) for i in range(1, forecast_days+1)]
        forecast_temps, forecast_aqi = [], []

        for i, d in enumerate(forecast_dates):
            hum = today_humidity + np.random.normal(0, 2)
            w = today_wind + np.random.normal(0, 1)
            rain = predicted_rain_mm + np.random.normal(0, 0.5)
            features = pd.DataFrame([[d.day, hum, w, rain]], columns=['day','humidity','wind','rain_mm'])
            temp_pred = temp_model.predict(features)[0] + np.random.normal(0, 1)
            forecast_temps.append(temp_pred)

            aqi_cat = aqi_model.predict(pd.DataFrame([[temp_pred, hum, w]], columns=['temp','humidity','wind']))[0]
            aqi_pred = [80,150,250][aqi_cat] + np.random.randint(-10, 10)
            forecast_aqi.append(max(0, aqi_pred))

        forecast_df = pd.DataFrame({"Date": forecast_dates, "Temp": forecast_temps, "AQI": forecast_aqi})

        with tab3:
            col_temp, col_aqi = st.columns(2)
            with col_temp:
                st.markdown("#### Temperature Forecast (°C)")
                temp_chart = alt.Chart(forecast_df).mark_line(point=True).encode(
                    x='Date:T', y='Temp:Q', tooltip=['Date','Temp']
                )
                st.altair_chart(temp_chart, use_container_width=True)

            with col_aqi:
                st.markdown("#### AQI Forecast")
                aqi_chart = alt.Chart(forecast_df).mark_bar().encode(
                    x='Date:T', y='AQI:Q', tooltip=['Date','AQI']
                )
                st.altair_chart(aqi_chart, use_container_width=True)

            st.markdown("**AQI Legend:** 🟢 Good | 🟡 Moderate | 🟠 Unhealthy (Sensitive) | 🔴 Unhealthy | 🟣 Very Unhealthy | ⚫ Hazardous")








