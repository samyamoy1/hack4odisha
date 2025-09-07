import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import requests
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
import datetime
import altair as alt

# -------------------- Generate synthetic data --------------------
np.random.seed(42)
days, hours_per_day = 30, 24
total_rows = days * hours_per_day

dates = [datetime.datetime.now() - datetime.timedelta(days=i//24, hours=i%24) for i in range(total_rows)]
dates = [d.strftime("%Y-%m-%d %H:%M:%S") for d in dates]

temp = np.random.normal(30, 3, total_rows)
humidity = np.random.normal(80, 5, total_rows)
wind = np.random.normal(10, 2, total_rows)
rain_mm = np.random.choice([0,0.5,1,2,5,10], total_rows, p=[0.5,0.2,0.1,0.1,0.05,0.05])

aqi = []
for t, h, w, r in zip(temp, humidity, wind, rain_mm):
    base = 50 + (t-25)*2 + (h-70)*0.5 - w*1.5 + r*2
    noise = np.random.normal(0,10)
    aqi.append(min(max(int(base+noise),10),300))

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
temp_model.fit(df[['day','humidity','wind','rain_mm']], df['temp'])

aqi_model = RandomForestClassifier(n_estimators=200, random_state=42)
aqi_model.fit(df[['temp','humidity','wind']], df['aqi_category'])

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
        if rain_mm>1: advice.append("🌂 Carry umbrella, might rain.")
        if temp<18: advice.append("🧥 Wear warm clothes.")
        if temp>32: advice.append("🧢 Stay hydrated & wear light clothes.")
        if aqi_value>150: advice.append("😷 High pollution, limit outdoor activity.")
        if activity in ["Jogging","Cycling","Walking"]:
            if rain_mm>1: advice.append("⚠️ Be careful on wet surfaces.")
            if temp<10: advice.append("❄️ Consider shorter duration.")
    if not advice: advice.append("✅ Weather looks good for your activity!")
    return " ".join(advice)

@st.cache_data(ttl=600)
def fetch_weather(city, api_key):
    return requests.get(f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric").json()

@st.cache_data(ttl=600)
def fetch_air_quality(lat, lon, api_key):
    data = requests.get(f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}").json()
    if "list" in data and data["list"]:
        mapping = {1:30,2:80,3:120,4:180,5:300}
        return mapping.get(data["list"][0]["main"]["aqi"],0)
    return 0

@st.cache_data(ttl=600)
def fetch_rain_probability(city, api_key):
    data = requests.get(f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={api_key}&units=metric").json()
    tomorrow = datetime.date.today() + datetime.timedelta(days=1)
    forecast_list = [item for item in data.get("list",[]) if datetime.datetime.fromtimestamp(item["dt"]).date() == tomorrow]
    if not forecast_list: return 0
    rain_hours = sum(1 for item in forecast_list if "rain" in item and item["rain"].get("3h",0)>0)
    return round((rain_hours/len(forecast_list))*100,1)

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="Weather + AQI App", layout="wide")
st.title("🌦 Weather + AQI + Activity Advisory")

tier1_cities = ["Mumbai","Delhi","Bangalore","Kolkata","Chennai","Hyderabad","Pune","Ahmedabad"]
tier2_cities = ["Surat","Jaipur","Lucknow","Kanpur","Nagpur","Indore","Thane","Bhopal","Patna","Vadodara"]
all_cities_display = [f"⭐ {c}" for c in tier1_cities] + tier2_cities

with st.form("weather_form"):
    city_selected = st.selectbox("Select your city:", all_cities_display)
    city = city_selected.replace("⭐ ","")
    activity = st.selectbox("Select your planned activity:", ["None","Jogging","Cycling","Walking","Outdoor Work","Picnic"])
    submitted = st.form_submit_button("Get Forecast & Advice")

if submitted:
    api_key = "332c7aeda1d896aa5c4ce26b89c28096"
    weather_data = fetch_weather(city, api_key)

    if weather_data.get("main"):
        today_temp = weather_data["main"]["temp"]
        today_humidity = weather_data["main"]["humidity"]
        today_wind = weather_data["wind"]["speed"]
        lat, lon = weather_data["coord"]["lat"], weather_data["coord"]["lon"]
        aqi_value = fetch_air_quality(lat, lon, api_key)
        category, badge = get_air_quality_category(aqi_value)
        rain_prob = fetch_rain_probability(city, api_key)

        tab1, tab2, tab3 = st.tabs(["📍 Today", "📅 Tomorrow (ML)", "📈 5-Day Forecast"])

        # Today
        with tab1:
            st.info(f"🌡 Temp: {today_temp}°C | 💧 Humidity: {today_humidity}% | 🌬 Wind: {today_wind} km/h")
            st.success(f"🌍 AQI: {badge} {category} — {aqi_value}/500")
            st.progress(int(rain_prob))
            st.caption(f"🌧 Rain Probability Tomorrow: {rain_prob}%")

        # Tomorrow (ML)
        tomorrow_day = pd.Timestamp.now().day + 1
        predicted_rain_mm = df['rain_mm'].mean()
        tomorrow_features = pd.DataFrame([[tomorrow_day,today_humidity,today_wind,predicted_rain_mm]], columns=['day','humidity','wind','rain_mm'])
        predicted_temp = temp_model.predict(tomorrow_features)[0]
        aqi_features = pd.DataFrame([[predicted_temp,today_humidity,today_wind]], columns=['temp','humidity','wind'])
        predicted_aqi_cat = aqi_model.predict(aqi_features)[0]
        predicted_aqi_value = [80,150,250][predicted_aqi_cat]
        pred_category, pred_badge = get_air_quality_category(predicted_aqi_value)

        with tab2:
            st.info(f"🌡 Predicted Temp: {predicted_temp:.2f}°C")
            st.success(f"🌍 Predicted AQI: {pred_badge} {pred_category} — {predicted_aqi_value}/500")
            st.warning(f"🌧 Predicted Rain (mm): {predicted_rain_mm:.1f}")
            st.info(activity_advice(predicted_temp, predicted_rain_mm, predicted_aqi_value, activity))

        # 5-Day Forecast
        forecast_days = 5
        forecast_dates = [datetime.date.today() + datetime.timedelta(days=i) for i in range(1, forecast_days+1)]
        forecast_temps, forecast_aqi = [], []

        temp_trend = np.linspace(0, np.random.uniform(-2,2), forecast_days)
        aqi_trend = np.linspace(0, np.random.randint(-15,15), forecast_days)

        for i, d in enumerate(forecast_dates):
            hum = today_humidity + np.random.normal(0, 2)
            w = today_wind + np.random.normal(0, 1)
            rain = predicted_rain_mm + np.random.normal(0, 0.5)

            features = pd.DataFrame([[d.day, hum, w, rain]], columns=['day','humidity','wind','rain_mm'])
            temp_pred = temp_model.predict(features)[0] + np.random.normal(0,1) + temp_trend[i]
            forecast_temps.append(temp_pred)

            aqi_cat = aqi_model.predict(pd.DataFrame([[temp_pred, hum, w]], columns=['temp','humidity','wind']))[0]
            aqi_pred = [80,150,250][aqi_cat] + np.random.randint(-10,10) + aqi_trend[i]
            forecast_aqi.append(max(0, aqi_pred))

        forecast_df = pd.DataFrame({"Date": forecast_dates, "Temp": forecast_temps, "AQI": forecast_aqi})

        def map_aqi_category(value):
            if value <= 50: return "Good", "#00ff00"
            elif value <= 100: return "Moderate", "#ffff00"
            elif value <= 150: return "Unhealthy for Sensitive Groups", "#ff8000"
            elif value <= 200: return "Unhealthy", "#ff0000"
            elif value <= 300: return "Very Unhealthy", "#800080"
            else: return "Hazardous", "#000000"

        forecast_df['AQI_Category'] = forecast_df['AQI'].apply(lambda x: map_aqi_category(x)[0])
        forecast_df['AQI_Color'] = forecast_df['AQI'].apply(lambda x: map_aqi_category(x)[1])

        with tab3:
            col_temp, col_aqi = st.columns(2)
            with col_temp:
                st.markdown("#### Temperature Forecast (°C)")
                temp_chart = alt.Chart(forecast_df).mark_line(point=True, interpolate='monotone').encode(
                    x=alt.X('Date:T'), y=alt.Y('Temp:Q'), tooltip=['Date','Temp']
                ).interactive()
                st.altair_chart(temp_chart, use_container_width=True)

            with col_aqi:
                st.markdown("#### AQI Forecast")
                aqi_chart = alt.Chart(forecast_df).mark_bar().encode(
                    x=alt.X('Date:T'), y=alt.Y('AQI:Q'),
                    color=alt.Color('AQI_Color:N', scale=None, legend=None),
                    tooltip=['Date','AQI','AQI_Category']
                ).interactive()
                st.altair_chart(aqi_chart, use_container_width=True)

            st.markdown("**AQI Legend:** 🟢 Good | 🟡 Moderate | 🟠 Unhealthy for Sensitive Groups | 🔴 Unhealthy | 🟣 Very Unhealthy | ⚫ Hazardous")
    else:
        st.error("❌ Could not fetch weather data. Check API key or city name.")



           






