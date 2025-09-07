import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import requests
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
import datetime

# ----------------------------
# Generate synthetic historical data for ML model
# ----------------------------
np.random.seed(42)
days = 30
hours_per_day = 24
total_rows = days * hours_per_day

dates = [datetime.datetime.now() - datetime.timedelta(days=i//24, hours=i%24) for i in range(total_rows)]
dates = [d.strftime("%Y-%m-%d %H:%M:%S") for d in dates]

temp = np.random.normal(loc=30, scale=3, size=total_rows)
humidity = np.random.normal(loc=80, scale=5, size=total_rows)
wind = np.random.normal(loc=10, scale=2, size=total_rows)
rain_mm = np.random.choice([0,0.5,1,2,5,10], size=total_rows, p=[0.5,0.2,0.1,0.1,0.05,0.05])

aqi = []
for t, h, w, r in zip(temp, humidity, wind, rain_mm):
    base = 50 + (t-25)*2 + (h-70)*0.5 - w*1.5 + r*2
    noise = np.random.normal(0,10)
    aqi_value = min(max(int(base+noise),10),300)
    aqi.append(aqi_value)

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

# ----------------------------
# Train ML models
# ----------------------------
X_temp = df[['day','humidity','wind','rain_mm']]
y_temp = df['temp']
temp_model = LinearRegression()
temp_model.fit(X_temp, y_temp)

X_aqi = df[['temp','humidity','wind']]
y_aqi = df['aqi_category']
aqi_model = RandomForestClassifier(n_estimators=200, random_state=42)
aqi_model.fit(X_aqi, y_aqi)

# ----------------------------
# Helper Functions
# ----------------------------
def get_air_quality_category(aqi_value: int) -> tuple:
    if aqi_value <= 50:
        return "Good", "🟢"
    elif aqi_value <= 100:
        return "Moderate", "🟡"
    elif aqi_value <= 150:
        return "Unhealthy for Sensitive Groups", "🟠"
    elif aqi_value <= 200:
        return "Unhealthy", "🔴"
    elif aqi_value <= 300:
        return "Very Unhealthy", "🟣"
    else:
        return "Hazardous", "⚫"

def activity_advice(temp: float, rain_mm: float, aqi_value: int, activity: str) -> str:
    advice = []
    if activity == "None":
        return "No specific advice for now."
    if rain_mm > 1:
        advice.append("🌂 Carry an umbrella, it might rain.")
    if temp < 18:
        advice.append("🧥 Wear a sweater or warm clothes, it's cold.")
    if temp > 32:
        advice.append("🧢 Stay hydrated and wear light clothing, it's hot.")
    if aqi_value > 150:
        advice.append("😷 High pollution! Consider wearing a mask or limiting outdoor activity.")
    if activity in ["Jogging", "Cycling", "Walking"]:
        if rain_mm > 1:
            advice.append("⚠️ Be careful on wet surfaces.")
        if temp < 10:
            advice.append("❄️ Consider shorter duration for outdoor activity.")
    if not advice:
        advice.append("✅ Weather looks good for your activity!")
    return " ".join(advice)

@st.cache_data(ttl=600)
def fetch_weather(city: str, api_key: str) -> dict:
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    return requests.get(url).json()

@st.cache_data(ttl=600)
def fetch_air_quality(lat: float, lon: float, api_key: str) -> int:
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}"
    data = requests.get(url).json()
    if "list" in data and data["list"]:
        aqi_code = data["list"][0]["main"]["aqi"]
        mapping = {1:30,2:80,3:120,4:180,5:300}
        return mapping.get(aqi_code,0)
    return 0

@st.cache_data(ttl=600)
def fetch_rain_probability(city: str, api_key: str) -> float:
    url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={api_key}&units=metric"
    data = requests.get(url).json()
    if "list" not in data:
        return 0
    tomorrow = datetime.date.today() + datetime.timedelta(days=1)
    forecast_list = [item for item in data["list"] if datetime.datetime.fromtimestamp(item["dt"]).date() == tomorrow]
    if not forecast_list:
        return 0
    rain_hours = sum(1 for item in forecast_list if "rain" in item and item["rain"].get("3h",0) > 0)
    probability = round((rain_hours / len(forecast_list)) * 100, 1)
    return probability

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Weather + AQI App", layout="wide")
st.title("🌦 Weather + AQI + Activity Advisory")

tier1_cities = ["Mumbai", "Delhi", "Bangalore", "Kolkata", "Chennai", "Hyderabad", "Pune", "Ahmedabad"]
tier2_cities = ["Surat", "Jaipur", "Lucknow", "Kanpur", "Nagpur", "Indore", "Thane", "Bhopal", "Patna", "Vadodara"]
tier1_cities_display = [f"⭐ {city}" for city in tier1_cities]
all_cities_display = tier1_cities_display + tier2_cities

with st.form("weather_form"):
    city_selected = st.selectbox("Select your city:", all_cities_display)
    city = city_selected.replace("⭐ ", "")
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

        # Tabs for Today and Tomorrow
        tab1, tab2 = st.tabs(["📍 Today's Weather", "📅 Tomorrow's Forecast (ML)"])

        with tab1:
            st.info(f"🌡 Temperature: {today_temp}°C  | 💧 Humidity: {today_humidity}%  | 🌬 Wind: {today_wind} km/h")
            st.success(f"🌍 AQI: {badge} {category} — {aqi_value}/500")
            st.progress(int(rain_prob))
            st.caption(f"🌧 Rain Probability Tomorrow: {rain_prob}%")

            st.markdown("#### 📝 AQI Legend")
            aqi_legend = {
                "Good": "🟢 0-50",
                "Moderate": "🟡 51-100",
                "Unhealthy for Sensitive Groups": "🟠 101-150",
                "Unhealthy": "🔴 151-200",
                "Very Unhealthy": "🟣 201-300",
                "Hazardous": "⚫ 301+"
            }
            for key, val in aqi_legend.items():
                st.markdown(f"<span style='display:block; margin:2px 0; font-size:14px;'>{val}</span>", unsafe_allow_html=True)

        with tab2:
            tomorrow_day = pd.Timestamp.now().day + 1
            predicted_rain_mm = df['rain_mm'].mean()
            tomorrow_features = pd.DataFrame([[tomorrow_day, today_humidity, today_wind, predicted_rain_mm]],
                                             columns=['day','humidity','wind','rain_mm'])
            predicted_temp = temp_model.predict(tomorrow_features)[0]

            aqi_features = pd.DataFrame([[predicted_temp, today_humidity, today_wind]], columns=['temp','humidity','wind'])
            predicted_aqi_cat = aqi_model.predict(aqi_features)[0]
            predicted_aqi_value = [80,150,250][predicted_aqi_cat]
            pred_category, pred_badge = get_air_quality_category(predicted_aqi_value)

            st.info(f"🌡 Predicted Temperature: {predicted_temp:.2f}°C")
            st.success(f"🌍 Predicted AQI: {pred_badge} {pred_category} — {predicted_aqi_value}/500")
            st.warning(f"🌧 Predicted Rain (mm): {predicted_rain_mm:.1f}")

            with st.expander("💡 Activity Advisory for Tomorrow"):
                advice_text = activity_advice(predicted_temp, predicted_rain_mm, predicted_aqi_value, activity)
                st.info(advice_text)

    else:
        st.error("❌ Could not fetch weather data. Check your API key or city name.")


           




