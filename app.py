import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
import datetime
import requests

# ----------------------------
# Tier 1 & Tier 2 Cities (India)
# ----------------------------
cities = {
    "Bengaluru": (12.97, 77.59),
    "Delhi": (28.61, 77.21),
    "Mumbai": (19.07, 72.87),
    "Chennai": (13.08, 80.27),
    "Hyderabad": (17.38, 78.47),
    "Kolkata": (22.57, 88.36),
    "Ahmedabad": (23.03, 72.58),
    "Pune": (18.52, 73.85),
    # Tier 2 Cities
    "Jaipur": (26.91, 75.79),
    "Lucknow": (26.85, 80.95),
    "Chandigarh": (30.74, 76.79),
    "Indore": (22.72, 75.85),
    "Coimbatore": (11.01, 76.96),
    "Nagpur": (21.15, 79.09),
    "Visakhapatnam": (17.70, 83.30),
    "Bhopal": (23.25, 77.41),
    "Vadodara": (22.30, 73.20),
    "Surat": (21.17, 72.83),
}

# ----------------------------
# Generate synthetic historical data for ML training
# ----------------------------
np.random.seed(42)
data_rows = []

for city_name, (lat, lon) in cities.items():
    for day in range(1, 366):  # 1 year
        month = (day // 30) + 1
        temp = np.random.normal(loc=25 + 5*np.sin((month/12)*2*np.pi) - lat*0.05, scale=4)
        humidity = np.random.normal(loc=70 - lat*0.1, scale=10)
        wind = np.random.normal(loc=10, scale=3)
        rain_mm = max(0, np.random.normal(loc=2, scale=3))
        base_aqi = 50 + (temp-25)*2 + (humidity-70)*0.5 - wind*1.5 + rain_mm*2
        noise = np.random.normal(0,15)
        aqi = min(max(int(base_aqi+noise),10),300)
        aqi_category = 0 if aqi<=100 else (1 if aqi<=200 else 2)
        data_rows.append([city_name, lat, lon, day, month, temp, humidity, wind, rain_mm, aqi, aqi_category])

df = pd.DataFrame(data_rows, columns=[
    "city","lat","lon","day","month","temp","humidity","wind","rain_mm","aqi","aqi_category"
])

# ----------------------------
# Train ML models
# ----------------------------
X_temp = df[['lat','lon','day','month','humidity','wind','rain_mm']]
y_temp = df['temp']
temp_model = LinearRegression()
temp_model.fit(X_temp, y_temp)

X_aqi = df[['temp','humidity','wind']]
y_aqi = df['aqi_category']
aqi_model = RandomForestClassifier(n_estimators=200, random_state=42)
aqi_model.fit(X_aqi, y_aqi)

# ----------------------------
# Helper functions
# ----------------------------
def get_air_quality_category(aqi_value):
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

def activity_advice(temp, rain_mm, aqi_value, activity):
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
    if activity in ["Jogging","Cycling","Walking"]:
        if rain_mm>1:
            advice.append("⚠️ Be careful on wet surfaces.")
        if temp<10:
            advice.append("❄️ Consider shorter duration for outdoor activity.")
    if not advice:
        advice.append("✅ Weather looks good for your activity!")
    return " ".join(advice)

def rain_probability(city, df):
    city_data = df[df['city']==city]
    rain_days = city_data[city_data['rain_mm']>0].shape[0]
    total_days = city_data.shape[0]
    return round((rain_days/total_days)*100,1)

def fetch_today_weather(city, api_key):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    data = requests.get(url).json()
    if "main" in data:
        today_temp = data["main"]["temp"]
        today_humidity = data["main"]["humidity"]
        today_wind = data["wind"]["speed"]
        lat, lon = data["coord"]["lat"], data["coord"]["lon"]
        return today_temp, today_humidity, today_wind, lat, lon
    else:
        return None

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🌦 ML Weather + AQI + Activity Advisory (India)")

with st.form("weather_form"):
    city_list = list(cities.keys())
    city = st.selectbox("Select your city:", city_list)
    activity = st.selectbox("Select your planned activity:",
                            ["None","Jogging","Cycling","Walking","Outdoor Work","Picnic"])
    submitted = st.form_submit_button("Get Forecast & Advice")

if submitted:
    api_key = "YOUR_OPENWEATHERMAP_API_KEY"  # replace with your key
    today_weather = fetch_today_weather(city, api_key)
    
    if today_weather:
        today_temp, today_humidity, today_wind, lat, lon = today_weather
        
        # Predict tomorrow
        today_date = datetime.datetime.now()
        tomorrow_day = today_date.timetuple().tm_yday + 1
        tomorrow_month = today_date.month if today_date.day<28 else today_date.month+1
        predicted_rain_mm = df[df['city']==city]['rain_mm'].mean()
        
        temp_features = pd.DataFrame([[lat, lon, tomorrow_day, tomorrow_month, today_humidity, today_wind, predicted_rain_mm]],
                                     columns=['lat','lon','day','month','humidity','wind','rain_mm'])
        predicted_temp = temp_model.predict(temp_features)[0]
        
        aqi_features = pd.DataFrame([[predicted_temp, today_humidity, today_wind]], columns=['temp','humidity','wind'])
        predicted_aqi_cat = aqi_model.predict(aqi_features)[0]
        predicted_aqi_value = [80,150,250][predicted_aqi_cat]
        pred_category, pred_badge = get_air_quality_category(predicted_aqi_value)
        
        st.subheader(f"📅 Tomorrow's Forecast in {city}")
        st.write(f"🌡 Predicted Temperature: {predicted_temp:.1f}°C")
        st.write(f"🌍 Predicted AQI: {pred_badge} {pred_category} — {predicted_aqi_value}/500")
        st.write(f"🌧 Predicted Rain (mm): {predicted_rain_mm:.1f}")
        st.write(f"🌧 Historical Rain Probability: {rain_probability(city, df)}%")
        
        advice = activity_advice(predicted_temp, predicted_rain_mm, predicted_aqi_value, activity)
        st.subheader("💡 Activity Advisory")
        st.write(advice)
    else:
        st.error("❌ Could not fetch today's weather. Check your API key or city name.")


           

