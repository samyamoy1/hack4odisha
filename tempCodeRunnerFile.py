import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

API_KEY = "332c7aeda1d896aa5c4ce26b89c28096"  # Replace with your real OpenWeatherMap API key


# -------------------
# 1. Get Current Weather
# -------------------
def get_weather(city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        weather = {
            "temperature": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "description": data["weather"][0]["description"],
            "wind_speed": round(data["wind"]["speed"] * 3.6, 2),  # convert m/s to km/h
            "lat": data["coord"]["lat"],
            "lon": data["coord"]["lon"]
        }
        return weather
    return None


# -------------------
# 2. Get Forecast (rain chance in next 3h)
# -------------------
def get_rain_chance(city):
    url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if "list" in data and len(data["list"]) > 0:
            rain_chance = data["list"][0].get("pop", 0) * 100  # 'pop' = probability of precipitation
            return round(rain_chance, 2)
    return None


# -------------------
# 3. Get 7-day Forecast
# -------------------
def get_weekly_forecast(lat, lon):
    url = f"https://api.openweathermap.org/data/2.5/onecall?lat={lat}&lon={lon}&exclude=current,minutely,hourly,alerts&appid={API_KEY}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        forecast = []
        for day in data["daily"][:7]:  # next 7 days
            forecast.append({
                "date": day["dt"],  # Unix timestamp
                "min_temp": day["temp"]["min"],
                "max_temp": day["temp"]["max"],
                "description": day["weather"][0]["description"],
                "rain_chance": round(day.get("pop", 0) * 100, 2)
            })
        return forecast
    return None


# -------------------
# 4. Get Air Quality
# -------------------
def get_air_quality(lat, lon):
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if "list" in data and len(data["list"]) > 0:
            aqi = data["list"][0]["main"]["aqi"]
            aqi_map = {1: 25, 2: 50, 3: 100, 4: 150, 5: 200}
            return aqi_map.get(aqi, 100)
    return None


# -------------------
# 5. Personalized Advice
# -------------------
def personalized_advice(weather, activity):
    advice = ""

    if "rain" in weather["description"].lower():
        advice += "Carry an umbrella. "
    if weather["temperature"] > 35:
        advice += "Stay hydrated and avoid outdoor activities. "
    if weather["temperature"] < 10:
        advice += "Wear warm clothes. "

    if activity.lower() in ["jogging", "running"]:
        if weather["temperature"] > 30:
            advice += "Best to jog early morning or late evening. "
        elif "rain" in weather["description"].lower():
            advice += "Better to skip jogging today. "
    elif activity.lower() == "cycling":
        if weather["wind_speed"] > 20:
            advice += "Windy day, cycling may be tough. "
    elif activity.lower() == "gardening":
        if "rain" in weather["description"].lower():
            advice += "Rainy conditions, not ideal for gardening. "

    if not advice:
        advice = "Weather looks good for your activity."

    return advice


# -------------------
# 6. Flask Route
# -------------------
@app.route("/get_advice", methods=["GET"])
def get_advice():
    city = request.args.get("city")
    activity = request.args.get("activity", "general")

    if not city:
        return jsonify({"error": "Please provide city"}), 400

    weather = get_weather(city)
    if not weather:
        return jsonify({"error": "Could not fetch weather data"}), 500

    rain_chance = get_rain_chance(city)
    air_quality = get_air_quality(weather["lat"], weather["lon"])
    weekly_forecast = get_weekly_forecast(weather["lat"], weather["lon"])
    advice = personalized_advice(weather, activity)

    return jsonify({
        "city": city,
        "temperature": weather["temperature"],
        "humidity": weather["humidity"],
        "description": weather["description"],
        "wind_speed": weather["wind_speed"],
        "rain_chance": rain_chance,
        "air_quality": air_quality,
        "activity": activity,
        "advice": advice,
        "weekly_forecast": weekly_forecast
    })


if __name__ == "__main__":
    app.run(debug=True)
