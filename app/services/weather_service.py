import os
import json
import logging
import requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

# ─────────────────────────────────────────
# BUILT-IN INDIA CITY/STATE COORDINATES
# ─────────────────────────────────────────
_INDIA_COORDS = {
    "chennai":(13.0827,80.2707),"mumbai":(19.0760,72.8777),"delhi":(28.6139,77.2090),
    "new delhi":(28.6139,77.2090),"bangalore":(12.9716,77.5946),"bengaluru":(12.9716,77.5946),
    "hyderabad":(17.3850,78.4867),"coimbatore":(11.0168,76.9558),"kolkata":(22.5726,88.3639),
    "pune":(18.5204,73.8567),"ahmedabad":(23.0225,72.5714),"jaipur":(26.9124,75.7873),
    "lucknow":(26.8467,80.9462),"chandigarh":(30.7333,76.7794),"bhopal":(23.2599,77.4126),
    "patna":(25.5941,85.1376),"bhubaneswar":(20.2961,85.8245),"guwahati":(26.1445,91.7362),
    "thiruvananthapuram":(8.5241,76.9366),"trivandrum":(8.5241,76.9366),
    "visakhapatnam":(17.6868,83.2185),"vizag":(17.6868,83.2185),
    "nagpur":(21.1458,79.0882),"amritsar":(31.6340,74.8723),"surat":(21.1702,72.8311),
    "kochi":(9.9312,76.2673),"indore":(22.7196,75.8577),"madurai":(9.9252,78.1198),
    "salem":(11.6643,78.1460),"tiruppur":(11.1085,77.3411),"vellore":(12.9165,79.1325),
    "vadodara":(22.3072,73.1812),"rajkot":(22.3039,70.8022),"agra":(27.1767,78.0081),
    "varanasi":(25.3176,82.9739),"ludhiana":(30.9010,75.8573),
    # States
    "tamil nadu":(11.1271,78.6569),"maharashtra":(19.7515,75.7139),
    "karnataka":(15.3173,75.7139),"kerala":(10.8505,76.2711),
    "andhra pradesh":(15.9129,79.7400),"telangana":(18.1124,79.0193),
    "gujarat":(22.2587,71.1924),"rajasthan":(27.0238,74.2179),
    "punjab":(31.1471,75.3412),"haryana":(29.0588,76.0856),
    "uttar pradesh":(26.8467,80.9462),"madhya pradesh":(22.9734,78.6569),
    "west bengal":(22.9868,87.8550),"odisha":(20.9517,85.0985),
    "assam":(26.2006,92.9376),"bihar":(25.0961,85.3131),
    "jharkhand":(23.6102,85.2799),"chhattisgarh":(21.2787,81.8661),
    "himachal pradesh":(31.1048,77.1734),"uttarakhand":(30.0668,79.0193),"goa":(15.2993,74.1240),
}

# Wind direction → numeric (matches training data encoding)
_WIND_DIR_MAP = {
    "N":0,"NNE":1,"NE":2,"ENE":3,"E":4,"ESE":5,"SE":6,"SSE":7,
    "S":8,"SSW":9,"SW":10,"WSW":11,"W":12,"WNW":13,"NW":14,"NNW":15,
}


def get_coordinates_for_location(location: str):
    key = location.strip().lower()
    if key in _INDIA_COORDS:
        return _INDIA_COORDS[key]
    for city_key, coords in _INDIA_COORDS.items():
        if city_key in key or key in city_key:
            return coords
    if not OPENWEATHER_API_KEY:
        return None
    try:
        url = f"http://api.openweathermap.org/geo/1.0/direct?q={location},IN&limit=1&appid={OPENWEATHER_API_KEY}"
        r = requests.get(url, timeout=8); r.raise_for_status()
        data = r.json()
        if data:
            return (data[0]["lat"], data[0]["lon"])
    except Exception as e:
        logger.warning("Geocoding failed for '%s': %s", location, e)
    return None


def get_weather_forecast(lat: float, lon: float) -> dict:
    """
    Fetches 5-day/3h forecast. Returns 5 slots with ALL fields
    needed by the improved 16-feature rainfall model.
    """
    if not OPENWEATHER_API_KEY:
        return {"error": "OPENWEATHER_API_KEY not set in .env"}

    url = (f"https://api.openweathermap.org/data/2.5/forecast"
           f"?lat={lat}&lon={lon}&units=metric&appid={OPENWEATHER_API_KEY}")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.error("Weather API failed: %s", e)
        return {"error": f"Weather fetch failed: {e}"}

    data = response.json()
    forecasts = []

    for entry in data["list"][:5]:
        main    = entry["main"]
        wind    = entry["wind"]
        clouds  = entry["clouds"]
        wdir_str = _deg_to_cardinal(wind.get("deg", 180))
        wdir_enc = _WIND_DIR_MAP.get(wdir_str, 8)

        temp        = main["temp"]
        humidity    = main["humidity"]
        feels_like  = main.get("feels_like", temp)
        visibility  = entry.get("visibility", 10000) / 1000   # m → km
        gust_kph    = round(wind.get("gust", wind["speed"]) * 3.6, 2)
        wind_kph    = round(wind["speed"] * 3.6, 2)
        wind_deg    = wind.get("deg", 180)
        cloud       = clouds["all"]
        pressure_mb = main["pressure"]

        # Engineered features (must match training)
        dew_point   = temp - ((100 - humidity) / 5)
        feels_delta = feels_like - temp
        gust_ratio  = gust_kph / (wind_kph + 0.1)
        vis_inv     = 1 / (visibility + 0.1)
        humid_cloud = humidity * cloud / 100

        forecasts.append({
            # Core
            "timestamp":   entry["dt_txt"],
            "temp":        temp,
            "humidity":    humidity,
            "pressure_mb": pressure_mb,
            "cloud":       cloud,
            "wind_kph":    wind_kph,
            "rain_3h":     entry.get("rain", {}).get("3h", 0.0),
            # New fields for improved model
            "gust_kph":          gust_kph,
            "visibility_km":     visibility,
            "uv_index":          0,           # not in free forecast API
            "wind_degree":       wind_deg,
            "feels_like_celsius":feels_like,
            "wind_dir_enc":      wdir_enc,
            # Engineered
            "dew_point":   round(dew_point,   2),
            "feels_delta": round(feels_delta, 2),
            "gust_ratio":  round(gust_ratio,  4),
            "vis_inv":     round(vis_inv,      4),
            "humid_cloud": round(humid_cloud,  2),
        })

    return {
        "city":     data["city"]["name"],
        "country":  data["city"]["country"],
        "lat":      lat,
        "lon":      lon,
        "forecast": forecasts,
    }


def get_weather_for_location(location: str) -> dict:
    coords = get_coordinates_for_location(location)
    if coords is None:
        return {"error": f"Could not find coordinates for '{location}'"}
    lat, lon = coords
    result = get_weather_forecast(lat, lon)
    result["resolved_location"] = location
    return result


def _deg_to_cardinal(deg: float) -> str:
    dirs = ["N","NNE","NE","ENE","E","ESE","SE","SSE",
            "S","SSW","SW","WSW","W","WNW","NW","NNW"]
    idx = round(deg / 22.5) % 16
    return dirs[idx]