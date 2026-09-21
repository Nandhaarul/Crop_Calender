import json
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database import CalendarRecord, get_db
from app.services.ml_service import (
    predict_fertilizer,
    predict_rainfall,
    predict_yield,
)
from app.services.weather_service import (
    get_weather_for_location,
    get_weather_forecast,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/insights", tags=["Insights"])


class InsightsRequest(BaseModel):
    calendar_id: int

    nitrogen_level: Optional[float] = Field(default=None, ge=0, le=140)
    phosphorus_level: Optional[float] = Field(default=None, ge=0, le=145)
    potassium_level: Optional[float] = Field(default=None, ge=0, le=205)
    soil_ph: Optional[float] = Field(default=None, ge=3.5, le=9.9)
    soil_moisture: Optional[float] = Field(default=None, ge=0, le=100)
    soil_type: Optional[str] = Field(default=None)
    organic_carbon: Optional[float] = Field(default=None, ge=0)
    electrical_conductivity: Optional[float] = Field(default=None, ge=0)
    irrigation_type: Optional[str] = Field(default=None)
    previous_crop: Optional[str] = Field(default=None)
    region: Optional[str] = Field(default=None)

    latitude: Optional[float] = Field(default=None)
    longitude: Optional[float] = Field(default=None)


_SEASON_DEFAULTS = {
    "kharif": {
        "temperature": 30.0, "humidity": 80.0, "pressure_mb": 1005.0,
        "cloud": 70.0, "wind_kph": 12.0, "rainfall": 120.0,
        "gust_kph": 15.6, "visibility_km": 8.0, "uv_index": 8,
        "wind_degree": 180, "feels_like_celsius": 34.0, "wind_dir_enc": 8,
        "dew_point": 26.0, "feels_delta": 4.0, "gust_ratio": 1.3,
        "vis_inv": 0.118, "humid_cloud": 56.0,
    },
    "rabi": {
        "temperature": 18.0, "humidity": 55.0, "pressure_mb": 1015.0,
        "cloud": 30.0, "wind_kph": 8.0, "rainfall": 30.0,
        "gust_kph": 10.4, "visibility_km": 12.0, "uv_index": 4,
        "wind_degree": 90, "feels_like_celsius": 16.0, "wind_dir_enc": 4,
        "dew_point": 8.0, "feels_delta": -2.0, "gust_ratio": 1.3,
        "vis_inv": 0.077, "humid_cloud": 16.5,
    },
    "summer": {
        "temperature": 36.0, "humidity": 40.0, "pressure_mb": 1000.0,
        "cloud": 10.0, "wind_kph": 15.0, "rainfall": 5.0,
        "gust_kph": 19.5, "visibility_km": 15.0, "uv_index": 11,
        "wind_degree": 270, "feels_like_celsius": 40.0, "wind_dir_enc": 12,
        "dew_point": 19.0, "feels_delta": 4.0, "gust_ratio": 1.3,
        "vis_inv": 0.062, "humid_cloud": 4.0,
    },
    "zaid": {
        "temperature": 33.0, "humidity": 50.0, "pressure_mb": 1003.0,
        "cloud": 20.0, "wind_kph": 10.0, "rainfall": 15.0,
        "gust_kph": 13.0, "visibility_km": 12.0, "uv_index": 9,
        "wind_degree": 225, "feels_like_celsius": 36.0, "wind_dir_enc": 9,
        "dew_point": 22.0, "feels_delta": 3.0, "gust_ratio": 1.3,
        "vis_inv": 0.077, "humid_cloud": 10.0,
    },
}


def _pick_value(request_value, snapshot: dict, key: str, default):
    if request_value is not None:
        return request_value

    value = snapshot.get(key)
    if value is None:
        return default

    return value


def _load_calendar_context(calendar_id: int, db: Session) -> dict:
    record = db.query(CalendarRecord).filter(
        CalendarRecord.calendar_id == calendar_id
    ).first()

    if not record:
        raise HTTPException(status_code=404, detail=f"Calendar {calendar_id} not found")

    try:
        payload = json.loads(record.calendar_json) if record.calendar_json else {}
    except json.JSONDecodeError:
        payload = {}

    input_snapshot = payload.get("input_snapshot", {})

    return {
        "calendar_id": record.calendar_id,
        "farmer_name": record.farmer_name,
        "crop": payload.get("crop", record.crop),
        "season": payload.get("season", record.season),
        "location": payload.get("location", record.location),
        "soil_type": payload.get("soil_type", record.soil_type),
        "region": payload.get("region", input_snapshot.get("region", "South")),
        "input_snapshot": input_snapshot,
    }


def _get_weather_context(
    season: str,
    location: str,
    latitude: Optional[float],
    longitude: Optional[float],
) -> dict:
    weather = None

    if latitude is not None and longitude is not None:
        weather = get_weather_forecast(latitude, longitude)
        if "error" in weather:
            weather = None

    if weather is None:
        weather = get_weather_for_location(location)
        if "error" in weather:
            weather = None
            logger.warning("Could not geocode '%s' - using seasonal defaults", location)

    if weather and weather.get("forecast"):
        slots = weather["forecast"]
        return {
            "temperature": sum(s["temp"] for s in slots) / len(slots),
            "humidity": sum(s["humidity"] for s in slots) / len(slots),
            "pressure_mb": sum(s["pressure_mb"] for s in slots) / len(slots),
            "cloud": sum(s["cloud"] for s in slots) / len(slots),
            "wind_kph": sum(s["wind_kph"] for s in slots) / len(slots),
            "rainfall": sum(s["rain_3h"] for s in slots) * 8,
            "slots": slots,
            "source": "live",
            "city": weather.get("city", location),
        }

    defaults = _SEASON_DEFAULTS.get(season.lower(), _SEASON_DEFAULTS["kharif"])
    return {**defaults, "slots": None, "source": "seasonal_defaults"}


def _build_rainfall_forecast(weather_ctx: dict) -> list:
    slots = weather_ctx.get("slots")

    if slots:
        forecast = []
        for index, slot in enumerate(slots, start=1):
            result = predict_rainfall(
                temperature_celsius=slot["temp"],
                humidity=slot["humidity"],
                pressure_mb=slot["pressure_mb"],
                cloud=slot["cloud"],
                wind_kph=slot["wind_kph"],
                gust_kph=slot.get("gust_kph", slot["wind_kph"] * 1.3),
                visibility_km=slot.get("visibility_km", 10.0),
                uv_index=slot.get("uv_index", 0),
                wind_degree=slot.get("wind_degree", 180),
                feels_like_celsius=slot.get("feels_like_celsius", slot["temp"]),
                wind_dir_enc=slot.get("wind_dir_enc", 8),
                dew_point=slot.get("dew_point"),
                feels_delta=slot.get("feels_delta"),
                gust_ratio=slot.get("gust_ratio"),
                vis_inv=slot.get("vis_inv"),
                humid_cloud=slot.get("humid_cloud"),
            )

            api_rain_mm = round(float(slot.get("rain_3h", 0.0)), 2)
            ai_rain_mm = result.get("predicted_rainfall_mm", 0.0) if "error" not in result else 0.0
            rain_probability_pct = result.get("rain_probability_pct", None) if "error" not in result else None
            rainy_amount_if_rain_mm = result.get("rainy_amount_if_rain_mm", None) if "error" not in result else None

            forecast.append({
                "slot": index,
                "label": f"Slot {index}",
                "timestamp": slot.get("timestamp"),
                "source": "weather_api",
                "display_rain_mm": api_rain_mm,
                "actual_rain_mm": api_rain_mm,
                "predicted_rain_mm": round(float(ai_rain_mm), 2),
                "rain_probability_pct": rain_probability_pct,
                "rainy_amount_if_rain_mm": rainy_amount_if_rain_mm,
            })

        return forecast

    result = predict_rainfall(
        temperature_celsius=weather_ctx["temperature"],
        humidity=weather_ctx["humidity"],
        pressure_mb=weather_ctx["pressure_mb"],
        cloud=weather_ctx["cloud"],
        wind_kph=weather_ctx["wind_kph"],
        gust_kph=weather_ctx.get("gust_kph", weather_ctx["wind_kph"] * 1.3),
        visibility_km=weather_ctx.get("visibility_km", 10.0),
        uv_index=weather_ctx.get("uv_index", 0),
        wind_degree=weather_ctx.get("wind_degree", 180),
        feels_like_celsius=weather_ctx.get("feels_like_celsius", weather_ctx["temperature"]),
        wind_dir_enc=weather_ctx.get("wind_dir_enc", 8),
        dew_point=weather_ctx.get("dew_point"),
        feels_delta=weather_ctx.get("feels_delta"),
        gust_ratio=weather_ctx.get("gust_ratio"),
        vis_inv=weather_ctx.get("vis_inv"),
        humid_cloud=weather_ctx.get("humid_cloud"),
    )

    ai_rain_mm = result.get("predicted_rainfall_mm", 0.0) if "error" not in result else 0.0
    rain_probability_pct = result.get("rain_probability_pct", None) if "error" not in result else None
    rainy_amount_if_rain_mm = result.get("rainy_amount_if_rain_mm", None) if "error" not in result else None

    return [
        {
            "slot": index,
            "label": f"Slot {index}",
            "timestamp": None,
            "source": "ai_fallback",
            "display_rain_mm": round(float(ai_rain_mm), 2),
            "actual_rain_mm": None,
            "predicted_rain_mm": round(float(ai_rain_mm), 2),
            "rain_probability_pct": rain_probability_pct,
            "rainy_amount_if_rain_mm": rainy_amount_if_rain_mm,
        }
        for index in range(1, 6)
    ]


def _health_label(risk: float) -> str:
    if risk < 35:
        return "Excellent"
    if risk < 55:
        return "Good"
    if risk < 70:
        return "Moderate"
    return "At Risk"


@router.post("/generate")
def generate_insights(req: InsightsRequest, db: Session = Depends(get_db)):
    try:
        stored = _load_calendar_context(req.calendar_id, db)
        snapshot = stored["input_snapshot"]

        crop = stored["crop"]
        season = stored["season"]
        location = stored["location"]
        region = _pick_value(req.region, snapshot, "region", stored["region"] or "South")
        soil_type = _pick_value(req.soil_type, snapshot, "soil_type", stored["soil_type"] or "Clay")

        nitrogen_level = _pick_value(req.nitrogen_level, snapshot, "nitrogen_level", 50.0)
        phosphorus_level = _pick_value(req.phosphorus_level, snapshot, "phosphorus_level", 40.0)
        potassium_level = _pick_value(req.potassium_level, snapshot, "potassium_level", 40.0)
        soil_ph = _pick_value(req.soil_ph, snapshot, "soil_ph", 6.5)
        soil_moisture = _pick_value(req.soil_moisture, snapshot, "soil_moisture", 40.0)
        organic_carbon = _pick_value(req.organic_carbon, snapshot, "organic_carbon", 0.5)
        electrical_conductivity = _pick_value(req.electrical_conductivity, snapshot, "electrical_conductivity", 0.8)
        irrigation_type = _pick_value(req.irrigation_type, snapshot, "irrigation_type", "Canal")
        previous_crop = _pick_value(req.previous_crop, snapshot, "previous_crop", "Wheat")

        weather_ctx = _get_weather_context(
            season=season,
            location=location,
            latitude=req.latitude,
            longitude=req.longitude,
        )
        weather_source = weather_ctx.get("source", "seasonal_defaults")
        resolved_city = weather_ctx.get("city", location)

        fert_result = predict_fertilizer(
            soil_type=soil_type,
            crop_type=crop,
            growth_stage="Vegetative",
            season=season,
            irrigation_type=irrigation_type,
            previous_crop=previous_crop,
            region=region,
            soil_ph=soil_ph,
            soil_moisture=soil_moisture,
            organic_carbon=organic_carbon,
            electrical_conductivity=electrical_conductivity,
            nitrogen_level=nitrogen_level,
            phosphorus_level=phosphorus_level,
            potassium_level=potassium_level,
            temperature=weather_ctx["temperature"],
            humidity=weather_ctx["humidity"],
            rainfall=weather_ctx["rainfall"],
        )

        rainfall_slots = _build_rainfall_forecast(weather_ctx)
        rainfall_forecast = [slot["display_rain_mm"] for slot in rainfall_slots]
        actual_rainfall = [slot["actual_rain_mm"] for slot in rainfall_slots]
        ai_rainfall = [slot["predicted_rain_mm"] for slot in rainfall_slots]

        yield_result = predict_yield(
            crop=crop,
            season=season,
            region=region,
        )

        humidity_risk = max(0, (weather_ctx["humidity"] - 60) / 40) * 30
        k_deficiency = max(0, (60 - potassium_level) / 60) * 20
        temp_stress = max(0, (weather_ctx["temperature"] - 32) / 10) * 20
        ph_stress = abs(soil_ph - 6.5) * 5
        risk_score = min(100, round(
            humidity_risk + k_deficiency + temp_stress + ph_stress
        ))

        pest_risk = (
            "Low" if risk_score < 35 else
            "Medium" if risk_score < 60 else
            "High"
        )

        avg_rain = sum(rainfall_forecast) / len(rainfall_forecast) if rainfall_forecast else 0
        irrigation_advice = (
            "Reduce irrigation - adequate rainfall expected" if avg_rain > 10 else
            "Maintain normal irrigation schedule" if avg_rain > 5 else
            "Increase irrigation - low rainfall predicted"
        )

        rainfall_display_mode = (
            "weather_api_primary"
            if any(slot["source"] == "weather_api" for slot in rainfall_slots)
            else "ai_fallback"
        )

        return {
            "calendar_id": req.calendar_id,
            "farmer_name": stored["farmer_name"],
            "crop": crop,
            "season": season,
            "location": location,
            "region": region,
            "soil_type": soil_type,
            "weather_source": weather_source,
            "resolved_city": resolved_city,
            "rainfall_display_mode": rainfall_display_mode,
            "fertilizer_recommendation": fert_result,
            "rainfall_forecast_mm": rainfall_forecast,
            "actual_rainfall_mm": actual_rainfall,
            "ai_rainfall_mm": ai_rainfall,
            "rainfall_forecast_slots": rainfall_slots,
            "yield_prediction": yield_result,
            "overall_health": _health_label(risk_score),
            "risk_score": risk_score,
            "pest_risk": pest_risk,
            "irrigation_advice": irrigation_advice,
            "alerts": [
                f"Apply {fert_result.get('recommended_fertilizer', 'NPK')} during vegetative stage",
                f"Pest pressure is {pest_risk.lower()} - monitor {'closely' if pest_risk == 'High' else 'regularly'}",
                "Rainfall shown to users uses Weather API first when live forecast is available",
                f"Expected yield: {yield_result.get('yield_quintals_per_acre', 'N/A')} quintals/acre based on {region} region historical data",
            ],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Insights generation failed")
        raise HTTPException(status_code=500, detail=f"Insights generation failed: {e}")
