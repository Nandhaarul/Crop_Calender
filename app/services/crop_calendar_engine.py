import json
import os
from datetime import datetime, timedelta

from app.services.ml_service import predict_fertilizer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CROP_CONFIG_PATH = os.path.join(BASE_DIR, "data", "crop_stages.json")

FERTILIZER_NPK = {
    "Urea": {"Nitrogen": 46, "Phosphorus": 0, "Potassium": 0},
    "DAP": {"Nitrogen": 18, "Phosphorus": 46, "Potassium": 0},
    "MOP": {"Nitrogen": 0, "Phosphorus": 0, "Potassium": 60},
    "NPK": {"Nitrogen": 17, "Phosphorus": 17, "Potassium": 17},
    "SSP": {"Nitrogen": 0, "Phosphorus": 16, "Potassium": 0},
    "Compost": {"Nitrogen": 2, "Phosphorus": 1, "Potassium": 1},
    "Zinc Sulphate": {"Nitrogen": 0, "Phosphorus": 0, "Potassium": 0},
}

STAGE_MODEL_MAP = {
    "Sowing": "Sowing",
    "Vegetative": "Vegetative",
    "Flowering": "Flowering",
    "Harvest": "Harvest",
}

RISK_LEVELS = ["Low", "Medium", "High"]


def _load_calendar_rules():
    with open(CROP_CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


CALENDAR_RULES = _load_calendar_rules()


def _compute_stage_window(cursor_date, duration_days):
    start_date = cursor_date
    end_date = start_date + timedelta(days=duration_days - 1)
    next_date = end_date + timedelta(days=1)
    return start_date, end_date, next_date


def _normalize_key(value: str) -> str:
    return value.strip().title()


def _get_crop_season_rules(crop: str, season: str) -> dict:
    crops = CALENDAR_RULES.get("crops", {})
    if crop not in crops:
        raise ValueError(
            f"Crop '{crop}' is not supported. Supported: {list(crops.keys())}"
        )

    crop_rules = crops[crop]
    season_key = _normalize_key(season)

    if season_key in crop_rules:
        return crop_rules[season_key]

    if "Kharif" in crop_rules:
        return crop_rules["Kharif"]

    return next(iter(crop_rules.values()))


def _get_region_profile(region: str) -> dict:
    profiles = CALENDAR_RULES.get("region_profiles", {})
    profile = profiles.get(_normalize_key(region))
    if profile:
        return profile

    return {
        "duration_factor": 1.0,
        "irrigation_factor": 1.0,
        "pest_risk_offset": {
            "Sowing": 0,
            "Vegetative": 0,
            "Flowering": 0,
            "Harvest": 0,
        },
    }


def _apply_duration_factor(stage_map: dict, duration_factor: float) -> dict:
    adjusted = {}
    for stage_name, duration_days in stage_map.items():
        adjusted[stage_name] = max(5, int(round(float(duration_days) * duration_factor)))
    return adjusted


def _shift_risk_level(base_risk: str, offset: int) -> str:
    try:
        idx = RISK_LEVELS.index(base_risk)
    except ValueError:
        idx = 1

    adjusted_idx = max(0, min(len(RISK_LEVELS) - 1, idx + int(offset)))
    return RISK_LEVELS[adjusted_idx]


def _adjust_irrigation(stage_name: str, irrigation_rules: dict, rainfall: float, region_factor: float) -> dict:
    base = dict(irrigation_rules.get(stage_name, {
        "predicted_water_level": 40,
        "frequency": "As needed",
    }))

    base_level = float(base["predicted_water_level"]) * float(region_factor)

    if rainfall >= 140:
        weather_factor = 0.75
        note = "Reduced due to high expected rainfall"
    elif rainfall >= 90:
        weather_factor = 0.85
        note = "Slightly reduced due to wet conditions"
    elif rainfall <= 35:
        weather_factor = 1.20
        note = "Increased due to low expected rainfall"
    else:
        weather_factor = 1.00
        note = "Normal schedule for current weather conditions"

    if stage_name == "Harvest":
        weather_factor = min(weather_factor, 1.0)

    base["predicted_water_level"] = round(max(5.0, base_level * weather_factor), 1)
    base["weather_adjustment"] = note
    return base


def generate_weather_intelligent_calendar(
    crop: str,
    season: str,
    sowing_date: str,
    location: str,
    farmer_name: str = "Farmer",
    soil_type: str = "Clay",
    irrigation_type: str = "Canal",
    previous_crop: str = "Wheat",
    region: str = "South",
    soil_ph: float = 6.5,
    soil_moisture: float = 40.0,
    organic_carbon: float = 0.5,
    electrical_conductivity: float = 0.8,
    nitrogen_level: float = 50.0,
    phosphorus_level: float = 40.0,
    potassium_level: float = 40.0,
    temperature: float = 25.0,
    humidity: float = 60.0,
    rainfall: float = 100.0,
) -> dict:
    sowing_dt = datetime.strptime(sowing_date, "%Y-%m-%d").date()

    season_rules = _get_crop_season_rules(crop, season)
    region_profile = _get_region_profile(region)

    stages = _apply_duration_factor(
        season_rules.get("stages", {}),
        region_profile.get("duration_factor", 1.0),
    )
    pest_rules = season_rules.get("pest_risk", {})
    irrigation_rules = season_rules.get("irrigation", {})
    pest_risk_offset = region_profile.get("pest_risk_offset", {})
    irrigation_factor = region_profile.get("irrigation_factor", 1.0)

    calendar = []
    cursor_date = sowing_dt

    for stage_name, duration_days in stages.items():
        start_date, end_date, cursor_date = _compute_stage_window(cursor_date, duration_days)

        fert_result = predict_fertilizer(
            soil_type=soil_type,
            crop_type=crop,
            growth_stage=STAGE_MODEL_MAP.get(stage_name, "Vegetative"),
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
            temperature=temperature,
            humidity=humidity,
            rainfall=rainfall,
        )

        if "error" in fert_result:
            fert_name = "NPK"
            fert_confidence = 0.0
            recommendation_source = "fallback"
            fallback_reason = fert_result.get("error")
            top_alternatives = fert_result.get("top_alternatives", [])
            nutrient_family = "Balanced"
            decision_mode = "engine_fallback"
        else:
            fert_name = fert_result["recommended_fertilizer"]
            fert_confidence = fert_result["confidence"]
            recommendation_source = "model"
            fallback_reason = fert_result.get("fallback_reason")
            top_alternatives = fert_result.get("top_alternatives", [])
            nutrient_family = fert_result.get("recommended_nutrient_family")
            decision_mode = fert_result.get("decision_mode", "exact_product")

        npk = FERTILIZER_NPK.get(
            fert_name,
            {"Nitrogen": 17, "Phosphorus": 17, "Potassium": 17},
        )

        base_risk = pest_rules.get(stage_name, "Medium")
        region_offset = pest_risk_offset.get(stage_name, 0)
        pest_risk = _shift_risk_level(base_risk, region_offset)

        calendar.append({
            "stage": stage_name,
            "duration_days": duration_days,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "fertilizer_recommendation": {
                "recommended_fertilizer": fert_name,
                "recommended_nutrient_family": nutrient_family,
                "decision_mode": decision_mode,
                "model_confidence_pct": fert_confidence,
                "recommendation_source": recommendation_source,
                "fallback_reason": fallback_reason,
                "top_alternatives": top_alternatives,
                "npk_content_pct": npk,
            },
            "irrigation_advice": _adjust_irrigation(
                stage_name=stage_name,
                irrigation_rules=irrigation_rules,
                rainfall=rainfall,
                region_factor=irrigation_factor,
            ),
            "pest_disease_risk": {
                "risk_level": pest_risk,
            },
        })

    total_duration_days = sum(stages.values())
    expected_harvest_date = (sowing_dt + timedelta(days=total_duration_days - 1)).isoformat()

    return {
        "farmer_name": farmer_name,
        "crop": crop,
        "season": season,
        "location": location,
        "soil_type": soil_type,
        "sowing_date": sowing_date,
        "expected_harvest_date": expected_harvest_date,
        "weather_inputs_used": {
            "temperature": round(float(temperature), 2),
            "humidity": round(float(humidity), 2),
            "rainfall": round(float(rainfall), 2),
        },
        "calendar": calendar,
    }
