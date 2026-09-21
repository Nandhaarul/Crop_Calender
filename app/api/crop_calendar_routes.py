import json
import random
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.services.crop_calendar_engine import generate_weather_intelligent_calendar
from app.services.ml_service import predict_crop
from app.services.weather_service import get_weather_for_location
from app.database import get_db, CalendarRecord, ProgressRecord

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/crop-calendar", tags=["Crop Calendar"])

STAGE_ORDER = ["Sowing", "Vegetative", "Flowering", "Harvest"]


_LOCATION_REGION = {
    "chennai": "South", "coimbatore": "South", "madurai": "South",
    "salem": "South", "trichy": "South", "vellore": "South",
    "tiruppur": "South", "tirunelveli": "South", "erode": "South",
    "tamil nadu": "South",

    "bangalore": "South", "bengaluru": "South", "mysore": "South",
    "hyderabad": "South", "visakhapatnam": "South", "vizag": "South",
    "kochi": "South", "thiruvananthapuram": "South", "trivandrum": "South",
    "kerala": "South", "karnataka": "South", "andhra pradesh": "South",
    "telangana": "South",

    "mumbai": "Central", "pune": "Central", "nagpur": "Central",
    "bhopal": "Central", "indore": "Central", "raipur": "Central",
    "maharashtra": "Central", "madhya pradesh": "Central",
    "chhattisgarh": "Central",

    "ahmedabad": "West", "surat": "West", "vadodara": "West",
    "jaipur": "West", "jodhpur": "West", "udaipur": "West",
    "gujarat": "West", "rajasthan": "West", "goa": "West",

    "delhi": "North", "new delhi": "North", "chandigarh": "North",
    "amritsar": "North", "ludhiana": "North", "lucknow": "North",
    "agra": "North", "varanasi": "North",
    "punjab": "North", "haryana": "North", "uttar pradesh": "North",
    "himachal pradesh": "North", "uttarakhand": "North",

    "kolkata": "East", "bhubaneswar": "East", "patna": "East",
    "guwahati": "East", "ranchi": "East",
    "west bengal": "East", "odisha": "East", "bihar": "East",
    "assam": "East", "jharkhand": "East",
}


def _derive_region(location: str) -> str:
    key = location.strip().lower()
    if key in _LOCATION_REGION:
        return _LOCATION_REGION[key]

    for loc_key, region in _LOCATION_REGION.items():
        if loc_key in key or key in loc_key:
            return region

    return "South"


def _record_payload(record: CalendarRecord) -> dict:
    try:
        payload = json.loads(record.calendar_json) if record.calendar_json else {}
    except json.JSONDecodeError:
        payload = {}

    payload.setdefault("crop", record.crop)
    payload.setdefault("season", record.season)
    payload.setdefault("location", record.location)

    if record.soil_type and "soil_type" not in payload:
        payload["soil_type"] = record.soil_type

    return payload


def _resolve_generation_weather(req) -> dict:
    fallback = {
        "temperature": float(req.temperature or 25.0),
        "humidity": float(req.humidity or 60.0),
        "rainfall": float(req.rainfall or 100.0),
        "weather_source": "request_defaults",
        "resolved_city": req.location,
    }

    weather = get_weather_for_location(req.location)
    if not weather or "error" in weather:
        return fallback

    slots = weather.get("forecast") or []
    if not slots:
        return fallback

    avg_temp = sum(s["temp"] for s in slots) / len(slots)
    avg_humidity = sum(s["humidity"] for s in slots) / len(slots)
    total_rainfall = sum(s["rain_3h"] for s in slots) * 8

    return {
        "temperature": round(avg_temp, 2),
        "humidity": round(avg_humidity, 2),
        "rainfall": round(total_rainfall, 2),
        "weather_source": "live",
        "resolved_city": weather.get("city", req.location),
        "lat": weather.get("lat"),
        "lon": weather.get("lon"),
    }


def _build_crop_fit_analysis(req, region: str, weather_context: dict) -> dict:
    result = predict_crop(
        N=req.nitrogen_level,
        P=req.phosphorus_level,
        K=req.potassium_level,
        temperature=weather_context["temperature"],
        humidity=weather_context["humidity"],
        ph=req.soil_ph,
        rainfall=weather_context["rainfall"],
        season=req.season,
        soil_type=req.soil_type,
        region=region,
        selected_crop=req.crop,
    )

    if "error" in result:
        return {
            "available": False,
            "selected_crop": req.crop,
            "recommended_crop": None,
            "selected_crop_matches_model": None,
            "selected_crop_probability_pct": None,
            "recommended_crop_probability_pct": None,
            "message": result["error"],
            "reason": "prediction_failed",
            "reason_detail": result["error"],
            "top_alternatives": [],
            "model_supported_crops": [],
        }

    if "crop_fit_analysis" in result:
        return result["crop_fit_analysis"]

    return {
        "available": bool(result.get("crop_fit_available", False)),
        "selected_crop": req.crop,
        "recommended_crop": result.get("recommended_crop"),
        "selected_crop_matches_model": None,
        "selected_crop_probability_pct": None,
        "recommended_crop_probability_pct": result.get("confidence"),
        "message": result.get("message"),
        "reason": None,
        "reason_detail": None,
        "top_alternatives": result.get("top_alternatives", []),
        "model_supported_crops": result.get("model_supported_crops", []),
    }


def _get_stage_progress_map(calendar_id: int, db: Session) -> dict:
    history = db.query(ProgressRecord).filter(
        ProgressRecord.calendar_id == calendar_id
    ).order_by(ProgressRecord.logged_at.desc()).all()

    stage_progress = {}
    for row in history:
        if row.stage not in stage_progress:
            stage_progress[row.stage] = row.progress_percent

    return stage_progress


def _validate_progress_update(calendar_id: int, stage: str, progress_percent: int, db: Session):
    if stage not in STAGE_ORDER:
        raise HTTPException(
            status_code=422,
            detail=f"Stage '{stage}' is invalid. Allowed stages: {STAGE_ORDER}"
        )

    stage_progress = _get_stage_progress_map(calendar_id, db)
    current_saved = stage_progress.get(stage)

    if current_saved is not None and progress_percent < current_saved:
        raise HTTPException(
            status_code=422,
            detail=f"You already saved {current_saved}% for {stage}. Progress cannot go backwards."
        )

    stage_index = STAGE_ORDER.index(stage)

    if stage_index == 0:
        return

    previous_stage = STAGE_ORDER[stage_index - 1]
    previous_stage_progress = stage_progress.get(previous_stage, 0)

    if previous_stage_progress < 100:
        raise HTTPException(
            status_code=422,
            detail=f"Complete {previous_stage} to 100% before updating {stage}."
        )


class CalendarRequest(BaseModel):
    crop: str
    season: str
    sowing_date: str = Field(..., example="2026-06-15")
    location: str
    farmer_name: str
    soil_type: str = Field(default="Clay")
    irrigation_type: str = Field(default="Canal")
    previous_crop: str = Field(default="Wheat")

    soil_ph: float = Field(default=6.5, ge=3.5, le=9.9)
    soil_moisture: float = Field(default=40.0, ge=0, le=100)
    organic_carbon: float = Field(default=0.5, ge=0)
    electrical_conductivity: float = Field(default=0.8, ge=0)
    nitrogen_level: float = Field(default=50.0, ge=0, le=140)
    phosphorus_level: float = Field(default=40.0, ge=0, le=145)
    potassium_level: float = Field(default=40.0, ge=0, le=205)
    temperature: Optional[float] = Field(default=25.0)
    humidity: Optional[float] = Field(default=60.0)
    rainfall: Optional[float] = Field(default=100.0)


class ProgressUpdateRequest(BaseModel):
    calendar_id: int
    stage: str
    progress_percent: int = Field(..., ge=0, le=100)
    notes: str = Field(default="")


@router.post("/generate")
def generate_calendar(req: CalendarRequest, db: Session = Depends(get_db)):
    region = _derive_region(req.location)
    weather_context = _resolve_generation_weather(req)
    crop_fit_analysis = _build_crop_fit_analysis(req, region, weather_context)

    input_snapshot = {
        "crop": req.crop,
        "season": req.season,
        "sowing_date": req.sowing_date,
        "location": req.location,
        "farmer_name": req.farmer_name,
        "soil_type": req.soil_type,
        "irrigation_type": req.irrigation_type,
        "previous_crop": req.previous_crop,
        "region": region,
        "soil_ph": req.soil_ph,
        "soil_moisture": req.soil_moisture,
        "organic_carbon": req.organic_carbon,
        "electrical_conductivity": req.electrical_conductivity,
        "nitrogen_level": req.nitrogen_level,
        "phosphorus_level": req.phosphorus_level,
        "potassium_level": req.potassium_level,
        "temperature": req.temperature,
        "humidity": req.humidity,
        "rainfall": req.rainfall,
    }

    try:
        result = generate_weather_intelligent_calendar(
            crop=req.crop,
            season=req.season,
            sowing_date=req.sowing_date,
            location=req.location,
            farmer_name=req.farmer_name,
            soil_type=req.soil_type,
            irrigation_type=req.irrigation_type,
            previous_crop=req.previous_crop,
            region=region,
            soil_ph=req.soil_ph,
            soil_moisture=req.soil_moisture,
            organic_carbon=req.organic_carbon,
            electrical_conductivity=req.electrical_conductivity,
            nitrogen_level=req.nitrogen_level,
            phosphorus_level=req.phosphorus_level,
            potassium_level=req.potassium_level,
            temperature=weather_context["temperature"],
            humidity=weather_context["humidity"],
            rainfall=weather_context["rainfall"],
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.exception("Calendar engine failed")
        raise HTTPException(status_code=500, detail=f"Calendar generation failed: {e}")

    calendar_id = random.randint(1_000_000_000, 9_999_999_999)

    stored_payload = {
        **result,
        "region": region,
        "weather_context": weather_context,
        "crop_fit_analysis": crop_fit_analysis,
        "input_snapshot": input_snapshot,
    }

    record = CalendarRecord(
        calendar_id=calendar_id,
        farmer_name=result["farmer_name"],
        crop=result["crop"],
        season=result["season"],
        location=result["location"],
        soil_type=result.get("soil_type"),
        sowing_date=req.sowing_date,
        calendar_json=json.dumps(stored_payload),
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    return {
        "calendar_id": calendar_id,
        **stored_payload,
    }


@router.get("/all")
def get_all_calendars(db: Session = Depends(get_db)):
    records = db.query(CalendarRecord).order_by(CalendarRecord.created_at.desc()).all()

    response = []
    for record in records:
        payload = _record_payload(record)
        response.append({
            "calendar_id": record.calendar_id,
            "farmer_name": record.farmer_name,
            "crop": record.crop,
            "season": record.season,
            "location": record.location,
            "soil_type": record.soil_type,
            "sowing_date": record.sowing_date,
            "created_at": record.created_at.isoformat() if record.created_at else None,
            **payload,
        })

    return response


@router.get("/{calendar_id}")
def get_calendar(calendar_id: int, db: Session = Depends(get_db)):
    record = db.query(CalendarRecord).filter(
        CalendarRecord.calendar_id == calendar_id
    ).first()

    if not record:
        raise HTTPException(status_code=404, detail=f"Calendar {calendar_id} not found")

    payload = _record_payload(record)

    return {
        "calendar_id": record.calendar_id,
        "farmer_name": record.farmer_name,
        "created_at": record.created_at.isoformat() if record.created_at else None,
        **payload,
    }


@router.post("/progress/update")
def update_progress(req: ProgressUpdateRequest, db: Session = Depends(get_db)):
    cal = db.query(CalendarRecord).filter(
        CalendarRecord.calendar_id == req.calendar_id
    ).first()

    if not cal:
        raise HTTPException(status_code=404, detail=f"Calendar {req.calendar_id} not found")

    _validate_progress_update(
        calendar_id=req.calendar_id,
        stage=req.stage,
        progress_percent=req.progress_percent,
        db=db,
    )

    entry = ProgressRecord(
        calendar_id=req.calendar_id,
        stage=req.stage,
        progress_percent=req.progress_percent,
        notes=req.notes or "",
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)

    return {
        "status": "success",
        "calendar_id": req.calendar_id,
        "stage": req.stage,
        "progress_percent": req.progress_percent,
        "logged_at": entry.logged_at.isoformat() if entry.logged_at else None,
    }


@router.get("/progress/{calendar_id}")
def get_progress_history(calendar_id: int, db: Session = Depends(get_db)):
    cal = db.query(CalendarRecord).filter(
        CalendarRecord.calendar_id == calendar_id
    ).first()

    if not cal:
        raise HTTPException(status_code=404, detail=f"Calendar {calendar_id} not found")

    history = db.query(ProgressRecord).filter(
        ProgressRecord.calendar_id == calendar_id
    ).order_by(ProgressRecord.logged_at.desc()).all()

    return {
        "calendar_id": calendar_id,
        "crop": cal.crop,
        "farmer_name": cal.farmer_name,
        "season": cal.season,
        "location": cal.location,
        "history": [
            {
                "stage": h.stage,
                "progress_percent": h.progress_percent,
                "notes": h.notes,
                "logged_at": h.logged_at.strftime("%d %b %Y, %I:%M %p") if h.logged_at else None,
            }
            for h in history
        ],
    }
