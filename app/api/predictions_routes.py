import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from app.services.ml_service import predict_crop, predict_fertilizer, predict_rainfall

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/predict", tags=["Predictions"])


class CropPredictionRequest(BaseModel):
    N: float = Field(..., ge=0, le=140, description="Nitrogen level (kg/ha)")
    P: float = Field(..., ge=0, le=145, description="Phosphorus level (kg/ha)")
    K: float = Field(..., ge=0, le=205, description="Potassium level (kg/ha)")
    temperature: float = Field(..., ge=0, le=50, description="Temperature (deg C)")
    humidity: float = Field(..., ge=0, le=100, description="Humidity (%)")
    ph: float = Field(..., ge=3.5, le=9.9, description="Soil pH")
    rainfall: float = Field(..., ge=0, description="Rainfall (mm)")
    season: Optional[str] = Field(default=None, description="Kharif | Rabi | Zaid")
    soil_type: Optional[str] = Field(default=None, description="Clay | Loamy | Sandy | Silt")
    region: Optional[str] = Field(default=None, description="Central | East | North | South | West")
    selected_crop: Optional[str] = Field(default=None, description="Optional crop selected by user")


class FertilizerPredictionRequest(BaseModel):
    soil_type: str = Field(..., description="Clay | Loamy | Sandy | Silt")
    crop_type: str = Field(..., description="Rice | Wheat | Maize | Cotton | Sugarcane | Potato | Tomato")
    growth_stage: str = Field(..., description="Sowing | Vegetative | Flowering | Harvest")
    season: str = Field(..., description="Kharif | Rabi | Zaid")

    irrigation_type: str = Field(default="Canal", description="Canal | Drip | Rainfed | Sprinkler")
    previous_crop: str = Field(default="Wheat", description="Previous season crop")
    region: str = Field(default="South", description="Central | East | North | South | West")

    soil_ph: float = Field(default=6.5, ge=3.5, le=9.9)
    soil_moisture: float = Field(default=40.0, ge=0, le=100)
    organic_carbon: float = Field(default=0.5, ge=0)
    electrical_conductivity: float = Field(default=0.8, ge=0)
    nitrogen_level: float = Field(default=50.0, ge=0, le=140)
    phosphorus_level: float = Field(default=40.0, ge=0, le=145)
    potassium_level: float = Field(default=40.0, ge=0, le=205)
    temperature: float = Field(default=25.0, ge=0, le=50)
    humidity: float = Field(default=60.0, ge=0, le=100)
    rainfall: float = Field(default=100.0, ge=0)
    fertilizer_used_last_season: float = Field(default=100.0, ge=0)
    yield_last_season: float = Field(default=3.0, ge=0)


class RainfallPredictionRequest(BaseModel):
    temperature_celsius: float = Field(..., ge=-10, le=50)
    humidity: float = Field(..., ge=0, le=100)
    pressure_mb: float = Field(..., ge=900, le=1100)
    cloud: float = Field(..., ge=0, le=100, description="Cloud cover (%)")
    wind_kph: float = Field(..., ge=0, description="Wind speed (kph)")


@router.post("/crop")
def crop_recommendation(req: CropPredictionRequest):
    try:
        result = predict_crop(
            N=req.N,
            P=req.P,
            K=req.K,
            temperature=req.temperature,
            humidity=req.humidity,
            ph=req.ph,
            rainfall=req.rainfall,
            season=req.season,
            soil_type=req.soil_type,
            region=req.region,
            selected_crop=req.selected_crop,
        )
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        if not result.get("model_status", {}).get("available", False):
            raise HTTPException(
                status_code=503,
                detail=result.get("message") or result.get("model_status", {}).get("reason_detail") or "Crop-fit model is unavailable."
            )
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Crop-fit prediction failed")
        raise HTTPException(status_code=500, detail=f"Crop-fit prediction failed: {e}")


@router.post("/fertilizer")
def fertilizer_recommendation(req: FertilizerPredictionRequest):
    try:
        result = predict_fertilizer(
            soil_type=req.soil_type,
            crop_type=req.crop_type,
            growth_stage=req.growth_stage,
            season=req.season,
            irrigation_type=req.irrigation_type,
            previous_crop=req.previous_crop,
            region=req.region,
            soil_ph=req.soil_ph,
            soil_moisture=req.soil_moisture,
            organic_carbon=req.organic_carbon,
            electrical_conductivity=req.electrical_conductivity,
            nitrogen_level=req.nitrogen_level,
            phosphorus_level=req.phosphorus_level,
            potassium_level=req.potassium_level,
            temperature=req.temperature,
            humidity=req.humidity,
            rainfall=req.rainfall,
            fertilizer_used_last_season=req.fertilizer_used_last_season,
            yield_last_season=req.yield_last_season,
        )
        if "error" in result:
            raise HTTPException(status_code=503, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Fertilizer prediction failed")
        raise HTTPException(status_code=500, detail=f"Fertilizer prediction failed: {e}")


@router.post("/rainfall")
def rainfall_prediction(req: RainfallPredictionRequest):
    try:
        result = predict_rainfall(
            temperature_celsius=req.temperature_celsius,
            humidity=req.humidity,
            pressure_mb=req.pressure_mb,
            cloud=req.cloud,
            wind_kph=req.wind_kph,
        )
        if "error" in result:
            raise HTTPException(status_code=503, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Rainfall prediction failed")
        raise HTTPException(status_code=500, detail=f"Rainfall prediction failed: {e}")
