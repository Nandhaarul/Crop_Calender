from fastapi import APIRouter
from pydantic import BaseModel
from app.services.weather_service import get_weather_forecast

router = APIRouter(prefix="/weather", tags=["Weather"])

class WeatherInput(BaseModel):
    latitude: float
    longitude: float

@router.post("/forecast")
def weather_forecast(input: WeatherInput):
    return get_weather_forecast(input.latitude, input.longitude)
