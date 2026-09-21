from pydantic import BaseModel

class SoilData(BaseModel):
    ph: float
    nitrogen: float
    phosphorus: float
    potassium: float
    organic_carbon: float | None = None
    soil_type: str | None = None

class LocationData(BaseModel):
    latitude: float
    longitude: float
    district: str | None = None
    state: str | None = None

class FarmInput(BaseModel):
    farmer_id: str | None = None
    crop_type: str
    season: str  # kharif / rabi / summer
    soil: SoilData
    location: LocationData
    area_acres: float | None = None
