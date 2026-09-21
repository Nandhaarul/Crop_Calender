from pydantic import BaseModel, Field
from typing import List, Optional


# ----- USER INPUT MODEL -----
class CropCalendarRequest(BaseModel):
    crop_name: str = Field(..., example="Rice")
    location: str = Field(..., example="Tamil Nadu")
    sowing_date: str = Field(..., example="2026-01-10")
    field_area_acres: float = Field(..., example=2.5)
    soil_type: Optional[str] = Field(None, example="Clayey")
    irrigation_type: Optional[str] = Field(None, example="Canal")
    duration_days: Optional[int] = Field(None, example=120)


# ----- CALENDAR EVENT MODEL -----
class CalendarEvent(BaseModel):
    day_start: int
    day_end: int
    task_name: str
    description: str


# ----- FINAL CALENDAR OUTPUT -----
class CropCalendarResponse(BaseModel):
    crop_name: str
    total_duration: int
    schedule: List[CalendarEvent]
