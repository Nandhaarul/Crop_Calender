import json
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.weather_routes import router as weather_router
from app.api.farm_routes import router as farm_router
from app.api.predictions_routes import router as prediction_router
from app.api.crop_calendar_routes import router as calendar_router
from app.api.insights_routes import router as insights_router
from app.database import init_db

app = FastAPI(
    title="Crop Calendar API",
    description="AI-powered crop calendar and fertilizer recommendation system",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    init_db()
    print("Database initialised")


app.include_router(farm_router)
app.include_router(weather_router)
app.include_router(prediction_router)
app.include_router(calendar_router)
app.include_router(insights_router)


def _load_metrics_summary():
    metrics_path = os.path.join(os.path.dirname(__file__), "models", "model_metrics.json")
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload.get("summary")
    except Exception:
        return None


@app.get("/")
def root():
    return {"status": "running", "version": "1.0.0"}


@app.get("/health")
def health():
    from app.services.ml_service import crop_model, fertilizer_model, rainfall_model, yield_model

    return {
        "status": "ok",
        "models": {
            "crop": crop_model is not None,
            "fertilizer": fertilizer_model is not None,
            "rainfall": rainfall_model is not None,
            "yield": yield_model is not None,
        },
        "evaluation_summary": _load_metrics_summary(),
    }
