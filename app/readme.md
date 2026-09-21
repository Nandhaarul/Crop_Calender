# 🌾 AI Crop Calendar — Smart Farming Assistant

An AI-powered crop calendar and farming recommendation system built with **FastAPI**, **React**, and three **Machine Learning models** trained on real agricultural datasets.

---

## 📌 Project Overview

This system helps farmers plan their crop season by providing:
- **AI-recommended fertilizer** for each growth stage (Sowing → Vegetative → Flowering → Harvest)
- **Crop recommendation** based on soil nutrient levels
- **Rainfall forecasting** using atmospheric data
- **Pest risk assessment** per growth stage and season
- **Irrigation scheduling** tailored to each crop stage
- **Persistent crop calendars** stored in a local SQLite database

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     React Frontend                       │
│   Dashboard │ CalendarPage │ InsightsPage │ Progress     │
└──────────────────────┬──────────────────────────────────┘
                       │ HTTP (axios)
┌──────────────────────▼──────────────────────────────────┐
│               FastAPI Backend (port 8000)                │
│                                                          │
│  /crop-calendar/generate   →  crop_calendar_engine.py   │
│  /insights/generate        →  insights_routes.py        │
│  /predict/crop             →  ml_service.py             │
│  /predict/fertilizer       →  ml_service.py             │
│  /predict/rainfall         →  ml_service.py             │
│  /weather/forecast         →  weather_service.py        │
│  /health                   →  main.py                   │
└──────────┬──────────────────────────┬───────────────────┘
           │                          │
┌──────────▼──────────┐   ┌───────────▼──────────────────┐
│   SQLite Database    │   │      ML Models (pkl)         │
│   crop_calendar.db   │   │  crop_model.pkl      99.3%  │
│   (persists data)    │   │  fertilizer_model.pkl 99.5% │
└──────────────────────┘   │  rainfall_model.pkl  49% R² │
                            └──────────────────────────────┘
```

---

## 🤖 ML Models

| Model | Algorithm | Dataset | Accuracy | Features |
|---|---|---|---|---|
| Crop Recommendation | Random Forest (200 trees) | Crop_recommendation.csv (2,200 rows) | **99.3%** | N, P, K, temperature, humidity, pH, rainfall |
| Fertilizer Recommendation | Random Forest (300 trees) + Oversampling | Fertilizer_dataset.csv (10,000 rows) | **99.5%** | 19 features → 46 one-hot encoded |
| Rainfall Prediction | Random Forest Regressor (200 trees) | Rainfall_data.csv (24,070 rows) | **R² = 0.49** | temperature, humidity, pressure, cloud, wind |

> **Note on Fertilizer Model:** The original dataset had severe class imbalance (SSP class: 36 samples vs Urea: 3,000+). Manual oversampling of minority classes to 300 samples each resolved this — SSP recall improved from 5% to 72%.

---

## 🛠️ Tech Stack

**Backend**
- Python 3.11
- FastAPI + Uvicorn
- SQLAlchemy + SQLite
- scikit-learn, joblib, pandas
- python-dotenv, requests

**Frontend**
- React 18
- React Router v6
- Chart.js + react-chartjs-2
- Axios
- Tailwind CSS

---

## 🚀 Setup & Running

### Prerequisites
- Python 3.9+
- Node.js 16+

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd crop_calender_backend
```

### 2. Set up environment variables
```bash
cp .env.example .env
# Open .env and add your OpenWeatherMap API key
# Get a free key at: https://openweathermap.org/api
```

### 3. Install Python dependencies
```bash
pip install fastapi uvicorn sqlalchemy scikit-learn joblib pandas \
            python-dotenv requests
```

### 4. Train the ML models (first time only)
```bash
python app/services/train_models.py
```
This creates `app/models/crop_model.pkl`, `fertilizer_model.pkl`, and `rainfall_model.pkl`.

### 5. Start the backend
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 6. Start the frontend (new terminal)
```bash
cd frontend
npm install
npm start
```

### 7. Open the app
| URL | Purpose |
|---|---|
| http://localhost:3000 | Frontend application |
| http://localhost:8000/docs | Interactive API docs (Swagger UI) |
| http://localhost:8000/health | Model status check |

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/crop-calendar/generate` | Generate AI crop calendar with per-stage fertilizer |
| GET | `/crop-calendar/all` | List all saved calendars |
| GET | `/crop-calendar/{id}` | Get a specific calendar by ID |
| POST | `/predict/crop` | Recommend crop from soil + weather data |
| POST | `/predict/fertilizer` | Recommend fertilizer for given crop/stage |
| POST | `/predict/rainfall` | Predict rainfall (mm) from atmospheric data |
| POST | `/insights/generate` | Full AI insights: health, risk, alerts |
| POST | `/weather/forecast` | 5-day weather forecast by lat/lon |
| GET | `/health` | Model load status check |

---

## 📁 Project Structure

```
crop_calender_backend/
├── app/
│   ├── main.py                    # FastAPI app, CORS, startup
│   ├── database.py                # SQLite setup (SQLAlchemy)
│   ├── api/
│   │   ├── crop_calendar_routes.py
│   │   ├── insights_routes.py
│   │   ├── predictions_routes.py
│   │   ├── weather_routes.py
│   │   └── farm_routes.py
│   ├── services/
│   │   ├── ml_service.py          # All 3 model prediction functions
│   │   ├── crop_calendar_engine.py # Per-stage calendar generation
│   │   ├── weather_service.py     # OpenWeatherMap integration
│   │   └── train_models.py        # Model training script
│   ├── models/
│   │   ├── crop_model.pkl
│   │   ├── fertilizer_model.pkl
│   │   └── rainfall_model.pkl
│   ├── data/
│   │   └── raw/                   # Training datasets
│   └── schemas/
│       └── farm_schema.py
└── frontend/
    └── src/
        ├── pages/
        │   ├── Dashboard.js
        │   ├── CalendarPage.js
        │   ├── CalendarDetails.js
        │   ├── InsightsPage.js
        │   └── ProgressPage.js
        ├── components/
        │   ├── Header.js
        │   ├── Sidebar.js
        │   ├── CropForm.js
        │   └── AnalysisCards.js
        └── services/
            └── api.js
```

---

## 🌱 Supported Crops

Rice · Wheat · Maize · Cotton · Sugarcane · Potato · Tomato

## 🌦️ Seasons

Kharif (Jun–Oct) · Rabi (Nov–Mar) · Zaid (Apr–Jun)

## 🌍 Regions

North · South · East · West · Central

---

## 📊 Sample Request

**Generate a crop calendar:**
```json
POST /crop-calendar/generate
{
  "crop": "Rice",
  "season": "Kharif",
  "sowing_date": "2026-06-15",
  "location": "Chennai",
  "farmer_name": "Arjun",
  "soil_type": "Clay",
  "region": "South",
  "nitrogen_level": 60,
  "phosphorus_level": 40,
  "potassium_level": 35,
  "soil_ph": 6.2
}
```

**Response includes:**
- Unique calendar ID (persisted to database)
- 4 growth stages with duration in days
- AI-recommended fertilizer per stage with confidence %
- NPK content of recommended fertilizer
- Irrigation schedule (mm + frequency)
- Pest risk level (Low / Medium / High)

---

## 👨‍💻 Team / Author
 Nandha Arul(RA2211003010288)
 Harini M(RA22110030102)