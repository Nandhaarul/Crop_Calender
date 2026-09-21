# AI-Powered Crop Calendar

An AI-assisted agricultural decision-support platform that converts farm inputs into a stage-wise crop calendar with crop-fit analysis, fertilizer recommendations, rainfall insights, yield estimation, and progress tracking.

---

## Overview

Farmers often receive crop, weather, fertilizer, and yield advice from separate sources. This project combines those decisions into one practical workflow.

The user enters farm details such as crop, season, sowing date, location, soil type, nutrient levels, pH, and soil moisture. The system then generates a crop calendar containing:

- Crop-fit suitability analysis
- Stage-wise fertilizer recommendation
- Irrigation guidance adjusted by rainfall context
- Pest-risk guidance
- Rainfall prediction and weather insights
- Yield estimation
- Saved calendar history and manual progress tracking

---

## Key Features

- Crop-fit recommendation for Rice, Wheat, Maize, Cotton, Sugarcane, and Tomato
- Two-stage fertilizer recommendation:
  - Fertilizer family prediction
  - Exact fertilizer product prediction with safe fallback logic
- Three-part rainfall prediction:
  - Rain/no-rain prediction
  - Meaningful-rain prediction
  - Rainfall amount prediction
- Yield estimation based on crop, season, and region
- Crop calendar generation with Sowing, Vegetative, Flowering, and Harvest stages
- SQLite-based calendar and progress storage
- Insights dashboard for fertilizer, rainfall, yield, health, and risk information
- Confidence thresholds that withhold weak ML predictions instead of showing unreliable output

---

## ML Models and Performance

| Model | Task | Algorithm | Main Metric |
|---------|---------|---------|---------|
| Crop-Fit Model | Suitable crop recommendation | RandomForestClassifier | 99.2% holdout accuracy |
| Fertilizer Product Model | Exact fertilizer recommendation | Calibrated RandomForestClassifier | 87.6% accuracy |
| Fertilizer Family Model | Nutrient-family recommendation | Calibrated RandomForestClassifier | 88.5% accuracy |
| Rain Event Model | Rain/no-rain classification | Calibrated RandomForestClassifier | 82.7% F1 Score |
| Meaningful Rain Model | Significant rainfall classification | Calibrated RandomForestClassifier | 68.0% F1 Score |
| Rainfall Amount Model | Rainfall amount prediction (mm) | RandomForestRegressor | 70.3% R² Score |
| Yield Model | Yield estimation (tonnes/hectare) | RandomForestRegressor | 62.7% R² Score |

---

## Datasets Used

### Crop-Fit Dataset

`app/data/processed/crop_fit_dataset_prepared.csv`

Crop-fit training data using:

- N, P, K
- Temperature
- Humidity
- pH
- Season
- Soil Type

### Fertilizer Dataset

`app/data/raw/Fertilizer_dataset.csv`

Contains:

- Soil characteristics
- Crop details
- Growth stages
- Nutrient information
- Weather features
- Irrigation features
- Regional features

### Rainfall Dataset

`app/data/raw/Rainfall_data.csv`

Used for:

- Rain event prediction
- Meaningful rain prediction
- Rainfall amount prediction

### Yield Dataset

`tests/crop_production.csv`

Historical Indian crop production data used for yield estimation.

---

## Architecture

```text
React Frontend
      |
      v
FastAPI Backend
      |
      +--> ML Service
      |      +--> Crop-Fit Model
      |      +--> Fertilizer Models
      |      +--> Rainfall Models
      |      +--> Yield Model
      |
      +--> Crop Calendar Engine
      |      +--> Stage Rules
      |      +--> Irrigation Rules
      |      +--> Pest-Risk Rules
      |
      +--> SQLite Database
             +--> Calendars
             +--> Progress History
```

---

## Tech Stack

### Frontend

- React
- React Router
- Axios
- Chart.js
- Tailwind CSS
- Node.js

### Backend

- Python
- FastAPI
- SQLAlchemy
- Uvicorn

### Machine Learning

- scikit-learn
- pandas
- numpy
- joblib
- Random Forest Classifier
- Random Forest Regressor

### Database

- SQLite

---

## Project Structure

```text
crop_calendar_backend/
├── app/
│   ├── api/                 # FastAPI API routes
│   ├── data/
│   │   ├── raw/             # Raw datasets
│   │   └── processed/       # Prepared datasets
│   ├── models/              # Trained ML models and metadata
│   ├── services/            # ML, weather, training, and calendar logic
│   ├── database.py          # SQLite configuration and tables
│   └── main.py              # FastAPI entry point
│
├── frontend/
│   └── src/                 # React frontend application
│
├── tests/
│   └── crop_production.csv  # Yield training dataset
│
└── crop_calendar.db         # Local SQLite database
```

---

## How It Works

1. The farmer enters crop, soil, season, sowing date, location, and nutrient details.
2. The frontend sends the data to the FastAPI backend.
3. The backend gets weather context and runs crop-fit analysis.
4. The crop calendar engine creates stage-wise dates.
5. The fertilizer model gives fertilizer recommendations for each stage.
6. Rule-based logic adjusts irrigation and pest-risk guidance.
7. The generated calendar is saved in SQLite.
8. The farmer can later view calendar details, update progress, and generate insights.

---

## Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
cd YOUR_REPOSITORY_NAME
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

#### Windows

```bash
.venv\Scripts\activate
```

#### macOS/Linux

```bash
source .venv/bin/activate
```

### 3. Install Backend Dependencies

```bash
pip install fastapi uvicorn sqlalchemy pandas numpy scikit-learn joblib requests
```

### 4. Start Backend

```bash
uvicorn app.main:app --reload
```

Backend URL:

```text
http://127.0.0.1:8000
```

Swagger Documentation:

```text
http://127.0.0.1:8000/docs
```

### 5. Start Frontend

Open a new terminal:

```bash
cd frontend
npm install
npm start
```

Frontend URL:

```text
http://localhost:3000
```

---

## Model Training

### Prepare the Crop-Fit Dataset

```bash
python app/services/prepare_crop_fit_dataset.py
```

### Train All ML Models

```bash
python app/services/train_models.py
```

Trained models are stored in:

```text
app/models/
```

---

## Database

The project uses SQLite.

Database file:

```text
crop_calendar.db
```

### Tables

#### calendars

Stores generated crop calendars.

#### progress

Stores crop-stage progress updates.

The complete generated calendar is stored as JSON in the `calendar_json` column of the `calendars` table.

---

## Important Notes

- Crop-fit currently supports Rice, Wheat, Maize, Cotton, Sugarcane, and Tomato.
- Potato is not currently included in crop-fit model training data.
- Pest-risk prediction and stage duration logic are rule-based.
- Progress tracking is manually updated by the farmer.
- Rainfall prediction uses current or recent weather context and is not a full seasonal climate forecast.
- Do not upload:
  - `.env`
  - API keys
  - `.venv`
  - `node_modules`
  - Local database files

---

## Future Improvements

- Add Potato crop-fit training data
- Add richer field-level features for yield prediction
- Add IoT/sensor-based field monitoring
- Add image-based pest and disease detection
- Add user authentication
- Move from SQLite to PostgreSQL for large-scale multi-user deployment

---

## License

This project is intended for educational and portfolio use.
