import json
import os

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

RAW_INPUT_PATH = os.path.join(RAW_DIR, "crop_fit_dataset.csv")
OUTPUT_PATH = os.path.join(PROCESSED_DIR, "crop_fit_dataset_prepared.csv")
REPORT_PATH = os.path.join(PROCESSED_DIR, "crop_fit_prepare_report.json")

SUPPORTED_CROPS = ["Rice", "Wheat", "Maize", "Cotton", "Sugarcane", "Potato", "Tomato"]


def _normalize_crop(value):
    if pd.isna(value):
        return None

    cleaned = str(value).strip().lower()
    crop_map = {
        "rice": "Rice",
        "wheat": "Wheat",
        "maize": "Maize",
        "corn": "Maize",
        "cotton": "Cotton",
        "sugarcane": "Sugarcane",
        "potato": "Potato",
        "tomato": "Tomato",
    }
    return crop_map.get(cleaned)


def _normalize_season(value):
    if pd.isna(value):
        return None

    cleaned = str(value).strip().lower()
    season_map = {
        "kharif": "Kharif",
        "rabi": "Rabi",
        "zaid": "Zaid",
        "summer": "Zaid",
    }
    return season_map.get(cleaned)


def _normalize_soil_type(value):
    if pd.isna(value):
        return None

    cleaned = str(value).strip().lower()

    if "well-drained" in cleaned or "well drained" in cleaned:
        return "Loamy"
    if "friable" in cleaned:
        return "Loamy"
    if "clay" in cleaned or "black" in cleaned:
        return "Clay"
    if "loam" in cleaned or "laterite" in cleaned:
        return "Loamy"
    if "alluvial" in cleaned or "silt" in cleaned:
        return "Silt"
    if "sand" in cleaned or "red" in cleaned:
        return "Sandy"

    return None


def _normalize_month(value):
    if pd.isna(value):
        return None

    cleaned = str(value).strip().lower()[:3]
    month_map = {
        "jan": "Jan",
        "feb": "Feb",
        "mar": "Mar",
        "apr": "Apr",
        "may": "May",
        "jun": "Jun",
        "jul": "Jul",
        "aug": "Aug",
        "sep": "Sep",
        "oct": "Oct",
        "nov": "Nov",
        "dec": "Dec",
    }
    return month_map.get(cleaned)


def _normalize_water_source(value):
    if pd.isna(value):
        return None

    cleaned = str(value).strip().lower()
    if "irrig" in cleaned:
        return "Irrigated"
    if "rain" in cleaned:
        return "Rainfed"
    return cleaned.title()


def main():
    if not os.path.exists(RAW_INPUT_PATH):
        raise FileNotFoundError(f"Raw crop-fit dataset not found: {RAW_INPUT_PATH}")

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    df = pd.read_csv(RAW_INPUT_PATH)
    original_rows = len(df)
    original_columns = list(df.columns)

    df.columns = [str(col).strip().upper() for col in df.columns]

    required_raw_columns = [
        "CROPS",
        "SOIL",
        "SEASON",
        "SOWN",
        "HARVESTED",
        "WATER_SOURCE",
        "SOIL_PH",
        "CROPDURATION",
        "TEMP",
        "RELATIVE_HUMIDITY",
        "N",
        "P",
        "K",
    ]
    missing_raw_columns = [col for col in required_raw_columns if col not in df.columns]
    if missing_raw_columns:
        raise ValueError(
            f"Raw crop-fit dataset is missing required columns: {missing_raw_columns}"
        )

    prepared = pd.DataFrame({
        "crop": df["CROPS"].apply(_normalize_crop),
        "season": df["SEASON"].apply(_normalize_season),
        "soil_type": df["SOIL"].apply(_normalize_soil_type),
        "sown_month": df["SOWN"].apply(_normalize_month),
        "harvested_month": df["HARVESTED"].apply(_normalize_month),
        "water_source": df["WATER_SOURCE"].apply(_normalize_water_source),
        "ph": pd.to_numeric(df["SOIL_PH"], errors="coerce"),
        "crop_duration_days": pd.to_numeric(df["CROPDURATION"], errors="coerce"),
        "temperature": pd.to_numeric(df["TEMP"], errors="coerce"),
        "humidity": pd.to_numeric(df["RELATIVE_HUMIDITY"], errors="coerce"),
        "n": pd.to_numeric(df["N"], errors="coerce"),
        "p": pd.to_numeric(df["P"], errors="coerce"),
        "k": pd.to_numeric(df["K"], errors="coerce"),
    })

    prepared = prepared[prepared["crop"].isin(SUPPORTED_CROPS)].copy()

    prepared = prepared.dropna(subset=[
        "crop",
        "season",
        "soil_type",
        "ph",
        "temperature",
        "humidity",
        "n",
        "p",
        "k",
    ]).copy()

    prepared["ph"] = prepared["ph"].clip(lower=3.5, upper=9.5)
    prepared["temperature"] = prepared["temperature"].clip(lower=0.0, upper=50.0)
    prepared["humidity"] = prepared["humidity"].clip(lower=0.0, upper=100.0)
    prepared["n"] = prepared["n"].clip(lower=0.0)
    prepared["p"] = prepared["p"].clip(lower=0.0)
    prepared["k"] = prepared["k"].clip(lower=0.0)
    prepared["crop_duration_days"] = prepared["crop_duration_days"].clip(lower=1.0)

    prepared = prepared.reset_index(drop=True)
    prepared.to_csv(OUTPUT_PATH, index=False)

    report = {
        "raw_input_path": RAW_INPUT_PATH,
        "prepared_output_path": OUTPUT_PATH,
        "original_rows": int(original_rows),
        "prepared_rows": int(len(prepared)),
        "original_columns": original_columns,
        "prepared_columns": list(prepared.columns),
        "supported_crops": SUPPORTED_CROPS,
        "class_distribution": {
            crop: int(count)
            for crop, count in prepared["crop"].value_counts().sort_index().items()
        },
        "notes": [
            "Raw dataset was normalized to trainer-friendly schema.",
            "Well-drained and friable soil labels were mapped into Loamy so wheat rows are preserved honestly.",
            "Rainfall and region were not created because they do not exist in the raw file.",
            "Prepared dataset currently supports inference features: crop, season, soil_type, N, P, K, temperature, humidity, ph.",
            "Potato is still absent as a true crop label in the raw dataset; sweet potato was not remapped to potato.",
        ],
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Prepared dataset saved to: {OUTPUT_PATH}")
    print(f"Preparation report saved to: {REPORT_PATH}")
    print(f"Rows: {original_rows} -> {len(prepared)}")
    print("Class distribution:")
    for crop, count in report["class_distribution"].items():
        print(f"  {crop}: {count}")


if __name__ == "__main__":
    main()
