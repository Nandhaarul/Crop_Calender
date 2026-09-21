"""
train_models.py
Run once to train all 4 ML models and save evaluation metrics.

Crop-fit trains from:
app/data/processed/crop_fit_dataset_prepared.csv

Fertilizer training now also saves:
app/models/fertilizer_model_meta.json

Run this after updating code:
python app/services/train_models.py
"""

import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils import resample

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
TEST_DIR = os.path.join(BASE_DIR, os.pardir, "tests")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

CROP_FIT_PREPARED_PATH = os.path.join(PROCESSED_DIR, "crop_fit_dataset_prepared.csv")
FERTILIZER_PATH = os.path.join(RAW_DIR, "Fertilizer_dataset.csv")
RAINFALL_PATH = os.path.join(RAW_DIR, "Rainfall_data.csv")
DATA_CORE_PATH = os.path.join(TEST_DIR, "data_core.csv")
CROP_PROD_PATH = os.path.join(TEST_DIR, "crop_production.csv")

CROP_MODEL_PATH = os.path.join(MODEL_DIR, "crop_model.pkl")
FERTILIZER_MODEL_PATH = os.path.join(MODEL_DIR, "fertilizer_model.pkl")
FERTILIZER_FAMILY_MODEL_PATH = os.path.join(MODEL_DIR, "fertilizer_family_model.pkl")
RAINFALL_MODEL_PATH = os.path.join(MODEL_DIR, "rainfall_model.pkl")
RAINFALL_EVENT_MODEL_PATH = os.path.join(MODEL_DIR, "rainfall_event_model.pkl")
RAINFALL_MEANINGFUL_EVENT_MODEL_PATH = os.path.join(MODEL_DIR, "rainfall_meaningful_event_model.pkl")
YIELD_MODEL_PATH = os.path.join(MODEL_DIR, "yield_model.pkl")

METRICS_PATH = os.path.join(MODEL_DIR, "model_metrics.json")
CROP_META_PATH = os.path.join(MODEL_DIR, "crop_model_meta.json")
FERTILIZER_META_PATH = os.path.join(MODEL_DIR, "fertilizer_model_meta.json")
YIELD_ENCODERS_PATH = os.path.join(MODEL_DIR, "yield_encoders.json")
YIELD_META_PATH = os.path.join(MODEL_DIR, "yield_model_meta.json")
RAINFALL_FEATURES_PATH = os.path.join(MODEL_DIR, "rainfall_features.json")

APP_SUPPORTED_CROPS = ["Rice", "Wheat", "Maize", "Cotton", "Sugarcane", "Potato", "Tomato"]
CROP_FIT_TARGET_CROPS = ["Rice", "Wheat", "Maize", "Cotton", "Sugarcane", "Tomato"]
CROP_FIT_EXCLUDED_APP_CROPS = [crop for crop in APP_SUPPORTED_CROPS if crop not in CROP_FIT_TARGET_CROPS]

CROP_BASE_NUMERIC_FEATURES = ["n", "p", "k", "temperature", "humidity", "ph"]
CROP_BASE_CATEGORICAL_FEATURES = ["season", "soil_type"]
CROP_OPTIONAL_NUMERIC_FEATURES = ["rainfall"]
CROP_OPTIONAL_CATEGORICAL_FEATURES = ["region"]

CROP_MIN_ROWS_PER_CLASS = 150
CROP_MIN_HOLDOUT_ACCURACY_PCT = 70.0
CROP_MIN_CV_ACCURACY_PCT = 70.0
CROP_MIN_PREDICTION_CONFIDENCE_PCT = 60.0
CROP_MIN_PROBABILITY_MARGIN_PCT = 8.0

FERTILIZER_MIN_HOLDOUT_ACCURACY_PCT = 80.0
FERTILIZER_MIN_HOLDOUT_MACRO_F1_PCT = 65.0
FERTILIZER_MIN_PREDICTION_CONFIDENCE_PCT = 55.0
FERTILIZER_MIN_PROBABILITY_MARGIN_PCT = 10.0
FERTILIZER_MIN_CLASS_PRECISION_PCT = 45.0
FERTILIZER_MIN_CLASS_RECALL_PCT = 45.0

def _round_or_none(value, digits=4):
    if value is None:
        return None
    return round(float(value), digits)


def _pct_or_none(value, digits=1):
    if value is None:
        return None
    return round(float(value) * 100.0, digits)


def _joblib_save(obj, path):
    joblib.dump(obj, path, compress=3)


def _save_metrics(payload):
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print("  Saved -> model_metrics.json")


def _save_crop_meta(payload):
    with open(CROP_META_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print("  Saved -> crop_model_meta.json")


def _save_fertilizer_meta(payload):
    with open(FERTILIZER_META_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print("  Saved -> fertilizer_model_meta.json")


def _standardize_crop_name(value):
    if pd.isna(value):
        return None
    cleaned = str(value).strip()
    if not cleaned:
        return None
    crop_map = {crop.lower(): crop for crop in APP_SUPPORTED_CROPS}
    return crop_map.get(cleaned.lower(), cleaned.title())


def _standardize_season(value):
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


def _standardize_soil_type(value):
    if pd.isna(value):
        return None
    cleaned = str(value).strip().lower()
    soil_map = {
        "clay": "Clay",
        "clayey": "Clay",
        "black": "Clay",
        "loamy": "Loamy",
        "loam": "Loamy",
        "sandy": "Sandy",
        "red": "Sandy",
        "silt": "Silt",
        "silty": "Silt",
    }
    return soil_map.get(cleaned, str(value).strip().title())


def _standardize_region(value):
    if pd.isna(value):
        return None
    cleaned = str(value).strip().lower()
    region_map = {
        "north": "North",
        "south": "South",
        "east": "East",
        "west": "West",
        "central": "Central",
    }
    return region_map.get(cleaned, str(value).strip().title())


def _disabled_crop_meta(reason_detail, dataset_rows=0, represented_labels=None, class_distribution=None):
    represented_labels = represented_labels or []
    class_distribution = class_distribution or {}

    return {
        "available": False,
        "reason": "dataset_not_ready",
        "reason_detail": reason_detail,
        "training_source": os.path.basename(CROP_FIT_PREPARED_PATH),
        "supported_crops": CROP_FIT_TARGET_CROPS,
        "app_supported_crops": APP_SUPPORTED_CROPS,
        "excluded_app_crops": CROP_FIT_EXCLUDED_APP_CROPS,
        "represented_labels": represented_labels,
        "class_distribution": class_distribution,
        "dataset_rows_used": int(dataset_rows),
        "numeric_features": [],
        "categorical_features": [],
        "min_rows_per_class": CROP_MIN_ROWS_PER_CLASS,
        "min_holdout_accuracy_pct": CROP_MIN_HOLDOUT_ACCURACY_PCT,
        "min_cv_accuracy_pct": CROP_MIN_CV_ACCURACY_PCT,
        "min_prediction_confidence_pct": CROP_MIN_PREDICTION_CONFIDENCE_PCT,
        "min_probability_margin_pct": CROP_MIN_PROBABILITY_MARGIN_PCT,
        "holdout_accuracy_pct": None,
        "holdout_macro_f1_pct": None,
        "holdout_balanced_accuracy_pct": None,
        "cv_accuracy_mean_pct": None,
        "cv_macro_f1_mean_pct": None,
    }


def _disabled_crop_metrics(reason_detail, dataset_rows_original=0, dataset_rows_used=0, represented_labels=None, class_distribution=None):
    represented_labels = represented_labels or []
    class_distribution = class_distribution or {}
    missing_labels = [crop for crop in CROP_FIT_TARGET_CROPS if crop not in represented_labels]

    return {
        "available": False,
        "dataset_rows_original": int(dataset_rows_original),
        "dataset_rows_used": int(dataset_rows_used),
        "num_labels_original": int(len(represented_labels)),
        "num_labels_used": int(len(represented_labels)),
        "model_target_crops": CROP_FIT_TARGET_CROPS,
        "app_supported_crops": APP_SUPPORTED_CROPS,
        "excluded_app_crops": CROP_FIT_EXCLUDED_APP_CROPS,
        "app_supported_labels": represented_labels,
        "missing_supported_labels": missing_labels,
        "class_distribution": class_distribution,
        "target_coverage_pct": _pct_or_none(len(represented_labels) / len(CROP_FIT_TARGET_CROPS), 1) if CROP_FIT_TARGET_CROPS else None,
        "app_supported_coverage_pct": _pct_or_none(len(represented_labels) / len(APP_SUPPORTED_CROPS), 1) if APP_SUPPORTED_CROPS else None,
        "holdout_accuracy_pct": None,
        "holdout_macro_f1_pct": None,
        "holdout_balanced_accuracy_pct": None,
        "cv_accuracy_mean_pct": None,
        "cv_accuracy_std_pct_points": None,
        "cv_macro_f1_mean_pct": None,
        "per_class_summary": {},
        "confusion_matrix_labels": [],
        "confusion_matrix": [],
        "top_feature_importances": [],
        "validation": "Prepared crop_fit_dataset_prepared.csv quality gate",
        "note": reason_detail,
    }


def train_crop_model():
    print("\n" + "=" * 60)
    print("  MODEL 1 - Crop Fit Recommendation")
    print("=" * 60)

    if not os.path.exists(CROP_FIT_PREPARED_PATH):
        note = (
            "Prepared crop-fit dataset is missing. Run "
            "python app/services/prepare_crop_fit_dataset.py first."
        )
        crop_metrics = _disabled_crop_metrics(reason_detail=note)
        _save_crop_meta(_disabled_crop_meta(reason_detail=note))
        print(f"  {note}")
        return crop_metrics

    df = pd.read_csv(CROP_FIT_PREPARED_PATH).copy()
    dataset_rows_original = len(df)
    df.columns = [str(col).strip().lower() for col in df.columns]

    required_columns = ["crop"] + CROP_BASE_NUMERIC_FEATURES + CROP_BASE_CATEGORICAL_FEATURES
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        note = (
            "Prepared crop-fit dataset is missing required columns: "
            f"{missing_columns}. Crop-fit model disabled."
        )
        crop_metrics = _disabled_crop_metrics(
            reason_detail=note,
            dataset_rows_original=dataset_rows_original,
        )
        _save_crop_meta(_disabled_crop_meta(reason_detail=note))
        print(f"  {note}")
        return crop_metrics

    df["crop"] = df["crop"].apply(_standardize_crop_name)
    if "season" in df.columns:
        df["season"] = df["season"].apply(_standardize_season)
    if "soil_type" in df.columns:
        df["soil_type"] = df["soil_type"].apply(_standardize_soil_type)
    if "region" in df.columns:
        df["region"] = df["region"].apply(_standardize_region)

    numeric_features = list(CROP_BASE_NUMERIC_FEATURES)
    categorical_features = list(CROP_BASE_CATEGORICAL_FEATURES)

    for col in CROP_OPTIONAL_NUMERIC_FEATURES:
        if col in df.columns:
            numeric_features.append(col)

    for col in CROP_OPTIONAL_CATEGORICAL_FEATURES:
        if col in df.columns:
            categorical_features.append(col)

    for col in numeric_features:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["crop"] + numeric_features + categorical_features).copy()
    df = df[df["crop"].isin(CROP_FIT_TARGET_CROPS)].copy()
    df = df[df["season"].isin(["Kharif", "Rabi", "Zaid"])].copy()
    df = df[df["soil_type"].isin(["Clay", "Loamy", "Sandy", "Silt"])].copy()

    if "region" in categorical_features:
        df = df[df["region"].isin(["Central", "East", "North", "South", "West"])].copy()

    dataset_rows_used = len(df)
    represented_labels = sorted(df["crop"].unique().tolist()) if not df.empty else []
    class_distribution = {
        crop: int(count)
        for crop, count in df["crop"].value_counts().sort_index().items()
    } if not df.empty else {}

    missing_labels = [crop for crop in CROP_FIT_TARGET_CROPS if crop not in represented_labels]
    low_count_labels = [
        crop for crop, count in class_distribution.items()
        if count < CROP_MIN_ROWS_PER_CLASS
    ]

    if df.empty:
        note = "Prepared crop-fit dataset has no usable rows after cleaning. Crop-fit model disabled."
        crop_metrics = _disabled_crop_metrics(
            reason_detail=note,
            dataset_rows_original=dataset_rows_original,
            dataset_rows_used=dataset_rows_used,
            represented_labels=represented_labels,
            class_distribution=class_distribution,
        )
        _save_crop_meta(_disabled_crop_meta(
            reason_detail=note,
            dataset_rows=dataset_rows_used,
            represented_labels=represented_labels,
            class_distribution=class_distribution,
        ))
        print(f"  {note}")
        return crop_metrics

    if missing_labels or low_count_labels:
        note = (
            "Prepared crop-fit dataset is not ready yet. You need all target crop-fit classes and at least "
            f"{CROP_MIN_ROWS_PER_CLASS} cleaned rows per crop. "
            f"Target crops: {CROP_FIT_TARGET_CROPS}. Missing crops: {missing_labels}. "
            f"Low-count crops: {low_count_labels}. Excluded app crops without true raw rows: {CROP_FIT_EXCLUDED_APP_CROPS}."
        )
        crop_metrics = _disabled_crop_metrics(
            reason_detail=note,
            dataset_rows_original=dataset_rows_original,
            dataset_rows_used=dataset_rows_used,
            represented_labels=represented_labels,
            class_distribution=class_distribution,
        )
        _save_crop_meta(_disabled_crop_meta(
            reason_detail=note,
            dataset_rows=dataset_rows_used,
            represented_labels=represented_labels,
            class_distribution=class_distribution,
        ))
        print(f"  {note}")
        return crop_metrics

    X = df[numeric_features + categorical_features].copy()
    y = df["crop"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                ]),
                numeric_features,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical_features,
            ),
        ]
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=250,
            max_depth=18,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )),
    ])

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    holdout_accuracy = accuracy_score(y_test, preds)
    holdout_macro_f1 = f1_score(y_test, preds, average="macro")
    holdout_balanced_accuracy = balanced_accuracy_score(y_test, preds)

    min_class_count = int(y.value_counts().min())
    cv_splits = min(5, min_class_count)
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)

    cv_acc = cross_val_score(
        model,
        X,
        y,
        cv=skf,
        scoring="accuracy",
        n_jobs=-1,
    )
    cv_f1 = cross_val_score(
        model,
        X,
        y,
        cv=skf,
        scoring="f1_macro",
        n_jobs=-1,
    )

    class_report = classification_report(y_test, preds, output_dict=True, zero_division=0)
    per_class_summary = {}
    for crop in CROP_FIT_TARGET_CROPS:
        if crop in class_report:
            per_class_summary[crop] = {
                "precision_pct": _pct_or_none(class_report[crop]["precision"]),
                "recall_pct": _pct_or_none(class_report[crop]["recall"]),
                "f1_pct": _pct_or_none(class_report[crop]["f1-score"]),
                "support": int(class_report[crop]["support"]),
            }

    cm_labels = CROP_FIT_TARGET_CROPS
    cm = confusion_matrix(y_test, preds, labels=cm_labels)

    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    importances = model.named_steps["classifier"].feature_importances_
    top_feature_importances = sorted(
        [
            {
                "feature": str(name),
                "importance_pct": round(float(score) * 100.0, 2),
            }
            for name, score in zip(feature_names, importances)
        ],
        key=lambda x: -x["importance_pct"],
    )[:20]

    holdout_accuracy_pct = _pct_or_none(holdout_accuracy)
    cv_accuracy_mean_pct = _pct_or_none(np.mean(cv_acc))

    is_reliable = (
        holdout_accuracy_pct is not None
        and cv_accuracy_mean_pct is not None
        and holdout_accuracy_pct >= CROP_MIN_HOLDOUT_ACCURACY_PCT
        and cv_accuracy_mean_pct >= CROP_MIN_CV_ACCURACY_PCT
    )

    if is_reliable:
        note = "Crop-fit model trained successfully and passed reliability gates."
    else:
        note = (
            "Crop-fit model trained, but it did not pass reliability gates. "
            "Predictions will remain disabled until the dataset improves."
        )

    _joblib_save(model, CROP_MODEL_PATH)

    crop_meta = {
        "available": is_reliable,
        "reason": None if is_reliable else "metrics_below_threshold",
        "reason_detail": note,
        "training_source": os.path.basename(CROP_FIT_PREPARED_PATH),
        "supported_crops": CROP_FIT_TARGET_CROPS,
        "app_supported_crops": APP_SUPPORTED_CROPS,
        "excluded_app_crops": CROP_FIT_EXCLUDED_APP_CROPS,
        "represented_labels": represented_labels,
        "class_distribution": class_distribution,
        "dataset_rows_used": int(dataset_rows_used),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "min_rows_per_class": CROP_MIN_ROWS_PER_CLASS,
        "min_holdout_accuracy_pct": CROP_MIN_HOLDOUT_ACCURACY_PCT,
        "min_cv_accuracy_pct": CROP_MIN_CV_ACCURACY_PCT,
        "min_prediction_confidence_pct": CROP_MIN_PREDICTION_CONFIDENCE_PCT,
        "min_probability_margin_pct": CROP_MIN_PROBABILITY_MARGIN_PCT,
        "holdout_accuracy_pct": holdout_accuracy_pct,
        "holdout_macro_f1_pct": _pct_or_none(holdout_macro_f1),
        "holdout_balanced_accuracy_pct": _pct_or_none(holdout_balanced_accuracy),
        "cv_accuracy_mean_pct": cv_accuracy_mean_pct,
        "cv_macro_f1_mean_pct": _pct_or_none(np.mean(cv_f1)),
        "top_feature_importances": top_feature_importances,
    }
    _save_crop_meta(crop_meta)

    print(f"  Dataset rows original: {dataset_rows_original:,}")
    print(f"  Dataset rows used:     {dataset_rows_used:,}")
    print(f"  Labels used:           {represented_labels}")
    print(f"  Holdout Acc:           {holdout_accuracy_pct}%")
    print(f"  Holdout Macro-F1:      {_pct_or_none(holdout_macro_f1)}%")
    print(f"  Balanced Acc:          {_pct_or_none(holdout_balanced_accuracy)}%")
    print(f"  {cv_splits}-fold CV Acc:        {cv_accuracy_mean_pct}%")
    print(f"  {cv_splits}-fold CV Macro-F1:   {_pct_or_none(np.mean(cv_f1))}%")
    print(f"  Reliable:              {is_reliable}")
    print("  Saved -> crop_model.pkl")
    print("  Saved -> crop_model_meta.json")

    return {
        "available": is_reliable,
        "dataset_rows_original": int(dataset_rows_original),
        "dataset_rows_used": int(dataset_rows_used),
        "num_labels_original": int(len(represented_labels)),
        "num_labels_used": int(len(represented_labels)),
        "model_target_crops": CROP_FIT_TARGET_CROPS,
        "app_supported_crops": APP_SUPPORTED_CROPS,
        "excluded_app_crops": CROP_FIT_EXCLUDED_APP_CROPS,
        "app_supported_labels": represented_labels,
        "missing_supported_labels": missing_labels,
        "class_distribution": class_distribution,
        "target_coverage_pct": _pct_or_none(len(represented_labels) / len(CROP_FIT_TARGET_CROPS), 1),
        "app_supported_coverage_pct": _pct_or_none(len(represented_labels) / len(APP_SUPPORTED_CROPS), 1),
        "holdout_accuracy_pct": holdout_accuracy_pct,
        "holdout_macro_f1_pct": _pct_or_none(holdout_macro_f1),
        "holdout_balanced_accuracy_pct": _pct_or_none(holdout_balanced_accuracy),
        "cv_accuracy_mean_pct": cv_accuracy_mean_pct,
        "cv_accuracy_std_pct_points": _pct_or_none(np.std(cv_acc), 2),
        "cv_macro_f1_mean_pct": _pct_or_none(np.mean(cv_f1)),
        "per_class_summary": per_class_summary,
        "confusion_matrix_labels": cm_labels,
        "confusion_matrix": cm.tolist(),
        "top_feature_importances": top_feature_importances,
        "validation": "Prepared crop_fit_dataset_prepared.csv with stratified 80/20 holdout + stratified CV",
        "note": note,
    }


def train_fertilizer_model():
    print("\n" + "=" * 60)
    print("  MODEL 2 - Fertilizer Recommendation")
    print("=" * 60)

    fd = pd.read_csv(FERTILIZER_PATH)

    fert_map = {
        "Urea": "Urea",
        "DAP": "DAP",
        "14-35-14": "DAP",
        "28-28": "NPK",
        "17-17-17": "NPK",
        "20-20": "NPK",
        "10-26-26": "NPK",
    }
    soil_map = {
        "Sandy": "Sandy",
        "Loamy": "Loamy",
        "Black": "Clay",
        "Red": "Sandy",
        "Clayey": "Clay",
    }
    crop_map = {
        "Paddy": "Rice",
        "Maize": "Maize",
        "Sugarcane": "Sugarcane",
        "Cotton": "Cotton",
        "Wheat": "Wheat",
        "Tobacco": "Rice",
        "Barley": "Wheat",
        "Millets": "Maize",
        "Oil seeds": "Cotton",
        "Pulses": "Rice",
        "Ground Nuts": "Cotton",
    }
    fertilizer_family_map = {
        "Urea": "Nitrogen",
        "DAP": "Phosphorus",
        "SSP": "Phosphorus",
        "MOP": "Potassium",
        "NPK": "Balanced",
        "Compost": "Organic",
        "Zinc Sulphate": "Micronutrient",
    }
    family_fallback_product_map = {
        "Nitrogen": "Urea",
        "Phosphorus": "DAP",
        "Potassium": "MOP",
        "Balanced": "NPK",
        "Organic": "Compost",
        "Micronutrient": "Zinc Sulphate",
    }

    train_base, test_base = train_test_split(
        fd,
        test_size=0.2,
        random_state=42,
        stratify=fd["Recommended_Fertilizer"],
    )

    extra_rows = 0
    try:
        dc = pd.read_csv(DATA_CORE_PATH)
        dc_mapped = pd.DataFrame({
            "Soil_Type": dc["Soil Type"].map(soil_map),
            "Soil_pH": 6.5,
            "Soil_Moisture": dc["Moisture"],
            "Organic_Carbon": 0.5,
            "Electrical_Conductivity": 0.8,
            "Nitrogen_Level": dc["Nitrogen"],
            "Phosphorus_Level": dc["Phosphorous"],
            "Potassium_Level": dc["Potassium"],
            "Temperature": dc["Temparature"],
            "Humidity": dc["Humidity"],
            "Rainfall": 100.0,
            "Crop_Type": dc["Crop Type"].map(crop_map),
            "Crop_Growth_Stage": "Vegetative",
            "Season": "Kharif",
            "Irrigation_Type": "Canal",
            "Previous_Crop": "Wheat",
            "Region": "South",
            "Fertilizer_Used_Last_Season": 100.0,
            "Yield_Last_Season": 3.0,
            "Recommended_Fertilizer": dc["Fertilizer Name"].map(fert_map),
        }).dropna()
        extra_rows = len(dc_mapped)
        train_pool = pd.concat([train_base, dc_mapped], ignore_index=True)
    except FileNotFoundError:
        train_pool = train_base

    train_pool = train_pool.copy()
    target_size = max(300, int(train_pool["Recommended_Fertilizer"].value_counts().median()))
    balanced_parts = []

    for cls in sorted(train_pool["Recommended_Fertilizer"].unique()):
        subset = train_pool[train_pool["Recommended_Fertilizer"] == cls]
        if len(subset) < target_size:
            subset = resample(subset, replace=True, n_samples=target_size, random_state=42)
        balanced_parts.append(subset)

    train_balanced = pd.concat(balanced_parts, ignore_index=True)

    X_train = pd.get_dummies(train_balanced.drop("Recommended_Fertilizer", axis=1))
    y_train = train_balanced["Recommended_Fertilizer"]
    y_train_family = y_train.map(fertilizer_family_map)

    feature_columns = list(X_train.columns)

    X_test = pd.get_dummies(test_base.drop("Recommended_Fertilizer", axis=1))
    y_test = test_base["Recommended_Fertilizer"]
    y_test_family = y_test.map(fertilizer_family_map)
    X_test = X_test.reindex(columns=feature_columns, fill_value=0)

    product_base_model = RandomForestClassifier(
        n_estimators=220,
        max_depth=18,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    product_model = CalibratedClassifierCV(estimator=product_base_model, method="sigmoid", cv=3)
    product_model.fit(X_train, y_train)

    family_base_model = RandomForestClassifier(
        n_estimators=180,
        max_depth=16,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    family_model = CalibratedClassifierCV(estimator=family_base_model, method="sigmoid", cv=3)
    family_model.fit(X_train, y_train_family)

    preds = product_model.predict(X_test)
    probas = product_model.predict_proba(X_test)
    family_preds = family_model.predict(X_test)
    family_probas = family_model.predict_proba(X_test)
    holdout_accuracy = accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average="macro")
    balanced_acc = balanced_accuracy_score(y_test, preds)
    family_holdout_accuracy = accuracy_score(y_test_family, family_preds)
    family_macro_f1 = f1_score(y_test_family, family_preds, average="macro")

    class_report = classification_report(y_test, preds, output_dict=True, zero_division=0)
    class_summary = {}
    for cls in sorted(y_test.unique()):
        if cls in class_report:
            class_summary[cls] = {
                "precision_pct": _pct_or_none(class_report[cls]["precision"]),
                "recall_pct": _pct_or_none(class_report[cls]["recall"]),
                "f1_pct": _pct_or_none(class_report[cls]["f1-score"]),
                "support": int(class_report[cls]["support"]),
            }

    classes_ = list(product_model.classes_)
    class_to_index = {label: idx for idx, label in enumerate(classes_)}
    avg_top_confidence_pct = _pct_or_none(np.mean(np.max(probas, axis=1)))
    avg_probability_margin_pct = _pct_or_none(
        np.mean(
            np.sort(probas, axis=1)[:, -1] - np.sort(probas, axis=1)[:, -2]
        )
    )
    family_classes = list(family_model.classes_)
    family_avg_top_confidence_pct = _pct_or_none(np.mean(np.max(family_probas, axis=1)))

    reliable_classes = []
    unreliable_classes = []
    for label in classes_:
        metrics = class_summary.get(label, {})
        precision_pct = float(metrics.get("precision_pct") or 0.0)
        recall_pct = float(metrics.get("recall_pct") or 0.0)
        if (
            precision_pct >= FERTILIZER_MIN_CLASS_PRECISION_PCT
            and recall_pct >= FERTILIZER_MIN_CLASS_RECALL_PCT
        ):
            reliable_classes.append(label)
        else:
            unreliable_classes.append(label)

    is_globally_reliable = (
        (_pct_or_none(holdout_accuracy) or 0.0) >= FERTILIZER_MIN_HOLDOUT_ACCURACY_PCT
        and (_pct_or_none(macro_f1) or 0.0) >= FERTILIZER_MIN_HOLDOUT_MACRO_F1_PCT
        and len(reliable_classes) >= 5
    )

    if is_globally_reliable:
        note = (
            "Fertilizer model passed reliability gates. Predictions are still withheld "
            "for weak classes or low-confidence cases."
        )
        reason = None
    else:
        note = (
            "Fertilizer model trained, but overall metrics did not pass reliability gates. "
            "Predictions should be treated as unavailable until the dataset improves."
        )
        reason = "metrics_below_threshold"

    fertilizer_meta = {
        "available": is_globally_reliable,
        "reason": reason,
        "reason_detail": note,
        "training_source": os.path.basename(FERTILIZER_PATH),
        "feature_columns": feature_columns,
        "model_classes": classes_,
        "family_model_classes": family_classes,
        "reliable_classes": reliable_classes,
        "unreliable_classes": unreliable_classes,
        "fertilizer_family_map": fertilizer_family_map,
        "family_fallback_product_map": family_fallback_product_map,
        "per_class_summary": class_summary,
        "holdout_accuracy_pct": _pct_or_none(holdout_accuracy),
        "holdout_macro_f1_pct": _pct_or_none(macro_f1),
        "holdout_balanced_accuracy_pct": _pct_or_none(balanced_acc),
        "family_holdout_accuracy_pct": _pct_or_none(family_holdout_accuracy),
        "family_holdout_macro_f1_pct": _pct_or_none(family_macro_f1),
        "avg_top_confidence_pct": avg_top_confidence_pct,
        "avg_probability_margin_pct": avg_probability_margin_pct,
        "family_avg_top_confidence_pct": family_avg_top_confidence_pct,
        "min_holdout_accuracy_pct": FERTILIZER_MIN_HOLDOUT_ACCURACY_PCT,
        "min_holdout_macro_f1_pct": FERTILIZER_MIN_HOLDOUT_MACRO_F1_PCT,
        "min_prediction_confidence_pct": FERTILIZER_MIN_PREDICTION_CONFIDENCE_PCT,
        "min_probability_margin_pct": FERTILIZER_MIN_PROBABILITY_MARGIN_PCT,
        "min_class_precision_pct": FERTILIZER_MIN_CLASS_PRECISION_PCT,
        "min_class_recall_pct": FERTILIZER_MIN_CLASS_RECALL_PCT,
        "class_probability_indexes": class_to_index,
        "target_resample_size": int(target_size),
    }

    _joblib_save(product_model, FERTILIZER_MODEL_PATH)
    _joblib_save(family_model, FERTILIZER_FAMILY_MODEL_PATH)
    _save_fertilizer_meta(fertilizer_meta)

    print(f"  Base rows:        {len(fd):,}")
    print(f"  Train rows:       {len(train_base):,} base + {extra_rows:,} extra")
    print(f"  Test rows:        {len(test_base):,} untouched")
    print(f"  Holdout Acc:      {_pct_or_none(holdout_accuracy)}%")
    print(f"  Macro F1:         {_pct_or_none(macro_f1)}%")
    print(f"  Balanced Acc:     {_pct_or_none(balanced_acc)}%")
    print(f"  Family Acc:       {_pct_or_none(family_holdout_accuracy)}%")
    print(f"  Family Macro F1:  {_pct_or_none(family_macro_f1)}%")
    print(f"  Reliable classes: {reliable_classes}")
    print(f"  Blocked classes:  {unreliable_classes}")
    print("  Saved -> fertilizer_model.pkl")
    print("  Saved -> fertilizer_family_model.pkl")
    print("  Saved -> fertilizer_model_meta.json")

    return {
        "available": is_globally_reliable,
        "dataset_rows_base": int(len(fd)),
        "train_rows_base_only": int(len(train_base)),
        "test_rows_base_only": int(len(test_base)),
        "extra_training_rows": int(extra_rows),
        "holdout_accuracy_pct": _pct_or_none(holdout_accuracy),
        "holdout_macro_f1_pct": _pct_or_none(macro_f1),
        "holdout_balanced_accuracy_pct": _pct_or_none(balanced_acc),
        "family_holdout_accuracy_pct": _pct_or_none(family_holdout_accuracy),
        "family_holdout_macro_f1_pct": _pct_or_none(family_macro_f1),
        "avg_top_confidence_pct": avg_top_confidence_pct,
        "avg_probability_margin_pct": avg_probability_margin_pct,
        "family_avg_top_confidence_pct": family_avg_top_confidence_pct,
        "reliable_classes": reliable_classes,
        "unreliable_classes": unreliable_classes,
        "per_class_summary": class_summary,
        "validation": "Stratified 80/20 holdout from base dataset; calibrated exact-product and nutrient-family classifiers; extra rows used for training only",
        "note": note + " The system now falls back to nutrient-family guidance when exact product confidence is weak.",
    }


def _build_rainfall_feature_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()

    out["wind_dir_enc"] = pd.to_numeric(out.get("wind_dir_enc", 8), errors="coerce").fillna(8)
    out["temperature_celsius"] = pd.to_numeric(out["temperature_celsius"], errors="coerce")
    out["humidity"] = pd.to_numeric(out["humidity"], errors="coerce").clip(lower=1, upper=100)
    out["pressure_mb"] = pd.to_numeric(out["pressure_mb"], errors="coerce")
    out["cloud"] = pd.to_numeric(out["cloud"], errors="coerce").clip(lower=0, upper=100)
    out["wind_kph"] = pd.to_numeric(out["wind_kph"], errors="coerce").clip(lower=0)
    out["gust_kph"] = pd.to_numeric(out["gust_kph"], errors="coerce").clip(lower=0)
    out["visibility_km"] = pd.to_numeric(out["visibility_km"], errors="coerce").clip(lower=0.1)
    out["uv_index"] = pd.to_numeric(out["uv_index"], errors="coerce").clip(lower=0)
    out["wind_degree"] = pd.to_numeric(out["wind_degree"], errors="coerce").fillna(180)
    out["feels_like_celsius"] = pd.to_numeric(out["feels_like_celsius"], errors="coerce")

    out["dew_point"] = out["temperature_celsius"] - ((100 - out["humidity"]) / 5)
    out["feels_delta"] = out["feels_like_celsius"] - out["temperature_celsius"]
    out["gust_ratio"] = out["gust_kph"] / (out["wind_kph"] + 0.1)
    out["vis_inv"] = 1 / (out["visibility_km"] + 0.1)
    out["humid_cloud"] = out["humidity"] * out["cloud"] / 100

    radians = np.deg2rad(out["wind_degree"])
    out["wind_u"] = out["wind_kph"] * np.cos(radians)
    out["wind_v"] = out["wind_kph"] * np.sin(radians)
    out["gust_delta"] = out["gust_kph"] - out["wind_kph"]
    out["temp_humidity"] = out["temperature_celsius"] * out["humidity"] / 100
    out["pressure_cloud"] = out["pressure_mb"] * out["cloud"] / 100

    sat_vapor_pressure = 0.6108 * np.exp((17.27 * out["temperature_celsius"]) / (out["temperature_celsius"] + 237.3))
    actual_vapor_pressure = sat_vapor_pressure * out["humidity"] / 100
    out["vpd"] = np.maximum(sat_vapor_pressure - actual_vapor_pressure, 0)
    out["dryness_index"] = out["temperature_celsius"] * (100 - out["humidity"]) / 100
    out["uv_clear_sky"] = out["uv_index"] * (100 - out["cloud"]) / 100

    features = [
        "temperature_celsius",
        "humidity",
        "pressure_mb",
        "cloud",
        "wind_kph",
        "gust_kph",
        "visibility_km",
        "uv_index",
        "wind_degree",
        "feels_like_celsius",
        "wind_dir_enc",
        "dew_point",
        "feels_delta",
        "gust_ratio",
        "vis_inv",
        "humid_cloud",
        "wind_u",
        "wind_v",
        "gust_delta",
        "temp_humidity",
        "pressure_cloud",
        "vpd",
        "dryness_index",
        "uv_clear_sky",
    ]

    return out[features].copy(), features


def _select_probability_threshold(y_true, probs, threshold_grid, recall_floor=0.0):
    best = None
    for threshold in threshold_grid:
        preds = (probs >= threshold).astype(int)
        precision = precision_score(y_true, preds, zero_division=0)
        recall = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        candidate = {
            "threshold": round(float(threshold), 4),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

        if recall < recall_floor:
            continue

        if best is None or candidate["f1"] > best["f1"] or (
            candidate["f1"] == best["f1"] and candidate["precision"] > best["precision"]
        ):
            best = candidate

    if best is not None:
        return best

    for threshold in threshold_grid:
        preds = (probs >= threshold).astype(int)
        precision = precision_score(y_true, preds, zero_division=0)
        recall = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        candidate = {
            "threshold": round(float(threshold), 4),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
        if best is None or candidate["f1"] > best["f1"] or (
            candidate["f1"] == best["f1"] and candidate["precision"] > best["precision"]
        ):
            best = candidate

    return best


def _combine_rainfall_predictions(
    rainy_amount_preds,
    event_probs,
    event_threshold,
    meaningful_probs,
    meaningful_threshold,
    params,
):
    event_preds = (event_probs >= event_threshold).astype(int)
    meaningful_preds = (meaningful_probs >= meaningful_threshold).astype(int)

    preds = np.asarray(rainy_amount_preds, dtype=float).copy()
    low_rain_scale = float(params["low_rain_scale"])
    low_rain_floor = float(params["low_rain_floor"])
    meaningful_boost = float(params["meaningful_boost"])
    moderate_base = float(params["moderate_base"])
    moderate_slope = float(params["moderate_slope"])
    moderate_cap = float(params["moderate_cap"])

    preds = np.where(
        event_preds == 1,
        preds,
        preds * np.clip(event_probs * low_rain_scale, low_rain_floor, low_rain_scale),
    )
    preds = np.where(
        (event_preds == 1) & (meaningful_preds == 1),
        np.maximum(preds, meaningful_probs * meaningful_boost),
        preds,
    )
    preds = np.where(
        (event_preds == 1) & (meaningful_preds == 0),
        preds * np.clip(moderate_base + moderate_slope * event_probs, moderate_base, moderate_cap),
        preds,
    )

    return np.maximum(preds, 0)


def train_rainfall_model():
    print("\n" + "=" * 60)
    print("  MODEL 3 - Rainfall Prediction")
    print("=" * 60)

    df = pd.read_csv(RAINFALL_PATH)

    wind_dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    dir_map = {d: i for i, d in enumerate(wind_dirs)}
    df["wind_dir_enc"] = df["wind_direction"].map(dir_map).fillna(8)

    X, features = _build_rainfall_feature_frame(df)
    y = df["precip_mm"]
    rain_event_cutoff_mm = 0.1
    meaningful_rain_cutoff_mm = 1.0
    rain_event = (y >= rain_event_cutoff_mm).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        stratify=rain_event,
        test_size=0.2,
        random_state=42,
    )

    rain_event_train = (y_train >= rain_event_cutoff_mm).astype(int)
    rain_event_test = (y_test >= rain_event_cutoff_mm).astype(int)
    meaningful_event_test = (y_test >= meaningful_rain_cutoff_mm).astype(int)

    X_model_train, X_val, y_model_train, y_val = train_test_split(
        X_train,
        y_train,
        stratify=rain_event_train,
        test_size=0.2,
        random_state=42,
    )

    rain_event_model_train = (y_model_train >= rain_event_cutoff_mm).astype(int)
    rain_event_val = (y_val >= rain_event_cutoff_mm).astype(int)
    meaningful_event_model_train = (y_model_train >= meaningful_rain_cutoff_mm).astype(int)
    meaningful_event_val = (y_val >= meaningful_rain_cutoff_mm).astype(int)

    event_base_model = RandomForestClassifier(
        n_estimators=320,
        max_depth=20,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    event_model = CalibratedClassifierCV(estimator=event_base_model, method="sigmoid", cv=3)
    event_model.fit(X_model_train, rain_event_model_train)
    event_val_probs = event_model.predict_proba(X_val)[:, 1]
    event_threshold_info = _select_probability_threshold(
        y_true=rain_event_val,
        probs=event_val_probs,
        threshold_grid=np.arange(0.25, 0.71, 0.02),
        recall_floor=0.82,
    )
    event_threshold = event_threshold_info["threshold"]

    meaningful_base_model = RandomForestClassifier(
        n_estimators=340,
        max_depth=20,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    meaningful_event_model = CalibratedClassifierCV(
        estimator=meaningful_base_model,
        method="sigmoid",
        cv=3,
    )
    meaningful_event_model.fit(X_model_train, meaningful_event_model_train)
    meaningful_val_probs = meaningful_event_model.predict_proba(X_val)[:, 1]
    meaningful_threshold_info = _select_probability_threshold(
        y_true=meaningful_event_val,
        probs=meaningful_val_probs,
        threshold_grid=np.arange(0.18, 0.61, 0.02),
        recall_floor=0.55,
    )
    meaningful_event_threshold = meaningful_threshold_info["threshold"]

    rainy_model_train_mask = y_model_train >= rain_event_cutoff_mm
    amount_model = RandomForestRegressor(
        n_estimators=420,
        max_depth=20,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    amount_targets = np.log1p(y_model_train[rainy_model_train_mask])
    amount_weights = 1.0 + np.log1p(y_model_train[rainy_model_train_mask]) * 3.0
    amount_model.fit(
        X_model_train[rainy_model_train_mask],
        amount_targets,
        sample_weight=amount_weights,
    )

    rainy_amount_val_preds = np.expm1(amount_model.predict(X_val))
    blend_grid = [
        {
            "low_rain_scale": low_rain_scale,
            "low_rain_floor": low_rain_floor,
            "meaningful_boost": meaningful_boost,
            "moderate_base": moderate_base,
            "moderate_slope": moderate_slope,
            "moderate_cap": moderate_cap,
        }
        for low_rain_scale in [0.25, 0.30, 0.35]
        for low_rain_floor in [0.01, 0.02, 0.03]
        for meaningful_boost in [1.6, 2.0, 2.4]
        for moderate_base in [0.45, 0.50, 0.55]
        for moderate_slope in [0.35, 0.40, 0.45]
        for moderate_cap in [0.85, 0.90, 0.95]
    ]

    meaningful_rain_val_actual = (y_val >= meaningful_rain_cutoff_mm).astype(int)
    best_blend = None
    for params in blend_grid:
        val_preds = _combine_rainfall_predictions(
            rainy_amount_preds=rainy_amount_val_preds,
            event_probs=event_val_probs,
            event_threshold=event_threshold,
            meaningful_probs=meaningful_val_probs,
            meaningful_threshold=meaningful_event_threshold,
            params=params,
        )
        val_meaningful_pred = (val_preds >= meaningful_rain_cutoff_mm).astype(int)
        val_event_pred = (event_val_probs >= event_threshold).astype(int)
        val_meaningful_f1 = f1_score(meaningful_rain_val_actual, val_meaningful_pred, zero_division=0)
        val_meaningful_recall = recall_score(meaningful_rain_val_actual, val_meaningful_pred, zero_division=0)
        val_event_f1 = f1_score(rain_event_val, val_event_pred, zero_division=0)
        val_r2 = r2_score(y_val, val_preds)
        score = val_meaningful_f1 + (0.25 * val_event_f1) + (0.10 * max(val_r2, 0))
        candidate = {
            "params": params,
            "score": float(score),
            "meaningful_f1": float(val_meaningful_f1),
            "meaningful_recall": float(val_meaningful_recall),
            "event_f1": float(val_event_f1),
            "r2": float(val_r2),
        }
        if (
            best_blend is None
            or candidate["score"] > best_blend["score"]
            or (
                candidate["score"] == best_blend["score"]
                and candidate["meaningful_recall"] > best_blend["meaningful_recall"]
            )
        ):
            best_blend = candidate

    event_model.fit(X_train, rain_event_train)
    meaningful_event_train = (y_train >= meaningful_rain_cutoff_mm).astype(int)
    meaningful_event_model.fit(X_train, meaningful_event_train)

    rainy_train_mask = y_train >= rain_event_cutoff_mm
    amount_model.fit(
        X_train[rainy_train_mask],
        np.log1p(y_train[rainy_train_mask]),
        sample_weight=1.0 + np.log1p(y_train[rainy_train_mask]) * 3.0,
    )

    event_probs = event_model.predict_proba(X_test)[:, 1]
    event_preds = (event_probs >= event_threshold).astype(int)
    meaningful_event_probs = meaningful_event_model.predict_proba(X_test)[:, 1]
    meaningful_event_preds = (meaningful_event_probs >= meaningful_event_threshold).astype(int)
    rainy_amount_preds = np.expm1(amount_model.predict(X_test))
    preds = _combine_rainfall_predictions(
        rainy_amount_preds=rainy_amount_preds,
        event_probs=event_probs,
        event_threshold=event_threshold,
        meaningful_probs=meaningful_event_probs,
        meaningful_threshold=meaningful_event_threshold,
        params=best_blend["params"],
    )

    baseline = DummyRegressor(strategy="mean")
    baseline.fit(X_train, y_train)
    baseline_preds = np.maximum(baseline.predict(X_test), 0)

    overall_r2 = r2_score(y_test, preds)
    overall_mae = mean_absolute_error(y_test, preds)
    baseline_r2 = r2_score(y_test, baseline_preds)
    baseline_mae = mean_absolute_error(y_test, baseline_preds)

    rainy_mask = y_test >= rain_event_cutoff_mm
    if rainy_mask.sum() > 1:
        rainy_r2 = r2_score(y_test[rainy_mask], preds[rainy_mask])
        rainy_mae = mean_absolute_error(y_test[rainy_mask], preds[rainy_mask])
    else:
        rainy_r2 = None
        rainy_mae = None

    rain_event_recall = recall_score(rain_event_test, event_preds, zero_division=0)
    rain_event_precision = precision_score(rain_event_test, event_preds, zero_division=0)
    rain_event_f1 = f1_score(rain_event_test, event_preds, zero_division=0)

    meaningful_rain_actual = (y_test >= meaningful_rain_cutoff_mm).astype(int)
    meaningful_rain_pred = (preds >= 1.0).astype(int)
    meaningful_event_precision = precision_score(
        meaningful_event_test,
        meaningful_event_preds,
        zero_division=0,
    )
    meaningful_rain_recall = recall_score(
        meaningful_rain_actual,
        meaningful_rain_pred,
        zero_division=0,
    )
    meaningful_event_recall = recall_score(
        meaningful_event_test,
        meaningful_event_preds,
        zero_division=0,
    )
    meaningful_event_f1 = f1_score(
        meaningful_event_test,
        meaningful_event_preds,
        zero_division=0,
    )
    meaningful_rain_precision = precision_score(
        meaningful_rain_actual,
        meaningful_rain_pred,
        zero_division=0,
    )
    meaningful_rain_f1 = f1_score(
        meaningful_rain_actual,
        meaningful_rain_pred,
        zero_division=0,
    )

    feature_meta = {
        "features": features,
        "wind_dir_map": dir_map,
        "event_threshold": event_threshold,
        "meaningful_event_threshold": meaningful_event_threshold,
        "event_threshold_precision_pct": _pct_or_none(event_threshold_info["precision"]),
        "event_threshold_recall_pct": _pct_or_none(event_threshold_info["recall"]),
        "event_threshold_f1_pct": _pct_or_none(event_threshold_info["f1"]),
        "meaningful_threshold_precision_pct": _pct_or_none(meaningful_threshold_info["precision"]),
        "meaningful_threshold_recall_pct": _pct_or_none(meaningful_threshold_info["recall"]),
        "meaningful_threshold_f1_pct": _pct_or_none(meaningful_threshold_info["f1"]),
        "combination_params": best_blend["params"],
        "rain_event_cutoff_mm": rain_event_cutoff_mm,
        "meaningful_rain_cutoff_mm": meaningful_rain_cutoff_mm,
        "strategy": "three_stage_event_meaningful_amount_v2",
    }

    _joblib_save(amount_model, RAINFALL_MODEL_PATH)
    _joblib_save(event_model, RAINFALL_EVENT_MODEL_PATH)
    _joblib_save(meaningful_event_model, RAINFALL_MEANINGFUL_EVENT_MODEL_PATH)
    with open(RAINFALL_FEATURES_PATH, "w", encoding="utf-8") as f:
        json.dump(feature_meta, f)

    print(f"  Dataset rows:     {len(df):,}")
    print(f"  Zero-rain share:  {_pct_or_none((df['precip_mm'] == 0).mean())}%")
    print(f"  Event threshold:  {event_threshold}")
    print(f"  Meaningful thr:   {meaningful_event_threshold}")
    print(f"  Overall R2:       {_pct_or_none(overall_r2)}%")
    print(f"  Overall MAE:      {_round_or_none(overall_mae, 4)} mm")
    print(f"  Rainy-only MAE:   {_round_or_none(rainy_mae, 4)} mm")
    print(f"  Event precision:  {_pct_or_none(rain_event_precision)}%")
    print(f"  Event recall:     {_pct_or_none(rain_event_recall)}%")
    print(f"  Event F1:         {_pct_or_none(rain_event_f1)}%")
    print(f"  Meaningful cls P: {_pct_or_none(meaningful_event_precision)}%")
    print(f"  Meaningful cls R: {_pct_or_none(meaningful_event_recall)}%")
    print(f"  Meaningful cls F1:{_pct_or_none(meaningful_event_f1)}%")
    print(f"  >=1mm precision:  {_pct_or_none(meaningful_rain_precision)}%")
    print(f"  >=1mm recall:     {_pct_or_none(meaningful_rain_recall)}%")
    print(f"  >=1mm F1:         {_pct_or_none(meaningful_rain_f1)}%")
    print("  Saved -> rainfall_model.pkl")
    print("  Saved -> rainfall_event_model.pkl")
    print("  Saved -> rainfall_meaningful_event_model.pkl")
    print("  Saved -> rainfall_features.json")

    return {
        "dataset_rows": int(len(df)),
        "zero_rain_share_pct": _pct_or_none((df["precip_mm"] == 0).mean()),
        "overall_r2_pct": _pct_or_none(overall_r2),
        "overall_mae_mm": _round_or_none(overall_mae, 4),
        "baseline_r2_pct": _pct_or_none(baseline_r2),
        "baseline_mae_mm": _round_or_none(baseline_mae, 4),
        "rainy_rows_in_test": int(rainy_mask.sum()),
        "rainy_only_r2_pct": _pct_or_none(rainy_r2),
        "rainy_only_mae_mm": _round_or_none(rainy_mae, 4),
        "rain_event_precision_pct": _pct_or_none(rain_event_precision),
        "rain_event_recall_pct": _pct_or_none(rain_event_recall),
        "rain_event_f1_pct": _pct_or_none(rain_event_f1),
        "meaningful_event_precision_pct": _pct_or_none(meaningful_event_precision),
        "meaningful_event_recall_pct": _pct_or_none(meaningful_event_recall),
        "meaningful_event_f1_pct": _pct_or_none(meaningful_event_f1),
        "meaningful_rain_precision_pct": _pct_or_none(meaningful_rain_precision),
        "meaningful_rain_recall_pct": _pct_or_none(meaningful_rain_recall),
        "meaningful_rain_f1_pct": _pct_or_none(meaningful_rain_f1),
        "validation": "Stratified 80/20 holdout with calibrated rain-event classifier, calibrated meaningful-rain classifier, and rainy-amount regressor",
        "note": "Rainfall now models any-rain, meaningful-rain, and rainy amount separately to improve useful rainfall detection.",
    }


def train_yield_model():
    print("\n" + "=" * 60)
    print("  MODEL 4 - Yield Prediction")
    print("=" * 60)

    df = pd.read_csv(CROP_PROD_PATH)
    df["Season"] = df["Season"].str.strip()
    df = df.dropna(subset=["Production"])
    df["yield_per_ha"] = df["Production"] / df["Area"]

    q99 = df["yield_per_ha"].quantile(0.99)
    q01 = df["yield_per_ha"].quantile(0.01)
    df = df[(df["yield_per_ha"] >= q01) & (df["yield_per_ha"] <= q99)]

    crop_map = {
        "Rice": "Rice",
        "Wheat": "Wheat",
        "Maize": "Maize",
        "Sugarcane": "Sugarcane",
        "Potato": "Potato",
        "Tomato": "Tomato",
        "Cotton(lint)": "Cotton",
    }
    season_map = {
        "Kharif": "Kharif",
        "Rabi": "Rabi",
        "Summer": "Summer",
        "Zaid": "Kharif",
        "Whole Year": "Kharif",
        "Autumn": "Kharif",
        "Winter": "Rabi",
    }
    state_region = {
        "Tamil Nadu": "South",
        "Karnataka": "South",
        "Kerala": "South",
        "Andhra Pradesh": "South",
        "Telangana": "South",
        "Maharashtra": "Central",
        "Madhya Pradesh": "Central",
        "Chhattisgarh": "Central",
        "Gujarat": "West",
        "Rajasthan": "West",
        "Goa": "West",
        "Punjab": "North",
        "Haryana": "North",
        "Uttar Pradesh": "North",
        "Uttarakhand": "North",
        "Himachal Pradesh": "North",
        "Jammu and Kashmir": "North",
        "West Bengal": "East",
        "Odisha": "East",
        "Jharkhand": "East",
        "Bihar": "East",
        "Assam": "East",
    }

    df = df[df["Crop"].isin(crop_map.keys())].copy()
    df["Crop_std"] = df["Crop"].map(crop_map)
    df["Season_std"] = df["Season"].map(season_map).fillna("Kharif")
    df["Region"] = df["State_Name"].map(state_region).fillna("Central")

    le_crop = LabelEncoder()
    le_season = LabelEncoder()
    le_region = LabelEncoder()

    df["crop_enc"] = le_crop.fit_transform(df["Crop_std"])
    df["season_enc"] = le_season.fit_transform(df["Season_std"])
    df["region_enc"] = le_region.fit_transform(df["Region"])
    df["support_key"] = df["Crop_std"] + "|" + df["Season_std"] + "|" + df["Region"]

    X = df[["crop_enc", "season_enc", "region_enc"]]
    y = df["yield_per_ha"]
    groups = df["State_Name"]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    model = RandomForestRegressor(
        n_estimators=320,
        max_depth=14,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    preds = np.maximum(model.predict(X_test), 0)
    tree_preds = np.array([
        np.maximum(est.predict(X_test), 0)
        for est in model.estimators_
    ])
    pred_lower = np.percentile(tree_preds, 10, axis=0)
    pred_upper = np.percentile(tree_preds, 90, axis=0)
    avg_prediction_spread_t_ha = float(np.mean(pred_upper - pred_lower))

    baseline = DummyRegressor(strategy="mean")
    baseline.fit(X_train, y_train)
    baseline_preds = np.maximum(baseline.predict(X_test), 0)

    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    baseline_r2 = r2_score(y_test, baseline_preds)
    baseline_mae = mean_absolute_error(y_test, baseline_preds)

    support_counts = {
        str(key): int(count)
        for key, count in df["support_key"].value_counts().sort_index().items()
    }
    residuals = np.abs(y_test - preds)
    global_mae = float(mean_absolute_error(y_test, preds))
    p75_residual = float(np.percentile(residuals, 75))
    p90_residual = float(np.percentile(residuals, 90))
    min_support_rows = 120

    yield_meta = {
        "available": True,
        "training_source": os.path.basename(CROP_PROD_PATH),
        "holdout_r2_pct": _pct_or_none(r2),
        "holdout_mae_t_ha": _round_or_none(mae, 4),
        "baseline_r2_pct": _pct_or_none(baseline_r2),
        "baseline_mae_t_ha": _round_or_none(baseline_mae, 4),
        "global_mae_t_ha": _round_or_none(global_mae, 4),
        "p75_residual_t_ha": _round_or_none(p75_residual, 4),
        "p90_residual_t_ha": _round_or_none(p90_residual, 4),
        "avg_prediction_spread_t_ha": _round_or_none(avg_prediction_spread_t_ha, 4),
        "min_support_rows": min_support_rows,
        "supported_crops": list(le_crop.classes_),
        "supported_seasons": list(le_season.classes_),
        "supported_regions": list(le_region.classes_),
        "support_counts": support_counts,
    }

    _joblib_save(model, YIELD_MODEL_PATH)
    with open(YIELD_ENCODERS_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "crops": list(le_crop.classes_),
                "seasons": list(le_season.classes_),
                "regions": list(le_region.classes_),
            },
            f,
        )
    with open(YIELD_META_PATH, "w", encoding="utf-8") as f:
        json.dump(yield_meta, f, indent=2)

    print(f"  Dataset rows:     {len(df):,}")
    print(f"  Holdout states:   {df.iloc[test_idx]['State_Name'].nunique():,}")
    print(f"  Grouped R2:       {_pct_or_none(r2)}%")
    print(f"  Grouped MAE:      {_round_or_none(mae, 4)} t/ha")
    print(f"  Avg spread:       {_round_or_none(avg_prediction_spread_t_ha, 4)} t/ha")
    print("  Saved -> yield_model.pkl + yield_encoders.json + yield_model_meta.json")

    return {
        "dataset_rows": int(len(df)),
        "holdout_states": int(df.iloc[test_idx]["State_Name"].nunique()),
        "grouped_holdout_r2_pct": _pct_or_none(r2),
        "grouped_holdout_mae_t_ha": _round_or_none(mae, 4),
        "baseline_r2_pct": _pct_or_none(baseline_r2),
        "baseline_mae_t_ha": _round_or_none(baseline_mae, 4),
        "avg_prediction_spread_t_ha": _round_or_none(avg_prediction_spread_t_ha, 4),
        "min_support_rows": min_support_rows,
        "validation": "Grouped holdout by state with random-forest uncertainty spread and support-count tracking",
        "note": "Yield now returns an estimate range and support-aware confidence band instead of a fake exact-only value.",
    }


if __name__ == "__main__":
    crop_metrics = train_crop_model()
    fertilizer_metrics = train_fertilizer_model()
    rainfall_metrics = train_rainfall_model()
    yield_metrics = train_yield_model()

    summary = {
        "crop": {
            "available": crop_metrics["available"],
            "validation": crop_metrics["validation"],
            "holdout_accuracy_pct": crop_metrics["holdout_accuracy_pct"],
            "holdout_macro_f1_pct": crop_metrics["holdout_macro_f1_pct"],
            "cv_accuracy_mean_pct": crop_metrics["cv_accuracy_mean_pct"],
            "target_coverage_pct": crop_metrics.get("target_coverage_pct"),
            "app_supported_coverage_pct": crop_metrics["app_supported_coverage_pct"],
            "note": crop_metrics["note"],
        },
        "fertilizer": {
            "available": fertilizer_metrics["available"],
            "validation": fertilizer_metrics["validation"],
            "holdout_accuracy_pct": fertilizer_metrics["holdout_accuracy_pct"],
            "holdout_macro_f1_pct": fertilizer_metrics["holdout_macro_f1_pct"],
            "holdout_balanced_accuracy_pct": fertilizer_metrics["holdout_balanced_accuracy_pct"],
            "reliable_classes": fertilizer_metrics["reliable_classes"],
            "unreliable_classes": fertilizer_metrics["unreliable_classes"],
            "note": fertilizer_metrics["note"],
        },
        "rainfall": {
            "validation": rainfall_metrics["validation"],
            "overall_r2_pct": rainfall_metrics["overall_r2_pct"],
            "overall_mae_mm": rainfall_metrics["overall_mae_mm"],
            "rainy_only_mae_mm": rainfall_metrics["rainy_only_mae_mm"],
            "rain_event_precision_pct": rainfall_metrics["rain_event_precision_pct"],
            "rain_event_recall_pct": rainfall_metrics["rain_event_recall_pct"],
            "meaningful_event_recall_pct": rainfall_metrics["meaningful_event_recall_pct"],
            "meaningful_rain_precision_pct": rainfall_metrics["meaningful_rain_precision_pct"],
            "meaningful_rain_recall_pct": rainfall_metrics["meaningful_rain_recall_pct"],
            "meaningful_rain_f1_pct": rainfall_metrics["meaningful_rain_f1_pct"],
            "note": rainfall_metrics["note"],
        },
        "yield": {
            "validation": yield_metrics["validation"],
            "grouped_holdout_r2_pct": yield_metrics["grouped_holdout_r2_pct"],
            "grouped_holdout_mae_t_ha": yield_metrics["grouped_holdout_mae_t_ha"],
            "note": yield_metrics["note"],
        },
    }

    payload = {
        "generated_by": "train_models.py",
        "summary": summary,
        "details": {
            "crop": crop_metrics,
            "fertilizer": fertilizer_metrics,
            "rainfall": rainfall_metrics,
            "yield": yield_metrics,
        },
    }

    _save_metrics(payload)

    print("\n" + "=" * 60)
    print("  ALL 4 MODELS TRAINED SUCCESSFULLY")
    print("=" * 60 + "\n")


def createFunction():
    print("Hello World Nandha")