import json
import os
import warnings
from typing import Optional

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

SUPPORTED_CROPS = ["Rice", "Wheat", "Maize", "Cotton", "Sugarcane", "Potato", "Tomato"]
SUPPORTED_CROP_SET = set(SUPPORTED_CROPS)

DEFAULT_MIN_HOLDOUT_ACCURACY_PCT = 70.0
DEFAULT_MIN_CV_ACCURACY_PCT = 70.0
DEFAULT_MIN_PREDICTION_CONFIDENCE_PCT = 60.0
DEFAULT_MIN_PROBABILITY_MARGIN_PCT = 8.0

DEFAULT_FERTILIZER_MIN_HOLDOUT_ACCURACY_PCT = 80.0
DEFAULT_FERTILIZER_MIN_HOLDOUT_MACRO_F1_PCT = 65.0
DEFAULT_FERTILIZER_MIN_PREDICTION_CONFIDENCE_PCT = 55.0
DEFAULT_FERTILIZER_MIN_PROBABILITY_MARGIN_PCT = 10.0
DEFAULT_FERTILIZER_MIN_CLASS_PRECISION_PCT = 45.0
DEFAULT_FERTILIZER_MIN_CLASS_RECALL_PCT = 45.0


def _load(filename):
    path = os.path.join(MODEL_DIR, filename)
    try:
        obj = joblib.load(path)
        print(f"Loaded {filename}")
        return obj
    except FileNotFoundError:
        print(f"Not found: {path}. Run train_models.py first.")
        return None
    except Exception as e:
        print(f"Failed to load {filename}: {e}")
        return None


def _load_json(filename):
    path = os.path.join(MODEL_DIR, filename)
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _standardize_crop_name(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = str(value).strip()
    if not cleaned:
        return None
    crop_map = {crop.lower(): crop for crop in SUPPORTED_CROPS}
    return crop_map.get(cleaned.lower(), cleaned.title())


def _normalize_season(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = str(value).strip().lower()
    season_map = {
        "kharif": "Kharif",
        "rabi": "Rabi",
        "zaid": "Zaid",
        "summer": "Zaid",
    }
    return season_map.get(cleaned)


def _normalize_soil_type(value: Optional[str]) -> Optional[str]:
    if value is None:
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


def _normalize_region(value: Optional[str]) -> Optional[str]:
    if value is None:
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


def _get_crop_model_classes():
    if crop_model is None:
        return []

    if hasattr(crop_model, "classes_"):
        return [_standardize_crop_name(v) for v in crop_model.classes_]

    named_steps = getattr(crop_model, "named_steps", {})
    classifier = named_steps.get("classifier")
    if classifier is not None and hasattr(classifier, "classes_"):
        return [_standardize_crop_name(v) for v in classifier.classes_]

    return []


def _get_fertilizer_model_classes():
    if fertilizer_model is None:
        return []

    if hasattr(fertilizer_model, "classes_"):
        return list(fertilizer_model.classes_)

    calibrated_classifiers = getattr(fertilizer_model, "calibrated_classifiers_", None)
    if calibrated_classifiers:
        first = calibrated_classifiers[0]
        estimator = getattr(first, "estimator", None)
        if estimator is not None and hasattr(estimator, "classes_"):
            return list(estimator.classes_)

    return []


def _get_crop_model_status() -> dict:
    meta = crop_model_meta or {}
    available_classes = sorted(set(filter(None, _get_crop_model_classes())))

    if not available_classes:
        available_classes = sorted(set(meta.get("represented_labels", [])))

    holdout_accuracy_pct = meta.get("holdout_accuracy_pct")
    cv_accuracy_mean_pct = meta.get("cv_accuracy_mean_pct")

    min_holdout_accuracy_pct = float(meta.get("min_holdout_accuracy_pct", DEFAULT_MIN_HOLDOUT_ACCURACY_PCT))
    min_cv_accuracy_pct = float(meta.get("min_cv_accuracy_pct", DEFAULT_MIN_CV_ACCURACY_PCT))
    min_prediction_confidence_pct = float(
        meta.get("min_prediction_confidence_pct", DEFAULT_MIN_PREDICTION_CONFIDENCE_PCT)
    )
    min_probability_margin_pct = float(
        meta.get("min_probability_margin_pct", DEFAULT_MIN_PROBABILITY_MARGIN_PCT)
    )

    status = {
        "available": True,
        "reason": None,
        "reason_detail": None,
        "holdout_accuracy_pct": holdout_accuracy_pct,
        "cv_accuracy_mean_pct": cv_accuracy_mean_pct,
        "min_holdout_accuracy_pct": min_holdout_accuracy_pct,
        "min_cv_accuracy_pct": min_cv_accuracy_pct,
        "min_prediction_confidence_pct": min_prediction_confidence_pct,
        "min_probability_margin_pct": min_probability_margin_pct,
        "model_supported_crops": available_classes,
        "training_source": meta.get("training_source"),
        "numeric_features": meta.get("numeric_features", []),
        "categorical_features": meta.get("categorical_features", []),
    }

    if crop_model is None:
        status["available"] = False
        status["reason"] = "model_not_loaded"
        status["reason_detail"] = "Crop-fit model file is not loaded."
        return status

    if not meta:
        status["available"] = False
        status["reason"] = "meta_missing"
        status["reason_detail"] = "Crop-fit model metadata is missing."
        return status

    if not meta.get("available", False):
        status["available"] = False
        status["reason"] = meta.get("reason", "model_disabled")
        status["reason_detail"] = meta.get(
            "reason_detail",
            "Crop-fit model is disabled because the training dataset is not ready.",
        )
        return status

    if holdout_accuracy_pct is None or cv_accuracy_mean_pct is None:
        status["available"] = False
        status["reason"] = "metrics_missing"
        status["reason_detail"] = "Crop-fit model metrics are missing."
        return status

    if (
        float(holdout_accuracy_pct) < min_holdout_accuracy_pct
        or float(cv_accuracy_mean_pct) < min_cv_accuracy_pct
    ):
        status["available"] = False
        status["reason"] = "metrics_below_threshold"
        status["reason_detail"] = (
            "Crop-fit model did not pass reliability gates, so predictions are disabled."
        )
        return status

    return status


def _get_fertilizer_model_status() -> dict:
    meta = fertilizer_model_meta or {}
    model_classes = _get_fertilizer_model_classes()
    if not model_classes:
        model_classes = meta.get("model_classes", [])

    holdout_accuracy_pct = meta.get("holdout_accuracy_pct")
    holdout_macro_f1_pct = meta.get("holdout_macro_f1_pct")

    min_holdout_accuracy_pct = float(
        meta.get("min_holdout_accuracy_pct", DEFAULT_FERTILIZER_MIN_HOLDOUT_ACCURACY_PCT)
    )
    min_holdout_macro_f1_pct = float(
        meta.get("min_holdout_macro_f1_pct", DEFAULT_FERTILIZER_MIN_HOLDOUT_MACRO_F1_PCT)
    )
    min_prediction_confidence_pct = float(
        meta.get("min_prediction_confidence_pct", DEFAULT_FERTILIZER_MIN_PREDICTION_CONFIDENCE_PCT)
    )
    min_probability_margin_pct = float(
        meta.get("min_probability_margin_pct", DEFAULT_FERTILIZER_MIN_PROBABILITY_MARGIN_PCT)
    )
    min_class_precision_pct = float(
        meta.get("min_class_precision_pct", DEFAULT_FERTILIZER_MIN_CLASS_PRECISION_PCT)
    )
    min_class_recall_pct = float(
        meta.get("min_class_recall_pct", DEFAULT_FERTILIZER_MIN_CLASS_RECALL_PCT)
    )

    status = {
        "available": True,
        "reason": None,
        "reason_detail": None,
        "holdout_accuracy_pct": holdout_accuracy_pct,
        "holdout_macro_f1_pct": holdout_macro_f1_pct,
        "min_holdout_accuracy_pct": min_holdout_accuracy_pct,
        "min_holdout_macro_f1_pct": min_holdout_macro_f1_pct,
        "min_prediction_confidence_pct": min_prediction_confidence_pct,
        "min_probability_margin_pct": min_probability_margin_pct,
        "min_class_precision_pct": min_class_precision_pct,
        "min_class_recall_pct": min_class_recall_pct,
        "model_classes": model_classes,
        "family_model_classes": meta.get("family_model_classes", []),
        "reliable_classes": meta.get("reliable_classes", []),
        "unreliable_classes": meta.get("unreliable_classes", []),
        "feature_columns": meta.get("feature_columns", []),
        "per_class_summary": meta.get("per_class_summary", {}),
        "fertilizer_family_map": meta.get("fertilizer_family_map", {}),
        "family_fallback_product_map": meta.get("family_fallback_product_map", {}),
        "family_holdout_accuracy_pct": meta.get("family_holdout_accuracy_pct"),
        "family_holdout_macro_f1_pct": meta.get("family_holdout_macro_f1_pct"),
    }

    if fertilizer_model is None or fertilizer_family_model is None:
        status["available"] = False
        status["reason"] = "model_not_loaded"
        status["reason_detail"] = "Fertilizer model files are not loaded."
        return status

    if not meta:
        status["available"] = False
        status["reason"] = "meta_missing"
        status["reason_detail"] = "Fertilizer model metadata is missing."
        return status

    if not meta.get("available", False):
        status["available"] = False
        status["reason"] = meta.get("reason", "model_disabled")
        status["reason_detail"] = meta.get(
            "reason_detail",
            "Fertilizer model is disabled because it did not pass reliability gates.",
        )
        return status

    if holdout_accuracy_pct is None or holdout_macro_f1_pct is None:
        status["available"] = False
        status["reason"] = "metrics_missing"
        status["reason_detail"] = "Fertilizer model metrics are missing."
        return status

    if (
        float(holdout_accuracy_pct) < min_holdout_accuracy_pct
        or float(holdout_macro_f1_pct) < min_holdout_macro_f1_pct
    ):
        status["available"] = False
        status["reason"] = "metrics_below_threshold"
        status["reason_detail"] = (
            "Fertilizer model did not pass reliability gates, so predictions are disabled."
        )
        return status

    return status


def _build_crop_input_df(
    N,
    P,
    K,
    temperature,
    humidity,
    ph,
    rainfall,
    season: Optional[str] = None,
    soil_type: Optional[str] = None,
    region: Optional[str] = None,
) -> pd.DataFrame:
    numeric_features = crop_model_meta.get("numeric_features", []) if crop_model_meta else []
    categorical_features = crop_model_meta.get("categorical_features", []) if crop_model_meta else []

    row = {
        "n": float(N),
        "p": float(P),
        "k": float(K),
        "temperature": float(temperature),
        "humidity": float(humidity),
        "ph": float(ph),
        "rainfall": float(rainfall),
        "season": _normalize_season(season),
        "soil_type": _normalize_soil_type(soil_type),
        "region": _normalize_region(region),
    }

    feature_names = numeric_features + categorical_features
    if not feature_names:
        feature_names = ["n", "p", "k", "temperature", "humidity", "ph", "season", "soil_type"]

    return pd.DataFrame([{name: row.get(name) for name in feature_names}])


crop_model = _load("crop_model.pkl")
crop_model_meta = _load_json("crop_model_meta.json")
fertilizer_model = _load("fertilizer_model.pkl")
fertilizer_family_model = _load("fertilizer_family_model.pkl")
fertilizer_model_meta = _load_json("fertilizer_model_meta.json")
rainfall_model = _load("rainfall_model.pkl")
rainfall_event_model = _load("rainfall_event_model.pkl")
rainfall_meaningful_event_model = _load("rainfall_meaningful_event_model.pkl")
yield_model = _load("yield_model.pkl")
yield_encoders = _load_json("yield_encoders.json")
yield_model_meta = _load_json("yield_model_meta.json")
rainfall_features = _load_json("rainfall_features.json")


_FERTILIZER_CATEGORIES = {
    "Soil_Type": ["Clay", "Loamy", "Sandy", "Silt"],
    "Crop_Type": ["Cotton", "Maize", "Potato", "Rice", "Sugarcane", "Tomato", "Wheat"],
    "Crop_Growth_Stage": ["Flowering", "Harvest", "Sowing", "Vegetative"],
    "Season": ["Kharif", "Rabi", "Zaid"],
    "Irrigation_Type": ["Canal", "Drip", "Rainfed", "Sprinkler"],
    "Previous_Crop": ["Cotton", "Maize", "Potato", "Rice", "Sugarcane", "Tomato", "Wheat"],
    "Region": ["Central", "East", "North", "South", "West"],
}


def _encode_fertilizer_input(
    soil_type, crop_type, growth_stage, season,
    irrigation_type, previous_crop, region, soil_ph, soil_moisture,
    organic_carbon, electrical_conductivity, nitrogen_level,
    phosphorus_level, potassium_level, temperature, humidity, rainfall,
    fertilizer_used_last_season=100.0, yield_last_season=3.0
) -> pd.DataFrame:
    row = {
        "Soil_pH": soil_ph,
        "Soil_Moisture": soil_moisture,
        "Organic_Carbon": organic_carbon,
        "Electrical_Conductivity": electrical_conductivity,
        "Nitrogen_Level": nitrogen_level,
        "Phosphorus_Level": phosphorus_level,
        "Potassium_Level": potassium_level,
        "Temperature": temperature,
        "Humidity": humidity,
        "Rainfall": rainfall,
        "Fertilizer_Used_Last_Season": fertilizer_used_last_season,
        "Yield_Last_Season": yield_last_season,
    }
    cat_vals = {
        "Soil_Type": soil_type,
        "Crop_Type": crop_type,
        "Crop_Growth_Stage": growth_stage,
        "Season": season,
        "Irrigation_Type": irrigation_type,
        "Previous_Crop": previous_crop,
        "Region": region,
    }
    for col, options in _FERTILIZER_CATEGORIES.items():
        selected = cat_vals[col]
        for opt in options:
            row[f"{col}_{opt}"] = 1 if selected == opt else 0
    return pd.DataFrame([row])


def predict_crop(
    N,
    P,
    K,
    temperature,
    humidity,
    ph,
    rainfall,
    season: Optional[str] = None,
    soil_type: Optional[str] = None,
    region: Optional[str] = None,
    selected_crop: Optional[str] = None,
) -> dict:
    status = _get_crop_model_status()

    base_result = {
        "recommended_crop": None,
        "confidence": None,
        "top_alternatives": [],
        "supported_crops": SUPPORTED_CROPS,
        "model_supported_crops": status["model_supported_crops"],
        "crop_fit_available": False,
        "model_status": status,
    }

    selected_crop_std = _standardize_crop_name(selected_crop) if selected_crop is not None else None

    if not status["available"]:
        if selected_crop is not None:
            base_result["crop_fit_analysis"] = {
                "available": False,
                "selected_crop": selected_crop_std or selected_crop,
                "recommended_crop": None,
                "selected_crop_matches_model": None,
                "selected_crop_in_model": selected_crop_std in status["model_supported_crops"] if selected_crop_std else False,
                "selected_crop_probability_pct": None,
                "recommended_crop_probability_pct": None,
                "message": status["reason_detail"],
                "reason": status["reason"],
                "reason_detail": status["reason_detail"],
                "top_alternatives": [],
                "model_supported_crops": status["model_supported_crops"],
                "holdout_accuracy_pct": status["holdout_accuracy_pct"],
                "cv_accuracy_mean_pct": status["cv_accuracy_mean_pct"],
            }
        else:
            base_result["message"] = status["reason_detail"]
        return base_result

    try:
        df = _build_crop_input_df(
            N=N,
            P=P,
            K=K,
            temperature=temperature,
            humidity=humidity,
            ph=ph,
            rainfall=rainfall,
            season=season,
            soil_type=soil_type,
            region=region,
        )

        probas = crop_model.predict_proba(df)[0]
        available_classes = _get_crop_model_classes()
        if not available_classes:
            available_classes = status["model_supported_crops"]

        ranked = sorted(zip(available_classes, probas), key=lambda x: -x[1])

        top3 = ranked[:3]
        best_crop, best_prob = ranked[0]
        second_prob = ranked[1][1] if len(ranked) > 1 else 0.0

        best_prob_pct = round(float(best_prob) * 100, 1)
        margin_pct = round(float(best_prob - second_prob) * 100, 1)

        low_confidence = best_prob_pct < status["min_prediction_confidence_pct"]
        low_margin = margin_pct < status["min_probability_margin_pct"]

        if low_confidence or low_margin:
            reason_detail = (
                "Crop-fit model is not confident enough for this input, so the result is withheld "
                "instead of showing a weak recommendation."
            )

            base_result.update({
                "recommended_crop": None,
                "confidence": None,
                "top_alternatives": [
                    {"crop": crop, "probability": round(float(prob) * 100, 1)}
                    for crop, prob in top3
                ],
                "crop_fit_available": False,
                "message": reason_detail,
            })

            if selected_crop is not None:
                selected_prob = None
                for crop_name, prob in ranked:
                    if crop_name == selected_crop_std:
                        selected_prob = prob
                        break

                base_result["crop_fit_analysis"] = {
                    "available": False,
                    "selected_crop": selected_crop_std or selected_crop,
                    "recommended_crop": None,
                    "selected_crop_matches_model": None,
                    "selected_crop_in_model": selected_crop_std in available_classes if selected_crop_std else False,
                    "selected_crop_probability_pct": round(float(selected_prob) * 100, 1) if selected_prob is not None else None,
                    "recommended_crop_probability_pct": best_prob_pct,
                    "message": reason_detail,
                    "reason": "low_prediction_confidence",
                    "reason_detail": reason_detail,
                    "top_alternatives": [
                        {"crop": crop, "probability": round(float(prob) * 100, 1)}
                        for crop, prob in top3
                    ],
                    "model_supported_crops": available_classes,
                    "holdout_accuracy_pct": status["holdout_accuracy_pct"],
                    "cv_accuracy_mean_pct": status["cv_accuracy_mean_pct"],
                    "top_confidence_pct": best_prob_pct,
                    "probability_margin_pct": margin_pct,
                }
            return base_result

        base_result.update({
            "recommended_crop": best_crop,
            "confidence": best_prob_pct,
            "top_alternatives": [
                {"crop": crop, "probability": round(float(prob) * 100, 1)}
                for crop, prob in top3
            ],
            "crop_fit_available": True,
        })

        if selected_crop is not None:
            selected_prob = None
            for crop_name, prob in ranked:
                if crop_name == selected_crop_std:
                    selected_prob = prob
                    break

            selected_in_app = selected_crop_std in SUPPORTED_CROP_SET if selected_crop_std else False
            selected_in_model = selected_crop_std in available_classes if selected_crop_std else False
            selected_matches_model = bool(selected_crop_std) and best_crop == selected_crop_std

            if not selected_in_app:
                message = "The selected crop is not supported by this app."
            elif not selected_in_model:
                message = "The selected crop is not represented in the currently trained crop-fit model."
            elif selected_matches_model:
                message = "Your selected crop looks suitable for these farm conditions."
            else:
                message = "Your selected crop may not be the best fit for these farm conditions."

            base_result["crop_fit_analysis"] = {
                "available": True,
                "selected_crop": selected_crop_std or selected_crop,
                "recommended_crop": best_crop,
                "selected_crop_matches_model": selected_matches_model,
                "selected_crop_in_model": selected_in_model,
                "selected_crop_probability_pct": round(float(selected_prob) * 100, 1) if selected_prob is not None else None,
                "recommended_crop_probability_pct": best_prob_pct,
                "message": message,
                "reason": None,
                "reason_detail": None,
                "top_alternatives": [
                    {"crop": crop, "probability": round(float(prob) * 100, 1)}
                    for crop, prob in top3
                ],
                "model_supported_crops": available_classes,
                "holdout_accuracy_pct": status["holdout_accuracy_pct"],
                "cv_accuracy_mean_pct": status["cv_accuracy_mean_pct"],
                "top_confidence_pct": best_prob_pct,
                "probability_margin_pct": margin_pct,
            }

        return base_result
    except Exception as e:
        return {"error": f"Crop prediction failed: {e}"}


def predict_fertilizer(
    soil_type, crop_type, growth_stage, season,
    irrigation_type="Canal", previous_crop="Wheat", region="South",
    soil_ph=6.5, soil_moisture=40.0, organic_carbon=0.5,
    electrical_conductivity=0.8, nitrogen_level=50.0, phosphorus_level=40.0,
    potassium_level=40.0, temperature=25.0, humidity=60.0, rainfall=100.0,
    fertilizer_used_last_season=100.0, yield_last_season=3.0
) -> dict:
    status = _get_fertilizer_model_status()
    if not status["available"]:
        return {"error": status["reason_detail"]}

    sm = {s.lower(): s for s in _FERTILIZER_CATEGORIES["Soil_Type"]}
    cm = {c.lower(): c for c in _FERTILIZER_CATEGORIES["Crop_Type"]}
    gm = {g.lower(): g for g in _FERTILIZER_CATEGORIES["Crop_Growth_Stage"]}
    sn = {s.lower(): s for s in _FERTILIZER_CATEGORIES["Season"]}
    im = {i.lower(): i for i in _FERTILIZER_CATEGORIES["Irrigation_Type"]}
    pm = {p.lower(): p for p in _FERTILIZER_CATEGORIES["Previous_Crop"]}
    rm = {r.lower(): r for r in _FERTILIZER_CATEGORIES["Region"]}

    soil_type = sm.get(str(soil_type).lower(), "Clay")
    crop_type = cm.get(str(crop_type).lower(), "Rice")
    growth_stage = gm.get(str(growth_stage).lower(), "Vegetative")
    season = sn.get(str(season).lower(), "Kharif")
    irrigation_type = im.get(str(irrigation_type).lower(), "Canal")
    previous_crop = pm.get(str(previous_crop).lower(), "Wheat")
    region = rm.get(str(region).lower(), "South")

    try:
        df = _encode_fertilizer_input(
            soil_type=soil_type,
            crop_type=crop_type,
            growth_stage=growth_stage,
            season=season,
            irrigation_type=irrigation_type,
            previous_crop=previous_crop,
            region=region,
            soil_ph=soil_ph,
            soil_moisture=soil_moisture,
            organic_carbon=organic_carbon,
            electrical_conductivity=electrical_conductivity,
            nitrogen_level=nitrogen_level,
            phosphorus_level=phosphorus_level,
            potassium_level=potassium_level,
            temperature=temperature,
            humidity=humidity,
            rainfall=rainfall,
            fertilizer_used_last_season=fertilizer_used_last_season,
            yield_last_season=yield_last_season,
        )

        feature_columns = status["feature_columns"]
        if feature_columns:
            df = df.reindex(columns=feature_columns, fill_value=0)

        family_classes = list(getattr(fertilizer_family_model, "classes_", [])) or status["family_model_classes"]
        family_probas = fertilizer_family_model.predict_proba(df)[0]
        ranked_families = sorted(zip(family_classes, family_probas), key=lambda x: -x[1])
        family_label, family_prob = ranked_families[0]
        family_second_prob = ranked_families[1][1] if len(ranked_families) > 1 else 0.0
        family_confidence_pct = round(float(family_prob) * 100, 1)
        family_margin_pct = round(float(family_prob - family_second_prob) * 100, 1)

        product_probas = fertilizer_model.predict_proba(df)[0]
        classes_ = _get_fertilizer_model_classes() or status["model_classes"]
        ranked_products = sorted(zip(classes_, product_probas), key=lambda x: -x[1])
        top3 = ranked_products[:3]

        best_label, best_prob = ranked_products[0]
        second_prob = ranked_products[1][1] if len(ranked_products) > 1 else 0.0

        best_prob_pct = round(float(best_prob) * 100, 1)
        margin_pct = round(float(best_prob - second_prob) * 100, 1)

        class_metrics = status["per_class_summary"].get(best_label, {})
        class_precision_pct = float(class_metrics.get("precision_pct") or 0.0)
        class_recall_pct = float(class_metrics.get("recall_pct") or 0.0)
        predicted_family_from_product = status["fertilizer_family_map"].get(best_label)
        safe_family_product = status["family_fallback_product_map"].get(family_label, "NPK")

        family_low_confidence = family_confidence_pct < 50.0 or family_margin_pct < 8.0
        low_confidence = best_prob_pct < status["min_prediction_confidence_pct"]
        low_margin = margin_pct < status["min_probability_margin_pct"]
        weak_class = (
            best_label in status["unreliable_classes"]
            or class_precision_pct < status["min_class_precision_pct"]
            or class_recall_pct < status["min_class_recall_pct"]
        )
        family_mismatch = bool(predicted_family_from_product) and predicted_family_from_product != family_label

        if family_low_confidence:
            return {
                "error": "Fertilizer recommendation withheld because even the nutrient-family prediction is not confident enough.",
                "reason": "family_low_confidence",
                "recommended_fertilizer": None,
                "recommended_nutrient_family": None,
                "confidence": None,
                "top_alternatives": [
                    {"fertilizer": label, "probability": round(float(prob) * 100, 1)}
                    for label, prob in top3
                ],
            }

        if low_confidence or low_margin or weak_class or family_mismatch:
            return {
                "recommended_fertilizer": safe_family_product,
                "recommended_nutrient_family": family_label,
                "decision_mode": "safe_family_fallback",
                "confidence": family_confidence_pct,
                "probability_margin_pct": family_margin_pct,
                "top_alternatives": [
                    {"fertilizer": label, "probability": round(float(prob) * 100, 1)}
                    for label, prob in top3
                ],
                "model_status": {
                    "available": True,
                    "holdout_accuracy_pct": status["holdout_accuracy_pct"],
                    "holdout_macro_f1_pct": status["holdout_macro_f1_pct"],
                    "family_holdout_accuracy_pct": status["family_holdout_accuracy_pct"],
                    "family_holdout_macro_f1_pct": status["family_holdout_macro_f1_pct"],
                    "class_precision_pct": class_precision_pct,
                    "class_recall_pct": class_recall_pct,
                },
                "fallback_reason": (
                    "Exact fertilizer product was not reliable enough, so the system returned a safe product within the predicted nutrient family."
                ),
            }

        return {
            "recommended_fertilizer": best_label,
            "recommended_nutrient_family": predicted_family_from_product or family_label,
            "decision_mode": "exact_product",
            "confidence": best_prob_pct,
            "probability_margin_pct": margin_pct,
            "top_alternatives": [
                {"fertilizer": label, "probability": round(float(prob) * 100, 1)}
                for label, prob in top3
            ],
            "model_status": {
                "available": True,
                "holdout_accuracy_pct": status["holdout_accuracy_pct"],
                "holdout_macro_f1_pct": status["holdout_macro_f1_pct"],
                "family_holdout_accuracy_pct": status["family_holdout_accuracy_pct"],
                "family_holdout_macro_f1_pct": status["family_holdout_macro_f1_pct"],
                "class_precision_pct": class_precision_pct,
                "class_recall_pct": class_recall_pct,
            },
            "fallback_reason": None,
        }
    except Exception as e:
        return {"error": f"Fertilizer prediction failed: {e}"}


def _build_rainfall_input_frame(
    temperature_celsius: float,
    humidity: float,
    pressure_mb: float,
    cloud: float,
    wind_kph: float,
    gust_kph: float,
    visibility_km: float,
    uv_index: float,
    wind_degree: float,
    feels_like_celsius: float,
    wind_dir_enc: int,
    dew_point: float,
    feels_delta: float,
    gust_ratio: float,
    vis_inv: float,
    humid_cloud: float,
) -> pd.DataFrame:
    row = {
        "temperature_celsius": float(temperature_celsius),
        "humidity": float(humidity),
        "pressure_mb": float(pressure_mb),
        "cloud": float(cloud),
        "wind_kph": float(wind_kph),
        "gust_kph": float(gust_kph),
        "visibility_km": float(visibility_km),
        "uv_index": float(uv_index),
        "wind_degree": float(wind_degree),
        "feels_like_celsius": float(feels_like_celsius),
        "wind_dir_enc": float(wind_dir_enc),
        "dew_point": float(dew_point),
        "feels_delta": float(feels_delta),
        "gust_ratio": float(gust_ratio),
        "vis_inv": float(vis_inv),
        "humid_cloud": float(humid_cloud),
    }

    radians = np.deg2rad(row["wind_degree"])
    row["wind_u"] = row["wind_kph"] * np.cos(radians)
    row["wind_v"] = row["wind_kph"] * np.sin(radians)
    row["gust_delta"] = row["gust_kph"] - row["wind_kph"]
    row["temp_humidity"] = row["temperature_celsius"] * row["humidity"] / 100
    row["pressure_cloud"] = row["pressure_mb"] * row["cloud"] / 100

    sat_vapor_pressure = 0.6108 * np.exp((17.27 * row["temperature_celsius"]) / (row["temperature_celsius"] + 237.3))
    actual_vapor_pressure = sat_vapor_pressure * row["humidity"] / 100
    row["vpd"] = max(float(sat_vapor_pressure - actual_vapor_pressure), 0.0)
    row["dryness_index"] = row["temperature_celsius"] * (100 - row["humidity"]) / 100
    row["uv_clear_sky"] = row["uv_index"] * (100 - row["cloud"]) / 100

    features = (rainfall_features or {}).get("features", [
        "temperature_celsius", "humidity", "pressure_mb", "cloud", "wind_kph",
        "gust_kph", "visibility_km", "uv_index", "wind_degree", "feels_like_celsius", "wind_dir_enc",
        "dew_point", "feels_delta", "gust_ratio", "vis_inv", "humid_cloud",
        "wind_u", "wind_v", "gust_delta", "temp_humidity", "pressure_cloud",
        "vpd", "dryness_index", "uv_clear_sky",
    ])

    return pd.DataFrame([[row[f] for f in features]], columns=features)


def predict_rainfall(
    temperature_celsius: float,
    humidity: float,
    pressure_mb: float,
    cloud: float,
    wind_kph: float,
    gust_kph: float = None,
    visibility_km: float = 10.0,
    uv_index: float = 0,
    wind_degree: float = 180,
    feels_like_celsius: float = None,
    wind_dir_enc: int = 8,
    dew_point: float = None,
    feels_delta: float = None,
    gust_ratio: float = None,
    vis_inv: float = None,
    humid_cloud: float = None,
) -> dict:
    if rainfall_model is None:
        return {"error": "Rainfall model not loaded."}

    if gust_kph is None:
        gust_kph = wind_kph * 1.3
    if feels_like_celsius is None:
        feels_like_celsius = temperature_celsius
    if dew_point is None:
        dew_point = temperature_celsius - ((100 - humidity) / 5)
    if feels_delta is None:
        feels_delta = feels_like_celsius - temperature_celsius
    if gust_ratio is None:
        gust_ratio = gust_kph / (wind_kph + 0.1)
    if vis_inv is None:
        vis_inv = 1 / (visibility_km + 0.1)
    if humid_cloud is None:
        humid_cloud = humidity * cloud / 100

    df = _build_rainfall_input_frame(
        temperature_celsius=temperature_celsius,
        humidity=humidity,
        pressure_mb=pressure_mb,
        cloud=cloud,
        wind_kph=wind_kph,
        gust_kph=gust_kph,
        visibility_km=visibility_km,
        uv_index=uv_index,
        wind_degree=wind_degree,
        feels_like_celsius=feels_like_celsius,
        wind_dir_enc=wind_dir_enc,
        dew_point=dew_point,
        feels_delta=feels_delta,
        gust_ratio=gust_ratio,
        vis_inv=vis_inv,
        humid_cloud=humid_cloud,
    )
    try:
        rainfall_strategy = (rainfall_features or {}).get("strategy", "single_stage_amount_only")
        raw_amount_pred = float(rainfall_model.predict(df)[0])
        rainy_amount_mm = (
            max(float(np.expm1(raw_amount_pred)), 0.0)
            if rainfall_strategy in {"two_stage_event_plus_amount", "three_stage_event_meaningful_amount", "three_stage_event_meaningful_amount_v2"}
            else max(raw_amount_pred, 0.0)
        )

        if rainfall_event_model is not None and rainfall_strategy in {"two_stage_event_plus_amount", "three_stage_event_meaningful_amount", "three_stage_event_meaningful_amount_v2"}:
            rain_probability = float(rainfall_event_model.predict_proba(df)[0][1])
            threshold = float((rainfall_features or {}).get("event_threshold", 0.35))
            meaningful_rain_probability = None

            if rainfall_strategy in {"three_stage_event_meaningful_amount", "three_stage_event_meaningful_amount_v2"} and rainfall_meaningful_event_model is not None:
                meaningful_rain_probability = float(rainfall_meaningful_event_model.predict_proba(df)[0][1])
                meaningful_threshold = float((rainfall_features or {}).get("meaningful_event_threshold", 0.28))
                combo = (rainfall_features or {}).get("combination_params", {})
                low_rain_scale = float(combo.get("low_rain_scale", 0.30))
                low_rain_floor = float(combo.get("low_rain_floor", 0.02))
                meaningful_boost = float(combo.get("meaningful_boost", 2.0))
                moderate_base = float(combo.get("moderate_base", 0.50))
                moderate_slope = float(combo.get("moderate_slope", 0.40))
                moderate_cap = float(combo.get("moderate_cap", 0.90))

                predicted_mm = rainy_amount_mm
                if rain_probability < threshold:
                    predicted_mm = rainy_amount_mm * max(rain_probability * low_rain_scale, low_rain_floor)
                elif meaningful_rain_probability >= meaningful_threshold:
                    predicted_mm = max(rainy_amount_mm, meaningful_rain_probability * meaningful_boost)
                else:
                    predicted_mm = rainy_amount_mm * min(
                        max(moderate_base + moderate_slope * rain_probability, moderate_base),
                        moderate_cap,
                    )
            else:
                predicted_mm = (
                    rainy_amount_mm
                    if rain_probability >= threshold
                    else rainy_amount_mm * rain_probability * 0.35
                )
            strategy = rainfall_strategy
        else:
            rain_probability = 100.0 if rainy_amount_mm >= 0.1 else 0.0
            meaningful_rain_probability = 100.0 if rainy_amount_mm >= 1.0 else 0.0
            predicted_mm = rainy_amount_mm
            strategy = "single_stage_amount_only"

        if rainfall_event_model is None:
            rain_probability_pct = round(rain_probability, 1)
        else:
            rain_probability_pct = round(rain_probability * 100.0, 1)

        if meaningful_rain_probability is None:
            meaningful_probability_pct = 100.0 if rainy_amount_mm >= 1.0 else 0.0
        elif rainfall_meaningful_event_model is not None and isinstance(meaningful_rain_probability, float) and meaningful_rain_probability <= 1.0:
            meaningful_probability_pct = round(meaningful_rain_probability * 100.0, 1)
        else:
            meaningful_probability_pct = round(float(meaningful_rain_probability), 1)

        if meaningful_probability_pct >= 75.0 or rain_probability_pct >= 85.0:
            confidence_band = "high"
        elif meaningful_probability_pct >= 45.0 or rain_probability_pct >= 60.0:
            confidence_band = "medium"
        else:
            confidence_band = "low"

        return {
            "predicted_rainfall_mm": round(max(predicted_mm, 0.0), 2),
            "rain_probability_pct": rain_probability_pct,
            "meaningful_rain_probability_pct": meaningful_probability_pct,
            "rainy_amount_if_rain_mm": round(rainy_amount_mm, 2),
            "confidence_band": confidence_band,
            "model_strategy": strategy,
        }
    except Exception as e:
        return {"error": f"Rainfall prediction failed: {e}"}


def predict_yield(crop: str, season: str, region: str = "South") -> dict:
    if yield_model is None or yield_encoders is None or yield_model_meta is None:
        return {"error": "Yield model not loaded. Run train_models.py first."}

    crops = yield_encoders["crops"]
    seasons = yield_encoders["seasons"]
    regions = yield_encoders["regions"]

    crop_std = crop.strip().title()
    season_std = {
        "Zaid": "Kharif",
        "Summer": "Kharif",
        "Whole Year": "Kharif",
        "Autumn": "Kharif",
        "Winter": "Rabi",
    }.get(season.strip().title(), season.strip().title())
    region_std = region.strip().title()

    crop_enc = crops.index(crop_std) if crop_std in crops else 0
    season_enc = seasons.index(season_std) if season_std in seasons else 0
    region_enc = regions.index(region_std) if region_std in regions else 0

    try:
        support_key = f"{crop_std}|{season_std}|{region_std}"
        support_rows = int((yield_model_meta.get("support_counts") or {}).get(support_key, 0))
        min_support_rows = int(yield_model_meta.get("min_support_rows", 120))

        if support_rows < min_support_rows:
            return {
                "error": "Yield prediction withheld because this crop-season-region combination is weakly supported by training data.",
                "crop": crop_std,
                "season": season_std,
                "region": region_std,
                "support_rows": support_rows,
                "minimum_support_rows": min_support_rows,
            }

        feature_row = [[crop_enc, season_enc, region_enc]]
        t_per_ha = max(float(yield_model.predict(feature_row)[0]), 0.0)

        estimators = getattr(yield_model, "estimators_", [])
        if estimators:
            tree_preds = np.array([
                max(float(est.predict(feature_row)[0]), 0.0)
                for est in estimators
            ])
            lower = max(float(np.percentile(tree_preds, 10)), 0.0)
            upper = max(float(np.percentile(tree_preds, 90)), lower)
        else:
            pad = float(yield_model_meta.get("global_mae_t_ha", 1.5))
            lower = max(t_per_ha - pad, 0.0)
            upper = t_per_ha + pad

        spread = upper - lower
        if support_rows >= min_support_rows * 2 and spread <= 2.0:
            confidence_band = "high"
            support_level = "strong"
        elif spread <= 4.0:
            confidence_band = "medium"
            support_level = "moderate"
        else:
            confidence_band = "low"
            support_level = "limited"

        quintals_acre = round(t_per_ha * 4.04685, 1)
        return {
            "crop": crop_std,
            "season": season_std,
            "region": region_std,
            "yield_tonnes_per_ha": round(t_per_ha, 2),
            "yield_range_tonnes_per_ha": [round(lower, 2), round(upper, 2)],
            "yield_quintals_per_acre": quintals_acre,
            "yield_range_quintals_per_acre": [round(lower * 4.04685, 1), round(upper * 4.04685, 1)],
            "confidence_band": confidence_band,
            "support_level": support_level,
            "support_rows": support_rows,
            "minimum_support_rows": min_support_rows,
            "data_source": "Indian govt crop production data (1997-2015, 246K records)",
            "supported": crop_std in crops,
        }
    except Exception as e:
        return {"error": f"Yield prediction failed: {e}"}
