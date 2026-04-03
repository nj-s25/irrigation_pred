import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

MODEL_FILES = {
    "lgb": "lgb_model.pkl",
    "xgb": "xgb_model.pkl",
    "cat": "cat_model.pkl",
    "label_encoders": "label_encoders.pkl",
    "target_encoder": "target_encoder.pkl",
}

NUMERIC_FIELDS = [
    ("Soil_pH", 6.5),
    ("Soil_Moisture", 30.0),
    ("Organic_Carbon", 1.0),
    ("Electrical_Conductivity", 0.5),
    ("Temperature_C", 25.0),
    ("Humidity", 60.0),
    ("Rainfall_mm", 50.0),
    ("Sunlight_Hours", 8.0),
    ("Wind_Speed_kmh", 10.0),
    ("Field_Area_hectare", 1.0),
    ("Previous_Irrigation_mm", 10.0),
]

CATEGORICAL_FIELDS = [
    "Soil_Type",
    "Crop_Type",
    "Crop_Growth_Stage",
    "Season",
    "Irrigation_Type",
    "Water_Source",
    "Mulching_Used",
    "Region",
]


@st.cache_resource(show_spinner=False)
def load_artifacts() -> Dict[str, Any]:
    artifacts: Dict[str, Any] = {}
    missing_files: List[str] = []

    for key, filename in MODEL_FILES.items():
        path = Path(filename)
        if not path.exists():
            missing_files.append(filename)
            continue
        with path.open("rb") as f:
            artifacts[key] = pickle.load(f)

    if missing_files:
        raise FileNotFoundError(f"Missing required files: {', '.join(missing_files)}")

    return artifacts


def get_encoder_classes(label_encoders: Dict[str, Any], feature: str) -> Optional[List[str]]:
    encoder = label_encoders.get(feature)
    if encoder is None or not hasattr(encoder, "classes_"):
        return None
    return [str(x) for x in encoder.classes_]


def render_sidebar() -> None:
    st.sidebar.header("How it works")
    st.sidebar.markdown(
        """
        This app predicts **Irrigation Need** using an ensemble of three models:
        - LightGBM
        - XGBoost
        - CatBoost

        Steps:
        1. Collect field and crop information
        2. Apply the same feature engineering used in training
        3. Encode categorical fields with saved encoders
        4. Average class probabilities from all models
        5. Return final prediction and confidence scores
        """
    )


def render_input_form(label_encoders: Dict[str, Any]) -> Dict[str, Any]:
    defaults = {"Mulching_Used": ["Yes", "No"]}
    user_input: Dict[str, Any] = {}

    with st.form("prediction_form"):
        st.subheader("Input Features")

        col1, col2 = st.columns(2)

        with col1:
            for idx, feature in enumerate(CATEGORICAL_FIELDS):
                classes = get_encoder_classes(label_encoders, feature)
                options = classes if classes else defaults.get(feature, ["Unknown"])
                user_input[feature] = st.selectbox(feature, options=options, index=0, key=f"cat_{idx}")

        with col2:
            for idx, (feature, default_val) in enumerate(NUMERIC_FIELDS):
                user_input[feature] = st.number_input(
                    feature,
                    value=float(default_val),
                    format="%.4f",
                    key=f"num_{idx}",
                )

        submitted = st.form_submit_button("Predict Irrigation Need")

    user_input["submitted"] = submitted
    return user_input


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    temp_safe = df["Temperature_C"].replace(0, np.nan)
    sun_safe = df["Sunlight_Hours"].replace(0, np.nan)

    df["Moisture_Temp_Ratio"] = (df["Soil_Moisture"] / temp_safe).fillna(0.0)
    df["Rain_Sun_Ratio"] = (df["Rainfall_mm"] / sun_safe).fillna(0.0)
    df["Soil_Crop"] = df["Soil_Type"].astype(str) + "_" + df["Crop_Type"].astype(str)
    df["Season_Irrigation"] = df["Season"].astype(str) + "_" + df["Irrigation_Type"].astype(str)
    return df


def encode_categorical_features(
    df: pd.DataFrame, label_encoders: Dict[str, Any]
) -> Tuple[pd.DataFrame, List[str]]:
    df_encoded = df.copy()
    warnings: List[str] = []

    for feature, encoder in label_encoders.items():
        if feature not in df_encoded.columns:
            continue

        value = str(df_encoded.at[0, feature])
        known = set(map(str, getattr(encoder, "classes_", [])))
        if value not in known:
            warnings.append(
                f"Unknown category for '{feature}': '{value}'. Please choose a known category."
            )
            continue

        try:
            df_encoded[feature] = encoder.transform(df_encoded[feature].astype(str))
        except Exception as exc:
            warnings.append(f"Encoding failed for '{feature}': {exc}")

    return df_encoded, warnings


def align_features_for_model(model: Any, df: pd.DataFrame) -> pd.DataFrame:
    if hasattr(model, "feature_name_") and model.feature_name_ is not None:
        cols = list(model.feature_name_)
        return df.reindex(columns=cols, fill_value=0)

    if hasattr(model, "feature_names_in_"):
        cols = list(model.feature_names_in_)
        return df.reindex(columns=cols, fill_value=0)

    return df


def predict_ensemble(df: pd.DataFrame, artifacts: Dict[str, Any]) -> np.ndarray:
    models = [artifacts["lgb"], artifacts["xgb"], artifacts["cat"]]
    probs = []

    for model in models:
        model_input = align_features_for_model(model, df)
        prob = model.predict_proba(model_input)[0]
        probs.append(np.asarray(prob, dtype=float))

    return np.mean(probs, axis=0)


def decode_prediction(avg_prob: np.ndarray, target_encoder: Any) -> Tuple[str, Dict[str, float]]:
    pred_idx = int(np.argmax(avg_prob))

    if hasattr(target_encoder, "inverse_transform"):
        pred_label = str(target_encoder.inverse_transform([pred_idx])[0])
        class_labels = [str(x) for x in getattr(target_encoder, "classes_", range(len(avg_prob)))]
    else:
        pred_label = str(pred_idx)
        class_labels = [f"Class {i}" for i in range(len(avg_prob))]

    confidence = {class_labels[i]: float(avg_prob[i]) for i in range(len(avg_prob))}
    return pred_label, confidence

def get_water_saving_tips(prediction, input_data):
    tips = []

    moisture = input_data["Soil_Moisture"]
    rainfall = input_data["Rainfall_mm"]
    temp = input_data["Temperature_C"]

    if prediction == "High":
        tips.append("⚠️ High irrigation needed. Consider drip irrigation to save water.")
        tips.append("💧 Irrigate during early morning or late evening to reduce evaporation.")

    elif prediction == "Medium":
        tips.append("🌤 Moderate irrigation needed. Monitor soil moisture regularly.")
        tips.append("🚿 Use sprinkler systems for balanced distribution.")

    else:  # Low
        tips.append("✅ Low irrigation needed. Avoid overwatering.")
        tips.append("🌱 Use mulching to retain soil moisture.")

    # Extra smart conditions
    if rainfall > 1500:
        tips.append("🌧 High rainfall detected — reduce irrigation frequency.")

    if temp > 35:
        tips.append("🔥 High temperature — increase irrigation efficiency (drip recommended).")

    if moisture > 50:
        tips.append("💦 Soil already moist — avoid excess irrigation.")

    return tips

def get_region_tips(region):
    region = region.lower()

    tips = {
        "north": "❄️ Cooler climate — irrigation frequency can be lower.",
        "south": "🌡 Warmer region — monitor evaporation rates.",
        "east": "🌧 Higher rainfall — avoid over-irrigation.",
        "west": "☀️ Drier conditions — consider efficient irrigation systems."
    }

    return tips.get(region, "🌍 Monitor local conditions for best irrigation practices.")




def main() -> None:
    st.set_page_config(page_title="Smart Irrigation Prediction System", layout="wide")
    st.title("Smart Irrigation Prediction System")
    render_sidebar()

    try:
        artifacts = load_artifacts()
    except Exception as exc:
        st.error(f"Unable to load model artifacts: {exc}")
        return

    label_encoders = artifacts["label_encoders"]
    target_encoder = artifacts["target_encoder"]

    user_input = render_input_form(label_encoders)
    if not user_input.get("submitted"):
        return

    row = {k: v for k, v in user_input.items() if k != "submitted"}
    input_df = pd.DataFrame([row])

    engineered_df = engineer_features(input_df)
    encoded_df, encoding_warnings = encode_categorical_features(engineered_df, label_encoders)

    if encoding_warnings:
        for w in encoding_warnings:
            st.warning(w)
        st.stop()

    try:
        avg_prob = predict_ensemble(encoded_df, artifacts)
        prediction, confidence = decode_prediction(avg_prob, target_encoder)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        return

    st.success(f"Predicted Irrigation Need: **{prediction}**")
    st.subheader("Confidence Scores")

    conf_df = (
        pd.DataFrame(
            {"Irrigation Need": list(confidence.keys()), "Probability": list(confidence.values())}
        )
        .sort_values("Probability", ascending=False)
        .reset_index(drop=True)
    )

    st.dataframe(conf_df, use_container_width=True)
    st.bar_chart(conf_df.set_index("Irrigation Need"))

    st.subheader("🌱 Water Saving Recommendations")
    tips = get_water_saving_tips(prediction, user_input_dict)
    for tip in tips:
        st.write(tip)

    st.subheader("🌍 Region-Based Advice")
    region_tip = get_region_tips(user_input_dict["Region"])
    st.info(region_tip)


if __name__ == "__main__":
    main()
