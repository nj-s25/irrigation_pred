# 🌾 Smart Irrigation Prediction System

A machine learning powered web app to predict irrigation needs using environmental and agricultural data.

---

## 🚀 Live App
(Insert your Streamlit link here after deployment)

---

## 📌 Features

- 🌱 Predict Irrigation Need (Low / Medium / High)
- 📊 Probability Confidence Visualization
- 🌍 Region-based Recommendations
- 💧 Water-saving Suggestions
- 🤖 Ensemble ML Model (LightGBM + XGBoost + CatBoost)

---

## 🧠 Model Details

This project uses an ensemble of three models:

- LightGBM
- XGBoost
- CatBoost

Final prediction is computed by averaging probabilities from all models.

---

## 📊 Dataset

- Source: Kaggle Playground Series (S6E4)
- Target: `Irrigation_Need`
- Classes: Low, Medium, High

---

## ⚙️ Feature Engineering

- Moisture_Temp_Ratio = Soil_Moisture / Temperature_C
- Rain_Sun_Ratio = Rainfall_mm / Sunlight_Hours
- Soil_Crop = Soil_Type + Crop_Type
- Season_Irrigation = Season + Irrigation_Type

---

## 🏗️ Project Structure

