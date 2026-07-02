"""
backend/main.py
FastAPI backend for the Edge AI Vehicle Health & Predictive Maintenance system.

Serves:
  - REST API endpoints that run the trained models (fault detection,
    component classification, RUL regression) against sensor data.
  - The static frontend (frontend/) at the root path.

Run with (from the project root):
    uvicorn backend.main:app --reload --port 8000
Then open http://localhost:8000 in your browser.
"""

import os
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"

FEATURES = [
    "engine_temp_c", "vibration_g", "battery_voltage_v", "brake_pad_mm",
    "tire_pressure_psi", "fuel_efficiency_kmpl", "rpm", "fault_code_present",
]
COMPONENTS = ["engine", "battery", "brakes", "tires"]

HEALTHY_RANGES = {
    "engine_temp_c": (80, 100),
    "vibration_g": (0.15, 0.45),
    "battery_voltage_v": (12.2, 14.4),
    "brake_pad_mm": (3.0, 12.0),
    "tire_pressure_psi": (28, 36),
    "fuel_efficiency_kmpl": (11, 18),
}

# ---------------------------------------------------------------------------
# Load models + data once at startup
# ---------------------------------------------------------------------------
fault_clf = joblib.load(BASE_DIR / "models" / "fault_classifier.joblib")
comp_clf = joblib.load(BASE_DIR / "models" / "component_classifier.joblib")
label_encoder = joblib.load(BASE_DIR / "models" / "component_label_encoder.joblib")
rul_models = joblib.load(BASE_DIR / "models" / "rul_regressors.joblib")
DF = pd.read_csv(BASE_DIR / "vehicle_sensor_data.csv")

app = FastAPI(title="Vehicle Health AI API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # relaxed for hackathon demo; restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class SensorReading(BaseModel):
    engine_temp_c: float = Field(..., example=92.5)
    vibration_g: float = Field(..., example=0.32)
    battery_voltage_v: float = Field(..., example=12.5)
    brake_pad_mm: float = Field(..., example=7.2)
    tire_pressure_psi: float = Field(..., example=31.0)
    fuel_efficiency_kmpl: float = Field(..., example=13.8)
    rpm: float = Field(..., example=2300)
    fault_code_present: int = Field(..., example=0)


# ---------------------------------------------------------------------------
# Core prediction logic (shared by all endpoints)
# ---------------------------------------------------------------------------
def compute_health_score(row: dict, fault_prob: float) -> int:
    penalty = 0.0
    for feat, (lo, hi) in HEALTHY_RANGES.items():
        val = row[feat]
        if val < lo or val > hi:
            span = hi - lo
            overflow = max(lo - val, val - hi, 0)
            penalty += min(25, 25 * overflow / max(span, 1e-6))
    penalty += fault_prob * 40
    return int(round(max(0, 100 - penalty)))


def score_band(score: int) -> str:
    if score >= 80:
        return "Excellent"
    if score >= 60:
        return "Good"
    if score >= 40:
        return "Attention Needed"
    return "Critical"


def run_prediction(reading: dict) -> dict:
    X_row = pd.DataFrame([reading])[FEATURES]

    fault_prob = float(fault_clf.predict_proba(X_row)[0][1])
    comp_idx = int(comp_clf.predict(X_row)[0])
    comp_pred = label_encoder.inverse_transform([comp_idx])[0]
    comp_proba = comp_clf.predict_proba(X_row)[0]

    health = compute_health_score(reading, fault_prob)
    status = score_band(health)

    rul = {}
    for comp in COMPONENTS:
        pred = float(rul_models[comp].predict(X_row)[0])
        rul[comp] = round(max(0, min(999, pred)), 1)

    return {
        "health_score": health,
        "status": status,
        "fault_probability": round(fault_prob, 4),
        "predicted_component": str(comp_pred),
        "component_probabilities": {
            cls: round(float(p), 4) for cls, p in zip(label_encoder.classes_, comp_proba)
        },
        "remaining_useful_life_days": rul,
        "sensor_reading": reading,
    }


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health_check():
    return {"status": "ok", "vehicles_loaded": int(DF["vehicle_id"].nunique())}


@app.get("/api/vehicles")
def list_vehicles():
    ids = sorted(DF["vehicle_id"].unique().tolist())
    return {"vehicle_ids": ids}


@app.get("/api/vehicle/{vehicle_id}/meta")
def vehicle_meta(vehicle_id: int):
    vdf = DF[DF["vehicle_id"] == vehicle_id]
    if vdf.empty:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    return {
        "vehicle_id": vehicle_id,
        "num_days": int(len(vdf)),
        "odometer_start_km": int(vdf["odometer_km"].iloc[0]),
        "odometer_end_km": int(vdf["odometer_km"].iloc[-1]),
    }


@app.get("/api/vehicle/{vehicle_id}/day/{day}")
def vehicle_day_prediction(vehicle_id: int, day: int):
    """Runs the edge AI model against a specific day's sensor snapshot for a vehicle."""
    vdf = DF[DF["vehicle_id"] == vehicle_id].reset_index(drop=True)
    if vdf.empty:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    if day < 0 or day >= len(vdf):
        raise HTTPException(status_code=400, detail=f"day must be between 0 and {len(vdf) - 1}")

    row = vdf.iloc[day]
    reading = {f: float(row[f]) for f in FEATURES}
    reading["fault_code_present"] = int(row["fault_code_present"])

    result = run_prediction(reading)
    result["vehicle_id"] = vehicle_id
    result["day"] = int(row["day"])
    result["odometer_km"] = int(row["odometer_km"])
    return result


@app.get("/api/vehicle/{vehicle_id}/history")
def vehicle_history(vehicle_id: int, up_to_day: Optional[int] = None):
    """Returns raw sensor history for charting, optionally truncated at up_to_day."""
    vdf = DF[DF["vehicle_id"] == vehicle_id].reset_index(drop=True)
    if vdf.empty:
        raise HTTPException(status_code=404, detail="Vehicle not found")
    if up_to_day is not None:
        vdf = vdf.iloc[: up_to_day + 1]

    cols = ["day"] + FEATURES
    return {"vehicle_id": vehicle_id, "history": vdf[cols].to_dict(orient="records")}


@app.post("/api/predict")
def predict_custom(reading: SensorReading):
    """Accepts a live sensor reading (e.g. from an OBD-II device or IoT gateway)
    and returns the edge AI model's prediction. This is the endpoint a real
    edge device would call in production."""
    return run_prediction(reading.dict())


# ---------------------------------------------------------------------------
# Serve the frontend (must be added AFTER the /api routes above)
# ---------------------------------------------------------------------------
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
