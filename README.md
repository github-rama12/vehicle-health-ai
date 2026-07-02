# Edge AI Vehicle Health & Predictive Maintenance — Full-Stack Version

A real client-server architecture: a **FastAPI backend** that runs the trained
ML models and exposes them as a REST API, plus a **static HTML/JS frontend**
that calls that API. One command starts the whole app.

This is a step up from the earlier single-file Streamlit prototype — it
mirrors how the system would actually be architected in production, with a
clear separation between the "edge inference service" (backend) and the
"driver / fleet manager dashboard" (frontend).

## Project structure

```
vehicle_health_fullstack/
├── backend/
│   ├── main.py              # FastAPI app: loads models, exposes REST endpoints, serves frontend
│   ├── requirements.txt
│   ├── vehicle_sensor_data.csv
│   └── models/               # pre-trained scikit-learn models (.joblib)
├── frontend/
│   ├── index.html            # dashboard layout
│   ├── style.css             # dark theme styling
│   └── app.js                # calls the backend API and renders charts (Chart.js)
└── README.md
```

## How to run it locally

1. Open a terminal in the project root (the folder containing `backend/` and `frontend/`).
2. Install backend dependencies:
   ```bash
   pip install -r backend/requirements.txt
   ```
3. Start the server:
   ```bash
   uvicorn backend.main:app --reload --port 8000
   ```
4. Open your browser to **http://localhost:8000**

That single `uvicorn` command runs the API *and* serves the frontend — no
second server or `streamlit run` needed.

## API endpoints (for judges / testing)

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Backend + model status check |
| GET | `/api/vehicles` | List all vehicle IDs in the dataset |
| GET | `/api/vehicle/{id}/meta` | Basic metadata (days of history, odometer range) |
| GET | `/api/vehicle/{id}/day/{day}` | Full prediction for one day: health score, alert, component probabilities, RUL |
| GET | `/api/vehicle/{id}/history?up_to_day=N` | Raw sensor history for charting |
| POST | `/api/predict` | **Send a live sensor reading, get a prediction back** — this is the endpoint a real OBD-II/IoT edge device would call |

You can also explore and test the API interactively at
**http://localhost:8000/docs** (FastAPI's built-in Swagger UI) — useful to
show judges the API contract directly.

Example of calling the live-prediction endpoint from a terminal:
```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"engine_temp_c":95,"vibration_g":0.9,"battery_voltage_v":12.4,"brake_pad_mm":8,"tire_pressure_psi":31,"fuel_efficiency_kmpl":13,"rpm":2500,"fault_code_present":1}'
```

## Deploying it live (shareable link)

Because this is a FastAPI app (not Streamlit), deploy it on a platform that
runs a Python web server, for example **Render** (free tier, easiest for a
hackathon):

1. Push this project to GitHub.
2. Go to [render.com](https://render.com) → **New +** → **Web Service** → connect your repo.
3. Settings:
   - **Root directory:** leave blank (repo root)
   - **Build command:** `pip install -r backend/requirements.txt`
   - **Start command:** `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
4. Deploy. Render gives you a public URL like `https://vehicle-health-ai.onrender.com`.

(Alternatives: Railway, Fly.io, or a small AWS/Azure VM work the same way —
any host that can run `uvicorn` and exposes a port.)

## Model performance (unchanged from the original training run)

- Fault detection classifier: **98.7% accuracy**, 96.6% recall on the fault class.
- Failing-component classifier: **94.8% accuracy** across engine/battery/brakes/tires/none.
- RUL regressors: mean absolute error of 12–30 days depending on component.

## From prototype to production (talking points for judges)

- Swap the scikit-learn models for **TensorFlow Lite / ONNX Runtime**,
  quantized and running directly on a Raspberry Pi or NVIDIA Jetson Nano —
  the `/api/predict` contract stays identical, only what's behind it changes.
- Replace `vehicle_sensor_data.csv` with a live **MQTT** feed from OBD-II /
  IoT sensors, pushed into the backend via a small ingestion worker.
- Add authentication + per-vehicle access control for a real fleet-management
  deployment.
- Persist prediction history in SQLite/Postgres instead of recomputing from CSV.
