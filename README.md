# 🛠️ Chatter Detection — FFT Features + Streamlit UI

Detect machining **chatter** from time-series force signals using a clean **Streamlit** frontend and a reusable **Python backend**.  
This app ingests `.csv/.xlsx` files, extracts **time & frequency features** (FFT on `FZ`), visualizes signals/FFT, trains **ML models** (RandomForest/XGBoost), and lets you **download** a ready-to-use model bundle.

---

## 🔎 What is this project?

**Goal:** Provide an end-to-end, reproducible workflow to identify machining chatter from sensor data (e.g., cutting force `FZ`).  
**Why:** Chatter harms surface finish, reduces tool life, and risks machine safety. Early detection enables quick corrective actions (speed/feed/DOC changes).

**Pipeline (high-level):**
1. Read files containing `FZ, DOC, SPEED, FEED, CHATTER`.
2. Window the `FZ` signal (segment/step).
3. Compute FFT (one-sided) + statistical features (energy, kurtosis, entropy, crest factor, peak frequency, etc.).
4. Standardize → `SelectKBest` (ANOVA) → Train (RF/XGBoost).
5. Inspect metrics/plots; export a `.joblib` bundle containing the model, scaler, selector, and feature names.

**Frontend (Streamlit):**
- Point to a **data folder**.
- Preview **time domain** and **zoomed FFT (200–1000 Hz)** plots.
- See **correlations** and **top-K features**.
- Tune **hyperparameters** and **train** interactively.
- **Download** the trained bundle for reuse.

---

## 📂 Project Structure

chatter-app/
├─ app/
│ └─ streamlit_app.py # Streamlit UI (orchestration & visuals)
├─ backend/
│ ├─ init.py
│ └─ pipeline.py # Feature extraction & ML helpers (reusable)
├─ models/ # Exported model bundles (.joblib)
├─ data/ # (optional) put your .csv/.xlsx here
├─ requirements.txt
└─ README.md

---

## 🧾 Data Requirements

- Supported: **`.csv`, `.xlsx`** (top-level only; no subfolders).
- Required columns (case-sensitive):

| Column    | Type     | Description                                 |
|-----------|----------|---------------------------------------------|
| `FZ`      | float[]  | Force signal used for FFT & features        |
| `DOC`     | float    | Depth of cut                                |
| `SPEED`   | float    | Spindle speed (or equivalent)               |
| `FEED`    | float    | Feed rate                                   |
| `CHATTER` | int/bool | Target label (e.g., 0 = no chatter, 1 = yes)|

**Minimal CSV example:**
```csv
FZ,DOC,SPEED,FEED,CHATTER
0.012,0.5,1200,0.15,0
0.011,0.5,1200,0.15,0
0.017,0.5,1200,0.15,1
...
##🖥️ Using the App

In the sidebar, set the Data folder path to your .csv/.xlsx files.

Adjust segment size (e.g., 1024), step size (e.g., 256), and sampling rate (e.g., 10005 Hz).

Click 🔎 Extract Features to build the feature table (downloadable as CSV).

Review time/FFT previews, correlation heatmaps, and top correlations vs chatter.

Pick a model (RandomForest or XGBoost), set hyperparameters, and click 🚀 Train Model.

Inspect the classification report, confusion matrix, train/test accuracy, and download the .joblib bundle.