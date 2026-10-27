# Continuous Sepsis Intelligence: Gradient-Boosted ICU Decision Support
### Research Prototype

This directory contains the self-contained deployment package for the XGBoost Sepsis Clinical Decision Support Streamlit application.

---

## Directory Structure

```text
sepsis_streamlit_app/
├── app.py                            # Main Streamlit web application
├── xgb_sepsis_model.json              # Trained XGBoost sepsis prediction model (15-seed ensemble representation)
├── model_metadata.json               # Operating threshold (0.055), feature schema, and external benchmark metrics
├── streamlit_cohort_pool.parquet     # Stratified sample of 500 patient stays from external Hospital Set B (238 KB)
├── streamlit_features_pool.parquet   # Pre-engineered feature matrix matching the 500 cohort patients (677 KB)
├── requirements.txt                  # Python dependencies
├── README.md                         # Deployment documentation and execution instructions
└── .streamlit/
    └── config.toml                   # Clinical high-contrast light theme configuration
```

**Total Package Footprint:** ~1.4 MB (fully self-contained, no external database required).

---

## Quick Start: Running Locally

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch Application
```bash
streamlit run app.py
```
The application will open automatically in your browser at `http://localhost:8501`.

---

## Features & Clinical Architecture

1. **ICU Monitoring Board (24, 32, or 50 Beds):**
   - Stratified patient sampling maintaining the exact population sepsis prevalence (6.25%).
   - Live vital signs, 3-hour trajectory indicators, and operating threshold flags (`0.055`).
2. **Live Simulation Engine:**
   - Non-blocking client-side hourly stepping (`Step -1h`, `Step +1h`, `Play Live Simulation`, `Next Alert`, `Reset Clock`).
3. **Bedside Chart & Longitudinal Physiological Timelines:**
   - 4-panel trajectory plots: Risk trajectory, Vital signs (HR, Resp, MAP), Pulmonary SF ratio ($SpO_2 / FiO_2$), and Serum Lactate / Staleness.
4. **TreeSHAP Explainability Waterfall:**
   - Additive Shapley feature attribution displaying the top clinical drivers shifting patient risk log-odds.
5. **Clinical Trade-offs & Benchmark Metrics:**
   - Decision threshold slider ($t^*$) illustrating the balance between early detection (7.6%/hr septic shock mortality window) and alarm fatigue (150–350 alarms/bed/day, 40:1 utility penalty).
6. **Clinician Triage Protocol & Audit Trail:**
   - Bedside actions (`Reviewed`, `Escalated`, `Dismissed`) with CSV export.
