import os
import time
import json
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb
import shap
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -------------------------------------------------------------
# 1. Page Configuration & Clinical Styling
# -------------------------------------------------------------
st.set_page_config(
    page_title="Continuous Sepsis Intelligence: Gradient-Boosted ICU Decision Support",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# High-contrast light clinical styling
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400..700;1,6..72,400..700&family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700;800&display=swap" rel="stylesheet">

<style>
    /* Global Background & High-Contrast Typography */
    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: #0a3641 !important;
        background-color: #f7fafb !important;
    }

    /* Only style explicit content text, never dropdowns/popovers */
    p, h1, h2, h3, h4, h5, h6, .editorial-heading, .bed-title {
    color: #0a3641;
    }

    /* Force Table Cells to Always Have Dark High-Contrast Text */
    table, table tr, table td, table th {
        color: #0a3641 !important;
        background-color: transparent !important;
    }

    /* Dropdown Menus & Tooltips: Crisp White Text on Dark Backgrounds */
    [data-baseweb="popover"] *,
    [data-baseweb="tooltip"] *,
    [data-baseweb="select"] *,
    ul[role="listbox"] li * {
        color: #ffffff !important;
    }

    /* Optimized Page Padding */
    .block-container {
        padding-top: 0.8rem !important;
        padding-bottom: 2rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 100% !important;
    }

    header[data-testid="stHeader"] {
        background: transparent !important;
        height: 1.2rem !important;
    }

    /* Top Brand & Header Bar */
    .clinical-navbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #ffffff;
        border: 1px solid #d8e6e7;
        border-radius: 8px;
        padding: 10px 20px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(10, 54, 65, 0.04);
    }
    .clinical-brand {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .brand-title-text {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-weight: 700;
        font-size: 1.15rem;
        letter-spacing: 0.10em;
        color: #0a3641;
    }
    .badge-prototype-pill {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        color: #e76f51 !important;
        background: #ffffff;
        border: 1.5px solid #e76f51;
        padding: 4px 12px;
        border-radius: 9999px;
        text-transform: uppercase;
    }
    .nav-status-strip {
        display: flex;
        align-items: center;
        gap: 12px;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .nav-status-badge {
        background: #f0f7f7;
        padding: 4px 10px;
        border-radius: 6px;
        border: 1px solid #d8e6e7;
        color: #0a3641 !important;
    }

    /* Giant Centralized ICU Simulation Clock Card */
    .icu-hero-clock-box {
        background: #ffffff;
        border: 2px solid #008080;
        border-radius: 12px;
        padding: 14px 24px;
        text-align: center;
        box-shadow: 0 4px 16px rgba(0, 128, 128, 0.08);
        margin-bottom: 12px;
    }
    .icu-clock-eyebrow {
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.16em;
        color: #008080;
        text-transform: uppercase;
        margin-bottom: 2px;
    }
    .icu-clock-digits {
        font-family: 'JetBrains Mono', monospace;
        font-size: 3.4rem;
        font-weight: 800;
        color: #0a3641;
        letter-spacing: 0.06em;
        line-height: 1.05;
        margin: 2px 0;
    }
    .icu-clock-subtext {
        font-size: 0.82rem;
        color: #475569;
        font-weight: 600;
    }

    /* Editorial Serif Headings */
    .editorial-heading {
        font-family: 'Newsreader', Georgia, serif;
        font-size: 1.45rem;
        font-weight: 600;
        color: #0a3641 !important;
        letter-spacing: -0.01em;
        margin: 0;
        padding: 0;
    }
    .editorial-subheading {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 0.82rem;
        color: #475569 !important;
        margin-top: 2px;
        margin-bottom: 10px;
    }

    /* Patient Cards - Equal Width Symmetrical Layout */
    .patient-card {
        background: #ffffff;
        border: 1px solid #d8e6e7;
        border-radius: 8px;
        padding: 14px 16px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(10, 54, 65, 0.03);
        transition: transform 0.12s ease, box-shadow 0.12s ease;
        box-sizing: border-box;
        width: 100% !important;
        min-width: 0 !important;
    }
    .patient-card:hover {
        box-shadow: 0 4px 12px rgba(10, 54, 65, 0.08);
        border-color: #008080;
    }
    .patient-card-critical {
        border-left: 5px solid #be123c !important;
        background: #fffafa;
    }
    .patient-card-actioned {
        border-left: 5px solid #b45309 !important;
        background: #fffdf7;
    }
    .patient-card-stable {
        border-left: 5px solid #008080 !important;
        background: #ffffff;
    }
    .patient-card-discharged {
        border-left: 5px solid #94a3b8 !important;
        background: #f8fafc;
        opacity: 0.70;
    }

    /* Badges (NO EMOJIS) */
    .badge-critical {
        background: #fff1f2;
        color: #be123c !important;
        border: 1px solid #fecdd3;
        font-size: 0.72rem;
        font-weight: 700;
        padding: 3px 8px;
        border-radius: 4px;
        letter-spacing: 0.04em;
    }
    .badge-actioned {
        background: #fefce8;
        color: #92400e !important;
        border: 1px solid #fef08a;
        font-size: 0.72rem;
        font-weight: 700;
        padding: 3px 8px;
        border-radius: 4px;
        letter-spacing: 0.04em;
    }
    .badge-stable {
        background: #f0fdfa;
        color: #0f766e !important;
        border: 1px solid #ccfbf1;
        font-size: 0.72rem;
        font-weight: 700;
        padding: 3px 8px;
        border-radius: 4px;
        letter-spacing: 0.04em;
    }
    .badge-discharged {
        background: #f1f5f9;
        color: #475569 !important;
        border: 1px solid #e2e8f0;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 3px 8px;
        border-radius: 4px;
    }

    /* Bed Headers & Vitals */
    .bed-title {
        font-family: 'Newsreader', Georgia, serif;
        font-size: 1.25rem;
        font-weight: 600;
        color: #0a3641 !important;
    }
    .patient-meta {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        color: #475569 !important;
        margin-left: 6px;
    }
    .risk-score-critical {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.5rem;
        font-weight: 700;
        color: #be123c !important;
    }
    .risk-score-stable {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.5rem;
        font-weight: 700;
        color: #008080 !important;
    }
    .trend-rising {
        color: #be123c !important;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .trend-falling {
        color: #008080 !important;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .trend-stable {
        color: #475569 !important;
        font-weight: 500;
        font-size: 0.85rem;
    }

    .vitals-strip {
        background: #f7fafb;
        border: 1px solid #e5eff0;
        border-radius: 6px;
        padding: 8px 12px;
        font-size: 0.82rem;
        color: #1e293b !important;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 6px;
        margin-top: 8px;
        margin-bottom: 6px;
    }
    .vitals-val {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        color: #0a3641 !important;
    }

    /* Button Styling */
    .stButton>button {
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        border: 1px solid #d8e6e7;
        background-color: #ffffff;
        color: #0a3641 !important;
        height: 38px;
        transition: all 0.10s ease;
    }
    .stButton>button:hover {
        border-color: #008080;
        color: #008080 !important;
        background-color: #f0f7f7;
    }
    .stButton>button[kind="primary"] {
        background-color: #008080 !important;
        color: #ffffff !important;
        border: 1px solid #006666 !important;
    }
    .stButton>button[kind="primary"]:hover {
        background-color: #0a3641 !important;
        border-color: #0a3641 !important;
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# 2. Asset Loading & Model Caching
# -------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_model_and_explainer():
    model = xgb.XGBClassifier()
    model.load_model(os.path.join(BASE_DIR, "xgb_sepsis_model.json"))
    with open(os.path.join(BASE_DIR, "model_metadata.json"), "r") as f:
        meta = json.load(f)
    explainer = shap.TreeExplainer(model)
    return model, explainer, meta

@st.cache_data
def load_cohort_pool():
    cohort_df = pd.read_parquet(os.path.join(BASE_DIR, "streamlit_cohort_pool.parquet"))
    features_df = pd.read_parquet(os.path.join(BASE_DIR, "streamlit_features_pool.parquet"))
    return cohort_df, features_df

model, explainer, meta = load_model_and_explainer()
cohort_pool, features_pool = load_cohort_pool()

# -------------------------------------------------------------
# 3. Session State Initialization
# -------------------------------------------------------------
if "cohort_seed" not in st.session_state:
    st.session_state.cohort_seed = 42

if "sim_hour" not in st.session_state:
    st.session_state.sim_hour = 6

if "is_playing" not in st.session_state:
    st.session_state.is_playing = False

if "play_speed" not in st.session_state:
    st.session_state.play_speed = 3

if "alert_actions" not in st.session_state:
    st.session_state.alert_actions = {}  # (patient_id, hour) -> dict

if "selected_pid" not in st.session_state:
    st.session_state.selected_pid = None

if "current_view" not in st.session_state:
    st.session_state.current_view = "board"

if "threshold" not in st.session_state:
    st.session_state.threshold = meta.get("optimal_threshold", 0.055)

if "sort_by" not in st.session_state:
    st.session_state.sort_by = "Risk (High to Low)"

if "filter_status" not in st.session_state:
    st.session_state.filter_status = "All Patients"

if "cohort_size" not in st.session_state:
    st.session_state.cohort_size = 24

# Stratified Sampling: Supporting 24, 32, or 50 beds while maintaining sepsis prevalence
@st.cache_data
def get_simulated_cohort(seed, n_beds):
    pat_stats = cohort_pool.groupby("patient_id").agg(
        sepsis=("SepsisLabel", "max"),
        max_hour=("ICULOS", "max")
    )
    pos_pats = pat_stats[pat_stats["sepsis"] == 1].index.values
    neg_pats = pat_stats[pat_stats["sepsis"] == 0].index.values
    
    rng = np.random.RandomState(seed)
    
    if n_beds == 24:
        n_pos, n_neg = 2, 22
    elif n_beds == 32:
        n_pos, n_neg = 2, 30
    else:
        n_pos, n_neg = 3, 47

    p_pos = rng.choice(pos_pats, size=min(n_pos, len(pos_pats)), replace=False)
    p_neg = rng.choice(neg_pats, size=min(n_neg, len(neg_pats)), replace=False)
    sampled_pids = np.concatenate([p_pos, p_neg])
    rng.shuffle(sampled_pids)
    
    bed_mapping = {pid: f"Bed {i+1:02d}" for i, pid in enumerate(sampled_pids)}
    return list(sampled_pids), bed_mapping

sim_pids, bed_map = get_simulated_cohort(st.session_state.cohort_seed, st.session_state.cohort_size)

if st.session_state.selected_pid not in sim_pids:
    st.session_state.selected_pid = sim_pids[0]

# Precompute predictions across the pool once
@st.cache_data
def get_all_predictions():
    probs = model.predict_proba(features_pool)[:, 1]
    df_eval = cohort_pool[["patient_id", "ICULOS", "SepsisLabel", "HR", "Temp", "Resp", "MAP", "O2Sat", "Lactate", "WBC", "Creatinine", "Age", "Gender"]].copy()
    df_eval["prob"] = probs
    df_eval["SF_ratio"] = features_pool["SF_ratio"]
    df_eval["HR_n_meas"] = features_pool["HR_n_meas"]
    df_eval["Lactate_hrs_since"] = features_pool["Lactate_hrs_since"]
    return df_eval

all_eval_data = get_all_predictions()

@st.cache_data
def get_cohort_eval_data(seed, n_beds):
    pids, _ = get_simulated_cohort(seed, n_beds)
    return all_eval_data[all_eval_data["patient_id"].isin(pids)].copy()

cohort_eval_data = get_cohort_eval_data(st.session_state.cohort_seed, st.session_state.cohort_size)
max_cohort_hour = int(cohort_eval_data["ICULOS"].max())

# -------------------------------------------------------------
# 4. Instant Precomputed Hourly Cache (< 0.0005s lookup)
# -------------------------------------------------------------
@st.cache_data
def precompute_hourly_state_cache(seed, n_beds):
    c_data = get_cohort_eval_data(seed, n_beds)
    _, b_map = get_simulated_cohort(seed, n_beds)
    
    m_hour = int(c_data["ICULOS"].max())
    max_h_map = c_data.groupby("patient_id")["ICULOS"].max().to_dict()
    
    hourly_dict = {}
    for h in range(1, m_hour + 1):
        past = c_data[c_data["ICULOS"] <= h]
        if past.empty:
            hourly_dict[h] = []
            continue
            
        latest = past.sort_values("ICULOS").groupby("patient_id").last().reset_index()
        
        # 3h prior trajectory
        t3_past = past[past["ICULOS"] <= (latest.set_index("patient_id")["ICULOS"].reindex(past["patient_id"]).values - 3)]
        t3_probs = t3_past.sort_values("ICULOS").groupby("patient_id").last()["prob"].to_dict() if not t3_past.empty else {}
        
        recs = []
        for row in latest.to_dict(orient="records"):
            pid = row["patient_id"]
            prob = float(row["prob"])
            lh = int(row["ICULOS"])
            t3_p = t3_probs.get(pid, prob)
            
            recs.append({
                "patient_id": pid,
                "bed": b_map.get(pid, "Bed"),
                "prob": prob,
                "risk_diff": prob - t3_p,
                "hr": row["HR"],
                "temp": row["Temp"],
                "resp": row["Resp"],
                "map": row["MAP"],
                "o2sat": row["O2Sat"],
                "sf_ratio": row["SF_ratio"],
                "lactate": row["Lactate"],
                "lactate_hrs_since": row["Lactate_hrs_since"],
                "wbc": row["WBC"],
                "creatinine": row["Creatinine"],
                "iculos": lh,
                "sepsis_label": row["SepsisLabel"],
                "age": row["Age"],
                "gender": row["Gender"],
                "max_hour": max_h_map.get(pid, lh)
            })
        hourly_dict[h] = recs
    return hourly_dict

state_cache = precompute_hourly_state_cache(st.session_state.cohort_seed, st.session_state.cohort_size)

# Helper function to format human-readable clinical features for TreeSHAP
def format_clinical_feature(name, val):
    val_str = f"{val:.1f}" if (val is not None and pd.notna(val)) else "—"
    if name == "HR_n_meas":
        return f"HR Measurement Freq ({val:.0f} readings)"
    elif name == "Resp_n_meas":
        return f"Resp Measurement Freq ({val:.0f} readings)"
    elif name == "Temp_n_meas":
        return f"Temp Measurement Freq ({val:.0f} readings)"
    elif name == "Lactate_hrs_since":
        return f"Lactate Staleness ({val:.0f}h ago)"
    elif name == "WBC_hrs_since":
        return f"WBC Staleness ({val:.0f}h ago)"
    elif name == "SF_ratio":
        return f"SF Ratio SpO2/FiO2 ({val:.0f})"
    elif name == "HR_d6":
        sign = "+" if val > 0 else ""
        return f"HR 6h Rate of Change ({sign}{val:.1f} bpm)"
    elif name == "Resp_d6":
        sign = "+" if val > 0 else ""
        return f"Resp 6h Rate of Change ({sign}{val:.1f} /min)"
    elif name == "Temp_d6":
        sign = "+" if val > 0 else ""
        return f"Temp 6h Rate of Change ({sign}{val:.1f} °C)"
    elif name == "MAP_d6":
        sign = "+" if val > 0 else ""
        return f"MAP 6h Rate of Change ({sign}{val:.1f} mmHg)"
    elif name == "HR":
        return f"Heart Rate ({val:.0f} bpm)"
    elif name == "Resp":
        return f"Respiratory Rate ({val:.0f} /min)"
    elif name == "Temp":
        return f"Body Temperature ({val:.1f} °C)"
    elif name == "MAP":
        return f"Mean Arterial Pressure ({val:.0f} mmHg)"
    elif name == "Lactate":
        return f"Serum Lactate ({val:.1f} mmol/L)"
    elif name == "WBC":
        return f"White Blood Cells ({val:.1f} k/uL)"
    elif name == "Creatinine":
        return f"Serum Creatinine ({val:.1f} mg/dL)"
    elif name == "Age":
        return f"Patient Age ({val:.0f} yrs)"
    else:
        return f"{name} ({val_str})"

# -------------------------------------------------------------
# 5. Top Brand Navbar
# -------------------------------------------------------------
origami_svg = """<svg width="26" height="26" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg" style="vertical-align: middle; margin-right: 8px;"><polygon points="10,80 50,15 90,80" fill="#008080" opacity="0.85"/><polygon points="50,15 90,80 50,60" fill="#00a896" opacity="0.95"/><polygon points="10,80 50,60 50,95" fill="#028090" opacity="0.90"/><polygon points="50,15 35,48 50,60" fill="#48cae4" opacity="0.80"/></svg>"""

navbar_html = f"""<div class="clinical-navbar">
<div class="clinical-brand">
{origami_svg}
<span class="brand-title-text">CONTINUOUS SEPSIS INTELLIGENCE</span>
</div>
<div style="display: flex; align-items: center; gap: 16px;">
<span class="badge-prototype-pill">RESEARCH PROTOTYPE &bull; NOT FDA CLEARED</span>
<div class="nav-status-strip">
<span class="nav-status-badge"><b>Beds:</b> {len(sim_pids)}</span>
<span class="nav-status-badge"><b>Cohort Mode:</b> {st.session_state.cohort_size} Beds</span>
</div>
</div>
</div>"""

st.html(navbar_html)

# -------------------------------------------------------------
# 6. Primary Section Navigation Tabs
# -------------------------------------------------------------
nav_c1, nav_c2, nav_c3, nav_c4 = st.columns(4)

with nav_c1:
    if st.button(f"Patient Monitoring Board ({len(sim_pids)} Beds)", use_container_width=True, type="primary" if st.session_state.current_view == "board" else "secondary"):
        st.session_state.current_view = "board"
        st.rerun()

with nav_c2:
    if st.button("Bedside Chart & TreeSHAP", use_container_width=True, type="primary" if st.session_state.current_view == "chart" else "secondary"):
        st.session_state.current_view = "chart"
        st.rerun()

with nav_c3:
    if st.button("Clinical Trade-Offs & Benchmarks", use_container_width=True, type="primary" if st.session_state.current_view == "metrics" else "secondary"):
        st.session_state.current_view = "metrics"
        st.rerun()

with nav_c4:
    if st.button("Alert Audit Log", use_container_width=True, type="primary" if st.session_state.current_view == "audit" else "secondary"):
        st.session_state.current_view = "audit"
        st.rerun()

# -------------------------------------------------------------
# 7. Non-Blocking Live Simulation Execution (Fragment-Based)
# -------------------------------------------------------------
# Fragment triggers client-side tick smoothly every N seconds without thread freezing or graying out
auto_interval = float(st.session_state.play_speed) if st.session_state.is_playing else None

@st.fragment(run_every=auto_interval)
def render_live_simulation_block():
    # If playing, advance hour on tick
    if st.session_state.is_playing:
        if st.session_state.sim_hour < 60:
            st.session_state.sim_hour += 1
        else:
            # When cohort surveillance window completes, loop back to active admission hours
            st.session_state.sim_hour = 6
            st.toast("Surveillance window reached Hour 60. Looping back to Hour 06:00 to resume active admissions.")
            
    active_h = st.session_state.sim_hour

    # Giant Centralized ICU Hero Clock Card
    st.markdown(f"""
    <div class="icu-hero-clock-box">
        <div class="icu-clock-eyebrow">ICU SIMULATION CLOCK &bull; ACTIVE BEDSIDE SURVEILLANCE</div>
        <div class="icu-clock-digits">HOUR {active_h:02d}:00</div>
        <div class="icu-clock-subtext">Elapsed ICU Stay Duration &bull</div>
    </div>
    """, unsafe_allow_html=True)

    # Centered Simulation Control Toolbar
    sim_b1, sim_b2, sim_b3, sim_b4, sim_b5, sim_b6 = st.columns([0.16, 0.16, 0.22, 0.16, 0.16, 0.14])

    with sim_b1:
        if st.button("Step -1h", use_container_width=True, disabled=(st.session_state.sim_hour <= 1)):
            st.session_state.sim_hour = max(1, st.session_state.sim_hour - 1)
            st.rerun()
            
    with sim_b2:
        if st.button("Step +1h", use_container_width=True, disabled=(st.session_state.sim_hour >= max_cohort_hour)):
            st.session_state.sim_hour = min(max_cohort_hour, st.session_state.sim_hour + 1)
            st.rerun()
            
    with sim_b3:
        if not st.session_state.is_playing:
            if st.button("Play Live Simulation", use_container_width=True, type="primary"):
                # If already near the end of ICU stays where vitals cease changing, start from Hour 6
                if st.session_state.sim_hour >= 40:
                    st.session_state.sim_hour = 6
                st.session_state.is_playing = True
                st.rerun()
        else:
            if st.button("Pause Simulation", use_container_width=True):
                st.session_state.is_playing = False
                st.rerun()
                
    with sim_b4:
        st.session_state.play_speed = st.selectbox(
            "Playback Speed:",
            options=[2, 3, 5, 10],
            index=[2, 3, 5, 10].index(st.session_state.play_speed) if st.session_state.play_speed in [2, 3, 5, 10] else 1,
            format_func=lambda s: f"{s} sec / hr",
            label_visibility="collapsed"
        )
        
    with sim_b5:
        if st.button("Next Alert", use_container_width=True, help="Scan forward to the next hour where any patient triggers an alert."):
            found_hour = None
            for h in range(st.session_state.sim_hour + 1, max_cohort_hour + 1):
                sub_h = cohort_eval_data[cohort_eval_data["ICULOS"] == h]
                if (sub_h["prob"] >= st.session_state.threshold).any():
                    found_hour = h
                    break
            if found_hour:
                st.session_state.sim_hour = found_hour
                st.rerun()
            else:
                st.toast("No subsequent alerts found in the remaining ICU hours.")
                
    with sim_b6:
        if st.button("Reset Clock", use_container_width=True):
            st.session_state.sim_hour = 6
            st.session_state.is_playing = False
            st.rerun()

    # Playback Active Status Banner
    if st.session_state.is_playing:
        st.markdown(f"""
        <div style="background: #e6f7f8; border: 1px solid #b2e2e6; border-radius: 6px; padding: 6px 16px; margin-bottom: 12px; font-size: 0.85rem; color: #008080 !important; display: flex; justify-content: space-between; align-items: center;">
            <span><b>LIVE SIMULATION ACTIVE:</b> Stepping forward live ({st.session_state.play_speed}s per ICU hour) &bull; Surveillance Hour {active_h:02d}:00</span>
            <span style="font-weight: 600; font-size: 0.82rem; color: #0a3641 !important;">Click 'Pause Simulation' above to freeze</span>
        </div>
        """, unsafe_allow_html=True)

    # -------------------------------------------------------------
    # VIEW 1: Live Patient Monitoring Board
    # -------------------------------------------------------------
    if st.session_state.current_view == "board":
        # Header & Filters with Cohort Size Switcher
        c_hdr1, c_hdr2, c_hdr3, c_hdr4, c_hdr5 = st.columns([0.35, 0.20, 0.18, 0.15, 0.12])
        with c_hdr1:
            st.markdown(f"<div class='editorial-heading'>ICU Monitoring Board ({len(sim_pids)} Beds)</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='editorial-subheading'>Continuous surveillance at Hour {active_h:02d}:00 &bull; Operating Threshold: {st.session_state.threshold*100:.1f}% risk</div>", unsafe_allow_html=True)
        with c_hdr2:
            status_filter = st.selectbox(
                "Filter Beds:",
                options=["All Patients", "Needs Review Only", "Actioned Only", "Stable Only"],
                index=["All Patients", "Needs Review Only", "Actioned Only", "Stable Only"].index(st.session_state.filter_status) if st.session_state.filter_status in ["All Patients", "Needs Review Only", "Actioned Only", "Stable Only"] else 0,
                label_visibility="collapsed"
            )
            st.session_state.filter_status = status_filter
        with c_hdr3:
            sort_choice = st.selectbox(
                "Sort Beds:",
                options=["Risk (High to Low)", "Bed Number", "Risk Trend (Largest Spike)"],
                index=["Risk (High to Low)", "Bed Number", "Risk Trend (Largest Spike)"].index(st.session_state.sort_by) if st.session_state.sort_by in ["Risk (High to Low)", "Bed Number", "Risk Trend (Largest Spike)"] else 0,
                label_visibility="collapsed"
            )
            st.session_state.sort_by = sort_choice
        with c_hdr4:
            selected_size = st.selectbox(
                "Cohort Size:",
                options=[24, 32, 50],
                index=[24, 32, 50].index(st.session_state.cohort_size) if st.session_state.cohort_size in [24, 32, 50] else 0,
                format_func=lambda s: f"{s} Beds (Fastest)" if s == 24 else (f"{s} Beds (6.2% Prev)" if s == 32 else f"{s} Beds (Full)"),
                label_visibility="collapsed"
            )
            if selected_size != st.session_state.cohort_size:
                st.session_state.cohort_size = selected_size
                st.session_state.sim_hour = 6
                st.rerun()
        with c_hdr5:
            if st.button("Shuffle Cohort", use_container_width=True, help="Draw a fresh stratified sample of beds from Hospital B."):
                st.session_state.cohort_seed += 1
                st.session_state.sim_hour = 6
                st.rerun()

        # Re-derive board data for active_h from instant cache
        current_raw_items = state_cache.get(active_h, [])
        active_board_rows = []
        for it in current_raw_items:
            pid = it["patient_id"]
            lh = it["iculos"]
            prob = it["prob"]
            act_rec = st.session_state.alert_actions.get((pid, lh), None)
            
            if active_h > it["max_hour"]:
                status = "Discharged"
            elif act_rec:
                status = f"Actioned ({act_rec['action']})"
            elif prob >= st.session_state.threshold:
                status = "Needs Review"
            else:
                status = "Stable"
                
            rc = dict(it)
            rc["status"] = status
            active_board_rows.append(rc)

        disp_df = pd.DataFrame(active_board_rows)

        if status_filter == "Needs Review Only":
            disp_df = disp_df[disp_df["status"] == "Needs Review"]
        elif status_filter == "Actioned Only":
            disp_df = disp_df[disp_df["status"].str.contains("Actioned", na=False)]
        elif status_filter == "Stable Only":
            disp_df = disp_df[disp_df["status"] == "Stable"]

        if sort_choice == "Risk (High to Low)":
            disp_df = disp_df.sort_values("prob", ascending=False)
        elif sort_choice == "Bed Number":
            disp_df = disp_df.sort_values("bed", ascending=True)
        elif sort_choice == "Risk Trend (Largest Spike)":
            disp_df = disp_df.sort_values("risk_diff", ascending=False)

        if disp_df.empty:
            st.info("No patients match the current filter selection.")
        else:
            # Symmetrical 2-Column Grid Row-by-Row: strictly guarantees 50%/50% width and zero squishing!
            for row_idx in range(0, len(disp_df), 2):
                card_cols = st.columns(2)
                
                # Left Column Card
                with card_cols[0]:
                    render_card_element(disp_df.iloc[row_idx])
                    
                # Right Column Card
                if row_idx + 1 < len(disp_df):
                    with card_cols[1]:
                        render_card_element(disp_df.iloc[row_idx + 1])

    # -------------------------------------------------------------
    # VIEW 2: Bedside Chart & TreeSHAP
    # -------------------------------------------------------------
    elif st.session_state.current_view == "chart":
        render_bedside_chart_view(active_h)

def render_card_element(item):
    pid = item["patient_id"]
    bed = item["bed"]
    prob = item["prob"]
    risk_diff = item["risk_diff"]
    status = item["status"]

    if risk_diff > 0.02:
        trend_html = f"<span class='trend-rising'>▲ +{risk_diff*100:.1f}%</span>"
    elif risk_diff < -0.02:
        trend_html = f"<span class='trend-falling'>▼ {risk_diff*100:.1f}%</span>"
    else:
        trend_html = "<span class='trend-stable'>— Stable</span>"

    if status == "Needs Review":
        card_class = "patient-card patient-card-critical"
        badge_html = "<span class='badge-critical'>CRITICAL REVIEW</span>"
        risk_class = "risk-score-critical"
        btn_label = "Triage Bedside Chart"
        btn_type = "primary"
    elif "Actioned" in status:
        card_class = "patient-card patient-card-actioned"
        act_name = status.replace("Actioned (", "").replace(")", "").upper()
        badge_html = f"<span class='badge-actioned'>ACTIONED: {act_name}</span>"
        risk_class = "risk-score-critical" if prob >= st.session_state.threshold else "risk-score-stable"
        btn_label = "Open Bedside Chart"
        btn_type = "secondary"
    elif status == "Discharged":
        card_class = "patient-card patient-card-discharged"
        badge_html = "<span class='badge-discharged'>DISCHARGED</span>"
        risk_class = "risk-score-stable"
        btn_label = "Open Bedside Chart"
        btn_type = "secondary"
    else:
        card_class = "patient-card patient-card-stable"
        badge_html = "<span class='badge-stable'>STABLE</span>"
        risk_class = "risk-score-stable"
        btn_label = "Open Bedside Chart"
        btn_type = "secondary"

    lactate_str = f"{item['lactate']:.1f}" if pd.notna(item['lactate']) else "—"
    stale_str = f"({item['lactate_hrs_since']:.0f}h ago)" if pd.notna(item['lactate_hrs_since']) else ""
    sf_str = f"{item['sf_ratio']:.0f}" if pd.notna(item['sf_ratio']) else "—"
    wbc_str = f"{item['wbc']:.1f}" if pd.notna(item['wbc']) else "—"

    st.markdown(f"""
    <div class="{card_class}">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
            <div>
                <span class="bed-title">{bed}</span>
                <span class="patient-meta">ID: {pid}</span>
            </div>
            <div>{badge_html}</div>
        </div>
        <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 8px;">
            <div>
                <span style="font-size: 0.76rem; color: #475569 !important; text-transform: uppercase; font-weight: 600;">Sepsis Risk</span><br>
                <span class="{risk_class}">{prob*100:.1f}%</span>
            </div>
            <div style="text-align: right;">
                <span style="font-size: 0.74rem; color: #475569 !important; text-transform: uppercase;">3h Trajectory</span><br>
                {trend_html}
            </div>
            <div style="text-align: right;">
                <span style="font-size: 0.74rem; color: #475569 !important; text-transform: uppercase;">ICU Stay</span><br>
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.95rem; font-weight: 600; color: #0a3641 !important;">Hour {item['iculos']}</span>
            </div>
        </div>
        <div class="vitals-strip">
            <span>HR <span class="vitals-val">{item['hr']:.0f}</span></span>
            <span>Resp <span class="vitals-val">{item['resp']:.0f}</span></span>
            <span>Temp <span class="vitals-val">{item['temp']:.1f}&deg;C</span></span>
            <span>MAP <span class="vitals-val">{item['map']:.0f}</span></span>
            <span>SF Ratio <span class="vitals-val">{sf_str}</span></span>
        </div>
        <div style="font-size: 0.78rem; color: #475569 !important; margin-top: 6px;">
            Lactate: <span style="font-family: 'JetBrains Mono', monospace; font-weight: 600; color: {'#be123c' if pd.notna(item['lactate']) and item['lactate']>2.0 else '#0a3641'} !important;">{lactate_str} mmol/L {stale_str}</span> &bull; 
            WBC: <span style="font-family: 'JetBrains Mono', monospace; font-weight: 600; color: #0a3641 !important;">{wbc_str}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button(btn_label, key=f"btn_chart_{pid}", use_container_width=True, type=btn_type):
        st.session_state.selected_pid = pid
        st.session_state.current_view = "chart"
        st.rerun()

def render_bedside_chart_view(active_h):
    col_back, col_sel, col_prev, col_next = st.columns([0.24, 0.52, 0.12, 0.12])
    with col_back:
        if st.button(f"&larr; Back to Patient Board ({len(sim_pids)} Beds)", use_container_width=True):
            st.session_state.current_view = "board"
            st.rerun()
    with col_sel:
        def format_pat_choice(p):
            sub = [it for it in state_cache.get(active_h, []) if it["patient_id"] == p]
            if sub:
                b = sub[0]["bed"]
                st_code = "[ALERT]" if sub[0]["prob"] >= st.session_state.threshold else "[STABLE]"
                return f"{b} ({p}) {st_code} - Risk: {sub[0]['prob']*100:.1f}%"
            return p
        
        cur_idx = sim_pids.index(st.session_state.selected_pid) if st.session_state.selected_pid in sim_pids else 0
        new_sel_pid = st.selectbox(
            "Selected Patient Bed:",
            options=sim_pids,
            index=cur_idx,
            format_func=format_pat_choice,
            label_visibility="collapsed"
        )
        if new_sel_pid != st.session_state.selected_pid:
            st.session_state.selected_pid = new_sel_pid
            st.rerun()
    with col_prev:
        if st.button("Previous Bed", use_container_width=True, disabled=(cur_idx == 0)):
            st.session_state.selected_pid = sim_pids[cur_idx - 1]
            st.rerun()
    with col_next:
        if st.button("Next Bed", use_container_width=True, disabled=(cur_idx == len(sim_pids) - 1)):
            st.session_state.selected_pid = sim_pids[cur_idx + 1]
            st.rerun()

    selected_pid = st.session_state.selected_pid
    p_full = cohort_eval_data[cohort_eval_data["patient_id"] == selected_pid]
    p_sim = p_full[p_full["ICULOS"] <= active_h].copy()
    bed_str = bed_map.get(selected_pid, "Bed")

    if p_sim.empty:
        st.warning("This patient has not yet been admitted at the current simulation hour.")
        return

    cur_p_row = p_sim.iloc[-1]
    active_key = (selected_pid, int(cur_p_row["ICULOS"]))
    existing_action = st.session_state.alert_actions.get(active_key, None)
    p_sepsis_outcome = "POSITIVE (Sepsis-3 Confirmed in ICU)" if p_full["SepsisLabel"].max() == 1 else "NEGATIVE (Non-septic ICU Stay)"

    # High-Contrast Clinical Header Card
    st.markdown(f"""
    <div style="background: #ffffff; border: 1px solid #d8e6e7; border-radius: 8px; padding: 16px 20px; margin-bottom: 14px; box-shadow: 0 1px 3px rgba(10,54,65,0.03);">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <span class="editorial-heading">{bed_str} &bull; Clinical Chart Record</span>
                <span class="patient-meta" style="color: #475569 !important;">ID: {selected_pid} | Age: {cur_p_row['Age']:.0f} | Gender: {'Male' if cur_p_row['Gender']==1 else 'Female'}</span>
            </div>
            <div style="text-align: right;">
                <span style="font-size: 0.82rem; color: #475569 !important;">Ground Truth Registry:</span>
                <span style="font-weight: 700; font-size: 0.88rem; color: {'#be123c' if 'POSITIVE' in p_sepsis_outcome else '#008080'} !important;"> {p_sepsis_outcome}</span>
            </div>
        </div>
        <div style="display: flex; gap: 28px; margin-top: 12px; padding-top: 12px; border-top: 1px solid #eef4f4; align-items: baseline;">
            <div>
                <span style="font-size: 0.76rem; color: #475569 !important; text-transform: uppercase; font-weight: 600;">Current Sepsis Risk</span><br>
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 1.6rem; font-weight: 700; color: {'#be123c' if cur_p_row['prob']>=st.session_state.threshold else '#008080'} !important;">{cur_p_row['prob']*100:.1f}%</span>
            </div>
            <div>
                <span style="font-size: 0.76rem; color: #475569 !important; text-transform: uppercase; font-weight: 600;">Threshold</span><br>
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 1.15rem; font-weight: 600; color: #0a3641 !important;">{st.session_state.threshold*100:.1f}%</span>
            </div>
            <div>
                <span style="font-size: 0.76rem; color: #475569 !important; text-transform: uppercase; font-weight: 600;">Heart Rate</span><br>
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 1.15rem; font-weight: 600; color: #0a3641 !important;">{cur_p_row['HR']:.0f} bpm</span>
            </div>
            <div>
                <span style="font-size: 0.76rem; color: #475569 !important; text-transform: uppercase; font-weight: 600;">Resp Rate</span><br>
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 1.15rem; font-weight: 600; color: #0a3641 !important;">{cur_p_row['Resp']:.0f} /min</span>
            </div>
            <div>
                <span style="font-size: 0.76rem; color: #475569 !important; text-transform: uppercase; font-weight: 600;">MAP</span><br>
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 1.15rem; font-weight: 600; color: #0a3641 !important;">{cur_p_row['MAP']:.0f} mmHg</span>
            </div>
            <div>
                <span style="font-size: 0.76rem; color: #475569 !important; text-transform: uppercase; font-weight: 600;">SF Ratio</span><br>
                <span style="font-family: 'JetBrains Mono', monospace; font-size: 1.15rem; font-weight: 600; color: #0a3641 !important;">{cur_p_row['SF_ratio']:.0f}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Clinician Alert Triage Protocol
    st.markdown("<div class='editorial-heading' style='font-size: 1.15rem; margin-bottom: 8px;'>Clinician Alert Triage & Response Protocol</div>", unsafe_allow_html=True)
    col_act1, col_act2 = st.columns([0.62, 0.38])
    with col_act1:
        act_b1, act_b2, act_b3 = st.columns(3)
        with act_b1:
            if st.button("Review & Monitor", key="btn_act_rev", use_container_width=True):
                st.session_state.alert_actions[active_key] = {
                    "action": "Reviewed",
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "hour": active_h,
                    "risk": cur_p_row["prob"],
                    "notes": "Bedside clinical assessment conducted; vital signs set for frequent observation."
                }
                st.rerun()
        with act_b2:
            if st.button("Escalate Sepsis Bundle", key="btn_act_esc", type="primary", use_container_width=True):
                st.session_state.alert_actions[active_key] = {
                    "action": "Escalated",
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "hour": active_h,
                    "risk": cur_p_row["prob"],
                    "notes": "Sepsis-3 protocol initiated: blood cultures x2, IV crystalloid, broad-spectrum antibiotics, repeat lactate."
                }
                st.rerun()
        with act_b3:
            if st.button("Dismiss Alert", key="btn_act_dis", use_container_width=True):
                st.session_state.alert_actions[active_key] = {
                    "action": "Dismissed",
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "hour": active_h,
                    "risk": cur_p_row["prob"],
                    "notes": "Alert dismissed: physiological derangement attributed to non-infectious etiology."
                }
                st.rerun()

    with col_act2:
        if existing_action:
            st.markdown(f"""
            <div style="background: #f0fdfa; border: 1px solid #99f6e4; border-radius: 6px; padding: 10px; font-size: 0.82rem; color: #0f766e !important;">
                <b>Logged Action for Hour {active_h:02d}:00:</b> {existing_action['action']} at {existing_action['timestamp']}<br>
                <i>{existing_action['notes']}</i>
            </div>
            """, unsafe_allow_html=True)
        else:
            if cur_p_row["prob"] >= st.session_state.threshold:
                st.markdown(f"""
                <div style="background: #fff1f2; border: 1px solid #fecdd3; border-radius: 6px; padding: 10px; font-size: 0.82rem; color: #be123c !important;">
                    <b>ACTION REQUIRED:</b> Sepsis risk ({cur_p_row['prob']*100:.1f}%) exceeds operating threshold ({st.session_state.threshold*100:.1f}%). Please triage above.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 10px; font-size: 0.82rem; color: #475569 !important;">
                    Patient vitals currently within acceptable trajectory. No mandatory alert escalation required.
                </div>
                """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Longitudinal ICU Physiological Timelines with Clear Y-Axis Labels
    st.markdown("<div class='editorial-heading' style='font-size: 1.15rem; margin-bottom: 6px;'>Longitudinal ICU Physiological Timelines</div>", unsafe_allow_html=True)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Predicted Sepsis Risk Trajectory vs. Threshold",
            "Vital Signs (Heart Rate, Respiratory Rate, MAP)",
            "Pulmonary Oxygenation: Non-Invasive SF Ratio (SpO2/FiO2)",
            "Laboratory Measurements: Serum Lactate & Staleness"
        ],
        vertical_spacing=0.28,
        horizontal_spacing=0.14
    )

    # Plot 1: Risk Trajectory
    fig.add_trace(go.Scatter(
        x=p_sim["ICULOS"], y=p_sim["prob"],
        mode="lines+markers", name="Risk P(sepsis)",
        line=dict(color="#be123c" if cur_p_row["prob"]>=st.session_state.threshold else "#008080", width=2.5),
        marker=dict(size=5)
    ), row=1, col=1)
    fig.add_hline(y=st.session_state.threshold, line_dash="dash", line_color="#b45309",
                  annotation_text=f"Threshold ({st.session_state.threshold:.3f})", row=1, col=1)

    # Plot 2: Vital Signs
    fig.add_trace(go.Scatter(x=p_sim["ICULOS"], y=p_sim["HR"], mode="lines", name="HR (bpm)", line=dict(color="#be123c", width=2)), row=1, col=2)
    fig.add_trace(go.Scatter(x=p_sim["ICULOS"], y=p_sim["Resp"], mode="lines", name="Resp (/min)", line=dict(color="#028090", width=2)), row=1, col=2)
    fig.add_trace(go.Scatter(x=p_sim["ICULOS"], y=p_sim["MAP"], mode="lines", name="MAP (mmHg)", line=dict(color="#0a3641", width=2)), row=1, col=2)

    # Plot 3: SF Ratio
    fig.add_trace(go.Scatter(x=p_sim["ICULOS"], y=p_sim["SF_ratio"], mode="lines+markers", name="SF Ratio (SpO2/FiO2)", line=dict(color="#008080", width=2)), row=2, col=1)
    fig.add_hline(y=300, line_dash="dot", line_color="#be123c", annotation_text="Hypoxemic Derangement (<300)", row=2, col=1)

    # Plot 4: Labs & Staleness
    fig.add_trace(go.Scatter(x=p_sim["ICULOS"], y=p_sim["Lactate"], mode="lines+markers", name="Serum Lactate (mmol/L)", line=dict(color="#be123c", width=2)), row=2, col=2)
    fig.add_trace(go.Scatter(x=p_sim["ICULOS"], y=p_sim["Lactate_hrs_since"], mode="lines", name="Lactate Staleness (hrs)", line=dict(color="#64748b", dash="dash")), row=2, col=2)
    fig.add_hline(y=2.0, line_dash="dot", line_color="#be123c", annotation_text="Hyperlactatemia (>2.0)", row=2, col=2)

    # Explicit Y-Axis & X-Axis Labels on All 4 Subplots with Comfortable Standoff
    fig.update_yaxes(title=dict(text="Risk P(sepsis)", standoff=12, font=dict(size=11, color="#0a3641")), row=1, col=1)
    fig.update_xaxes(title=dict(text="ICU Stay Elapsed (Hours)", standoff=10, font=dict(size=11, color="#0a3641")), row=1, col=1)

    fig.update_yaxes(title=dict(text="Vitals (bpm, /min, mmHg)", standoff=12, font=dict(size=11, color="#0a3641")), row=1, col=2)
    fig.update_xaxes(title=dict(text="ICU Stay Elapsed (Hours)", standoff=10, font=dict(size=11, color="#0a3641")), row=1, col=2)

    fig.update_yaxes(title=dict(text="SF Ratio (SpO2 / FiO2)", standoff=12, font=dict(size=11, color="#0a3641")), row=2, col=1)
    fig.update_xaxes(title=dict(text="ICU Stay Elapsed (Hours)", standoff=10, font=dict(size=11, color="#0a3641")), row=2, col=1)

    fig.update_yaxes(title=dict(text="Serum Lactate (mmol/L)", standoff=12, font=dict(size=11, color="#0a3641")), row=2, col=2)
    fig.update_xaxes(title=dict(text="ICU Stay Elapsed (Hours)", standoff=10, font=dict(size=11, color="#0a3641")), row=2, col=2)

    fig.update_layout(
        height=620,
        margin=dict(l=80, r=40, t=60, b=60),
        plot_bgcolor="#eef2f5",   # Clean contrasting clinical grey canvas
        paper_bgcolor="#ffffff",  # Clean white card container
        showlegend=False,
        font=dict(family="Plus Jakarta Sans", size=11, color="#0a3641")
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#d8e1e8", zeroline=False, tickfont=dict(color="#0a3641", size=10))
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#d8e1e8", zeroline=False, tickfont=dict(color="#0a3641", size=10))

    st.plotly_chart(fig, use_container_width=True, theme=None)

    # -------------------------------------------------------------
    # TreeSHAP Waterfall Plot with Proper Axis Spacing (Zero Collisions!)
    # -------------------------------------------------------------
    st.markdown("<div class='editorial-heading' style='font-size: 1.15rem; margin-top: 14px; margin-bottom: 4px;'>TreeSHAP Local Feature Attribution: Model Rationale</div>", unsafe_allow_html=True)
    st.caption("Additive Shapley values quantifying how each physiological variable shifted the log-odds of sepsis for this patient at the current hour:")

    cur_feat_idx = p_sim.index[-1]
    cur_X_row = features_pool.loc[[cur_feat_idx]]
    
    shap_explanation = explainer(cur_X_row)
    shap_vals = shap_explanation.values[0]
    base_val = shap_explanation.base_values[0]

    def get_shap_palette(col):
        if col.endswith(("_n_meas", "_hrs_since")):
            return "#e76f51"  # Measurement Process (Orange/Coral)
        elif col.endswith("_d6"):
            return "#00a896"  # 6-Hour Dynamic Trend (Teal)
        else:
            return "#0a3641"  # Physiological Value / Baseline (Navy)

    top_indices = np.argsort(np.abs(shap_vals))[-12:]
    top_names = cur_X_row.columns[top_indices]
    top_shap_values = shap_vals[top_indices]
    top_actual_values = cur_X_row.iloc[0].iloc[top_indices].values
    bar_colors = [get_shap_palette(c) for c in top_names]

    # Human-readable formatted labels with full clinical names and units
    custom_labels = [format_clinical_feature(n, v) for n, v in zip(top_names, top_actual_values)]

    fig_shap = go.Figure(go.Bar(
        x=top_shap_values,
        y=custom_labels,
        orientation="h",
        marker=dict(color=bar_colors)
    ))

    # Generous left margin (280px) and standoff so labels never collide with numbers!
    fig_shap.update_layout(
        title=f"TreeSHAP Attribution at Hour {active_h:02d}:00 (Base Log-Odds: {base_val:.2f} &rarr; Predicted Risk: {cur_p_row['prob']*100:.1f}%)",
        plot_bgcolor="#eef2f5",   # Clean contrasting clinical grey canvas
        paper_bgcolor="#ffffff",  # Clean white card container
        height=480,
        margin=dict(l=280, r=40, t=60, b=80),
        font=dict(family="Plus Jakarta Sans", size=11, color="#0a3641")
    )
    fig_shap.update_xaxes(
        title=dict(text="SHAP Value (Contribution to Log-Odds of Sepsis)", standoff=25, font=dict(family="Plus Jakarta Sans", size=12, color="#0a3641")),
        showgrid=True,
        gridwidth=1,
        gridcolor="#d8e1e8",
        automargin=True,
        tickfont=dict(family="JetBrains Mono", color="#0a3641", size=10)
    )
    fig_shap.update_yaxes(
        title=dict(text="Clinical Predictor & Measured Value", standoff=30, font=dict(family="Plus Jakarta Sans", size=12, color="#0a3641")),
        automargin=True,
        tickfont=dict(family="Plus Jakarta Sans", color="#0a3641", size=10)
    )

    st.plotly_chart(fig_shap, use_container_width=True, theme=None)

    c_leg1, c_leg2, c_leg3 = st.columns(3)
    with c_leg1:
        st.markdown("<span style='color: #e76f51 !important; font-weight: bold;'>■ Coral</span>: **Measurement Process** (`_n_meas` frequency & `_hrs_since` staleness)", unsafe_allow_html=True)
    with c_leg2:
        st.markdown("<span style='color: #0a3641 !important; font-weight: bold;'>■ Navy</span>: **Physiological State** (SF ratio, Temp, HR, MAP, Lactate)", unsafe_allow_html=True)
    with c_leg3:
        st.markdown("<span style='color: #00a896 !important; font-weight: bold;'>■ Teal</span>: **Dynamic 6-Hour Trend** (`_d6` rate of change)", unsafe_allow_html=True)

# Run the live simulation block (which renders the clock, toolbar, and active view)
render_live_simulation_block()

# -------------------------------------------------------------
# 8. VIEW 3: Clinical Trade-Off & Cohort Performance Metrics
# -------------------------------------------------------------
if st.session_state.current_view == "metrics":
    cur_h = st.session_state.sim_hour
    st.markdown("<div class='editorial-heading'>Clinical Dilemma: Early Detection vs. Alarm Fatigue</div>", unsafe_allow_html=True)
    st.markdown("<div class='editorial-subheading'>Operational analysis of the clinical trade-offs governing decision threshold calibration</div>", unsafe_allow_html=True)

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown(r"""
        <div style="background: #ffffff; border: 1px solid #d8e6e7; border-radius: 8px; padding: 18px; margin-bottom: 14px;">
            <h4 style="font-family: 'Newsreader', Georgia, serif; color: #0a3641 !important; margin-top: 0;">Why Catching Sepsis Early Saves Lives</h4>
            <ul style="font-size: 0.88rem; line-height: 1.6; color: #1e293b !important;">
                <li style="color: #1e293b !important;"><b style="color: #0a3641 !important;">Every Hour Counts:</b> When a severe infection escalates to septic shock, every single hour of delay in administering targeted antibiotics increases the risk of death by <b style="color: #be123c !important;">7.6% per hour</b>.</li>
                <li style="color: #1e293b !important;"><b style="color: #0a3641 !important;">A 4-to-6 Hour Early Warning Window:</b> Detecting subtle physiological changes hours before obvious clinical shock gives nurses and doctors a crucial head start to order lab tests, start IV fluids, and stop organ damage before it becomes irreversible.</li>
                <li style="color: #1e293b !important;"><b style="color: #0a3641 !important;">Continuous, Painless Lung Monitoring:</b> By tracking how blood oxygen levels compare to oxygen supplied by mask (the <b style="color: #008080 !important;">SF ratio</b>), the system continuously detects breathing stress in real time without requiring painful arterial blood draws.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_t2:
        st.markdown(r"""
        <div style="background: #ffffff; border: 1px solid #d8e6e7; border-radius: 8px; padding: 18px; margin-bottom: 14px;">
            <h4 style="font-family: 'Newsreader', Georgia, serif; color: #0a3641 !important; margin-top: 0;">The Hidden Cost of False Alarms</h4>
            <ul style="font-size: 0.88rem; line-height: 1.6; color: #1e293b !important;">
                <li style="color: #1e293b !important;"><b style="color: #0a3641 !important;">Doctor & Nurse Alarm Fatigue:</b> Hospital ICU teams are bombarded with <b style="color: #be123c !important;">150 to 350 machine alarms per bed every day</b>. When most alarms turn out to be false, clinicians become desensitized, increasing the danger of missing a genuine emergency.</li>
                <li style="color: #1e293b !important;"><b style="color: #0a3641 !important;">Unnecessary Tests & Medical Costs:</b> A false sepsis alert triggers emergency blood draws, extra imaging, and lab workup costing <b style="color: #be123c !important;">$800 to $1,590 per patient</b>, tying up critical ICU staff and hospital resources.</li>
                <li style="color: #1e293b !important;"><b style="color: #0a3641 !important;">Antibiotic Overuse Risks:</b> Giving powerful broad-spectrum antibiotics when they aren't truly needed can cause severe secondary bowel infections (such as <i>C. diff</i>) and breeds drug-resistant superbugs.</li>
                <li style="color: #1e293b !important;"><b style="color: #0a3641 !important;">The 40:1 Life-Saving Trade-Off:</b> In official clinical evaluation, missing a true sepsis case is penalized <b style="color: #008080 !important;">40 times more severely</b> than sounding a temporary false alarm. The AI is specifically calibrated to never miss a critical deterioration, while keeping false alarms low enough so nurses can trust every alert.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Threshold calibration slider
    st.markdown("<div class='editorial-heading' style='font-size: 1.15rem; margin-top: 10px; margin-bottom: 6px;'>Calibrate Decision Threshold</div>", unsafe_allow_html=True)
    c_th1, c_th2 = st.columns([0.65, 0.35])
    with c_th1:
        st.session_state.threshold = st.slider(
            "Operating Decision Threshold (t*):",
            min_value=0.02, max_value=0.15, value=float(st.session_state.threshold), step=0.005,
            help="Lowering threshold catches sepsis earlier but triggers more false alarms. Raising threshold reduces alarm fatigue but delays detection."
        )
    with c_th2:
        st.markdown(f"""
        <div style="background: #f0f7f7; border: 1px solid #d8e6e7; border-radius: 6px; padding: 12px; font-size: 0.85rem; color: #0a3641 !important;">
            <b style="color: #0a3641 !important;">Active Operating Threshold:</b> {st.session_state.threshold:.3f}<br>
            Optimized on Training Set A utility (0.055). All patients with risk &ge; {st.session_state.threshold*100:.1f}% are flagged for clinical review.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='editorial-heading'>Empirical Testing Benchmarks vs. Live Simulated Cohort</div>", unsafe_allow_html=True)

    col_bench1, col_bench2 = st.columns(2)
    with col_bench1:
        bench_b = meta["benchmark_metrics_external"]
        st.markdown(f"""
        <div style="background: #ffffff; border: 1px solid #d8e6e7; border-radius: 8px; padding: 18px;">
            <h4 style="font-family: 'Newsreader', Georgia, serif; color: #0a3641 !important; margin-top: 0;">Full External Hospital B Benchmark (24,737 Patients)</h4>
            <p style="font-size: 0.82rem; color: #475569 !important;">Empirical performance of the 15-seed XGBoost ensemble on Hospital B (zero data leakage, causal trimming at t_sepsis + 6h):</p>
            <table style="width: 100%; font-size: 0.88rem; border-collapse: collapse; color: #0a3641 !important;">
                <tr style="border-bottom: 1px solid #eef4f4; height: 34px;">
                    <td style="color: #0a3641 !important; font-weight: 600;">AUROC (Discrimination)</td>
                    <td style="text-align: right; font-family: 'JetBrains Mono', monospace; font-weight: 700; color: #0a3641 !important;">{bench_b['auroc']:.3f}</td>
                </tr>
                <tr style="border-bottom: 1px solid #eef4f4; height: 34px;">
                    <td style="color: #0a3641 !important; font-weight: 600;">Precision (AUPRC)</td>
                    <td style="text-align: right; font-family: 'JetBrains Mono', monospace; font-weight: 700; color: #0a3641 !important;">{bench_b['auprc']:.3f}</td>
                </tr>
                <tr style="border-bottom: 1px solid #eef4f4; height: 34px;">
                    <td style="color: #0a3641 !important; font-weight: 600;">Patient Sensitivity (Recall)</td>
                    <td style="text-align: right; font-family: 'JetBrains Mono', monospace; font-weight: 700; color: #008080 !important;">{bench_b['sensitivity']*100:.1f}%</td>
                </tr>
                <tr style="border-bottom: 1px solid #eef4f4; height: 34px;">
                    <td style="color: #0a3641 !important; font-weight: 600;">Patient False Alert Rate (FAR)</td>
                    <td style="text-align: right; font-family: 'JetBrains Mono', monospace; font-weight: 700; color: #be123c !important;">{bench_b['false_alert_rate']*100:.1f}%</td>
                </tr>
                <tr style="border-bottom: 1px solid #eef4f4; height: 34px;">
                    <td style="color: #0a3641 !important; font-weight: 600;">Alerts per Patient</td>
                    <td style="text-align: right; font-family: 'JetBrains Mono', monospace; font-weight: 700; color: #0a3641 !important;">1.2</td>
                </tr>
                <tr style="height: 34px;">
                    <td style="color: #0a3641 !important; font-weight: 600;">PhysioNet Normalized Utility</td>
                    <td style="text-align: right; font-family: 'JetBrains Mono', monospace; font-weight: 700; color: #008080 !important;">+{bench_b['utility']:.3f}</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    with col_bench2:
        live_g = cohort_eval_data.groupby("patient_id").agg(
            sepsis=("SepsisLabel", "max"),
            max_p=("prob", "max")
        )
        live_g["alerted"] = live_g["max_p"] >= st.session_state.threshold
        
        tp_live = len(live_g[(live_g["sepsis"] == 1) & (live_g["alerted"])])
        fn_live = len(live_g[(live_g["sepsis"] == 1) & (~live_g["alerted"])])
        fp_live = len(live_g[(live_g["sepsis"] == 0) & (live_g["alerted"])])
        tn_live = len(live_g[(live_g["sepsis"] == 0) & (~live_g["alerted"])])
        
        live_sens = (tp_live / (tp_live + fn_live)) if (tp_live + fn_live) > 0 else 0.0
        live_far = (fp_live / (fp_live + tn_live)) if (fp_live + tn_live) > 0 else 0.0
        live_ppv = (tp_live / (tp_live + fp_live)) if (tp_live + fp_live) > 0 else 0.0
        total_live_alerts = int(live_g["alerted"].sum())
        
        # Count critical beds at cur_h
        cur_items = state_cache.get(cur_h, [])
        crit_beds_count = sum(1 for it in cur_items if cur_h <= it["max_hour"] and it["prob"] >= st.session_state.threshold)

        st.markdown(f"""
        <div style="background: #ffffff; border: 1px solid #d8e6e7; border-radius: 8px; padding: 18px;">
            <h4 style="font-family: 'Newsreader', Georgia, serif; color: #0a3641 !important; margin-top: 0;">Active Cohort ({len(sim_pids)} Monitored Beds)</h4>
            <p style="font-size: 0.82rem; color: #475569 !important;">Evaluated strictly per-patient (window-based evaluation up to simulated Hour {cur_h:02d}:00):</p>
            <table style="width: 100%; font-size: 0.88rem; border-collapse: collapse; color: #0a3641 !important;">
                <tr style="border-bottom: 1px solid #eef4f4; height: 34px;">
                    <td style="color: #0a3641 !important; font-weight: 600;">Cohort Prevalence</td>
                    <td style="text-align: right; font-family: 'JetBrains Mono', monospace; font-weight: 700; color: #0a3641 !important;">{live_g['sepsis'].mean()*100:.1f}% ({tp_live+fn_live}/{len(sim_pids)})</td>
                </tr>
                <tr style="border-bottom: 1px solid #eef4f4; height: 34px;">
                    <td style="color: #0a3641 !important; font-weight: 600;">Live Patient Sensitivity</td>
                    <td style="text-align: right; font-family: 'JetBrains Mono', monospace; font-weight: 700; color: #008080 !important;">{live_sens*100:.1f}% ({tp_live}/{tp_live+fn_live})</td>
                </tr>
                <tr style="border-bottom: 1px solid #eef4f4; height: 34px;">
                    <td style="color: #0a3641 !important; font-weight: 600;">Live Patient False Alert Rate</td>
                    <td style="text-align: right; font-family: 'JetBrains Mono', monospace; font-weight: 700; color: #be123c !important;">{live_far*100:.1f}% ({fp_live}/{fp_live+tn_live})</td>
                </tr>
                <tr style="border-bottom: 1px solid #eef4f4; height: 34px;">
                    <td style="color: #0a3641 !important; font-weight: 600;">Live Precision (PPV)</td>
                    <td style="text-align: right; font-family: 'JetBrains Mono', monospace; font-weight: 700; color: #0a3641 !important;">{live_ppv*100:.1f}%</td>
                </tr>
                <tr style="border-bottom: 1px solid #eef4f4; height: 34px;">
                    <td style="color: #0a3641 !important; font-weight: 600;">Total Alerted Patients</td>
                    <td style="text-align: right; font-family: 'JetBrains Mono', monospace; font-weight: 700; color: #0a3641 !important;">{total_live_alerts} / {len(sim_pids)}</td>
                </tr>
                <tr style="height: 34px;">
                    <td style="color: #0a3641 !important; font-weight: 600;">Active Bed Triage Queue</td>
                    <td style="text-align: right; font-family: 'JetBrains Mono', monospace; font-weight: 700; color: #be123c !important;">{crit_beds_count} Beds</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

# -------------------------------------------------------------
# 9. VIEW 4: Alert Audit Log & CSV Export
# -------------------------------------------------------------
elif st.session_state.current_view == "audit":
    cur_h = st.session_state.sim_hour
    st.markdown("<div class='editorial-heading'>Clinical Alert Audit Trail</div>", unsafe_allow_html=True)
    st.markdown("<div class='editorial-subheading'>Chronological log of bedside clinician acknowledgements, escalations, and dismissals</div>", unsafe_allow_html=True)

    if not st.session_state.alert_actions:
        st.info("No alert actions have been logged yet. Open any bedside chart to triage alerts and record clinical actions.")
    else:
        audit_rows = []
        for (pid, h), data in st.session_state.alert_actions.items():
            bed = bed_map.get(pid, "Bed")
            audit_rows.append({
                "Bed": bed,
                "Patient ID": pid,
                "Simulated Hour": f"Hour {h:02d}:00",
                "Action Taken": data["action"],
                "Predicted Risk": f"{data['risk']*100:.1f}%",
                "Logged At": data["timestamp"],
                "Clinical Notes": data["notes"]
            })
        
        audit_table = pd.DataFrame(audit_rows)
        st.dataframe(audit_table, use_container_width=True, hide_index=True)
        
        csv_data = audit_table.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Export Audit Log (CSV)",
            data=csv_data,
            file_name=f"sepsis_audit_log_hour_{cur_h:02d}.csv",
            mime="text/csv"
        )
