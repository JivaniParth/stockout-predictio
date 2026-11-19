"""
Enterprise Streamlit Entry Point for Healthcare Supply Chain Medicine Stockout Predictor.
Combined Research Project: Cloud Computing (CC) & Deep Learning (DL).
Author: Parth Jivani | Supervisor: Dr. Yogesh Chaudhari | Navrachana University
"""

import sys
import pickle
from datetime import date
from pathlib import Path
from typing import Dict, Any, Tuple
import torch
import streamlit as st

# 1. Page Configuration
st.set_page_config(
    page_title="HSCM Stockout Risk Predictor (CC + DL)",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Path Setup
CURRENT_DIR = Path(__file__).resolve().parent
if (CURRENT_DIR / "src").exists():
    BASE_DIR = CURRENT_DIR
else:
    BASE_DIR = CURRENT_DIR / "Implementation"

sys.path.insert(0, str(BASE_DIR))

# 3. Import UI Architecture & Design System Modules
from src.ui.theme import inject_custom_theme
from src.ui.state import SessionStateManager
from src.ui.components.primitives import render_header
from src.ui.views.prediction_view import render_prediction_tab
from src.ui.views.batch_view import render_batch_tab
from src.ui.views.benchmarks_view import render_benchmarks_tab
from src.ui.views.architecture_view import render_architecture_tab

from src.container import Container

RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = RESULTS_DIR / "models"
FIGURES_DIR = RESULTS_DIR / "figures"
DATA_PATH = BASE_DIR / "SCMS_Delivery_History_Dataset.csv"

# 4. Inject Theme & Initialize State
inject_custom_theme()
SessionStateManager.initialize()


# 5. Access Dependency Injection Container
@st.cache_resource
def get_app_container():
    return Container.get_instance(models_dir=MODELS_DIR)


container = get_app_container()
preprocessor = container.get_preprocessor()
model_names = container.get_available_model_names()

# 6. Sidebar Configuration
st.sidebar.header("Model & Inference Configuration")

selected_model_name = st.sidebar.selectbox(
    "Select Forecasting Architecture:",
    model_names,
    index=0,
    help="Choose between Deep Learning sequence/attention models or classical tabular baselines.",
)

active_predictor = container.get_predictor(selected_model_name)
default_thresh = active_predictor.optimal_threshold

# Adapter wrapper for backwards compatibility with any views expecting models_dict
models_dict = {
    selected_model_name: {
        "model": active_predictor,
        "type": "clean_adapter",
        "threshold": default_thresh,
    }
}


use_calibrated = st.sidebar.checkbox(
    f"Use Validation-Calibrated Threshold (τ* = {default_thresh:.2f})",
    value=True,
    help="Decision threshold calibrated on the 2014 prospective validation split to maximize minority-class F1 score.",
)

if use_calibrated:
    decision_threshold = default_thresh
else:
    decision_threshold = st.sidebar.slider(
        "Custom Decision Threshold (τ):",
        0.05,
        0.95,
        default_thresh,
        0.01,
        help="Lowering threshold increases recall (catches more stockouts at risk of false alarms); raising threshold increases precision.",
    )

st.sidebar.markdown("---")
st.sidebar.subheader("Preload Test Scenarios")
preset_choice = st.sidebar.radio(
    "Quick Preset Scenarios:",
    [
        "None (Custom Input)",
        "High-Risk Underserved (Congo DRC — Pediatric ARV)",
        "Moderate-Risk Regional Node (Côte d'Ivoire — Adult ARV)",
        "Low-Risk Central Hub (South Africa — HIV Rapid Test)",
    ],
    index=0,
    help="Select a representative real-world pharmaceutical consignment to populate input parameters instantly.",
)

preset_vals = {}
if "Congo DRC" in preset_choice:
    preset_vals = {
        "Source_Country": "India",
        "Destination_Country": "Congo, DRC",
        "Scheduled_Delivery_Date": date(2026, 11, 20),
        "PO_Sent_to_Vendor_Date": date(2026, 8, 1),
        "Manufacturer_Lead_Time_Days": 120.0,
        "Product_Group": "ARV",
        "Sub_Classification": "Pediatric",
        "Shipment_Mode": "Air",
        "Fulfill_Via": "Direct Drop",
        "Vendor_INCO_Term": "CIP",
        "Vendor": "Aurobindo Pharma Limited",
        "Manufacturing_Site": "Aurobindo Unit III, India",
        "First_Line_Designation": "Yes",
        "Line_Item_Quantity": 25000.0,
        "Line_Item_Value": 125000.0,
        "Pack_Price": 5.0,
        "Unit_Price": 0.08,
        "Unit_of_Measure_Per_Pack": 60,
        "Weight_Kilograms": 1450.0,
        "Freight_Cost_USD": 8900.0,
        "Line_Item_Insurance_USD": 140.0,
    }
elif "Côte d'Ivoire" in preset_choice:
    preset_vals = {
        "Source_Country": "India",
        "Destination_Country": "Cte d'Ivoire",
        "Scheduled_Delivery_Date": date(2026, 9, 15),
        "PO_Sent_to_Vendor_Date": date(2026, 7, 10),
        "Manufacturer_Lead_Time_Days": 90.0,
        "Product_Group": "ARV",
        "Sub_Classification": "Adult",
        "Shipment_Mode": "Air",
        "Fulfill_Via": "Direct Drop",
        "Vendor_INCO_Term": "EXW",
        "Vendor": "Cipla Limited",
        "Manufacturing_Site": "Cipla, Patalganga, India",
        "First_Line_Designation": "Yes",
        "Line_Item_Quantity": 15000.0,
        "Line_Item_Value": 65000.0,
        "Pack_Price": 4.33,
        "Unit_Price": 0.07,
        "Unit_of_Measure_Per_Pack": 60,
        "Weight_Kilograms": 820.0,
        "Freight_Cost_USD": 4500.0,
        "Line_Item_Insurance_USD": 75.0,
    }
elif "South Africa" in preset_choice:
    preset_vals = {
        "Source_Country": "South Africa",
        "Destination_Country": "South Africa",
        "Scheduled_Delivery_Date": date(2026, 10, 15),
        "PO_Sent_to_Vendor_Date": date(2026, 9, 1),
        "Manufacturer_Lead_Time_Days": 30.0,
        "Product_Group": "HRDT",
        "Sub_Classification": "HIV test",
        "Shipment_Mode": "Truck",
        "Fulfill_Via": "From RDC",
        "Vendor_INCO_Term": "EXW",
        "Vendor": "Ranbaxy Fine Chemicals LTD.",
        "Manufacturing_Site": "Ranbaxy Fine Chemicals LTD",
        "First_Line_Designation": "Yes",
        "Line_Item_Quantity": 1000.0,
        "Line_Item_Value": 29000.0,
        "Pack_Price": 29.0,
        "Unit_Price": 0.97,
        "Unit_of_Measure_Per_Pack": 30,
        "Weight_Kilograms": 80.0,
        "Freight_Cost_USD": 650.0,
        "Line_Item_Insurance_USD": 35.0,
    }

# 7. Render App Header
render_header(
    title="Cloud-Native Medicine Stockout & Delivery Risk Predictor",
    subtitle="Combined Research Project: Cloud Computing (CC) & Deep Learning (DL) | "
             "Author: Parth Jivani | Supervisor: Dr. Yogesh Chaudhari, Prof. Rupali Shinde | Navrachana University",
)

# 8. Render Main Tabs Navigation
tabs = st.tabs([
    "1. Real-Time Shipment Risk Prediction",
    "2. High-Throughput Batch Inference",
    "3. Research Gaps & Model Benchmarks",
    "4. Cloud Architecture & Microservice QoS",
])

# Tab 1: Real-Time Prediction View
with tabs[0]:
    render_prediction_tab(
        preprocessor=preprocessor,
        models_dict=models_dict,
        selected_model_name=selected_model_name,
        decision_threshold=decision_threshold,
        preset_vals=preset_vals,
    )

# Tab 2: Batch Inference View
with tabs[1]:
    render_batch_tab(
        preprocessor=preprocessor,
        models_dict=models_dict,
        selected_model_name=selected_model_name,
        decision_threshold=decision_threshold,
        data_path=DATA_PATH,
    )

# Tab 3: Benchmarks & Literature Gaps View
with tabs[2]:
    render_benchmarks_tab(
        results_dir=RESULTS_DIR,
        figures_dir=FIGURES_DIR,
    )

# Tab 4: Architecture & QoS Telemetry View
with tabs[3]:
    render_architecture_tab()
