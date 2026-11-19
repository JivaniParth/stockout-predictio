"""
Batch Inference View (Tab 2) for Streamlit HSCM Platform.
High-Throughput Batch Inference, Filterable Grid, and Inventory Sync.
"""

from typing import Dict, Any
import time
from pathlib import Path
import pandas as pd
import streamlit as st

from src.ui.components.primitives import (
    render_empty_state,
    render_metric_card,
    render_skeleton,
    render_alert,
)
from src.ui.state import SessionStateManager


def render_batch_tab(
    preprocessor: Any,
    models_dict: Dict[str, Any],
    selected_model_name: str,
    decision_threshold: float,
    data_path: Path,
):
    """Renders the High-Throughput Batch Inference & Inventory Synchronization View."""
    st.subheader("High-Throughput Batch Inference & Inventory Synchronization")
    st.markdown(
        "Upload multiple shipment records or test prospective consignments simultaneously. "
        "The system evaluates **batch throughput (samples/sec)** and highlights impending facility stockouts."
    )

    batch_df = SessionStateManager.get_batch_data()

    col_btn1, col_btn2 = st.columns([1.5, 2.5])
    with col_btn1:
        load_sample = st.button("Load 25 Prospective Test Samples (2015 Unseen Data)", type="secondary")
    with col_btn2:
        if batch_df is not None:
            if st.button("Clear Batch Dataset", type="tertiary"):
                SessionStateManager.clear_batch_data()
                st.rerun()

    # Load 25 prospective test samples on demand
    if load_sample:
        raw_full = pd.read_csv(data_path)
        sched_d = pd.to_datetime(raw_full["Scheduled Delivery Date"], errors="coerce", format="mixed")
        test_2015 = raw_full[sched_d.dt.year >= 2015].sample(n=25, random_state=42).copy()
        SessionStateManager.set_batch_data(test_2015)
        batch_df = test_2015
        st.rerun()

    # Empty State Handling
    if batch_df is None or len(batch_df) == 0:
        render_empty_state(
            title="No Batch Dataset Loaded",
            description="Click 'Load 25 Prospective Test Samples' above to populate unseen 2015 pharmaceutical consignments, or upload an operational delivery CSV.",
            action_label="Load 25 Prospective Samples",
            action_key="empty_state_load_btn",
            on_action=lambda: None,
        )
        return

    st.write(f"Loaded **{len(batch_df)}** shipments ready for inference:")
    display_cols = [
        "ID", "Country", "Product Group", "Shipment Mode",
        "Line Item Quantity", "Line Item Value", "Scheduled Delivery Date",
    ]
    st.dataframe(batch_df[[c for c in display_cols if c in batch_df.columns]].head(8), use_container_width=True)

    # Trigger Batch Inference
    run_batch = st.button("Run High-Speed Batch Inference on All Records", type="primary", use_container_width=True)

    if run_batch:
        progress_bar = st.progress(0, text="Initializing high-throughput batch vectorization...")
        total_rows = len(batch_df)

        from src.container import Container
        from src.domain.entities import ShipmentEntity

        entities = []
        for idx, (_, r) in enumerate(batch_df.iterrows()):
            hist_sched = pd.to_datetime(r["Scheduled Delivery Date"], errors="coerce")
            hist_po = pd.to_datetime(r["PO Sent to Vendor Date"], errors="coerce")
            mfg_lt_hist = (
                (hist_sched - hist_po).days
                if (pd.notna(hist_sched) and pd.notna(hist_po) and (hist_sched - hist_po).days > 0)
                else 90.0
            )

            ins_val = r.get("Line Item Insurance (USD)", 50.0)
            ins_clean = float(ins_val) if pd.notna(ins_val) else 50.0

            entities.append(
                ShipmentEntity(
                    country=str(r["Country"]),
                    scheduled_delivery_date=str(r["Scheduled Delivery Date"]),
                    po_sent_to_vendor_date=str(r["PO Sent to Vendor Date"]),
                    manufacturer_lead_time_days=float(mfg_lt_hist),
                    product_group=str(r["Product Group"]),
                    sub_classification=str(r["Sub Classification"]),
                    shipment_mode=str(r["Shipment Mode"]),
                    fulfill_via=str(r["Fulfill Via"]),
                    vendor_inco_term=str(r["Vendor INCO Term"]),
                    vendor=str(r["Vendor"]),
                    manufacturing_site=str(r["Manufacturing Site"]),
                    first_line_designation=str(r["First Line Designation"]),
                    line_item_quantity=float(r["Line Item Quantity"]),
                    line_item_value=float(r["Line Item Value"]),
                    pack_price=float(r["Pack Price"]),
                    unit_price=float(r["Unit Price"]),
                    unit_of_measure_per_pack=float(r["Unit of Measure (Per Pack)"]),
                    weight_kilograms=str(r["Weight (Kilograms)"]),
                    freight_cost_usd=str(r["Freight Cost (USD)"]),
                    line_item_insurance_usd=ins_clean,
                )
            )

        progress_bar.progress(50, text="Executing vectorized forward pass via BatchPredictShipmentsUseCase...")
        container = Container.get_instance()
        batch_uc = container.get_batch_predict_use_case(selected_model_name)
        prognoses, total_lat_ms, throughput = batch_uc.execute(entities, custom_threshold=decision_threshold)

        flags_list = [p.late_delivery_risk for p in prognoses]
        probs_list = [p.risk_probability for p in prognoses]

        progress_bar.progress(100, text="Batch processing complete!")
        progress_bar.empty()


        # Update dataframe
        batch_df = batch_df.copy()
        batch_df["Predicted_Late_Risk"] = flags_list
        batch_df["Risk_Probability"] = [round(p, 4) for p in probs_list]
        batch_df["Risk_Label"] = ["High Risk (Stockout Alert)" if f == 1 else "On-Time" for f in flags_list]
        SessionStateManager.set_batch_data(batch_df)

        # Telemetry Metrics Grid
        st.markdown("#### Batch QoS Execution Telemetry")
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            render_metric_card("Total Shipments", str(total_rows), meta="Batch record volume")
        with m2:
            at_risk_count = sum(flags_list)
            render_metric_card(
                "Flagged At-Risk",
                str(at_risk_count),
                delta=f"{(at_risk_count / total_rows) * 100:.1f}% Risk Rate",
                delta_positive=False,
                meta="Impending stockout warnings",
            )
        with m3:
            render_metric_card("Execution Latency", f"{total_lat_ms:.1f} ms", meta="Total forward pass time")
        with m4:
            render_metric_card(
                "Throughput Capacity",
                f"{throughput:.1f} rec/s",
                delta="Real-Time Scalable",
                delta_positive=True,
                meta="Microservice inference speed",
            )

    # Render results table if predictions exist
    if "Risk_Label" in batch_df.columns:
        st.write("##### Predicted Consignments & Risk Stratification:")

        filter_choice = st.radio(
            "Filter Table by Risk Status:",
            ["All Records", "High Risk (Stockout Alerts Only)", "On-Time Records Only"],
            horizontal=True,
        )

        filtered_df = batch_df
        if filter_choice == "High Risk (Stockout Alerts Only)":
            filtered_df = batch_df[batch_df["Predicted_Late_Risk"] == 1]
        elif filter_choice == "On-Time Records Only":
            filtered_df = batch_df[batch_df["Predicted_Late_Risk"] == 0]

        if len(filtered_df) == 0:
            render_alert("No records match the selected filter criteria.", variant="info")
        else:
            show_cols = [
                "ID", "Country", "Product Group", "Shipment Mode",
                "Line Item Quantity", "Risk_Probability", "Risk_Label",
            ]
            st.dataframe(filtered_df[[c for c in show_cols if c in filtered_df.columns]], use_container_width=True)

            csv_data = filtered_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Filtered Predictions CSV",
                data=csv_data,
                file_name="batch_stockout_predictions.csv",
                mime="text/csv",
                type="primary",
            )
