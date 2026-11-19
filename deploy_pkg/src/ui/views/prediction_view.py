"""
Prediction View (Tab 1) for Streamlit HSCM Platform.
Real-Time Consignment Risk & Medicine Stockout Assessment.
"""

from typing import Dict, Any, Tuple, Optional
from datetime import date
import time
import pandas as pd
import streamlit as st

from src.container import Container
from src.domain.entities import ShipmentEntity
from src.ui.components.primitives import render_alert, render_empty_state
from src.ui.components.telemetry import (
    render_route_corridor_banner,
    render_qos_telemetry_bar,
    render_interactive_corridor_map,
)
from src.ui.components.scorecards import render_lead_time_tracker, render_prediction_scorecard
from src.ui.state import SessionStateManager


def execute_single_prediction(
    model_name: str,
    models_dict: Optional[Dict[str, Any]] = None,
    preprocessor: Optional[Any] = None,
    input_dict: Optional[Dict[str, Any]] = None,
    decision_threshold: Optional[float] = None,
) -> Tuple[int, float, float]:
    """
    Runs single prediction via Layer 2 Use Case, tracks inference latency,
    and records session QoS telemetry.
    """
    if input_dict is None:
        raise ValueError("input_dict is required for prediction.")

    container = Container.get_instance()
    entity = ShipmentEntity(
        country=str(input_dict["Country"]),
        scheduled_delivery_date=str(input_dict["Scheduled_Delivery_Date"]),
        po_sent_to_vendor_date=str(input_dict.get("PO_Sent_to_Vendor_Date", "")),
        manufacturer_lead_time_days=float(input_dict.get("Manufacturer_Lead_Time_Days", 90.0)),
        product_group=str(input_dict["Product_Group"]),
        sub_classification=str(input_dict["Sub_Classification"]),
        shipment_mode=str(input_dict["Shipment_Mode"]),
        fulfill_via=str(input_dict["Fulfill_Via"]),
        vendor_inco_term=str(input_dict["Vendor_INCO_Term"]),
        vendor=str(input_dict["Vendor"]),
        manufacturing_site=str(input_dict["Manufacturing_Site"]),
        first_line_designation=str(input_dict["First_Line_Designation"]),
        line_item_quantity=float(input_dict["Line_Item_Quantity"]),
        line_item_value=float(input_dict["Line_Item_Value"]),
        pack_price=float(input_dict["Pack_Price"]),
        unit_price=float(input_dict["Unit_Price"]),
        unit_of_measure_per_pack=float(input_dict["Unit_of_Measure_Per_Pack"]),
        weight_kilograms=str(input_dict["Weight_Kilograms"]),
        freight_cost_usd=str(input_dict["Freight_Cost_USD"]),
        line_item_insurance_usd=float(input_dict.get("Line_Item_Insurance_USD", 50.0)),
    )

    use_case = container.get_predict_shipment_use_case(model_name)
    prognosis = use_case.execute(entity, custom_threshold=decision_threshold)

    # Record QoS telemetry in UI session state
    SessionStateManager.record_telemetry(prognosis.qos_latency_ms, model_name, "/predict")

    return prognosis.late_delivery_risk, prognosis.risk_probability, prognosis.qos_latency_ms



def render_prediction_tab(
    preprocessor: Any,
    models_dict: Dict[str, Any],
    selected_model_name: str,
    decision_threshold: float,
    preset_vals: Dict[str, Any],
):
    """Renders the Real-Time Consignment Risk & Stockout Prediction View."""
    st.subheader("Real-Time Consignment Risk & Stockout Assessment")
    st.markdown(
        "Configure shipment parameters below. Every input includes a help tooltip **(?)** explaining its operational significance. "
        "The system evaluates **logistics features**, **manufacturer-quoted SLAs**, and **geographic resource tiers** in real-time."
    )

    # 1. Geographic Origin & Destination
    st.markdown("#### 1. Geographic Supply Chain Corridor & Route Logistics")
    geo1, geo2, geo3 = st.columns(3)

    source_country_list = [
        "India", "South Africa", "United States", "Germany", "United Kingdom",
        "Japan", "France", "Netherlands", "Ireland", "South Korea", "China", "Italy", "Canada", "International / Other",
    ]
    destination_country_list = [
        "Congo, DRC", "Zambia", "Burundi", "Mozambique", "South Sudan", "Nigeria",
        "Cte d'Ivoire", "Uganda", "Tanzania", "Zimbabwe", "Haiti", "Rwanda",
        "Guyana", "Kenya", "South Africa", "Vietnam", "Ethiopia", "Namibia", "Botswana", "Cameroon",
    ]

    with geo1:
        def_src = preset_vals.get("Source_Country", "India")
        src_idx = source_country_list.index(def_src) if def_src in source_country_list else 0
        source_country = st.selectbox(
            "Source Country (Manufacturing Origin):",
            source_country_list,
            index=src_idx,
            help="Country where medication or test kits are produced and packaged. Over 70% of PEPFAR antiretrovirals originate in India.",
        )

    with geo2:
        def_dst = preset_vals.get("Destination_Country", "Congo, DRC")
        dst_idx = destination_country_list.index(def_dst) if def_dst in destination_country_list else 0
        destination_country = st.selectbox(
            "Destination Country (Recipient Facility):",
            destination_country_list,
            index=dst_idx,
            help="Recipient sovereign country receiving the consignment. Landlocked countries face higher port and border clearance delays.",
        )

    with geo3:
        dest_tier = preprocessor.assign_country_tier(destination_country)
        tier_color = "#EF4444" if "Tier_3" in dest_tier else ("#F59E0B" if "Tier_2" in dest_tier else "#10B981")
        tier_readable = (
            "Tier 3: High-Risk Underserved"
            if "Tier_3" in dest_tier
            else ("Tier 2: Moderate-Risk Regional" if "Tier_2" in dest_tier else "Tier 1: Low-Risk Centralized Hub")
        )
        st.markdown("**Health Equity Tier:**")
        st.markdown(f"<span style='color:{tier_color};font-weight:700;font-size:1.05rem;'>{tier_readable}</span>", unsafe_allow_html=True)
        st.caption("Based on historical stockout rates and healthcare logistical resilience.")

    # Route Logistics: Transportation Mode, Fulfillment Channel & Commercial Trade Terms
    route_col1, route_col2, route_col3 = st.columns(3)

    shipment_mode_options = {
        "Air — Commercial Air Freight (Fast / Standard)": "Air",
        "Air Charter — Dedicated Charter Flight (Urgent / Emergency)": "Air Charter",
        "Truck — Overland Road Transport (Regional Transit)": "Truck",
        "Ocean — Sea Freight Container (High Volume / Slow)": "Ocean",
    }
    def_mode_raw = preset_vals.get("Shipment_Mode", "Air")
    matching_modes = [k for k, v in shipment_mode_options.items() if v == def_mode_raw]
    mode_default_idx = list(shipment_mode_options.keys()).index(matching_modes[0]) if matching_modes else 0

    with route_col1:
        selected_mode_label = st.selectbox(
            "Shipment Transportation Mode:",
            list(shipment_mode_options.keys()),
            index=mode_default_idx,
            help="Logistical transport mode. Air Charter is used for urgent stockouts; Ocean is slowest.",
        )
        shipment_mode = shipment_mode_options[selected_mode_label]

    fulfill_options = {
        "Direct Drop — Direct Dispatch from Manufacturer to Recipient Country": "Direct Drop",
        "From RDC — Dispatched from Regional Distribution Center Warehouse": "From RDC",
    }
    def_fulfill_raw = preset_vals.get("Fulfill_Via", "Direct Drop")
    matching_fulfills = [k for k, v in fulfill_options.items() if v == def_fulfill_raw]
    fulfill_default_idx = list(fulfill_options.keys()).index(matching_fulfills[0]) if matching_fulfills else 0

    with route_col2:
        selected_fulfill_label = st.selectbox(
            "Fulfillment Channel:",
            list(fulfill_options.keys()),
            index=fulfill_default_idx,
            help="Direct Drop ships directly from the plant; From RDC utilizes regional buffer warehouses.",
        )
        fulfill_via = fulfill_options[selected_fulfill_label]

    incoterm_options = {
        "CIP — Carriage & Insurance Paid (Seller covers freight + insurance)": "CIP",
        "EXW — Ex Works (Buyer covers pickup directly from factory door)": "EXW",
        "FCA — Free Carrier (Seller delivers to nominated carrier)": "FCA",
        "DDP — Delivered Duty Paid (Seller covers all costs to destination door)": "DDP",
        "CIF — Cost, Insurance & Freight (Maritime sea cargo insurance included)": "CIF",
        "DAP — Delivered at Place (Seller delivers ready for unloading)": "DAP",
    }
    def_inco_raw = preset_vals.get("Vendor_INCO_Term", "CIP")
    matching_incos = [k for k, v in incoterm_options.items() if v == def_inco_raw]
    inco_default_idx = list(incoterm_options.keys()).index(matching_incos[0]) if matching_incos else 0

    with route_col3:
        selected_inco_label = st.selectbox(
            "Vendor INCO Term (Commercial Trade Agreement):",
            list(incoterm_options.keys()),
            index=inco_default_idx,
            help="International commercial terms defining transit risk and insurance allocation.",
        )
        vendor_inco = incoterm_options[selected_inco_label]

    # Render Supply Route Corridor Banner (Strictly reflects the active user-selected shipment_mode & fulfill_via)
    render_route_corridor_banner(
        source_country=source_country,
        destination_country=destination_country,
        destination_tier=dest_tier,
        shipment_mode=shipment_mode,
        fulfill_via=fulfill_via,
    )

    with st.expander("Interactive 3D Geospatial Transit Corridor", expanded=False):
        render_interactive_corridor_map(
            source_country=source_country,
            destination_country=destination_country,
            destination_tier=dest_tier,
            shipment_mode=shipment_mode,
        )

    # 2. Detailed Parameters Form (Encapsulated in st.form to eliminate intermediate page reruns)
    with st.form(key="consignment_risk_evaluation_form", clear_on_submit=False):
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.markdown("##### Manufacturer SLA & Delivery Schedule")
            min_allowed_date = date(2000, 1, 1)
            max_allowed_date = date(2050, 12, 31)

            def_sched = preset_vals.get("Scheduled_Delivery_Date", date(2026, 11, 20))
            def_po = preset_vals.get("PO_Sent_to_Vendor_Date", date(2026, 8, 1))
            def_mfg_lt = float(preset_vals.get("Manufacturer_Lead_Time_Days", 90.0))

            mfg_lead_time_days = st.number_input(
                "Manufacturer / Distributor Quoted Lead Time (Days):",
                min_value=1.0,
                max_value=600.0,
                value=def_mfg_lt,
                step=5.0,
                help="Contracted turnaround time defined directly by the manufacturer or distributor SLA. NOT calculated as a calendar difference.",
            )

            po_date = st.date_input(
                "PO Sent to Vendor Date (Order Placement):",
                value=def_po,
                min_value=min_allowed_date,
                max_value=max_allowed_date,
                help="Date when the official Purchase Order was issued to the manufacturer or distributor.",
            )

            sched_date = st.date_input(
                "Scheduled Delivery Date (Client Facility Deadline):",
                value=def_sched,
                min_value=min_allowed_date,
                max_value=max_allowed_date,
                help="Agreed arrival deadline at client facility. You can pick any date up to year 2050.",
            )

            # Render Lead Time Squeeze Analysis
            client_window_days, is_squeezed = render_lead_time_tracker(mfg_lead_time_days, po_date, sched_date)

        with col_b:
            st.markdown("##### Pharmaceutical & Product Profile")

            product_group_options = {
                "ARV — Antiretroviral (HIV/AIDS Life-Saving Medication)": "ARV",
                "HRDT — HIV Rapid Diagnostic Test (Screening & Confirmation Kits)": "HRDT",
                "ANTM — Antimalarial (Malaria Treatment Therapeutics)": "ANTM",
                "ACT — Artemisinin Combination Therapy (WHO Frontline Malaria Drug)": "ACT",
                "MRDT — Malaria Rapid Diagnostic Test (Malaria Screening Kits)": "MRDT",
            }
            def_prod_raw = preset_vals.get("Product_Group", "ARV")
            def_prod_label = [k for k, v in product_group_options.items() if v == def_prod_raw][0]
            selected_prod_label = st.selectbox(
                "Product Group (Therapeutic Class):",
                list(product_group_options.keys()),
                index=list(product_group_options.keys()).index(def_prod_label),
                help="Medical therapeutic category. ARVs treat HIV; HRDTs are rapid diagnostic test kits; ACT/ANTM treat Malaria.",
            )
            product_group = product_group_options[selected_prod_label]

            # Dynamic sub-classifications matching selected Product Group
            if product_group == "ARV":
                sub_class_options = {
                    "Adult Formulations (Pills / Tablets for Adults)": "Adult",
                    "Pediatric Formulations (Oral Suspensions / Syrups for Children)": "Pediatric",
                }
            elif product_group == "HRDT":
                sub_class_options = {
                    "HIV Rapid Screening Test Kits": "HIV test",
                    "HIV Test - Ancillary (Lab Buffer Liquids & Consumables)": "HIV test - Ancillary",
                }
            elif product_group == "ANTM":
                sub_class_options = {"Malaria Therapeutics (General Antimalarials)": "Malaria"}
            elif product_group == "ACT":
                sub_class_options = {"ACT Formulations (Artemisinin Combination Therapies)": "ACT"}
            else:
                sub_class_options = {"Malaria Rapid Diagnostic Test Kits": "Malaria"}

            def_sub_raw = preset_vals.get("Sub_Classification", "Adult")
            matching_sub_labels = [k for k, v in sub_class_options.items() if v == def_sub_raw]
            sub_default_idx = list(sub_class_options.keys()).index(matching_sub_labels[0]) if matching_sub_labels else 0

            selected_sub_label = st.selectbox(
                "Sub-Classification (Target Patient Formulation):",
                list(sub_class_options.keys()),
                index=sub_default_idx,
                help="Patient formulation or diagnostic screening reagent tier.",
            )
            sub_classification = sub_class_options[selected_sub_label]

            first_line_options = {
                "Yes — WHO First-Line Priority Drug (Essential Frontline Therapy)": "Yes",
                "No — Second-Line or Alternative Formulation": "No",
            }
            def_fl_raw = preset_vals.get("First_Line_Designation", "Yes")
            def_fl_label = [k for k, v in first_line_options.items() if v == def_fl_raw][0]
            selected_fl_label = st.selectbox(
                "First-Line Medicine Designation:",
                list(first_line_options.keys()),
                index=list(first_line_options.keys()).index(def_fl_label),
                help="Frontline essential medicine under WHO guidelines. Stockout carries immediate life-threatening risk.",
            )
            first_line_desig = first_line_options[selected_fl_label]

            vendor_options = [
                "Aurobindo Pharma Limited", "Ranbaxy Fine Chemicals LTD.", "Cipla Limited",
                "Hetero Labs Limited", "Mylan Laboratories", "Other_Vendor",
            ]
            def_ven = preset_vals.get("Vendor", "Aurobindo Pharma Limited")
            vendor = st.selectbox(
                "Pharmaceutical Supplier / Vendor:",
                vendor_options,
                index=vendor_options.index(def_ven) if def_ven in vendor_options else 0,
                help="Contracted certified pharmaceutical supplier.",
            )

            site_options = [
                "Aurobindo Unit III, India", "Ranbaxy Fine Chemicals LTD", "Cipla, Patalganga, India",
                "Cipla, Goa, India", "Hetero Unit III Hyderabad IN", "ABBVIE Ludwigshafen Germany",
                "Trinity Biotech, Plc", "Alere Medical Co., Ltd.", "Aspen-OSD, Port Elizabeth, SA", "Other_Site",
            ]
            def_site = preset_vals.get("Manufacturing_Site", "Aurobindo Unit III, India")
            manufacturing_site = st.selectbox(
                "Certified Manufacturing Facility:",
                site_options,
                index=site_options.index(def_site) if def_site in site_options else 0,
                help="Certified production site where drug lot is synthesized and packaged.",
            )

            uom = st.number_input(
                "Unit of Measure (Pills/Kits per Pack):",
                min_value=1,
                max_value=1000,
                value=int(preset_vals.get("Unit_of_Measure_Per_Pack", 60)),
                help="Count of individual units per physical pack.",
            )

        with col_c:
            st.markdown("##### Order Volume, Pricing & Freight")

            qty = st.number_input(
                "Line Item Quantity (Packs):",
                min_value=1.0,
                max_value=1000000.0,
                value=float(preset_vals.get("Line_Item_Quantity", 25000.0)),
                step=500.0,
                help="Total count of packs ordered in this consignment.",
            )

            pack_price = st.number_input(
                "Pack Price (USD per Pack):",
                min_value=0.01,
                max_value=1000.0,
                value=float(preset_vals.get("Pack_Price", 5.0)),
                step=0.5,
                help="Procurement price per unit pack in USD.",
            )

            unit_price = st.number_input(
                "Unit Price (USD per Pill/Kit):",
                min_value=0.001,
                max_value=100.0,
                value=float(preset_vals.get("Unit_Price", 0.08)),
                step=0.01,
                help="Calculated unit price per individual pill or test.",
            )

            auto_calc_val = qty * pack_price
            val = st.number_input(
                "Line Item Total Value (USD):",
                min_value=1.0,
                max_value=10000000.0,
                value=float(preset_vals.get("Line_Item_Value", auto_calc_val)),
                step=1000.0,
                help="Gross invoice value of the entire shipment line in USD.",
            )
            st.caption(f"Calculated Value: {qty:,.0f} packs × ${pack_price:.2f} = **${auto_calc_val:,.2f}**")

            weight = st.number_input(
                "Consignment Weight (Kilograms):",
                min_value=0.1,
                max_value=100000.0,
                value=float(preset_vals.get("Weight_Kilograms", 1450.0)),
                step=50.0,
                help="Gross cargo weight in kg.",
            )

            freight = st.number_input(
                "Freight Cost (USD):",
                min_value=0.0,
                max_value=500000.0,
                value=float(preset_vals.get("Freight_Cost_USD", 8900.0)),
                step=100.0,
                help="International shipping and transport fee.",
            )

            insurance = st.number_input(
                "Line Item Insurance (USD):",
                min_value=0.0,
                max_value=50000.0,
                value=float(preset_vals.get("Line_Item_Insurance_USD", 140.0)),
                step=10.0,
                help="Cargo transit insurance coverage fee.",
            )

        st.markdown("---")
        predict_btn = st.form_submit_button(
            "Predict Stockout & Late-Delivery Risk Now",
            type="primary",
            use_container_width=True,
        )

    if predict_btn:
        input_data = {
            "Country": destination_country,
            "Scheduled_Delivery_Date": str(sched_date),
            "PO_Sent_to_Vendor_Date": str(po_date),
            "Manufacturer_Lead_Time_Days": mfg_lead_time_days,
            "Product_Group": product_group,
            "Sub_Classification": sub_classification,
            "Shipment_Mode": shipment_mode,
            "Fulfill_Via": fulfill_via,
            "Vendor_INCO_Term": vendor_inco,
            "Vendor": vendor,
            "Manufacturing_Site": manufacturing_site,
            "First_Line_Designation": first_line_desig,
            "Line_Item_Quantity": qty,
            "Line_Item_Value": val,
            "Pack_Price": pack_price,
            "Unit_Price": unit_price,
            "Unit_of_Measure_Per_Pack": uom,
            "Weight_Kilograms": weight,
            "Freight_Cost_USD": freight,
            "Line_Item_Insurance_USD": insurance,
        }

        with st.spinner("Processing deep tensor forward pass & profiling QoS latency..."):
            flag, prob, lat_ms = execute_single_prediction(
                model_name=selected_model_name,
                models_dict=models_dict,
                preprocessor=preprocessor,
                input_dict=input_data,
                decision_threshold=decision_threshold,
            )

        # Record prediction to session state for persistence across tab navigations
        SessionStateManager.record_prediction({
            "flag": flag,
            "prob": prob,
            "threshold": decision_threshold,
            "country": destination_country,
            "country_tier": dest_tier,
            "mfg_lead_time_days": mfg_lead_time_days,
            "client_window_days": client_window_days,
            "shipment_mode": shipment_mode,
            "fulfill_via": fulfill_via,
            "first_line_desig": first_line_desig,
            "latency_ms": lat_ms,
            "model_name": selected_model_name,
        })

    # Render Scorecard & Telemetry from active or persistent state
    last_pred = SessionStateManager.get_last_prediction()
    if last_pred is not None:
        render_prediction_scorecard(
            flag=last_pred["flag"],
            prob=last_pred["prob"],
            threshold=last_pred["threshold"],
            country=last_pred["country"],
            country_tier=last_pred["country_tier"],
            mfg_lead_time_days=last_pred["mfg_lead_time_days"],
            client_window_days=last_pred["client_window_days"],
            shipment_mode=last_pred["shipment_mode"],
            fulfill_via=last_pred["fulfill_via"],
            first_line_desig=last_pred["first_line_desig"],
        )

        st.write("")
        render_qos_telemetry_bar(
            latency_ms=last_pred["latency_ms"],
            model_name=last_pred["model_name"],
            threshold=last_pred["threshold"],
        )
