"""
Domain Composite Scorecards for Pharmaceutical Stockout Prediction.
Includes calibrated probability gauges, lead time SLA analysis,
and public health clinical mitigation playbooks.
"""

from typing import Dict, Any, Tuple
from datetime import date, timedelta
import streamlit as st
from src.ui.components.primitives import render_alert, render_badge


def render_lead_time_tracker(
    mfg_lead_time_days: float,
    po_date: date,
    sched_date: date,
) -> Tuple[int, bool]:
    """
    Renders visual analysis of manufacturer-quoted SLA lead time versus client delivery target.
    Returns (client_window_days, is_squeezed).
    """
    est_supplier_delivery = po_date + timedelta(days=int(mfg_lead_time_days))
    st.caption(
        f"**Supplier SLA Target:** **{est_supplier_delivery.strftime('%d-%b-%Y')}** "
        f"(contracted {int(mfg_lead_time_days)} days manufacturer lead time)"
    )

    client_window_days = (sched_date - po_date).days
    is_squeezed = False

    if client_window_days < 0:
        render_alert(
            f"Scheduled arrival date ({sched_date}) is earlier than Purchase Order date ({po_date})! "
            f"Please verify shipment timeline dates.",
            variant="error",
            title="Invalid Timeline Sequence",
        )
        is_squeezed = True
    elif client_window_days < mfg_lead_time_days:
        deficit = int(mfg_lead_time_days - client_window_days)
        render_alert(
            f"Client deadline allows **{client_window_days} days**, but manufacturer contract mandates "
            f"**{int(mfg_lead_time_days)} days** ({deficit}-day deficit!). "
            "This manufacturer lead time squeeze is a primary driver of delivery delays and facility stockouts.",
            variant="warning",
            title="Lead Time Squeeze Alert",
        )
        is_squeezed = True
    else:
        buffer = int(client_window_days - mfg_lead_time_days)
        render_alert(
            f"Client window allows **{client_window_days} days** "
            f"(+{buffer} days safety buffer over manufacturer's {int(mfg_lead_time_days)}d quoted turnaround).",
            variant="success",
            title="Contract SLA Aligned",
        )

    return client_window_days, is_squeezed


def render_prediction_scorecard(
    flag: int,
    prob: float,
    threshold: float,
    country: str,
    country_tier: str,
    mfg_lead_time_days: float,
    client_window_days: int,
    shipment_mode: str,
    fulfill_via: str,
    first_line_desig: str,
):
    """
    Renders accessible, production-ready stockout risk scorecard.
    Includes calibrated probability gauge, explainable risk factors, and actionable interventions.
    """
    prob_pct = prob * 100.0

    st.markdown("#### Clinical Risk Prognosis")
    score_col1, score_col2 = st.columns([1.6, 1])

    with score_col1:
        if flag == 1:
            st.error(
                f"### HIGH STOCKOUT & DELAY RISK DETECTED\n\n"
                f"**Consignment flagged as AT-RISK of clinical delivery failure.**\n\n"
                f"Calibrated Failure Probability: **{prob_pct:.1f}%** | Operating Threshold: **{threshold:.2f}**"
            )
        else:
            st.success(
                f"### ON-TIME DELIVERY ASSURED\n\n"
                f"**Consignment projected to arrive within schedule without clinical stockout risk.**\n\n"
                f"Calibrated Failure Probability: **{prob_pct:.1f}%** | Operating Threshold: **{threshold:.2f}**"
            )

        # Calibrated Probability Gauge
        st.write("**Calibrated Risk Probability Gauge:**")
        st.progress(float(min(max(prob, 0.0), 1.0)))

    with score_col2:
        st.markdown("##### Key Risk Characteristics")
        st.markdown(f"- **Destination:** `{country}`")
        st.markdown(f"- **Health Equity Tier:** `{country_tier}`")
        st.markdown(f"- **Contract Lead Time:** `{int(mfg_lead_time_days)} days`")
        st.markdown(f"- **Client Window:** `{client_window_days} days`")
        st.markdown(f"- **Essential Priority:** `{first_line_desig}`")

    # Explainable Risk Factor Breakdown Cards
    st.markdown("#### Explainable Risk Factor Analysis")
    exp1, exp2, exp3, exp4 = st.columns(4)

    with exp1:
        st.info(f"**Lead Time SLA:**\n\n{int(mfg_lead_time_days)}d Quoted vs {client_window_days}d Allowed")
    with exp2:
        st.info(f"**Geographic Tier:**\n\n{country_tier}")
    with exp3:
        st.info(f"**Logistics Channel:**\n\n{shipment_mode} ({fulfill_via})")
    with exp4:
        st.info(f"**Clinical Criticality:**\n\n{first_line_desig} Essential Medicine")

    # Operational Mitigation Playbook
    st.markdown("#### Public Health Supply Chain Mitigation Playbook")
    if flag == 1:
        st.warning(
            f"**High Vulnerability Action Required:** This shipment is destined for **{country} ({country_tier})** with elevated probability of stockout. "
            "\n\n**Immediate Interventions:**"
            "\n1. **Logistics Rerouting:** Expedite consignment to dedicated Air Charter or buffer dispatch from Regional Distribution Center (RDC)."
            "\n2. **Early Health Facility Alert:** Notify destination provincial clinics to activate secondary safety stock."
            "\n3. **Supplier Escalation:** Issue high-priority tracking ticket to supplier to guarantee production slot."
        )
    else:
        st.success(
            f"**Standard Operational Handling:** Consignment to **{country}** exhibits stable logistical margins. "
            "Proceed with scheduled distribution channel."
        )
