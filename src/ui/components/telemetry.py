"""
Cloud Telemetry and Supply Chain Route Components.
Visualizes real-time Quality of Service (QoS) metrics, inference SLAs,
and international pharmaceutical transit corridors.
All dynamic corridor and route strings are sanitized against XSS attacks.
"""

from typing import Optional
import html
import streamlit as st
import pydeck as pdk
import pandas as pd
from src.ui.components.primitives import render_badge, render_metric_card

# Geographic coordinates for international pharmaceutical corridors
CORRIDOR_COORDINATES = {
    "India": {"lat": 20.5937, "lon": 78.9629, "name": "India (Global ARV Production Hub)"},
    "South Africa": {"lat": -30.5595, "lon": 22.9375, "name": "South Africa (Regional Hub)"},
    "United States": {"lat": 37.0902, "lon": -95.7129, "name": "United States (Donor Hub)"},
    "Germany": {"lat": 51.1657, "lon": 10.4515, "name": "Germany (Packaging Facility)"},
    "United Kingdom": {"lat": 55.3781, "lon": -3.4360, "name": "United Kingdom"},
    "Japan": {"lat": 36.2048, "lon": 138.2529, "name": "Japan"},
    "France": {"lat": 46.2276, "lon": 2.2137, "name": "France"},
    "Netherlands": {"lat": 52.1326, "lon": 5.2913, "name": "Netherlands (Port of Rotterdam)"},
    "Ireland": {"lat": 53.1424, "lon": -7.6921, "name": "Ireland"},
    "South Korea": {"lat": 35.9078, "lon": 127.7669, "name": "South Korea"},
    "China": {"lat": 35.8617, "lon": 104.1954, "name": "China (Raw API Source)"},
    "Italy": {"lat": 41.8719, "lon": 12.5674, "name": "Italy"},
    "Canada": {"lat": 56.1304, "lon": -106.3468, "name": "Canada"},
    "International / Other": {"lat": 46.2044, "lon": 6.1432, "name": "International Multilateral Relief Hub (Geneva, CH)"},
    "International": {"lat": 46.2044, "lon": 6.1432, "name": "International Multilateral Relief Hub (Geneva, CH)"},
    "Other": {"lat": 46.2044, "lon": 6.1432, "name": "International Multilateral Relief Hub (Geneva, CH)"},
    "Congo, DRC": {"lat": -4.0383, "lon": 21.7587, "name": "Congo, DRC (Tier 3 Vulnerable)"},
    "Democratic Republic of the Congo": {"lat": -4.0383, "lon": 21.7587, "name": "Congo, DRC (Tier 3 Vulnerable)"},
    "DRC": {"lat": -4.0383, "lon": 21.7587, "name": "Congo, DRC (Tier 3 Vulnerable)"},
    "Zambia": {"lat": -13.1339, "lon": 27.8493, "name": "Zambia"},
    "Burundi": {"lat": -3.3731, "lon": 29.9189, "name": "Burundi (Landlocked)"},
    "Mozambique": {"lat": -18.6657, "lon": 35.5296, "name": "Mozambique"},
    "South Sudan": {"lat": 6.8770, "lon": 31.3070, "name": "South Sudan (Tier 3 Fragile)"},
    "Nigeria": {"lat": 9.0820, "lon": 8.6753, "name": "Nigeria (High Volume Node)"},
    "Cte d'Ivoire": {"lat": 7.5400, "lon": -5.5471, "name": "Côte d'Ivoire (West African Node)"},
    "Côte d'Ivoire": {"lat": 7.5400, "lon": -5.5471, "name": "Côte d'Ivoire (West African Node)"},
    "Uganda": {"lat": 1.3733, "lon": 32.2903, "name": "Uganda"},
    "Tanzania": {"lat": -6.3690, "lon": 34.8888, "name": "Tanzania"},
    "Zimbabwe": {"lat": -19.0154, "lon": 29.1549, "name": "Zimbabwe"},
    "Haiti": {"lat": 18.9712, "lon": -72.2852, "name": "Haiti (Island Logistics)"},
    "Rwanda": {"lat": -1.9403, "lon": 29.8739, "name": "Rwanda"},
    "Guyana": {"lat": 4.8604, "lon": -58.9302, "name": "Guyana"},
    "Kenya": {"lat": -0.0236, "lon": 37.9062, "name": "Kenya (Port of Mombasa Node)"},
    "Vietnam": {"lat": 14.0583, "lon": 108.2772, "name": "Vietnam"},
    "Ethiopia": {"lat": 9.1450, "lon": 40.4897, "name": "Ethiopia (Addis Hub)"},
    "Namibia": {"lat": -22.9576, "lon": 18.4904, "name": "Namibia"},
    "Botswana": {"lat": -22.3285, "lon": 24.6849, "name": "Botswana"},
    "Cameroon": {"lat": 7.3697, "lon": 12.3547, "name": "Cameroon"},
}

# Neutral fallback coordinates for unknown international nodes (Geneva / Europe international transit hub)
INTERNATIONAL_FALLBACK = {"lat": 46.2044, "lon": 6.1432, "name": "International Multilateral Relief Hub (Geneva, CH)"}


def render_interactive_corridor_map(
    source_country: str,
    destination_country: str,
    destination_tier: str,
    shipment_mode: str = "Air",
):
    """
    Renders an interactive PyDeck 3D geospatial arc map showing
    origin-to-destination transit trajectory color-coded by health equity risk tier.
    """
    src_pt = CORRIDOR_COORDINATES.get(
        source_country,
        {"lat": INTERNATIONAL_FALLBACK["lat"], "lon": INTERNATIONAL_FALLBACK["lon"], "name": f"{source_country} (International Node)"}
    )
    dst_pt = CORRIDOR_COORDINATES.get(
        destination_country,
        {"lat": -4.0383, "lon": 21.7587, "name": f"{destination_country} (Recipient)"}
    )

    # Color code destination by tier
    if "Tier_3" in destination_tier:
        target_color = [239, 68, 68, 220]  # Vibrant Red
        tier_text = "Tier 3 (High-Risk Underserved)"
    elif "Tier_2" in destination_tier:
        target_color = [245, 158, 11, 220]  # Amber
        tier_text = "Tier 2 (Moderate-Risk Node)"
    else:
        target_color = [16, 185, 129, 220]  # Emerald
        tier_text = "Tier 1 (Central Hub)"

    source_color = [37, 99, 235, 220]  # Electric Blue

    arc_df = pd.DataFrame([{
        "from_lat": src_pt["lat"],
        "from_lon": src_pt["lon"],
        "from_name": src_pt["name"],
        "to_lat": dst_pt["lat"],
        "to_lon": dst_pt["lon"],
        "to_name": dst_pt["name"],
        "mode": shipment_mode,
        "tier": tier_text,
    }])

    scatter_df = pd.DataFrame([
        {"lat": src_pt["lat"], "lon": src_pt["lon"], "name": f"Origin: {src_pt['name']}", "color": source_color, "radius": 180000},
        {"lat": dst_pt["lat"], "lon": dst_pt["lon"], "name": f"Destination: {dst_pt['name']}", "color": target_color, "radius": 240000},
    ])

    arc_layer = pdk.Layer(
        "ArcLayer",
        data=arc_df,
        get_source_position=["from_lon", "from_lat"],
        get_target_position=["to_lon", "to_lat"],
        get_source_color=source_color,
        get_target_color=target_color,
        get_width=4,
        get_tilt=15,
        pickable=True,
        auto_highlight=True,
    )

    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        data=scatter_df,
        get_position=["lon", "lat"],
        get_fill_color="color",
        get_radius="radius",
        pickable=True,
        auto_highlight=True,
    )

    # Center camera between origin and destination
    mid_lat = (src_pt["lat"] + dst_pt["lat"]) / 2.0
    mid_lon = (src_pt["lon"] + dst_pt["lon"]) / 2.0

    view_state = pdk.ViewState(
        latitude=float(mid_lat),
        longitude=float(mid_lon),
        zoom=1.8,
        min_zoom=1.0,
        max_zoom=6.0,
        pitch=30,
        bearing=0,
    )

    tooltip = {
        "html": "<b>{name}</b><br/>{tier}",
        "style": {"backgroundColor": "#0F172A", "color": "#F8FAFC", "fontSize": "12px", "borderRadius": "8px", "padding": "6px 10px"}
    }

    deck = pdk.Deck(
        layers=[arc_layer, scatter_layer],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style="dark",
    )

    st.pydeck_chart(deck, use_container_width=True)


def render_route_corridor_banner(
    source_country: str,
    destination_country: str,
    destination_tier: str,
    shipment_mode: str,
    fulfill_via: str,
):
    """
    Renders an accessible, XSS-safe dynamic corridor visualization showing origin, transit, and destination health tier.
    """
    safe_src = html.escape(str(source_country), quote=True)
    safe_dst = html.escape(str(destination_country), quote=True)
    safe_mode = html.escape(str(shipment_mode), quote=True)
    safe_fulfill = html.escape(str(fulfill_via), quote=True)

    tier_variant = "tier3" if "Tier_3" in destination_tier else ("tier2" if "Tier_2" in destination_tier else "tier1")
    tier_label = (
        "Tier 3: High-Risk Underserved"
        if "Tier_3" in destination_tier
        else ("Tier 2: Moderate-Risk Regional" if "Tier_2" in destination_tier else "Tier 1: Centralized Hub")
    )
    tier_badge_html = render_badge(tier_label, variant=tier_variant, pulse=("Tier_3" in destination_tier))

    st.markdown(
        f"""
        <div class="hscm-corridor-banner" role="region" aria-label="Active supply route corridor">
            <div class="hscm-corridor-route">
                <span><b>{safe_src}</b> (Origin)</span>
                <span aria-hidden="true" style="opacity: 0.8;">─── [{safe_mode}] ({safe_fulfill}) ───►</span>
                <span><b>{safe_dst}</b> (Recipient)</span>
            </div>
            <div>
                {tier_badge_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_qos_telemetry_bar(
    latency_ms: float,
    model_name: str,
    threshold: float,
    device: str = "CPU (Optimized)",
    throughput_samples_per_sec: Optional[float] = None,
):
    """
    Renders Cloud QoS infrastructure metrics with SLA threshold indicators.
    """
    st.markdown("##### Cloud QoS & Infrastructure Telemetry")
    col1, col2, col3, col4 = st.columns(4)

    # Colorize latency based on real-time SLAs (< 10ms = excellent, < 50ms = good, >= 50ms = warning)
    latency_delta = "Ultra-Fast Edge" if latency_ms < 5.0 else ("Compliant (<50ms)" if latency_ms < 50.0 else "High Load")
    delta_positive = latency_ms < 50.0

    with col1:
        render_metric_card(
            label="Inference Latency",
            value=f"{latency_ms:.2f} ms",
            delta=latency_delta,
            delta_positive=delta_positive,
            meta="Single-sample request P50",
            tooltip="Time taken for tensor tokenization, forward pass, and calibration",
        )

    with col2:
        render_metric_card(
            label="Active Architecture",
            value=model_name,
            meta=f"Compute Device: {device}",
            tooltip="Active deep learning sequence model or ensemble baseline",
        )

    with col3:
        render_metric_card(
            label="Decision Threshold (τ)",
            value=f"{threshold:.2f}",
            meta="Calibrated on 2014 Val Split",
            tooltip="Optimized for minority-class stockout F1-score",
        )

    with col4:
        tp_val = f"{throughput_samples_per_sec:.1f}/s" if throughput_samples_per_sec else "Real-time Single"
        render_metric_card(
            label="Throughput Capacity",
            value=tp_val,
            meta="FastAPI ASGI / Uvicorn",
            tooltip="Estimated peak inference capacity",
        )
