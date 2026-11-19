"""
Reusable UI Primitives for Streamlit Design System.
Provides accessible, composable atoms with strict WCAG 2.1 AA compliance,
semantic HTML5 tags, and responsive CSS styling.
All interpolated user and telemetry strings are strictly sanitized via html.escape
to eliminate Cross-Site Scripting (XSS) injection vectors.
"""

from typing import Optional, Callable, Literal
import html
import streamlit as st


def render_header(
    title: str,
    subtitle: Optional[str] = None,
    badge_text: Optional[str] = None,
    badge_variant: Literal["success", "warning", "danger", "primary"] = "primary",
):
    """Renders accessible application or section header with optional sanitized status badge."""
    safe_title = html.escape(str(title), quote=True)
    safe_subtitle = html.escape(str(subtitle), quote=True) if subtitle else None
    safe_badge_text = html.escape(str(badge_text), quote=True) if badge_text else None

    badge_html = (
        f'<span class="hscm-badge hscm-badge-{badge_variant}">{safe_badge_text}</span>'
        if safe_badge_text
        else ""
    )
    st.markdown(
        f"""
        <header class="hscm-header" role="banner">
            <h1 class="hscm-header-title">
                {safe_title} {badge_html}
            </h1>
            {f'<p class="hscm-header-subtitle">{safe_subtitle}</p>' if safe_subtitle else ''}
        </header>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(
    label: str,
    value: str,
    delta: Optional[str] = None,
    delta_positive: bool = True,
    meta: Optional[str] = None,
    tooltip: Optional[str] = None,
):
    """
    Renders an accessible stat/metric card.
    Uses sanitized HTML and ARIA live regions for assistive technologies.
    """
    safe_label = html.escape(str(label), quote=True)
    safe_value = html.escape(str(value), quote=True)
    safe_meta = html.escape(str(meta), quote=True) if meta else ""
    safe_tooltip = html.escape(str(tooltip), quote=True) if tooltip else ""

    delta_html = ""
    if delta:
        safe_delta = html.escape(str(delta), quote=True)
        delta_color = "var(--hscm-success)" if delta_positive else "var(--hscm-danger)"
        arrow = "▲" if delta_positive else "▼"
        delta_html = f'<span style="color: {delta_color}; font-weight: 700; margin-left: 0.4rem;">{arrow} {safe_delta}</span>'

    meta_html = f'<div class="hscm-stat-meta">{safe_meta}</div>' if safe_meta else ""
    title_attr = f'title="{safe_tooltip}"' if safe_tooltip else ""

    st.markdown(
        f"""
        <div class="hscm-stat-card" {title_attr} role="region" aria-label="{safe_label} metric">
            <div class="hscm-stat-label">{safe_label}</div>
            <div class="hscm-stat-value">
                {safe_value}
                {delta_html}
            </div>
            {meta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_badge(
    text: str,
    variant: Literal["tier1", "tier2", "tier3", "success", "warning", "danger", "neutral"] = "neutral",
    pulse: bool = False,
) -> str:
    """Returns HTML for an accessible inline badge/pill with sanitized content."""
    safe_text = html.escape(str(text), quote=True)
    pulse_dot = ""
    if pulse:
        pulse_class = "hscm-pulse-red" if variant in ["tier3", "danger"] else "hscm-pulse-green"
        pulse_dot = f'<span class="hscm-pulse-dot {pulse_class}" aria-hidden="true"></span>'

    return f'<span class="hscm-badge hscm-badge-{variant}" role="status">{pulse_dot} {safe_text}</span>'


def render_skeleton(height: str = "45px", width: str = "100%", count: int = 1):
    """Renders content-aware shimmer skeleton to eliminate jarring layout shifts."""
    safe_height = html.escape(str(height), quote=True)
    safe_width = html.escape(str(width), quote=True)
    for _ in range(count):
        st.markdown(
            f'<div class="hscm-skeleton" style="height: {safe_height}; width: {safe_width}; margin-bottom: 0.5rem;" aria-hidden="true"></div>',
            unsafe_allow_html=True,
        )


def render_empty_state(
    title: str,
    description: str,
    icon: Optional[str] = None,
    action_label: Optional[str] = None,
    action_key: Optional[str] = None,
    on_action: Optional[Callable] = None,
) -> bool:
    """
    Renders an accessible empty state with sanitized copy and call-to-action button.
    """
    safe_title = html.escape(str(title), quote=True)
    safe_desc = html.escape(str(description), quote=True)
    icon_html = f'<div class="hscm-empty-icon" aria-hidden="true">{html.escape(str(icon), quote=True)}</div>' if icon else ""

    st.markdown(
        f"""
        <div class="hscm-empty-state" role="status" aria-live="polite">
            {icon_html}
            <div class="hscm-empty-title">{safe_title}</div>
            <div class="hscm-empty-desc">{safe_desc}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if action_label:
        col1, col2, col3 = st.columns([1, 1.2, 1])
        with col2:
            clicked = st.button(action_label, key=action_key, type="primary", use_container_width=True)
            if clicked and on_action:
                on_action()
            return clicked
    return False


def render_alert(
    message: str,
    variant: Literal["info", "warning", "error", "success"] = "info",
    title: Optional[str] = None,
):
    """Renders theme-adaptive alert banner using native accessible Streamlit channels."""
    formatted_msg = f"**{title}**\n\n{message}" if title else message
    if variant == "error":
        st.error(formatted_msg)
    elif variant == "warning":
        st.warning(formatted_msg)
    elif variant == "success":
        st.success(formatted_msg)
    else:
        st.info(formatted_msg)
