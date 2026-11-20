"""
Streamlit Production Design System: UI Tokens, Styles, and Themes.
Provides WCAG 2.1 AA compliant color palettes, responsive typography scales,
and dynamic dark/light mode CSS custom properties for enterprise healthcare applications.
"""

import streamlit as st


def inject_custom_theme():
    """
    Injects production-grade design tokens, shimmer keyframes, and accessible utility styles.
    Scoped to adapt automatically to both Streamlit Dark and Light themes.
    """
    st.markdown(
        """
        <style>
        /* =====================================================================
           1. CSS Custom Properties / Design Tokens (WCAG 2.1 AA Compliant)
           ===================================================================== */
        :root {
            --hscm-primary: #2563EB;
            --hscm-primary-hover: #1D4ED8;
            --hscm-primary-subtle: rgba(37, 99, 235, 0.12);
            
            --hscm-success: #16A34A;
            --hscm-success-subtle: rgba(22, 163, 74, 0.14);
            --hscm-success-border: #86EFAC;
            
            --hscm-warning: #D97706;
            --hscm-warning-subtle: rgba(217, 119, 6, 0.14);
            --hscm-warning-border: #FCD34D;
            
            --hscm-danger: #DC2626;
            --hscm-danger-subtle: rgba(220, 38, 38, 0.14);
            --hscm-danger-border: #FCA5A5;
            
            --hscm-tier1: #059669;
            --hscm-tier2: #D97706;
            --hscm-tier3: #DC2626;

            --hscm-radius-sm: 6px;
            --hscm-radius-md: 10px;
            --hscm-radius-lg: 14px;
            --hscm-radius-full: 9999px;
            
            --hscm-font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        }

        /* Responsive Container Fixes */
        .block-container {
            padding-top: 1.5rem !important;
            padding-bottom: 2.5rem !important;
            max-width: 95% !important;
        }

        /* =====================================================================
           2. Typography & Header Components
           ===================================================================== */
        .hscm-header {
            margin-bottom: 1.25rem;
            border-bottom: 1px solid var(--border-color, rgba(128, 128, 128, 0.2));
            padding-bottom: 0.75rem;
        }
        
        .hscm-header-title {
            font-size: 1.95rem;
            font-weight: 750;
            letter-spacing: -0.02em;
            color: var(--text-color, #111827);
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin: 0;
        }

        .hscm-header-subtitle {
            font-size: 0.95rem;
            color: var(--text-color, #6B7280);
            opacity: 0.85;
            margin-top: 0.35rem;
            line-height: 1.4;
        }

        /* =====================================================================
           3. Card Primitives (Theme Adaptive)
           ===================================================================== */
        .hscm-card {
            background-color: var(--secondary-background-color, rgba(255, 255, 255, 0.04));
            border: 1px solid var(--border-color, rgba(128, 128, 128, 0.2));
            border-radius: var(--hscm-radius-md);
            padding: 1.15rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }

        .hscm-card:hover {
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        }

        /* Stat / Metric Card */
        .hscm-stat-card {
            background-color: var(--secondary-background-color, rgba(255, 255, 255, 0.04));
            border: 1px solid var(--border-color, rgba(128, 128, 128, 0.2));
            border-radius: var(--hscm-radius-md);
            padding: 1rem 1.25rem;
            position: relative;
            overflow: hidden;
        }

        .hscm-stat-label {
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-weight: 600;
            color: var(--text-color, #6B7280);
            opacity: 0.8;
            margin-bottom: 0.25rem;
        }

        .hscm-stat-value {
            font-size: 1.75rem;
            font-weight: 800;
            color: var(--text-color, #111827);
            letter-spacing: -0.02em;
            line-height: 1.1;
        }

        .hscm-stat-meta {
            font-size: 0.8rem;
            margin-top: 0.35rem;
            color: var(--text-color, #6B7280);
            opacity: 0.85;
            display: flex;
            align-items: center;
            gap: 0.35rem;
        }

        /* =====================================================================
           4. Badges & Status Chips
           ===================================================================== */
        .hscm-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            font-size: 0.78rem;
            font-weight: 650;
            padding: 0.25rem 0.65rem;
            border-radius: var(--hscm-radius-full);
            white-space: nowrap;
        }

        .hscm-badge-tier1 {
            background-color: rgba(16, 185, 129, 0.15);
            color: #10B981;
            border: 1px solid rgba(16, 185, 129, 0.35);
        }

        .hscm-badge-tier2 {
            background-color: rgba(245, 158, 11, 0.15);
            color: #F59E0B;
            border: 1px solid rgba(245, 158, 11, 0.35);
        }

        .hscm-badge-tier3 {
            background-color: rgba(239, 68, 68, 0.15);
            color: #EF4444;
            border: 1px solid rgba(239, 68, 68, 0.35);
        }

        .hscm-badge-success {
            background-color: var(--hscm-success-subtle);
            color: var(--hscm-success);
            border: 1px solid var(--hscm-success-border);
        }

        .hscm-badge-danger {
            background-color: var(--hscm-danger-subtle);
            color: var(--hscm-danger);
            border: 1px solid var(--hscm-danger-border);
        }

        /* Pulse indicator for live streaming or high-risk alert */
        .hscm-pulse-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            display: inline-block;
        }

        .hscm-pulse-red {
            background-color: #EF4444;
            box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7);
            animation: hscm-pulse-animation 1.8s infinite;
        }

        .hscm-pulse-green {
            background-color: #10B981;
            box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
            animation: hscm-pulse-animation 1.8s infinite;
        }

        @keyframes hscm-pulse-animation {
            0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); }
            70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(239, 68, 68, 0); }
            100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
        }

        /* =====================================================================
           5. Shimmer Skeleton Loaders
           ===================================================================== */
        .hscm-skeleton {
            background: linear-gradient(
                90deg,
                rgba(128, 128, 128, 0.08) 25%,
                rgba(128, 128, 128, 0.18) 50%,
                rgba(128, 128, 128, 0.08) 75%
            );
            background-size: 200% 100%;
            animation: hscm-shimmer 1.5s infinite;
            border-radius: var(--hscm-radius-sm);
            display: inline-block;
        }

        @keyframes hscm-shimmer {
            0% { background-position: 200% 0; }
            100% { background-position: -200% 0; }
        }

        /* =====================================================================
           6. Accessible Empty State Pattern
           ===================================================================== */
        .hscm-empty-state {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            padding: 3rem 1.5rem;
            background-color: var(--secondary-background-color, rgba(255, 255, 255, 0.02));
            border: 2px dashed var(--border-color, rgba(128, 128, 128, 0.25));
            border-radius: var(--hscm-radius-lg);
            margin: 1.25rem 0;
        }

        .hscm-empty-icon {
            font-size: 2.75rem;
            margin-bottom: 0.75rem;
            opacity: 0.8;
        }

        .hscm-empty-title {
            font-size: 1.15rem;
            font-weight: 700;
            color: var(--text-color, #111827);
            margin-bottom: 0.35rem;
        }

        .hscm-empty-desc {
            font-size: 0.88rem;
            color: var(--text-color, #6B7280);
            max-width: 440px;
            margin-bottom: 1.25rem;
            line-height: 1.45;
            opacity: 0.85;
        }

        /* =====================================================================
           7. Dynamic Supply Route Corridor Banner
           ===================================================================== */
        .hscm-corridor-banner {
            background: linear-gradient(90deg, #1E3A8A 0%, #1D4ED8 50%, #2563EB 100%);
            color: #FFFFFF !important;
            padding: 0.85rem 1.25rem;
            border-radius: var(--hscm-radius-md);
            font-weight: 600;
            font-size: 0.98rem;
            margin-bottom: 1.25rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            box-shadow: 0 4px 10px rgba(37, 99, 235, 0.25);
            flex-wrap: wrap;
            gap: 0.5rem;
        }

        .hscm-corridor-route {
            display: flex;
            align-items: center;
            gap: 0.65rem;
        }

        /* Ensure proper high contrast in dark mode tables */
        div[data-testid="stDataFrame"] {
            border-radius: var(--hscm-radius-md);
            overflow: hidden;
            border: 1px solid var(--border-color, rgba(128, 128, 128, 0.2));
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
