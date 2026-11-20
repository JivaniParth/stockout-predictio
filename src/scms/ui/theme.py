"""
Design System and Custom Styling for HSCM Decision Support Platform.
Accessible typography, high-contrast badges, clinical triage cards, and responsive layout.
"""

import streamlit as st


def inject_custom_theme():
    """Injects custom CSS styling for high-contrast, accessible clinical supply chain UI."""
    st.markdown(
        """
        <style>
        /* Base typography & containers */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        /* Metric cards */
        .metric-card {
            background-color: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(128, 128, 128, 0.2);
            border-radius: 10px;
            padding: 18px 20px;
            margin-bottom: 15px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }

        /* Decision Badges */
        .badge-alert {
            background-color: #fee2e2;
            color: #991b1b;
            border: 1px solid #f87171;
            padding: 4px 12px;
            border-radius: 9999px;
            font-weight: 700;
            font-size: 0.88rem;
            display: inline-block;
        }
        .badge-safe {
            background-color: #dcfce7;
            color: #166534;
            border: 1px solid #4ade80;
            padding: 4px 12px;
            border-radius: 9999px;
            font-weight: 700;
            font-size: 0.88rem;
            display: inline-block;
        }
        .badge-boundary {
            background-color: #fef3c7;
            color: #92400e;
            border: 1px solid #facc15;
            padding: 4px 12px;
            border-radius: 9999px;
            font-weight: 700;
            font-size: 0.88rem;
            display: inline-block;
        }

        /* Callout banners */
        .disclosure-box {
            background-color: rgba(245, 158, 11, 0.1);
            border-left: 4px solid #f59e0b;
            padding: 14px 18px;
            border-radius: 4px;
            margin: 12px 0 20px 0;
            font-size: 0.92rem;
        }
        .warning-box {
            background-color: rgba(239, 68, 68, 0.1);
            border-left: 4px solid #ef4444;
            padding: 14px 18px;
            border-radius: 4px;
            margin: 12px 0 20px 0;
            font-size: 0.92rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
