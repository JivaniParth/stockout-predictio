"""
Layer 1: Domain Policies and Business Invariants.
Encapsulates pharmaceutical supply chain rules, health equity classification,
and manufacturer SLA turnaround evaluation.
"""

from src.domain.entities import HealthEquityTier, RiskClassification

# Historical vulnerability gradients established in review paper draft
TIER_1_COUNTRIES = {
    "South Africa", "Vietnam", "Ethiopia", "Namibia", "Botswana",
    "Dominican Republic", "Guatemala",
}

TIER_3_HIGH_RISK_COUNTRIES = {
    "Burundi", "Congo, DRC", "Mozambique", "Zambia", "South Sudan",
    "Togo", "Senegal", "Cameroon",
}


class HealthEquityPolicy:
    """Evaluates healthcare infrastructure vulnerability across geographic tiers."""

    @staticmethod
    def classify_country(country: str) -> HealthEquityTier:
        if country in TIER_3_HIGH_RISK_COUNTRIES:
            return HealthEquityTier.TIER_3_HIGH_RISK_UNDERSERVED
        elif country in TIER_1_COUNTRIES:
            return HealthEquityTier.TIER_1_LOW_RISK_HUB
        return HealthEquityTier.TIER_2_MODERATE_RISK


class RiskClassificationPolicy:
    """Classifies continuous probability into clinical severity tiers."""

    @staticmethod
    def classify(probability: float) -> RiskClassification:
        if probability < 0.25:
            return RiskClassification.LOW_RISK
        elif probability < 0.50:
            return RiskClassification.MODERATE_RISK
        elif probability < 0.75:
            return RiskClassification.HIGH_RISK
        return RiskClassification.CRITICAL_RISK


class LeadTimePolicy:
    """Analyzes manufacturer-quoted lead times versus client arrival targets."""

    @staticmethod
    def evaluate_sla(mfg_lead_time_days: float, client_window_days: int) -> dict:
        if client_window_days < 0:
            return {
                "status": "INVALID_TIMELINE",
                "is_squeezed": True,
                "difference_days": client_window_days,
                "message": "Scheduled delivery deadline occurs prior to Purchase Order placement.",
            }
        elif client_window_days < mfg_lead_time_days:
            deficit = int(mfg_lead_time_days - client_window_days)
            return {
                "status": "LEAD_TIME_SQUEEZE",
                "is_squeezed": True,
                "difference_days": -deficit,
                "message": f"Client window is short by {deficit} days against manufacturer SLA.",
            }
        buffer = int(client_window_days - mfg_lead_time_days)
        return {
            "status": "SLA_ALIGNED",
            "is_squeezed": False,
            "difference_days": buffer,
            "message": f"Client window provides {buffer} days safety buffer over manufacturer SLA.",
        }
