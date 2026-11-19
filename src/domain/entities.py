"""
Layer 1: Domain Entities and Value Objects.
Pure domain models independent of any framework, database, or UI library.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class HealthEquityTier(str, Enum):
    TIER_1_LOW_RISK_HUB = "Tier_1_Low_Risk_Hub"
    TIER_2_MODERATE_RISK = "Tier_2_Moderate_Risk"
    TIER_3_HIGH_RISK_UNDERSERVED = "Tier_3_High_Risk_Underserved"


class RiskClassification(str, Enum):
    LOW_RISK = "Low Risk"
    MODERATE_RISK = "Moderate Risk"
    HIGH_RISK = "High Risk"
    CRITICAL_RISK = "Critical Stockout Risk"


@dataclass(frozen=True)
class ShipmentEntity:
    """Immutable domain representation of an international pharmaceutical consignment."""
    country: str
    scheduled_delivery_date: str
    po_sent_to_vendor_date: Optional[str]
    manufacturer_lead_time_days: float
    product_group: str
    sub_classification: str
    shipment_mode: str
    fulfill_via: str
    vendor_inco_term: str
    vendor: str
    manufacturing_site: str
    first_line_designation: str
    line_item_quantity: float
    line_item_value: float
    pack_price: float
    unit_price: float
    unit_of_measure_per_pack: float
    weight_kilograms: str
    freight_cost_usd: str
    line_item_insurance_usd: float


@dataclass(frozen=True)
class PredictionPrognosis:
    """Domain result of stockout risk prognostication."""
    late_delivery_risk: int
    risk_probability: float
    risk_tier: RiskClassification
    country_tier: HealthEquityTier
    qos_latency_ms: float
