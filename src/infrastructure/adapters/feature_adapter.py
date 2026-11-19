"""
Layer 3 Adapter: Feature Extraction and Preprocessing Adapter.
Bridges domain entities to the HSCM tabular preprocessor.
"""

from typing import List
import numpy as np
import pandas as pd
from src.domain.entities import ShipmentEntity
from src.application.ports.transformer import IFeatureTransformer
from src.data.preprocessor import HSCMPreprocessor


class HSCMFeatureAdapter(IFeatureTransformer):
    """Adapter bridging domain ShipmentEntities to tabular preprocessor."""

    def __init__(self, preprocessor: HSCMPreprocessor):
        self._preprocessor = preprocessor

    def transform_entities(self, entities: List[ShipmentEntity]) -> np.ndarray:
        rows = []
        for e in entities:
            row = {
                "Country": e.country,
                "Scheduled Delivery Date": e.scheduled_delivery_date,
                "PO Sent to Vendor Date": e.po_sent_to_vendor_date,
                "Manufacturer_Lead_Time_Days": e.manufacturer_lead_time_days,
                "Delivered to Client Date": e.scheduled_delivery_date,
                "Product Group": e.product_group,
                "Sub Classification": e.sub_classification,
                "Shipment Mode": e.shipment_mode,
                "Fulfill Via": e.fulfill_via,
                "Vendor INCO Term": e.vendor_inco_term,
                "Vendor": e.vendor,
                "Manufacturing Site": e.manufacturing_site,
                "First Line Designation": e.first_line_designation,
                "Line Item Quantity": e.line_item_quantity,
                "Line Item Value": e.line_item_value,
                "Pack Price": e.pack_price,
                "Unit Price": e.unit_price,
                "Unit of Measure (Per Pack)": e.unit_of_measure_per_pack,
                "Weight (Kilograms)": e.weight_kilograms,
                "Freight Cost (USD)": e.freight_cost_usd,
                "Line Item Insurance (USD)": e.line_item_insurance_usd,
            }
            rows.append(row)

        raw_df = pd.DataFrame(rows)
        clean_df = self._preprocessor.clean_and_feature_engineer(raw_df)
        return self._preprocessor.transform(clean_df)
