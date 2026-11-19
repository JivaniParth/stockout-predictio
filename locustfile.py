"""
Locust High-Throughput Load Testing Harness for HSCM Stockout Prediction Platform.
Simulates concurrent healthcare supply chain officers querying:
1. Single consignment stockout prognostication (/predict)
2. High-volume consignment manifests (/predict/batch)
3. Health and Readiness probes (/health, /ready, /metrics)
"""

from locust import HttpUser, task, between
import random


SAMPLE_SHIPMENTS = [
    {
        "Country": "Congo, DRC",
        "Scheduled_Delivery_Date": "2015-11-20",
        "PO_Sent_to_Vendor_Date": "2015-08-01",
        "Product_Group": "ARV",
        "Sub_Classification": "Adult",
        "Shipment_Mode": "Air",
        "Fulfill_Via": "Direct Drop",
        "Vendor_INCO_Term": "CIP",
        "Vendor": "Aurobindo Pharma Limited",
        "Manufacturing_Site": "Aurobindo Unit III, India",
        "First_Line_Designation": "Yes",
        "Line_Item_Quantity": 25000.0,
        "Line_Item_Value": 125000.0,
        "Pack_Price": 5.0,
        "Unit_Price": 0.08,
        "Unit_of_Measure_Per_Pack": 60.0,
        "Weight_Kilograms": "1450.0",
        "Freight_Cost_USD": "8900.0",
        "Line_Item_Insurance_USD": 140.0,
    },
    {
        "Country": "South Africa",
        "Scheduled_Delivery_Date": "2015-10-15",
        "PO_Sent_to_Vendor_Date": "2015-09-01",
        "Product_Group": "HRDT",
        "Sub_Classification": "HIV test",
        "Shipment_Mode": "Truck",
        "Fulfill_Via": "From RDC",
        "Vendor_INCO_Term": "EXW",
        "Vendor": "Ranbaxy Fine Chemicals LTD.",
        "Manufacturing_Site": "Ranbaxy Fine Chemicals LTD",
        "First_Line_Designation": "Yes",
        "Line_Item_Quantity": 1000.0,
        "Line_Item_Value": 29000.0,
        "Pack_Price": 29.0,
        "Unit_Price": 0.97,
        "Unit_of_Measure_Per_Pack": 30.0,
        "Weight_Kilograms": "80.0",
        "Freight_Cost_USD": "650.0",
        "Line_Item_Insurance_USD": 35.0,
    },
    {
        "Country": "Burundi",
        "Scheduled_Delivery_Date": "2015-12-05",
        "PO_Sent_to_Vendor_Date": "2015-06-10",
        "Product_Group": "ARV",
        "Sub_Classification": "Pediatric",
        "Shipment_Mode": "Air",
        "Fulfill_Via": "Direct Drop",
        "Vendor_INCO_Term": "CIP",
        "Vendor": "Mylan Laboratories GmbH",
        "Manufacturing_Site": "Mylan, India",
        "First_Line_Designation": "Yes",
        "Line_Item_Quantity": 8000.0,
        "Line_Item_Value": 48000.0,
        "Pack_Price": 6.0,
        "Unit_Price": 0.10,
        "Unit_of_Measure_Per_Pack": 60.0,
        "Weight_Kilograms": "450.0",
        "Freight_Cost_USD": "3800.0",
        "Line_Item_Insurance_USD": 55.0,
    },
]


class SupplyChainUser(HttpUser):
    wait_time = between(0.01, 0.05)

    @task(6)
    def test_predict_single(self):
        shipment = random.choice(SAMPLE_SHIPMENTS)
        self.client.post("/predict", json=shipment)

    @task(2)
    def test_predict_batch(self):
        payload = {"shipments": random.choices(SAMPLE_SHIPMENTS, k=random.randint(5, 20))}
        self.client.post("/predict/batch", json=payload)

    @task(1)
    def test_health_check(self):
        self.client.get("/health")
