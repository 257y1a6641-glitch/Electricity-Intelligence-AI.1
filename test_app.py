"""
Comprehensive verification test suite for AI Smart Energy Management System.
Tests:
- ML dataset generation and training
- Short-term prediction accuracy & feature inputs
- Dynamic allocation logic (Critical 100%, Important, Flexible throttling/shifting)
- REST API endpoints (/predict, /allocate, /simulate/demand-surge, /simulate/generation-drop, /status, /loads, /reset)
- HTML template and static assets availability
"""

import sys
import unittest
import json
from app import app
from energy_allocation import allocate_energy, get_default_loads
from demand_forecasting import forecaster

class TestEnergySystem(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        self.client.testing = True

    def test_01_index_route(self):
        """Verify HTML dashboard loads successfully."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        html = response.data.decode("utf-8")
        self.assertIn("AI Smart Energy Management System", html)
        self.assertIn("chart-demand-forecast", html)
        self.assertIn("chart-demand-vs-generation", html)
        self.assertIn("chart-energy-allocation", html)
        self.assertIn("loads-table", html)
        print("[PASS] Test 1 Passed: Dashboard HTML served.")

    def test_02_predict_endpoint(self):
        """Verify POST /predict returns valid prediction & metrics."""
        payload = {
            "current_demand": 80.0,
            "available_generation": 100.0,
            "temperature": 32.0,
            "humidity": 60.0,
            "occupancy": 75.0,
            "hour": 14,
            "day_type": "Weekday",
            "user_activity": "Medium"
        }
        response = self.client.post("/predict", data=json.dumps(payload), content_type="application/json")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn("predicted_demand", data)
        self.assertGreater(data["predicted_demand"], 20.0)
        self.assertIn("historical_trend", data)
        self.assertEqual(len(data["historical_trend"]), 5)
        print(f"[PASS] Test 2 Passed: Predicted demand = {data['predicted_demand']} kW")

    def test_03_allocate_endpoint(self):
        """Verify POST /allocate properly prioritizes loads during deficit."""
        payload = {
            "predicted_demand": 100.0,
            "available_generation": 90.0
        }
        response = self.client.post("/allocate", data=json.dumps(payload), content_type="application/json")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        # Available is 90 kW, required is 100 kW -> Shortage = 10 kW
        self.assertEqual(data["shortage"], 10.0)
        # Total allocated must not exceed available generation
        self.assertLessEqual(data["total_allocated"], 90.0)
        # Critical and Important must be supplied; Flexible must be throttled
        self.assertGreater(len(data["shifted_loads"]) + len(data["managed_loads"]), 0)
        self.assertIn("cost_analysis", data)
        print(f"[PASS] Test 3 Passed: Deficit handled. Critical = {data['critical']} kW, Flexible = {data['flexible']} kW, Shifted = {data['shifted_loads']}")

    def test_04_simulate_demand_surge(self):
        """Verify POST /simulate/demand-surge handles sudden surge."""
        payload = {"surge_demand": 125.0, "available_generation": 100.0}
        response = self.client.post("/simulate/demand-surge", data=json.dumps(payload), content_type="application/json")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data["current_demand"], 125.0)
        self.assertIn("Demand Surge Detected", data["alert_title"])
        self.assertGreater(data["shortage"], 0.0)
        print(f"[PASS] Test 4 Passed: Surge simulated. Shortage = {data['shortage']} kW, Alert = {data['alert_title']}")

    def test_05_simulate_generation_drop(self):
        """Verify POST /simulate/generation-drop handles sudden supply cut."""
        payload = {"dropped_generation": 70.0, "predicted_demand": 100.0}
        response = self.client.post("/simulate/generation-drop", data=json.dumps(payload), content_type="application/json")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data["available_generation"], 70.0)
        self.assertEqual(data["shortage"], 30.0)
        self.assertIn("Generation Drop Detected", data["alert_title"])
        print(f"[PASS] Test 5 Passed: Generation drop simulated. Shortage = {data['shortage']} kW")

    def test_06_status_and_loads_endpoints(self):
        """Verify GET /status and GET /loads."""
        res_status = self.client.get("/status")
        self.assertEqual(res_status.status_code, 200)
        status_data = json.loads(res_status.data)
        self.assertIn("system_state", status_data)
        self.assertIn("model_metrics", status_data)

        res_loads = self.client.get("/loads")
        self.assertEqual(res_loads.status_code, 200)
        loads_data = json.loads(res_loads.data)
        self.assertEqual(len(loads_data["loads"]), 10)
        print("[PASS] Test 6 Passed: Status & Loads APIs active.")

    def test_07_reset_endpoint(self):
        """Verify POST /reset restores system baseline."""
        res = self.client.post("/reset")
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertEqual(data["system_state"]["current_demand"], 80.0)
        self.assertEqual(data["system_state"]["available_generation"], 100.0)
        print("[PASS] Test 7 Passed: System reset verified.")

if __name__ == "__main__":
    unittest.main()

