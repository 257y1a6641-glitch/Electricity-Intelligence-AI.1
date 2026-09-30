"""
AI/ML Demand Forecasting Module for Hackathon Prototype.
Trains a RandomForestRegressor model on synthetic electricity demand data
and predicts short-term electricity demand based on weather, occupancy,
behavioral patterns, and historical demand.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from data_generator import generate_energy_dataset

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "demand_model.pkl")
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "energy_data.csv")

FEATURE_COLUMNS = [
    "previous_demand",
    "temperature",
    "humidity",
    "occupancy",
    "hour",
    "day_type",
    "user_activity"
]

class DemandForecaster:
    def __init__(self, model_path: str = MODEL_PATH, data_path: str = DATA_PATH):
        self.model_path = model_path
        self.data_path = data_path
        self.model = None
        self.metrics = {}
        self.is_loaded = False

    def load_or_train(self):
        """Loads trained model from disk or trains a new one if missing."""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, "rb") as f:
                    saved_data = pickle.load(f)
                    self.model = saved_data.get("model")
                    self.metrics = saved_data.get("metrics", {})
                    self.is_loaded = True
                    print(f"Loaded existing demand model from {self.model_path}")
                    return
            except Exception as e:
                print(f"Warning: Failed loading model ({e}), retraining...")

        self.train()

    def train(self) -> dict:
        """Trains RandomForestRegressor on historical/synthetic dataset."""
        if not os.path.exists(self.data_path):
            print(f"Dataset not found at {self.data_path}. Generating synthetic data...")
            generate_energy_dataset(n_samples=2000, output_path=self.data_path)

        df = pd.read_csv(self.data_path)
        X = df[FEATURE_COLUMNS]
        y = df["demand"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            min_samples_split=4,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))

        self.model = model
        self.metrics = {
            "mae": round(mae, 2),
            "r2_score": round(r2, 4),
            "dataset_samples": len(df),
            "features": FEATURE_COLUMNS,
            "data_source": "Synthetic demonstration dataset (Hackathon Prototype)"
        }
        self.is_loaded = True

        # Save to disk
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump({"model": self.model, "metrics": self.metrics}, f)

        print(f"Model successfully trained. MAE: {self.metrics['mae']} kW, R2: {self.metrics['r2_score']}")
        return self.metrics

    def predict(self, input_features: dict) -> dict:
        """
        Predicts short-term electricity demand based on input dictionary.
        """
        if not self.is_loaded or self.model is None:
            self.load_or_train()

        # Parse and sanitize input features with robust defaults
        current_demand = float(input_features.get("current_demand", input_features.get("previous_demand", 80.0)))
        temperature = float(input_features.get("temperature", 30.0))
        humidity = float(input_features.get("humidity", 55.0))
        occupancy = float(input_features.get("occupancy", 70.0))
        hour = int(input_features.get("hour", 14)) % 24
        
        # Day type: 0 for weekday, 1 for weekend
        day_type_raw = input_features.get("day_type", 0)
        if isinstance(day_type_raw, str):
            day_type = 1 if "weekend" in day_type_raw.lower() else 0
        else:
            day_type = int(day_type_raw)

        # User activity level: 0: Low, 1: Medium, 2: High
        activity_raw = input_features.get("user_activity", 1)
        if isinstance(activity_raw, str):
            act_map = {"low": 0, "medium": 1, "high": 2}
            user_activity = act_map.get(activity_raw.lower(), 1)
        else:
            user_activity = int(activity_raw)

        row_df = pd.DataFrame([{
            "previous_demand": current_demand,
            "temperature": temperature,
            "humidity": humidity,
            "occupancy": occupancy,
            "hour": hour,
            "day_type": day_type,
            "user_activity": user_activity
        }])[FEATURE_COLUMNS]

        raw_pred = float(self.model.predict(row_df)[0])
        predicted_demand = round(max(10.0, raw_pred), 1)

        # Generate realistic historical trajectory for visualization
        # e.g., 5 past hourly readings leading into current demand + 1 prediction step
        np.random.seed(int(current_demand * 10) % 1000)
        fluctuations = [-0.15, -0.09, -0.04, -0.02, 0.0]
        historical_series = [
            round(max(15.0, current_demand * (1.0 + f) + np.random.uniform(-1.5, 1.5)), 1)
            for f in fluctuations
        ]
        
        # Timeline labels
        timeline_labels = [
            f"T-4h",
            f"T-3h",
            f"T-2h",
            f"T-1h",
            f"Current ({hour:02d}:00)",
            f"Forecast ({(hour + 1) % 24:02d}:00)"
        ]

        return {
            "current_demand": round(current_demand, 1),
            "predicted_demand": predicted_demand,
            "historical_trend": historical_series,
            "forecast_point": predicted_demand,
            "timeline_labels": timeline_labels,
            "model_metrics": self.metrics,
            "input_echo": {
                "temperature": temperature,
                "humidity": humidity,
                "occupancy": occupancy,
                "hour": hour,
                "day_type": "Weekend" if day_type == 1 else "Weekday",
                "user_activity": ["Low", "Medium", "High"][user_activity]
            }
        }

# Global singleton for fast reuse
forecaster = DemandForecaster()

if __name__ == "__main__":
    metrics = forecaster.train()
    sample = {
        "current_demand": 80.0,
        "temperature": 32.0,
        "humidity": 60.0,
        "occupancy": 75.0,
        "hour": 14,
        "day_type": 0,
        "user_activity": 1
    }
    result = forecaster.predict(sample)
    print("Sample Prediction Result:", result)
