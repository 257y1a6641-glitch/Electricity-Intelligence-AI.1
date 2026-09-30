"""
Synthetic Energy Data Generator for Hackathon Demonstration.
Generates realistic electricity demand data reflecting diurnal patterns,
temperature-driven HVAC spikes, occupancy fluctuations, and user activity.
"""

import os
import numpy as np
import pandas as pd

def generate_energy_dataset(n_samples: int = 1500, output_path: str = "data/energy_data.csv") -> pd.DataFrame:
    """
    Generates synthetic electricity demand records.
    NOTE: This dataset is synthetically generated for hackathon prototyping and evaluation.
    """
    np.random.seed(42)

    # Time & behavioral variables
    hours = np.random.randint(0, 24, size=n_samples)
    day_types = np.random.choice([0, 1], size=n_samples, p=[0.71, 0.29])  # 0: Weekday, 1: Weekend
    user_activity = np.random.choice([0, 1, 2], size=n_samples, p=[0.25, 0.50, 0.25])  # 0: Low, 1: Med, 2: High

    # Environmental variables
    # Higher temps in afternoon hours (12-16)
    temp_base = 24.0 + 8.0 * np.sin((hours - 8) / 24.0 * np.pi)
    temperatures = np.clip(temp_base + np.random.normal(0, 4, size=n_samples), 16.0, 42.0)
    humidity = np.clip(60.0 - (temperatures - 25.0) * 0.8 + np.random.normal(0, 8, size=n_samples), 25.0, 95.0)

    # Occupancy depends on hour and weekday/weekend
    occupancy_base = np.where(
        day_types == 0,
        np.where((hours >= 8) & (hours <= 18), 75.0, 25.0),  # Weekday office/commercial
        np.where((hours >= 10) & (hours <= 20), 60.0, 40.0)   # Weekend residential/retail
    )
    occupancy = np.clip(occupancy_base + np.random.normal(0, 12, size=n_samples), 5.0, 100.0)

    # Previous demand (baseline with diurnal cycle)
    prev_demand_base = 50.0 + 25.0 * np.sin((hours - 6) / 24.0 * 2 * np.pi)
    previous_demand = np.clip(prev_demand_base + np.random.normal(0, 8, size=n_samples), 35.0, 130.0)

    # Physics & behavior-inspired electricity demand calculation
    # Base load:
    demand = 25.0 + 0.30 * previous_demand
    
    # Hour peak factor (morning ramp 9-11, evening peak 17-21)
    hour_factor = np.where((hours >= 9) & (hours <= 12), 15.0, 0.0) + \
                  np.where((hours >= 17) & (hours <= 21), 20.0, 0.0) + \
                  np.where((hours >= 1) & (hours <= 5), -10.0, 0.0)
    demand += hour_factor

    # Temperature sensitivity (cooling degree effect above 24°C)
    cooling_effect = np.maximum(0.0, temperatures - 24.0) * 1.8
    demand += cooling_effect

    # Occupancy effect: more people = more computing, lights, amenities
    demand += (occupancy / 100.0) * 22.0

    # User activity tier
    demand += user_activity * 7.5

    # Day type offset (weekdays have higher industrial/office baseline)
    demand += (1 - day_types) * 6.0

    # Realistic random noise
    demand += np.random.normal(0, 2.5, size=n_samples)
    demand = np.clip(demand, 30.0, 160.0).round(2)

    df = pd.DataFrame({
        "previous_demand": previous_demand.round(2),
        "temperature": temperatures.round(1),
        "humidity": humidity.round(1),
        "occupancy": occupancy.round(1),
        "hour": hours,
        "day_type": day_types,
        "user_activity": user_activity,
        "demand": demand
    })

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset generated with {len(df)} rows at {output_path}")
    return df

if __name__ == "__main__":
    generate_energy_dataset()
