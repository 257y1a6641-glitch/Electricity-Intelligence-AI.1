# AI-Based Smart Electricity Demand Forecasting and Dynamic Energy Allocation

> **Hackathon Prototype** | Smart Energy Management System balancing short-term electricity demand forecasting with priority-based dynamic load shedding and time-of-use cost optimization.

---

## 1. Project Title
**AI-Based Smart Electricity Demand Forecasting and Dynamic Energy Allocation**

---

## 2. Problem Statement
Electricity demand varies significantly with time, ambient weather (temperature & humidity), building occupancy, and consumer activity. This makes it difficult for microgrids, facility managers, and smart grid operators to balance fluctuating demand with limited, intermittent generation capacity (such as solar or limited grid feed). When demand suddenly surges or generation capacity drops, systems without dynamic control suffer from catastrophic grid overloads, blackouts, or excessive peak-rate tariff penalties.

---

## 3. Proposed Solution
A web-based **AI Smart Energy Management System** that bridges short-term demand forecasting with intelligent multi-tier load control:
1. **Forecasts Short-Term Demand**: Uses an AI/ML regression model (`RandomForestRegressor`) to predict demand based on previous load, temperature, humidity, occupancy, hour of day, day type, and user activity.
2. **Classifies Loads into Three Priority Tiers**:
   - **Critical Loads** (Priority 1): Medical equipment, emergency alarms, data servers — guaranteed 100% uninterrupted power.
   - **Important Loads** (Priority 2): Office lighting, workstations, primary HVAC — powered immediately after critical loads.
   - **Flexible Loads** (Priority 3): EV fast charging, water heating, auxiliary HVAC, industrial washing — dynamically throttled or shifted to off-peak periods during deficits rather than blacking out the whole system.
3. **Automated Deficit / Surge Mitigation**: Reacts in real time to demand surges and generation dropouts, reallocating energy dynamically and computing cost savings.
4. **Interactive Command Center**: Real-time interactive dashboard featuring Chart.js timelines, live load allocation tables, and 1-click hackathon test scenarios.

---

## 4. System Workflow

```
INPUT TELEMETRY & BEHAVIORAL DATA
(Demand, Temp, Humidity, Occupancy, Hour, Day Type, User Activity)
                        ↓
            AI/ML DEMAND PREDICTION
             (RandomForestRegressor)
                        ↓
    COMPARE PREDICTED DEMAND VS AVAILABLE GENERATION
                        ↓
           CLASSIFY LOADS BY PRIORITY
         (Critical → Important → Flexible)
                        ↓
        DYNAMIC ENERGY ALLOCATION ENGINE
      (Protect Critical → Supply Important → Manage/Shift Flexible)
                        ↓
    DETECT SUDDEN DEMAND RISE OR GENERATION DROP
                        ↓
             REAL-TIME REALLOCATION
                        ↓
        TIME-OF-USE ELECTRICITY COST ESTIMATION
                        ↓
      INTERACTIVE DASHBOARD & SYSTEM ALERTS
```

---

## 5. Key Features
- **AI-Powered Demand Forecast**: Short-term kW demand prediction with historical trend timeline.
- **Smart Load Shedding & Shifting**: Does not abruptly disconnect everything; intelligently throttles EV chargers and defers deferrable heating/processing tasks.
- **Real-Time Surge & Drop Simulation**: Dedicated buttons to trigger instant commercial demand surges (+25 kW) and generation dropouts (-30 kW).
- **Time-of-Use Cost Optimizer**: Calculates real-time cost differences, peak surcharge avoidance, and off-peak savings.
- **1-Click Hackathon Scenarios**: 5 pre-configured demo scenarios for judges.
- **Dynamic Fulfillment Badges**: Visual progress bars and status indicators for every connected appliance and subsystem.

---

## 6. AI/ML Approach
- **Algorithm**: `RandomForestRegressor` (100 estimators, max depth 12).
- **Features Used**:
  - `previous_demand`: Historical electricity demand (kW).
  - `temperature`: Ambient temperature (°C) driving HVAC cooling loads.
  - `humidity`: Humidity index (%).
  - `occupancy`: Building occupancy level (0-100%).
  - `hour`: Diurnal curve (0-23 hours).
  - `day_type`: Weekday vs. Weekend operational profile.
  - `user_activity`: Behavioral tier (Low, Medium, High).
- **Validation**: 80/20 train/test split on synthetic data.
- **Performance**:
  - **Mean Absolute Error (MAE)**: ~3.27 kW
  - **R² Score**: ~0.96
- *Note on Data*: A realistic synthetic dataset (`data/energy_data.csv`, 2,000 samples) calibrated to real-world diurnal and thermodynamic HVAC curves is provided for hackathon demonstration.

---

## 7. Dynamic Energy Allocation Approach
When total demand exceeds available generation:
1. **Critical Loads**: Medical equipment (20 kW), Emergency systems (10 kW), Essential servers (15 kW) receive power first.
2. **Important Loads**: Lighting (15 kW), Workstations (15 kW), Essential HVAC (10 kW) are fulfilled next.
3. **Flexible Loads**: EV Fast Chargers (25 kW), Water Heaters (15 kW), Chillers (15 kW), Heavy Equipment (10 kW) are intelligently throttled (e.g., 50% eco-charge) or shifted to off-peak periods, maintaining overall system stability within available generation.

---

## 8. Technology Stack
- **Frontend**: HTML5, CSS3 (Modern Glassmorphic Dark UI), Vanilla JavaScript (ES6+), Chart.js (CDN).
- **Backend**: Python 3.10+, Flask, REST API.
- **AI/ML**: `scikit-learn`, `pandas`, `numpy`.
- **Packaging/Environment**: `uv` or standard Python `pip`.

---

## 9. Project Structure

```
AI-Energy-Demand-Forecasting/
│
├── app.py                     # Flask web server and REST API endpoints
├── demand_forecasting.py      # ML model training and short-term prediction logic
├── energy_allocation.py       # Multi-tier priority allocation algorithm & cost analysis
├── data_generator.py          # Synthetic dataset generator with diurnal & weather patterns
├── requirements.txt           # Python dependencies
├── README.md                  # Complete hackathon documentation
│
├── data/
│   └── energy_data.csv        # Synthetic demonstration dataset
│
├── models/
│   └── demand_model.pkl       # Serialized trained RandomForestRegressor model
│
├── templates/
│   └── index.html             # Responsive dashboard layout
│
└── static/
    ├── style.css              # Dark theme styling, cards, tables, and alert badges
    └── script.js              # REST client, Chart.js managers, and scenario drivers
```

---

## 10. Installation & Prerequisites

### Prerequisites
- Python 3.9, 3.10, 3.11, or 3.12+

### Step-by-Step Installation
1. Open a terminal in the project directory:
   ```bash
   cd AI-Energy-Demand-Forecasting
   ```

2. (Optional but recommended) Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # Windows PowerShell:
   .\.venv\Scripts\activate
   # Linux / macOS:
   source .venv/bin/activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

---

## 11. How to Run the Application

1. Start the Flask server:
   ```bash
   python app.py
   ```

2. Open your web browser and navigate to:
   ```
   http://127.0.0.1:5000
   ```

The dashboard will open with the trained AI model loaded and initial load allocations displayed.

---

## 12. REST API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the interactive frontend dashboard. |
| `POST` | `/predict` | Evaluates ML model against operating features and returns predicted demand. |
| `POST` | `/allocate` | Runs dynamic energy allocation across load priority tiers. |
| `POST` | `/simulate/demand-surge` | Simulates a sudden demand spike (125 kW), reallocates power, and returns alert. |
| `POST` | `/simulate/generation-drop`| Simulates a generation capacity drop (70 kW), reallocates power, and returns alert. |
| `GET` | `/status` | Returns system health, active telemetry, and ML model performance metrics. |
| `GET` | `/loads` | Returns default classified loads specification. |
| `POST` | `/reset` | Restores default baseline inputs and normal operational state. |

---

## 13. Predefined Demo Scenarios for Judges

Use the 1-click buttons in the **Demo Scenarios** panel on the dashboard:
1. **Scenario 1: Normal Demand** (80 kW demand, 100 kW generation, 25°C, 50% occ) → System operates normally with surplus energy buffer.
2. **Scenario 2: High Demand** (95 kW demand, 90 kW generation, 34°C, 85% occ) → AI forecasts ~108 kW demand; system protects Critical & Important loads, throttles EV charging.
3. **Scenario 3: Sudden Demand Surge** → Instantly injects a 125 kW load surge; triggers critical shortage defense protocol.
4. **Scenario 4: Generation Drop** → Generation drops from 100 kW to 70 kW; non-critical flexible loads are shifted to avoid blackout.
5. **Scenario 5: Heatwave & High Occupancy** (39.5°C, 98% occ) → Simulates peak summer HVAC strain with automated load balancing.

---

## 14. Example API Request & Response

### Request: `POST /predict`
```json
{
  "current_demand": 80.0,
  "available_generation": 90.0,
  "temperature": 32.0,
  "humidity": 60.0,
  "occupancy": 75.0,
  "hour": 14,
  "day_type": "Weekday",
  "user_activity": "Medium"
}
```

### Response:
```json
{
  "available_generation": 90.0,
  "current_demand": 80.0,
  "predicted_demand": 94.0,
  "shortage": 4.0,
  "surplus": 0.0,
  "status": "Deficit Expected",
  "alert_message": "Predicted demand (94.0 kW) exceeds available generation (90.0 kW). Shortage: 4.0 kW."
}
```

---

## 15. Future Improvements
- **Real-Time IoT Integration**: Connect with smart meters and MQTT telemetry streams.
- **Battery Energy Storage (BESS) Orchestration**: Automatically dispatch battery storage during peak tariff intervals.
- **Deep Reinforcement Learning**: Train a PPO/DQN agent to balance real-time electricity spot prices with thermal comfort constraints.
- **Weather API Integration**: Fetch live 24-hour solar irradiance and temperature forecasts from OpenWeather or NOAA.
