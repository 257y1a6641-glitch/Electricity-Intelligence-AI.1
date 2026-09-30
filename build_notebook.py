"""
Generates the complete AI_Smart_Energy_Management.ipynb Jupyter Notebook
containing all source code, data generation, ML training, dynamic allocation,
simulations, visualizations, Flask backend, and frontend code.
"""

import json
import os

def make_cell(cell_type, source, outputs=None):
    if isinstance(source, list):
        src_lines = [line if line.endswith("\n") else line + "\n" for line in source]
        # Remove trailing newline from last line for clean json
        if src_lines:
            src_lines[-1] = src_lines[-1].rstrip("\n")
    else:
        src_lines = [source]

    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": src_lines
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = outputs or []
    return cell

def build_notebook():
    cells = []

    # Title & Header
    cells.append(make_cell("markdown", [
        "# AI-Based Smart Electricity Demand Forecasting and Dynamic Energy Allocation",
        "",
        "> **Hackathon End-to-End Prototype & Project Notebook**  ",
        "> Combining Machine Learning Short-Term Demand Forecasting with Priority-Based Dynamic Energy Allocation, Demand Surge/Generation Drop Simulations, and Interactive Dashboard Telemetry.",
        "",
        "---",
        "## Table of Contents",
        "1. **Project Overview & System Architecture**",
        "2. **Step 1: Environment Setup & Library Imports**",
        "3. **Step 2: Synthetic Energy Dataset Generation (`data_generator.py`)**",
        "4. **Step 3: Exploratory Data Analysis & Feature Engineering**",
        "5. **Step 4: AI/ML Demand Forecasting (`demand_forecasting.py`)**",
        "6. **Step 5: Load Classification & Dynamic Energy Allocation (`energy_allocation.py`)**",
        "7. **Step 6: Hackathon Scenario Simulations (Surge & Generation Drop)**",
        "8. **Step 7: Flask Backend API Implementation (`app.py`)**",
        "9. **Step 8: Full Frontend Dashboard Code (HTML, CSS, JS)**",
        "10. **Step 9: Testing, Verification & Deployment**"
    ]))

    # Step 1: Imports
    cells.append(make_cell("markdown", [
        "---",
        "## 1. Environment Setup & Library Imports",
        "",
        "Install dependencies if running in Colab or a fresh environment:",
        "```bash",
        "!pip install Flask numpy pandas scikit-learn matplotlib seaborn",
        "```"
    ]))

    cells.append(make_cell("code", [
        "import os",
        "import sys",
        "import copy",
        "import pickle",
        "import numpy as np",
        "import pandas as pd",
        "import matplotlib.pyplot as plt",
        "from sklearn.ensemble import RandomForestRegressor",
        "from sklearn.model_selection import train_test_split",
        "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score",
        "",
        "# Configure visual plot styling",
        "plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')",
        "plt.rcParams['figure.figsize'] = (10, 5)",
        "print('Libraries imported successfully!')"
    ]))

    # Step 2: Synthetic Data Generator
    cells.append(make_cell("markdown", [
        "---",
        "## 2. Synthetic Energy Dataset Generation (`data_generator.py`)",
        "",
        "Generates realistic electricity demand data reflecting diurnal patterns, temperature-driven HVAC spikes, occupancy fluctuations, and user behavior.",
        "*Note: Synthetic data calibrated for hackathon demonstration.*"
    ]))

    cells.append(make_cell("code", [
        "def generate_energy_dataset(n_samples: int = 2000, output_path: str = 'data/energy_data.csv') -> pd.DataFrame:",
        "    np.random.seed(42)",
        "",
        "    # Time & behavioral variables",
        "    hours = np.random.randint(0, 24, size=n_samples)",
        "    day_types = np.random.choice([0, 1], size=n_samples, p=[0.71, 0.29])  # 0: Weekday, 1: Weekend",
        "    user_activity = np.random.choice([0, 1, 2], size=n_samples, p=[0.25, 0.50, 0.25])  # 0: Low, 1: Med, 2: High",
        "",
        "    # Environmental variables (warmer afternoons)",
        "    temp_base = 24.0 + 8.0 * np.sin((hours - 8) / 24.0 * np.pi)",
        "    temperatures = np.clip(temp_base + np.random.normal(0, 4, size=n_samples), 16.0, 42.0)",
        "    humidity = np.clip(60.0 - (temperatures - 25.0) * 0.8 + np.random.normal(0, 8, size=n_samples), 25.0, 95.0)",
        "",
        "    # Occupancy depends on diurnal cycle and weekday/weekend",
        "    occupancy_base = np.where(",
        "        day_types == 0,",
        "        np.where((hours >= 8) & (hours <= 18), 75.0, 25.0),",
        "        np.where((hours >= 10) & (hours <= 20), 60.0, 40.0)",
        "    )",
        "    occupancy = np.clip(occupancy_base + np.random.normal(0, 12, size=n_samples), 5.0, 100.0)",
        "",
        "    # Previous demand (baseline with diurnal cycle)",
        "    prev_demand_base = 50.0 + 25.0 * np.sin((hours - 6) / 24.0 * 2 * np.pi)",
        "    previous_demand = np.clip(prev_demand_base + np.random.normal(0, 8, size=n_samples), 35.0, 130.0)",
        "",
        "    # Physics- & behavior-based demand calculation",
        "    demand = 25.0 + 0.30 * previous_demand",
        "",
        "    # Hour peak factor (morning ramp 9-12, evening peak 17-21)",
        "    hour_factor = np.where((hours >= 9) & (hours <= 12), 15.0, 0.0) + \\",
        "                  np.where((hours >= 17) & (hours <= 21), 20.0, 0.0) + \\",
        "                  np.where((hours >= 1) & (hours <= 5), -10.0, 0.0)",
        "    demand += hour_factor",
        "",
        "    # Temperature cooling degree effect above 24°C",
        "    cooling_effect = np.maximum(0.0, temperatures - 24.0) * 1.8",
        "    demand += cooling_effect",
        "",
        "    # Occupancy effect",
        "    demand += (occupancy / 100.0) * 22.0",
        "",
        "    # User activity tier effect",
        "    demand += user_activity * 7.5",
        "",
        "    # Weekday base commercial activity",
        "    demand += (1 - day_types) * 6.0",
        "",
        "    # Gaussian noise",
        "    demand += np.random.normal(0, 2.5, size=n_samples)",
        "    demand = np.clip(demand, 30.0, 160.0).round(2)",
        "",
        "    df = pd.DataFrame({",
        "        'previous_demand': previous_demand.round(2),",
        "        'temperature': temperatures.round(1),",
        "        'humidity': humidity.round(1),",
        "        'occupancy': occupancy.round(1),",
        "        'hour': hours,",
        "        'day_type': day_types,",
        "        'user_activity': user_activity,",
        "        'demand': demand",
        "    })",
        "",
        "    os.makedirs(os.path.dirname(output_path), exist_ok=True)",
        "    df.to_csv(output_path, index=False)",
        "    print(f'Generated {len(df)} samples saved to {output_path}')",
        "    return df",
        "",
        "df_energy = generate_energy_dataset(2000)",
        "df_energy.head()"
    ]))

    # Step 3: EDA Visualizations
    cells.append(make_cell("markdown", [
        "---",
        "## 3. Exploratory Data Analysis & Feature Distributions"
    ]))

    cells.append(make_cell("code", [
        "fig, axes = plt.subplots(2, 2, figsize=(14, 10))",
        "",
        "# Plot 1: Temperature vs Electricity Demand",
        "axes[0, 0].scatter(df_energy['temperature'], df_energy['demand'], alpha=0.3, color='#38bdf8')",
        "axes[0, 0].set_title('Temperature (°C) vs. Demand (kW)')",
        "axes[0, 0].set_xlabel('Temperature (°C)')",
        "axes[0, 0].set_ylabel('Electricity Demand (kW)')",
        "",
        "# Plot 2: Average Hourly Demand Profile",
        "hourly_avg = df_energy.groupby('hour')['demand'].mean()",
        "axes[0, 1].plot(hourly_avg.index, hourly_avg.values, marker='o', color='#10b981', linewidth=2.5)",
        "axes[0, 1].set_title('Diurnal Electricity Demand Curve (Hour of Day)')",
        "axes[0, 1].set_xlabel('Hour (0 - 23)')",
        "axes[0, 1].set_ylabel('Average Demand (kW)')",
        "axes[0, 1].set_xticks(range(0, 24, 2))",
        "",
        "# Plot 3: Occupancy vs Demand",
        "axes[1, 0].scatter(df_energy['occupancy'], df_energy['demand'], alpha=0.3, color='#a855f7')",
        "axes[1, 0].set_title('Occupancy (%) vs. Demand (kW)')",
        "axes[1, 0].set_xlabel('Occupancy (%)')",
        "axes[1, 0].set_ylabel('Electricity Demand (kW)')",
        "",
        "# Plot 4: Demand Distribution Histogram",
        "axes[1, 1].hist(df_energy['demand'], bins=25, color='#f59e0b', edgecolor='black', alpha=0.75)",
        "axes[1, 1].set_title('Electricity Demand Distribution')",
        "axes[1, 1].set_xlabel('Demand (kW)')",
        "axes[1, 1].set_ylabel('Count')",
        "",
        "plt.tight_layout()",
        "plt.show()"
    ]))

    # Step 4: Demand Forecasting Model Training
    cells.append(make_cell("markdown", [
        "---",
        "## 4. AI/ML Demand Forecasting (`demand_forecasting.py`)",
        "",
        "We train a `RandomForestRegressor` model to predict short-term demand based on lag demand, weather, occupancy, and behavioral patterns."
    ]))

    cells.append(make_cell("code", [
        "FEATURE_COLUMNS = [",
        "    'previous_demand',",
        "    'temperature',",
        "    'humidity',",
        "    'occupancy',",
        "    'hour',",
        "    'day_type',",
        "    'user_activity'",
        "]",
        "",
        "X = df_energy[FEATURE_COLUMNS]",
        "y = df_energy['demand']",
        "",
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)",
        "",
        "rf_model = RandomForestRegressor(",
        "    n_estimators=100,",
        "    max_depth=12,",
        "    min_samples_split=4,",
        "    random_state=42,",
        "    n_jobs=-1",
        ")",
        "rf_model.fit(X_train, y_train)",
        "",
        "y_pred = rf_model.predict(X_test)",
        "mae = mean_absolute_error(y_test, y_pred)",
        "rmse = np.sqrt(mean_squared_error(y_test, y_pred))",
        "r2 = r2_score(y_test, y_pred)",
        "",
        "print('=== MODEL PERFORMANCE METRICS ===')",
        "print(f'Mean Absolute Error (MAE) : {mae:.2f} kW')",
        "print(f'Root Mean Squared Error   : {rmse:.2f} kW')",
        "print(f'R² Accuracy Score         : {r2:.4f} ({r2*100:.1f}%)')",
        "",
        "# Save trained model",
        "os.makedirs('models', exist_ok=True)",
        "with open('models/demand_model.pkl', 'wb') as f:",
        "    pickle.dump({'model': rf_model, 'metrics': {'mae': mae, 'r2_score': r2}}, f)",
        "print('Model saved to models/demand_model.pkl')"
    ]))

    # Feature Importance Plot
    cells.append(make_cell("code", [
        "plt.figure(figsize=(10, 4))",
        "importances = pd.Series(rf_model.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=True)",
        "importances.plot(kind='barh', color='#38bdf8', edgecolor='black')",
        "plt.title('Random Forest Feature Importance in Electricity Demand Forecasting')",
        "plt.xlabel('Relative Importance Score')",
        "plt.show()"
    ]))

    # Step 5: Energy Allocation Engine
    cells.append(make_cell("markdown", [
        "---",
        "## 5. Load Classification & Dynamic Energy Allocation (`energy_allocation.py`)",
        "",
        "### Load Categories:",
        "1. **Critical Loads (Priority 1)**: Medical equipment (20 kW), Emergency systems (10 kW), Essential servers (15 kW). Must never be dropped.",
        "2. **Important Loads (Priority 2)**: Lighting (15 kW), Workstations (15 kW), Essential HVAC (10 kW). Supplied immediately after critical.",
        "3. **Flexible Loads (Priority 3)**: EV Fast Chargers (25 kW), Water Heating (15 kW), Auxiliary HVAC (15 kW), Washing (10 kW). Dynamically throttled/shifted during deficits to eliminate blackout risk."
    ]))

    cells.append(make_cell("code", [
        "DEFAULT_LOADS = [",
        "    {'id': 'load_1', 'name': 'Medical Equipment', 'category': 'Critical', 'priority': 1, 'is_flexible': False, 'required_kw': 20.0, 'allocated_kw': 20.0, 'status': 'Active'},",
        "    {'id': 'load_2', 'name': 'Emergency Systems', 'category': 'Critical', 'priority': 1, 'is_flexible': False, 'required_kw': 10.0, 'allocated_kw': 10.0, 'status': 'Active'},",
        "    {'id': 'load_3', 'name': 'Essential Servers', 'category': 'Critical', 'priority': 1, 'is_flexible': False, 'required_kw': 15.0, 'allocated_kw': 15.0, 'status': 'Active'},",
        "    {'id': 'load_4', 'name': 'Lighting', 'category': 'Important', 'priority': 2, 'is_flexible': False, 'required_kw': 15.0, 'allocated_kw': 15.0, 'status': 'Active'},",
        "    {'id': 'load_5', 'name': 'Computers & Workstations', 'category': 'Important', 'priority': 2, 'is_flexible': False, 'required_kw': 15.0, 'allocated_kw': 15.0, 'status': 'Active'},",
        "    {'id': 'load_6', 'name': 'Essential HVAC', 'category': 'Important', 'priority': 2, 'is_flexible': False, 'required_kw': 10.0, 'allocated_kw': 10.0, 'status': 'Active'},",
        "    {'id': 'load_7', 'name': 'EV Fast Chargers', 'category': 'Flexible', 'priority': 3, 'is_flexible': True, 'required_kw': 25.0, 'allocated_kw': 25.0, 'status': 'Active'},",
        "    {'id': 'load_8', 'name': 'Water Heating System', 'category': 'Flexible', 'priority': 3, 'is_flexible': True, 'required_kw': 15.0, 'allocated_kw': 15.0, 'status': 'Active'},",
        "    {'id': 'load_9', 'name': 'Non-Critical HVAC / Chillers', 'category': 'Flexible', 'priority': 3, 'is_flexible': True, 'required_kw': 15.0, 'allocated_kw': 15.0, 'status': 'Active'},",
        "    {'id': 'load_10', 'name': 'Washing & Heavy Equipment', 'category': 'Flexible', 'priority': 3, 'is_flexible': True, 'required_kw': 10.0, 'allocated_kw': 10.0, 'status': 'Active'}",
        "]",
        "",
        "def scale_loads_to_demand(loads, target_demand):",
        "    current_sum = sum(l['required_kw'] for l in loads)",
        "    if current_sum <= 0 or target_demand <= 0:",
        "        return loads",
        "    scale_factor = target_demand / current_sum",
        "    scaled = copy.deepcopy(loads)",
        "    allocated_so_far = 0.0",
        "    for i, l in enumerate(scaled):",
        "        if i == len(scaled) - 1:",
        "            l['required_kw'] = round(target_demand - allocated_so_far, 1)",
        "        else:",
        "            l['required_kw'] = round(l['required_kw'] * scale_factor, 1)",
        "            allocated_so_far += l['required_kw']",
        "        l['allocated_kw'] = l['required_kw']",
        "    return scaled",
        "",
        "def allocate_energy(available_generation: float, predicted_demand: float = None, loads: list = None, electricity_price_kwh: float = 0.16) -> dict:",
        "    loads = copy.deepcopy(loads if loads else DEFAULT_LOADS)",
        "    available_generation = float(max(0.0, available_generation))",
        "    ",
        "    if predicted_demand is not None and predicted_demand > 0:",
        "        loads = scale_loads_to_demand(loads, predicted_demand)",
        "",
        "    total_required = round(sum(l['required_kw'] for l in loads), 1)",
        "    critical_loads = [l for l in loads if l['category'] == 'Critical']",
        "    important_loads = [l for l in loads if l['category'] == 'Important']",
        "    flexible_loads = [l for l in loads if l['category'] == 'Flexible']",
        "",
        "    req_critical = round(sum(l['required_kw'] for l in critical_loads), 1)",
        "    req_important = round(sum(l['required_kw'] for l in important_loads), 1)",
        "    req_flexible = round(sum(l['required_kw'] for l in flexible_loads), 1)",
        "",
        "    shifted_loads, managed_loads = [], []",
        "    remaining = available_generation",
        "",
        "    # 1. Critical Loads Priority",
        "    alloc_critical = 0.0",
        "    for load in critical_loads:",
        "        req = load['required_kw']",
        "        if remaining >= req:",
        "            load['allocated_kw'] = req",
        "            load['status'] = 'Active'",
        "            remaining -= req",
        "        else:",
        "            load['allocated_kw'] = round(max(0.0, remaining), 1)",
        "            load['status'] = 'Protected Partial' if remaining > 0 else 'Offline'",
        "            remaining = 0.0",
        "        alloc_critical += load['allocated_kw']",
        "",
        "    # 2. Important Loads Priority",
        "    alloc_important = 0.0",
        "    for load in important_loads:",
        "        req = load['required_kw']",
        "        if remaining >= req:",
        "            load['allocated_kw'] = req",
        "            load['status'] = 'Active'",
        "            remaining -= req",
        "        else:",
        "            load['allocated_kw'] = round(max(0.0, remaining), 1)",
        "            load['status'] = 'Reduced' if remaining > 0 else 'Offline'",
        "            remaining = 0.0",
        "        alloc_important += load['allocated_kw']",
        "",
        "    # 3. Flexible Loads Dynamic Throttling/Shifting",
        "    alloc_flexible = 0.0",
        "    if len(flexible_loads) > 0:",
        "        if remaining >= req_flexible:",
        "            for load in flexible_loads:",
        "                load['allocated_kw'] = load['required_kw']",
        "                load['status'] = 'Active'",
        "                remaining -= load['required_kw']",
        "                alloc_flexible += load['allocated_kw']",
        "        else:",
        "            flex_budget = max(0.0, remaining)",
        "            for load in flexible_loads:",
        "                req = load['required_kw']",
        "                name = load['name'].lower()",
        "                if ('ev' in name or 'hvac' in name or 'chiller' in name) and flex_budget > 0:",
        "                    portion = min(flex_budget, req * 0.50)",
        "                    load['allocated_kw'] = round(portion, 1)",
        "                    flex_budget -= load['allocated_kw']",
        "                    load['status'] = 'Managed (Eco-Throttled)' if load['allocated_kw'] > 0 else 'Shifted (Off-Peak)'",
        "                    (managed_loads if load['allocated_kw'] > 0 else shifted_loads).append(load['name'])",
        "                elif flex_budget >= req * 0.40:",
        "                    load['allocated_kw'] = round(req * 0.40, 1)",
        "                    flex_budget -= load['allocated_kw']",
        "                    load['status'] = 'Managed (Reduced)'",
        "                    managed_loads.append(load['name'])",
        "                else:",
        "                    load['allocated_kw'] = 0.0",
        "                    load['status'] = 'Shifted (Off-Peak)'",
        "                    shifted_loads.append(load['name'])",
        "                alloc_flexible += load['allocated_kw']",
        "",
        "    all_loads = critical_loads + important_loads + flexible_loads",
        "    total_allocated = round(alloc_critical + alloc_important + alloc_flexible, 1)",
        "    shortage = round(max(0.0, total_required - available_generation), 1)",
        "    surplus = round(max(0.0, available_generation - total_allocated), 1)",
        "",
        "    # Cost Analysis",
        "    off_peak_price = round(electricity_price_kwh * 0.55, 3)",
        "    peak_penalty = round(electricity_price_kwh * 1.50, 3)",
        "    unopt_cost = round(total_required * (peak_penalty if shortage > 0 else electricity_price_kwh), 2)",
        "    opt_cost = round((total_allocated * electricity_price_kwh) + ((total_required - total_allocated) * off_peak_price), 2)",
        "    saving = round(max(0.0, unopt_cost - opt_cost), 2)",
        "",
        "    return {",
        "        'critical': round(alloc_critical, 1),",
        "        'important': round(alloc_important, 1),",
        "        'flexible': round(alloc_flexible, 1),",
        "        'total_required': total_required,",
        "        'total_allocated': total_allocated,",
        "        'available_generation': available_generation,",
        "        'shortage': shortage,",
        "        'surplus': surplus,",
        "        'shifted_loads': shifted_loads,",
        "        'managed_loads': managed_loads,",
        "        'estimated_saving': saving,",
        "        'loads': all_loads",
        "    }"
    ]))

    # Step 6: Simulation Testing
    cells.append(make_cell("markdown", [
        "---",
        "## 6. Hackathon Scenario Simulations",
        "",
        "### Testing Demand Surge & Generation Drop Scenarios"
    ]))

    cells.append(make_cell("code", [
        "# Scenario 1: Normal Operation",
        "res_normal = allocate_energy(available_generation=100.0, predicted_demand=94.0)",
        "print('=== SCENARIO 1: NORMAL DEMAND ===')",
        "print(f'Required: {res_normal[\"total_required\"]} kW | Generation: {res_normal[\"available_generation\"]} kW | Surplus: {res_normal[\"surplus\"]} kW')",
        "",
        "# Scenario 2: Sudden Demand Surge (+25 kW Spike)",
        "res_surge = allocate_energy(available_generation=100.0, predicted_demand=125.0)",
        "print('\\n=== SCENARIO 2: DEMAND SURGE (125 kW) ===')",
        "print(f'Required: {res_surge[\"total_required\"]} kW | Generation: {res_surge[\"available_generation\"]} kW | Shortage: {res_surge[\"shortage\"]} kW')",
        "print(f'Critical: {res_surge[\"critical\"]} kW | Important: {res_surge[\"important\"]} kW | Flexible: {res_surge[\"flexible\"]} kW')",
        "print(f'Managed Loads: {res_surge[\"managed_loads\"]}')",
        "print(f'Estimated Cost Savings: ${res_surge[\"estimated_saving\"]}/hr')",
        "",
        "# Scenario 3: Sudden Generation Drop (70 kW Supply Cut)",
        "res_drop = allocate_energy(available_generation=70.0, predicted_demand=100.0)",
        "print('\\n=== SCENARIO 3: GENERATION DROP (70 kW) ===')",
        "print(f'Required: {res_drop[\"total_required\"]} kW | Generation: {res_drop[\"available_generation\"]} kW | Shortage: {res_drop[\"shortage\"]} kW')",
        "print(f'Critical: {res_drop[\"critical\"]} kW | Important: {res_drop[\"important\"]} kW | Flexible: {res_drop[\"flexible\"]} kW')",
        "print(f'Shifted Loads: {res_drop[\"shifted_loads\"]}')",
        "print(f'Estimated Cost Savings: ${res_drop[\"estimated_saving\"]}/hr')"
    ]))

    # Step 6.1: Visualization of Scenarios
    cells.append(make_cell("code", [
        "scenarios = ['Normal (94 kW)', 'Demand Surge (125 kW)', 'Generation Drop (70 kW)']",
        "crit_vals = [res_normal['critical'], res_surge['critical'], res_drop['critical']]",
        "imp_vals = [res_normal['important'], res_surge['important'], res_drop['important']]",
        "flex_vals = [res_normal['flexible'], res_surge['flexible'], res_drop['flexible']]",
        "",
        "x = np.arange(len(scenarios))",
        "width = 0.55",
        "",
        "plt.figure(figsize=(10, 5))",
        "p1 = plt.bar(x, crit_vals, width, label='Critical Loads', color='#ef4444')",
        "p2 = plt.bar(x, imp_vals, width, bottom=crit_vals, label='Important Loads', color='#f59e0b')",
        "p3 = plt.bar(x, flex_vals, width, bottom=np.array(crit_vals) + np.array(imp_vals), label='Flexible Loads', color='#38bdf8')",
        "",
        "plt.ylabel('Allocated Power (kW)')",
        "plt.title('Dynamic Energy Allocation across Hackathon Scenarios')",
        "plt.xticks(x, scenarios)",
        "plt.legend()",
        "plt.show()"
    ]))

    # Step 7: Flask App code
    cells.append(make_cell("markdown", [
        "---",
        "## 7. Flask Web Application & REST APIs (`app.py`)",
        "",
        "The complete Flask server exposing endpoints:",
        "- `POST /predict`",
        "- `POST /allocate`",
        "- `POST /simulate/demand-surge`",
        "- `POST /simulate/generation-drop`",
        "- `GET /status`, `GET /loads`, `POST /reset`"
    ]))

    # Read existing app.py code
    with open("app.py", "r", encoding="utf-8") as f:
        app_code = f.read()

    cells.append(make_cell("code", [app_code]))

    # Step 8: Frontend Template & Assets
    cells.append(make_cell("markdown", [
        "---",
        "## 8. Frontend Dashboard Code",
        "",
        "### 8.1 HTML Dashboard (`templates/index.html`)"
    ]))

    with open("templates/index.html", "r", encoding="utf-8") as f:
        html_code = f.read()

    cells.append(make_cell("code", [
        "# HTML Template Code (Stored in templates/index.html)",
        f'HTML_CONTENT = """{html_code}"""',
        "print('HTML template loaded. Characters:', len(HTML_CONTENT))"
    ]))

    cells.append(make_cell("markdown", [
        "### 8.2 Modern CSS Theme (`static/style.css`)"
    ]))

    with open("static/style.css", "r", encoding="utf-8") as f:
        css_code = f.read()

    cells.append(make_cell("code", [
        "# CSS Stylesheet (Stored in static/style.css)",
        f'CSS_CONTENT = """{css_code}"""',
        "print('CSS stylesheet loaded. Characters:', len(CSS_CONTENT))"
    ]))

    cells.append(make_cell("markdown", [
        "### 8.3 JavaScript & Chart.js Controller (`static/script.js`)"
    ]))

    with open("static/script.js", "r", encoding="utf-8") as f:
        js_code = f.read()

    cells.append(make_cell("code", [
        "# JavaScript Controller (Stored in static/script.js)",
        f'JS_CONTENT = """{js_code}"""',
        "print('JavaScript controller loaded. Characters:', len(JS_CONTENT))"
    ]))

    # Summary
    cells.append(make_cell("markdown", [
        "---",
        "## 9. Summary & Deployment",
        "",
        "### How to Run this Project:",
        "1. **Local Server**:",
        "   ```bash",
        "   python app.py",
        "   ```",
        "   Navigate to `http://127.0.0.1:5000`",
        "",
        "2. **Live Public URL via Cloudflare Tunnel**:",
        "   ```bash",
        "   cloudflared tunnel --url http://127.0.0.1:5000",
        "   ```",
        "",
        "3. **GitHub Repository**:",
        "   [https://github.com/257y1a6641-glitch/Electricity-Intelligence-AI.1](https://github.com/257y1a6641-glitch/Electricity-Intelligence-AI.1)"
    ]))

    notebook_data = {
        "cells": cells,
        "metadata": {
            "language_info": {
                "name": "python",
                "version": "3.11"
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    out_file = "c:/Users/Afrin/OneDrive/Desktop/AI/AI_Smart_Energy_Management.ipynb"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(notebook_data, f, indent=2)

    sub_file = "c:/Users/Afrin/OneDrive/Desktop/AI/AI-Energy-Demand-Forecasting/AI_Smart_Energy_Management.ipynb"
    with open(sub_file, "w", encoding="utf-8") as f:
        json.dump(notebook_data, f, indent=2)

    print(f"Jupyter Notebook successfully written to:\n - {out_file}\n - {sub_file}")

if __name__ == "__main__":
    build_notebook()
