"""
AI Smart Energy Management System - Flask Backend & REST APIs.
Coordinates AI Demand Forecasting, Priority-Based Dynamic Energy Allocation,
Demand Surge and Generation Drop simulations, and real-time dashboard telemetry.
"""

import os
from flask import Flask, render_template, request, jsonify
from demand_forecasting import forecaster
from energy_allocation import allocate_energy, get_default_loads

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)

# Preload or train ML model on startup
print("Initializing Demand Forecasting Model...")
forecaster.load_or_train()
print("Demand Forecasting Model ready.")

# Current in-memory operational state (can be reset)
system_state = {
    "current_demand": 80.0,
    "available_generation": 100.0,
    "temperature": 32.0,
    "humidity": 60.0,
    "occupancy": 75.0,
    "hour": 14,
    "day_type": "Weekday",
    "user_activity": "Medium",
    "last_prediction": 94.0,
    "last_allocation": None,
    "loads": get_default_loads(),
    "last_event": "System initialized normally."
}

# Run initial allocation so the dashboard loads with complete data
system_state["last_allocation"] = allocate_energy(
    available_generation=system_state["available_generation"],
    predicted_demand=system_state["last_prediction"],
    loads=system_state["loads"]
)


@app.route("/")
def index():
    """Serves the main interactive dashboard."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    Receives current operating parameters, invokes the AI/ML model,
    and returns predicted demand, comparison with generation, and historical trend data.
    """
    try:
        data = request.get_json(force=True) or {}
        
        # Update system state with incoming parameters
        system_state["current_demand"] = float(data.get("current_demand", system_state["current_demand"]))
        system_state["available_generation"] = float(data.get("available_generation", system_state["available_generation"]))
        system_state["temperature"] = float(data.get("temperature", system_state["temperature"]))
        system_state["humidity"] = float(data.get("humidity", system_state["humidity"]))
        system_state["occupancy"] = float(data.get("occupancy", system_state["occupancy"]))
        system_state["hour"] = int(data.get("hour", system_state["hour"]))
        system_state["day_type"] = str(data.get("day_type", system_state["day_type"]))
        system_state["user_activity"] = str(data.get("user_activity", system_state["user_activity"]))

        # Execute ML prediction
        prediction_result = forecaster.predict({
            "current_demand": system_state["current_demand"],
            "temperature": system_state["temperature"],
            "humidity": system_state["humidity"],
            "occupancy": system_state["occupancy"],
            "hour": system_state["hour"],
            "day_type": system_state["day_type"],
            "user_activity": system_state["user_activity"]
        })

        pred_val = prediction_result["predicted_demand"]
        system_state["last_prediction"] = pred_val
        
        avail = system_state["available_generation"]
        shortage = round(max(0.0, pred_val - avail), 1)
        surplus = round(max(0.0, avail - pred_val), 1)

        # Status categorization
        if shortage > 0:
            status_text = "Deficit Expected"
            alert_type = "warning"
            alert_msg = f"Predicted demand ({pred_val} kW) exceeds available generation ({avail} kW). Shortage: {shortage} kW."
        else:
            status_text = "Balanced"
            alert_type = "success"
            alert_msg = f"Available generation ({avail} kW) covers predicted demand ({pred_val} kW). Surplus: {surplus} kW."

        response_payload = {
            "current_demand": system_state["current_demand"],
            "predicted_demand": pred_val,
            "available_generation": avail,
            "shortage": shortage,
            "surplus": surplus,
            "status": status_text,
            "alert_type": alert_type,
            "alert_message": alert_msg,
            "historical_trend": prediction_result["historical_trend"],
            "timeline_labels": prediction_result["timeline_labels"],
            "metrics": prediction_result["model_metrics"],
            "input_echo": prediction_result["input_echo"]
        }
        return jsonify(response_payload), 200

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 400


@app.route("/allocate", methods=["POST"])
def allocate():
    """
    POST /allocate
    Allocates available generation across load priorities (Critical -> Important -> Flexible).
    """
    try:
        data = request.get_json(force=True) or {}
        
        # Allow caller to override or use system state
        pred_demand = float(data.get("predicted_demand", system_state["last_prediction"]))
        avail_gen = float(data.get("available_generation", system_state["available_generation"]))
        loads = data.get("loads", system_state["loads"])

        allocation_result = allocate_energy(
            available_generation=avail_gen,
            predicted_demand=pred_demand,
            loads=loads
        )

        # Update cache
        system_state["last_allocation"] = allocation_result
        system_state["loads"] = allocation_result["loads"]
        system_state["available_generation"] = avail_gen
        system_state["last_prediction"] = pred_demand

        return jsonify(allocation_result), 200

    except Exception as e:
        return jsonify({"error": f"Allocation calculation failed: {str(e)}"}), 400


@app.route("/simulate/demand-surge", methods=["POST"])
def simulate_demand_surge():
    """
    POST /simulate/demand-surge
    Simulates a sudden demand spike (e.g. from 80 kW to 125 kW).
    Detects shortage, recalculates allocation, protects critical/important loads,
    and shifts flexible loads.
    """
    try:
        data = request.get_json(silent=True) or {}
        
        # Surge parameters
        surge_demand = float(data.get("surge_demand", 125.0))
        avail_gen = float(data.get("available_generation", 100.0))

        system_state["current_demand"] = surge_demand
        system_state["available_generation"] = avail_gen
        system_state["user_activity"] = "High"

        # Re-run ML prediction with surge inputs
        prediction_result = forecaster.predict({
            "current_demand": surge_demand,
            "temperature": max(34.0, system_state["temperature"]),
            "humidity": system_state["humidity"],
            "occupancy": 90.0,
            "hour": system_state["hour"],
            "day_type": system_state["day_type"],
            "user_activity": "High"
        })

        pred_demand = prediction_result["predicted_demand"]
        system_state["last_prediction"] = pred_demand

        # Run dynamic allocation
        allocation_result = allocate_energy(
            available_generation=avail_gen,
            predicted_demand=pred_demand,
            loads=get_default_loads()
        )

        shortage = round(max(0.0, pred_demand - avail_gen), 1)
        alert_msg = "Demand Surge Detected — Available energy is insufficient for current demand. Flexible loads are being managed."

        system_state["last_allocation"] = allocation_result
        system_state["loads"] = allocation_result["loads"]
        system_state["last_event"] = alert_msg

        return jsonify({
            "simulation": "Demand Surge",
            "current_demand": surge_demand,
            "predicted_demand": pred_demand,
            "available_generation": avail_gen,
            "shortage": shortage,
            "surplus": 0.0,
            "allocation": allocation_result,
            "historical_trend": prediction_result["historical_trend"],
            "timeline_labels": prediction_result["timeline_labels"],
            "alert_type": "danger",
            "alert_title": "Demand Surge Detected",
            "alert_message": alert_msg
        }), 200

    except Exception as e:
        return jsonify({"error": f"Demand surge simulation failed: {str(e)}"}), 500


@app.route("/simulate/generation-drop", methods=["POST"])
def simulate_generation_drop():
    """
    POST /simulate/generation-drop
    Simulates a sudden generation drop (e.g. from 100 kW to 70 kW due to cloud cover or trip).
    Detects generation deficit, recalculates allocation, protects priority loads,
    and manages flexible loads.
    """
    try:
        data = request.get_json(silent=True) or {}

        dropped_gen = float(data.get("dropped_generation", 70.0))
        current_demand = float(data.get("current_demand", 80.0))
        target_demand = float(data.get("predicted_demand", 100.0))

        system_state["available_generation"] = dropped_gen
        system_state["current_demand"] = current_demand
        system_state["last_prediction"] = target_demand

        # Run dynamic allocation with reduced generation
        allocation_result = allocate_energy(
            available_generation=dropped_gen,
            predicted_demand=target_demand,
            loads=get_default_loads()
        )

        shortage = round(max(0.0, target_demand - dropped_gen), 1)
        alert_msg = f"Generation Drop Detected — Available Energy: {dropped_gen} kW | Required Energy: {target_demand} kW | Shortage: {shortage} kW. Flexible loads throttled."

        system_state["last_allocation"] = allocation_result
        system_state["loads"] = allocation_result["loads"]
        system_state["last_event"] = alert_msg

        # Trend for visualization
        pred_res = forecaster.predict({
            "current_demand": current_demand,
            "temperature": system_state["temperature"],
            "humidity": system_state["humidity"],
            "occupancy": system_state["occupancy"],
            "hour": system_state["hour"],
            "day_type": system_state["day_type"],
            "user_activity": system_state["user_activity"]
        })

        return jsonify({
            "simulation": "Generation Drop",
            "current_demand": current_demand,
            "predicted_demand": target_demand,
            "available_generation": dropped_gen,
            "shortage": shortage,
            "surplus": 0.0,
            "allocation": allocation_result,
            "historical_trend": pred_res["historical_trend"],
            "timeline_labels": pred_res["timeline_labels"],
            "alert_type": "warning",
            "alert_title": "Generation Drop Detected",
            "alert_message": alert_msg
        }), 200

    except Exception as e:
        return jsonify({"error": f"Generation drop simulation failed: {str(e)}"}), 500


@app.route("/status", methods=["GET"])
def get_status():
    """
    GET /status
    Returns current telemetry, operational state, load allocation, and model health.
    """
    try:
        return jsonify({
            "system_state": {
                "current_demand": system_state["current_demand"],
                "available_generation": system_state["available_generation"],
                "temperature": system_state["temperature"],
                "humidity": system_state["humidity"],
                "occupancy": system_state["occupancy"],
                "hour": system_state["hour"],
                "day_type": system_state["day_type"],
                "user_activity": system_state["user_activity"],
                "last_prediction": system_state["last_prediction"],
                "last_event": system_state["last_event"]
            },
            "last_allocation": system_state["last_allocation"],
            "model_metrics": forecaster.metrics
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/loads", methods=["GET"])
def get_loads():
    """
    GET /loads
    Returns the initial default categorized loads.
    """
    return jsonify({
        "loads": get_default_loads()
    }), 200


@app.route("/reset", methods=["POST"])
def reset():
    """
    POST /reset
    Restores baseline default inputs and allocation.
    """
    system_state["current_demand"] = 80.0
    system_state["available_generation"] = 100.0
    system_state["temperature"] = 32.0
    system_state["humidity"] = 60.0
    system_state["occupancy"] = 75.0
    system_state["hour"] = 14
    system_state["day_type"] = "Weekday"
    system_state["user_activity"] = "Medium"
    system_state["last_prediction"] = 94.0
    system_state["loads"] = get_default_loads()
    system_state["last_event"] = "System reset to baseline operation."

    alloc = allocate_energy(
        available_generation=100.0,
        predicted_demand=94.0,
        loads=system_state["loads"]
    )
    system_state["last_allocation"] = alloc

    pred_res = forecaster.predict({
        "current_demand": 80.0,
        "temperature": 32.0,
        "humidity": 60.0,
        "occupancy": 75.0,
        "hour": 14,
        "day_type": 0,
        "user_activity": 1
    })

    return jsonify({
        "message": "System reset to baseline successfully.",
        "system_state": system_state,
        "allocation": alloc,
        "historical_trend": pred_res["historical_trend"],
        "timeline_labels": pred_res["timeline_labels"]
    }), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting AI Smart Energy Management System on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=True)
