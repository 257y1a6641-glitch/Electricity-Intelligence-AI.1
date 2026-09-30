"""
AI Smart Energy Management System — Flask REST API Backend.

PROBLEM STATEMENT ALIGNMENT
----------------------------
Electricity demand varies significantly with time, weather, occupancy, and user
behavior, making it difficult to balance demand with limited generation capacity.
This backend coordinates:
  1. AI/ML Short-Term Demand Forecasting (RandomForestRegressor)
  2. Priority-Based Dynamic Energy Allocation (Critical > Important > Flexible)
  3. Demand Surge & Generation Drop Detection & Response
  4. Real-Time Cost Estimation & Savings Calculation
  5. REST API for interactive dashboard telemetry

UN SDG Alignment:
  - SDG 7: Affordable and Clean Energy
  - SDG 9: Industry, Innovation and Infrastructure
  - SDG 13: Climate Action (reduced energy waste)

NOTE: This prototype uses a synthetic dataset for hackathon demonstration purposes.
Predictions are illustrative and not intended for real-world grid operations.

Author: AI Smart Energy Team
Version: 2.0.0
"""

import os
import time
import logging
from flask import Flask, render_template, request, jsonify, g
from flask_compress import Compress
from demand_forecasting import forecaster
from energy_allocation import allocate_energy, get_default_loads

# ─── Logging Setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ─── App Initialization ───────────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)
app.config["JSON_SORT_KEYS"]      = False
app.config["COMPRESS_MIMETYPES"]  = [
    "application/json", "text/html", "text/css",
    "text/javascript", "application/javascript"
]
app.config["COMPRESS_LEVEL"]      = 6      # gzip level 6 — good balance speed/size
app.config["COMPRESS_MIN_SIZE"]   = 500    # compress responses ≥ 500 bytes
app.config["COMPRESS_ALGORITHM"]  = ["br", "gzip", "deflate"]  # prefer Brotli
Compress(app)                              # attach gzip/brotli to all responses

# ─── In-Memory Cache ──────────────────────────────────────────────────────────
_prediction_cache: dict = {}
CACHE_TTL_SECONDS = 30  # Cache predictions for 30 seconds to reduce redundant ML calls

def _cache_key(data: dict) -> str:
    """Generate a deterministic cache key from request features."""
    keys = ["current_demand", "temperature", "humidity", "occupancy",
            "hour", "day_type", "user_activity"]
    return "|".join(str(round(float(data.get(k, 0)), 1)) if k not in ("day_type", "user_activity")
                    else str(data.get(k, "")) for k in keys)

# ─── Input Validation ─────────────────────────────────────────────────────────
VALID_DAY_TYPES = {"Weekday", "Weekend"}
VALID_ACTIVITIES = {"Low", "Medium", "High"}

def validate_inputs(data: dict) -> tuple[bool, str]:
    """
    Validates incoming request payload for type safety and business-rule bounds.

    Args:
        data: Parsed JSON request body.

    Returns:
        (is_valid: bool, error_message: str)
    """
    try:
        demand = float(data.get("current_demand", 80))
        if not (0 < demand <= 10000):
            return False, "current_demand must be between 0 and 10000 kW"

        generation = float(data.get("available_generation", 100))
        if not (0 < generation <= 10000):
            return False, "available_generation must be between 0 and 10000 kW"

        temp = float(data.get("temperature", 30))
        if not (-50 <= temp <= 60):
            return False, "temperature must be between -50 and 60 °C"

        humidity = float(data.get("humidity", 60))
        if not (0 <= humidity <= 100):
            return False, "humidity must be between 0 and 100 %"

        occupancy = float(data.get("occupancy", 75))
        if not (0 <= occupancy <= 100):
            return False, "occupancy must be between 0 and 100 %"

        hour = int(data.get("hour", 14))
        if not (0 <= hour <= 23):
            return False, "hour must be between 0 and 23"

        day_type = str(data.get("day_type", "Weekday"))
        if day_type not in VALID_DAY_TYPES:
            return False, f"day_type must be one of: {VALID_DAY_TYPES}"

        activity = str(data.get("user_activity", "Medium"))
        if activity not in VALID_ACTIVITIES:
            return False, f"user_activity must be one of: {VALID_ACTIVITIES}"

    except (ValueError, TypeError) as e:
        return False, f"Type error in input: {str(e)}"

    return True, ""


# ─── Timing Middleware ────────────────────────────────────────────────────────
@app.before_request
def start_timer():
    """Record request start time for latency tracking."""
    g.start_time = time.perf_counter()


@app.after_request
def add_latency_header(response):
    """Attach X-Response-Time header in milliseconds to every response."""
    if hasattr(g, "start_time"):
        elapsed_ms = round((time.perf_counter() - g.start_time) * 1000, 2)
        response.headers["X-Response-Time"] = f"{elapsed_ms}ms"
        response.headers["X-API-Version"] = "2.0.0"
    response.headers["Cache-Control"] = "no-store"
    return response


# ─── Model Warmup ─────────────────────────────────────────────────────────────
logger.info("Initializing Demand Forecasting Model...")
forecaster.load_or_train()
logger.info("Demand Forecasting Model ready. Metrics: %s", forecaster.metrics)

# ─── Operational State ────────────────────────────────────────────────────────
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
    "last_event": "System initialized normally.",
    "request_count": 0,
    "version": "2.0.0"
}

# Run initial allocation so the dashboard loads with complete data
system_state["last_allocation"] = allocate_energy(
    available_generation=system_state["available_generation"],
    predicted_demand=system_state["last_prediction"],
    loads=system_state["loads"]
)


# ─── Routes ──────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    """Serves the main AI Smart Energy Management System dashboard."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict — AI Demand Forecasting Endpoint.

    Receives current operating parameters, checks the prediction cache,
    invokes the RandomForestRegressor ML model, and returns:
      - predicted_demand (kW)
      - shortage / surplus vs available generation
      - historical trend for chart visualization
      - model performance metrics (MAE, R²)
      - cache hit status and latency

    Request Body (JSON):
        current_demand      (float, kW)   : Current observed demand
        available_generation(float, kW)   : Current available power
        temperature         (float, °C)   : Ambient temperature
        humidity            (float, %)    : Relative humidity
        occupancy           (float, %)    : Building/zone occupancy
        hour                (int, 0-23)   : Current hour
        day_type            (str)         : "Weekday" | "Weekend"
        user_activity       (str)         : "Low" | "Medium" | "High"

    Returns:
        200: Prediction result JSON
        400: Validation error or prediction failure
    """
    try:
        t0 = time.perf_counter()
        data = request.get_json(force=True) or {}
        system_state["request_count"] += 1

        # Input validation
        is_valid, err_msg = validate_inputs(data)
        if not is_valid:
            return jsonify({"error": err_msg, "field_hint": err_msg}), 400

        # Update system state
        system_state["current_demand"]       = float(data.get("current_demand", system_state["current_demand"]))
        system_state["available_generation"] = float(data.get("available_generation", system_state["available_generation"]))
        system_state["temperature"]          = float(data.get("temperature", system_state["temperature"]))
        system_state["humidity"]             = float(data.get("humidity", system_state["humidity"]))
        system_state["occupancy"]            = float(data.get("occupancy", system_state["occupancy"]))
        system_state["hour"]                 = int(data.get("hour", system_state["hour"]))
        system_state["day_type"]             = str(data.get("day_type", system_state["day_type"]))
        system_state["user_activity"]        = str(data.get("user_activity", system_state["user_activity"]))

        # Cache lookup
        cache_key = _cache_key(data)
        now = time.time()
        cached = _prediction_cache.get(cache_key)
        cache_hit = False
        if cached and (now - cached["ts"]) < CACHE_TTL_SECONDS:
            prediction_result = cached["result"]
            cache_hit = True
            logger.info("Prediction cache hit for key=%s", cache_key[:20])
        else:
            prediction_result = forecaster.predict({
                "current_demand": system_state["current_demand"],
                "temperature":    system_state["temperature"],
                "humidity":       system_state["humidity"],
                "occupancy":      system_state["occupancy"],
                "hour":           system_state["hour"],
                "day_type":       system_state["day_type"],
                "user_activity":  system_state["user_activity"]
            })
            _prediction_cache[cache_key] = {"result": prediction_result, "ts": now}

        pred_val = prediction_result["predicted_demand"]
        system_state["last_prediction"] = pred_val

        avail    = system_state["available_generation"]
        shortage = round(max(0.0, pred_val - avail), 1)
        surplus  = round(max(0.0, avail - pred_val), 1)

        if shortage > 0:
            status_text = "Deficit Expected"
            alert_type  = "warning"
            alert_msg   = (f"Predicted demand ({pred_val} kW) exceeds available generation "
                           f"({avail} kW). Shortage: {shortage} kW. Dynamic reallocation recommended.")
        else:
            status_text = "Balanced"
            alert_type  = "success"
            alert_msg   = (f"Available generation ({avail} kW) covers predicted demand "
                           f"({pred_val} kW). Surplus: {surplus} kW.")

        inference_ms = round((time.perf_counter() - t0) * 1000, 2)
        logger.info("Prediction complete: %.1f kW in %.2f ms (cache=%s)", pred_val, inference_ms, cache_hit)

        return jsonify({
            "current_demand":      system_state["current_demand"],
            "predicted_demand":    pred_val,
            "available_generation": avail,
            "shortage":            shortage,
            "surplus":             surplus,
            "status":              status_text,
            "alert_type":          alert_type,
            "alert_message":       alert_msg,
            "historical_trend":    prediction_result["historical_trend"],
            "timeline_labels":     prediction_result["timeline_labels"],
            "metrics":             prediction_result["model_metrics"],
            "input_echo":          prediction_result["input_echo"],
            "performance": {
                "inference_ms": inference_ms,
                "cache_hit":    cache_hit,
                "cache_ttl_s":  CACHE_TTL_SECONDS
            }
        }), 200

    except Exception as e:
        logger.error("Prediction failed: %s", str(e), exc_info=True)
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 400


@app.route("/allocate", methods=["POST"])
def allocate():
    """
    POST /allocate — Dynamic Energy Allocation Endpoint.

    Implements the core allocation algorithm:
      1. Critical loads receive full allocation unconditionally.
      2. Important loads receive energy after critical loads are satisfied.
      3. Flexible loads are eco-throttled or shifted to off-peak hours.
      4. Total allocated power never exceeds available_generation.

    Request Body (JSON):
        predicted_demand    (float, kW)   : ML-predicted demand
        available_generation(float, kW)   : Available power supply
        loads               (list, opt)   : Override load list

    Returns:
        200: Allocation result with cost analysis
        400: Validation or computation error
    """
    try:
        t0 = time.perf_counter()
        data = request.get_json(force=True) or {}

        pred_demand = float(data.get("predicted_demand", system_state["last_prediction"]))
        avail_gen   = float(data.get("available_generation", system_state["available_generation"]))
        loads       = data.get("loads", system_state["loads"])

        # Guard against nonsensical inputs
        if pred_demand <= 0 or avail_gen <= 0:
            return jsonify({"error": "predicted_demand and available_generation must be > 0"}), 400

        allocation_result = allocate_energy(
            available_generation=avail_gen,
            predicted_demand=pred_demand,
            loads=loads
        )

        system_state["last_allocation"]      = allocation_result
        system_state["loads"]                = allocation_result["loads"]
        system_state["available_generation"] = avail_gen
        system_state["last_prediction"]      = pred_demand

        allocation_result["performance"] = {"allocation_ms": round((time.perf_counter() - t0) * 1000, 2)}
        logger.info("Allocation done: total_allocated=%.1f / avail=%.1f kW", allocation_result["total_allocated"], avail_gen)
        return jsonify(allocation_result), 200

    except Exception as e:
        logger.error("Allocation failed: %s", str(e), exc_info=True)
        return jsonify({"error": f"Allocation calculation failed: {str(e)}"}), 400


@app.route("/simulate/demand-surge", methods=["POST"])
def simulate_demand_surge():
    """
    POST /simulate/demand-surge — Sudden Demand Spike Simulation.

    Models a real-world scenario where demand suddenly spikes (e.g., industrial
    load connect, heat wave peak). System detects the surge, recalculates the
    allocation, protects critical/important loads, and manages flexible loads.

    Default surge: 80 kW → 125 kW (demand), generation held at 100 kW.

    Returns:
        200: Simulation result with shortage, allocation, and alert details
        500: Unexpected runtime error
    """
    try:
        data        = request.get_json(silent=True) or {}
        surge_demand = float(data.get("surge_demand", 125.0))
        avail_gen    = float(data.get("available_generation", 100.0))

        # Clamp to safe ranges
        surge_demand = max(0.0, min(surge_demand, 10000.0))
        avail_gen    = max(0.0, min(avail_gen, 10000.0))

        system_state["current_demand"]       = surge_demand
        system_state["available_generation"] = avail_gen
        system_state["user_activity"]        = "High"

        prediction_result = forecaster.predict({
            "current_demand": surge_demand,
            "temperature":    max(34.0, system_state["temperature"]),
            "humidity":       system_state["humidity"],
            "occupancy":      90.0,
            "hour":           system_state["hour"],
            "day_type":       system_state["day_type"],
            "user_activity":  "High"
        })

        pred_demand = prediction_result["predicted_demand"]
        system_state["last_prediction"] = pred_demand

        allocation_result = allocate_energy(
            available_generation=avail_gen,
            predicted_demand=pred_demand,
            loads=get_default_loads()
        )

        shortage  = round(max(0.0, pred_demand - avail_gen), 1)
        alert_msg = ("Demand Surge Detected — Available energy is insufficient for current demand. "
                     "Flexible loads are being managed. Critical and Important loads protected.")

        system_state["last_allocation"] = allocation_result
        system_state["loads"]           = allocation_result["loads"]
        system_state["last_event"]      = alert_msg

        logger.warning("Demand Surge: %.1f kW demand vs %.1f kW generation. Shortage=%.1f kW", surge_demand, avail_gen, shortage)
        return jsonify({
            "simulation":          "Demand Surge",
            "current_demand":      surge_demand,
            "predicted_demand":    pred_demand,
            "available_generation": avail_gen,
            "shortage":            shortage,
            "surplus":             0.0,
            "allocation":          allocation_result,
            "historical_trend":    prediction_result["historical_trend"],
            "timeline_labels":     prediction_result["timeline_labels"],
            "alert_type":          "danger",
            "alert_title":         "Demand Surge Detected",
            "alert_message":       alert_msg
        }), 200

    except Exception as e:
        logger.error("Demand surge simulation failed: %s", str(e), exc_info=True)
        return jsonify({"error": f"Demand surge simulation failed: {str(e)}"}), 500


@app.route("/simulate/generation-drop", methods=["POST"])
def simulate_generation_drop():
    """
    POST /simulate/generation-drop — Sudden Generation Capacity Drop.

    Models a real-world scenario where available generation suddenly falls
    (e.g., cloud cover on solar panels, grid trip, generator fault).
    System detects the deficit, recalculates allocation, protects priority
    loads, and defers flexible loads.

    Default: generation drops from 100 kW → 70 kW.

    Returns:
        200: Simulation result with shortage, allocation breakdown, and alert
        500: Unexpected runtime error
    """
    try:
        data          = request.get_json(silent=True) or {}
        dropped_gen   = float(data.get("dropped_generation", 70.0))
        current_demand = float(data.get("current_demand", 80.0))
        target_demand  = float(data.get("predicted_demand", 100.0))

        # Clamp inputs
        dropped_gen    = max(0.0, min(dropped_gen, 10000.0))
        current_demand = max(0.0, min(current_demand, 10000.0))
        target_demand  = max(0.0, min(target_demand, 10000.0))

        system_state["available_generation"] = dropped_gen
        system_state["current_demand"]       = current_demand
        system_state["last_prediction"]      = target_demand

        allocation_result = allocate_energy(
            available_generation=dropped_gen,
            predicted_demand=target_demand,
            loads=get_default_loads()
        )

        shortage  = round(max(0.0, target_demand - dropped_gen), 1)
        alert_msg = (f"Generation Drop Detected — Available Energy: {dropped_gen} kW | "
                     f"Required Energy: {target_demand} kW | Shortage: {shortage} kW. "
                     "Flexible loads throttled or deferred.")

        system_state["last_allocation"] = allocation_result
        system_state["loads"]           = allocation_result["loads"]
        system_state["last_event"]      = alert_msg

        pred_res = forecaster.predict({
            "current_demand": current_demand,
            "temperature":    system_state["temperature"],
            "humidity":       system_state["humidity"],
            "occupancy":      system_state["occupancy"],
            "hour":           system_state["hour"],
            "day_type":       system_state["day_type"],
            "user_activity":  system_state["user_activity"]
        })

        logger.warning("Generation Drop: avail=%.1f kW, demand=%.1f kW, shortage=%.1f kW", dropped_gen, target_demand, shortage)
        return jsonify({
            "simulation":          "Generation Drop",
            "current_demand":      current_demand,
            "predicted_demand":    target_demand,
            "available_generation": dropped_gen,
            "shortage":            shortage,
            "surplus":             0.0,
            "allocation":          allocation_result,
            "historical_trend":    pred_res["historical_trend"],
            "timeline_labels":     pred_res["timeline_labels"],
            "alert_type":          "warning",
            "alert_title":         "Generation Drop Detected",
            "alert_message":       alert_msg
        }), 200

    except Exception as e:
        logger.error("Generation drop simulation failed: %s", str(e), exc_info=True)
        return jsonify({"error": f"Generation drop simulation failed: {str(e)}"}), 500


@app.route("/status", methods=["GET"])
def get_status():
    """
    GET /status — System Health & Telemetry.

    Returns current operational state, active load allocation, ML model
    health metrics, and cumulative request count.

    Returns:
        200: Full system telemetry JSON
        500: Internal error
    """
    try:
        return jsonify({
            "system_state": {
                "current_demand":      system_state["current_demand"],
                "available_generation": system_state["available_generation"],
                "temperature":         system_state["temperature"],
                "humidity":            system_state["humidity"],
                "occupancy":           system_state["occupancy"],
                "hour":                system_state["hour"],
                "day_type":            system_state["day_type"],
                "user_activity":       system_state["user_activity"],
                "last_prediction":     system_state["last_prediction"],
                "last_event":          system_state["last_event"],
                "request_count":       system_state["request_count"],
                "version":             system_state["version"]
            },
            "last_allocation": system_state["last_allocation"],
            "model_metrics":   forecaster.metrics,
            "cache_entries":   len(_prediction_cache),
            "health":          "ok"
        }), 200
    except Exception as e:
        logger.error("Status endpoint failed: %s", str(e), exc_info=True)
        return jsonify({"error": str(e), "health": "degraded"}), 500


@app.route("/loads", methods=["GET"])
def get_loads():
    """
    GET /loads — Default Load Configuration.

    Returns the full categorized load list (Critical, Important, Flexible)
    with required power, priority, and flexibility metadata.

    Returns:
        200: Load list with category summary
    """
    loads = get_default_loads()
    return jsonify({
        "loads": loads,
        "summary": {
            "critical_count":  sum(1 for l in loads if l["category"] == "Critical"),
            "important_count": sum(1 for l in loads if l["category"] == "Important"),
            "flexible_count":  sum(1 for l in loads if l["category"] == "Flexible"),
            "total_kw":        round(sum(l["required_kw"] for l in loads), 1)
        }
    }), 200


@app.route("/reset", methods=["POST"])
def reset():
    """
    POST /reset — Restore Baseline State.

    Resets all system parameters to default demonstration values and
    clears the prediction cache. Useful after simulations.

    Returns:
        200: Baseline state and initial allocation
    """
    global _prediction_cache
    _prediction_cache = {}  # Clear cache on reset

    system_state.update({
        "current_demand":      80.0,
        "available_generation": 100.0,
        "temperature":         32.0,
        "humidity":            60.0,
        "occupancy":           75.0,
        "hour":                14,
        "day_type":            "Weekday",
        "user_activity":       "Medium",
        "last_prediction":     94.0,
        "loads":               get_default_loads(),
        "last_event":          "System reset to baseline operation.",
        "request_count":       0
    })

    alloc = allocate_energy(
        available_generation=100.0,
        predicted_demand=94.0,
        loads=system_state["loads"]
    )
    system_state["last_allocation"] = alloc

    pred_res = forecaster.predict({
        "current_demand":  80.0,
        "temperature":     32.0,
        "humidity":        60.0,
        "occupancy":       75.0,
        "hour":            14,
        "day_type":        "Weekday",
        "user_activity":   "Medium"
    })

    logger.info("System reset to baseline.")
    return jsonify({
        "message":          "System reset to baseline successfully.",
        "system_state":     {k: v for k, v in system_state.items() if k not in ("loads", "last_allocation")},
        "allocation":       alloc,
        "historical_trend": pred_res["historical_trend"],
        "timeline_labels":  pred_res["timeline_labels"]
    }), 200


@app.route("/health", methods=["GET"])
def health_check():
    """
    GET /health — Lightweight liveness probe.

    Returns:
        200: { "status": "ok", "model_loaded": bool }
    """
    return jsonify({
        "status":       "ok",
        "model_loaded": forecaster.is_loaded,
        "version":      "2.0.0"
    }), 200


@app.errorhandler(404)
def not_found(e):
    """Handle unknown routes gracefully."""
    return jsonify({"error": "Endpoint not found", "available_endpoints": [
        "GET /", "POST /predict", "POST /allocate",
        "POST /simulate/demand-surge", "POST /simulate/generation-drop",
        "GET /status", "GET /loads", "POST /reset", "GET /health"
    ]}), 404


@app.errorhandler(405)
def method_not_allowed(e):
    """Handle wrong HTTP methods with a clear message."""
    return jsonify({"error": "Method not allowed for this endpoint"}), 405


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info("Starting AI Smart Energy Management System v2.0.0 on port %d...", port)
    app.run(host="0.0.0.0", port=port, debug=False)
