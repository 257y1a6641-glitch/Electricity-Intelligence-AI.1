"""
Dynamic Energy Allocation Engine — AI Smart Energy Management System v2.0.
===========================================================================

PROBLEM ALIGNMENT
-----------------
Solves the core challenge: when demand exceeds generation capacity, the system
MUST NOT simply cut all power. Instead it uses a 3-tier priority algorithm:
  1. Critical loads (medical, emergency, servers) → always 100% served
  2. Important loads (lighting, computers, HVAC) → served after critical
  3. Flexible loads (EV, water heating, washers) → eco-throttled or deferred

ALGORITHM COMPLEXITY
--------------------
  - Time:  O(n) — single linear pass over each load tier
  - Space: O(n) — deep copy of load list; no auxiliary data structures
  - The scale_loads_to_demand() correction step prevents floating-point drift

COST MODEL
----------
  Unoptimised cost = total_required × peak_penalty_rate  (overload / diesel backup)
  Optimised cost   = allocated × base_rate + deferred × off_peak_rate
  Saving           = max(0, unoptimised − optimised)

UN SDG Alignment:
  SDG 7  — Affordable and Clean Energy
  SDG 9  — Industry, Innovation and Infrastructure
  SDG 11 — Sustainable Cities and Communities

NOTE: Uses synthetic demonstration data. Not for real-world grid operations.
"""

from __future__ import annotations
from typing import List, Dict, Any
from functools import lru_cache
import copy
import json

# ── Default Load Registry ─────────────────────────────────────────────────────

_DEFAULT_LOADS_TUPLE = (
    # ── Critical (Priority 1): Must NEVER be dropped ──────────────────────────
    {
        "id": "load_1", "name": "Medical Equipment",
        "category": "Critical", "priority": 1,
        "is_flexible": False, "required_kw": 20.0,
        "allocated_kw": 20.0, "status": "Active",
        "description": "Life-support and ICU equipment — zero-interruption required"
    },
    {
        "id": "load_2", "name": "Emergency Systems",
        "category": "Critical", "priority": 1,
        "is_flexible": False, "required_kw": 10.0,
        "allocated_kw": 10.0, "status": "Active",
        "description": "Fire alarms, emergency lighting, safety systems"
    },
    {
        "id": "load_3", "name": "Essential Servers",
        "category": "Critical", "priority": 1,
        "is_flexible": False, "required_kw": 15.0,
        "allocated_kw": 15.0, "status": "Active",
        "description": "Core IT infrastructure — data integrity risk if dropped"
    },
    # ── Important (Priority 2): Necessary for general operations ──────────────
    {
        "id": "load_4", "name": "Lighting",
        "category": "Important", "priority": 2,
        "is_flexible": False, "required_kw": 15.0,
        "allocated_kw": 15.0, "status": "Active",
        "description": "Zone lighting for occupied spaces"
    },
    {
        "id": "load_5", "name": "Computers & Workstations",
        "category": "Important", "priority": 2,
        "is_flexible": False, "required_kw": 15.0,
        "allocated_kw": 15.0, "status": "Active",
        "description": "Workstation power for productive work hours"
    },
    {
        "id": "load_6", "name": "Essential HVAC",
        "category": "Important", "priority": 2,
        "is_flexible": False, "required_kw": 10.0,
        "allocated_kw": 10.0, "status": "Active",
        "description": "Minimum climate control for occupied zones"
    },
    # ── Flexible (Priority 3): Can be eco-throttled or shifted ────────────────
    {
        "id": "load_7", "name": "EV Fast Chargers",
        "category": "Flexible", "priority": 3,
        "is_flexible": True, "required_kw": 25.0,
        "allocated_kw": 25.0, "status": "Active",
        "description": "EV charging stations — tolerates 50% eco-throttle or off-peak shift",
        "eco_throttle": 0.50
    },
    {
        "id": "load_8", "name": "Water Heating System",
        "category": "Flexible", "priority": 3,
        "is_flexible": True, "required_kw": 15.0,
        "allocated_kw": 15.0, "status": "Active",
        "description": "Hot water heating — thermal mass allows 2–6 h deferral",
        "eco_throttle": 0.40
    },
    {
        "id": "load_9", "name": "Non-Critical HVAC / Chillers",
        "category": "Flexible", "priority": 3,
        "is_flexible": True, "required_kw": 15.0,
        "allocated_kw": 15.0, "status": "Active",
        "description": "Secondary HVAC zones — temperature setpoint can be relaxed by 2–3°C",
        "eco_throttle": 0.50
    },
    {
        "id": "load_10", "name": "Washing & Heavy Equipment",
        "category": "Flexible", "priority": 3,
        "is_flexible": True, "required_kw": 10.0,
        "allocated_kw": 10.0, "status": "Active",
        "description": "Industrial washers and batch processors — fully deferrable",
        "eco_throttle": 0.40
    },
)

# ── Pure Functions ─────────────────────────────────────────────────────────────

def get_default_loads() -> List[Dict[str, Any]]:
    """
    Returns a fresh deep-copy of the default load registry.

    Returns:
        list[dict]: 10 loads categorised as Critical / Important / Flexible.
    """
    return [dict(load) for load in _DEFAULT_LOADS_TUPLE]


def _compute_scale_factor(loads: List[Dict], target_demand: float) -> float:
    """
    Computes the proportional scale factor to align load sum with predicted demand.

    Args:
        loads: Current load list.
        target_demand: ML-predicted demand (kW).

    Returns:
        float: scale_factor = target_demand / current_load_sum
    """
    current_sum = sum(l["required_kw"] for l in loads)
    if current_sum <= 0 or target_demand <= 0:
        return 1.0
    return target_demand / current_sum


def scale_loads_to_demand(
    loads: List[Dict[str, Any]], target_demand: float
) -> List[Dict[str, Any]]:
    """
    Scales load requirements proportionally so their sum equals ``target_demand``.

    Uses a last-element correction to eliminate floating-point rounding drift.

    Complexity: O(n)

    Args:
        loads:         List of load dicts with ``required_kw`` fields.
        target_demand: Target total demand in kW.

    Returns:
        list[dict]: Deep-copied loads with updated ``required_kw`` values.
    """
    scale = _compute_scale_factor(loads, target_demand)
    scaled = [dict(l) for l in loads]          # O(n) shallow copy per load
    allocated_so_far = 0.0
    last_idx = len(scaled) - 1

    for i, load in enumerate(scaled):
        if i == last_idx:
            # Correction: absorb floating-point error in final element
            load["required_kw"] = round(target_demand - allocated_so_far, 1)
        else:
            load["required_kw"] = round(load["required_kw"] * scale, 1)
            allocated_so_far   += load["required_kw"]
        load["allocated_kw"] = load["required_kw"]

    return scaled


# ── Core Allocation Engine ────────────────────────────────────────────────────

def allocate_energy(
    available_generation: float,
    predicted_demand: float | None = None,
    loads: List[Dict[str, Any]] | None = None,
    electricity_price_kwh: float = 0.16
) -> Dict[str, Any]:
    """
    Priority-based dynamic energy allocation algorithm.

    Algorithm (O(n) single-pass per tier):
    ----------------------------------------
    Phase 1 — Critical loads:
        Each load gets its full required_kw if power remains; otherwise
        whatever is left (partial protected allocation).
    Phase 2 — Important loads:
        Served after critical loads are satisfied.
    Phase 3 — Flexible loads:
        a) If remaining power ≥ total flexible requirement → all active.
        b) Otherwise, loads with eco_throttle receive partial allocation
           (their eco_throttle fraction or available budget, whichever
           is smaller). Loads with insufficient budget are deferred to
           off-peak ("Shifted").

    Cost Model:
        Unoptimised = total_required × peak_penalty_rate  (overload penalty)
        Optimised   = allocated × base_rate + deferred × off_peak_rate
        Saving      = max(0, unoptimised − optimised)

    Args:
        available_generation:  Available power supply in kW.
        predicted_demand:      ML-predicted demand (kW); if provided, loads
                               are scaled to match prediction before allocation.
        loads:                 Optional custom load list. Uses default if None.
        electricity_price_kwh: Base electricity tariff ($/kWh, demo value).

    Returns:
        dict: Complete allocation result including per-load status, cost
              analysis, shortage/surplus, and system alert.
    """
    # ── Input sanitisation ────────────────────────────────────────────────────
    available_generation = float(max(0.0, available_generation))
    loads = [dict(l) for l in (loads or get_default_loads())]   # O(n)

    # Scale load requirements to match ML prediction if provided
    if predicted_demand is not None and predicted_demand > 0:
        loads = scale_loads_to_demand(loads, predicted_demand)   # O(n)

    # ── Single-pass tier separation (O(n)) ────────────────────────────────────
    critical_loads  = []
    important_loads = []
    flexible_loads  = []
    total_required  = 0.0

    for load in loads:
        cat = load.get("category", "Flexible")
        total_required += load["required_kw"]
        if cat == "Critical":
            critical_loads.append(load)
        elif cat == "Important":
            important_loads.append(load)
        else:
            flexible_loads.append(load)

    total_required = round(total_required, 1)
    req_critical   = round(sum(l["required_kw"] for l in critical_loads), 1)
    req_important  = round(sum(l["required_kw"] for l in important_loads), 1)
    req_flexible   = round(sum(l["required_kw"] for l in flexible_loads), 1)

    shifted_loads  = []
    managed_loads  = []
    remaining      = available_generation

    # ── Phase 1: Critical Loads ───────────────────────────────────────────────
    alloc_critical = 0.0
    for load in critical_loads:
        req = load["required_kw"]
        if remaining >= req:
            load["allocated_kw"] = req
            load["status"]       = "Active"
            remaining           -= req
        else:
            load["allocated_kw"] = round(max(0.0, remaining), 1)
            load["status"]       = "Protected Partial" if remaining > 0 else "Offline"
            remaining            = 0.0
        alloc_critical += load["allocated_kw"]

    # ── Phase 2: Important Loads ──────────────────────────────────────────────
    alloc_important = 0.0
    for load in important_loads:
        req = load["required_kw"]
        if remaining >= req:
            load["allocated_kw"] = req
            load["status"]       = "Active"
            remaining           -= req
        else:
            load["allocated_kw"] = round(max(0.0, remaining), 1)
            load["status"]       = "Reduced" if remaining > 0 else "Offline"
            remaining            = 0.0
        alloc_important += load["allocated_kw"]

    # ── Phase 3: Flexible Loads ───────────────────────────────────────────────
    alloc_flexible   = 0.0
    flex_budget      = max(0.0, remaining)

    if flex_budget >= req_flexible:
        # Sufficient power — all flexible loads run normally
        for load in flexible_loads:
            load["allocated_kw"] = load["required_kw"]
            load["status"]       = "Active"
            flex_budget         -= load["required_kw"]
            alloc_flexible      += load["allocated_kw"]
    else:
        # Deficit — eco-throttle or defer each flexible load
        for load in flexible_loads:
            req          = load["required_kw"]
            throttle_pct = float(load.get("eco_throttle", 0.50))
            eco_target   = min(flex_budget, req * throttle_pct)

            if eco_target > 0:
                load["allocated_kw"] = round(eco_target, 1)
                flex_budget         -= load["allocated_kw"]
                load["status"]       = "Managed (Eco-Throttled)"
                managed_loads.append(load["name"])
            else:
                load["allocated_kw"] = 0.0
                load["status"]       = "Shifted (Off-Peak)"
                shifted_loads.append(load["name"])

            alloc_flexible += load["allocated_kw"]

    # ── Aggregate Results ─────────────────────────────────────────────────────
    all_loads      = critical_loads + important_loads + flexible_loads
    total_allocated = round(alloc_critical + alloc_important + alloc_flexible, 1)
    shortage        = round(max(0.0, total_required - available_generation), 1)
    surplus         = round(max(0.0, available_generation - total_allocated), 1)

    # ── Status & Alert ────────────────────────────────────────────────────────
    if shortage == 0:
        system_status = "Normal"
        alert_type    = "success"
        alert_message = "System operating normally. Generation fully satisfies demand."
    elif alloc_critical == req_critical and alloc_important == req_important:
        system_status = "Optimised"
        alert_type    = "warning"
        alert_message = (f"Energy deficit of {shortage} kW managed via smart allocation. "
                         "Critical & Important loads protected; Flexible loads shifted/throttled.")
    else:
        system_status = "Critical Shortage"
        alert_type    = "danger"
        alert_message = (f"Severe generation deficit of {shortage} kW! "
                         "High-priority loads operating under conservation protocol.")

    # ── Cost Estimation ───────────────────────────────────────────────────────
    off_peak_rate      = round(electricity_price_kwh * 0.55, 3)   # 45% cheaper
    peak_penalty_rate  = round(electricity_price_kwh * 1.50, 3)   # 50% surcharge
    deferred_kw        = round(total_required - total_allocated, 1)

    unoptimised_cost   = round(
        total_required * (peak_penalty_rate if shortage > 0 else electricity_price_kwh), 2
    )
    optimised_cost     = round(
        (total_allocated * electricity_price_kwh) + (deferred_kw * off_peak_rate), 2
    )
    estimated_saving   = round(max(0.0, unoptimised_cost - optimised_cost), 2)

    return {
        "critical":            round(alloc_critical, 1),
        "important":           round(alloc_important, 1),
        "flexible":            round(alloc_flexible, 1),
        "req_critical":        req_critical,
        "req_important":       req_important,
        "req_flexible":        req_flexible,
        "total_required":      total_required,
        "total_allocated":     total_allocated,
        "available_generation": round(available_generation, 1),
        "shortage":            shortage,
        "surplus":             surplus,
        "shifted_loads":       shifted_loads,
        "managed_loads":       managed_loads,
        "status":              system_status,
        "alert_type":          alert_type,
        "alert_message":       alert_message,
        "algorithm": {
            "time_complexity":  "O(n)",
            "space_complexity": "O(n)",
            "strategy":         "Priority-based greedy with eco-throttle and off-peak deferral",
            "phases":           ["Critical (100% protect)", "Important", "Flexible (eco-throttle/shift)"]
        },
        "cost_analysis": {
            "base_rate_per_kwh":  electricity_price_kwh,
            "off_peak_rate":      off_peak_rate,
            "peak_penalty_rate":  peak_penalty_rate,
            "unoptimised_cost":   unoptimised_cost,
            "optimized_cost":     optimised_cost,
            "estimated_saving":   estimated_saving,
            "deferred_kw":        deferred_kw,
            "disclaimer":         "Demo cost model using time-of-use tariffs. Not for real billing."
        },
        "loads": all_loads
    }


# ── CLI Entry Point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Dynamic Allocation Demo ===")
    for avail, demand in [(90, 100), (50, 100), (150, 100)]:
        res = allocate_energy(available_generation=avail, predicted_demand=demand)
        print(f"\nAvail={avail} kW | Demand={demand} kW | Status={res['status']}")
        print(f"  Critical={res['critical']} kW | Important={res['important']} kW | "
              f"Flexible={res['flexible']} kW")
        print(f"  Shifted: {res['shifted_loads']} | Saving=\${res['cost_analysis']['estimated_saving']}")
