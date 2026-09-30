"""
Dynamic Energy Allocation Module for Hackathon Prototype.
Prioritizes Critical loads, supplies Important loads, and dynamically throttles
or shifts Flexible loads during generation deficits or demand surges to eliminate
overload conditions and save energy costs.
"""

from typing import List, Dict, Any
import copy

DEFAULT_LOADS = [
    # Critical Loads (Priority 1: Must never be dropped)
    {
        "id": "load_1",
        "name": "Medical Equipment",
        "category": "Critical",
        "priority": 1,
        "is_flexible": False,
        "required_kw": 20.0,
        "allocated_kw": 20.0,
        "status": "Active"
    },
    {
        "id": "load_2",
        "name": "Emergency Systems",
        "category": "Critical",
        "priority": 1,
        "is_flexible": False,
        "required_kw": 10.0,
        "allocated_kw": 10.0,
        "status": "Active"
    },
    {
        "id": "load_3",
        "name": "Essential Servers",
        "category": "Critical",
        "priority": 1,
        "is_flexible": False,
        "required_kw": 15.0,
        "allocated_kw": 15.0,
        "status": "Active"
    },
    # Important Loads (Priority 2: Necessary for general operations)
    {
        "id": "load_4",
        "name": "Lighting",
        "category": "Important",
        "priority": 2,
        "is_flexible": False,
        "required_kw": 15.0,
        "allocated_kw": 15.0,
        "status": "Active"
    },
    {
        "id": "load_5",
        "name": "Computers & Workstations",
        "category": "Important",
        "priority": 2,
        "is_flexible": False,
        "required_kw": 15.0,
        "allocated_kw": 15.0,
        "status": "Active"
    },
    {
        "id": "load_6",
        "name": "Essential HVAC",
        "category": "Important",
        "priority": 2,
        "is_flexible": False,
        "required_kw": 10.0,
        "allocated_kw": 10.0,
        "status": "Active"
    },
    # Flexible Loads (Priority 3: Can be dynamically throttled or shifted)
    {
        "id": "load_7",
        "name": "EV Fast Chargers",
        "category": "Flexible",
        "priority": 3,
        "is_flexible": True,
        "required_kw": 25.0,
        "allocated_kw": 25.0,
        "status": "Active"
    },
    {
        "id": "load_8",
        "name": "Water Heating System",
        "category": "Flexible",
        "priority": 3,
        "is_flexible": True,
        "required_kw": 15.0,
        "allocated_kw": 15.0,
        "status": "Active"
    },
    {
        "id": "load_9",
        "name": "Non-Critical HVAC / Chillers",
        "category": "Flexible",
        "priority": 3,
        "is_flexible": True,
        "required_kw": 15.0,
        "allocated_kw": 15.0,
        "status": "Active"
    },
    {
        "id": "load_10",
        "name": "Washing & Heavy Equipment",
        "category": "Flexible",
        "priority": 3,
        "is_flexible": True,
        "required_kw": 10.0,
        "allocated_kw": 10.0,
        "status": "Active"
    }
]

def get_default_loads() -> List[Dict[str, Any]]:
    """Returns a fresh deep copy of default loads."""
    return copy.deepcopy(DEFAULT_LOADS)

def scale_loads_to_demand(loads: List[Dict[str, Any]], target_demand: float) -> List[Dict[str, Any]]:
    """
    Scales the load requirements proportionally so their sum matches target predicted demand,
    giving a consistent representation between ML predicted demand and load table.
    """
    current_sum = sum(l["required_kw"] for l in loads)
    if current_sum <= 0 or target_demand <= 0:
        return loads

    scale_factor = target_demand / current_sum
    scaled = copy.deepcopy(loads)
    allocated_so_far = 0.0
    for i, l in enumerate(scaled):
        if i == len(scaled) - 1:
            l["required_kw"] = round(target_demand - allocated_so_far, 1)
        else:
            l["required_kw"] = round(l["required_kw"] * scale_factor, 1)
            allocated_so_far += l["required_kw"]
        l["allocated_kw"] = l["required_kw"]
    return scaled


def allocate_energy(
    available_generation: float,
    predicted_demand: float = None,
    loads: List[Dict[str, Any]] = None,
    electricity_price_kwh: float = 0.16
) -> Dict[str, Any]:
    """
    Dynamic allocation engine.
    - Ensures critical loads get 100% priority.
    - Allocates to important loads next.
    - Manages/shifts flexible loads when available generation is insufficient.
    - Avoids overload conditions and computes demo cost savings.
    """
    if loads is None or len(loads) == 0:
        loads = get_default_loads()
    else:
        loads = copy.deepcopy(loads)

    available_generation = float(max(0.0, available_generation))
    
    # If predicted_demand is specified and differs from current sum of loads,
    # scale the loads to reflect the forecast
    if predicted_demand is not None and predicted_demand > 0:
        loads = scale_loads_to_demand(loads, predicted_demand)

    total_required = round(sum(l["required_kw"] for l in loads), 1)

    critical_loads = [l for l in loads if l.get("category") == "Critical"]
    important_loads = [l for l in loads if l.get("category") == "Important"]
    flexible_loads = [l for l in loads if l.get("category") == "Flexible"]

    req_critical = round(sum(l["required_kw"] for l in critical_loads), 1)
    req_important = round(sum(l["required_kw"] for l in important_loads), 1)
    req_flexible = round(sum(l["required_kw"] for l in flexible_loads), 1)

    shifted_loads = []
    managed_loads = []
    remaining_power = available_generation

    # 1. Allocate to CRITICAL LOADS first
    alloc_critical = 0.0
    for load in critical_loads:
        req = load["required_kw"]
        if remaining_power >= req:
            load["allocated_kw"] = req
            load["status"] = "Active"
            remaining_power -= req
        else:
            # Under severe grid collapse: allocate whatever is left
            load["allocated_kw"] = round(max(0.0, remaining_power), 1)
            load["status"] = "Protected Partial" if remaining_power > 0 else "Offline"
            remaining_power = 0.0
        alloc_critical += load["allocated_kw"]

    # 2. Allocate to IMPORTANT LOADS second
    alloc_important = 0.0
    for load in important_loads:
        req = load["required_kw"]
        if remaining_power >= req:
            load["allocated_kw"] = req
            load["status"] = "Active"
            remaining_power -= req
        else:
            load["allocated_kw"] = round(max(0.0, remaining_power), 1)
            load["status"] = "Reduced" if remaining_power > 0 else "Offline"
            remaining_power = 0.0
        alloc_important += load["allocated_kw"]

    # 3. Manage FLEXIBLE LOADS third
    alloc_flexible = 0.0
    if len(flexible_loads) > 0:
        if remaining_power >= req_flexible:
            # Surplus or exact match: All flexible loads run normally
            for load in flexible_loads:
                load["allocated_kw"] = load["required_kw"]
                load["status"] = "Active"
                remaining_power -= load["required_kw"]
                alloc_flexible += load["allocated_kw"]
        else:
            # Deficit: Intelligently throttle and shift flexible loads!
            # We don't shut everything off; we prioritize partial power to EV/HVAC,
            # and shift deferrable tasks (water heating, laundry/heavy processing)
            flexible_budget = max(0.0, remaining_power)
            
            for load in flexible_loads:
                req = load["required_kw"]
                name = load["name"].lower()
                
                # EV Chargers & Non-Critical HVAC can operate in low-power eco-mode (Managed)
                if ("ev" in name or "hvac" in name or "chiller" in name) and flexible_budget > 0:
                    target_portion = min(flexible_budget, req * 0.50)  # Throttled to 50% or available budget
                    load["allocated_kw"] = round(target_portion, 1)
                    flexible_budget -= load["allocated_kw"]
                    if load["allocated_kw"] > 0:
                        load["status"] = "Managed (Eco-Throttled)"
                        managed_loads.append(load["name"])
                    else:
                        load["status"] = "Shifted (Off-Peak)"
                        shifted_loads.append(load["name"])
                # Water heating & Heavy equipment can be fully shifted to off-peak hours
                elif flexible_budget >= req * 0.40:
                    load["allocated_kw"] = round(req * 0.40, 1)
                    flexible_budget -= load["allocated_kw"]
                    load["status"] = "Managed (Reduced)"
                    managed_loads.append(load["name"])
                else:
                    load["allocated_kw"] = 0.0
                    load["status"] = "Shifted (Off-Peak)"
                    shifted_loads.append(load["name"])

                alloc_flexible += load["allocated_kw"]

            remaining_power = max(0.0, flexible_budget)

    # Reassemble all loads preserving order
    all_loads = critical_loads + important_loads + flexible_loads
    total_allocated = round(alloc_critical + alloc_important + alloc_flexible, 1)
    
    shortage = round(max(0.0, total_required - available_generation), 1)
    surplus = round(max(0.0, available_generation - total_allocated), 1)

    # Determine status & alert message
    if shortage == 0:
        system_status = "Normal"
        alert_type = "success"
        alert_message = "System operating normally. Generation fully satisfies demand."
    elif alloc_critical == req_critical and alloc_important == req_important:
        system_status = "Optimized"
        alert_type = "warning"
        alert_message = f"Energy deficit of {shortage} kW managed. Critical & Important loads protected; Flexible loads shifted/throttled."
    else:
        system_status = "Critical Shortage"
        alert_type = "danger"
        alert_message = f"Severe generation deficit of {shortage} kW! High priority loads operating under conservation protocol."

    # Cost Estimation calculation
    # Demo pricing: Peak penalty occurs during unmanaged shortage; shifted loads run at off-peak rates
    off_peak_price = round(electricity_price_kwh * 0.55, 3)
    peak_penalty_price = round(electricity_price_kwh * 1.50, 3)
    
    # Without smart allocation: grid overload penalties or diesel backup costs
    unoptimized_cost = round(total_required * (peak_penalty_price if shortage > 0 else electricity_price_kwh), 2)
    
    # With smart allocation: supplied energy at base rate + deferred shifted loads at off-peak rate
    deferred_kw = round(total_required - total_allocated, 1)
    optimized_cost = round((total_allocated * electricity_price_kwh) + (deferred_kw * off_peak_price), 2)
    estimated_saving = round(max(0.0, unoptimized_cost - optimized_cost), 2)

    return {
        "critical": round(alloc_critical, 1),
        "important": round(alloc_important, 1),
        "flexible": round(alloc_flexible, 1),
        "req_critical": req_critical,
        "req_important": req_important,
        "req_flexible": req_flexible,
        "total_required": total_required,
        "total_allocated": total_allocated,
        "available_generation": round(available_generation, 1),
        "shortage": shortage,
        "surplus": surplus,
        "shifted_loads": shifted_loads,
        "managed_loads": managed_loads,
        "status": system_status,
        "alert_type": alert_type,
        "alert_message": alert_message,
        "cost_analysis": {
            "base_rate_per_kwh": electricity_price_kwh,
            "off_peak_rate": off_peak_price,
            "peak_penalty_rate": peak_penalty_price,
            "unoptimized_cost": unoptimized_cost,
            "optimized_cost": optimized_cost,
            "estimated_saving": estimated_saving,
            "deferred_kw": deferred_kw,
            "disclaimer": "Demo calculation based on time-of-use tariffs and load shifting savings."
        },
        "loads": all_loads
    }

if __name__ == "__main__":
    res = allocate_energy(available_generation=90.0, predicted_demand=100.0)
    print("Allocation Result:")
    print(f"Total Required: {res['total_required']} kW, Total Allocated: {res['total_allocated']} kW")
    print(f"Critical: {res['critical']} kW, Important: {res['important']} kW, Flexible: {res['flexible']} kW")
    print(f"Shifted Loads: {res['shifted_loads']}")
    print(f"Cost Savings: ${res['cost_analysis']['estimated_saving']}")
