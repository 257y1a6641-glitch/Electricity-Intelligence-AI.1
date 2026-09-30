/**
 * AI Smart Energy Management System - Frontend Controller
 * Communicates with Flask backend REST endpoints, animates Chart.js visualizations,
 * and handles interactive scenario simulations.
 */

let forecastChart = null;
let demandVsGenChart = null;
let allocationChart = null;

// Initialize on page load
document.addEventListener("DOMContentLoaded", () => {
  initCharts();
  fetchInitialStatus();
});

/* ================= CHART INITIALIZATION ================= */
function initCharts() {
  // Chart 1: Demand Forecast Timeline
  const ctxForecast = document.getElementById("chart-demand-forecast").getContext("2d");
  forecastChart = new Chart(ctxForecast, {
    type: "line",
    data: {
      labels: ["T-4h", "T-3h", "T-2h", "T-1h", "Current", "Forecast (T+1h)"],
      datasets: [
        {
          label: "Demand Curve (kW)",
          data: [68, 72, 75, 78, 80, 94],
          borderColor: "#38bdf8",
          backgroundColor: "rgba(56, 189, 248, 0.12)",
          fill: true,
          tension: 0.35,
          borderWidth: 3,
          pointBackgroundColor: ["#38bdf8", "#38bdf8", "#38bdf8", "#38bdf8", "#38bdf8", "#c084fc"],
          pointBorderColor: "#fff",
          pointRadius: [4, 4, 4, 4, 6, 8],
          pointHoverRadius: 9
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "#1e293b",
          titleColor: "#f8fafc",
          bodyColor: "#38bdf8",
          borderColor: "rgba(255,255,255,0.1)",
          borderWidth: 1,
          padding: 10,
          callbacks: {
            label: (ctx) => `${ctx.dataset.label}: ${ctx.raw} kW`
          }
        }
      },
      scales: {
        x: {
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          ticks: { color: "#94a3b8", font: { size: 11 } }
        },
        y: {
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          ticks: {
            color: "#94a3b8",
            font: { size: 11 },
            callback: (v) => `${v} kW`
          }
        }
      }
    }
  });

  // Chart 2: Demand vs Generation Bar Chart
  const ctxCompare = document.getElementById("chart-demand-vs-generation").getContext("2d");
  demandVsGenChart = new Chart(ctxCompare, {
    type: "bar",
    data: {
      labels: ["Predicted Demand", "Available Generation"],
      datasets: [
        {
          label: "kW",
          data: [94, 100],
          backgroundColor: ["rgba(168, 85, 247, 0.8)", "rgba(16, 185, 129, 0.8)"],
          borderColor: ["#c084fc", "#34d399"],
          borderWidth: 1.5,
          borderRadius: 6,
          barPercentage: 0.55
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "#1e293b",
          padding: 8,
          callbacks: { label: (ctx) => `${ctx.raw} kW` }
        }
      },
      scales: {
        x: {
          grid: { display: false },
          ticks: { color: "#94a3b8", font: { size: 11 } }
        },
        y: {
          grid: { color: "rgba(255, 255, 255, 0.05)" },
          ticks: { color: "#94a3b8", font: { size: 11 } }
        }
      }
    }
  });

  // Chart 3: Energy Allocation Doughnut Chart
  const ctxAlloc = document.getElementById("chart-energy-allocation").getContext("2d");
  allocationChart = new Chart(ctxAlloc, {
    type: "doughnut",
    data: {
      labels: ["Critical Loads", "Important Loads", "Flexible Loads"],
      datasets: [
        {
          data: [45, 40, 9],
          backgroundColor: [
            "rgba(239, 68, 68, 0.85)",   // Red/Coral
            "rgba(245, 158, 11, 0.85)",  // Amber
            "rgba(56, 189, 248, 0.85)"   // Cyan
          ],
          borderColor: "#111827",
          borderWidth: 2,
          hoverOffset: 4
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: "68%",
      plugins: {
        legend: {
          position: "bottom",
          labels: {
            color: "#94a3b8",
            boxWidth: 12,
            font: { size: 11 },
            padding: 8
          }
        },
        tooltip: {
          backgroundColor: "#1e293b",
          padding: 8,
          callbacks: {
            label: (ctx) => ` ${ctx.label}: ${ctx.raw} kW`
          }
        }
      }
    }
  });
}

/* ================= INITIAL STATUS FETCH ================= */
async function fetchInitialStatus() {
  try {
    const res = await fetch("/status");
    if (!res.ok) throw new Error("Status endpoint not responding");
    const data = await res.json();
    
    if (data.system_state) {
      document.getElementById("input-current-demand").value = data.system_state.current_demand;
      document.getElementById("input-available-generation").value = data.system_state.available_generation;
      document.getElementById("input-temperature").value = data.system_state.temperature;
      document.getElementById("input-humidity").value = data.system_state.humidity;
      document.getElementById("input-occupancy").value = data.system_state.occupancy;
      document.getElementById("input-hour").value = data.system_state.hour;
      document.getElementById("input-day-type").value = data.system_state.day_type;
      document.getElementById("input-user-activity").value = data.system_state.user_activity;
      
      document.getElementById("val-current-demand").textContent = data.system_state.current_demand.toFixed(1);
      document.getElementById("val-predicted-demand").textContent = data.system_state.last_prediction.toFixed(1);
      document.getElementById("val-available-generation").textContent = data.system_state.available_generation.toFixed(1);
    }

    if (data.model_metrics) {
      if (data.model_metrics.mae) {
        document.getElementById("meta-mae").textContent = `${data.model_metrics.mae} kW`;
      }
      if (data.model_metrics.r2_score) {
        document.getElementById("meta-r2").textContent = data.model_metrics.r2_score;
      }
    }

    if (data.last_allocation) {
      updateAllocationUI(data.last_allocation);
    }
    
    setConnectionActive(true);
  } catch (err) {
    console.error("Initial load error:", err);
    setConnectionActive(false);
  }
}

/* ================= API CALL: RUN PREDICTION ================= */
async function runPrediction() {
  const payload = getFormPayload();
  showBannerLoading("Running AI demand forecasting model...");

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.error || "Prediction failed");
    }

    const data = await res.json();

    // Update Top Metric Cards
    document.getElementById("val-current-demand").textContent = data.current_demand.toFixed(1);
    document.getElementById("val-predicted-demand").textContent = data.predicted_demand.toFixed(1);
    document.getElementById("val-available-generation").textContent = data.available_generation.toFixed(1);

    const delta = data.predicted_demand - data.current_demand;
    const deltaSign = delta >= 0 ? "+" : "";
    document.getElementById("val-pred-delta").textContent = `${deltaSign}${delta.toFixed(1)} kW from current load`;

    // Update Forecast Chart
    if (forecastChart && data.historical_trend) {
      const chartPoints = [...data.historical_trend, data.predicted_demand];
      forecastChart.data.labels = data.timeline_labels || ["T-4h", "T-3h", "T-2h", "T-1h", "Current", "Forecast (T+1h)"];
      forecastChart.data.datasets[0].data = chartPoints;
      forecastChart.update();
    }

    // Immediately trigger allocation with the newly predicted demand
    await allocateEnergy(data.predicted_demand, data.available_generation);

  } catch (err) {
    showAlert("danger", "Prediction Error", err.message);
  }
}

/* ================= API CALL: ALLOCATE ENERGY ================= */
async function allocateEnergy(overrideDemand = null, overrideGen = null) {
  const currentDemand = overrideDemand !== null ? overrideDemand : parseFloat(document.getElementById("val-predicted-demand").textContent);
  const availGen = overrideGen !== null ? overrideGen : parseFloat(document.getElementById("input-available-generation").value);

  try {
    const res = await fetch("/allocate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        predicted_demand: currentDemand,
        available_generation: availGen
      })
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.error || "Allocation failed");
    }

    const data = await res.json();
    updateAllocationUI(data);

  } catch (err) {
    showAlert("danger", "Allocation Error", err.message);
  }
}

/* ================= API CALL: DEMAND SURGE SIMULATION ================= */
async function simulateDemandSurge() {
  showBannerLoading("Simulating sudden high-demand spike...");
  
  try {
    const res = await fetch("/simulate/demand-surge", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        surge_demand: 125.0,
        available_generation: 100.0
      })
    });

    if (!res.ok) throw new Error("Demand surge simulation failed");
    const data = await res.json();

    // Update form values
    document.getElementById("input-current-demand").value = data.current_demand;
    document.getElementById("input-available-generation").value = data.available_generation;
    document.getElementById("input-user-activity").value = "High";

    // Update Top Metric Cards
    document.getElementById("val-current-demand").textContent = data.current_demand.toFixed(1);
    document.getElementById("val-predicted-demand").textContent = data.predicted_demand.toFixed(1);
    document.getElementById("val-available-generation").textContent = data.available_generation.toFixed(1);
    document.getElementById("val-pred-delta").textContent = "+25.0 kW sudden surge detected";

    // Update Chart
    if (forecastChart && data.historical_trend) {
      forecastChart.data.labels = data.timeline_labels;
      forecastChart.data.datasets[0].data = [...data.historical_trend, data.predicted_demand];
      forecastChart.update();
    }

    // Update Allocation and Alert
    updateAllocationUI(data.allocation);
    showAlert("danger", data.alert_title, data.alert_message);

  } catch (err) {
    showAlert("danger", "Surge Simulation Error", err.message);
  }
}

/* ================= API CALL: GENERATION DROP SIMULATION ================= */
async function simulateGenerationDrop() {
  showBannerLoading("Simulating generation loss / grid trip...");

  try {
    const res = await fetch("/simulate/generation-drop", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        dropped_generation: 70.0,
        predicted_demand: 100.0
      })
    });

    if (!res.ok) throw new Error("Generation drop simulation failed");
    const data = await res.json();

    // Update form values
    document.getElementById("input-available-generation").value = data.available_generation;

    // Update Top Metric Cards
    document.getElementById("val-available-generation").textContent = data.available_generation.toFixed(1);
    document.getElementById("val-predicted-demand").textContent = data.predicted_demand.toFixed(1);

    // Update Allocation and Alert
    updateAllocationUI(data.allocation);
    showAlert("warning", data.alert_title, data.alert_message);

  } catch (err) {
    showAlert("danger", "Generation Drop Error", err.message);
  }
}

/* ================= PREDEFINED DEMO SCENARIOS ================= */
async function applyScenario(num) {
  if (num === 3) {
    // Scenario 3: Sudden Demand Surge
    await simulateDemandSurge();
    return;
  }
  if (num === 4) {
    // Scenario 4: Generation Drop
    await simulateGenerationDrop();
    return;
  }

  // Define inputs for other scenarios
  let scenarioConfig = {};
  if (num === 1) {
    // Scenario 1: Normal Demand
    scenarioConfig = {
      current_demand: 80.0,
      available_generation: 100.0,
      temperature: 25.0,
      humidity: 50,
      occupancy: 50,
      hour: 10,
      day_type: "Weekday",
      user_activity: "Medium"
    };
  } else if (num === 2) {
    // Scenario 2: High Demand
    scenarioConfig = {
      current_demand: 95.0,
      available_generation: 90.0,
      temperature: 34.0,
      humidity: 65,
      occupancy: 85,
      hour: 14,
      day_type: "Weekday",
      user_activity: "High"
    };
  } else if (num === 5) {
    // Scenario 5: High Temperature + High Occupancy
    scenarioConfig = {
      current_demand: 100.0,
      available_generation: 95.0,
      temperature: 39.5,
      humidity: 78,
      occupancy: 98,
      hour: 15,
      day_type: "Weekday",
      user_activity: "High"
    };
  }

  // Populate form fields
  document.getElementById("input-current-demand").value = scenarioConfig.current_demand;
  document.getElementById("input-available-generation").value = scenarioConfig.available_generation;
  document.getElementById("input-temperature").value = scenarioConfig.temperature;
  document.getElementById("input-humidity").value = scenarioConfig.humidity;
  document.getElementById("input-occupancy").value = scenarioConfig.occupancy;
  document.getElementById("input-hour").value = scenarioConfig.hour;
  document.getElementById("input-day-type").value = scenarioConfig.day_type;
  document.getElementById("input-user-activity").value = scenarioConfig.user_activity;

  // Run full pipeline
  await runPrediction();
}

/* ================= SYSTEM RESET ================= */
async function resetSystem() {
  try {
    const res = await fetch("/reset", { method: "POST" });
    const data = await res.json();
    
    // Reset Form
    document.getElementById("input-current-demand").value = 80.0;
    document.getElementById("input-available-generation").value = 100.0;
    document.getElementById("input-temperature").value = 32.0;
    document.getElementById("input-humidity").value = 60;
    document.getElementById("input-occupancy").value = 75;
    document.getElementById("input-hour").value = 14;
    document.getElementById("input-day-type").value = "Weekday";
    document.getElementById("input-user-activity").value = "Medium";

    // Update Cards
    document.getElementById("val-current-demand").textContent = "80.0";
    document.getElementById("val-predicted-demand").textContent = "94.0";
    document.getElementById("val-available-generation").textContent = "100.0";
    document.getElementById("val-pred-delta").textContent = "+14.0 kW from current load";

    // Update Forecast Chart
    if (forecastChart && data.historical_trend) {
      forecastChart.data.labels = data.timeline_labels;
      forecastChart.data.datasets[0].data = [...data.historical_trend, 94.0];
      forecastChart.update();
    }

    updateAllocationUI(data.allocation);
    showAlert("success", "System Reset", "All operational parameters restored to baseline.");

  } catch (err) {
    showAlert("danger", "Reset Error", err.message);
  }
}

/* ================= UI RENDER HELPERS ================= */
function updateAllocationUI(alloc) {
  if (!alloc) return;

  // 1. Balance Card (Shortage / Surplus)
  const isShortage = alloc.shortage > 0;
  const labelBalance = document.getElementById("label-balance");
  const badgeBalance = document.getElementById("badge-balance");
  const valBalance = document.getElementById("val-balance");
  const footerBalance = document.getElementById("footer-balance");

  if (isShortage) {
    labelBalance.textContent = "Energy Shortage";
    badgeBalance.textContent = "Deficit";
    badgeBalance.className = "card-badge badge-red";
    valBalance.textContent = alloc.shortage.toFixed(1);
    valBalance.style.color = "#f87171";
    footerBalance.textContent = "Flexible loads throttled/deferred";
  } else {
    labelBalance.textContent = "Energy Surplus";
    badgeBalance.textContent = "Surplus";
    badgeBalance.className = "card-badge badge-green";
    valBalance.textContent = alloc.surplus.toFixed(1);
    valBalance.style.color = "#34d399";
    footerBalance.textContent = "Clean capacity buffer available";
  }

  // 2. Status Card
  const valStatus = document.getElementById("val-system-status");
  const systemDot = document.getElementById("system-dot");
  const badgeStatus = document.getElementById("badge-status");
  valStatus.textContent = alloc.status;
  
  if (alloc.status === "Normal") {
    valStatus.style.color = "#34d399";
    systemDot.style.background = "#10b981";
    systemDot.style.boxShadow = "0 0 10px #10b981";
    badgeStatus.textContent = "Optimal";
    badgeStatus.className = "card-badge badge-green";
  } else if (alloc.status === "Optimized") {
    valStatus.style.color = "#fbbf24";
    systemDot.style.background = "#f59e0b";
    systemDot.style.boxShadow = "0 0 10px #f59e0b";
    badgeStatus.textContent = "Managed";
    badgeStatus.className = "card-badge badge-savings";
  } else {
    valStatus.style.color = "#f87171";
    systemDot.style.background = "#ef4444";
    systemDot.style.boxShadow = "0 0 10px #ef4444";
    badgeStatus.textContent = "Critical";
    badgeStatus.className = "card-badge badge-red";
  }

  // 3. Cost & Savings Card
  if (alloc.cost_analysis) {
    document.getElementById("val-cost").textContent = alloc.cost_analysis.optimized_cost.toFixed(2);
    document.getElementById("cost-unoptimized").textContent = `$${alloc.cost_analysis.unoptimized_cost.toFixed(2)}/hr`;
    document.getElementById("cost-optimized").textContent = `$${alloc.cost_analysis.optimized_cost.toFixed(2)}/hr`;
    document.getElementById("cost-savings").textContent = `$${alloc.cost_analysis.estimated_saving.toFixed(2)}/hr`;
    document.getElementById("badge-savings").textContent = `Save $${alloc.cost_analysis.estimated_saving.toFixed(2)}`;
    
    const pct = alloc.cost_analysis.unoptimized_cost > 0
      ? Math.round((alloc.cost_analysis.estimated_saving / alloc.cost_analysis.unoptimized_cost) * 100)
      : 0;
    document.getElementById("cost-savings-sub").textContent = `~${pct}% energy cost reduction`;
  }

  // 4. Update Charts
  if (demandVsGenChart) {
    demandVsGenChart.data.datasets[0].data = [alloc.total_required, alloc.available_generation];
    demandVsGenChart.update();
  }

  if (allocationChart) {
    allocationChart.data.datasets[0].data = [alloc.critical, alloc.important, alloc.flexible];
    allocationChart.update();
  }

  // 5. Update Table Summary Badges
  const pctCrit = alloc.req_critical > 0 ? Math.round((alloc.critical / alloc.req_critical) * 100) : 100;
  const pctImp = alloc.req_important > 0 ? Math.round((alloc.important / alloc.req_important) * 100) : 100;
  const pctFlex = alloc.req_flexible > 0 ? Math.round((alloc.flexible / alloc.req_flexible) * 100) : 100;

  document.getElementById("badge-critical-total").textContent = `Critical: ${alloc.critical.toFixed(1)} kW (${pctCrit}%)`;
  document.getElementById("badge-important-total").textContent = `Important: ${alloc.important.toFixed(1)} kW (${pctImp}%)`;
  document.getElementById("badge-flexible-total").textContent = `Flexible: ${alloc.flexible.toFixed(1)} kW (${pctFlex}%)`;

  // 6. Render Dynamic Loads Table
  renderLoadsTable(alloc.loads);

  // 7. Update Alert Banner if not already explicitly showing surge/drop
  if (alloc.alert_message) {
    showAlert(alloc.alert_type, `System Alert [${alloc.status}]`, alloc.alert_message);
  }
}

function renderLoadsTable(loads) {
  const tbody = document.getElementById("loads-table-body");
  if (!tbody || !loads) return;

  tbody.innerHTML = "";

  loads.forEach(load => {
    const tr = document.createElement("tr");

    // Category Badge
    const catClass = load.category === "Critical" ? "tag-critical" :
                     load.category === "Important" ? "tag-important" : "tag-flexible";

    // Status Badge
    let statusClass = "status-active";
    if (load.status.includes("Managed")) statusClass = "status-managed";
    else if (load.status.includes("Shifted")) statusClass = "status-shifted";
    else if (load.status.includes("Reduced")) statusClass = "status-reduced";

    // Fulfillment Percentage
    const pct = load.required_kw > 0 ? Math.min(100, Math.round((load.allocated_kw / load.required_kw) * 100)) : 100;
    const barColor = pct >= 99 ? "#10b981" : pct >= 40 ? "#f59e0b" : "#a855f7";

    tr.innerHTML = `
      <td><strong>${escapeHtml(load.name)}</strong></td>
      <td><span class="table-badge ${catClass}">${load.category}</span></td>
      <td>Priority ${load.priority}</td>
      <td>${load.required_kw.toFixed(1)}</td>
      <td><strong>${load.allocated_kw.toFixed(1)}</strong></td>
      <td>
        <span class="progress-bar-bg">
          <span class="progress-bar-fill" style="width: ${pct}%; background: ${barColor};"></span>
        </span>
        ${pct}%
      </td>
      <td><span class="table-badge ${statusClass}">${escapeHtml(load.status)}</span></td>
    `;

    tbody.appendChild(tr);
  });
}

function showAlert(type, title, message) {
  const banner = document.getElementById("alert-banner");
  const alertIcon = document.getElementById("alert-icon");
  const alertTitle = document.getElementById("alert-title");
  const alertMsg = document.getElementById("alert-message");
  const alertTime = document.getElementById("alert-timestamp");

  banner.className = `alert alert-${type}`;
  alertIcon.textContent = type === "danger" ? "⚠" : type === "warning" ? "⚡" : "✓";
  alertTitle.textContent = title ? `${title}:` : "";
  alertMsg.textContent = message;
  
  const now = new Date();
  alertTime.textContent = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

function showBannerLoading(msg) {
  showAlert("warning", "Computing", msg);
}

function setConnectionActive(active) {
  const dot = document.getElementById("connection-dot");
  const txt = document.getElementById("connection-text");
  if (active) {
    dot.className = "status-dot pulse-green";
    txt.textContent = "Backend Active";
  } else {
    dot.className = "status-dot";
    dot.style.background = "#ef4444";
    txt.textContent = "Backend Disconnected";
  }
}

function getFormPayload() {
  return {
    current_demand: parseFloat(document.getElementById("input-current-demand").value),
    available_generation: parseFloat(document.getElementById("input-available-generation").value),
    temperature: parseFloat(document.getElementById("input-temperature").value),
    humidity: parseFloat(document.getElementById("input-humidity").value),
    occupancy: parseFloat(document.getElementById("input-occupancy").value),
    hour: parseInt(document.getElementById("input-hour").value),
    day_type: document.getElementById("input-day-type").value,
    user_activity: document.getElementById("input-user-activity").value
  };
}

function escapeHtml(str) {
  if (!str) return "";
  return str.replace(/[&<>'"]/g, 
    tag => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[tag] || tag)
  );
}
