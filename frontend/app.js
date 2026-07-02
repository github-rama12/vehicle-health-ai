const API = ""; // same-origin, since FastAPI serves this frontend too

let vehicleIds = [];
let currentVehicle = null;
let currentDay = 0;
let numDays = 0;
let autoplayTimer = null;

const el = (id) => document.getElementById(id);

// ---------------------------------------------------------------------
// Bootstrapping
// ---------------------------------------------------------------------
async function init() {
  try {
    const health = await fetchJSON(`${API}/api/health`);
    el("apiStatus").textContent = `online (${health.vehicles_loaded} vehicles)`;
    el("apiStatus").style.color = "#16a34a";
  } catch (e) {
    el("apiStatus").textContent = "offline — is the backend running?";
    el("apiStatus").style.color = "#dc2626";
    return;
  }

  const { vehicle_ids } = await fetchJSON(`${API}/api/vehicles`);
  vehicleIds = vehicle_ids;
  const select = el("vehicleSelect");
  select.innerHTML = vehicleIds.map((v) => `<option value="${v}">${v}</option>`).join("");
  select.addEventListener("change", () => loadVehicle(parseInt(select.value)));

  el("daySlider").addEventListener("input", (e) => {
    currentDay = parseInt(e.target.value);
    el("dayLabel").textContent = currentDay;
    renderDay(currentDay);
  });

  el("autoplayToggle").addEventListener("change", (e) => {
    if (e.target.checked) startAutoplay();
    else stopAutoplay();
  });

  await loadVehicle(vehicleIds[0]);
}

async function fetchJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Request failed: ${url}`);
  return res.json();
}

// ---------------------------------------------------------------------
// Vehicle loading
// ---------------------------------------------------------------------
async function loadVehicle(vehicleId) {
  currentVehicle = vehicleId;
  const meta = await fetchJSON(`${API}/api/vehicle/${vehicleId}/meta`);
  numDays = meta.num_days;
  const slider = el("daySlider");
  slider.max = numDays - 1;
  currentDay = numDays - 1;
  slider.value = currentDay;
  el("dayLabel").textContent = currentDay;
  await renderDay(currentDay);
}

// ---------------------------------------------------------------------
// Rendering a single day's prediction
// ---------------------------------------------------------------------
async function renderDay(day) {
  const data = await fetchJSON(`${API}/api/vehicle/${currentVehicle}/day/${day}`);

  el("subtitle").textContent =
    `Vehicle #${data.vehicle_id} • Day ${data.day} • Odometer: ${data.odometer_km.toLocaleString()} km • ` +
    `Inference running via edge AI backend`;

  drawGauge(data.health_score);
  const statusEl = el("statusLabel");
  statusEl.textContent = data.status;
  statusEl.className = data.status.toLowerCase().includes("excellent") ? "excellent"
    : data.status.toLowerCase().includes("good") ? "good"
    : data.status.toLowerCase().includes("attention") ? "attention" : "critical";

  renderAlert(data);
  renderComponentBars(data.component_probabilities);
  renderRUL(data.remaining_useful_life_days);
  renderSensorTiles(data.sensor_reading);

  const history = await fetchJSON(`${API}/api/vehicle/${currentVehicle}/history?up_to_day=${day}`);
  renderCharts(history.history);
}

function renderAlert(data) {
  const box = el("alertBox");
  const p = data.fault_probability;
  if (p > 0.5) {
    box.className = "alert-box alert-critical";
    box.innerHTML = `<b>Predicted issue: ${data.predicted_component.toUpperCase()}</b><br/><br/>
      Model confidence: ${(p * 100).toFixed(0)}% probability of near-term fault.<br/><br/>
      Recommend inspection within the next few days.`;
  } else if (p > 0.25) {
    box.className = "alert-box alert-warning";
    box.innerHTML = `Early drift detected, possibly related to <b>${data.predicted_component}</b>.
      Confidence: ${(p * 100).toFixed(0)}%. Monitor closely.`;
  } else {
    box.className = "alert-box alert-neutral";
    box.textContent = "No faults detected. All monitored systems nominal.";
  }
}

function renderComponentBars(probs) {
  const container = el("componentBars");
  const entries = Object.entries(probs).sort((a, b) => b[1] - a[1]);
  container.innerHTML = entries.map(([name, prob]) => `
    <div class="component-bar-row">
      <span>${name}</span>
      <div class="component-bar-track">
        <div class="component-bar-fill" style="width:${(prob * 100).toFixed(0)}%"></div>
      </div>
      <span>${(prob * 100).toFixed(0)}%</span>
    </div>
  `).join("");
}

function renderRUL(rul) {
  const container = el("rulList");
  const order = ["engine", "battery", "brakes", "tires"];
  container.innerHTML = order.map((comp) => {
    const days = rul[comp];
    const healthy = days > 900;
    const display = healthy ? "999+ (healthy)" : `${days.toFixed(0)} days`;
    const pct = healthy ? 100 : Math.max(2, Math.min(100, days / 3));
    return `
      <div class="rul-row">
        <div class="rul-label">${comp.charAt(0).toUpperCase() + comp.slice(1)} — ${display}</div>
        <div class="rul-track"><div class="rul-fill" style="width:${pct}%"></div></div>
      </div>
    `;
  }).join("");
}

const BASELINES = {
  engine_temp_c: 90, vibration_g: 0.30, battery_voltage_v: 12.6,
  brake_pad_mm: 9.0, tire_pressure_psi: 32, fuel_efficiency_kmpl: 14.5,
};
const LABELS = {
  engine_temp_c: ["Engine Temp", "°C"], vibration_g: ["Vibration", "g"],
  battery_voltage_v: ["Battery", "V"], brake_pad_mm: ["Brake Pad", "mm"],
  tire_pressure_psi: ["Tire Pressure", "psi"], fuel_efficiency_kmpl: ["Fuel Efficiency", "km/l"],
};

function renderSensorTiles(reading) {
  const container = el("sensorTiles");
  container.innerHTML = Object.keys(BASELINES).map((key) => {
    const [label, unit] = LABELS[key];
    const val = reading[key];
    const baseline = BASELINES[key];
    const delta = val - baseline;
    const deltaClass = delta > 0 ? "delta-up" : "delta-down";
    const digits = key === "vibration_g" ? 3 : 1;
    return `
      <div class="sensor-tile">
        <div class="label">${label}</div>
        <div class="value">${val.toFixed(digits)} ${unit}</div>
        <div class="delta ${deltaClass}">${delta >= 0 ? "+" : ""}${delta.toFixed(digits)} vs baseline</div>
      </div>
    `;
  }).join("");
}

// ---------------------------------------------------------------------
// Gauge (canvas semicircle) + charts
// ---------------------------------------------------------------------
function drawGauge(score) {
  const canvas = el("gaugeCanvas");
  const ctx = canvas.getContext("2d");
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  const cx = w / 2, cy = h - 10, r = 90;
  const startAngle = Math.PI, endAngle = 2 * Math.PI;

  // background bands
  const bands = [
    [0, 40, "#4d1f1f"], [40, 60, "#4d3a10"], [60, 80, "#3a4310"], [80, 100, "#1e3a26"],
  ];
  bands.forEach(([lo, hi, color]) => {
    ctx.beginPath();
    const a0 = startAngle + (lo / 100) * Math.PI;
    const a1 = startAngle + (hi / 100) * Math.PI;
    ctx.arc(cx, cy, r, a0, a1);
    ctx.lineWidth = 18;
    ctx.strokeStyle = color;
    ctx.stroke();
  });

  // value arc
  const color = score >= 80 ? "#16a34a" : score >= 60 ? "#65a30d" : score >= 40 ? "#f59e0b" : "#dc2626";
  ctx.beginPath();
  ctx.arc(cx, cy, r, startAngle, startAngle + (score / 100) * Math.PI);
  ctx.lineWidth = 18;
  ctx.strokeStyle = color;
  ctx.stroke();

  // needle-less readout
  ctx.fillStyle = "#e6e6e6";
  ctx.font = "bold 28px sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(`${score} / 100`, cx, cy - 20);
}

let chart1 = null, chart2 = null;
function renderCharts(history) {
  const days = history.map((r) => r.day);

  const ctx1 = el("chartEngineVib").getContext("2d");
  if (chart1) chart1.destroy();
  chart1 = new Chart(ctx1, {
    type: "line",
    data: {
      labels: days,
      datasets: [
        { label: "Engine Temp (°C)", data: history.map((r) => r.engine_temp_c), borderColor: "#2563eb", tension: 0.2, pointRadius: 0 },
        { label: "Vibration (x100 g)", data: history.map((r) => r.vibration_g * 100), borderColor: "#f59e0b", tension: 0.2, pointRadius: 0 },
      ],
    },
    options: chartOptions("Engine Temp & Vibration"),
  });

  const ctx2 = el("chartBattBrake").getContext("2d");
  if (chart2) chart2.destroy();
  chart2 = new Chart(ctx2, {
    type: "line",
    data: {
      labels: days,
      datasets: [
        { label: "Battery (V)", data: history.map((r) => r.battery_voltage_v), borderColor: "#16a34a", tension: 0.2, pointRadius: 0 },
        { label: "Brake Pad (mm)", data: history.map((r) => r.brake_pad_mm), borderColor: "#dc2626", tension: 0.2, pointRadius: 0 },
      ],
    },
    options: chartOptions("Battery & Brake Pad Wear"),
  });
}

function chartOptions(title) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      title: { display: true, text: title, color: "#e6e6e6" },
      legend: { labels: { color: "#9aa0ac" } },
    },
    scales: {
      x: { ticks: { color: "#9aa0ac" }, grid: { color: "#262b38" } },
      y: { ticks: { color: "#9aa0ac" }, grid: { color: "#262b38" } },
    },
  };
}

// ---------------------------------------------------------------------
// Autoplay
// ---------------------------------------------------------------------
function startAutoplay() {
  stopAutoplay();
  autoplayTimer = setInterval(async () => {
    if (currentDay >= numDays - 1) {
      currentDay = 0;
    } else {
      currentDay += 1;
    }
    el("daySlider").value = currentDay;
    el("dayLabel").textContent = currentDay;
    await renderDay(currentDay);
  }, 400);
}
function stopAutoplay() {
  if (autoplayTimer) clearInterval(autoplayTimer);
  autoplayTimer = null;
}

init();
