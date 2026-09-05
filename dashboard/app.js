const state = { payload: null, selected: "" };
const $ = (id) => document.getElementById(id);
const format = (value, digits = 1) => value == null || Number.isNaN(Number(value)) ? "--" : Number(value).toFixed(digits);
const cleanCity = (name) => name.replace("_", " ");

async function loadData() {
  const response = await fetch("/api/data", { cache: "no-store" });
  if (!response.ok) throw new Error("Could not load live data.");
  state.payload = await response.json(); render(); $("connection").textContent = "Live connection";
}
function render() {
  const { latest, history, updated } = state.payload; const anomalies = latest.filter((row) => Number(row.Ensemble_Anomaly) === 1); const highest = Math.max(0, ...latest.map((row) => Number(row.Model_Agreement) || 0));
  $("station-count").textContent = latest.length; $("anomaly-count").textContent = anomalies.length; $("health").textContent = latest.length ? `${Math.round((latest.length - anomalies.length) / latest.length * 100)}%` : "--"; $("health-note").textContent = anomalies.length ? "Review flagged stations" : "No ensemble anomalies detected"; $("agreement").textContent = highest ? `${highest} / 4` : "--";
  $("updated").textContent = updated ? new Date(updated.replace(" ", "T") + "Z").toLocaleString([], { dateStyle: "medium", timeStyle: "short" }) : "No observation"; $("status-count").textContent = `${anomalies.length} flagged`;
  if (!state.selected || !latest.some((row) => row.Location === state.selected)) state.selected = latest[0]?.Location || "";
  renderSelect(latest); renderCities(latest); renderTable(latest); renderChart(history[state.selected] || []);
}
function renderSelect(rows) { $("city-select").innerHTML = [...new Set(rows.map((row) => row.Location))].map((city) => `<option value="${city}" ${city === state.selected ? "selected" : ""}>${cleanCity(city)}</option>`).join(""); $("selected-city").textContent = cleanCity(state.selected); }
function renderCities(rows) { $("city-list").innerHTML = rows.map((row) => { const severity = row.Anomaly_Severity || "Normal"; return `<div class="city ${row.Location === state.selected ? "selected" : ""}" data-city="${row.Location}"><div><div class="city-name">${cleanCity(row.Location)}</div><div class="city-reading">${format(row.Temperature_C)} C / ${format(row.Humidity_Percent, 0)}% RH</div></div><div class="status ${severity.toLowerCase()}">${severity}<br><small>${row.Model_Agreement || 0}/4 vote</small></div></div>`; }).join(""); document.querySelectorAll(".city").forEach((city) => city.addEventListener("click", () => { state.selected = city.dataset.city; render(); })); }
function renderTable(rows) { $("observations").innerHTML = rows.map((row) => { const severity = row.Anomaly_Severity || "Normal"; const time = row.DateTime?.split(" ")[1]?.slice(0, 5) || "--"; return `<tr><td>${cleanCity(row.Location)}</td><td>${time}</td><td>${format(row.Temperature_C)} C</td><td>${format(row.Humidity_Percent, 0)}%</td><td>${format(row.Pressure_hPa)} hPa</td><td>${row.Model_Agreement || 0} / 4</td><td><span class="badge ${severity.toLowerCase()}">${severity}</span></td><td class="reason">${row.Anomaly_Reason || "--"}</td></tr>`; }).join(""); }
function renderChart(points) {
  const svg = $("trend-chart"); const width = 720; const height = 290; const pad = { left: 42, right: 18, top: 18, bottom: 35 }; const innerW = width - pad.left - pad.right; const innerH = height - pad.top - pad.bottom;
  if (!points.length) { svg.innerHTML = `<text x="${width / 2}" y="145" text-anchor="middle" class="chart-label">No history available for this station</text>`; return; }
  const values = points.flatMap((point) => [point.temperature, point.humidity]).filter((value) => value != null); const min = Math.floor(Math.min(...values) - 5); const max = Math.ceil(Math.max(...values) + 5); const x = (index) => pad.left + index * innerW / Math.max(1, points.length - 1); const y = (value) => pad.top + (max - value) * innerH / (max - min); const line = (key) => points.map((point, index) => `${index ? "L" : "M"}${x(index).toFixed(1)},${y(point[key] ?? min).toFixed(1)}`).join(" ");
  let grid = ""; for (let index = 0; index < 5; index += 1) { const value = min + (max - min) * index / 4; const yPos = y(value); grid += `<line x1="${pad.left}" y1="${yPos}" x2="${width - pad.right}" y2="${yPos}" class="chart-grid"/><text x="5" y="${yPos + 4}" class="chart-label">${Math.round(value)}</text>`; }
  svg.innerHTML = `${grid}<path d="${line("temperature")}" class="chart-line" stroke="var(--blue)"/><path d="${line("humidity")}" class="chart-line" stroke="var(--amber)"/>${points.map((point, index) => `<circle cx="${x(index)}" cy="${y(point.temperature ?? min)}" r="3.5" fill="var(--blue)" class="chart-point"/>`).join("")}<text x="${pad.left}" y="${height - 8}" class="chart-label">${points[0].time?.split(" ")[0] || ""}</text><text x="${width - pad.right}" y="${height - 8}" text-anchor="end" class="chart-label">${points.at(-1).time?.split(" ")[0] || ""}</text>`;
}
$("city-select").addEventListener("change", (event) => { state.selected = event.target.value; render(); });
$("refresh").addEventListener("click", async () => { const button = $("refresh"); button.disabled = true; button.textContent = "Refreshing..."; $("error").textContent = ""; try { const response = await fetch("/api/refresh", { method: "POST" }); if (!response.ok) throw new Error((await response.json()).error || "Refresh failed."); state.payload = await response.json(); render(); } catch (error) { $("error").textContent = error.message; } finally { button.disabled = false; button.textContent = "Refresh data"; } });
loadData().catch((error) => { $("connection").textContent = "Offline"; $("error").textContent = error.message + " Start dashboard_server.py first."; });