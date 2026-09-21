import React, { useState, useEffect } from "react";
import API from "../services/api";
import { Bar, Line } from "react-chartjs-2";
import {
  Chart as ChartJS, BarElement, LineElement, CategoryScale,
  LinearScale, PointElement, Tooltip, Legend,
} from "chart.js";

ChartJS.register(BarElement, LineElement, CategoryScale, LinearScale, PointElement, Tooltip, Legend);

function getSoilStatus(value, type) {
  const t = {
    nitrogen: { low: 25, good: 50 },
    phosphorus: { low: 15, good: 35 },
    potassium: { low: 20, good: 50 },
  }[type];

  if (!t) return { label: "N/A", color: "bg-gray-100 text-gray-500", bar: "#94a3b8", pct: 50 };
  if (value < t.low) return { label: "Low", color: "bg-red-100 text-red-700", bar: "#ef4444", pct: Math.round((value / t.low) * 40) };
  if (value < t.good) return { label: "Fair", color: "bg-yellow-100 text-yellow-700", bar: "#f59e0b", pct: Math.round(40 + ((value - t.low) / (t.good - t.low)) * 30) };
  return { label: "Good", color: "bg-green-100 text-green-700", bar: "#22c55e", pct: Math.min(100, Math.round(70 + ((value - t.good) / t.good) * 30)) };
}

function getPhStatus(ph) {
  if (ph < 5.5) return { label: "Too Acidic", color: "bg-red-100 text-red-700", bar: "#ef4444" };
  if (ph <= 7.5) return { label: "Optimal", color: "bg-green-100 text-green-700", bar: "#22c55e" };
  if (ph <= 8.5) return { label: "Alkaline", color: "bg-yellow-100 text-yellow-700", bar: "#f59e0b" };
  return { label: "Too Alkaline", color: "bg-red-100 text-red-700", bar: "#ef4444" };
}

function SoilHealthCard({ n, p, k, ph, crop }) {
  const ns = getSoilStatus(n, "nitrogen");
  const ps = getSoilStatus(p, "phosphorus");
  const ks = getSoilStatus(k, "potassium");
  const phs = getPhStatus(ph);
  const anyLow = [ns, ps, ks].some((s) => s.label === "Low") || phs.label.includes("Acidic") || phs.label.includes("Alkaline");
  const allGood = [ns, ps, ks].every((s) => s.label === "Good") && phs.label === "Optimal";

  const rows = [
    { label: "Nitrogen (N)", value: n, unit: "kg/ha", status: ns },
    { label: "Phosphorus (P)", value: p, unit: "kg/ha", status: ps },
    { label: "Potassium (K)", value: k, unit: "kg/ha", status: ks },
    { label: "Soil pH", value: ph, unit: "", status: { ...phs, pct: Math.round(((ph - 3.5) / 6.4) * 100) } },
  ];

  return (
    <div className="bg-white p-5 rounded shadow">
      <div className="flex justify-between items-center mb-4">
        <h2 className="font-bold">Soil Health Score</h2>
        <span className={`text-xs font-semibold px-2 py-1 rounded ${allGood ? "bg-green-100 text-green-700" : anyLow ? "bg-red-100 text-red-700" : "bg-yellow-100 text-yellow-700"}`}>
          {allGood ? "Healthy" : anyLow ? "Needs Attention" : "Fair"}
        </span>
      </div>

      <div className="space-y-3">
        {rows.map((r, i) => (
          <div key={i}>
            <div className="flex justify-between items-center mb-1">
              <span className="text-sm font-medium text-gray-700">{r.label}</span>
              <div className="flex items-center gap-2">
                <span className="text-sm text-gray-500">{r.value}{r.unit}</span>
                <span className={`text-xs font-semibold px-2 py-0.5 rounded ${r.status.color}`}>{r.status.label}</span>
              </div>
            </div>
            <div className="w-full bg-gray-100 rounded-full h-2">
              <div className="h-2 rounded-full" style={{ width: `${Math.min(r.status.pct, 100)}%`, background: r.status.bar }} />
            </div>
          </div>
        ))}
      </div>

      {anyLow && (
        <div className="mt-4 bg-orange-50 border border-orange-200 rounded p-3 text-xs text-orange-800">
          <b>Tip:</b> Apply the recommended fertilizer to fix low nutrients before sowing.
        </div>
      )}

      {allGood && (
        <div className="mt-4 bg-green-50 border border-green-200 rounded p-3 text-xs text-green-800">
          Your soil is in excellent condition for {crop || "your crop"}.
        </div>
      )}
    </div>
  );
}

function formatTimestamp(value) {
  if (!value) return null;
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return value;
  return d.toLocaleString("en-IN", {
    day: "2-digit",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export default function InsightsPage() {
  const [calendarIdInput, setCalendarIdInput] = useState("");
  const [calendarInfo, setCalendarInfo] = useState(null);
  const [loadError, setLoadError] = useState("");
  const [loadingCal, setLoadingCal] = useState(false);
  const [insights, setInsights] = useState(null);
  const [loading, setLoading] = useState(false);

  const [soilInputs, setSoilInputs] = useState({
    nitrogen_level: 50,
    phosphorus_level: 40,
    potassium_level: 40,
    soil_ph: 6.5,
    soil_moisture: 40,
    soil_type: "Clay",
    region: "South",
  });

  useEffect(() => {
    const saved = JSON.parse(localStorage.getItem("lastCalendar") || "null");
    if (saved?.calendar_id) {
      setCalendarIdInput(String(saved.calendar_id));
      setCalendarInfo(saved);
      setSoilInputs({
        nitrogen_level: saved.nitrogen_level ?? 50,
        phosphorus_level: saved.phosphorus_level ?? 40,
        potassium_level: saved.potassium_level ?? 40,
        soil_ph: saved.soil_ph ?? 6.5,
        soil_moisture: saved.soil_moisture ?? 40,
        soil_type: saved.soil_type || "Clay",
        region: saved.region || "South",
      });
    }
  }, []);

  const loadCalendar = async (idOverride) => {
    const id = idOverride || calendarIdInput.trim();
    if (!id) {
      setLoadError("Please enter a Calendar ID.");
      return;
    }

    setLoadingCal(true);
    setLoadError("");
    setInsights(null);

    try {
      const res = await API.get(`/crop-calendar/${id}`);
      setCalendarInfo(res.data);
      setSoilInputs((prev) => ({
        ...prev,
        soil_type: res.data.soil_type || prev.soil_type,
        region: res.data.region || prev.region,
      }));
    } catch (err) {
      setLoadError(
        err.response?.status === 404
          ? `Calendar ID "${id}" not found.`
          : "Failed to load calendar. Make sure the backend is running."
      );
      setCalendarInfo(null);
    } finally {
      setLoadingCal(false);
    }
  };

  const generateInsights = async () => {
    if (!calendarInfo) {
      alert("Load a calendar first.");
      return;
    }

    try {
      setLoading(true);
      const res = await API.post("/insights/generate", {
        calendar_id: calendarInfo.calendar_id,
        ...soilInputs,
      });
      setInsights(res.data);
    } catch (err) {
      alert("Failed to generate insights.");
    } finally {
      setLoading(false);
    }
  };

  const riskValue = insights?.risk_score ?? 0;
  const rainfallSlots = insights?.rainfall_forecast_slots || [];

  const rainfallUsed = rainfallSlots.length > 0
    ? rainfallSlots.map((slot) => slot.display_rain_mm ?? 0)
    : (insights?.rainfall_forecast_mm || []);

  const rainfallActual = rainfallSlots.length > 0
    ? rainfallSlots.map((slot) => slot.actual_rain_mm ?? null)
    : (insights?.actual_rainfall_mm || []);

  const rainfallLabels = rainfallSlots.length > 0
    ? rainfallSlots.map((slot) => slot.label)
    : rainfallUsed.map((_, i) => `Slot ${i + 1}`);

  const fertRec = insights?.fertilizer_recommendation || null;
  const fertName = fertRec?.recommended_fertilizer || null;
  const fertConf = fertRec?.confidence ?? null;
  const yieldData = insights?.yield_prediction || null;
  const yieldQA = yieldData?.yield_quintals_per_acre ?? null;
  const yieldTha = yieldData?.yield_tonnes_per_ha ?? null;
  const isLive = insights?.weather_source === "live";
  const city = insights?.resolved_city || calendarInfo?.location || "";
  const displayMode = insights?.rainfall_display_mode || "ai_fallback";

  const totalUsedRain = rainfallUsed.reduce((a, b) => a + (Number(b) || 0), 0).toFixed(2);
  const totalActualRain = rainfallActual.reduce((a, b) => a + (Number(b) || 0), 0).toFixed(2);

  const hasActualSeries = rainfallActual.some((v) => v !== null && v !== undefined);

  const ic = "border border-gray-300 p-2 rounded w-full text-sm";
  const lc = "text-xs text-gray-500 mb-1 block font-medium";

  return (
    <div className="space-y-6 p-6">
      <h1 className="text-3xl font-bold">Insights & AI Suggestions</h1>

      <div className="bg-white p-5 rounded shadow space-y-3">
        <h2 className="font-semibold text-gray-700">
          Load Calendar <span className="text-xs text-gray-400 font-normal ml-1">- enter any Calendar ID to revisit insights</span>
        </h2>

        <div className="flex gap-3 flex-wrap">
          <input
            className="flex-1 min-w-[220px] border p-2 rounded text-sm"
            placeholder="Enter Calendar ID e.g. 3202031342"
            value={calendarIdInput}
            onChange={(e) => setCalendarIdInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && loadCalendar()}
          />
          <button
            onClick={() => loadCalendar()}
            disabled={loadingCal}
            className="bg-blue-600 hover:bg-blue-700 text-white px-5 py-2 rounded font-semibold text-sm"
          >
            {loadingCal ? "Loading..." : "Load Calendar"}
          </button>
        </div>

        {loadError && <p className="text-red-600 text-sm">{loadError}</p>}

        {calendarInfo && (
          <div className="bg-green-50 border border-green-200 rounded p-3 text-sm flex flex-wrap gap-x-5 gap-y-1">
            <span><b>{calendarInfo.crop}</b></span>
            <span>{calendarInfo.season}</span>
            <span>{calendarInfo.location}</span>
            {calendarInfo.farmer_name && <span>{calendarInfo.farmer_name}</span>}
            <span className="text-gray-400 text-xs">ID: {calendarInfo.calendar_id}</span>
          </div>
        )}
      </div>

      {calendarInfo && (
        <div className="bg-white p-5 rounded shadow">
          <h2 className="font-semibold text-gray-700 mb-3">
            Soil Details <span className="text-xs text-gray-400 font-normal ml-1">- adjust if needed</span>
          </h2>

          <div className="grid md:grid-cols-4 gap-3">
            <div>
              <label className={lc}>Nitrogen (N) kg/ha</label>
              <input type="number" className={ic} value={soilInputs.nitrogen_level} onChange={(e) => setSoilInputs({ ...soilInputs, nitrogen_level: +e.target.value })} />
            </div>
            <div>
              <label className={lc}>Phosphorus (P) kg/ha</label>
              <input type="number" className={ic} value={soilInputs.phosphorus_level} onChange={(e) => setSoilInputs({ ...soilInputs, phosphorus_level: +e.target.value })} />
            </div>
            <div>
              <label className={lc}>Potassium (K) kg/ha</label>
              <input type="number" className={ic} value={soilInputs.potassium_level} onChange={(e) => setSoilInputs({ ...soilInputs, potassium_level: +e.target.value })} />
            </div>
            <div>
              <label className={lc}>Soil pH</label>
              <input type="number" step="0.1" className={ic} value={soilInputs.soil_ph} onChange={(e) => setSoilInputs({ ...soilInputs, soil_ph: +e.target.value })} />
            </div>
            <div>
              <label className={lc}>Soil Type</label>
              <select className={ic} value={soilInputs.soil_type} onChange={(e) => setSoilInputs({ ...soilInputs, soil_type: e.target.value })}>
                <option>Clay</option>
                <option>Loamy</option>
                <option>Sandy</option>
                <option>Silt</option>
              </select>
            </div>
            <div>
              <label className={lc}>Soil Moisture (%)</label>
              <input type="number" className={ic} value={soilInputs.soil_moisture} onChange={(e) => setSoilInputs({ ...soilInputs, soil_moisture: +e.target.value })} />
            </div>
            <div className="md:col-span-2 flex items-end">
              <button
                onClick={generateInsights}
                disabled={loading}
                className="w-full bg-green-600 hover:bg-green-700 text-white px-6 py-2 rounded font-semibold"
              >
                {loading ? "Generating..." : "Generate Insights"}
              </button>
            </div>
          </div>
        </div>
      )}

      {!calendarInfo && !loadError && (
        <div className="bg-yellow-50 border border-yellow-200 p-4 rounded text-sm text-yellow-800">
          Enter a Calendar ID above or go to <b>Dashboard</b> to generate a new calendar first.
        </div>
      )}

      {insights && (
        <>
          <div className="bg-white p-4 rounded shadow text-sm flex flex-wrap gap-x-6 gap-y-1 items-center">
            <span><b>Health:</b> <span className={insights.overall_health === "Excellent" ? "text-green-600 font-semibold" : insights.overall_health === "Good" ? "text-blue-600 font-semibold" : insights.overall_health === "At Risk" ? "text-red-600 font-semibold" : "text-yellow-600 font-semibold"}>{insights.overall_health}</span></span>
            <span><b>Irrigation:</b> {insights.irrigation_advice}</span>
            <span><b>Pest Risk:</b> {insights.pest_risk}</span>
            <span className="ml-auto text-xs">
              {isLive ? <span className="text-green-600">Live - {city}</span> : <span className="text-yellow-600">Typical {calendarInfo?.season} weather</span>}
            </span>
          </div>

          <div className="grid lg:grid-cols-3 gap-6">
            <div className="bg-white p-5 rounded shadow">
              <h2 className="font-bold mb-2">Expected Yield</h2>
              {yieldQA !== null ? (
                <>
                  <div className="bg-green-50 border border-green-200 rounded p-4 text-center mb-3">
                    <p className="text-4xl font-bold text-green-700">{yieldQA}</p>
                    <p className="text-sm text-gray-600 mt-1">quintals / acre</p>
                    <p className="text-xs text-gray-400 mt-0.5">{yieldTha} tonnes/hectare</p>
                  </div>
                  <Bar
                    data={{
                      labels: ["Your Crop"],
                      datasets: [{
                        label: calendarInfo?.crop,
                        data: [yieldQA],
                        backgroundColor: "#22c55e",
                      }],
                    }}
                    options={{ responsive: true, scales: { y: { beginAtZero: true } } }}
                  />
                  <p className="text-xs text-gray-400 mt-2">
                    From 246K records of Indian govt crop production data (1997-2015)
                  </p>
                </>
              ) : (
                <p className="text-sm text-gray-400">Run train_models.py to enable yield prediction</p>
              )}
            </div>

            <div className="bg-white p-5 rounded shadow">
              <h2 className="font-bold mb-3">Crop Risk Score</h2>
              <div className={`text-center p-5 rounded-lg mb-3 ${riskValue < 35 ? "bg-green-50" : riskValue < 60 ? "bg-yellow-50" : "bg-red-50"}`}>
                <p className={`text-5xl font-bold ${riskValue < 35 ? "text-green-700" : riskValue < 60 ? "text-yellow-600" : "text-red-600"}`}>{riskValue}</p>
                <p className="text-sm text-gray-500 mt-1">out of 100</p>
                <p className={`text-sm font-semibold mt-1 ${riskValue < 35 ? "text-green-700" : riskValue < 60 ? "text-yellow-600" : "text-red-600"}`}>{insights.overall_health}</p>
              </div>
              <p className="text-xs text-gray-400">Calculated from weather humidity, temperature stress, potassium deficiency, and soil pH</p>
            </div>

            <div className="bg-white p-5 rounded shadow border-l-4 border-green-600">
              <h2 className="font-bold mb-3">Fertilizer to Apply Now</h2>
              {fertName ? (
                <div className="space-y-3">
                  <div className="bg-green-50 border border-green-200 rounded p-4 text-center">
                    <p className="text-3xl font-bold text-green-700">{fertName}</p>
                    {fertConf !== null && <p className="text-xs text-gray-500 mt-1">AI confidence: <b>{fertConf}%</b></p>}
                  </div>
                  <p className="text-xs text-gray-500">
                    Recommended for <b>{calendarInfo?.crop}</b> in Vegetative stage based on {soilInputs.soil_type} soil and {calendarInfo?.season} season.
                  </p>
                  <div className="bg-blue-50 border border-blue-200 rounded p-3 text-xs text-blue-800">
                    Apply in small doses every 2 weeks. Check your crop calendar stages for phase-specific quantities.
                  </div>
                </div>
              ) : (
                <p className="text-sm text-gray-500">No recommendation available</p>
              )}
            </div>
          </div>

          <SoilHealthCard
            n={soilInputs.nitrogen_level}
            p={soilInputs.phosphorus_level}
            k={soilInputs.potassium_level}
            ph={soilInputs.soil_ph}
            crop={calendarInfo?.crop}
          />

          <div className="bg-white p-6 rounded shadow">
            <div className="flex justify-between items-center mb-4">
              <h2 className="font-bold">Rainfall Forecast (Next {rainfallLabels.length} x 3h Slots)</h2>
              <span className={`text-xs font-medium px-2 py-1 rounded ${displayMode === "weather_api_primary" ? "bg-green-100 text-green-700" : "bg-yellow-100 text-yellow-700"}`}>
                {displayMode === "weather_api_primary" ? "Weather API" : "Fallback"}
              </span>
            </div>

            <div className="mb-4 text-xs text-gray-600 bg-blue-50 border border-blue-200 rounded p-3">
              {displayMode === "weather_api_primary"
                ? "This chart now shows only Weather API rainfall values."
                : "Live API rainfall is not available for this calendar, so fallback values are shown."}
            </div>

            {rainfallUsed.length > 0 ? (
              <>
                <Line
                  data={{
                    labels: rainfallLabels,
                    datasets: [
                      {
                        label: "Rainfall from API (mm)",
                        data: rainfallUsed,
                        borderColor: "#16a34a",
                        backgroundColor: "rgba(22,163,74,0.12)",
                        fill: true,
                        tension: 0.35,
                        pointRadius: 5,
                        pointBackgroundColor: "#16a34a",
                      },
                    ],
                  }}
                  options={{
                    responsive: true,
                    scales: {
                      y: {
                        beginAtZero: true,
                        title: { display: true, text: "mm" },
                      },
                    },
                  }}
                />

                <div className="mt-3 text-xs text-gray-500 space-y-1">
                  <p><b>Total rainfall:</b> {totalUsedRain} mm</p>
                  {hasActualSeries && <p><b>API total:</b> {totalActualRain} mm</p>}
                  <p>Each slot = 3 hours.</p>
                </div>

                {rainfallSlots.length > 0 && (
                  <div className="mt-4 overflow-x-auto">
                    <table className="min-w-full text-xs border rounded">
                      <thead className="bg-gray-50">
                        <tr>
                          <th className="text-left px-3 py-2 border-b">Slot</th>
                          <th className="text-left px-3 py-2 border-b">Time</th>
                          <th className="text-left px-3 py-2 border-b">Rainfall</th>
                          <th className="text-left px-3 py-2 border-b">Source</th>
                        </tr>
                      </thead>
                      <tbody>
                        {rainfallSlots.map((slot) => (
                          <tr key={slot.slot} className="border-b last:border-b-0">
                            <td className="px-3 py-2">{slot.label}</td>
                            <td className="px-3 py-2">{formatTimestamp(slot.timestamp) || "N/A"}</td>
                            <td className="px-3 py-2">{slot.display_rain_mm ?? "N/A"} mm</td>
                            <td className="px-3 py-2">{slot.source === "weather_api" ? "Weather API" : "Fallback"}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </>
            ) : (
              <p className="text-sm text-gray-400">No rainfall data available.</p>
            )}
          </div>

          <div className="bg-white p-6 rounded shadow">
            <h2 className="font-bold mb-3">AI Alerts & Recommendations</h2>
            <ul className="space-y-2">
              {insights.alerts?.map((a, i) => (
                <li key={i} className="flex items-start gap-2 text-sm">
                  <span className="text-green-600 font-bold mt-0.5">•</span>
                  <span>{a}</span>
                </li>
              ))}
            </ul>
          </div>
        </>
      )}
    </div>
  );
}
