import React, { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import API from "../services/api";
import { Bar } from "react-chartjs-2";
import {
  Chart as ChartJS, BarElement, CategoryScale,
  LinearScale, Tooltip, Legend,
} from "chart.js";

ChartJS.register(BarElement, CategoryScale, LinearScale, Tooltip, Legend);

const FERTILIZER_NPK = {
  Urea: { Nitrogen: 46, Phosphorus: 0, Potassium: 0 },
  DAP: { Nitrogen: 18, Phosphorus: 46, Potassium: 0 },
  MOP: { Nitrogen: 0, Phosphorus: 0, Potassium: 60 },
  NPK: { Nitrogen: 17, Phosphorus: 17, Potassium: 17 },
  SSP: { Nitrogen: 0, Phosphorus: 16, Potassium: 0 },
  Compost: { Nitrogen: 2, Phosphorus: 1, Potassium: 1 },
  "Zinc Sulphate": { Nitrogen: 0, Phosphorus: 0, Potassium: 0 },
};

function formatDate(value) {
  if (!value) return "N/A";
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return value;
  return d.toLocaleDateString("en-IN", {
    day: "2-digit",
    month: "short",
    year: "numeric",
  });
}

function formatWeatherSource(source) {
  if (source === "live") return "Live weather";
  if (source === "request_defaults") return "Fallback request values";
  if (source === "seasonal_defaults") return "Seasonal defaults";
  return source || "Unknown";
}

function formatPct(value) {
  if (value === null || value === undefined) return "N/A";
  return `${value}%`;
}

export default function CalendarDetails() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [calendar, setCalendar] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    API.get(`/crop-calendar/${id}`)
      .then((res) => setCalendar(res.data))
      .catch(() => setError("Calendar not found or server error."))
      .finally(() => setLoading(false));
  }, [id]);

  if (loading) return <p className="p-6 text-gray-500">Loading calendar...</p>;

  if (error) {
    return (
      <div className="p-6">
        <p className="text-red-600 mb-4">{error}</p>
        <button
          onClick={() => navigate("/calendar")}
          className="bg-green-600 text-white px-4 py-2 rounded"
        >
          Back to Calendars
        </button>
      </div>
    );
  }

  const stages = Array.isArray(calendar?.calendar) ? calendar.calendar : [];
  const weatherContext = calendar?.weather_context || {};
  const weatherInputsUsed = calendar?.weather_inputs_used || {};
  const cropFitAnalysis = calendar?.crop_fit_analysis || null;

  return (
    <div className="p-6 space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold">Calendar Details</h1>
        <button
          onClick={() => navigate("/calendar")}
          className="bg-gray-200 hover:bg-gray-300 px-4 py-2 rounded text-sm"
        >
          Back
        </button>
      </div>

      <div className="bg-white p-5 rounded shadow grid md:grid-cols-2 gap-3 text-sm">
        <div><b>Farmer:</b> {calendar.farmer_name}</div>
        <div><b>Crop:</b> {calendar.crop}</div>
        <div><b>Season:</b> {calendar.season}</div>
        <div><b>Location:</b> {calendar.location}</div>
        <div><b>Soil Type:</b> {calendar.soil_type || "N/A"}</div>
        <div><b>Calendar ID:</b> {calendar.calendar_id}</div>
        <div><b>Sowing Date:</b> {formatDate(calendar.sowing_date)}</div>
        <div><b>Expected Harvest:</b> {formatDate(calendar.expected_harvest_date)}</div>
        <div><b>Region:</b> {calendar.region || "N/A"}</div>
        <div><b>Weather Source:</b> {formatWeatherSource(weatherContext.weather_source)}</div>
        <div><b>Resolved City:</b> {weatherContext.resolved_city || calendar.location}</div>
        {calendar.created_at && (
          <div><b>Generated:</b> {new Date(calendar.created_at).toLocaleString()}</div>
        )}
      </div>

      <div className="bg-white p-5 rounded shadow">
        <h2 className="text-lg font-semibold mb-3">Weather Used For Generation</h2>
        <div className="grid md:grid-cols-3 gap-3 text-sm">
          <div><b>Temperature:</b> {weatherInputsUsed.temperature ?? "N/A"} deg C</div>
          <div><b>Humidity:</b> {weatherInputsUsed.humidity ?? "N/A"} %</div>
          <div><b>Rainfall:</b> {weatherInputsUsed.rainfall ?? "N/A"} mm</div>
        </div>
      </div>

      {cropFitAnalysis && (
        <div className="bg-white p-5 rounded shadow space-y-3">
          <div className="flex items-center justify-between gap-3 flex-wrap">
            <h2 className="text-lg font-semibold">Crop Fit Analysis</h2>
            <span className={`text-xs font-semibold px-2 py-1 rounded ${
              cropFitAnalysis.available
                ? "bg-green-100 text-green-800"
                : "bg-amber-100 text-amber-800"
            }`}>
              {cropFitAnalysis.available ? "Model Available" : "Limited / Unavailable"}
            </span>
          </div>

          <div className="grid md:grid-cols-2 gap-3 text-sm">
            <div><b>Selected Crop:</b> {cropFitAnalysis.selected_crop || "N/A"}</div>
            <div><b>Recommended Crop:</b> {cropFitAnalysis.recommended_crop || "N/A"}</div>
            <div><b>Selected Crop Probability:</b> {formatPct(cropFitAnalysis.selected_crop_probability_pct)}</div>
            <div><b>Recommended Crop Probability:</b> {formatPct(cropFitAnalysis.recommended_crop_probability_pct)}</div>
          </div>

          {cropFitAnalysis.message && (
            <div className={`text-sm rounded-lg px-3 py-2 ${
              cropFitAnalysis.available
                ? "bg-green-50 text-green-700 border border-green-100"
                : "bg-amber-50 text-amber-700 border border-amber-100"
            }`}>
              {cropFitAnalysis.message}
            </div>
          )}

          {Array.isArray(cropFitAnalysis.top_alternatives) && cropFitAnalysis.top_alternatives.length > 0 && (
            <div>
              <h3 className="text-sm font-semibold text-gray-800 mb-2">Top Suggestions</h3>
              <div className="grid md:grid-cols-3 gap-2">
                {cropFitAnalysis.top_alternatives.map((item, index) => (
                  <div key={`${item.crop}-${index}`} className="border rounded-lg p-3 text-sm bg-gray-50">
                    <div className="font-semibold text-gray-800">{item.crop}</div>
                    <div className="text-gray-500">{formatPct(item.probability)}</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {Array.isArray(cropFitAnalysis.model_supported_crops) && cropFitAnalysis.model_supported_crops.length > 0 && (
            <p className="text-xs text-gray-500">
              Supported by current crop-fit model: {cropFitAnalysis.model_supported_crops.join(", ")}
            </p>
          )}
        </div>
      )}

      <h2 className="text-xl font-semibold">Growth Stages</h2>

      <div className="grid md:grid-cols-2 gap-6">
        {stages.map((stage, i) => {
          const fertRec = stage.fertilizer_recommendation || {};
          const fertName = fertRec.recommended_fertilizer || "NPK";
          const confidence = fertRec.model_confidence_pct || 0;
          const npk = FERTILIZER_NPK[fertName] || { Nitrogen: 17, Phosphorus: 17, Potassium: 17 };
          const irrigation = stage.irrigation_advice || {};
          const pest = stage.pest_disease_risk || {};

          return (
            <div key={i} className="bg-white shadow rounded-xl p-5 border">
              <div className="flex justify-between items-center mb-1">
                <h3 className="text-lg font-bold text-green-700">{stage.stage}</h3>
                <span className="text-xs text-gray-400">{stage.duration_days} days</span>
              </div>

              <div className="text-sm text-gray-600 mb-3 space-y-1">
                <div><b>Start:</b> {formatDate(stage.start_date)}</div>
                <div><b>End:</b> {formatDate(stage.end_date)}</div>
              </div>

              <div className="flex items-center gap-2 mb-3">
                <span className="bg-green-100 text-green-800 text-xs font-semibold px-2 py-1 rounded">
                  {fertName}
                </span>
                {confidence > 0 && (
                  <span className="text-xs text-gray-400">{confidence}% confidence</span>
                )}
              </div>

              <Bar
                data={{
                  labels: ["Nitrogen", "Phosphorus", "Potassium"],
                  datasets: [{
                    label: "NPK Content (%)",
                    data: [npk.Nitrogen, npk.Phosphorus, npk.Potassium],
                    backgroundColor: ["#22c55e", "#3b82f6", "#f59e0b"],
                  }],
                }}
                options={{
                  responsive: true,
                  scales: { y: { beginAtZero: true, max: 60 } },
                  plugins: { legend: { display: true } },
                }}
              />

              <div className="mt-3 text-sm text-gray-700 space-y-1">
                <div><b>Water:</b> {irrigation.predicted_water_level ?? "N/A"} mm</div>
                <div className="text-gray-500">Frequency: {irrigation.frequency ?? "N/A"}</div>
                {irrigation.weather_adjustment && (
                  <div className="text-xs text-blue-600">
                    <b>Weather note:</b> {irrigation.weather_adjustment}
                  </div>
                )}
              </div>

              <div className="mt-2 text-sm">
                <b>Pest Risk:</b>{" "}
                <span className={
                  pest.risk_level === "High" ? "text-red-600 font-semibold" :
                  pest.risk_level === "Medium" ? "text-yellow-600 font-semibold" :
                  "text-green-600 font-semibold"
                }>
                  {pest.risk_level ?? "N/A"}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
