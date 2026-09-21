import React, { useState } from "react";
import { Bar } from "react-chartjs-2";
import {
  Chart as ChartJS, BarElement, CategoryScale,
  LinearScale, Tooltip, Legend,
} from "chart.js";
import API from "../services/api";

ChartJS.register(BarElement, CategoryScale, LinearScale, Tooltip, Legend);

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

export default function Dashboard() {
  const [calendarResult, setCalendarResult] = useState(null);
  const [calendarId, setCalendarId] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);

  const [formData, setFormData] = useState({
    farmer_name: "",
    crop: "",
    season: "",
    sowing_date: "",
    location: "",
    soil_type: "",
    nitrogen_level: 50,
    phosphorus_level: 40,
    potassium_level: 40,
    soil_ph: 6.5,
    soil_moisture: 40,
  });

  const handleChange = (e) => {
    const val = e.target.type === "number" ? +e.target.value : e.target.value;
    setFormData({ ...formData, [e.target.name]: val });
  };

  const generateCalendar = async () => {
    if (!formData.crop || !formData.season || !formData.sowing_date || !formData.location || !formData.farmer_name) {
      alert("Please fill in all required fields.");
      return;
    }

    try {
      setLoading(true);
      const response = await API.post("/crop-calendar/generate", formData);
      const data = response.data;

      localStorage.setItem("lastCalendar", JSON.stringify({
        calendar_id: data.calendar_id,
        crop: data.crop,
        season: data.season,
        location: data.location,
        farmer_name: formData.farmer_name,
        soil_type: formData.soil_type,
        region: data.region,
        nitrogen_level: formData.nitrogen_level,
        phosphorus_level: formData.phosphorus_level,
        potassium_level: formData.potassium_level,
        soil_ph: formData.soil_ph,
        soil_moisture: formData.soil_moisture,
      }));
      localStorage.setItem("lastCalendarFull", JSON.stringify(data));
      window.dispatchEvent(new Event("storage"));

      setCalendarResult(data);
      setCalendarId(data.calendar_id || null);
    } catch (error) {
      console.error("Error generating calendar:", error);
      alert(error.response?.data?.detail || "Backend connection failed. Make sure the server is running.");
    } finally {
      setLoading(false);
    }
  };

  const stages = Array.isArray(calendarResult?.calendar) ? calendarResult.calendar : [];
  const weatherContext = calendarResult?.weather_context || {};
  const weatherInputsUsed = calendarResult?.weather_inputs_used || {};
  const cropFitAnalysis = calendarResult?.crop_fit_analysis || null;

  const inputClass = "border border-gray-300 p-2 rounded w-full text-sm focus:outline-none focus:ring-2 focus:ring-green-400";
  const labelClass = "text-xs text-gray-500 mb-1 block font-medium";

  return (
    <div className="space-y-6 p-6">
      <h1 className="text-3xl font-bold text-gray-800">AI Crop Calendar</h1>

      <div className="bg-white p-6 rounded-xl shadow space-y-4">
        <h2 className="text-lg font-semibold text-green-700">Enter Your Farm Details</h2>

        <div className="grid md:grid-cols-3 gap-4">
          <div>
            <label className={labelClass}>Farmer Name *</label>
            <input name="farmer_name" placeholder="e.g. Arjun Kumar" onChange={handleChange} className={inputClass} />
          </div>

          <div>
            <label className={labelClass}>Crop *</label>
            <select name="crop" onChange={handleChange} className={inputClass}>
              <option value="">Select crop...</option>
              <option>Rice</option>
              <option>Wheat</option>
              <option>Maize</option>
              <option>Cotton</option>
              <option>Sugarcane</option>
              <option>Potato</option>
              <option>Tomato</option>
            </select>
            {formData.crop === "Potato" && (
              <p className="text-xs text-amber-600 mt-1">
                Crop-fit analysis is currently available for Rice, Wheat, Maize, Cotton, Sugarcane, and Tomato. Potato is excluded because it is not represented in the training data yet.
              </p>
            )}
          </div>

          <div>
            <label className={labelClass}>Season *</label>
            <select name="season" onChange={handleChange} className={inputClass}>
              <option value="">Select season...</option>
              <option>Kharif</option>
              <option>Rabi</option>
              <option>Zaid</option>
            </select>
          </div>

          <div>
            <label className={labelClass}>Sowing Date *</label>
            <input type="date" name="sowing_date" onChange={handleChange} className={inputClass} />
          </div>

          <div>
            <label className={labelClass}>Location (City / District) *</label>
            <input name="location" placeholder="e.g. Chennai, Punjab, Coimbatore" onChange={handleChange} className={inputClass} />
            <p className="text-xs text-gray-400 mt-0.5">Region will be detected automatically</p>
          </div>

          <div>
            <label className={labelClass}>Soil Type</label>
            <select name="soil_type" onChange={handleChange} className={inputClass}>
              <option value="">Select soil...</option>
              <option>Clay</option>
              <option>Loamy</option>
              <option>Sandy</option>
              <option>Silt</option>
            </select>
          </div>
        </div>

        <div>
          <button
            type="button"
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="text-sm text-green-600 hover:text-green-800 font-medium flex items-center gap-1"
          >
            {showAdvanced ? "Hide" : "Add"} Soil Test Values
            <span className="text-gray-400 font-normal">(optional - improves AI accuracy)</span>
          </button>

          {showAdvanced && (
            <div className="grid md:grid-cols-4 gap-4 mt-3 p-4 bg-green-50 rounded-lg border border-green-100">
              <div>
                <label className={labelClass}>Nitrogen (N) kg/ha</label>
                <input type="number" name="nitrogen_level" defaultValue={50} onChange={handleChange} className={inputClass} />
              </div>
              <div>
                <label className={labelClass}>Phosphorus (P) kg/ha</label>
                <input type="number" name="phosphorus_level" defaultValue={40} onChange={handleChange} className={inputClass} />
              </div>
              <div>
                <label className={labelClass}>Potassium (K) kg/ha</label>
                <input type="number" name="potassium_level" defaultValue={40} onChange={handleChange} className={inputClass} />
              </div>
              <div>
                <label className={labelClass}>Soil pH (3.5-9.9)</label>
                <input type="number" step="0.1" name="soil_ph" defaultValue={6.5} onChange={handleChange} className={inputClass} />
              </div>
              <div className="md:col-span-4">
                <p className="text-xs text-gray-400">Get these from a soil test kit or your nearest Krishi Vigyan Kendra.</p>
              </div>
            </div>
          )}
        </div>

        <button
          onClick={generateCalendar}
          className="w-full bg-green-600 hover:bg-green-700 text-white font-semibold py-3 rounded-lg transition text-base"
        >
          {loading ? "Generating your crop calendar..." : "Generate AI Crop Calendar"}
        </button>
      </div>

      {calendarId && (
        <div className="bg-green-50 border border-green-200 p-4 rounded-lg flex items-center justify-between flex-wrap gap-2">
          <div>
            <p className="text-sm font-semibold text-green-800">Calendar Generated Successfully</p>
            <p className="text-xs text-green-600 mt-0.5">
              Calendar ID: <b>{calendarId}</b> - saved to database
            </p>
          </div>
          <p className="text-xs text-gray-500">
            Go to <b>Insights</b> for AI analysis or <b>Progress</b> to track crop stages
          </p>
        </div>
      )}

      {calendarResult && (
        <div className="bg-white shadow p-4 rounded-lg grid md:grid-cols-3 gap-3 text-sm">
          <div><b>Farmer:</b> {calendarResult.farmer_name}</div>
          <div><b>Crop:</b> {calendarResult.crop}</div>
          <div><b>Season:</b> {calendarResult.season}</div>
          <div><b>Location:</b> {calendarResult.location}</div>
          <div><b>Sowing Date:</b> {formatDate(calendarResult.sowing_date)}</div>
          <div><b>Expected Harvest:</b> {formatDate(calendarResult.expected_harvest_date)}</div>
          <div><b>Region:</b> {calendarResult.region}</div>
          <div><b>Weather Source:</b> {formatWeatherSource(weatherContext.weather_source)}</div>
          <div><b>Resolved City:</b> {weatherContext.resolved_city || calendarResult.location}</div>
        </div>
      )}

      {calendarResult && (
        <div className="bg-white shadow p-5 rounded-lg">
          <h2 className="text-lg font-semibold mb-3">Weather Used For Generation</h2>
          <div className="grid md:grid-cols-3 gap-3 text-sm">
            <div><b>Temperature:</b> {weatherInputsUsed.temperature ?? "N/A"} deg C</div>
            <div><b>Humidity:</b> {weatherInputsUsed.humidity ?? "N/A"} %</div>
            <div><b>Rainfall:</b> {weatherInputsUsed.rainfall ?? "N/A"} mm</div>
          </div>
        </div>
      )}

      {cropFitAnalysis && (
        <div className="bg-white shadow p-5 rounded-lg space-y-3">
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

      {stages.length > 0 && (
        <div>
          <h2 className="text-2xl font-semibold mb-4">Your Crop Calendar</h2>
          <div className="grid md:grid-cols-2 gap-6">
            {stages.map((stage, index) => {
              const fertRec = stage.fertilizer_recommendation || {};
              const npk = fertRec.npk_content_pct || { Nitrogen: 0, Phosphorus: 0, Potassium: 0 };
              const fertName = fertRec.recommended_fertilizer || "N/A";
              const fertConf = fertRec.model_confidence_pct || 0;
              const irrigation = stage.irrigation_advice || { predicted_water_level: 0, frequency: "N/A" };
              const pest = stage.pest_disease_risk || { risk_level: "N/A" };

              return (
                <div key={index} className="bg-white shadow-md rounded-xl p-5 border">
                  <div className="flex justify-between items-center mb-1">
                    <h3 className="text-xl font-bold text-green-700">{stage.stage}</h3>
                    <span className="text-xs text-gray-400 bg-gray-100 px-2 py-0.5 rounded">{stage.duration_days} days</span>
                  </div>

                  <div className="text-sm text-gray-600 mb-3 space-y-1">
                    <div><b>Start:</b> {formatDate(stage.start_date)}</div>
                    <div><b>End:</b> {formatDate(stage.end_date)}</div>
                  </div>

                  <div className="mb-3 flex items-center gap-2">
                    <span className="bg-green-100 text-green-800 text-xs font-semibold px-2 py-1 rounded">{fertName}</span>
                    <span className="text-xs text-gray-400">{fertConf}% confidence</span>
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
                      plugins: { legend: { display: true } },
                      scales: { y: { beginAtZero: true, max: 60, title: { display: true, text: "%" } } },
                    }}
                  />

                  <div className="mt-3 text-sm text-gray-700 space-y-1">
                    <div><b>Water:</b> {irrigation.predicted_water_level} mm</div>
                    <div className="text-gray-500 text-xs">Frequency: {irrigation.frequency}</div>
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
                      {pest.risk_level}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
