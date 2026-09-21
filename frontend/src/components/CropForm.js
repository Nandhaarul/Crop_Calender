import { useState } from "react";
import axios from "axios";

function CropForm({ setCalendarData }) {
  const [formData, setFormData] = useState({
    crop: "",
    season: "",
    sowing_date: "",
    location: "",
    soil_type: "",
    farmer_name: "",
  });

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post(
        "http://127.0.0.1:8000/crop-calendar/generate",
        formData
      );
      setCalendarData(response.data.calendar);

      // Save to localStorage so Header + InsightsPage can read farmer name
      localStorage.setItem("lastCalendar", JSON.stringify({
        calendar_id: response.data.calendar_id,
        crop:         response.data.crop,
        season:       response.data.season,
        location:     response.data.location,
        farmer_name:  formData.farmer_name,
      }));
      localStorage.setItem("lastCalendarFull",
        JSON.stringify(response.data.calendar));

    } catch (error) {
      console.error("Error generating calendar:", error);
      alert("Failed to generate calendar. Check backend is running.");
    }
  };

  const inputClass = "border border-gray-300 p-2 rounded w-full text-sm focus:outline-none focus:ring-2 focus:ring-green-500";

  return (
    <form onSubmit={handleSubmit} className="bg-white p-6 rounded-xl shadow space-y-4">
      <h2 className="text-xl font-bold text-green-700">🌾 Generate Crop Calendar</h2>

      <div className="grid md:grid-cols-2 gap-4">
        <div>
          <label className="text-xs text-gray-500 mb-1 block">Farmer Name</label>
          <input name="farmer_name" placeholder="e.g. Arjun" onChange={handleChange} required className={inputClass} />
        </div>

        <div>
          <label className="text-xs text-gray-500 mb-1 block">Crop</label>
          <select name="crop" onChange={handleChange} required className={inputClass}>
            <option value="">Select crop...</option>
            <option>Rice</option>
            <option>Wheat</option>
            <option>Maize</option>
            <option>Cotton</option>
            <option>Sugarcane</option>
          </select>
        </div>

        <div>
          <label className="text-xs text-gray-500 mb-1 block">Season</label>
          <select name="season" onChange={handleChange} required className={inputClass}>
            <option value="">Select season...</option>
            <option>Kharif</option>
            <option>Rabi</option>
            <option>Zaid</option>
          </select>
        </div>

        <div>
          <label className="text-xs text-gray-500 mb-1 block">Soil Type</label>
          <select name="soil_type" onChange={handleChange} className={inputClass}>
            <option value="">Select soil type...</option>
            <option>Clay</option>
            <option>Loamy</option>
            <option>Sandy</option>
            <option>Silt</option>
          </select>
        </div>

        <div>
          <label className="text-xs text-gray-500 mb-1 block">Sowing Date</label>
          <input type="date" name="sowing_date" onChange={handleChange} required className={inputClass} />
        </div>

        <div>
          <label className="text-xs text-gray-500 mb-1 block">Location</label>
          <input name="location" placeholder="e.g. Chennai" onChange={handleChange} required className={inputClass} />
        </div>
      </div>

      <button type="submit"
        className="w-full bg-green-600 hover:bg-green-700 text-white font-semibold py-2 rounded transition">
        Generate AI Calendar
      </button>
    </form>
  );
}

export default CropForm;