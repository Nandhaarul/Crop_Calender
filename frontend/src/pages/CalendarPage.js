import React, { useEffect, useState } from "react";
import API from "../services/api";

export default function CropCalendarPage() {
  const [calendars, setCalendars] = useState([]);

  useEffect(() => {
    API.get("/crop-calendar/all")
      .then(res => setCalendars(res.data))
      .catch(err => console.error(err));
  }, []);

  return (
    <div>
      <h1 className="text-3xl font-bold mb-6">📅 Saved Crop Calendars</h1>

      {calendars.length === 0 && <p>No calendars generated yet.</p>}

      <div className="grid md:grid-cols-2 gap-6">
        {calendars.map((c, i) => (
          <div key={i} className="bg-white p-4 rounded shadow">
            <p><b>Calendar ID:</b> {c.calendar_id}</p>
            <p>🌾 Crop: {c.crop}</p>
            <p>🌱 Season: {c.season}</p>
            <p>📍 Location: {c.location}</p>
            <p>👨‍🌾 Farmer: {c.farmer_name}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
