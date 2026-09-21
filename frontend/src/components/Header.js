import { useEffect, useState } from "react";

export default function Header() {
  const [farmerName, setFarmerName] = useState("Guest");

  const readName = () => {
    const saved = JSON.parse(localStorage.getItem("lastCalendar") || "null");
    if (saved?.farmer_name) setFarmerName(saved.farmer_name);
  };

  useEffect(() => {
    readName();
    window.addEventListener("storage", readName);
    return () => window.removeEventListener("storage", readName);
  }, []);

  return (
    <div className="bg-white shadow p-4 flex justify-between items-center">
      <h2 className="text-xl font-semibold">AI Crop Calendar</h2>
      <div>
        <span className="text-gray-500 text-sm">Farmer: </span>
        <span className="font-semibold text-gray-800">{farmerName}</span>
      </div>
    </div>
  );
}