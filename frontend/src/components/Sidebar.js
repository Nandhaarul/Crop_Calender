import { Link, useLocation } from "react-router-dom";

export default function Sidebar() {
  const location = useLocation();

  const linkClass = (path) =>
    `block px-4 py-2 rounded ${
      location.pathname === path
        ? "bg-green-600 font-semibold text-white"
        : "hover:bg-green-700 text-white"
    }`;

  return (
    <div className="w-60 bg-green-800 text-white h-screen p-4">
      <h1 className="text-2xl font-bold mb-6">🌾 Agro AI</h1>

      <nav className="space-y-3">
        <Link to="/" className={linkClass("/")}>Dashboard</Link>
        <Link to="/calendar" className={linkClass("/calendar")}>Crop Calendar</Link>
        <Link to="/progress" className={linkClass("/progress")}>Progress</Link>
        <Link to="/insights" className={linkClass("/insights")}>Insights</Link>
      </nav>
    </div>
  );
}
