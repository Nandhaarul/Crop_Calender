import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import DashboardLayout from "./components/DashboardLayout";

import Dashboard from "./pages/Dashboard";
import CalendarPage from "./pages/CalendarPage";
import CalendarDetails from "./pages/CalendarDetails";
import ProgressPage from "./pages/ProgressPage";
import InsightsPage from "./pages/InsightsPage";

function App() {
  return (
    <Router>
      <Routes>

        {/* Layout Route */}
        <Route path="/" element={<DashboardLayout />}>
          <Route index element={<Dashboard />} />
          <Route path="calendar" element={<CalendarPage />} />
          <Route path="calendar/:id" element={<CalendarDetails />} />
          <Route path="progress" element={<ProgressPage />} />
          <Route path="insights" element={<InsightsPage />} />
        </Route>

      </Routes>
    </Router>
  );
}

export default App;
