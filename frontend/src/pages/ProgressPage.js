import React, { useState, useEffect } from "react";
import API from "../services/api";

const STAGES = ["Sowing", "Vegetative", "Flowering", "Harvest"];

export default function ProgressPage() {
  const [calendarIdInput, setCalendarIdInput] = useState("");
  const [calendarInfo, setCalendarInfo] = useState(null);
  const [calendarStages, setCalendarStages] = useState({});
  const [loadError, setLoadError] = useState("");
  const [loadingCal, setLoadingCal] = useState(false);

  const [stage, setStage] = useState("Sowing");
  const [progress, setProgress] = useState(0);
  const [notes, setNotes] = useState("");
  const [history, setHistory] = useState([]);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    const saved = JSON.parse(localStorage.getItem("lastCalendar") || "null");
    if (saved?.calendar_id) {
      setCalendarIdInput(String(saved.calendar_id));
      loadCalendar(String(saved.calendar_id));
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
    setCalendarInfo(null);
    setHistory([]);

    try {
      const calRes = await API.get(`/crop-calendar/${id}`);
      const cal = calRes.data;
      setCalendarInfo(cal);

      if (Array.isArray(cal.calendar)) {
        const map = {};
        cal.calendar.forEach((s) => { map[s.stage] = s; });
        setCalendarStages(map);
      }

      const progRes = await API.get(`/crop-calendar/progress/${id}`);
      setHistory(progRes.data.history || []);
    } catch (err) {
      if (err.response?.status === 404) {
        setLoadError(`Calendar ID "${id}" not found. Check the ID and try again.`);
      } else {
        setLoadError("Failed to load calendar. Make sure the backend is running.");
      }
    } finally {
      setLoadingCal(false);
    }
  };

  const updateProgress = async () => {
    if (!calendarInfo) {
      alert("Load a calendar first.");
      return;
    }

    setSaving(true);
    try {
      await API.post("/crop-calendar/progress/update", {
        calendar_id: calendarInfo.calendar_id,
        stage,
        progress_percent: Number(progress),
        notes,
      });

      const progRes = await API.get(`/crop-calendar/progress/${calendarInfo.calendar_id}`);
      setHistory(progRes.data.history || []);
      setNotes("");
    } catch (err) {
      const message = err.response?.data?.detail || "Failed to save progress.";
      alert(message);
    } finally {
      setSaving(false);
    }
  };

  const currentStageData = calendarStages[stage];
  const backendFertilizer = currentStageData?.fertilizer_recommendation?.recommended_fertilizer;
  const fertConf = currentStageData?.fertilizer_recommendation?.model_confidence_pct;
  const FERT_FALLBACK = { Sowing: "DAP", Vegetative: "Urea", Flowering: "Potash", Harvest: "None" };
  const fertDisplay = backendFertilizer
    ? `${backendFertilizer} (AI recommended)`
    : FERT_FALLBACK[stage];

  const harvestETA = progress >= 80 ? "7-10 days remaining" : "Estimated 3-4 weeks";
  const yieldStatus = progress < 50
    ? "Crop growth may be behind - consider checking irrigation"
    : "Yield potential looks stable";

  const latestForStage = history.find((h) => h.stage === stage);

  const latestProgressByStage = STAGES.reduce((acc, stageName) => {
    const row = history.find((h) => h.stage === stageName);
    acc[stageName] = row?.progress_percent ?? 0;
    return acc;
  }, {});

  const selectedStageIndex = STAGES.indexOf(stage);
  const previousStage = selectedStageIndex > 0 ? STAGES[selectedStageIndex - 1] : null;
  const previousStageProgress = previousStage ? latestProgressByStage[previousStage] : 100;
  const currentStageSavedProgress = latestProgressByStage[stage] || 0;

  const validationHint = previousStage && previousStageProgress < 100
    ? `Complete ${previousStage} to 100% before updating ${stage}.`
    : currentStageSavedProgress > 0
      ? `Latest saved progress for ${stage}: ${currentStageSavedProgress}%.`
      : `You can start logging progress for ${stage}.`;

  return (
    <div className="p-6 space-y-6">
      <h1 className="text-3xl font-bold">Crop Progress Tracker</h1>

      <div className="bg-white p-5 rounded shadow">
        <h2 className="font-semibold text-gray-700 mb-3">
          Load Your Calendar
          <span className="text-xs text-gray-400 font-normal ml-2">
            - enter your Calendar ID to track or continue logging progress
          </span>
        </h2>
        <div className="flex gap-3 items-start flex-wrap">
          <div className="flex-1 min-w-[220px]">
            <input
              className="border p-2 w-full rounded text-sm"
              placeholder="Enter Calendar ID e.g. 3202031342"
              value={calendarIdInput}
              onChange={(e) => setCalendarIdInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && loadCalendar()}
            />
          </div>
          <button
            onClick={() => loadCalendar()}
            disabled={loadingCal}
            className="bg-green-600 hover:bg-green-700 text-white px-5 py-2 rounded font-semibold text-sm"
          >
            {loadingCal ? "Loading..." : "Load Calendar"}
          </button>
          {!calendarInfo && (
            <p className="text-xs text-gray-400 w-full mt-1">
              Your Calendar ID is shown on the Dashboard after generating a calendar,
              or find it on the Calendars page.
            </p>
          )}
        </div>

        {loadError && (
          <p className="text-red-600 text-sm mt-2">{loadError}</p>
        )}

        {calendarInfo && (
          <div className="mt-3 bg-green-50 border border-green-200 rounded p-3 text-sm">
            <div className="flex flex-wrap gap-x-6 gap-y-1">
              <span><b>{calendarInfo.crop}</b></span>
              <span>{calendarInfo.season}</span>
              <span>{calendarInfo.location}</span>
              <span>{calendarInfo.farmer_name}</span>
              <span className="text-gray-400 text-xs">ID: {calendarInfo.calendar_id}</span>
            </div>
          </div>
        )}
      </div>

      {calendarInfo && (
        <>
          <div className="bg-white p-4 rounded shadow">
            <div className="flex items-center justify-between">
              {STAGES.map((s, i) => {
                const latestPct = latestProgressByStage[s] ?? 0;
                const isActive = s === stage;
                const isDone = latestPct === 100;

                return (
                  <React.Fragment key={i}>
                    <div
                      onClick={() => setStage(s)}
                      className={`flex flex-col items-center cursor-pointer px-3 py-2 rounded-lg transition ${
                        isActive ? "bg-green-600 text-white" :
                        isDone ? "bg-green-100 text-green-700" :
                        "bg-gray-100 hover:bg-gray-200 text-gray-600"
                      }`}
                    >
                      <span className="text-sm font-semibold">{s}</span>
                      <span className="text-xs mt-0.5 opacity-80">{latestPct}%</span>
                    </div>
                    {i < STAGES.length - 1 && (
                      <div className="flex-1 h-1 mx-1 bg-gray-200 rounded">
                        <div className="h-1 bg-green-400 rounded" style={{ width: `${latestPct}%` }} />
                      </div>
                    )}
                  </React.Fragment>
                );
              })}
            </div>
          </div>

          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-white p-5 rounded shadow space-y-4">
              <h2 className="font-bold text-gray-700">Log Progress - <span className="text-green-600">{stage}</span></h2>

              <div>
                <label className="text-xs text-gray-500 mb-1 block">Growth Stage</label>
                <select
                  className="border p-2 w-full rounded text-sm"
                  value={stage}
                  onChange={(e) => setStage(e.target.value)}
                >
                  {STAGES.map((s) => <option key={s}>{s}</option>)}
                </select>
              </div>

              <div className="bg-blue-50 border border-blue-200 rounded p-3 text-xs text-blue-800">
                {validationHint}
              </div>

              <div>
                <div className="flex justify-between items-center mb-1">
                  <label className="text-xs text-gray-500">Progress</label>
                  <span className="text-sm font-semibold text-green-700">{progress}%</span>
                </div>
                <input
                  type="range"
                  min={currentStageSavedProgress}
                  max="100"
                  value={progress}
                  onChange={(e) => setProgress(e.target.value)}
                  className="w-full accent-green-600"
                />
                <div className="flex justify-between text-xs text-gray-400 mt-0.5">
                  <span>{currentStageSavedProgress}%</span><span>50%</span><span>100%</span>
                </div>
              </div>

              <div>
                <label className="text-xs text-gray-500 mb-1 block">Farmer Notes (optional)</label>
                <textarea
                  className="border p-2 w-full rounded text-sm"
                  rows={3}
                  placeholder="e.g. Heavy rain observed, applied neem spray, soil moisture high..."
                  value={notes}
                  onChange={(e) => setNotes(e.target.value)}
                />
              </div>

              <button
                onClick={updateProgress}
                disabled={saving}
                className="bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded w-full font-semibold"
              >
                {saving ? "Saving..." : "Save Progress"}
              </button>
              <p className="text-xs text-gray-400 text-center">
                Progress is saved to the database - it will be here when you return
              </p>
            </div>

            <div className="bg-white p-5 rounded shadow space-y-3">
              <h2 className="font-bold">AI Analysis</h2>
              <div className="space-y-2 text-sm">
                <p>{yieldStatus}</p>
                <p>{progress > 70 ? "Crop nearing maturity" : "Weather conditions: monitor regularly"}</p>
                <p>
                  <b>Fertilizer for {stage}:</b> {fertDisplay}
                  {fertConf && (
                    <span className="text-gray-400 text-xs ml-1">({fertConf}% confidence)</span>
                  )}
                </p>
                <p><b>Harvest ETA:</b> {harvestETA}</p>
                {latestForStage && (
                  <div className="bg-blue-50 border border-blue-200 rounded p-2 text-xs text-blue-800">
                    Last logged for {stage}: <b>{latestForStage.progress_percent}%</b>
                    {latestForStage.notes && ` - "${latestForStage.notes}"`}
                    <span className="text-gray-400 ml-1">({latestForStage.logged_at})</span>
                  </div>
                )}
              </div>
            </div>
          </div>

          <div className="bg-white p-5 rounded shadow">
            <h2 className="font-bold mb-3">
              Progress History
              <span className="text-xs text-gray-400 font-normal ml-2">
                - all updates saved to database
              </span>
            </h2>
            {history.length === 0 ? (
              <p className="text-sm text-gray-400">
                No progress logged yet. Use the form above to log your first update.
              </p>
            ) : (
              <div className="space-y-2 max-h-72 overflow-y-auto">
                {history.map((h, i) => (
                  <div key={i} className="flex items-start gap-3 p-3 rounded border text-sm">
                    <div className={`w-2 h-2 rounded-full mt-1.5 flex-shrink-0 ${
                      h.progress_percent >= 80 ? "bg-green-500" :
                      h.progress_percent >= 50 ? "bg-yellow-500" : "bg-gray-400"
                    }`} />
                    <div className="flex-1">
                      <div className="flex justify-between items-center">
                        <span>
                          <b>{h.stage}</b> - {h.progress_percent}%
                        </span>
                        <span className="text-xs text-gray-400">{h.logged_at}</span>
                      </div>
                      {h.notes && (
                        <p className="text-xs text-gray-500 mt-0.5">{h.notes}</p>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}
