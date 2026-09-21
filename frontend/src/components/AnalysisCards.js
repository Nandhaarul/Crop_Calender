// AnalysisCards.js
// Used on InsightsPage — receives the full insights API response as prop

export default function AnalysisCards({ analysis }) {
  if (!analysis) return null;

  // ── CORRECT field names from /insights/generate response ──────────────────
  // OLD (broken): analysis.yield_prediction, analysis.risk_level,
  //               analysis.avg_temperature, analysis.avg_rainfall
  // NEW (correct): analysis.overall_health, analysis.risk_score,
  //                analysis.pest_risk, analysis.irrigation_advice

  const cards = [
    {
      label: "Crop Health",
      value: analysis.overall_health ?? "N/A",
      color:
        analysis.overall_health === "Excellent" ? "text-green-700" :
        analysis.overall_health === "Good"      ? "text-blue-600"  :
        analysis.overall_health === "At Risk"   ? "text-red-600"   :
        "text-yellow-600",
      icon: "💚",
    },
    {
      label: "Risk Score",
      value: analysis.risk_score != null ? `${analysis.risk_score} / 100` : "N/A",
      color:
        analysis.risk_score < 35 ? "text-green-700" :
        analysis.risk_score < 60 ? "text-yellow-600" :
        "text-red-600",
      icon: "⚠️",
    },
    {
      label: "Pest Risk",
      value: analysis.pest_risk ?? "N/A",
      color:
        analysis.pest_risk === "Low"    ? "text-green-700" :
        analysis.pest_risk === "Medium" ? "text-yellow-600" :
        "text-red-600",
      icon: "🐛",
    },
    {
      label: "Irrigation",
      value: analysis.irrigation_advice ?? "N/A",
      color: "text-blue-600",
      icon: "💧",
      small: true,
    },
  ];

  return (
    <div className="grid md:grid-cols-4 gap-4">
      {cards.map((card, i) => (
        <div key={i} className="bg-white shadow rounded-xl p-4">
          <p className="text-gray-500 text-sm mb-1">
            {card.icon} {card.label}
          </p>
          <h2 className={`font-bold ${card.small ? "text-sm" : "text-xl"} ${card.color}`}>
            {card.value}
          </h2>
        </div>
      ))}
    </div>
  );
}