// frontend/src/components/TelemetryDashboard.jsx
import { useEffect, useState } from "react";

const METRICS_URL = "http://localhost:8080/metrics";

export default function TelemetryDashboard() {
  const [metrics, setMetrics] = useState({
    fps: 0,
    gpuMemoryPercent: 0,
    decoderUtilization: 0,
  });
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const response = await fetch(METRICS_URL);
        const data = await response.json();
        setMetrics(data);
        setConnected(true);
      } catch (err) {
        setConnected(false);
      }
    }, 1000); // poll every second

    return () => clearInterval(interval);
  }, []);

  return (
    <div style={{ fontFamily: "sans-serif", marginTop: "1.5rem", padding: "1rem", border: "1px solid #ccc", borderRadius: "8px", maxWidth: "400px" }}>
      <h3>Telemetry Dashboard</h3>
      <p>Status: {connected ? "🟢 Connected" : "🔴 Waiting for backend..."}</p>

      <div style={{ marginTop: "1rem" }}>
        <MetricBar label="FPS" value={metrics.fps} max={60} unit="fps" />
        <MetricBar label="GPU Memory" value={metrics.gpuMemoryPercent} max={100} unit="%" />
        <MetricBar label="Decoder Utilization" value={metrics.decoderUtilization} max={100} unit="%" />
      </div>
    </div>
  );
}

function MetricBar({ label, value, max, unit }) {
  const percent = Math.min((value / max) * 100, 100);
  return (
    <div style={{ marginBottom: "0.75rem" }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.85rem" }}>
        <span>{label}</span>
        <span>{value?.toFixed(1)} {unit}</span>
      </div>
      <div style={{ background: "#eee", borderRadius: "4px", height: "8px" }}>
        <div style={{ width: `${percent}%`, background: "#4caf50", height: "100%", borderRadius: "4px" }} />
      </div>
    </div>
  );
}
