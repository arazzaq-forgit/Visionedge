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
    <section className={`telemetry ${connected ? "" : "telemetry--offline"}`}>
      <div className="telemetry__header">
        <h3 className="telemetry__title">Telemetry</h3>
        <span
          className={`status-pill ${connected ? "status-pill--live" : "status-pill--idle"
            }`}
        >
          <span className="status-dot" />
          {connected ? "connected" : "waiting for backend"}
        </span>
      </div>

      <MetricBar label="FPS" value={metrics.fps} max={60} unit="fps" />
      <MetricBar
        label="GPU Memory"
        value={metrics.gpuMemoryPercent}
        max={100}
        unit="%"
        amber
      />
      <MetricBar
        label="Decoder Utilization"
        value={metrics.decoderUtilization}
        max={100}
        unit="%"
      />
    </section>
  );
}

function MetricBar({ label, value, max, unit, amber }) {
  const safeValue = value ?? 0;
  const percent = Math.min((safeValue / max) * 100, 100);
  return (
    <div className="metric">
      <div className="metric__row">
        <span className="metric__label">{label}</span>
        <span className="metric__value">
          {safeValue.toFixed(1)}
          <span>{unit}</span>
        </span>
      </div>
      <div className="meter">
        <div
          className={`meter__fill ${amber ? "meter__fill--amber" : ""}`}
          style={{ width: `${percent}%` }}
        />
      </div>
    </div>
  );
}
