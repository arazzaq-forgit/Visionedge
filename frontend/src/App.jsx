import VideoStream from "./components/VideoStream.jsx";
import TelemetryDashboard from "./components/TelemetryDashboard.jsx";

export default function App() {
  return (
    <div style={{ fontFamily: "sans-serif", padding: "2rem" }}>
      <h1>VisionEdge — Live Stream</h1>
      <VideoStream />
      <TelemetryDashboard />
    </div>
  );
}
