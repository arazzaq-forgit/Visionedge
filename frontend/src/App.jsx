import VideoStream from "./components/VideoStream.jsx";
// TODO(Asha): wire in <TelemetryDashboard /> here once it's ready —
// leaving this to whoever merges next so we don't collide on layout.
// import TelemetryDashboard from "./components/TelemetryDashboard.jsx";

export default function App() {
  return (
    <div style={{ fontFamily: "sans-serif", padding: "2rem" }}>
      <h1>VisionEdge — Live Stream</h1>
      <VideoStream />
    </div>
  );
}
