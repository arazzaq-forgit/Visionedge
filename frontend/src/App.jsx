import { useState } from "react";
import VideoStream from "./components/VideoStream.jsx";
import TelemetryDashboard from "./components/TelemetryDashboard.jsx";
import PipelineFlow from "./components/PipelineFlow.jsx";
import StatStrip from "./components/StatStrip.jsx";

export default function App() {
  const [live, setLive] = useState(false);

  return (
    <div className="app">
      <div className="app__inner">
        <header className="app__header">
          <div>
            <div className="app__eyebrow">GPU-Accelerated Vision Pipeline</div>
            <h1 className="app__title">
              VisionEdge <span>/ live</span>
            </h1>
            <p className="app__subtitle">
              Hardware-decoded video, TensorRT inference, and a zero-copy GPU
              path — streamed to the browser over WebRTC.
            </p>
          </div>
        </header>

        <PipelineFlow active={live} />
        <StatStrip />

        <div className="dashboard-grid">
          <VideoStream onLiveChange={setLive} />
          <TelemetryDashboard />
        </div>

        <footer className="app__footer">
          Axlero Solutions Intern Project Program — Project 1, 2026 cycle
        </footer>
      </div>
    </div>
  );
}
