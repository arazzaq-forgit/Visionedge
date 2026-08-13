import { useRef, useState, useEffect } from "react";

const BACKEND_URL = "http://localhost:8080/offer";

export default function VideoStream({ onLiveChange }) {
  const videoRef = useRef(null);
  const pcRef = useRef(null);
  const [status, setStatus] = useState("disconnected");

  useEffect(() => {
    onLiveChange?.(status === "connected");
  }, [status, onLiveChange]);

  async function connect() {
    setStatus("connecting");

    const pc = new RTCPeerConnection();
    pcRef.current = pc;

    // we only receive video, we don't send any
    pc.addTransceiver("video", { direction: "recvonly" });

    pc.ontrack = (event) => {
      if (videoRef.current) {
        videoRef.current.srcObject = event.streams[0];
      }
    };

    pc.onconnectionstatechange = () => {
      setStatus(pc.connectionState);
    };

    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);

    const response = await fetch(BACKEND_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sdp: pc.localDescription.sdp,
        type: pc.localDescription.type,
      }),
    });

    const answer = await response.json();
    await pc.setRemoteDescription(answer);
  }

  function disconnect() {
    if (pcRef.current) {
      pcRef.current.close();
      pcRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setStatus("disconnected");
  }

  const isLive = status === "connected";
  const isPending = status === "connecting";
  const pillClass = isLive
    ? "status-pill status-pill--live"
    : isPending
      ? "status-pill status-pill--pending"
      : "status-pill status-pill--idle";

  return (
    <section className="stream-panel">
      <div className="stream-controls">
        <span className={pillClass}>
          <span className="status-dot" />
          {status}
        </span>

        <div className="btn-row">
          <button
            className="btn btn--primary"
            onClick={connect}
            disabled={isLive || isPending}
          >
            Connect
          </button>
          <button className="btn btn--ghost" onClick={disconnect}>
            Disconnect
          </button>
        </div>
      </div>

      <div className="video-frame">
        <video ref={videoRef} autoPlay playsInline muted />

        {!isLive && (
          <div className="video-frame__placeholder">
            {isPending ? "Negotiating connection…" : "No signal"}
          </div>
        )}

        {isLive && (
          <span className="video-frame__badge">
            <span className="status-dot" />
            live
          </span>
        )}

        <span className="video-frame__corner video-frame__corner--tl" />
        <span className="video-frame__corner video-frame__corner--tr" />
        <span className="video-frame__corner video-frame__corner--bl" />
        <span className="video-frame__corner video-frame__corner--br" />
      </div>
    </section>
  );
}
