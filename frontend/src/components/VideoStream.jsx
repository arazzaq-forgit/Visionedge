import { useRef, useState } from "react";

const BACKEND_URL = "http://localhost:8080/offer";

export default function VideoStream() {
  const videoRef = useRef(null);
  const pcRef = useRef(null);
  const [status, setStatus] = useState("disconnected");

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

  return (
    <div className="video-stream">
      <p>Status: {status}</p>
      <button onClick={connect} disabled={status === "connected" || status === "connecting"}>
        Connect
      </button>
      <button onClick={disconnect} style={{ marginLeft: "0.5rem" }}>
        Disconnect
      </button>
      <div style={{ marginTop: "1rem" }}>
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          style={{ width: "640px", background: "#000" }}
        />
      </div>
    </div>
  );
}
