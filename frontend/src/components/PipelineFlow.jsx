const STAGES = [
  { label: "Decode", tag: "PyAV / NVDEC" },
  { label: "Inference", tag: "TensorRT" },
  { label: "Zero-Copy", tag: "CuPy" },
  { label: "Stream", tag: "aiortc" },
];

export default function PipelineFlow({ active }) {
  return (
    <div className={`pipeline ${active ? "pipeline--active" : ""}`}>
      {STAGES.map((stage, i) => (
        <div className="pipeline__stage" key={stage.label}>
          <div className="pipeline__node">
            <span className="pipeline__index">{String(i + 1).padStart(2, "0")}</span>
            <span className="pipeline__label">{stage.label}</span>
            <span className="pipeline__tag">{stage.tag}</span>
          </div>
          {i < STAGES.length - 1 && <div className="pipeline__link" />}
        </div>
      ))}
    </div>
  );
}
