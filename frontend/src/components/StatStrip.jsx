const STATS = [
  { value: "3.05×", label: "TensorRT vs PyTorch", sub: "YOLOv8s, FP16" },
  { value: "3.99", unit: "ms", label: "Zero-copy latency", sub: "avg, P95 16.02 ms" },
  { value: "~250", unit: "fps", label: "Inference throughput", sub: "RTX 3050, 4GB" },
];

export default function StatStrip() {
  return (
    <div className="stat-strip">
      {STATS.map((stat) => (
        <div className="stat-strip__item" key={stat.label}>
          <div className="stat-strip__value">
            {stat.value}
            {stat.unit && <span>{stat.unit}</span>}
          </div>
          <div className="stat-strip__label">{stat.label}</div>
          <div className="stat-strip__sub">{stat.sub}</div>
        </div>
      ))}
    </div>
  );
}
