"""
VRAM leak profiler for continuous video streaming.

Usage:
    python -m backend.streaming.profiling.vram_profiler --duration 3600 --interval 1

Logs nvidia-smi memory.used to a CSV, then reports whether usage climbed
(possible leak) or plateaued (healthy) over the run.

NOTE: Ritam separately verified no VRAM leaks in the zero-copy pipeline
itself (steady at 7.38MB, merged in PR #8). This script profiles the
STREAMING layer specifically -- useful to confirm the WebRTC/encode path
doesn't leak independently, even though the decode/inference path is
already covered. Worth confirming with the team this isn't redundant
before treating it as a required deliverable.

Run this WHILE the stream is active (backend server + at least one
connected viewer).
"""
import argparse
import csv
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def read_gpu_memory_mb() -> float:
    """Return current GPU memory used, in MB, via nvidia-smi."""
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    # if multiple GPUs, this takes the first line/device
    first_line = result.stdout.strip().splitlines()[0]
    return float(first_line)


def run_profiler(duration_seconds: int, interval_seconds: float, out_path: Path):
    print(f"Logging GPU memory every {interval_seconds}s for {duration_seconds}s "
          f"-> {out_path}")

    samples = []
    start = time.time()

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "memory_used_mb"])

        while time.time() - start < duration_seconds:
            try:
                mem = read_gpu_memory_mb()
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                print(f"nvidia-smi failed: {e}", file=sys.stderr)
                print("Is this running on a machine with an NVIDIA GPU + drivers?")
                return

            ts = datetime.now().isoformat()
            writer.writerow([ts, mem])
            f.flush()
            samples.append(mem)
            time.sleep(interval_seconds)

    analyze(samples)


def analyze(samples: list[float]):
    if len(samples) < 10:
        print("Not enough samples collected to analyze meaningfully.")
        return

    # skip the first ~10% as warmup (initial allocations are expected)
    warmup_cutoff = max(1, len(samples) // 10)
    steady_state = samples[warmup_cutoff:]

    first_half = steady_state[: len(steady_state) // 2]
    second_half = steady_state[len(steady_state) // 2 :]

    avg_first = sum(first_half) / len(first_half)
    avg_second = sum(second_half) / len(second_half)
    drift_mb = avg_second - avg_first
    drift_pct = (drift_mb / avg_first * 100) if avg_first > 0 else 0

    print("\n--- VRAM Profiling Result ---")
    print(f"Samples collected: {len(samples)}")
    print(f"Avg memory (first half, post-warmup):  {avg_first:.1f} MB")
    print(f"Avg memory (second half, post-warmup): {avg_second:.1f} MB")
    print(f"Drift: {drift_mb:+.1f} MB ({drift_pct:+.1f}%)")

    if drift_pct > 5:
        print("\n⚠️  LIKELY LEAK: memory usage climbed >5% between the first "
              "and second half of the run. Check for per-frame tensors/CuPy "
              "arrays not being freed each loop iteration.")
    else:
        print("\n✅ No significant drift detected — memory usage looks stable.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile GPU memory during streaming")
    parser.add_argument("--duration", type=int, default=1800,
                         help="How long to run, in seconds (default: 1800 = 30 min)")
    parser.add_argument("--interval", type=float, default=1.0,
                         help="Sampling interval in seconds (default: 1.0)")
    parser.add_argument("--out", type=str, default="vram_log.csv",
                         help="Output CSV path (default: vram_log.csv)")
    args = parser.parse_args()

    run_profiler(args.duration, args.interval, Path(args.out))
