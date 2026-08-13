"""Long-running VRAM leak check. Run while browser streams are connected."""
import argparse
import statistics
import time

import torch


def mib(value):
    return value / 1024**2


def sample():
    return {"allocated_mib": mib(torch.cuda.memory_allocated()), "reserved_mib": mib(torch.cuda.memory_reserved())}


def profile(duration_seconds, interval_seconds, tolerance_mib):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; run this on the streaming GPU host")
    samples = []
    deadline = time.monotonic() + duration_seconds
    while time.monotonic() < deadline:
        value = sample()
        samples.append(value)
        print(f"allocated={value['allocated_mib']:.1f} MiB reserved={value['reserved_mib']:.1f} MiB")
        time.sleep(interval_seconds)
    third = max(1, len(samples) // 3)
    first = statistics.median(value["allocated_mib"] for value in samples[:third])
    last = statistics.median(value["allocated_mib"] for value in samples[-third:])
    growth = last - first
    print(f"VRAM allocated growth: {growth:+.1f} MiB (tolerance: {tolerance_mib:.1f} MiB)")
    return growth <= tolerance_mib


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes", type=float, default=30)
    parser.add_argument("--interval", type=float, default=10)
    parser.add_argument("--tolerance-mib", type=float, default=64)
    args = parser.parse_args()
    raise SystemExit(0 if profile(args.minutes * 60, args.interval, args.tolerance_mib) else 1)
