"""Run before/after benchmarks for the PhysNet optimization report.

This script does not read real UBFC-PHYS raw or cached data. It generates
synthetic tensors/signals that match the PhysNet setting used in the report.
"""

import argparse
import csv
import json
import platform
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import scipy
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "after"))

from physnet_optimizations import (  # noqa: E402
    VectorizedNegPearson,
    cached_detrend,
    fft_circular_macc,
    original_detrend_reference,
    original_macc_reference,
    original_neg_pearson_reference,
)


def _sync_if_needed():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _measure(fn, repeat, warmup):
    for _ in range(warmup):
        fn()
    _sync_if_needed()

    times = []
    peaks = []
    for _ in range(repeat):
        tracemalloc.start()
        start = time.perf_counter()
        fn()
        _sync_if_needed()
        elapsed = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        times.append(elapsed * 1000.0)
        peaks.append(peak / 1024.0)

    return {
        "mean_ms": statistics.mean(times),
        "std_ms": statistics.stdev(times) if len(times) > 1 else 0.0,
        "peak_kib": max(peaks),
    }


def _torch_signals(batch, length, seed, device):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    labels = torch.randn(batch, length, generator=gen, dtype=torch.float32)
    preds = labels * 0.8 + 0.2 * torch.randn(batch, length, generator=gen, dtype=torch.float32)
    return preds.to(device), labels.to(device)


def _np_signals(length, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(length, dtype=np.float64) / 35.0
    label = np.sin(2 * np.pi * 1.2 * t) + 0.03 * rng.standard_normal(length)
    pred = np.sin(2 * np.pi * 1.2 * t + 0.2) + 0.05 * rng.standard_normal(length)
    return pred, label


def _add_row(rows, category, input_size, variant, stats, correct):
    rows.append(
        {
            "category": category,
            "input_size": input_size,
            "variant": variant,
            "mean_ms": f"{stats['mean_ms']:.6f}",
            "std_ms": f"{stats['std_ms']:.6f}",
            "peak_kib": f"{stats['peak_kib']:.3f}",
            "correct_vs_reference": str(correct),
        }
    )


def benchmark_loss(rows, repeat, warmup, device):
    after_loss = VectorizedNegPearson().to(device)
    for batch, length in [(4, 128), (16, 128), (64, 128), (64, 512)]:
        preds, labels = _torch_signals(batch, length, seed=100 + batch + length, device=device)
        ref_value = original_neg_pearson_reference(preds, labels)
        after_value = after_loss(preds, labels)
        correct = bool(torch.allclose(ref_value, after_value, rtol=1e-5, atol=1e-6))
        input_size = f"B={batch},T={length}"

        _add_row(
            rows,
            "loss",
            input_size,
            "before_loop",
            _measure(lambda: original_neg_pearson_reference(preds, labels), repeat, warmup),
            True,
        )
        _add_row(
            rows,
            "loss",
            input_size,
            "after_vectorized",
            _measure(lambda: after_loss(preds, labels), repeat, warmup),
            correct,
        )


def benchmark_macc(rows, repeat, warmup):
    for length in [128, 256, 512, 1024]:
        pred, label = _np_signals(length, seed=200 + length)
        ref_value = original_macc_reference(pred, label)
        after_value = fft_circular_macc(pred, label)
        correct = bool(np.allclose(ref_value, after_value, rtol=1e-5, atol=1e-6))
        input_size = f"T={length}"

        _add_row(
            rows,
            "macc",
            input_size,
            "before_roll_corrcoef",
            _measure(lambda: original_macc_reference(pred, label), repeat, warmup),
            True,
        )
        _add_row(
            rows,
            "macc",
            input_size,
            "after_fft",
            _measure(lambda: fft_circular_macc(pred, label), repeat, warmup),
            correct,
        )


def benchmark_detrend(rows, repeat, warmup):
    for length in [32, 64, 128, 256]:
        pred, _ = _np_signals(length, seed=300 + length)
        ref_value = original_detrend_reference(pred, 100)
        after_value = cached_detrend(pred, 100)
        correct = bool(np.allclose(ref_value, after_value, rtol=1e-7, atol=1e-8))
        input_size = f"T={length}"

        _add_row(
            rows,
            "detrend",
            input_size,
            "before_rebuild_inverse",
            _measure(lambda: original_detrend_reference(pred, 100), repeat, warmup),
            True,
        )
        _add_row(
            rows,
            "detrend",
            input_size,
            "after_lru_cache",
            _measure(lambda: cached_detrend(pred, 100), repeat, warmup),
            correct,
        )


def environment_payload(device):
    return {
        "os": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "synthetic_video_shape_reference": "[B, 3, 128, 128, 128]",
        "synthetic_label_shape_reference": "[B, 128]",
        "seed": 100,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_dir", type=str, default=str(ROOT / "results"))
    args = parser.parse_args()

    np.random.seed(100)
    torch.manual_seed(100)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")

    rows = []
    benchmark_loss(rows, repeat=int(args.repeat), warmup=int(args.warmup), device=device)
    benchmark_macc(rows, repeat=int(args.repeat), warmup=int(args.warmup))
    benchmark_detrend(rows, repeat=int(args.repeat), warmup=int(args.warmup))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "benchmark_results.csv"
    env_path = output_dir / "environment.json"

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "category",
                "input_size",
                "variant",
                "mean_ms",
                "std_ms",
                "peak_kib",
                "correct_vs_reference",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    with env_path.open("w", encoding="utf-8") as f:
        json.dump(environment_payload(device), f, indent=2)

    print(f"Wrote {csv_path}")
    print(f"Wrote {env_path}")


if __name__ == "__main__":
    main()
