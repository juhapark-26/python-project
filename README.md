# PhysNet Advanced Python Final Project

This repository contains a safe, synthetic-data reproduction of optimization
candidates for the rPPG-Toolbox PhysNet research pipeline.

The original research repository and UBFC-PHYS data are not included. The
`src/before/` folder contains the relevant original code excerpts, and
`src/after/` contains isolated optimized implementations.

## Structure

```text
.
├── README.md
├── requirements.txt
├── src/
│   ├── before/
│   │   ├── PhysNetNegPearsonLoss_before.py
│   │   └── post_process_before.py
│   └── after/
│       └── physnet_safe_optimizations.py
├── benchmark/
│   └── run_benchmark.py
├── results/
│   ├── benchmark_results.csv
│   └── environment.json
└── report/
    ├── report.md
    ├── report.html
    ├── report.docx
    └── report.pdf
```

## Optimization Topics

- A. Data structure / complexity: FFT-based circular MACC replaces repeated lag-wise `np.roll + np.corrcoef`.
- D. Decorator / caching: `functools.lru_cache` reuses detrend projection matrices.
- E. Deep learning code optimization: batch-vectorized negative Pearson loss replaces Python batch loop.

## Run Benchmark

```bash
python benchmark/run_benchmark.py --repeat 10 --warmup 1 --device cpu
```

The benchmark uses only synthetic tensors/signals:

- video shape reference: `[B, 3, 128, 128, 128]`
- signal shape: `[B, 128]`
- seed: `100`

Results are written to:

- `results/benchmark_results.csv`
- `results/environment.json`

## Report

The main report is in `report/report.md`. `report/report.docx` and
`report/report.pdf` are generated copies for submission.

