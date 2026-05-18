"""
Export the XGBoost fraud model to ONNX, validate numerical equivalence,
and benchmark latency for both formats.

Run:
    python scripts/export_onnx.py

Dependencies (not in base requirements — install before running):
    pip install skl2onnx onnx onnxruntime
"""

import logging
import sys
import time
from pathlib import Path

import joblib
import numpy as np

# ---------------------------------------------------------------------------
# Logging: INFO → stdout, WARNING+ → stderr
# ---------------------------------------------------------------------------
_stdout_handler = logging.StreamHandler(sys.stdout)
_stdout_handler.setLevel(logging.DEBUG)
_stdout_handler.addFilter(lambda r: r.levelno < logging.WARNING)

_stderr_handler = logging.StreamHandler(sys.stderr)
_stderr_handler.setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[_stdout_handler, _stderr_handler],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PKL = REPO_ROOT / "models" / "xgboost_fraud_v1.pkl"
MODEL_ONNX = REPO_ROOT / "models" / "xgboost_fraud_v1.onnx"

BENCHMARK_RUNS = 1000
EQUIVALENCE_RUNS = 5
REL_TOL = 0.001  # 0.1% relative difference


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _percentile(times_ms: list[float], p: int) -> float:
    return float(np.percentile(times_ms, p))


def _benchmark(fn, n: int, X) -> list[float]:
    times: list[float] = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn(X)
        times.append((time.perf_counter() - t0) * 1000)
    return times


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    # ── 1. Verify optional dependencies ──────────────────────────────────
    try:
        import onnx
        import onnxruntime as ort
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except ImportError as exc:
        log.error(
            "Missing dependency: %s\n" "  Install with:  pip install skl2onnx onnx onnxruntime",
            exc,
        )
        sys.exit(1)

    # ── 2. Load model bundle ──────────────────────────────────────────────
    log.info("Loading model bundle from %s", MODEL_PKL)
    bundle = joblib.load(MODEL_PKL)
    model = bundle["model"]
    n_features = len(bundle["feature_names"])
    log.info(
        "Model loaded: version=%s  features=%d  threshold=%.4f",
        bundle["version"],
        n_features,
        bundle["threshold"],
    )

    # ── 3. Convert to ONNX ────────────────────────────────────────────────
    log.info("Converting to ONNX  (zipmap=False, FloatTensorType)")
    initial_type = [("X", FloatTensorType([None, n_features]))]
    onnx_model = convert_sklearn(
        model,
        initial_types=initial_type,
        options={type(model): {"zipmap": False}},
    )

    # ── 4. Validate ONNX graph ────────────────────────────────────────────
    log.info("Validating ONNX graph with onnx.checker")
    onnx.checker.check_model(onnx_model)
    log.info("ONNX graph check passed")

    # ── 5. Numerical equivalence on 5 random inputs ───────────────────────
    log.info("Checking numerical equivalence on %d random inputs", EQUIVALENCE_RUNS)
    rng = np.random.default_rng(42)
    X_check = rng.standard_normal((EQUIVALENCE_RUNS, n_features)).astype(np.float32)

    proba_pkl = model.predict_proba(X_check)[:, 1]

    sess = ort.InferenceSession(onnx_model.SerializeToString())
    proba_onnx = sess.run(None, {"X": X_check})[1][:, 1]

    rel_diff = np.abs(proba_pkl - proba_onnx) / (np.abs(proba_pkl) + 1e-10)
    max_rel = float(rel_diff.max())

    print()
    print("── Numerical Equivalence ────────────────────────────────────")
    for i, (p_pkl, p_onnx, rd) in enumerate(zip(proba_pkl, proba_onnx, rel_diff)):
        print(f"  sample {i}: pkl={p_pkl:.6f}  onnx={p_onnx:.6f}  rel_diff={rd:.2e}")

    if max_rel >= REL_TOL:
        log.error(
            "Equivalence check FAILED — max relative diff %.4f%% >= threshold %.1f%%",
            max_rel * 100,
            REL_TOL * 100,
        )
        sys.exit(1)
    log.info("Equivalence OK — max relative diff %.4f%%", max_rel * 100)

    # ── 6. Benchmark ──────────────────────────────────────────────────────
    log.info("Benchmarking %d runs each format (single-sample)", BENCHMARK_RUNS)
    X_bench = rng.standard_normal((1, n_features)).astype(np.float32)

    times_pkl = _benchmark(lambda X: model.predict_proba(X), BENCHMARK_RUNS, X_bench)
    times_onnx = _benchmark(lambda X: sess.run(None, {"X": X}), BENCHMARK_RUNS, X_bench)

    p50_pkl = _percentile(times_pkl, 50)
    p99_pkl = _percentile(times_pkl, 99)
    p50_onnx = _percentile(times_onnx, 50)
    p99_onnx = _percentile(times_onnx, 99)
    speedup = p50_pkl / p50_onnx if p50_onnx > 0 else float("inf")

    print()
    print("── Benchmark Results ────────────────────────────────────────")
    print(f"  {'Format':<12}  {'p50 (ms)':>10}  {'p99 (ms)':>10}")
    print(f"  {'-'*36}")
    print(f"  {'pkl':<12}  {p50_pkl:>10.3f}  {p99_pkl:>10.3f}")
    print(f"  {'onnx':<12}  {p50_onnx:>10.3f}  {p99_onnx:>10.3f}")
    print()
    print(f"  Speedup (pkl p50 / onnx p50): {speedup:.2f}x")

    # ── 7. Save ONNX model ────────────────────────────────────────────────
    MODEL_ONNX.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_ONNX, "wb") as f:
        f.write(onnx_model.SerializeToString())
    log.info("ONNX model saved → %s  (%d bytes)", MODEL_ONNX, MODEL_ONNX.stat().st_size)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        log.error("export_onnx failed: %s", exc, exc_info=True)
        sys.exit(1)
