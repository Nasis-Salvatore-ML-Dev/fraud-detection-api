"""
Verify the PSI training baseline produced by train.py and print a summary.

Run:
    python scripts/compute_baseline.py
"""

import json
import logging
import sys
from pathlib import Path

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
BASELINE_PATH = REPO_ROOT / "data" / "baselines" / "training_baseline.json"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    # ── 1. Locate baseline file ───────────────────────────────────────────
    log.info("Checking baseline at %s", BASELINE_PATH)
    if not BASELINE_PATH.exists():
        log.error(
            "Baseline file not found: %s\n" "  Run  python scripts/train.py  to generate it.",
            BASELINE_PATH,
        )
        sys.exit(1)

    # ── 2. Parse JSON ─────────────────────────────────────────────────────
    try:
        with open(BASELINE_PATH) as f:
            baseline: dict = json.load(f)
    except json.JSONDecodeError as exc:
        log.error("Baseline file is corrupt (invalid JSON): %s", exc)
        sys.exit(1)

    if not baseline:
        log.error("Baseline file is empty — re-run  python scripts/train.py.")
        sys.exit(1)

    # ── 3. Print summary ──────────────────────────────────────────────────
    print()
    print("── Training Baseline Summary ─────────────────────────────────")
    header = f"  {'Feature':<16}  {'n_bins':>6}  {'n_samples':>10}  {'mean':>10}  {'std':>10}  {'p5':>8}  {'p95':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for feature, stats in baseline.items():
        n_bins = len(stats["bin_edges"]) - 1
        print(
            f"  {feature:<16}  {n_bins:>6}  {stats['n_samples']:>10,}  "
            f"{stats['mean']:>10.4f}  {stats['std']:>10.4f}  "
            f"{stats['p5']:>8.4f}  {stats['p95']:>8.4f}"
        )

    print()
    log.info(
        "Baseline OK — %d feature(s), path: %s",
        len(baseline),
        BASELINE_PATH,
    )


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        log.error("compute_baseline failed: %s", exc, exc_info=True)
        sys.exit(1)
