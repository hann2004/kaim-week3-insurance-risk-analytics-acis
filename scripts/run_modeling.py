#!/usr/bin/env python3
"""Run Task-4 modeling pipeline and save report."""

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / "src"))

from modeling import run as run_modeling  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Run modeling pipeline (Task-4).")
    parser.add_argument(
        "--data-path",
        default="data/processed/insurance_data_final_cleaned.parquet",
        help="Parquet file to analyze",
    )
    parser.add_argument(
        "--report-path",
        default="reports/modeling_report.md",
        help="Output Markdown report path",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on number of rows to sample for modeling (reduces memory/compute).",
    )
    parser.add_argument(
        "--enable-shap",
        action="store_true",
        help="Compute SHAP feature importances (slower).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("Running modeling pipeline...")
    results = run_modeling(
        data_path=args.data_path,
        report_path=args.report_path,
        enable_shap=args.enable_shap,
        sample_rows=args.max_rows,
    )
    print(f"Report saved to {args.report_path}")

    # Quick console summary
    freq = results.get("frequency", {})
    prem = results.get("premium", {})
    sev = results.get("severity", {})

    def best_metric(metric_dict, key, higher_is_better=True):
        if not metric_dict:
            return "n/a", float("nan")
        best_name = None
        best_val = None
        for name, vals in metric_dict.items():
            if key not in vals:
                continue
            val = vals[key]
            if best_val is None or (higher_is_better and val > best_val) or (not higher_is_better and val < best_val):
                best_name, best_val = name, val
        return best_name or "n/a", best_val if best_val is not None else float("nan")

    freq_best, freq_val = best_metric(freq, "roc_auc", True)
    prem_best, prem_val = best_metric(prem, "rmse", False)
    sev_best, sev_val = best_metric(sev, "rmse", False)

    print("\nTop models")
    print(f"- Claim frequency: {freq_best} (ROC-AUC={freq_val:.3f})")
    print(f"- Premium regression: {prem_best} (RMSE={prem_val:.1f})")
    print(f"- Severity regression: {sev_best} (RMSE={sev_val:.1f})")


if __name__ == "__main__":
    main()