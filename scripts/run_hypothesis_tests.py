#!/usr/bin/env python3
"""CLI to run hypothesis tests for ACIS insurance analytics."""

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / "src"))

from hypothesis_testing import run as run_tests  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Run hypothesis tests on insurance data.")
    parser.add_argument(
        "--data-path",
        default="data/processed/insurance_data_final_cleaned.parquet",
        help="Parquet file to analyze",
    )
    parser.add_argument(
        "--report-path",
        default="reports/hypothesis_testing.md",
        help="Output Markdown report path",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("Running hypothesis tests...")
    results = run_tests(path=args.data_path, report_path=args.report_path)
    print(f"Report written to {args.report_path}")

    def decision(p):
        if p != p:
            return "insufficient data"
        return "reject H0" if p < 0.05 else "fail to reject H0"

    print("\nDecisions:")
    print(f"- Provinces frequency: {decision(results['provinces']['chi2_p'])}")
    print(f"- Provinces severity: {decision(results['provinces']['severity_p'])}")
    print(f"- Zip codes frequency: {decision(results['zipcodes']['chi2_p'])}")
    print(f"- Zip codes severity: {decision(results['zipcodes']['severity_p'])}")
    print(f"- Zip codes margin: {decision(results['zipcodes']['margin_p'])}")
    print(f"- Gender frequency: {decision(results['gender']['chi2_p'])}")
    print(f"- Gender severity: {decision(results['gender']['severity_p'])}")


if __name__ == "__main__":
    main()