#!/usr/bin/env python3
"""
Script to run automated insurance analytics pipeline.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from pipeline import InsuranceAnalyticsPipeline

def main():
    """Run the analytics pipeline."""
    print(" Starting Insurance Analytics Pipeline...")
    
    pipeline = InsuranceAnalyticsPipeline()
    results = pipeline.run_full_pipeline()
    
    print(f"\n Pipeline completed successfully!")
    print(f" Report generated: {results['report_path']}")
    
    # Print key findings
    metrics = results['results']['portfolio_metrics']
    print(f"\n Key Portfolio Metrics:")
    print(f"   • Loss Ratio: {metrics['loss_ratio']:.1%}")
    print(f"   • Claim Rate: {metrics['claim_rate']:.1f}%")
    print(f"   • Total Policies: {metrics['total_policies']:,}")
    print(f"   • Financial Impact: R{metrics['total_claims'] - metrics['total_premium']:,.0f}")

if __name__ == "__main__":
    main()