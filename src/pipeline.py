"""
Automated Insurance Analytics Pipeline
Provides reproducible, automated analysis for ACIS risk analytics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, List, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InsuranceAnalyticsPipeline:
    """Automated pipeline for insurance risk analytics."""
    
    def __init__(self, data_path: str = None):
        """
        Initialize the analytics pipeline.
        
        Parameters
        ----------
        data_path : str, optional
            Path to the insurance data file.
        """
        self.data_path = data_path or '../data/processed/insurance_data_final_cleaned.parquet'
        self.data = None
        self.results = {}
        
    def load_data(self) -> pd.DataFrame:
        """Load and validate insurance data."""
        logger.info(f"Loading data from: {self.data_path}")
        self.data = pd.read_parquet(self.data_path)
        logger.info(f"Data loaded: {self.data.shape[0]:,} rows, {self.data.shape[1]:,} columns")
        return self.data
    
    def calculate_portfolio_metrics(self) -> Dict:
        """Calculate key portfolio metrics."""
        logger.info("Calculating portfolio metrics...")
        
        metrics = {
            'total_policies': len(self.data),
            'total_premium': self.data['totalpremium'].sum(),
            'total_claims': self.data['totalclaims'].sum(),
            'loss_ratio': self.data['totalclaims'].sum() / self.data['totalpremium'].sum() if self.data['totalpremium'].sum() > 0 else 0,
            'claim_rate': (self.data['totalclaims'] > 0).mean() * 100,
            'avg_claim_amount': self.data[self.data['totalclaims'] > 0]['totalclaims'].mean() if (self.data['totalclaims'] > 0).any() else 0
        }
        
        self.results['portfolio_metrics'] = metrics
        return metrics
    
    def detect_outliers(self, variables: List[str] = None) -> Dict:
        """Detect outliers in specified variables using IQR method."""
        logger.info("Detecting outliers...")
        
        if variables is None:
            variables = ['totalpremium', 'totalclaims', 'calculatedpremiumperterm', 'suminsured']
        
        outliers = {}
        for var in variables:
            if var in self.data.columns:
                q1 = self.data[var].quantile(0.25)
                q3 = self.data[var].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outlier_data = self.data[(self.data[var] < lower_bound) | (self.data[var] > upper_bound)]
                outliers[var] = {
                    'count': len(outlier_data),
                    'percentage': (len(outlier_data) / len(self.data)) * 100,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'min': self.data[var].min(),
                    'max': self.data[var].max()
                }
        
        self.results['outliers'] = outliers
        return outliers
    
    def analyze_categorical_variables(self, variables: List[str] = None) -> Dict:
        """Analyze categorical variables with statistics."""
        logger.info("Analyzing categorical variables...")
        
        if variables is None:
            variables = ['province', 'vehicletype', 'gender_clean', 'covertype']
        
        categorical_analysis = {}
        for var in variables:
            if var in self.data.columns:
                analysis = {}
                value_counts = self.data[var].value_counts()
                
                for category in value_counts.head(5).index:  # Top 5 categories
                    category_data = self.data[self.data[var] == category]
                    analysis[category] = {
                        'policy_count': len(category_data),
                        'claim_rate': (category_data['totalclaims'] > 0).mean() * 100,
                        'avg_premium': category_data['totalpremium'].mean(),
                        'avg_claim': category_data[category_data['totalclaims'] > 0]['totalclaims'].mean() if (category_data['totalclaims'] > 0).any() else 0,
                        'loss_ratio': category_data['totalclaims'].sum() / category_data['totalpremium'].sum() if category_data['totalpremium'].sum() > 0 else 0
                    }
                
                categorical_analysis[var] = analysis
        
        self.results['categorical_analysis'] = categorical_analysis
        return categorical_analysis
    
    def generate_report(self, output_dir: str = '../reports') -> None:
        """Generate comprehensive analysis report."""
        logger.info("Generating analysis report...")
        
        report_path = Path(output_dir) / 'automated_analysis_report.md'
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write("# Automated Insurance Analytics Report\n\n")
            f.write(f"Generated: {pd.Timestamp.now()}\n\n")
            
            # Portfolio Metrics
            f.write("## 1. Portfolio Metrics\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            for key, value in self.results.get('portfolio_metrics', {}).items():
                if isinstance(value, float):
                    if 'ratio' in key or 'rate' in key:
                        f.write(f"| {key.replace('_', ' ').title()} | {value:.2%} |\n")
                    else:
                        f.write(f"| {key.replace('_', ' ').title()} | {value:,.2f} |\n")
                else:
                    f.write(f"| {key.replace('_', ' ').title()} | {value:,} |\n")
            
            # Outlier Analysis
            f.write("\n## 2. Outlier Analysis\n")
            outliers = self.results.get('outliers', {})
            for var, stats in outliers.items():
                f.write(f"\n### {var.replace('_', ' ').title()}\n")
                f.write(f"- Outlier count: {stats['count']:,} ({stats['percentage']:.1f}%)\n")
                f.write(f"- Bounds: [{stats['lower_bound']:,.0f}, {stats['upper_bound']:,.0f}]\n")
                f.write(f"- Range: [{stats['min']:,.0f}, {stats['max']:,.0f}]\n")
            
            # Categorical Analysis
            f.write("\n## 3. Categorical Variable Analysis\n")
            cat_analysis = self.results.get('categorical_analysis', {})
            for var, categories in cat_analysis.items():
                f.write(f"\n### {var.replace('_', ' ').title()}\n")
                f.write("| Category | Policies | Claim Rate | Avg Premium | Avg Claim | Loss Ratio |\n")
                f.write("|----------|----------|------------|-------------|-----------|------------|\n")
                for category, stats in categories.items():
                    f.write(f"| {category} | {stats['policy_count']:,} | {stats['claim_rate']:.1f}% | R{stats['avg_premium']:,.0f} | R{stats['avg_claim']:,.0f} | {stats['loss_ratio']:.1%} |\n")
        
        logger.info(f"Report saved to: {report_path}")
        return str(report_path)
    
    def run_full_pipeline(self) -> Dict:
        """Run the complete analytics pipeline."""
        logger.info("Starting automated analytics pipeline...")
        
        self.load_data()
        self.calculate_portfolio_metrics()
        self.detect_outliers()
        self.analyze_categorical_variables()
        report_path = self.generate_report()
        
        logger.info("Pipeline completed successfully!")
        return {
            'results': self.results,
            'report_path': report_path
        }


# Example usage
if __name__ == "__main__":
    pipeline = InsuranceAnalyticsPipeline()
    results = pipeline.run_full_pipeline()
    print(f" Analysis complete. Report: {results['report_path']}")