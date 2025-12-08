"""
Hypothesis Testing Module for Insurance Risk Analytics

Tests statistical hypotheses about risk differences across:
- Provinces
- Zip codes (postal codes)
- Gender

Uses appropriate statistical tests based on data distribution:
- Parametric tests (ANOVA, t-test) for normally distributed data
- Non-parametric tests (Kruskal-Wallis, Mann-Whitney U) for skewed data
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import (
    chi2_contingency,
    kruskal,
    mannwhitneyu,
    f_oneway,
    shapiro,
    normaltest
)
from pathlib import Path
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class HypothesisTester:
    """Statistical hypothesis testing for insurance risk analytics."""
    
    def __init__(self, data: pd.DataFrame, alpha: float = 0.05):
        """
        Initialize hypothesis tester.
        
        Parameters
        ----------
        data : pd.DataFrame
            Insurance data with columns: province, postalcode, gender, 
            totalclaims, totalpremium
        alpha : float, default=0.05
            Significance level for hypothesis tests
        """
        self.data = data.copy()
        self.alpha = alpha
        self.results = {}
        
        # Ensure required columns exist
        self._validate_columns()
        
        # Create derived metrics
        self.data['has_claim'] = (self.data['totalclaims'] > 0).astype(int)
        self.data['margin'] = self.data['totalpremium'] - self.data['totalclaims']
        
    def _validate_columns(self):
        """Check that required columns exist."""
        required = ['province', 'postalcode', 'gender', 'totalclaims', 'totalpremium']
        missing = [col for col in required if col not in self.data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    
    def check_normality(self, data: pd.Series, max_samples: int = 5000) -> Tuple[bool, float]:
        """
        Check if data is normally distributed.
        
        Parameters
        ----------
        data : pd.Series
            Data to test
        max_samples : int
            Maximum samples for normality test (for performance)
            
        Returns
        -------
        is_normal : bool
            True if data appears normally distributed
        p_value : float
            p-value from normality test
        """
        # Remove zeros and NaN
        test_data = data[(data > 0) & data.notna()].dropna()
        
        if len(test_data) < 3:
            return False, 1.0
        
        # Sample if too large (for performance)
        if len(test_data) > max_samples:
            test_data = test_data.sample(n=max_samples, random_state=42)
        
        # Use D'Agostino-Pearson test (more robust than Shapiro-Wilk for large samples)
        try:
            _, p_value = normaltest(test_data)
            is_normal = p_value > 0.05
        except:
            # Fallback to Shapiro-Wilk for small samples
            if len(test_data) < 5000:
                _, p_value = shapiro(test_data)
                is_normal = p_value > 0.05
            else:
                is_normal = False
                p_value = 0.0
        
        return is_normal, p_value
    
    def test_claim_frequency(self, group_col: str, min_group_size: int = 50) -> Dict:
        """
        Test if claim frequency (proportion with claims) differs across groups.
        Uses Chi-square test of independence.
        
        Parameters
        ----------
        group_col : str
            Column to group by (e.g., 'province', 'postalcode')
        min_group_size : int
            Minimum observations per group to include in test
            
        Returns
        -------
        dict
            Test results with p-value and statistics
        """
        # Filter groups with sufficient data
        group_counts = self.data[group_col].value_counts()
        valid_groups = group_counts[group_counts >= min_group_size].index
        
        if len(valid_groups) < 2:
            return {
                'chi2': np.nan,
                'p_value': np.nan,
                'dof': 0,
                'valid_groups': len(valid_groups),
                'message': 'Insufficient groups for testing'
            }
        
        # Create contingency table
        filtered_data = self.data[self.data[group_col].isin(valid_groups)]
        contingency = pd.crosstab(filtered_data[group_col], filtered_data['has_claim'])
        
        # Chi-square test
        chi2, p_value, dof, expected = chi2_contingency(contingency)
        
        return {
            'chi2': chi2,
            'p_value': p_value,
            'dof': dof,
            'valid_groups': len(valid_groups),
            'contingency_table': contingency,
            'expected_frequencies': expected
        }
    
    def test_claim_severity(self, group_col: str, min_group_size: int = 50, 
                           use_parametric: Optional[bool] = None) -> Dict:
        """
        Test if claim severity (claim amounts) differs across groups.
        Uses ANOVA (parametric) or Kruskal-Wallis (non-parametric) based on data distribution.
        
        Parameters
        ----------
        group_col : str
            Column to group by
        min_group_size : int
            Minimum observations per group
        use_parametric : bool, optional
            Force parametric (True) or non-parametric (False) test.
            If None, automatically selects based on normality.
            
        Returns
        -------
        dict
            Test results with p-value and test type used
        """
        # Filter to only policies with claims
        claims_data = self.data[self.data['totalclaims'] > 0].copy()
        
        if len(claims_data) < 10:
            return {
                'test_type': 'insufficient_data',
                'p_value': np.nan,
                'statistic': np.nan,
                'is_normal': False
            }
        
        # Filter groups with sufficient data
        group_counts = claims_data[group_col].value_counts()
        valid_groups = group_counts[group_counts >= min_group_size].index
        
        if len(valid_groups) < 2:
            return {
                'test_type': 'insufficient_groups',
                'p_value': np.nan,
                'statistic': np.nan,
                'valid_groups': len(valid_groups)
            }
        
        # Prepare data for testing
        filtered_data = claims_data[claims_data[group_col].isin(valid_groups)]
        groups = [filtered_data[filtered_data[group_col] == group]['totalclaims'].values 
                 for group in valid_groups]
        
        # Check normality if not specified
        if use_parametric is None:
            # Check normality of claim amounts
            all_claims = filtered_data['totalclaims']
            is_normal, normality_p = self.check_normality(all_claims)
            use_parametric = is_normal
        else:
            normality_p = np.nan
        
        # Perform appropriate test
        if use_parametric:
            # ANOVA (parametric)
            statistic, p_value = f_oneway(*groups)
            test_type = 'ANOVA (F-test)'
        else:
            # Kruskal-Wallis (non-parametric)
            statistic, p_value = kruskal(*groups)
            test_type = 'Kruskal-Wallis'
        
        return {
            'test_type': test_type,
            'p_value': p_value,
            'statistic': statistic,
            'is_normal': use_parametric if use_parametric is not None else is_normal,
            'normality_p': normality_p,
            'valid_groups': len(valid_groups),
            'group_means': {group: filtered_data[filtered_data[group_col] == group]['totalclaims'].mean() 
                           for group in valid_groups}
        }
    
    def test_margin_difference(self, group_col: str, min_group_size: int = 50,
                              use_parametric: Optional[bool] = None) -> Dict:
        """
        Test if profit margin differs across groups.
        Uses ANOVA (parametric) or Kruskal-Wallis (non-parametric).
        
        Parameters
        ----------
        group_col : str
            Column to group by
        min_group_size : int
            Minimum observations per group
        use_parametric : bool, optional
            Force parametric or non-parametric test
            
        Returns
        -------
        dict
            Test results
        """
        # Filter groups with sufficient data
        group_counts = self.data[group_col].value_counts()
        valid_groups = group_counts[group_counts >= min_group_size].index
        
        if len(valid_groups) < 2:
            return {
                'test_type': 'insufficient_groups',
                'p_value': np.nan,
                'statistic': np.nan,
                'valid_groups': len(valid_groups)
            }
        
        # Prepare data
        filtered_data = self.data[self.data[group_col].isin(valid_groups)]
        groups = [filtered_data[filtered_data[group_col] == group]['margin'].values 
                 for group in valid_groups]
        
        # Check normality if not specified
        if use_parametric is None:
            all_margins = filtered_data['margin']
            is_normal, normality_p = self.check_normality(all_margins)
            use_parametric = is_normal
        else:
            normality_p = np.nan
        
        # Perform test
        if use_parametric:
            statistic, p_value = f_oneway(*groups)
            test_type = 'ANOVA (F-test)'
        else:
            statistic, p_value = kruskal(*groups)
            test_type = 'Kruskal-Wallis'
        
        return {
            'test_type': test_type,
            'p_value': p_value,
            'statistic': statistic,
            'is_normal': use_parametric if use_parametric is not None else is_normal,
            'normality_p': normality_p,
            'valid_groups': len(valid_groups),
            'group_means': {group: filtered_data[filtered_data[group_col] == group]['margin'].mean() 
                           for group in valid_groups}
        }
    
    def test_gender_difference(self, metric: str = 'severity', 
                               min_group_size: int = 10) -> Dict:
        """
        Test if risk differs between men and women.
        Uses t-test (parametric) or Mann-Whitney U (non-parametric) for two groups.
        
        Parameters
        ----------
        metric : str
            'frequency' for claim rate, 'severity' for claim amounts
        min_group_size : int
            Minimum observations per gender group
            
        Returns
        -------
        dict
            Test results
        """
        # Filter to Male and Female only (exclude "Not specified")
        gender_data = self.data[self.data['gender'].isin(['Male', 'Female'])].copy()
        
        male_data = gender_data[gender_data['gender'] == 'Male']
        female_data = gender_data[gender_data['gender'] == 'Female']
        
        if len(male_data) < min_group_size or len(female_data) < min_group_size:
            return {
                'test_type': 'insufficient_data',
                'p_value': np.nan,
                'statistic': np.nan,
                'message': f'Insufficient data: Male={len(male_data)}, Female={len(female_data)}'
            }
        
        if metric == 'frequency':
            # Chi-square test for proportions
            contingency = pd.crosstab(gender_data['gender'], gender_data['has_claim'])
            chi2, p_value, dof, _ = chi2_contingency(contingency)
            return {
                'test_type': 'Chi-square',
                'p_value': p_value,
                'statistic': chi2,
                'dof': dof
            }
        
        elif metric == 'severity':
            # Test claim amounts (only policies with claims)
            male_claims = male_data[male_data['totalclaims'] > 0]['totalclaims']
            female_claims = female_data[female_data['totalclaims'] > 0]['totalclaims']
            
            if len(male_claims) < 3 or len(female_claims) < 3:
                return {
                    'test_type': 'insufficient_data',
                    'p_value': np.nan,
                    'statistic': np.nan,
                    'message': 'Insufficient claims for severity test'
                }
            
            # Check normality
            male_normal, _ = self.check_normality(male_claims)
            female_normal, _ = self.check_normality(female_claims)
            use_parametric = male_normal and female_normal
            
            # Perform test
            if use_parametric:
                statistic, p_value = stats.ttest_ind(male_claims, female_claims)
                test_type = 't-test (independent samples)'
            else:
                statistic, p_value = mannwhitneyu(male_claims, female_claims, 
                                                 alternative='two-sided')
                test_type = 'Mann-Whitney U'
            
            return {
                'test_type': test_type,
                'p_value': p_value,
                'statistic': statistic,
                'male_normal': male_normal,
                'female_normal': female_normal
            }
    
    def calculate_summary_stats(self, group_col: str) -> pd.DataFrame:
        """Calculate summary statistics by group."""
        summary = self.data.groupby(group_col).agg({
            'policyid': 'count',  # Count policies
            'has_claim': 'mean',  # Claim rate
            'totalpremium': 'sum',
            'totalclaims': 'sum'
        }).rename(columns={
            'policyid': 'policies',
            'has_claim': 'claim_rate'
        })
        
        # Calculate loss ratio
        summary['loss_ratio'] = summary['totalclaims'] / summary['totalpremium']
        
        # Calculate average claim amount (only for policies with claims)
        claims_data = self.data[self.data['totalclaims'] > 0]
        if len(claims_data) > 0:
            avg_claims = claims_data.groupby(group_col)['totalclaims'].mean()
            summary['avg_claim_given_claim'] = avg_claims
        else:
            summary['avg_claim_given_claim'] = 0
        
        # Calculate margin
        summary['margin'] = self.data.groupby(group_col)['margin'].mean()
        
        return summary.fillna(0).sort_values('policies', ascending=False)
    
    def test_provinces(self) -> Dict:
        """Test all province-related hypotheses."""
        print("Testing province hypotheses...")
        
        # Frequency test
        freq_result = self.test_claim_frequency('province', min_group_size=50)
        
        # Severity test
        sev_result = self.test_claim_severity('province', min_group_size=50)
        
        # Summary statistics
        summary = self.calculate_summary_stats('province')
        
        return {
            'chi2_p': freq_result['p_value'],
            'severity_p': sev_result['p_value'],
            'frequency_test': freq_result,
            'severity_test': sev_result,
            'summary': summary
        }
    
    def test_zipcodes(self) -> Dict:
        """Test all zip code-related hypotheses."""
        print("Testing zip code hypotheses...")
        
        # Frequency test
        freq_result = self.test_claim_frequency('postalcode', min_group_size=50)
        
        # Severity test
        sev_result = self.test_claim_severity('postalcode', min_group_size=50)
        
        # Margin test
        margin_result = self.test_margin_difference('postalcode', min_group_size=50)
        
        # Summary statistics (top 10)
        summary = self.calculate_summary_stats('postalcode').head(10)
        
        return {
            'chi2_p': freq_result['p_value'],
            'severity_p': sev_result['p_value'],
            'margin_p': margin_result['p_value'],
            'frequency_test': freq_result,
            'severity_test': sev_result,
            'margin_test': margin_result,
            'summary': summary
        }
    
    def test_gender(self) -> Dict:
        """Test gender-related hypotheses."""
        print("Testing gender hypotheses...")
        
        # Frequency test
        freq_result = self.test_gender_difference('frequency', min_group_size=10)
        
        # Severity test
        sev_result = self.test_gender_difference('severity', min_group_size=10)
        
        # Summary statistics
        summary = self.calculate_summary_stats('gender')
        
        return {
            'chi2_p': freq_result['p_value'],
            'severity_p': sev_result['p_value'],
            'frequency_test': freq_result,
            'severity_test': sev_result,
            'summary': summary
        }
    
    def run_all_tests(self) -> Dict:
        """Run all hypothesis tests."""
        print("=" * 60)
        print("Running All Hypothesis Tests")
        print("=" * 60)
        
        results = {
            'provinces': self.test_provinces(),
            'zipcodes': self.test_zipcodes(),
            'gender': self.test_gender()
        }
        
        self.results = results
        return results
    
    def generate_report(self, report_path: str = 'reports/hypothesis_testing.md') -> str:
        """Generate Markdown report from test results."""
        if not self.results:
            self.run_all_tests()
        
        report_dir = Path(report_path).parent
        report_dir.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write("# Hypothesis Testing Results\n\n")
            f.write(f"Generated: {pd.Timestamp.now()}\n\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            f.write("This report presents statistical hypothesis tests to validate risk differences across ")
            f.write("geographic regions (provinces, zip codes) and demographic segments (gender). ")
            f.write("Results inform pricing strategy and market segmentation decisions.\n\n")
            
            # Provinces
            f.write("## 1. Provinces - Risk Differences\n\n")
            f.write("### Null Hypothesis: There are no risk differences across provinces\n\n\n")
            
            prov_results = self.results['provinces']
            chi2_p = prov_results['chi2_p']
            sev_p = prov_results['severity_p']
            
            f.write(f"**Claim Frequency Test**: p={chi2_p:.4f} -> **{'reject H0' if chi2_p < self.alpha else 'fail to reject H0'}**\n\n")
            if chi2_p < self.alpha:
                summary = prov_results['summary']
                max_lr = summary['loss_ratio'].idxmax()
                min_lr = summary['loss_ratio'].idxmin()
                f.write(f"**REJECT H0**: Significant risk differences exist across provinces (p={chi2_p:.4f}). ")
                f.write(f"{max_lr} exhibits {summary.loc[max_lr, 'loss_ratio']:.1%} loss ratio vs {min_lr} at {summary.loc[min_lr, 'loss_ratio']:.1%}. ")
                f.write("Regional pricing adjustment recommended.\n\n")
            else:
                f.write("**FAIL TO REJECT H0**: No significant risk differences across provinces.\n\n")
            
            f.write(f"**Claim Severity Test**: p={sev_p:.4f} -> **{'reject H0' if sev_p < self.alpha else 'fail to reject H0'}**\n\n")
            if sev_p < self.alpha:
                summary = prov_results['summary']
                f.write(f"**REJECT H0**: Claim severity varies significantly by province (p={sev_p:.4f}). ")
                f.write(f"Average claim amount ranges from R{summary['avg_claim_given_claim'].min():,.0f} to R{summary['avg_claim_given_claim'].max():,.0f}. ")
                f.write("Consider province-specific excess amounts.\n\n")
            else:
                f.write("**FAIL TO REJECT H0**: No significant severity differences.\n\n")
            
            # Province summary table
            f.write("### Province Summary Statistics\n\n\n")
            summary = prov_results['summary']
            f.write(summary.to_markdown())
            f.write("\n\n\n")
            
            # Zip codes
            f.write("## 2. Zip Codes - Risk and Margin Differences\n\n")
            f.write("### Null Hypothesis 1: There are no risk differences between zip codes\n\n\n")
            
            zip_results = self.results['zipcodes']
            zip_chi2_p = zip_results['chi2_p']
            zip_sev_p = zip_results['severity_p']
            zip_margin_p = zip_results['margin_p']
            
            f.write(f"**Claim Frequency Test**: p={zip_chi2_p:.4f} -> **{'reject H0' if zip_chi2_p < self.alpha else 'fail to reject H0'}**\n\n")
            if zip_chi2_p < self.alpha:
                summary = zip_results['summary']
                if len(summary) > 0:
                    max_lr_idx = summary['loss_ratio'].idxmax()
                    f.write(f"**REJECT H0**: Significant risk differences exist between zip codes (p={zip_chi2_p:.4f}). ")
                    f.write(f"Postal code {max_lr_idx} shows {summary.loc[max_lr_idx, 'loss_ratio']:.1%} loss ratio. ")
                    f.write("Granular pricing by postal code may be warranted.\n\n")
            else:
                f.write("**FAIL TO REJECT H0**: No significant zip code differences.\n\n")
            
            f.write(f"**Claim Severity Test**: p={zip_sev_p:.4f} -> **{'reject H0' if zip_sev_p < self.alpha else 'fail to reject H0'}**\n\n")
            if zip_sev_p < self.alpha:
                f.write(f"**REJECT H0**: Statistically significant difference detected (p={zip_sev_p:.4f}). Further investigation recommended.\n\n")
            else:
                f.write("**FAIL TO REJECT H0**: No significant severity differences.\n\n")
            
            f.write("### Null Hypothesis 2: There is no significant margin (profit) difference between zip codes\n\n\n")
            f.write(f"**Margin Test**: p={zip_margin_p:.4f} -> **{'reject H0' if zip_margin_p < self.alpha else 'fail to reject H0'}**\n\n")
            if zip_margin_p < self.alpha:
                summary = zip_results['summary']
                if len(summary) > 0:
                    f.write(f"**REJECT H0**: Significant margin differences between zip codes (p={zip_margin_p:.4f}). ")
                    f.write(f"Average margin ranges from R{summary['margin'].min():,.0f} to R{summary['margin'].max():,.0f}. ")
                    f.write("Investigate high-risk postal codes for targeted interventions.\n\n")
            else:
                f.write("**FAIL TO REJECT H0**: No significant margin differences.\n\n")
            
            # Zip code summary
            f.write("### Top Zip Code Summary\n\n\n")
            summary = zip_results['summary']
            f.write(summary.to_markdown())
            f.write("\n\n\n")
            
            # Gender
            f.write("## 3. Gender - Risk Differences\n\n")
            f.write("### Null Hypothesis: There is no significant risk difference between Women and Men\n\n\n")
            
            gender_results = self.results['gender']
            gen_chi2_p = gender_results['chi2_p']
            gen_sev_p = gender_results['severity_p']
            
            f.write(f"**Claim Frequency Test**: p={gen_chi2_p:.4f} -> **{'reject H0' if gen_chi2_p < self.alpha else 'fail to reject H0'}**\n\n")
            if gen_chi2_p < self.alpha:
                summary = gender_results['summary']
                f.write(f"**REJECT H0**: Significant risk difference between genders (p={gen_chi2_p:.4f}). ")
                if 'Male' in summary.index and 'Female' in summary.index:
                    f.write(f"Male loss ratio: {summary.loc['Male', 'loss_ratio']:.1%}, Female loss ratio: {summary.loc['Female', 'loss_ratio']:.1%}. ")
                f.write("Consider gender-based pricing if legally permissible.\n\n")
            else:
                f.write("**FAIL TO REJECT H0**: No significant gender differences.\n\n")
            
            f.write(f"**Claim Severity Test**: p={gen_sev_p:.4f} -> **{'insufficient data' if pd.isna(gen_sev_p) else ('reject H0' if gen_sev_p < self.alpha else 'fail to reject H0')}**\n\n")
            if pd.isna(gen_sev_p):
                f.write("No statistically significant difference detected (p=nan). No immediate action required.\n\n")
            elif gen_sev_p < self.alpha:
                f.write(f"**REJECT H0**: Significant severity difference (p={gen_sev_p:.4f}).\n\n")
            else:
                f.write("**FAIL TO REJECT H0**: No significant severity difference.\n\n")
            
            # Gender summary
            f.write("### Gender Summary Statistics\n\n\n")
            summary = gender_results['summary']
            f.write(summary.to_markdown())
            f.write("\n\n\n")
            
            # Business recommendations
            f.write("## 4. Business Recommendations\n\n\n")
            f.write("### Immediate Actions (Next 30 Days)\n\n\n")
            f.write("Based on rejected null hypotheses, the following actions are recommended:\n\n\n")
            
            rejected = []
            if chi2_p < self.alpha or sev_p < self.alpha:
                rejected.append("1. **Province-based risk differences**: Implement segment-specific pricing adjustments.")
            if zip_chi2_p < self.alpha or zip_sev_p < self.alpha or zip_margin_p < self.alpha:
                rejected.append("2. **Zip code-based risk differences**: Implement segment-specific pricing adjustments.")
            if gen_chi2_p < self.alpha or gen_sev_p < self.alpha:
                rejected.append("3. **Gender-based risk differences**: Implement segment-specific pricing adjustments.")
            
            if rejected:
                for item in rejected:
                    f.write(f"{item}\n\n")
            else:
                f.write("No significant differences detected. Current pricing strategy appears appropriate.\n\n")
            
            f.write("### Strategic Recommendations\n\n\n")
            f.write("1. **Geographic Segmentation**: Develop province and postal code risk tiers for pricing.\n\n")
            f.write("2. **Risk-Based Pricing**: Adjust premiums based on statistically validated risk factors.\n\n")
            f.write("3. **Monitoring**: Establish quarterly review cycles to track risk evolution.\n\n")
            f.write("4. **Compliance**: Ensure all pricing adjustments comply with local insurance regulations.\n\n\n")
            
            # Methodology
            f.write("### Methodology Notes\n\n\n")
            f.write(f"- **Significance Level**: α = {self.alpha}\n\n")
            f.write("- **Claim Frequency**: Chi-square test of independence\n\n")
            f.write("- **Claim Severity**: ANOVA (if normal) or Kruskal-Wallis test (if non-normal)\n\n")
            f.write("- **Margin Analysis**: ANOVA (if normal) or Kruskal-Wallis test (if non-normal)\n\n")
            f.write("- **Gender Comparison**: t-test (if normal) or Mann-Whitney U test (if non-normal)\n\n")
            f.write("- Tests filter categories with <50 observations (10 for gender) to ensure statistical validity\n\n")
        
        return report_path


def run(data_path: str = 'data/processed/insurance_data_final_cleaned.parquet',
        report_path: str = 'reports/hypothesis_testing.md',
        alpha: float = 0.05) -> Dict:
    """
    Convenience function to run all hypothesis tests.
    
    Parameters
    ----------
    data_path : str
        Path to processed insurance data (parquet file)
    report_path : str
        Path to save the report
    alpha : float
        Significance level
        
    Returns
    -------
    dict
        Test results
    """
    print(f"Loading data from: {data_path}")
    data = pd.read_parquet(data_path)
    print(f"Data loaded: {data.shape[0]:,} rows, {data.shape[1]:,} columns\n")
    
    tester = HypothesisTester(data, alpha=alpha)
    results = tester.run_all_tests()
    
    print(f"\nGenerating report: {report_path}")
    tester.generate_report(report_path)
    
    return results


if __name__ == "__main__":
    # Example usage
    results = run()
    print("\nHypothesis testing complete!")

