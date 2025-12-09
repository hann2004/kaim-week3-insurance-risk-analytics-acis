"""Hypothesis testing utilities for ACIS insurance risk analysis.

Implements chi-square and non-parametric tests to assess differences in
claim frequency, claim severity, and margin across key segments.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# Columns that may exist in the cleaned datasets. Fallback order matters.
COL_CHOICES = {
    "province": ["province", "Province"],
    "zipcode": ["postalcode", "zip", "zipcode", "zip_code"],
    "gender": ["gender_clean", "gender", "Gender"],
    "total_premium": ["totalpremium", "TotalPremium"],
    "total_claims": ["totalclaims", "TotalClaims"],
}


def _pick_column(df: pd.DataFrame, candidates: List[str]) -> str:
    """Return the first column name that exists in the DataFrame."""
    for name in candidates:
        if name in df.columns:
            return name
    raise KeyError(f"None of the candidate columns are present: {candidates}")


def load_dataset(path: str) -> pd.DataFrame:
    """Load the parquet dataset and derive helper columns."""
    df = pd.read_parquet(path)

    premium_col = _pick_column(df, COL_CHOICES["total_premium"])
    claims_col = _pick_column(df, COL_CHOICES["total_claims"])

    df = df.copy()
    df["has_claim"] = df[claims_col].fillna(0) > 0
    df["margin"] = df[premium_col].fillna(0) - df[claims_col].fillna(0)
    df["claims_positive"] = df[claims_col].fillna(0)
    return df


def chi_square_frequency(df: pd.DataFrame, category: str, min_count: int = 50) -> Tuple[float, float, int]:
    """Chi-square test for association between category and claim frequency."""
    filtered = _filter_by_count(df, category, min_count)
    table = pd.crosstab(filtered[category], filtered["has_claim"])
    chi2, p, _, _ = stats.chi2_contingency(table)
    return chi2, p, table.shape[0]


def kruskal_by_group(df: pd.DataFrame, category: str, value_col: str, min_count: int = 50) -> Tuple[float, float, int]:
    """Kruskal-Wallis test comparing a numeric value across category groups."""
    filtered = _filter_by_count(df, category, min_count)
    groups = [grp[value_col].dropna().values for _, grp in filtered.groupby(category)]
    groups = [g for g in groups if len(g) > 0]
    if len(groups) < 2:
        return np.nan, np.nan, len(groups)
    stat, p = stats.kruskal(*groups)
    return stat, p, len(groups)


def mannwhitney_two_groups(df: pd.DataFrame, category: str, value_col: str) -> Tuple[float, float]:
    """Mann-Whitney test for two-group comparison."""
    groups = [grp[value_col].dropna().values for _, grp in df.groupby(category)]
    if len(groups) != 2 or any(len(g) == 0 for g in groups):
        return np.nan, np.nan
    stat, p = stats.mannwhitneyu(groups[0], groups[1], alternative="two-sided")
    return stat, p


def _filter_by_count(df: pd.DataFrame, category: str, min_count: int) -> pd.DataFrame:
    counts = df[category].value_counts()
    keep = counts[counts >= min_count].index
    return df[df[category].isin(keep)].copy()


def summarize_metric(df: pd.DataFrame, category: str, claims_col: str, premium_col: str) -> pd.DataFrame:
    """Compute loss ratio, claim rate, and severity per category."""
    grouped = df.groupby(category).agg(
        policies=(claims_col, "size"),
        claim_rate=("has_claim", "mean"),
        loss_ratio=(claims_col, "sum"),
        premium_sum=(premium_col, "sum"),
        avg_claim_given_claim=(claims_col, lambda s: s[s > 0].mean() if (s > 0).any() else 0),
    )
    grouped["loss_ratio"] = grouped["loss_ratio"] / grouped["premium_sum"].replace(0, np.nan)
    grouped = grouped.drop(columns=["premium_sum"])
    return grouped.reset_index()


def run_hypothesis_tests(df: pd.DataFrame) -> Dict:
    """Execute hypothesis tests and return structured results."""
    province_col = _pick_column(df, COL_CHOICES["province"])
    zip_col = _pick_column(df, COL_CHOICES["zipcode"])
    gender_col = _pick_column(df, COL_CHOICES["gender"])
    premium_col = _pick_column(df, COL_CHOICES["total_premium"])
    claims_col = _pick_column(df, COL_CHOICES["total_claims"])

    results = {}

    # Provinces: frequency and severity
    chi2, p, k = chi_square_frequency(df, province_col)
    sev_stat, sev_p, sev_groups = kruskal_by_group(df[df["has_claim"]], province_col, claims_col)
    results["provinces"] = {
        "chi2_stat": chi2,
        "chi2_p": p,
        "chi2_groups": k,
        "severity_stat": sev_stat,
        "severity_p": sev_p,
        "severity_groups": sev_groups,
        "summary": summarize_metric(df, province_col, claims_col, premium_col),
    }

    # Zip codes: frequency and severity
    chi2, p, k = chi_square_frequency(df, zip_col)
    sev_stat, sev_p, sev_groups = kruskal_by_group(df[df["has_claim"]], zip_col, claims_col)
    margin_stat, margin_p, margin_groups = kruskal_by_group(df, zip_col, "margin")
    results["zipcodes"] = {
        "chi2_stat": chi2,
        "chi2_p": p,
        "chi2_groups": k,
        "severity_stat": sev_stat,
        "severity_p": sev_p,
        "severity_groups": sev_groups,
        "margin_stat": margin_stat,
        "margin_p": margin_p,
        "margin_groups": margin_groups,
        "summary": summarize_metric(df, zip_col, claims_col, premium_col),
    }

    # Gender: frequency and severity (two-group)
    chi2, p, _ = chi_square_frequency(df, gender_col, min_count=10)
    mw_stat, mw_p = mannwhitney_two_groups(df[df["has_claim"]], gender_col, claims_col)
    results["gender"] = {
        "chi2_stat": chi2,
        "chi2_p": p,
        "severity_stat": mw_stat,
        "severity_p": mw_p,
        "summary": summarize_metric(df, gender_col, claims_col, premium_col),
    }

    return results


def write_report(results: Dict, output_path: str, df: pd.DataFrame = None) -> str:
    """Write a Markdown report summarizing hypothesis test outcomes with business recommendations."""
    lines = []
    lines.append("# Hypothesis Testing Results\n")
    lines.append(f"Generated: {pd.Timestamp.now()}\n")
    lines.append("\n## Executive Summary\n")
    lines.append("This report presents statistical hypothesis tests to validate risk differences across ")
    lines.append("geographic regions (provinces, zip codes) and demographic segments (gender). ")
    lines.append("Results inform pricing strategy and market segmentation decisions.\n")

    def decision(p: float) -> str:
        if np.isnan(p):
            return "insufficient data"
        return "reject H0" if p < 0.05 else "fail to reject H0"

    def get_business_interpretation(category: str, test_name: str, p_value: float, summary_df: pd.DataFrame, 
                                   margin_data: pd.DataFrame = None) -> str:
        """Generate business interpretation for test results."""
        if np.isnan(p_value) or p_value >= 0.05:
            return f"No statistically significant difference detected (p={p_value:.4f}). No immediate action required."
        
        # For rejected hypotheses, provide specific insights
        if category == "provinces":
            if test_name == "frequency":
                worst = summary_df.nlargest(1, "loss_ratio")
                best = summary_df.nsmallest(1, "loss_ratio")
                if len(worst) > 0 and len(best) > 0:
                    worst_prov = worst.iloc[0]
                    best_prov = best.iloc[0]
                    cat_col = summary_df.columns[0]  # First column is the category
                    return (f"**REJECT H0**: Significant risk differences exist across provinces (p={p_value:.4f}). "
                           f"{worst_prov[cat_col]} exhibits {worst_prov['loss_ratio']:.1%} loss ratio vs "
                           f"{best_prov[cat_col]} at {best_prov['loss_ratio']:.1%}. Regional pricing adjustment recommended.")
            elif test_name == "severity":
                worst = summary_df.nlargest(1, "avg_claim_given_claim")
                best = summary_df.nsmallest(1, "avg_claim_given_claim")
                if len(worst) > 0 and len(best) > 0:
                    return (f"**REJECT H0**: Claim severity varies significantly by province (p={p_value:.4f}). "
                           f"Average claim amount ranges from R{best.iloc[0]['avg_claim_given_claim']:,.0f} to "
                           f"R{worst.iloc[0]['avg_claim_given_claim']:,.0f}. Consider province-specific excess amounts.")
        
        elif category == "zipcodes":
            if test_name == "frequency":
                worst = summary_df.nlargest(1, "loss_ratio")
                best = summary_df.nsmallest(1, "loss_ratio")
                if len(worst) > 0 and len(best) > 0:
                    cat_col = summary_df.columns[0]
                    return (f"**REJECT H0**: Significant risk differences exist between zip codes (p={p_value:.4f}). "
                           f"Postal code {worst.iloc[0][cat_col]} shows {worst.iloc[0]['loss_ratio']:.1%} loss ratio. "
                           f"Granular pricing by postal code may be warranted.")
            elif test_name == "margin":
                # For margin, we need to calculate from the original data
                if margin_data is not None and len(margin_data) > 0:
                    margin_summary = margin_data.groupby(margin_data.columns[0])['margin'].agg(['mean', 'count'])
                    margin_summary = margin_summary[margin_summary['count'] >= 10].sort_values('mean')
                    if len(margin_summary) > 0:
                        worst_margin = margin_summary.iloc[0]
                        best_margin = margin_summary.iloc[-1]
                        return (f"**REJECT H0**: Significant margin differences between zip codes (p={p_value:.4f}). "
                               f"Average margin ranges from R{worst_margin['mean']:,.0f} to R{best_margin['mean']:,.0f}. "
                               f"Investigate high-risk postal codes for targeted interventions.")
                return (f"**REJECT H0**: Significant margin differences between zip codes (p={p_value:.4f}). "
                       f"Profitability varies substantially. Investigate high-risk postal codes for targeted interventions.")
        
        elif category == "gender":
            if len(summary_df) >= 2:
                cat_col = summary_df.columns[0]
                male = summary_df[summary_df[cat_col].astype(str).str.contains('M|Male', case=False, na=False)]
                female = summary_df[summary_df[cat_col].astype(str).str.contains('F|Female', case=False, na=False)]
                if len(male) > 0 and len(female) > 0:
                    m_lr = male.iloc[0]['loss_ratio']
                    f_lr = female.iloc[0]['loss_ratio']
                    return (f"**REJECT H0**: Significant risk difference between genders (p={p_value:.4f}). "
                           f"Male loss ratio: {m_lr:.1%}, Female loss ratio: {f_lr:.1%}. "
                           f"Consider gender-based pricing if legally permissible.")
        
        return f"**REJECT H0**: Statistically significant difference detected (p={p_value:.4f}). Further investigation recommended."

    # Provinces
    prov = results["provinces"]
    lines.append("## 1. Provinces - Risk Differences\n")
    lines.append("### Null Hypothesis: There are no risk differences across provinces\n\n")
    
    freq_decision = decision(prov['chi2_p'])
    sev_decision = decision(prov['severity_p'])
    
    lines.append(f"**Claim Frequency Test**: p={prov['chi2_p']:.4f} -> **{freq_decision}**\n")
    lines.append(f"{get_business_interpretation('provinces', 'frequency', prov['chi2_p'], prov['summary'])}\n\n")
    
    lines.append(f"**Claim Severity Test**: p={prov['severity_p']:.4f} -> **{sev_decision}**\n")
    lines.append(f"{get_business_interpretation('provinces', 'severity', prov['severity_p'], prov['summary'])}\n\n")
    
    lines.append("### Province Summary Statistics\n\n")
    lines.append(prov["summary"].sort_values("policies", ascending=False).head(10).to_markdown(index=False))
    lines.append("\n\n")

    # Zip codes
    zc = results["zipcodes"]
    lines.append("## 2. Zip Codes - Risk and Margin Differences\n")
    lines.append("### Null Hypothesis 1: There are no risk differences between zip codes\n\n")
    
    zc_freq_decision = decision(zc['chi2_p'])
    zc_sev_decision = decision(zc['severity_p'])
    zc_margin_decision = decision(zc['margin_p'])
    
    lines.append(f"**Claim Frequency Test**: p={zc['chi2_p']:.4f} -> **{zc_freq_decision}**\n")
    lines.append(f"{get_business_interpretation('zipcodes', 'frequency', zc['chi2_p'], zc['summary'])}\n\n")
    
    lines.append(f"**Claim Severity Test**: p={zc['severity_p']:.4f} -> **{zc_sev_decision}**\n")
    lines.append(f"{get_business_interpretation('zipcodes', 'severity', zc['severity_p'], zc['summary'])}\n\n")
    
    lines.append("### Null Hypothesis 2: There is no significant margin (profit) difference between zip codes\n\n")
    lines.append(f"**Margin Test**: p={zc['margin_p']:.4f} -> **{zc_margin_decision}**\n")
    # For margin interpretation, calculate from original dataframe if available
    margin_data = None
    if df is not None:
        zip_col = df.columns[df.columns.str.contains('postal|zip', case=False)][0] if len(df.columns[df.columns.str.contains('postal|zip', case=False)]) > 0 else None
        if zip_col:
            margin_data = df[[zip_col, 'margin']].copy()
    lines.append(f"{get_business_interpretation('zipcodes', 'margin', zc['margin_p'], zc['summary'], margin_data)}\n\n")
    
    lines.append("### Top Zip Code Summary\n\n")
    lines.append(zc["summary"].sort_values("policies", ascending=False).head(10).to_markdown(index=False))
    lines.append("\n\n")

    # Gender
    gen = results["gender"]
    lines.append("## 3. Gender - Risk Differences\n")
    lines.append("### Null Hypothesis: There is no significant risk difference between Women and Men\n\n")
    
    gen_freq_decision = decision(gen['chi2_p'])
    gen_sev_decision = decision(gen['severity_p'])
    
    lines.append(f"**Claim Frequency Test**: p={gen['chi2_p']:.4f} -> **{gen_freq_decision}**\n")
    lines.append(f"{get_business_interpretation('gender', 'frequency', gen['chi2_p'], gen['summary'])}\n\n")
    
    lines.append(f"**Claim Severity Test**: p={gen['severity_p']:.4f} -> **{gen_sev_decision}**\n")
    lines.append(f"{get_business_interpretation('gender', 'severity', gen['severity_p'], gen['summary'])}\n\n")
    
    lines.append("### Gender Summary Statistics\n\n")
    lines.append(gen["summary"].sort_values("policies", ascending=False).to_markdown(index=False))
    lines.append("\n\n")

    # Business Recommendations Section
    lines.append("## 4. Business Recommendations\n\n")
    lines.append("### Immediate Actions (Next 30 Days)\n\n")
    
    rejected_tests = []
    if prov['chi2_p'] < 0.05:
        rejected_tests.append("Province-based risk differences")
    if zc['chi2_p'] < 0.05:
        rejected_tests.append("Zip code-based risk differences")
    if zc['margin_p'] < 0.05:
        rejected_tests.append("Zip code margin differences")
    if gen['chi2_p'] < 0.05:
        rejected_tests.append("Gender-based risk differences")
    
    if rejected_tests:
        lines.append("Based on rejected null hypotheses, the following actions are recommended:\n\n")
        for i, test in enumerate(rejected_tests, 1):
            lines.append(f"{i}. **{test}**: Implement segment-specific pricing adjustments.\n")
        
        lines.append("\n### Strategic Recommendations\n\n")
        lines.append("1. **Geographic Segmentation**: Develop province and postal code risk tiers for pricing.\n")
        lines.append("2. **Risk-Based Pricing**: Adjust premiums based on statistically validated risk factors.\n")
        lines.append("3. **Monitoring**: Establish quarterly review cycles to track risk evolution.\n")
        lines.append("4. **Compliance**: Ensure all pricing adjustments comply with local insurance regulations.\n")
    else:
        lines.append("No statistically significant differences detected. Current pricing strategy appears appropriate.\n")
        lines.append("Continue monitoring and reassess quarterly.\n")
    
    lines.append("\n### Methodology Notes\n\n")
    lines.append("- **Significance Level**: α = 0.05\n")
    lines.append("- **Claim Frequency**: Chi-square test of independence\n")
    lines.append("- **Claim Severity**: Kruskal-Wallis test (non-parametric, handles non-normal distributions)\n")
    lines.append("- **Margin Analysis**: Kruskal-Wallis test comparing profit margins\n")
    lines.append("- **Gender Comparison**: Mann-Whitney U test for two-group comparison\n")
    lines.append("- Tests filter categories with <50 observations (10 for gender) to ensure statistical validity\n")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(lines))
    return str(output_path)


def run(path: str = "data/processed/insurance_data_final_cleaned.parquet", report_path: str = "reports/hypothesis_testing.md") -> Dict:
    """Convenience wrapper to load data, run tests, and write the report."""
    df = load_dataset(path)
    results = run_hypothesis_tests(df)
    write_report(results, report_path, df)
    return results


if __name__ == "__main__":
    run()