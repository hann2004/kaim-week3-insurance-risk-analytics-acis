# Hypothesis Testing Results

Generated: 2025-12-09 03:59:27.000667


## Executive Summary

This report presents statistical hypothesis tests to validate risk differences across 
geographic regions (provinces, zip codes) and demographic segments (gender). 
Results inform pricing strategy and market segmentation decisions.

## 1. Provinces - Risk Differences

### Null Hypothesis: There are no risk differences across provinces


**Claim Frequency Test**: p=0.0000 -> **reject H0**

**REJECT H0**: Significant risk differences exist across provinces (p=0.0000). Gauteng exhibits 122.2% loss ratio vs Northern Cape at 28.3%. Regional pricing adjustment recommended.


**Claim Severity Test**: p=0.0000 -> **reject H0**

**REJECT H0**: Claim severity varies significantly by province (p=0.0000). Average claim amount ranges from R11,186 to R32,266. Consider province-specific excess amounts.


### Province Summary Statistics


| province      |   policies |   claim_rate |   loss_ratio |   avg_claim_given_claim |
|:--------------|-----------:|-------------:|-------------:|------------------------:|
| Gauteng       |     393865 |   0.00335648 |     1.22247  |                 22243.9 |
| Western Cape  |     170796 |   0.00216633 |     1.05967  |                 28095.8 |
| KwaZulu-Natal |     169781 |   0.00284484 |     1.08051  |                 29609.5 |
| North West    |     143287 |   0.00243567 |     0.790367 |                 16963.5 |
| Mpumalanga    |      52718 |   0.00242801 |     0.721147 |                 15979.6 |
| Eastern Cape  |      30336 |   0.00164821 |     0.633755 |                 27128.5 |
| Limpopo       |      24836 |   0.0026977  |     0.661199 |                 15171.3 |
| Free State    |       8099 |   0.00135819 |     0.680758 |                 32265.7 |
| Northern Cape |       6380 |   0.00125392 |     0.282699 |                 11186.3 |



## 2. Zip Codes - Risk and Margin Differences

### Null Hypothesis 1: There are no risk differences between zip codes


**Claim Frequency Test**: p=0.0000 -> **reject H0**

**REJECT H0**: Significant risk differences exist between zip codes (p=0.0000). Postal code 1342.0 shows 4239.3% loss ratio. Granular pricing by postal code may be warranted.


**Claim Severity Test**: p=0.0029 -> **reject H0**

**REJECT H0**: Statistically significant difference detected (p=0.0029). Further investigation recommended.


### Null Hypothesis 2: There is no significant margin (profit) difference between zip codes


**Margin Test**: p=0.0000 -> **reject H0**

**REJECT H0**: Significant margin differences between zip codes (p=0.0000). Average margin ranges from R-2,104 to R197. Investigate high-risk postal codes for targeted interventions.


### Top Zip Code Summary


|   postalcode |   policies |   claim_rate |   loss_ratio |   avg_claim_given_claim |
|-------------:|-----------:|-------------:|-------------:|------------------------:|
|         2000 |     133498 |   0.0036405  |     1.13124  |                 19196.4 |
|          122 |      49171 |   0.00427081 |     1.41786  |                 18162   |
|         7784 |      28585 |   0.00174917 |     1.2804   |                 35156.7 |
|          299 |      25546 |   0.00262272 |     0.646227 |                 13622.7 |
|         7405 |      18518 |   0.00156604 |     0.652492 |                 21002   |
|          458 |      13775 |   0.00232305 |     0.911519 |                 20160.3 |
|         8000 |      11794 |   0.00432423 |     1.12371  |                 33685.3 |
|         2196 |      11048 |   0.00289645 |     1.16934  |                 50877.8 |
|          470 |      10226 |   0.00430276 |     0.947783 |                 12946.8 |
|         7100 |      10161 |   0.00275563 |     0.89561  |                 21165.2 |



## 3. Gender - Risk Differences

### Null Hypothesis: There is no significant risk difference between Women and Men


**Claim Frequency Test**: p=0.0266 -> **reject H0**

**REJECT H0**: Significant risk difference between genders (p=0.0266). Male loss ratio: 82.2%, Female loss ratio: 82.2%. Consider gender-based pricing if legally permissible.


**Claim Severity Test**: p=nan -> **insufficient data**

No statistically significant difference detected (p=nan). No immediate action required.


### Gender Summary Statistics


| gender        |   policies |   claim_rate |   loss_ratio |   avg_claim_given_claim |
|:--------------|-----------:|-------------:|-------------:|------------------------:|
| Not specified |     940990 |   0.00283319 |     1.05953  |                 23530.7 |
| Male          |      42817 |   0.00219539 |     0.869344 |                 14858.6 |
| Female        |       6755 |   0.00207254 |     0.821879 |                 17874.7 |



## 4. Business Recommendations


### Immediate Actions (Next 30 Days)


Based on rejected null hypotheses, the following actions are recommended:


1. **Province-based risk differences**: Implement segment-specific pricing adjustments.

2. **Zip code-based risk differences**: Implement segment-specific pricing adjustments.

3. **Zip code margin differences**: Implement segment-specific pricing adjustments.

4. **Gender-based risk differences**: Implement segment-specific pricing adjustments.


### Strategic Recommendations


1. **Geographic Segmentation**: Develop province and postal code risk tiers for pricing.

2. **Risk-Based Pricing**: Adjust premiums based on statistically validated risk factors.

3. **Monitoring**: Establish quarterly review cycles to track risk evolution.

4. **Compliance**: Ensure all pricing adjustments comply with local insurance regulations.


### Methodology Notes


- **Significance Level**: α = 0.05

- **Claim Frequency**: Chi-square test of independence

- **Claim Severity**: Kruskal-Wallis test (non-parametric, handles non-normal distributions)

- **Margin Analysis**: Kruskal-Wallis test comparing profit margins

- **Gender Comparison**: Mann-Whitney U test for two-group comparison

- Tests filter categories with <50 observations (10 for gender) to ensure statistical validity
