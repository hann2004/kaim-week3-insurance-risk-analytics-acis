# Modeling Report (Task-4)

Generated: 2025-12-09 05:14:34.226756

## Claim Frequency (has_claim)

| model | accuracy | f1 | precision | recall | roc_auc |
|---|---|---|---|---|---|
| logistic_regression | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| random_forest | 0.9967 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |



## Claim Severity (given claim)

| model | r2 | rmse |
|---|---|---|
| linear_regression | nan | 39784.8758 |
| random_forest | nan | 35745.9518 |



## Premium Regression (calculated premium)

| model | r2 | rmse |
|---|---|---|
| linear_regression | -1.6667 | 364.6704 |
| random_forest | 0.9824 | 29.5836 |



## SHAP Top Features

### frequency_random_forest

(SHAP not available or failed)


### premium_random_forest

| feature | mean_abs_shap |
|---|---|

| covertype_Own Damage | 106.726955 |

| covergroup_Comprehensive - Taxi | 22.606982 |

| suminsured | 21.676129 |

| covercategory_Third Party | 21.237280 |

| covertype_Third Party | 15.913942 |

| registrationyear | 5.118543 |

| totalpremium | 4.770519 |

| margin | 3.598070 |

| section_Motor Comprehensive | 3.197706 |

| cubiccapacity | 2.324690 |


