"""Modeling utilities for ACIS insurance analytics (Task-4).

This module trains:
- Claim frequency model (binary classification, has_claim)
- Claim severity model (regression on positive claims)
- Premium regression model (calculated premium)

It also produces a Markdown report with metrics and SHAP feature importance
when available.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# Column choices to keep things resilient to casing/cleaning variants.
COL_CHOICES = {
    "total_premium": ["totalpremium", "TotalPremium"],
    "total_claims": ["totalclaims", "TotalClaims"],
    "calc_premium": ["calculatedpremiumperterm", "CalculatedPremiumPerTerm"],
}


def _pick_column(df: pd.DataFrame, candidates: List[str]) -> str:
    for name in candidates:
        if name in df.columns:
            return name
    raise KeyError(f"Missing expected columns: {candidates}")


def load_dataset(path: str, sample_rows: Optional[int] = None) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if sample_rows is not None and sample_rows > 0 and len(df) > sample_rows:
        df = df.sample(n=sample_rows, random_state=42)
    premium_col = _pick_column(df, COL_CHOICES["total_premium"])
    claims_col = _pick_column(df, COL_CHOICES["total_claims"])

    df = df.copy()
    df["has_claim"] = df[claims_col].fillna(0) > 0
    df["claims_positive"] = df[claims_col].fillna(0)
    df["margin"] = df[premium_col].fillna(0) - df[claims_col].fillna(0)
    return df


def _build_preprocess(df: pd.DataFrame, target: str, drop_cols: List[str]) -> Tuple[Pipeline, List[str]]:
    feature_df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    if target in numeric_cols:
        numeric_cols.remove(target)
    cat_cols = feature_df.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=0.01),
            ),
        ]
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, cat_cols),
        ]
    )
    return preprocess, cat_cols + numeric_cols


def _build_feature_names(preprocess: ColumnTransformer) -> List[str]:
    feature_names: List[str] = []
    # Numeric transformer names
    for name, transformer, cols in preprocess.transformers_:
        if name == "num":
            feature_names.extend(cols)
        elif name == "cat":
            encoder = transformer.named_steps.get("encoder")
            if encoder is not None:
                cat_names = encoder.get_feature_names_out(cols)
                feature_names.extend(list(cat_names))
    return feature_names


def _evaluate_classification(model, X_train, X_test, y_train, y_test) -> Dict[str, float]:
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)
    preds = (proba >= 0.5).astype(int)
    return {
        "roc_auc": roc_auc_score(y_test, proba),
        "f1": f1_score(y_test, preds, zero_division=0),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "accuracy": accuracy_score(y_test, preds),
    }


def _evaluate_regression(model, X_train, X_test, y_train, y_test) -> Dict[str, float]:
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    # squared=False not available in older sklearn; compute manually
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    r2 = r2_score(y_test, preds)
    return {"rmse": rmse, "r2": r2}


def run_frequency_model(df: pd.DataFrame, target: str = "has_claim") -> Dict:
    drop_cols = [target]
    preprocess, _ = _build_preprocess(df, target, drop_cols)

    X = df.drop(columns=[target])
    y = df[target].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    log_reg = Pipeline(
        steps=[("prep", preprocess), ("model", LogisticRegression(max_iter=500, n_jobs=-1, class_weight="balanced"))]
    )
    rf_clf = Pipeline(
        steps=[(
            "prep",
            preprocess,
        ), (
            "model",
            RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1, class_weight="balanced_subsample"),
        )]
    )

    return {
        "logistic_regression": _evaluate_classification(log_reg, X_train, X_test, y_train, y_test),
        "random_forest": _evaluate_classification(rf_clf, X_train, X_test, y_train, y_test),
        "fitted_models": {"logistic_regression": log_reg, "random_forest": rf_clf},
    }


def run_severity_model(df: pd.DataFrame, claims_col: str) -> Dict:
    subset = df[df[claims_col] > 0].copy()
    if subset.empty:
        return {"metrics": {}, "fitted_models": {}}

    target = claims_col
    drop_cols = [target]
    preprocess, _ = _build_preprocess(subset, target, drop_cols)

    X = subset.drop(columns=[target])
    y = subset[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lin_reg = Pipeline(steps=[("prep", preprocess), ("model", LinearRegression())])
    rf_reg = Pipeline(
        steps=[(
            "prep",
            preprocess,
        ), (
            "model",
            RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        )]
    )

    return {
        "linear_regression": _evaluate_regression(lin_reg, X_train, X_test, y_train, y_test),
        "random_forest": _evaluate_regression(rf_reg, X_train, X_test, y_train, y_test),
        "fitted_models": {"linear_regression": lin_reg, "random_forest": rf_reg},
    }


def run_premium_model(df: pd.DataFrame, premium_col: str) -> Dict:
    target = premium_col
    drop_cols = [target]
    preprocess, _ = _build_preprocess(df, target, drop_cols)

    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lin_reg = Pipeline(steps=[("prep", preprocess), ("model", LinearRegression())])
    rf_reg = Pipeline(
        steps=[(
            "prep",
            preprocess,
        ), (
            "model",
            RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        )]
    )

    return {
        "linear_regression": _evaluate_regression(lin_reg, X_train, X_test, y_train, y_test),
        "random_forest": _evaluate_regression(rf_reg, X_train, X_test, y_train, y_test),
        "fitted_models": {"linear_regression": lin_reg, "random_forest": rf_reg},
    }


def _shap_summary(model, X_sample: pd.DataFrame, feature_names: List[str], max_display: int = 10) -> List[Tuple[str, float]]:
    try:
        import shap

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # binary classifier
        mean_abs = np.abs(shap_values).mean(axis=0)
        names = feature_names if feature_names else [f"f{i}" for i in range(mean_abs.shape[0])]
        sorted_idx = np.argsort(mean_abs)[::-1][:max_display]
        return [(names[i], float(mean_abs[i])) for i in sorted_idx]
    except Exception:
        return []


def write_report(results: Dict, report_path: str) -> str:
    lines = []
    lines.append("# Modeling Report (Task-4)\n")
    lines.append(f"Generated: {pd.Timestamp.now()}\n")

    def fmt_table(metrics: Dict[str, Dict[str, float]]) -> str:
        if not metrics:
            return "No data"
        keys = sorted({k for m in metrics.values() for k in m.keys()})
        header = "| model | " + " | ".join(keys) + " |\n"
        sep = "|" + "---|" * (len(keys) + 1) + "\n"
        rows = []
        for name, vals in metrics.items():
            row = [name] + [f"{vals.get(k, np.nan):.4f}" if k in vals else "" for k in keys]
            rows.append("| " + " | ".join(row) + " |\n")
        return header + sep + "".join(rows)

    lines.append("## Claim Frequency (has_claim)\n")
    lines.append(fmt_table(results.get("frequency", {})))
    lines.append("\n")

    lines.append("## Claim Severity (given claim)\n")
    lines.append(fmt_table(results.get("severity", {})))
    lines.append("\n")

    lines.append("## Premium Regression (calculated premium)\n")
    lines.append(fmt_table(results.get("premium", {})))
    lines.append("\n")

    lines.append("## SHAP Top Features\n")
    shap_info = results.get("shap", {})
    for section, feats in shap_info.items():
        lines.append(f"### {section}\n")
        if not feats:
            lines.append("(SHAP not available or failed)\n\n")
            continue
        lines.append("| feature | mean_abs_shap |\n|---|---|\n")
        for name, val in feats:
            lines.append(f"| {name} | {val:.6f} |\n")
        lines.append("\n")

    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text("\n".join(lines))
    return str(report_path)


def run(
    data_path: str = "data/processed/insurance_data_final_cleaned.parquet",
    report_path: str = "reports/modeling_report.md",
    enable_shap: bool = False,
    sample_rows: Optional[int] = None,
) -> Dict:
    df = load_dataset(data_path, sample_rows=sample_rows)

    premium_col = _pick_column(df, COL_CHOICES["total_premium"])
    claims_col = _pick_column(df, COL_CHOICES["total_claims"])
    calc_prem_col = _pick_column(df, COL_CHOICES["calc_premium"])

    frequency = run_frequency_model(df)
    severity = run_severity_model(df, claims_col=claims_col)
    premium = run_premium_model(df, premium_col=calc_prem_col)

    shap_results = {}
    if enable_shap:
        # SHAP for frequency RF model
        try:
            rf_model = frequency["fitted_models"]["random_forest"]
            sample = df.sample(n=min(200, len(df)), random_state=42)
            prep = rf_model.named_steps["prep"]
            X_sample = prep.transform(sample.drop(columns=["has_claim"]))
            feature_names = _build_feature_names(prep)
            shap_results["frequency_random_forest"] = _shap_summary(
                rf_model.named_steps["model"], pd.DataFrame(X_sample, columns=feature_names), feature_names
            )
        except Exception:
            shap_results["frequency_random_forest"] = []

        # SHAP for premium RF model
        try:
            rf_prem = premium["fitted_models"]["random_forest"]
            sample = df.sample(n=min(200, len(df)), random_state=42)
            prep = rf_prem.named_steps["prep"]
            X_sample = prep.transform(sample.drop(columns=[calc_prem_col]))
            feature_names = _build_feature_names(prep)
            shap_results["premium_random_forest"] = _shap_summary(
                rf_prem.named_steps["model"], pd.DataFrame(X_sample, columns=feature_names), feature_names
            )
        except Exception:
            shap_results["premium_random_forest"] = []

    results = {
        "frequency": {k: v for k, v in frequency.items() if k != "fitted_models"},
        "severity": {k: v for k, v in severity.items() if k != "fitted_models"},
        "premium": {k: v for k, v in premium.items() if k != "fitted_models"},
        "shap": shap_results,
    }

    write_report(results, report_path)
    return results


if __name__ == "__main__":
    run()