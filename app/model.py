"""Interpretable logistic churn-risk scorer for B2B SaaS accounts.

Uses an explicit logistic model with auditable feature weights rather than an
opaque black-box pickle. Every prediction returns a calibrated probability,
a risk band (LOW / MEDIUM / HIGH), and the top contributing risk factors.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, Field

MODEL_VERSION = "1.1.0"

# Log-odds weights. Positive values increase churn risk; negative values reduce it.
INTERCEPT = -0.65
FEATURE_WEIGHTS: dict[str, float] = {
    "failed_payments_90d": 1.15,       # per failed payment
    "days_since_last_login": 0.045,    # per inactive day (capped at 90)
    "support_tickets_30d": 0.28,       # per recent ticket (capped at 15)
    "low_monthly_logins": 0.16,        # per login below 15/month
    "low_feature_adoption": 0.025,     # per percentage point below 60%
    "tenure_months": -0.045,           # per month of tenure (capped at 36)
    "annual_contract": -0.85,          # 1 if annual, 0 if monthly
}

RISK_BAND_THRESHOLDS: dict[str, float] = {
    "HIGH": 0.65,
    "MEDIUM": 0.35,
    "LOW": 0.0,
}


class CustomerFeatures(BaseModel):
    customer_id: str | None = Field(None, max_length=64, examples=["acct-1042"])
    tenure_months: int = Field(..., ge=0, le=240, examples=[4])
    monthly_logins: int = Field(..., ge=0, le=5000, examples=[3])
    support_tickets_30d: int = Field(..., ge=0, le=200, examples=[4])
    failed_payments_90d: int = Field(..., ge=0, le=20, examples=[2])
    days_since_last_login: int = Field(..., ge=0, le=365, examples=[24])
    feature_adoption_pct: float = Field(..., ge=0.0, le=100.0, examples=[25.0])
    contract_type: Literal["monthly", "annual"] = "monthly"
    mrr_aud: float = Field(99.0, ge=0.0, le=1_000_000.0, examples=[299.0])


class ChurnPrediction(BaseModel):
    customer_id: str | None = None
    churn_probability: float = Field(..., ge=0.0, le=1.0)
    risk_band: Literal["LOW", "MEDIUM", "HIGH"]
    top_risk_factors: list[str]
    model_version: str = MODEL_VERSION


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, x))))


def _contributions(f: CustomerFeatures) -> list[tuple[str, float, str]]:
    """Return (feature_key, log_odds_delta, human_label) for each feature."""
    return [
        (
            "failed_payments_90d",
            FEATURE_WEIGHTS["failed_payments_90d"] * min(f.failed_payments_90d, 5),
            f"{f.failed_payments_90d} failed payment(s) in last 90 days",
        ),
        (
            "days_since_last_login",
            FEATURE_WEIGHTS["days_since_last_login"] * min(f.days_since_last_login, 90),
            f"{f.days_since_last_login} days since last login",
        ),
        (
            "support_tickets_30d",
            FEATURE_WEIGHTS["support_tickets_30d"] * min(f.support_tickets_30d, 15),
            f"{f.support_tickets_30d} support ticket(s) in last 30 days",
        ),
        (
            "low_monthly_logins",
            FEATURE_WEIGHTS["low_monthly_logins"] * max(0, 15 - f.monthly_logins),
            f"Low usage ({f.monthly_logins} monthly logins)",
        ),
        (
            "low_feature_adoption",
            FEATURE_WEIGHTS["low_feature_adoption"] * max(0.0, 60.0 - f.feature_adoption_pct),
            f"Low feature adoption ({f.feature_adoption_pct:.0f}%)",
        ),
        (
            "tenure_months",
            FEATURE_WEIGHTS["tenure_months"] * min(f.tenure_months, 36),
            f"Established tenure ({f.tenure_months} months)",
        ),
        (
            "annual_contract",
            FEATURE_WEIGHTS["annual_contract"] * (1 if f.contract_type == "annual" else 0),
            "Annual contract commitment",
        ),
    ]


def predict_churn(features: CustomerFeatures) -> ChurnPrediction:
    contribs = _contributions(features)
    logit = INTERCEPT + sum(delta for _, delta, _ in contribs)
    prob = round(_sigmoid(logit), 3)

    if prob >= RISK_BAND_THRESHOLDS["HIGH"]:
        band: Literal["LOW", "MEDIUM", "HIGH"] = "HIGH"
    elif prob >= RISK_BAND_THRESHOLDS["MEDIUM"]:
        band = "MEDIUM"
    else:
        band = "LOW"

    positive = sorted((c for c in contribs if c[1] > 0.25), key=lambda x: x[1], reverse=True)
    top_factors = [label for _, _, label in positive[:3]] or ["No elevated churn signals"]

    return ChurnPrediction(
        customer_id=features.customer_id,
        churn_probability=prob,
        risk_band=band,
        top_risk_factors=top_factors,
        model_version=MODEL_VERSION,
    )


# Small labelled benchmark set of synthetic B2B SaaS accounts (churned: 1 or 0).
BENCHMARK_ACCOUNTS: list[tuple[CustomerFeatures, int]] = [
    (CustomerFeatures(customer_id="bm-01", tenure_months=2, monthly_logins=1, support_tickets_30d=5, failed_payments_90d=2, days_since_last_login=30, feature_adoption_pct=15, contract_type="monthly"), 1),
    (CustomerFeatures(customer_id="bm-02", tenure_months=3, monthly_logins=2, support_tickets_30d=3, failed_payments_90d=1, days_since_last_login=25, feature_adoption_pct=20, contract_type="monthly"), 1),
    (CustomerFeatures(customer_id="bm-03", tenure_months=5, monthly_logins=0, support_tickets_30d=4, failed_payments_90d=0, days_since_last_login=45, feature_adoption_pct=10, contract_type="monthly"), 1),
    (CustomerFeatures(customer_id="bm-04", tenure_months=1, monthly_logins=3, support_tickets_30d=2, failed_payments_90d=1, days_since_last_login=18, feature_adoption_pct=25, contract_type="monthly"), 1),
    (CustomerFeatures(customer_id="bm-05", tenure_months=6, monthly_logins=4, support_tickets_30d=6, failed_payments_90d=1, days_since_last_login=22, feature_adoption_pct=30, contract_type="monthly"), 1),
    (CustomerFeatures(customer_id="bm-06", tenure_months=4, monthly_logins=2, support_tickets_30d=1, failed_payments_90d=2, days_since_last_login=14, feature_adoption_pct=35, contract_type="monthly"), 1),
    (CustomerFeatures(customer_id="bm-07", tenure_months=24, monthly_logins=42, support_tickets_30d=0, failed_payments_90d=0, days_since_last_login=1, feature_adoption_pct=85, contract_type="annual"), 0),
    (CustomerFeatures(customer_id="bm-08", tenure_months=18, monthly_logins=28, support_tickets_30d=1, failed_payments_90d=0, days_since_last_login=2, feature_adoption_pct=78, contract_type="annual"), 0),
    (CustomerFeatures(customer_id="bm-09", tenure_months=14, monthly_logins=20, support_tickets_30d=0, failed_payments_90d=0, days_since_last_login=3, feature_adoption_pct=70, contract_type="monthly"), 0),
    (CustomerFeatures(customer_id="bm-10", tenure_months=30, monthly_logins=55, support_tickets_30d=1, failed_payments_90d=0, days_since_last_login=0, feature_adoption_pct=92, contract_type="annual"), 0),
    (CustomerFeatures(customer_id="bm-11", tenure_months=12, monthly_logins=18, support_tickets_30d=1, failed_payments_90d=0, days_since_last_login=4, feature_adoption_pct=65, contract_type="annual"), 0),
    (CustomerFeatures(customer_id="bm-12", tenure_months=16, monthly_logins=22, support_tickets_30d=0, failed_payments_90d=0, days_since_last_login=2, feature_adoption_pct=72, contract_type="monthly"), 0),
]


def get_model_info() -> dict:
    correct = 0
    brier_sum = 0.0
    tp = fn = 0
    for feats, actual in BENCHMARK_ACCOUNTS:
        pred = predict_churn(feats)
        predicted = 1 if pred.churn_probability >= 0.5 else 0
        correct += int(predicted == actual)
        brier_sum += (pred.churn_probability - actual) ** 2
        if actual == 1 and predicted == 1:
            tp += 1
        elif actual == 1 and predicted == 0:
            fn += 1
    n = len(BENCHMARK_ACCOUNTS)
    return {
        "model_version": MODEL_VERSION,
        "model_type": "interpretable_logistic_regression",
        "intercept": INTERCEPT,
        "feature_weights": FEATURE_WEIGHTS,
        "risk_band_thresholds": RISK_BAND_THRESHOLDS,
        "validation_metrics": {
            "benchmark_samples": n,
            "accuracy": round(correct / n, 3),
            "churn_recall": round(tp / max(tp + fn, 1), 3),
            "brier_score": round(brier_sum / n, 4),
        },
    }
