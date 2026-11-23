"""Interpretable logistic churn-risk scorer for B2B SaaS accounts.

Uses an explicit logistic model with auditable feature weights rather than an
opaque black-box pickle. Every prediction returns a calibrated probability,
a risk band (LOW / MEDIUM / HIGH), and the top contributing risk factors.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, Field

MODEL_VERSION = "1.0.0"

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
