from fastapi.testclient import TestClient

from app.main import app
from app.model import CustomerFeatures, predict_churn

client = TestClient(app)

HIGH_RISK = CustomerFeatures(
    customer_id="acct-high",
    tenure_months=2,
    monthly_logins=1,
    support_tickets_30d=5,
    failed_payments_90d=2,
    days_since_last_login=28,
    feature_adoption_pct=15.0,
    contract_type="monthly",
    mrr_aud=499.0,
)

MEDIUM_RISK = CustomerFeatures(
    customer_id="acct-med",
    tenure_months=12,
    monthly_logins=11,
    support_tickets_30d=1,
    failed_payments_90d=0,
    days_since_last_login=10,
    feature_adoption_pct=50.0,
    contract_type="monthly",
    mrr_aud=199.0,
)

LOW_RISK = CustomerFeatures(
    customer_id="acct-low",
    tenure_months=24,
    monthly_logins=40,
    support_tickets_30d=0,
    failed_payments_90d=0,
    days_since_last_login=1,
    feature_adoption_pct=85.0,
    contract_type="annual",
    mrr_aud=1200.0,
)


def test_high_risk_account_scores_high():
    pred = predict_churn(HIGH_RISK)
    assert pred.risk_band == "HIGH"
    assert pred.churn_probability >= 0.65
    assert any("failed payment" in f for f in pred.top_risk_factors)


def test_healthy_account_scores_low():
    pred = predict_churn(LOW_RISK)
    assert pred.risk_band == "LOW"
    assert pred.churn_probability < 0.35


def test_annual_contract_reduces_churn_probability():
    monthly = predict_churn(MEDIUM_RISK)
    annual = predict_churn(MEDIUM_RISK.model_copy(update={"contract_type": "annual"}))
    assert annual.churn_probability < monthly.churn_probability


def test_predict_endpoint_and_demo_page():
    r = client.post("/predict", json=HIGH_RISK.model_dump())
    assert r.status_code == 200
    assert r.json()["risk_band"] == "HIGH"
    assert client.get("/").status_code == 200
    assert client.get("/health").json()["status"] == "ok"


def test_invalid_features_rejected():
    bad = {**HIGH_RISK.model_dump(), "feature_adoption_pct": 150.0}
    assert client.post("/predict", json=bad).status_code == 422


def test_batch_prediction_summary():
    payload = {"customers": [HIGH_RISK.model_dump(), MEDIUM_RISK.model_dump(), LOW_RISK.model_dump()]}
    r = client.post("/predict/batch", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert len(data["predictions"]) == 3
    summary = data["summary"]
    assert summary["total_customers"] == 3
    assert summary["high_risk_count"] == 1
    assert summary["low_risk_count"] == 1
    assert summary["at_risk_mrr_aud"] == 499.0
    assert 0.0 < summary["avg_churn_probability"] < 1.0


def test_batch_rejects_empty_list():
    assert client.post("/predict/batch", json={"customers": []}).status_code == 422
