# SaaS Churn Predictor

[![CI](https://github.com/nsriharshavardhan/saas-churn-predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/nsriharshavardhan/saas-churn-predictor/actions/workflows/ci.yml)

**Live demo:** https://saas-churn-predictor.onrender.com

A FastAPI scoring service that estimates 90-day churn risk for B2B SaaS accounts, lists the top
contributing risk factors, and helps customer-success teams prioritise outreach.

## Features

- **Interpretable logistic scorer.** Transparent log-odds weights over engagement, billing, support, and tenure features — no black-box pickle required.
- **Interactive web demo** at `/` with pre-filled high-risk, watchlist, and healthy account presets.
- **Cohort batch scoring.** `POST /predict/batch` scores up to 200 accounts at once and reports risk-band counts, average churn probability, and at-risk MRR (AUD).
- **Model transparency.** `GET /model-info` exposes the model version, feature weights, risk-band thresholds, and validation metrics on a built-in benchmark set.
- **Delivery:** pytest suite, GitHub Actions CI (tests + Docker build), Dockerfile, and one-click `render.yaml`.

## API

| Method | Path | Description |
|--------|------|-------------|
| GET  | `/` | Interactive churn scorer UI |
| POST | `/predict` | Score a single account |
| POST | `/predict/batch` | Score up to 200 accounts with cohort summary |
| GET  | `/model-info` | Model version, weights, thresholds, and benchmark metrics |
| GET  | `/health` | Health check |

```bash
curl -X POST https://saas-churn-predictor.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"customer_id":"acct-1042","tenure_months":4,"monthly_logins":3,
       "support_tickets_30d":4,"failed_payments_90d":2,
       "days_since_last_login":24,"feature_adoption_pct":25,
       "contract_type":"monthly","mrr_aud":299}'
```

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
uvicorn app.main:app --reload   # open http://127.0.0.1:8000
```

## Deploy (Render free tier)

1. Sign in to [Render](https://render.com) with GitHub, then choose **New → Blueprint** and pick this repo.
2. Render reads `render.yaml` and deploys. In the UI at `/`, check that the **High-risk account** preset returns `HIGH`.
