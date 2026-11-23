"""FastAPI service: SaaS customer churn risk scorer."""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from app.model import (
    MODEL_VERSION,
    ChurnPrediction,
    CustomerFeatures,
    get_model_info,
    predict_churn,
)

app = FastAPI(
    title="SaaS Churn Predictor",
    description="Scores B2B SaaS accounts for churn risk with interpretable factors and retention playbooks.",
    version=MODEL_VERSION,
)


class BatchPredictIn(BaseModel):
    customers: list[CustomerFeatures] = Field(..., min_length=1, max_length=200)


class CohortSummary(BaseModel):
    total_customers: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    avg_churn_probability: float
    at_risk_mrr_aud: float


class BatchPredictOut(BaseModel):
    predictions: list[ChurnPrediction]
    summary: CohortSummary


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.post("/predict", response_model=ChurnPrediction)
def predict(features: CustomerFeatures) -> ChurnPrediction:
    return predict_churn(features)


@app.post("/predict/batch", response_model=BatchPredictOut)
def predict_batch(body: BatchPredictIn) -> BatchPredictOut:
    preds = [predict_churn(c) for c in body.customers]
    high = sum(1 for p in preds if p.risk_band == "HIGH")
    med = sum(1 for p in preds if p.risk_band == "MEDIUM")
    low = sum(1 for p in preds if p.risk_band == "LOW")
    avg_p = round(sum(p.churn_probability for p in preds) / len(preds), 3)
    at_risk_mrr = round(
        sum(c.mrr_aud for c, p in zip(body.customers, preds) if p.risk_band == "HIGH"),
        2,
    )
    return BatchPredictOut(
        predictions=preds,
        summary=CohortSummary(
            total_customers=len(preds),
            high_risk_count=high,
            medium_risk_count=med,
            low_risk_count=low,
            avg_churn_probability=avg_p,
            at_risk_mrr_aud=at_risk_mrr,
        ),
    )


@app.get("/model-info")
def model_info() -> dict:
    return get_model_info()


DEMO_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SaaS Churn Predictor</title>
<style>
 body{font-family:system-ui,sans-serif;max-width:760px;margin:40px auto;padding:0 16px;color:#1f2937}
 label{display:block;font-size:13px;color:#4b5563;margin-top:10px}
 input,select{width:100%;box-sizing:border-box;padding:8px;border:1px solid #d1d5db;border-radius:8px;font:inherit}
 .grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}
 button{margin-top:14px;padding:10px 16px;border:0;border-radius:8px;background:#2563eb;color:#fff;cursor:pointer}
 button.ex{background:#eef2ff;color:#3730a3;padding:6px 10px;font-size:13px;margin-right:6px}
 .card{border:1px solid #e5e7eb;border-radius:10px;padding:16px;margin-top:20px}
 .pill{display:inline-block;padding:3px 10px;border-radius:99px;font-weight:600;margin-right:8px}
 .HIGH{background:#fee2e2;color:#b91c1c}.MEDIUM{background:#fef3c7;color:#92400e}.LOW{background:#dcfce7;color:#166534}
 .meta{font-size:12px;color:#6b7280}
</style></head><body>
<h1>SaaS Churn Predictor</h1>
<p class="meta">Score a B2B SaaS account's 90-day churn risk and inspect the driving factors.</p>
<div>
 <button class="ex" data-ex="0">High-risk account</button>
 <button class="ex" data-ex="1">Watchlist account</button>
 <button class="ex" data-ex="2">Healthy enterprise</button>
</div>
<form id="f">
 <div class="grid">
  <div><label>Tenure (months)</label><input id="tenure" type="number" min="0" value="4" required></div>
  <div><label>Monthly logins</label><input id="logins" type="number" min="0" value="3" required></div>
  <div><label>Support tickets (30d)</label><input id="tickets" type="number" min="0" value="4" required></div>
  <div><label>Failed payments (90d)</label><input id="failed" type="number" min="0" value="2" required></div>
  <div><label>Days since last login</label><input id="days" type="number" min="0" value="24" required></div>
  <div><label>Feature adoption (%)</label><input id="adopt" type="number" min="0" max="100" value="25" required></div>
  <div><label>Contract type</label><select id="contract"><option value="monthly" selected>monthly</option><option value="annual">annual</option></select></div>
  <div><label>MRR (AUD)</label><input id="mrr" type="number" min="0" value="299" required></div>
 </div>
 <button>Score account</button>
</form>
<div id="out"></div>
<p class="meta"><a href="/docs">API docs</a> · <a href="/model-info">Model weights &amp; metrics</a></p>
<script>
const EX=[
 [4,3,4,2,24,25,"monthly",299],
 [12,11,1,0,10,50,"monthly",199],
 [24,45,0,0,1,88,"annual",1499]];
const $=id=>document.getElementById(id);
const esc=s=>String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
document.querySelectorAll('.ex').forEach(b=>b.onclick=()=>{
  const e=EX[b.dataset.ex];
  ['tenure','logins','tickets','failed','days','adopt','contract','mrr'].forEach((k,i)=>$(k).value=e[i]);
});
$('f').onsubmit=async e=>{
  e.preventDefault();
  const body={
    tenure_months:+$('tenure').value, monthly_logins:+$('logins').value,
    support_tickets_30d:+$('tickets').value, failed_payments_90d:+$('failed').value,
    days_since_last_login:+$('days').value, feature_adoption_pct:+$('adopt').value,
    contract_type:$('contract').value, mrr_aud:+$('mrr').value
  };
  const r=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const d=await r.json();
  if(!r.ok){$('out').innerHTML=`<div class="card">Validation error: <pre>${esc(JSON.stringify(d.detail,null,1))}</pre></div>`;return;}
  const factors=d.top_risk_factors.map(x=>`<li>${esc(x)}</li>`).join('');
  const action=d.recommended_action?`<p><b>Recommended action:</b> ${esc(d.recommended_action)}</p>`:'';
  $('out').innerHTML=`<div class="card"><span class="pill ${d.risk_band}">${d.risk_band}</span><b>${(d.churn_probability*100).toFixed(1)}%</b> churn probability
   <ul>${factors}</ul>${action}<div class="meta">model v${esc(d.model_version)}</div></div>`;
};
</script></body></html>"""


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def demo() -> str:
    return DEMO_HTML
