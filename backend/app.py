import json, sys, joblib, pandas as pd, numpy as np, xgboost as xgb
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import create_model

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ml"))
from common import COLS, CAT, SEVERITY

bundle = joblib.load(ROOT / "models/preproc.joblib")
MODEL = xgb.XGBClassifier()
MODEL.load_model(str(ROOT / "models/xgb_model.json"))
PRE, CLASSES = bundle["pre"], bundle["classes"]
BOOSTER = MODEL.get_booster()
FEATS = bundle["feature_names"]
METRICS = json.load(open(ROOT / "models/metrics.json"))
REPLAY = pd.read_csv(ROOT / "models/replay_sample.csv")
state = {"cursor": 0}

Record = create_model("Record", **{c: (str if c in CAT else float, ...) for c in COLS})

def orig_feature(name):
    return next((c for c in CAT if name.startswith(c + "_")), name)
GROUP_IDX = {}
for i, n in enumerate(FEATS):
    GROUP_IDX.setdefault(orig_feature(n), []).append(i)

def classify(df, top_k=5):
    X = PRE.transform(df[COLS])
    proba = MODEL.predict_proba(X)
    contribs = BOOSTER.predict(xgb.DMatrix(X, feature_names=None), pred_contribs=True)  # (n, classes, feats+1)
    out = []
    for i in range(len(df)):
        k = int(proba[i].argmax())
        c = contribs[i, k, :-1]
        agg = sorted(((f, float(c[idx].sum())) for f, idx in GROUP_IDX.items()), key=lambda t: -abs(t[1]))[:top_k]
        cat = CLASSES[k]
        out.append({"category": cat, "confidence": round(float(proba[i, k]), 4), "severity": SEVERITY[cat],
                    "probabilities": {CLASSES[j]: round(float(proba[i, j]), 4) for j in range(len(CLASSES))},
                    "top_features": [{"feature": f, "value": df.iloc[i][f] if isinstance(df.iloc[i][f], str) else float(df.iloc[i][f]),
                                      "impact": round(v, 4)} for f, v in agg]})
    return out

app = FastAPI(title="NetGuard API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/api/health")
def health():
    return {"status": "ok", "classes": CLASSES}

@app.post("/api/predict")
def predict(rec: Record):
    return classify(pd.DataFrame([rec.model_dump()]))[0]

@app.post("/api/predict/batch")
def predict_batch(recs: list[Record]):
    if not 0 < len(recs) <= 500:
        raise HTTPException(422, "Send between 1 and 500 records")
    return classify(pd.DataFrame([r.model_dump() for r in recs]))

@app.get("/api/metrics")
def metrics():
    return METRICS

@app.get("/api/replay")
def replay(n: int = 5):
    """Simulated live traffic: replays held-out KDDTest+ rows, returns prediction + ground truth."""
    n = max(1, min(n, 20))
    idx = [(state["cursor"] + i) % len(REPLAY) for i in range(n)]
    state["cursor"] = (state["cursor"] + n) % len(REPLAY)
    rows = REPLAY.iloc[idx]
    preds = classify(rows)
    return [{"id": int(i), "true_label": r["label"], "true_category": r["category"],
             "protocol": r["protocol_type"], "service": r["service"], "src_bytes": int(r["src_bytes"]),
             "dst_bytes": int(r["dst_bytes"]), **p}
            for i, (_, r), p in zip(idx, rows.iterrows(), preds)]

app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="ui")
