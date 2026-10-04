import json, joblib, numpy as np, pandas as pd, sys
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
sys.path.insert(0, str(Path(__file__).parent))
from common import *

ROOT = Path(__file__).resolve().parent.parent
tr, te = load(ROOT/"data/KDDTrain+.txt"), load(ROOT/"data/KDDTest+.txt")
le = LabelEncoder().fit(tr["category"])
ytr, yte = le.transform(tr["category"]), le.transform(te["category"])
pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
                         ("num", StandardScaler(), NUM)])
Xtr, Xte = pre.fit_transform(tr[COLS]), pre.transform(te[COLS])
names = [n.split("__", 1)[1] for n in pre.get_feature_names_out()]

def evaluate(name, model, X, y):
    p = model.predict(X)
    rep = classification_report(y, p, target_names=le.classes_, output_dict=True, zero_division=0)
    res = {"model": name, "accuracy": accuracy_score(y, p), "macro_f1": f1_score(y, p, average="macro"),
           "per_class": {c: {k: rep[c][k] for k in ("precision", "recall", "f1-score", "support")} for c in le.classes_},
           "confusion_matrix": confusion_matrix(y, p).tolist(), "classes": list(le.classes_)}
    print(f"\n== {name}: acc={res['accuracy']:.4f} macroF1={res['macro_f1']:.4f}")
    print(classification_report(y, p, target_names=le.classes_, zero_division=0))
    return res

rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, class_weight="balanced", random_state=42).fit(Xtr, ytr)
r_rf = evaluate("RandomForest", rf, Xte, yte)
w = compute_sample_weight("balanced", ytr)
xgb = XGBClassifier(n_estimators=300, max_depth=7, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                    objective="multi:softprob", tree_method="hist", n_jobs=-1, random_state=42).fit(Xtr, ytr, sample_weight=w)
r_xgb = evaluate("XGBoost", xgb, Xte, yte)

imp = pd.Series(xgb.feature_importances_, index=names)
orig = lambda n: next((c for c in CAT if n.startswith(c + "_")), n)
glob = imp.groupby(orig).sum().sort_values(ascending=False).head(15)

xgb.save_model(str(ROOT/"models/xgb_model.json"))  # portable native format (works on any OS)
joblib.dump({"pre": pre, "classes": list(le.classes_), "feature_names": names}, ROOT/"models/preproc.joblib")
json.dump({"random_forest": r_rf, "xgboost": r_xgb, "global_importance": glob.round(5).to_dict(),
           "train_rows": len(tr), "test_rows": len(te),
           "note": "Evaluated on official KDDTest+ (contains attack types unseen in training)."},
          open(ROOT/"models/metrics.json", "w"), indent=1)
te.sample(3000, random_state=1)[COLS + ["label", "category"]].to_csv(ROOT/"models/replay_sample.csv", index=False)
print("\nsaved.")
