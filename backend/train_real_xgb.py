import json
import glob
import os
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

DATA_GLOB = "/opt/nginx-viewer-api/*_run_*.json"
OUT_DIR = "/opt/nginx-viewer-api/model"

# --- Filters to keep only meaningful HLS sessions ---
MIN_DURATION_S = 10
REQUIRE_M3U8 = True
REQUIRE_TS = True


def load_sessions(path):
    """Load sessions array from a capture json file. Skip empty/bad JSON files."""
    try:
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            print("Skipping empty file:", path)
            return []
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("sessions", [])
    except Exception as e:
        print("Skipping bad JSON:", path, "error:", repr(e))
        return []


def file_label_from_name(fp: str):
    """bots_run_*.json => 1 (bot), real_run_*.json => 0 (real)"""
    base = os.path.basename(fp)
    if base.startswith("bots_run_"):
        return 1
    if base.startswith("real_run_"):
        return 0
    return None


rows = []
files = sorted(glob.glob(DATA_GLOB))
if not files:
    raise SystemExit(f"No files found: {DATA_GLOB}")

used_files = []
skipped_files = []
per_file_kept = {}

for fp in files:
    file_label = file_label_from_name(fp)
    if file_label is None:
        skipped_files.append(fp)
        continue

    sessions = load_sessions(fp)
    if not sessions:
        skipped_files.append(fp)
        continue

    kept = 0
    for s in sessions:
        feats = s.get("features") or {}
        if not feats:
            continue

        # keep only real streaming-like sessions (remove healthchecks/random)
        if REQUIRE_M3U8 and float(feats.get("m3u8_count", 0) or 0) <= 0:
            continue
        if REQUIRE_TS and float(feats.get("ts_count", 0) or 0) <= 0:
            continue
        if float(feats.get("duration_s", 0) or 0) < MIN_DURATION_S:
            continue

        row = dict(feats)
        row["label"] = int(file_label)
        rows.append(row)
        kept += 1

    per_file_kept[fp] = kept
    if kept > 0:
        used_files.append(fp)
    else:
        skipped_files.append(fp)

if not rows:
    raise SystemExit("No usable rows after filtering. Increase window_seconds or reduce filters.")

df = pd.DataFrame(rows).fillna(0.0)

# Convert everything except label to numeric
for c in list(df.columns):
    if c == "label":
        continue
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

# Deterministic feature order (IMPORTANT)
feature_cols = sorted([c for c in df.columns if c != "label"])

y = df["label"].astype(int)
X = df[feature_cols]

if y.nunique() < 2:
    raise SystemExit(
        "Dataset still has only one class after filtering.\n"
        "Make sure you have BOTH bots_run_*.json and real_run_*.json with real streaming sessions."
    )

# Balance helper for bot/real imbalance
pos = int(y.sum())
neg = int((1 - y).sum())
scale_pos_weight = (neg / pos) if pos else 1.0

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

model = XGBClassifier(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    objective="binary:logistic",
    eval_metric="logloss",
    n_jobs=2,
    scale_pos_weight=scale_pos_weight,
    use_label_encoder=False,  # remove deprecation warning
)

model.fit(X_train, y_train)

proba = model.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)

print("\nUsed files (kept rows):")
for f in used_files:
    print(f"  - {f}  (kept={per_file_kept.get(f, 0)})")

if skipped_files:
    print("\nSkipped files (no usable rows or not matching pattern):")
    for f in skipped_files:
        print(f"  - {f}  (kept={per_file_kept.get(f, 0)})")

print("\nRows:", len(df), "Bots:", int(y.sum()), "Real:", int((1 - y).sum()))
print("scale_pos_weight:", scale_pos_weight)

print("\nConfusion matrix [ [TN FP], [FN TP] ]:")
print(confusion_matrix(y_test, pred))

print("\nClassification report:")
print(classification_report(y_test, pred))

print("AUC:", roc_auc_score(y_test, proba))

# Feature importance (top 15)
try:
    importances = model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:15]
    print("\nTop features:")
    for i in top_idx:
        print(f"  {feature_cols[i]}: {importances[i]:.4f}")
except Exception:
    pass

os.makedirs(OUT_DIR, exist_ok=True)
model_path = os.path.join(OUT_DIR, "xgb_bot_model.json")
schema_path = os.path.join(OUT_DIR, "feature_schema.json")

model.save_model(model_path)
with open(schema_path, "w", encoding="utf-8") as f:
    json.dump(feature_cols, f, indent=2)

print("\nSaved model:", model_path)
print("Saved schema:", schema_path)
