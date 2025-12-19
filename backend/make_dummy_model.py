import json
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

schema = [
  "req_count","duration_s","rps",
  "m3u8_count","ts_count","m3u8_ts_ratio",
  "bytes_sum","bytes_avg",
  "rt_avg","rt_p95",
  "gap_mean","gap_p50","gap_p95",
  "err_rate","status_4xx","status_5xx"
]

# create random data just to produce a valid model file
X = pd.DataFrame(np.random.rand(800, len(schema)), columns=schema)
y = ((X["rps"] > 0.35) | (X["err_rate"] > 0.2)).astype(int)

model = XGBClassifier(
    n_estimators=120,
    max_depth=4,
    learning_rate=0.08,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric="logloss",
)
model.fit(X, y)

out_dir = "/opt/nginx-viewer-api/model"
model.save_model(f"{out_dir}/xgb_bot_model.json")

with open(f"{out_dir}/feature_schema.json", "w") as f:
    json.dump(schema, f, indent=2)

print("Saved model + schema into", out_dir)
