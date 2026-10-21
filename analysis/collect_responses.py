"""Download all session files from the responses repo and flatten them to one row per trial.

Usage:
    pip install huggingface_hub pandas pyarrow
    python analysis/collect_responses.py Enzo-py/partial-prompting-responses [--out responses.parquet]

Adds a column `failed_attention_check` (rating > 2 on the control stimulus) so excluded
participants can be filtered out before analysis.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download

ap = argparse.ArgumentParser()
ap.add_argument("repo_id")
ap.add_argument("--out", default="responses.parquet")
args = ap.parse_args()

local = Path(snapshot_download(args.repo_id, repo_type="dataset", allow_patterns="responses/**"))
rows = []
for f in sorted(local.glob("responses/list*/*.json")):
    s = json.loads(f.read_text())
    p = s.get("participant", {})
    for t in s.get("trials", []):
        rows.append({
            "session_id": s["session_id"], "list_id": s["list_id"], "study_version": s.get("study_version"),
            "prolific_pid": p.get("prolific_pid"), "english": p.get("english"), "t2i_use": p.get("t2i"),
            "total_time_ms": s.get("total_time_ms"),
            **{k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in t.items()},
            "n_marks": len(t.get("marks", [])),
        })

df = pd.DataFrame(rows)
if df.empty:
    raise SystemExit("No responses yet.")
ctrl = df[df["role"] == "control"].groupby("session_id")["rating"].max()
df["failed_attention_check"] = df["session_id"].map(ctrl > 2).fillna(False)
df.to_parquet(args.out, index=False)
print(f"{df.session_id.nunique()} sessions, {len(df)} trials -> {args.out}")
print(df.groupby(["case_id", "role"])["rating"].agg(["count", "mean", "std"]).round(2))
