"""Download all session files from the responses repo and flatten them to one row per trial.

Usage:
    pip install huggingface_hub pandas pyarrow
    python analysis/collect_responses.py Enzo-py/partial-prompting-responses [--out responses.parquet]

Adds one column per quality check, the same rules as the Space (space/app.py, `quality_flags`):
  failed_attention_check  number of attention checks rated as matching (3+ for an absent
                          object, 4+ for a wrong attribute, count, relation or binding)
  too_fast                25% or more of the images answered in under 1 s
  same_answers            99% or more of the ratings are identical
then `n_failed_checks` (each failed attention check counts once) and `excluded` from 3 failed checks. `position` is the presentation order and `rt` the time
spent on the image (ms, from display to "Next").

Rankings go to a second file (--rankings-out), one row per ranked image (`image_id` from the dataset
group, `sample_id` when the image is also a rated stimulus): `rank` 1 is the favourite,
`shown_position` where the image was on screen, and the session's `excluded` flag.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download

ap = argparse.ArgumentParser()
ap.add_argument("repo_id")
ap.add_argument("--out", default="responses.parquet")
ap.add_argument("--rankings-out", default="rankings.parquet")
args = ap.parse_args()

local = Path(snapshot_download(args.repo_id, repo_type="dataset", allow_patterns="responses/**"))
rows, ranking_rows = [], []
for f in sorted(local.glob("responses/list*/*.json")):
    s = json.loads(f.read_text())
    for r in s.get("rankings") or []:
        # Rankings from dataset groups list image_ids (`order_sample_ids` alongside); older ones list sample_ids.
        grouped = "order_sample_ids" in r
        for rank, i in enumerate(r.get("order") or [], 1):
            ranking_rows.append({"session_id": s["session_id"], "list_id": s["list_id"], "set_id": r.get("set_id"),
                                 "prompt": r.get("prompt"), "visual_style": r.get("visual_style"),
                                 "image_id": i if grouped else None,
                                 "sample_id": r["order_sample_ids"][rank - 1] if grouped else i,
                                 "rank": rank, "n_images": len(r["order"]), "shown_position": (r.get("shown") or []).index(i) + 1
                                 if i in (r.get("shown") or []) else None,
                                 "ranking_position": r.get("position"), "rt": r.get("rt"), "resets": r.get("resets")})
    p = s.get("participant", {})
    for t in s.get("trials", []):
        rows.append({
            "session_id": s["session_id"], "list_id": s["list_id"], "study_version": s.get("study_version"),
            "source": p.get("source"), "pid_hash": p.get("pid_hash"), "english": p.get("english"), "t2i_use": p.get("t2i"),
            "total_time_ms": s.get("total_time_ms"),
            **{k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in t.items()},
            "n_marks": len(t.get("marks", [])),
            # Question 2 on segmented images: the selected parts (`segments`, JSON) and their categories.
            "segment_categories": ";".join(seg["category"] for seg in t.get("segments") or []) or None,
        })

df = pd.DataFrame(rows)
if df.empty:
    raise SystemExit("No responses yet.")
by_session = df.groupby("session_id")
# Attention checks: controls the image plainly contradicts (control_type set; case E1 in older
# datasets). Other "control" stimuli are a pair's control condition, analysed like the rest.
col = lambda c: df[c] if c in df else pd.Series(None, index=df.index)
is_check = (df["role"] == "control") & (col("control_type").notna() | (col("subcategory") == "Attention check") | (col("case_id") == "E1"))
# Failed = rated as matching: from 3 for an absent object, from 4 for a wrong colour, count or relation.
fail_at = col("control_type").fillna("absent").map(lambda k: 3 if k == "absent" else 4)
ctrl = (df["rating"] >= fail_at)[is_check].groupby(df["session_id"]).sum()
df["failed_attention_check"] = df["session_id"].map(ctrl).fillna(0).astype(int)
df["too_fast"] = df["session_id"].map(by_session["rt"].agg(lambda r: (r < 1000).mean() >= 0.25))
df["same_answers"] = df["session_id"].map(by_session["rating"].agg(lambda r: r.value_counts(normalize=True).iloc[0] >= 0.99))
df["n_failed_checks"] = df["failed_attention_check"] + df["too_fast"].astype(int) + df["same_answers"].astype(int)
df["excluded"] = df["n_failed_checks"] >= 3
df.to_parquet(args.out, index=False)
print(f"{df.session_id.nunique()} sessions ({df[df.excluded].session_id.nunique()} excluded), {len(df)} trials -> {args.out}")
print(df.groupby(["case_id", "role"])["rating"].agg(["count", "mean", "std"]).round(2))

if ranking_rows:
    rk = pd.DataFrame(ranking_rows)
    rk["excluded"] = rk["session_id"].map(df.groupby("session_id")["excluded"].first()).fillna(False)
    rk.to_parquet(args.rankings_out, index=False)
    print(f"{rk.groupby(['session_id', 'set_id']).ngroups} rankings -> {args.rankings_out}")
    print(rk[~rk.excluded].groupby(["set_id", rk.image_id.fillna(rk.sample_id)])["rank"].agg(["count", "mean"]).round(2))
