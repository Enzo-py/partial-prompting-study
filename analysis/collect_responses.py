"""Download all session files from the responses repo and flatten them to one row per trial.

Usage:
    pip install huggingface_hub pandas pyarrow
    python analysis/collect_responses.py Enzo-py/partial-prompting-responses [--out responses.parquet]

Adds one column per quality check, with the rules of the Space (space/human_export.py, `quality_flags`):
  failed_attention_check  number of attention checks rated as matching (3+ for a missing
                          object, 4+ for a wrong attribute, count or relation)
  too_fast                25% or more of the images answered in under 1 s
  same_answers            99% or more of the ratings are identical
then `n_failed_checks` (each failed attention check counts once) and `excluded` from EXCLUDE_AT (2)
failed checks. `position` is the presentation order and `rt` the time spent on the image (ms, from
display to "Next").

Rankings go to a second file (--rankings-out), one row per ranked member: `item` is what was ranked
(the image_id in an image group, the prompt in a prompt group, a sample_id in rankings from before the
dataset groups), `sample_id` the matching rated stimulus when there is one; `rank` 1 is the best,
`shown_position` where the member was on screen, and the session's `excluded` flag.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "space"))
import human_export  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("repo_id")
ap.add_argument("--out", default="responses.parquet")
ap.add_argument("--rankings-out", default="rankings.parquet")
args = ap.parse_args()

local = Path(snapshot_download(args.repo_id, repo_type="dataset", allow_patterns="responses/**"))
rows, ranking_rows = [], []
for f in sorted(local.glob("responses/list*/*.json")):
    s = json.loads(f.read_text())
    # The trials carry their labels, so the checks need no dataset.
    flags = Counter(human_export.quality_flags(s, {}))
    quality = {"failed_attention_check": flags["attention_check"], "too_fast": bool(flags["too_fast"]),
               "same_answers": bool(flags["same_answers"]), "n_failed_checks": sum(flags.values()),
               "excluded": sum(flags.values()) >= human_export.EXCLUDE_AT}
    for r in s.get("rankings") or []:
        # Rankings from dataset groups list image_ids or prompts (`order_sample_ids` alongside); older ones list sample_ids.
        grouped = "order_sample_ids" in r
        for rank, i in enumerate(r.get("order") or [], 1):
            ranking_rows.append({"session_id": s["session_id"], "list_id": s["list_id"], "set_id": r.get("set_id"),
                                 "axis": r.get("axis", "image"), "prompt": r.get("prompt"), "visual_style": r.get("visual_style"),
                                 "item": i, "sample_id": r["order_sample_ids"][rank - 1] if grouped else i,
                                 "rank": rank, "n_items": len(r["order"]), "shown_position": (r.get("shown") or []).index(i) + 1
                                 if i in (r.get("shown") or []) else None,
                                 "ranking_position": r.get("position"), "rt": r.get("rt"), "resets": r.get("resets"),
                                 "excluded": quality["excluded"]})
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
            **quality,
        })

df = pd.DataFrame(rows)
if df.empty:
    raise SystemExit("No responses yet.")
df.to_parquet(args.out, index=False)
print(f"{df.session_id.nunique()} sessions ({df[df.excluded].session_id.nunique()} excluded), {len(df)} trials -> {args.out}")
by = [c for c in ("category", "level") if c in df]
if by:
    print(df[~df.excluded].groupby(by)["rating"].agg(["count", "mean", "std"]).round(2))

if ranking_rows:
    rk = pd.DataFrame(ranking_rows)
    rk.to_parquet(args.rankings_out, index=False)
    print(f"{rk.groupby(['session_id', 'set_id']).ngroups} rankings -> {args.rankings_out}")
    print(rk[~rk.excluded].groupby(["set_id", "item"])["rank"].agg(["count", "mean"]).round(2))
