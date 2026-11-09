"""Export the human study results in the dataset's own format (see the dataset README, Pairs and Groups).

Usage:
    pip install huggingface_hub
    HF_TOKEN=... python analysis/export_human_results.py Enzo-py/partial-prompting-responses [--out human_results]
    python analysis/export_human_results.py path/to/responses      # a local copy (list*/<session>.json)

Reads every session the Space saved to the responses repo (it pushes them there a few minutes after
each submission), drops the excluded sessions with the same rules as the Space (space/app.py,
`quality_flags`), and writes to --out:

  pairs/pairs.jsonl      the dataset's pair table with `family_human` filled
  groups/groups.jsonl    the dataset's group table with `ranking_human` filled
  human/ratings.jsonl    per stimulus: number of ratings, mean, standard deviation, count of each score
  human/pairs.jsonl      per pair: the comparison behind `family_human` (difference, 95% interval, n)
  human/groups.jsonl     per group: number of rankings, mean rank and first places of each image
  human/sessions.jsonl   per session: list, failed checks, excluded or not

The first two keep every line and field of the dataset files, so they can replace them as they are.
Fields the README leaves for after the study stay null when the data cannot decide them yet.

family_human compares the ratings of the test member of a pair (the control when there is no test)
with those of its reference: d = mean(test) - mean(reference), with a bootstrap 95% interval.
  sensitivity  the interval is entirely below -MARGIN: the content clearly lowers the rating
  invariance   the interval is entirely inside [-MARGIN, MARGIN]: the rating does not move
  open         anything else (a small or uncertain change)
  null         fewer than --min-ratings ratings for a member
A pair with several test members gets a family only when they all agree, else `open`.

ranking_human orders a group's images (image_id) by mean rank over the included sessions, ties broken
by the number of first places; null below --min-rankings rankings, or for a group the Space did not
show (a prompt group, which has a single image).
"""
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev

from huggingface_hub import hf_hub_download, snapshot_download

ap = argparse.ArgumentParser()
ap.add_argument("responses", help="responses repo id, or a local folder of session files")
ap.add_argument("--dataset", default="Enzo-py/partial-prompting-ds")
ap.add_argument("--revision", default=None, help="dataset revision (default: latest)")
ap.add_argument("--out", default="human_results")
ap.add_argument("--margin", type=float, default=0.5, help="rating difference that counts as a change (default 0.5)")
ap.add_argument("--min-ratings", type=int, default=10, help="ratings per pair member needed to decide (default 10)")
ap.add_argument("--min-rankings", type=int, default=5, help="rankings per group needed to order it (default 5)")
ap.add_argument("--include-excluded", action="store_true", help="keep the sessions that fail the quality checks")
args = ap.parse_args()

# Quality checks: the same rules and thresholds as space/app.py.
FAST_TRIAL_MS, FAST_SHARE, SAME_ANSWER_SHARE, EXCLUDE_AT = 1000, 0.25, 0.99, 2
CHECK_FAIL_AT, CHECK_FAIL_AT_DEFAULT = {"absent": 3}, 4


def is_attention_check(s):
    if "attention_check" in s:
        return bool(s["attention_check"])
    return s.get("role") == "control" and bool(
        s.get("control_type") or s.get("subcategory") == "Attention check" or s.get("case_id") == "E1")


def quality_flags(session, by_id):
    trials = [t for t in session.get("trials", []) if t.get("rating")]
    flags = []
    for t in trials:
        s = t if "control_type" in t else by_id.get(t.get("sample_id"), {})  # labels saved with the trial
        if is_attention_check(s) and t["rating"] >= CHECK_FAIL_AT.get(s.get("control_type") or "absent", CHECK_FAIL_AT_DEFAULT):
            flags.append("attention_check")
    if trials and sum((t.get("rt") or 0) < FAST_TRIAL_MS for t in trials) / len(trials) >= FAST_SHARE:
        flags.append("too_fast")
    if trials and Counter(t["rating"] for t in trials).most_common(1)[0][1] / len(trials) >= SAME_ANSWER_SHARE:
        flags.append("same_answers")
    return flags


def read_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def bootstrap_diff(a, b, n=4000, seed=0):
    """95% bootstrap interval of mean(a) - mean(b)."""
    rng = random.Random(seed)
    d = sorted(mean(rng.choices(a, k=len(a))) - mean(rng.choices(b, k=len(b))) for _ in range(n))
    return d[int(0.025 * n)], d[int(0.975 * n) - 1]


get = lambda f: hf_hub_download(args.dataset, f, repo_type="dataset", revision=args.revision)
meta, pairs, groups = read_jsonl(get("data/metadata.jsonl")), read_jsonl(get("pairs/pairs.jsonl")), read_jsonl(get("groups/groups.jsonl"))
by_id = {s["sample_id"]: s for s in meta}

if Path(args.responses).is_dir():
    local = Path(args.responses)
else:
    local = Path(snapshot_download(args.responses, repo_type="dataset", allow_patterns="responses/**")) / "responses"
sessions = [json.loads(f.read_text()) for f in sorted(local.glob("list*/*.json"))]
if not sessions:
    raise SystemExit(f"No sessions in {args.responses}.")
out = Path(args.out)
session_rows, included = [], []
for s in sessions:
    flags = quality_flags(s, by_id)
    excluded = len(flags) >= EXCLUDE_AT
    session_rows.append({"session_id": s["session_id"], "list_id": s.get("list_id"), "study_version": s.get("study_version"),
                         "received_at": s.get("received_at"), "flags": flags, "excluded": excluded})
    if args.include_excluded or not excluded:
        included.append(s)
write_jsonl(out / "human/sessions.jsonl", session_rows)

# Ratings per stimulus.
ratings = defaultdict(list)
for s in included:
    for t in s.get("trials", []):
        if t.get("rating") and t.get("trial_kind", "stimulus") == "stimulus":
            ratings[t["sample_id"]].append(t["rating"])
write_jsonl(out / "human/ratings.jsonl", [
    {"sample_id": i, "n_ratings": len(r), "mean": round(mean(r), 3), "sd": round(stdev(r), 3) if len(r) > 1 else None,
     "counts": {str(k): r.count(k) for k in range(1, 6)}, "hypothetical_score": by_id.get(i, {}).get("hypothetical_score")}
    for i, r in sorted(ratings.items(), key=lambda kv: int(kv[0][1:]) if kv[0][1:].isdigit() else 0)])

# family_human per pair.
pair_rows = []
for p in pairs:
    roles = defaultdict(list)
    for m in p["members"]:
        roles[m["role"]].append(m["sample_id"])
    refs = roles.get("reference", [])
    tests = roles.get("test") or roles.get("control") or []
    ref_r = [x for i in refs for x in ratings.get(i, [])]
    comparisons, families = [], []
    for i in tests:
        r = ratings.get(i, [])
        row = {"sample_id": i, "n": len(r), "n_reference": len(ref_r), "diff": None, "ci95": None, "family": None}
        if len(r) >= args.min_ratings and len(ref_r) >= args.min_ratings:
            lo, hi = bootstrap_diff(r, ref_r)
            row.update(diff=round(mean(r) - mean(ref_r), 3), ci95=[round(lo, 3), round(hi, 3)],
                       family="sensitivity" if hi < -args.margin else
                              "invariance" if -args.margin <= lo and hi <= args.margin else "open")
        comparisons.append(row)
        families.append(row["family"])
    if not refs or not families or None in families:
        family = None
    else:
        family = families[0] if len(set(families)) == 1 else "open"
    p["family_human"] = family
    pair_rows.append({"pair_id": p["pair_id"], "family_expected": p.get("family_expected"), "family_human": family,
                      "reference": refs, "comparisons": comparisons})
write_jsonl(out / "pairs/pairs.jsonl", pairs)
write_jsonl(out / "human/pairs.jsonl", pair_rows)

# ranking_human per group.
ranks, firsts, n_rankings = defaultdict(lambda: defaultdict(list)), defaultdict(Counter), Counter()
for s in included:
    for r in s.get("rankings") or []:
        if "order_sample_ids" not in r:  # rankings from before the dataset groups
            continue
        n_rankings[r["set_id"]] += 1
        for k, i in enumerate(r["order"], 1):
            ranks[r["set_id"]][i].append(k)
        firsts[r["set_id"]][r["order"][0]] += 1
group_rows = []
for g in groups:
    gid, mr = g["group_id"], ranks.get(g["group_id"], {})
    images = {i: {"image_id": i, "mean_rank": round(mean(v), 3), "first": firsts[gid][i], "n": len(v)} for i, v in mr.items()}
    order = sorted(images, key=lambda i: (images[i]["mean_rank"], -images[i]["first"], i))
    g["ranking_human"] = order if n_rankings[gid] >= args.min_rankings else None
    group_rows.append({"group_id": gid, "axis": g.get("axis"), "n_rankings": n_rankings[gid], "ranking_human": g["ranking_human"],
                       "ranking_hypothetical": g.get("ranking_hypothetical"), "images": [images[i] for i in order]})
write_jsonl(out / "groups/groups.jsonl", groups)
write_jsonl(out / "human/groups.jsonl", group_rows)

n_ex = sum(r["excluded"] for r in session_rows)
print(f"{len(sessions)} sessions, {n_ex} excluded{' (kept)' if args.include_excluded else ''}; "
      f"{sum(map(len, ratings.values()))} ratings of {len(ratings)} stimuli")
print("family_human:", dict(Counter(p["family_human"] for p in pairs)))
print("ranking_human:", sum(g["ranking_human"] is not None for g in groups), "of", len(groups), "groups ordered")
print(f"-> {out}/")
