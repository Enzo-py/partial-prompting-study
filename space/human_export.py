"""Quality checks and the export of the human results in the dataset's own format.

Shared by the Space (app.py: exclusions, admin page, export button) and analysis/export_human_results.py,
so both always exclude the same sessions and produce the same files.

`export()` returns, by path:
  pairs/pairs.jsonl      the dataset's pair table with `family_human` filled
  groups/groups.jsonl    the dataset's group table with `ranking_human` filled
  human/ratings.jsonl    per sample: number of ratings, mean, standard deviation, count of each score
  human/pairs.jsonl      per pair: the comparison behind `family_human` (difference, 95% interval, n)
  human/groups.jsonl     per group: number of rankings, mean rank and first places of each image
  human/sessions.jsonl   per session: list, failed checks, excluded or not

The first two keep every line and field of the dataset files, so they can replace them as they are.
Fields the dataset README leaves for after the study stay null when the data cannot decide them yet.

family_human compares the ratings of the test member of a pair (the control when there is no test)
with those of its reference: d = mean(test) - mean(reference), with a bootstrap 95% interval.
  sensitivity  the interval is entirely below -margin: the content clearly lowers the rating
  invariance   the interval is entirely inside [-margin, margin]: the rating does not move
  open         anything else (a small or uncertain change)
  null         fewer than min_ratings ratings for a member
A pair with several test members gets a family only when they all agree, else `open`.

ranking_human orders a group's images (image_id) by mean rank over the included sessions, ties broken
by the number of first places; null below min_rankings rankings, or for a group never shown (a prompt
group, which has a single image).
"""
import json
import random
from collections import Counter, defaultdict
from statistics import mean, stdev

# Quality checks. Each failed attention check counts once; a session is excluded once it fails
# EXCLUDE_AT checks, fewer are only a warning.
FAST_TRIAL_MS = 1000       # an image answered faster than this...
FAST_SHARE = 0.25          # ...for this share of the images or more
SAME_ANSWER_SHARE = 0.99   # share of identical ratings across a session
EXCLUDE_AT = 2             # failed checks that exclude a session
# A check fails when the rating says the image matches. Only an absent object makes "Partly" (3)
# wrong; a wrong colour, count or relation can honestly be rated 3, so those fail from "Mostly" (4).
CHECK_FAIL_AT = {"absent": 3}
CHECK_FAIL_AT_DEFAULT = 4


def is_attention_check(s):
    """The dataset's `attention_check` flag when it has one. Older datasets (and sessions saved with
    them): a control whose text the image plainly contradicts (`control_type`: absent, relation, ...),
    or case E1; other controls are the control condition of a pair, counterbalanced like the rest."""
    if "attention_check" in s:
        return bool(s["attention_check"])
    return s.get("role") == "control" and bool(
        s.get("control_type") or s.get("subcategory") == "Attention check" or s.get("case_id") == "E1")


def failed_check(t, by_id):
    """Whether a trial is an attention check rated as matching. Uses the labels saved with the trial,
    so a session is judged on the samples it was shown even if the dataset changed since."""
    s = t if "control_type" in t else by_id.get(t.get("sample_id"), {})  # older sessions: current labels
    if not is_attention_check(s):
        return False
    kind = s.get("control_type") or "absent"  # the original check (E1, "a banana") is an absent object
    return t["rating"] >= CHECK_FAIL_AT.get(kind, CHECK_FAIL_AT_DEFAULT)


def quality_flags(session, by_id):
    """The checks a session fails, one entry per failed attention-check sample."""
    trials = [t for t in session.get("trials", []) if t.get("rating")]
    flags = ["attention_check" for t in trials if failed_check(t, by_id)]
    if trials and sum((t.get("rt") or 0) < FAST_TRIAL_MS for t in trials) / len(trials) >= FAST_SHARE:
        flags.append("too_fast")
    if trials and Counter(t["rating"] for t in trials).most_common(1)[0][1] / len(trials) >= SAME_ANSWER_SHARE:
        flags.append("same_answers")
    return flags


def bootstrap_diff(a, b, n=4000, seed=0):
    """95% bootstrap interval of mean(a) - mean(b)."""
    rng = random.Random(seed)
    d = sorted(mean(rng.choices(a, k=len(a))) - mean(rng.choices(b, k=len(b))) for _ in range(n))
    return d[int(0.025 * n)], d[int(0.975 * n) - 1]


def export(sessions, meta, pairs, groups, margin=0.5, min_ratings=10, min_rankings=5, include_excluded=False):
    """{path: rows} for the files listed in the module docstring, and a one-line summary."""
    by_id = {s["sample_id"]: s for s in meta}
    # A fixed order, so that the bootstrap intervals do not depend on how the sessions were read.
    sessions = sorted(sessions, key=lambda s: (str(s.get("received_at") or ""), str(s.get("session_id") or "")))
    pairs, groups = [dict(p) for p in pairs], [dict(g) for g in groups]
    session_rows, included = [], []
    for s in sessions:
        flags = quality_flags(s, by_id)
        excluded = len(flags) >= EXCLUDE_AT
        session_rows.append({"session_id": s.get("session_id"), "list_id": s.get("list_id"), "study_version": s.get("study_version"),
                             "received_at": s.get("received_at"), "flags": flags, "excluded": excluded})
        if include_excluded or not excluded:
            included.append(s)

    # Ratings per sample.
    ratings = defaultdict(list)
    for s in included:
        for t in s.get("trials", []):
            if t.get("rating") and t.get("trial_kind", "stimulus") == "stimulus":
                ratings[t["sample_id"]].append(t["rating"])
    order = lambda i: (int(i[1:]) if i[1:].isdigit() else 0, i)
    rating_rows = [
        {"sample_id": i, "n_ratings": len(r), "mean": round(mean(r), 3), "sd": round(stdev(r), 3) if len(r) > 1 else None,
         "counts": {str(k): r.count(k) for k in range(1, 6)}, "hypothetical_score": by_id.get(i, {}).get("hypothetical_score")}
        for i, r in sorted(ratings.items(), key=lambda kv: order(kv[0]))]

    # family_human per pair.
    pair_rows = []
    for p in pairs:
        roles = defaultdict(list)
        for m in p["members"]:
            roles[m["role"]].append(m["sample_id"])
        refs = roles.get("reference", [])
        tests = roles.get("test") or roles.get("control") or []
        ref_r = [x for i in refs for x in ratings.get(i, [])]
        comparisons = []
        for i in tests:
            r = ratings.get(i, [])
            row = {"sample_id": i, "n": len(r), "n_reference": len(ref_r), "diff": None, "ci95": None, "family": None}
            if len(r) >= min_ratings and len(ref_r) >= min_ratings:
                lo, hi = bootstrap_diff(r, ref_r)
                row.update(diff=round(mean(r) - mean(ref_r), 3), ci95=[round(lo, 3), round(hi, 3)],
                           family="sensitivity" if hi < -margin else
                                  "invariance" if -margin <= lo and hi <= margin else "open")
            comparisons.append(row)
        families = [c["family"] for c in comparisons]
        if not refs or not families or None in families:
            family = None
        else:
            family = families[0] if len(set(families)) == 1 else "open"
        p["family_human"] = family
        pair_rows.append({"pair_id": p["pair_id"], "family_expected": p.get("family_expected"), "family_human": family,
                          "reference": refs, "comparisons": comparisons})

    # ranking_human per group.
    ranks, firsts, n_rankings = defaultdict(lambda: defaultdict(list)), defaultdict(Counter), Counter()
    for s in included:
        for r in s.get("rankings") or []:
            if "order_sample_ids" not in r or not r.get("order"):  # rankings from before the dataset groups
                continue
            n_rankings[r["set_id"]] += 1
            for k, i in enumerate(r["order"], 1):
                ranks[r["set_id"]][i].append(k)
            firsts[r["set_id"]][r["order"][0]] += 1
    group_rows = []
    for g in groups:
        gid = g["group_id"]
        images = {i: {"image_id": i, "mean_rank": round(mean(v), 3), "first": firsts[gid][i], "n": len(v)}
                  for i, v in ranks.get(gid, {}).items()}
        ranked = sorted(images, key=lambda i: (images[i]["mean_rank"], -images[i]["first"], i))
        g["ranking_human"] = ranked if n_rankings[gid] >= min_rankings else None
        group_rows.append({"group_id": gid, "axis": g.get("axis"), "n_rankings": n_rankings[gid], "ranking_human": g["ranking_human"],
                           "ranking_hypothetical": g.get("ranking_hypothetical"), "images": [images[i] for i in ranked]})

    n_ex = sum(r["excluded"] for r in session_rows)
    note = (f"{len(sessions)} sessions, {n_ex} excluded{' (kept)' if include_excluded else ''}; "
            f"{sum(map(len, ratings.values()))} ratings of {len(ratings)} samples; "
            f"family_human {dict(Counter(p['family_human'] for p in pairs))}; "
            f"ranking_human {sum(g['ranking_human'] is not None for g in groups)} of {len(groups)} groups")
    return {"pairs/pairs.jsonl": pairs, "groups/groups.jsonl": groups, "human/ratings.jsonl": rating_rows,
            "human/pairs.jsonl": pair_rows, "human/groups.jsonl": group_rows, "human/sessions.jsonl": session_rows}, note


def jsonl(rows):
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
