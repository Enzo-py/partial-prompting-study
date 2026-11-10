"""Quality checks and the export of the human results in the dataset's own format.

Shared by the Space (app.py: exclusions, admin page, export button) and analysis/export_human_results.py,
so both always exclude the same sessions and produce the same files.

`export()` returns, by path:
  data/metadata.jsonl    the dataset's samples with `mean_human_score` and `details_human_score` filled
  groups/groups.jsonl    the dataset's groups with `ranking_human` and `details_ranking_human` filled
  human/sessions.jsonl   per session: list, failed checks, excluded or not

The first two keep every line and field of the dataset files, so they can replace them as they are.
Only the sessions that pass the quality checks count. A rating counts only when its saved text is the
sample's current text (sample ids of an older dataset may now name another sample), and a ranking only
when it ranked exactly the group's current members.

  details_human_score    one entry per rating: annotator (the session id), score, list, duration_s
  mean_human_score       the mean of those scores; null without ratings
  details_ranking_human  one entry per annotator: annotator, ranking (best first), list, duration_s
  ranking_human          the members (image_id for an image group, the prompt for a prompt group)
                         by mean rank, ties broken by the number of first places; null without rankings
"""
import json
from collections import Counter, defaultdict
from statistics import mean

# Quality checks. Each failed attention check counts once; a session is excluded once it fails
# EXCLUDE_AT checks, fewer are only a warning.
FAST_TRIAL_MS = 1000       # an image answered faster than this...
FAST_SHARE = 0.25          # ...for this share of the images or more
SAME_ANSWER_SHARE = 0.99   # share of identical ratings across a session
EXCLUDE_AT = 2             # failed checks that exclude a session
# A check fails when the rating says the image matches. Only a missing object makes "Partly" (3)
# wrong; a wrong colour, count or relation can honestly be rated 3, so those fail from "Mostly" (4).
CHECK_FAIL_AT = {"absent": 3, "missing_object": 3}
CHECK_FAIL_AT_DEFAULT = 4


def is_attention_check(s):
    """`category` = attention_check in the current dataset. Older datasets (and the sessions saved with
    them): an `attention_check` flag, or a control whose text the image plainly contradicts
    (`control_type`), or case E1; other controls were the control condition of a pair."""
    if s.get("category") == "attention_check":
        return True
    if "attention_check" in s:
        return bool(s["attention_check"])
    return s.get("role") == "control" and bool(
        s.get("control_type") or s.get("subcategory") == "Attention check" or s.get("case_id") == "E1")


def failed_check(t, by_id):
    """Whether a trial is an attention check rated as matching. Uses the labels saved with the trial,
    so a session is judged on the samples it was shown even if the dataset changed since."""
    s = t if ("category" in t or "control_type" in t) else by_id.get(t.get("sample_id"), {})  # older sessions: current labels
    if not is_attention_check(s):
        return False
    # The kind of error: `level` in the current dataset, `control_type` before; the original check
    # (E1, "a banana") is an absent object.
    kind = s.get("control_type") or (s.get("level") if s.get("category") == "attention_check" else None) or "absent"
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


def member_key(group, m):
    """What a ranking orders: the image for an image group, the prompt for a prompt group."""
    return m["prompt"] if group.get("axis") == "prompt" else m["image_id"]


def seconds(ms):
    return round(ms / 1000, 1) if isinstance(ms, (int, float)) else None


def export(sessions, meta, groups, include_excluded=False):
    """{path: rows} for the files listed in the module docstring, and a one-line summary."""
    by_id = {s["sample_id"]: s for s in meta}
    # A fixed order, so that the files do not depend on how the sessions were read.
    sessions = sorted(sessions, key=lambda s: (str(s.get("received_at") or ""), str(s.get("session_id") or "")))
    session_rows, included = [], []
    for s in sessions:
        flags = quality_flags(s, by_id)
        excluded = len(flags) >= EXCLUDE_AT
        session_rows.append({"session_id": s.get("session_id"), "list_id": s.get("list_id"), "study_version": s.get("study_version"),
                             "received_at": s.get("received_at"), "flags": flags, "excluded": excluded})
        if include_excluded or not excluded:
            included.append(s)

    # Ratings per sample.
    details, stale = defaultdict(list), 0
    for s in included:
        for t in s.get("trials", []):
            st = by_id.get(t.get("sample_id"))
            if not t.get("rating") or t.get("trial_kind", "stimulus") != "stimulus":
                continue
            if not st or t.get("prompt", st["prompt"]) != st["prompt"]:
                stale += 1
                continue
            details[st["sample_id"]].append({"annotator": s.get("session_id"), "score": t["rating"],
                                             "list": s.get("list_id"), "duration_s": seconds(t.get("rt"))})
    samples = []
    for row in meta:
        d = details.get(row["sample_id"], [])
        samples.append({**row, "mean_human_score": round(mean(x["score"] for x in d), 3) if d else None,
                        "details_human_score": d})

    # Rankings per group.
    rankings, stale_r = defaultdict(list), 0
    by_group = {g["group_id"]: g for g in groups}
    for s in included:
        for r in s.get("rankings") or []:
            g = by_group.get(r.get("set_id"))
            order = r.get("order") or []
            if not g or sorted(order) != sorted(member_key(g, m) for m in g["members"]):
                stale_r += 1
                continue
            rankings[g["group_id"]].append({"annotator": s.get("session_id"), "ranking": order,
                                            "list": s.get("list_id"), "duration_s": seconds(r.get("rt"))})
    out_groups = []
    for g in groups:
        d = rankings.get(g["group_id"], [])
        ranks, firsts = defaultdict(list), Counter()
        for x in d:
            for k, i in enumerate(x["ranking"], 1):
                ranks[i].append(k)
            firsts[x["ranking"][0]] += 1
        order = sorted(ranks, key=lambda i: (mean(ranks[i]), -firsts[i], i)) if d else None
        out_groups.append({**g, "ranking_human": order, "details_ranking_human": d})

    n_ex = sum(r["excluded"] for r in session_rows)
    note = (f"{len(sessions)} sessions, {n_ex} excluded{' (kept)' if include_excluded else ''}; "
            f"{sum(map(len, details.values()))} ratings of {len(details)} samples; "
            f"{sum(map(len, rankings.values()))} rankings of {len(rankings)} groups"
            + (f"; {stale} ratings and {stale_r} rankings left out (an older dataset)" if stale or stale_r else ""))
    return {"data/metadata.jsonl": samples, "groups/groups.jsonl": out_groups, "human/sessions.jsonl": session_rows}, note


def jsonl(rows):
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
