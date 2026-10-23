"""Backend for the partial-prompt human study (Hugging Face Space, Docker SDK).

- Builds counterbalanced lists from the stimuli dataset (one member of each minimal-pair group per participant).
- Assigns each new participant to the least-filled list.
- Receives the final session JSON and writes it to DATA_DIR; a CommitScheduler pushes DATA_DIR
  to a private dataset repo every few minutes.

Stimuli are loaded at startup from a Hugging Face dataset (DATASET_REPO), either an imagefolder
(`metadata.jsonl` + image files) or parquet with embedded images. Required columns: `sample_id`,
`prompt`, and the image (`file_name` / `image`). Optional: `human_study` (false = never shown),
`role` + `control_type` (attention checks), `pair_ids`,
`image_id`, plus any label columns (copied into the saved responses). Restart the Space to pick up
a new version of the dataset; the exact dataset commit is stored with every session.

Environment variables (set them as Space secrets / variables):
  DATASET_REPO              stimuli dataset (default "Enzo-py/partial-prompting-ds")
  DATASET_REVISION          optional branch / tag / commit to pin (default "main")
  DATASET_SPLIT             optional split (default "train")
  RESPONSES_REPO            dataset repo that receives responses, e.g. "Enzo-py/partial-prompting-responses"
  HF_TOKEN                  write token for RESPONSES_REPO (secret)
  PROLIFIC_COMPLETION_CODE  optional; participants are redirected to Prolific with this code
  STUDY_VERSION             optional label stored with every session (default "pilot-v1")
  ADMIN_KEY                 optional; protects /api/status and enables /admin
  STUDY_CONTACT             optional; researcher contact shown on the consent page
  PID_SALT                  optional secret; saves a salted hash of the Prolific ID (never the ID itself)
  LIST_COUNT, LIST_SIZE     optional; number of lists (default: just enough to show every stimulus) and
                            study images per list (default 20, at most)
  CHECKS_PER_LIST           optional; attention checks per list, rotating (default 4)
"""
import hashlib
import json
import math
import os
import random
import threading
import time
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent
STIM_DIR = Path(os.environ.get("STIM_DIR", ROOT / "stimuli_cache"))  # images extracted from the dataset
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

RESPONSES_REPO = os.environ.get("RESPONSES_REPO")
HF_TOKEN = os.environ.get("HF_TOKEN")
COMPLETION_CODE = os.environ.get("PROLIFIC_COMPLETION_CODE")
STUDY_VERSION = os.environ.get("STUDY_VERSION", "pilot-v1")
ADMIN_KEY = os.environ.get("ADMIN_KEY")
PID_SALT = os.environ.get("PID_SALT")            # optional secret; stores a hash of the Prolific ID
STUDY_CONTACT = os.environ.get("STUDY_CONTACT")  # optional; who to contact, shown on the consent page
PENDING_TTL = 30 * 60  # a started session counts towards its list for 30 min

# ---------------------------------------------------------------- stimuli & lists
DATASET_REPO = os.environ.get("DATASET_REPO", "Enzo-py/partial-prompting-ds")
DATASET_REVISION = os.environ.get("DATASET_REVISION", "main")
DATASET_SPLIT = os.environ.get("DATASET_SPLIT", "train")


def _unique_paths(api, sha, files):
    """Drop files whose content duplicates an earlier one (the same file uploaded at two paths)."""
    seen, unique = set(), []
    for info in sorted(api.get_paths_info(DATASET_REPO, files, repo_type="dataset", revision=sha),
                       key=lambda i: i.path):
        oid = info.lfs.sha256 if info.lfs else info.blob_id
        if oid in seen:
            print(f"Skipping {info.path}: identical to an already selected file")
            continue
        seen.add(oid)
        unique.append(info.path)
    return unique


def _parquet_rows(local, files):
    """Rows from parquet files with embedded images, as (row, image bytes, original image path)."""
    import pyarrow.parquet as pq
    for f in files:
        for r in pq.read_table(local / f).to_pylist():
            img = r.pop("image", None)
            data = img.get("bytes") if isinstance(img, dict) else None
            if not data:
                raise RuntimeError(f"{r.get('sample_id')}: image bytes are not embedded in {f}")
            yield r, data, img.get("path") or ""


def _imagefolder_rows(local, files):
    """Rows from imagefolder metadata.jsonl files; `file_name` is relative to the metadata file."""
    for f in files:
        folder = (local / f).parent
        for n, line in enumerate((local / f).read_text().splitlines(), 1):
            if not line.strip():
                continue
            r = json.loads(line)
            if not r.get("file_name"):
                raise RuntimeError(f"{f}, line {n}: no 'file_name'")
            path = folder / r.pop("file_name")
            if not path.is_file():
                raise RuntimeError(f"{f}, line {n}: image {path.relative_to(local)} not found")
            yield r, path.read_bytes(), path.name


def load_stimuli():
    """Download the stimuli, write the images to STIM_DIR, return (rows, commit sha).

    Reads either an imagefolder (`metadata.jsonl` next to the images) or parquet files with the
    images embedded, the two layouts the Hub uses for image datasets.
    """
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi(token=HF_TOKEN)
    sha = api.dataset_info(DATASET_REPO, revision=DATASET_REVISION).sha
    repo_files = api.list_repo_files(DATASET_REPO, repo_type="dataset", revision=sha)
    meta = [f for f in repo_files if f.split("/")[-1] == "metadata.jsonl"]
    if meta:
        # An imagefolder split lives in a folder named after it; a single folder is the default split.
        in_split = [f for f in meta if DATASET_SPLIT in f.split("/")[:-1]]
        files = _unique_paths(api, sha, in_split or meta)
        patterns = [f"{f.rsplit('/', 1)[0]}/**" if "/" in f else "**" for f in files]
        reader = _imagefolder_rows
    else:
        files = [f for f in repo_files
                 if f.endswith(".parquet") and (f"/{DATASET_SPLIT}-" in f or f"/{DATASET_SPLIT}/" in f
                                                or f.split("/")[-1].startswith(DATASET_SPLIT))]
        files = _unique_paths(api, sha, files) if files else []
        patterns, reader = files, _parquet_rows
    if not files:
        raise RuntimeError(f"No metadata.jsonl or parquet file for split '{DATASET_SPLIT}' in {DATASET_REPO}@{sha[:8]}")
    local = Path(snapshot_download(DATASET_REPO, repo_type="dataset", revision=sha,
                                   allow_patterns=patterns, token=HF_TOKEN))
    STIM_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for r, data, src in reader(local, sorted(files)):
        for col in ("sample_id", "prompt"):
            if r.get(col) is None:
                raise RuntimeError(f"Row without '{col}' in {', '.join(files)}")
        ext = Path(src).suffix.lower() or (".jpg" if data[:2] == b"\xff\xd8" else ".png")
        name = f"{r['sample_id']}{ext}"
        (STIM_DIR / name).write_bytes(data)
        r["file_name"] = name
        r["role"] = r.get("role") or "test"
        r["pair_ids"] = list(r.get("pair_ids") or [])
        r["image_id"] = r.get("image_id") or r["sample_id"]
        rows.append(r)
    ids = [r["sample_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("sample_id values must be unique")
    print(f"Loaded {len(rows)} stimuli from {DATASET_REPO}@{sha[:8]} ({', '.join(files)})")
    return rows, sha


STIMULI, DATASET_SHA = load_stimuli()
BY_ID = {s["sample_id"]: s for s in STIMULI}
# Labels for the saved files: every dataset column except the ones already sent or stored elsewhere.
LABEL_FIELDS = list(dict.fromkeys(k for s in STIMULI for k in s if k not in ("sample_id", "prompt", "file_name")))

# Public fields sent to the browser. Taxonomy labels stay server-side so participants never see them;
# visual_style is shown above the image so the drawing style is not mistaken for a mismatch.
PUBLIC_FIELDS = ("sample_id", "prompt", "file_name", "visual_style")


LIST_COUNT = int(os.environ.get("LIST_COUNT") or 0)   # number of lists; 0 = just enough to show every stimulus
LIST_SIZE = int(os.environ.get("LIST_SIZE", 20))      # study images per list (an upper bound), checks not included
CHECKS_PER_LIST = int(os.environ.get("CHECKS_PER_LIST", 4))  # attention checks per list, rotating


def in_study(s):
    """Stimuli marked human_study = false only serve the pair table and are never shown."""
    return s.get("human_study", True) is not False


def is_attention_check(s):
    """A control whose text the image plainly contradicts (`control_type`: absent, relation, ...): its
    answer is a rating of 1. Other controls (e.g. "every detail true") are the control condition of a
    pair and are counterbalanced like any other stimulus. Older datasets mark the check as case E1."""
    return s.get("role") == "control" and bool(
        s.get("control_type") or s.get("subcategory") == "Attention check" or s.get("case_id") == "E1")


def conflict(a, b):
    """Two stimuli never go in the same list: same minimal pair, same image, or same text."""
    return bool(set(a["pair_ids"]) & set(b["pair_ids"])) or a["image_id"] == b["image_id"] or a["prompt"] == b["prompt"]


def design_lists(ids, neighbours, n_lists, size, fixed=None, seed=0, restarts=10, steps=20000):
    """n_lists conflict-free lists of `size` items from `ids`, each item shown about equally often.

    `fixed[l]` are items already in list l (e.g. its attention checks): nothing conflicting with them
    goes in that list. Local search, deterministic for a given dataset: swap an item out of a list
    and another in, keeping the list conflict-free, whenever exposures get more even. Conflicting
    items (the members of a pair) are also pushed towards equal exposure, so cells stay balanced.
    """
    fixed = fixed or [set() for _ in range(n_lists)]
    pool = set(ids)
    pairs = [(a, b) for a in ids for b in neighbours[a] if b in pool and a < b]
    target = n_lists * size / len(ids)

    def cost(count):
        return (sum((count[i] - target) ** 2 + (1000 if count[i] == 0 else 0) for i in ids)
                + sum((count[a] - count[b]) ** 2 for a, b in pairs))

    best = None
    for r in range(restarts):
        rng = random.Random(seed * 1000 + r)
        count, lists = Counter(), []
        for l in range(n_lists):  # greedy start: least shown items first
            cur = set()
            for i in sorted(ids, key=lambda i: (count[i], rng.random())):
                if len(cur) < size and not neighbours[i] & (cur | fixed[l]):
                    cur.add(i)
            lists.append(cur)
            count.update(cur)
        if any(len(cur) < size for cur in lists):
            continue
        c = cost(count)
        for _ in range(steps):
            l = rng.randrange(n_lists)
            cur = lists[l]
            out = rng.choice(sorted(cur))
            blocked = (cur - {out}) | fixed[l]
            cand = [i for i in ids if i not in cur and not neighbours[i] & blocked]
            if not cand:
                continue
            new = rng.choice(cand)
            count[out] -= 1
            count[new] += 1
            c2 = cost(count)
            if c2 <= c:
                cur.remove(out)
                cur.add(new)
                c = c2
            else:
                count[out] += 1
                count[new] -= 1
        if best is None or c < best[0]:
            best = (c, [sorted(cur) for cur in lists])
    return best[1] if best else None


def build_lists(stimuli):
    """Counterbalanced lists: CHECKS_PER_LIST attention checks plus up to LIST_SIZE study images each,
    never two conflicting stimuli together (see `conflict`), so each comparison is between-subjects.
    Returns (lists, attention checks, stimuli not in the study).
    """
    study = [s for s in stimuli if in_study(s)]
    checks = sorted(s["sample_id"] for s in study if is_attention_check(s))
    items = sorted(s["sample_id"] for s in study if not is_attention_check(s))
    neighbours = {s["sample_id"]: set() for s in study}
    for a in study:
        for b in study:
            if a is not b and conflict(a, b):
                neighbours[a["sample_id"]].add(b["sample_id"])
    # Stimuli sharing a text, an image or a pair all conflict with each other, so each needs its own
    # list: the largest such group is a lower bound on the number of lists.
    groups = Counter()
    for s in study:
        if not is_attention_check(s):
            groups.update([("text", s["prompt"]), ("image", s["image_id"])] + [("pair", p) for p in s["pair_ids"]])
    min_lists = max(groups.values(), default=1)
    # Largest list size that still fits: there may be fewer conflict-free items than LIST_SIZE asks.
    for size in range(min(LIST_SIZE, len(items)), 0, -1):
        n_lists = LIST_COUNT or max(math.ceil(len(items) / size), min_lists)
        if not LIST_COUNT:  # spread the items evenly over those lists
            size = math.ceil(len(items) / n_lists)
        k = min(CHECKS_PER_LIST, len(checks))
        check_lists = (design_lists(checks, neighbours, n_lists, k) if k else None) or [[] for _ in range(n_lists)]
        lists = design_lists(items, neighbours, n_lists, size, fixed=[set(c) for c in check_lists])
        if lists:
            break
    else:
        check_lists, lists = [[]], [[]]
    not_shown = [s["sample_id"] for s in stimuli if not in_study(s)]
    return [sorted(c + l) for c, l in zip(check_lists, lists)], checks, not_shown


LISTS, CHECKS, NOT_IN_STUDY = build_lists(STIMULI)
STUDY = [s for s in STIMULI if in_study(s)]
EXPOSURE = Counter(i for l in LISTS for i in l)
(ROOT / "lists.json").write_text(json.dumps(
    {"n_lists": len(LISTS), "attention_checks": CHECKS, "not_in_study": NOT_IN_STUDY,
     "exposure": dict(sorted(EXPOSURE.items())), "lists": LISTS}, indent=2))
print(f"{len(LISTS)} lists of {sorted({len(l) for l in LISTS})} images, {min(CHECKS_PER_LIST, len(CHECKS))} attention "
      f"checks each; {len(NOT_IN_STUDY)} stimuli not in the study (human_study = false)")


# ---------------------------------------------------------------- persistence
scheduler = None
if RESPONSES_REPO and HF_TOKEN:
    from huggingface_hub import CommitScheduler
    scheduler = CommitScheduler(
        repo_id=RESPONSES_REPO, repo_type="dataset", folder_path=DATA_DIR,
        path_in_repo="responses", every=2, private=True, token=HF_TOKEN,
    )

lock = threading.Lock()
completed = Counter()
pending = {}  # session_id -> (list_id, started_at)


def recount():
    """Count completed sessions per list, from the responses repo and the local folder."""
    seen = set()
    if RESPONSES_REPO and HF_TOKEN:
        try:
            from huggingface_hub import HfApi
            for f in HfApi(token=HF_TOKEN).list_repo_files(RESPONSES_REPO, repo_type="dataset"):
                if f.startswith("responses/list") and f.endswith(".json"):
                    seen.add(f.split("/", 1)[1])
        except Exception as e:  # repo may not exist yet
            print("recount: could not list responses repo:", e)
    for p in DATA_DIR.glob("list*/*.json"):
        seen.add(f"{p.parent.name}/{p.name}")
    for f in seen:
        try:
            completed[int(f[4:6])] += 1
        except ValueError:
            pass


recount()


def choose_list():
    now = time.time()
    active = Counter(l for l, t in pending.values() if now - t < PENDING_TTL)
    return min(range(len(LISTS)), key=lambda k: (completed[k] + active[k], random.random()))


# ---------------------------------------------------------------- API
app = FastAPI(title="Partial prompt study")


@app.post("/api/start")
async def start(req: Request):
    body = await req.json()
    with lock:
        list_id = choose_list()
        session_id = uuid.uuid4().hex
        pending[session_id] = (list_id, time.time())
    stimuli = [{k: BY_ID[i].get(k) for k in PUBLIC_FIELDS} for i in LISTS[list_id]]
    return {"session_id": session_id, "list_id": list_id, "study_version": STUDY_VERSION,
            "stimuli": stimuli, "prolific": bool(body.get("prolific_pid")), "contact": STUDY_CONTACT}


@app.post("/api/submit")
async def submit(req: Request):
    body = await req.json()
    session_id = str(body.get("session_id", ""))
    if not session_id.isalnum() or len(session_id) > 64:
        raise HTTPException(400, "invalid session_id")
    list_id = int(body.get("list_id", -1))
    if not 0 <= list_id < len(LISTS):
        raise HTTPException(400, "invalid list_id")

    # Attach the taxonomy labels server-side so the stored file is self-describing.
    for t in body.get("trials", []):
        s = BY_ID.get(t.get("sample_id"))
        if s:
            for k in LABEL_FIELDS:
                t[k] = s.get(k)
    # Prolific IDs are never saved: only whether the participant came from Prolific, plus an optional
    # salted hash (PID_SALT) that lets the researchers match a Prolific submission to its session.
    participant = body.get("participant") or {}
    pid = participant.pop("prolific_pid", None)
    participant.pop("prolific_session_id", None)
    participant["source"] = "prolific" if pid else "direct"
    if pid and PID_SALT:
        participant["pid_hash"] = hashlib.sha256(f"{PID_SALT}:{pid}".encode()).hexdigest()[:16]
    body["participant"] = participant
    body["received_at"] = datetime.now(timezone.utc).isoformat()
    body["study_version"] = STUDY_VERSION
    body["dataset"] = {"repo": DATASET_REPO, "revision": DATASET_SHA, "split": DATASET_SPLIT}

    folder = DATA_DIR / f"list{list_id:02d}"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{session_id}.json"
    ctx = scheduler.lock if scheduler else lock
    with ctx:
        new = not path.exists()
        path.write_text(json.dumps(body, ensure_ascii=False, indent=1))
    with lock:
        pending.pop(session_id, None)
        if new:
            completed[list_id] += 1

    code = COMPLETION_CODE if pid else None
    redirect = f"https://app.prolific.com/submissions/complete?cc={code}" if code else None
    return {"ok": True, "completion_code": code, "redirect": redirect}


@app.get("/api/status")
def status(key: str = ""):
    if ADMIN_KEY and key != ADMIN_KEY:
        raise HTTPException(403, "add ?key=ADMIN_KEY")
    now = time.time()
    return {"study_version": STUDY_VERSION,
            "dataset": f"{DATASET_REPO}@{DATASET_SHA[:8]} ({DATASET_SPLIT}, {len(STIMULI)} stimuli)",
            "n_lists": len(LISTS), "items_per_list": [len(l) for l in LISTS],
            "completed_per_list": {k: completed[k] for k in range(len(LISTS))},
            "in_progress": sum(1 for _, t in pending.values() if now - t < PENDING_TTL),
            "saving_to": RESPONSES_REPO if scheduler else "local folder only (RESPONSES_REPO or HF_TOKEN missing)"}


# ---------------------------------------------------------------- admin summary
RESP_CACHE = ROOT / "responses_cache"  # local copy of the responses repo, refreshed by /api/summary
summary_lock = threading.Lock()
session_cache = {"at": 0.0, "sessions": {}}


def load_sessions(max_age=60):
    """Every saved session, keyed by 'listNN/<id>.json': the responses repo plus the local folder."""
    with summary_lock:
        if time.time() - session_cache["at"] < max_age:
            return session_cache["sessions"]
        files = []
        if scheduler:
            try:
                from huggingface_hub import snapshot_download
                snapshot_download(RESPONSES_REPO, repo_type="dataset", allow_patterns="responses/**",
                                  local_dir=RESP_CACHE, token=HF_TOKEN)
            except Exception as e:  # repo may not exist yet
                print("summary: could not download responses repo:", e)
            files += RESP_CACHE.glob("responses/list*/*.json")
        files += DATA_DIR.glob("list*/*.json")  # after the repo copy: local files may not be pushed yet
        sessions = {}
        for p in files:
            try:
                sessions[f"{p.parent.name}/{p.name}"] = json.loads(p.read_text())
            except (OSError, ValueError) as e:  # e.g. a file being written
                print(f"summary: skipping {p}: {e}")
        session_cache.update(at=time.time(), sessions=sessions)
        return sessions


# Quality checks. analysis/collect_responses.py applies the same rules. Each failed attention check
# counts once; a session is excluded once it fails EXCLUDE_AT checks, fewer are only a warning.
FAST_TRIAL_MS = 1000       # an image answered faster than this...
FAST_SHARE = 0.25          # ...for this share of the images or more
SAME_ANSWER_SHARE = 0.99   # share of identical ratings across a session
EXCLUDE_AT = 3             # failed checks that exclude a session


def quality_flags(session):
    """The checks a session fails, one entry per failed attention-check stimulus."""
    trials = [t for t in session.get("trials", []) if t.get("rating")]
    flags = ["attention_check" for t in trials  # each attention check expects a rating of 1
             if is_attention_check(BY_ID.get(t.get("sample_id"), {})) and t["rating"] > 2]
    if trials and sum((t.get("rt") or 0) < FAST_TRIAL_MS for t in trials) / len(trials) >= FAST_SHARE:
        flags.append("too_fast")
    ratings = Counter(t["rating"] for t in trials)
    if trials and ratings.most_common(1)[0][1] / len(trials) >= SAME_ANSWER_SHARE:
        flags.append("same_answers")
    return flags


def is_excluded(session):
    return len(quality_flags(session)) >= EXCLUDE_AT


def median(values):
    v = sorted(values)
    return (v[len(v) // 2] + v[(len(v) - 1) // 2]) / 2 if v else None


def mean(values):
    return sum(values) / len(values) if values else None


def summarize(sessions):
    """Progress per list and response statistics. Excluded sessions (EXCLUDE_AT failed checks) are
    counted in the progress but left out of the response statistics."""
    now = time.time()
    active = Counter(l for l, t in pending.values() if now - t < PENDING_TTL)
    done = Counter(s.get("list_id") for s in sessions)
    flags = {id(s): quality_flags(s) for s in sessions}
    included = [s for s in sessions if len(flags[id(s)]) < EXCLUDE_AT]

    trials = defaultdict(list)
    for s in included:
        for t in s.get("trials", []):
            if t.get("sample_id") in BY_ID and t.get("rating"):
                trials[t["sample_id"]].append(t)

    stimuli = []
    for st in STUDY:
        ts = trials[st["sample_id"]]
        ratings = [t["rating"] for t in ts]
        stimuli.append({
            **{k: st.get(k) for k in ("sample_id", "prompt", "file_name", "case_id", "category",
                                      "category_name", "subcategory", "role")},
            "attention_check": is_attention_check(st),
            "n": len(ts), "mean": mean(ratings),
            "dist": [ratings.count(r) for r in range(1, 6)],
            "unwanted_rate": mean([t.get("unwanted") == "yes" for t in ts]),
            "confidence": {c: sum(t.get("confidence") == c for t in ts) for c in ("low", "mid", "high")},
            "median_rt_s": (median([t["rt"] for t in ts if t.get("rt")]) or 0) / 1000 or None,
            "marks": [[m["x"], m["y"]] for t in ts for m in t.get("marks", [])],
        })

    by_cat = defaultdict(list)
    for st in STUDY:
        if not is_attention_check(st):
            by_cat[(st.get("category"), st.get("category_name"))] += trials[st["sample_id"]]
    categories = [{"category": c, "name": name, "n": len(ts), "mean": mean([t["rating"] for t in ts]),
                   "unwanted_rate": mean([t.get("unwanted") == "yes" for t in ts])}
                  for (c, name), ts in sorted(by_cat.items(), key=lambda kv: str(kv[0][0]))]

    by_pair = defaultdict(list)
    for st in STUDY:
        roles = list(st.get("pair_roles") or [])
        for i, pid in enumerate(st["pair_ids"]):
            ratings = [t["rating"] for t in trials[st["sample_id"]]]
            by_pair[pid].append({"sample_id": st["sample_id"], "prompt": st["prompt"],
                                 "pair_role": roles[i] if i < len(roles) else None,
                                 "n": len(ratings), "mean": mean(ratings)})
    pairs = [{"pair_id": pid, "members": members} for pid, members in sorted(by_pair.items())]

    recent = sorted(sessions, key=lambda s: s.get("received_at") or "", reverse=True)[:100]
    return {
        "study_version": STUDY_VERSION,
        "dataset": f"{DATASET_REPO}@{DATASET_SHA[:8]} ({DATASET_SPLIT}, {len(STIMULI)} stimuli)",
        "saving_to": RESPONSES_REPO if scheduler else "local folder only (RESPONSES_REPO or HF_TOKEN missing)",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "totals": {"completed": len(sessions), "excluded": len(sessions) - len(included),
                   "exclude_at": EXCLUDE_AT,
                   "flagged": sum(0 < len(flags[id(s)]) < EXCLUDE_AT for s in sessions),
                   "excluded_by": dict(Counter(f for s in sessions if len(flags[id(s)]) >= EXCLUDE_AT for f in flags[id(s)])),
                   "in_progress": sum(active.values()),
                   "median_minutes": (median([s["total_time_ms"] for s in sessions
                                              if s.get("total_time_ms")]) or 0) / 60000 or None},
        "lists": [{"list_id": k, "items": len(LISTS[k]), "completed": done[k], "in_progress": active[k]}
                  for k in range(len(LISTS))],
        "categories": categories, "pairs": pairs, "stimuli": stimuli,
        # No Prolific IDs here: the page only needs to tell sessions apart.
        "sessions": [{"session": str(s.get("session_id", ""))[:8], "list_id": s.get("list_id"),
                      "received_at": s.get("received_at"),
                      "minutes": (s.get("total_time_ms") or 0) / 60000 or None,
                      "prolific": s.get("participant", {}).get("source") == "prolific"
                                  or bool(s.get("participant", {}).get("prolific_pid")),  # sessions saved before IDs were dropped
                      "english": s.get("participant", {}).get("english"),
                      "t2i": s.get("participant", {}).get("t2i"),
                      "flags": flags[id(s)], "excluded": len(flags[id(s)]) >= EXCLUDE_AT}
                     for s in recent],
    }


@app.get("/api/summary")
def summary(key: str = "", refresh: bool = False):
    # Unlike /api/status, this exposes response content, so it stays closed until ADMIN_KEY is set.
    if not ADMIN_KEY:
        raise HTTPException(403, "set the ADMIN_KEY secret on the Space to enable the dashboard")
    if key != ADMIN_KEY:
        raise HTTPException(403, "wrong key")
    return summarize(list(load_sessions(max_age=0 if refresh else 60).values()))


# ---------------------------------------------------------------- participant feedback
FEEDBACK_MIN_OTHERS = 5   # a stimulus is compared only once this many other participants rated it
FEEDBACK_MIN_ITEMS = 3    # below this many comparable stimuli, no feedback is shown


def feedback_for(session):
    """Compare a participant's ratings with the other participants', after their answers are saved.

    Attention checks are never used (that would reveal them), nor are excluded sessions. Returns None until enough other participants have rated this participant's stimuli.
    """
    others = defaultdict(list)
    for s in load_sessions().values():
        if s.get("session_id") == session.get("session_id") or is_excluded(s):
            continue
        for t in s.get("trials", []):
            if t.get("rating"):
                others[t.get("sample_id")].append(t["rating"])
    rows = []
    for t in session.get("trials", []):
        st = BY_ID.get(t.get("sample_id"))
        if st and not is_attention_check(st) and t.get("rating") and len(others[st["sample_id"]]) >= FEEDBACK_MIN_OTHERS:
            ratings = others[st["sample_id"]]
            rows.append((st, t["rating"], mean(ratings), len(ratings)))
    if len(rows) < FEEDBACK_MIN_ITEMS:
        return None
    st, rating, m, n = max(rows, key=lambda r: abs(r[1] - r[2]))
    return {
        "n_compared": len(rows),
        "mean_diff": round(mean([r - m for _, r, m, _ in rows]), 2),
        "n_close": sum(abs(r - m) <= 1 for _, r, m, _ in rows),
        # Only a real disagreement is worth showing.
        "most_different": {**{k: st.get(k) for k in PUBLIC_FIELDS}, "rating": rating,
                           "others_mean": round(m, 1), "others_n": n} if abs(rating - m) >= 1 else None,
    }


@app.post("/api/feedback")
async def feedback(req: Request):
    body = await req.json()
    session_id, list_id = str(body.get("session_id", "")), body.get("list_id")
    if not session_id.isalnum() or len(session_id) > 64 or not isinstance(list_id, int):
        raise HTTPException(400, "invalid session")
    path = DATA_DIR / f"list{list_id:02d}" / f"{session_id}.json"
    if not path.exists():  # feedback only for saved sessions
        raise HTTPException(404, "session not found")
    return {"feedback": feedback_for(json.loads(path.read_text()))}


@app.get("/admin")
def admin():
    return FileResponse(ROOT / "static" / "admin.html")


app.mount("/stimuli", StaticFiles(directory=STIM_DIR), name="stimuli")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")
