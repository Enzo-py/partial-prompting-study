"""Backend for the partial-prompt human study (Hugging Face Space, Docker SDK).

- Builds counterbalanced lists from the stimuli dataset (one member of each minimal-pair group per participant).
- Assigns each new participant to the least-filled list.
- Receives the final session JSON and writes it to DATA_DIR; a CommitScheduler pushes DATA_DIR
  to a private dataset repo every few minutes.

Stimuli are loaded at startup from a Hugging Face dataset (DATASET_REPO). Required columns:
`sample_id`, `prompt`, `image`. Optional: `role` ("control" = shown to everyone), `pair_ids`,
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
  ADMIN_KEY                 optional; protects /api/status
"""
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
PENDING_TTL = 30 * 60  # a started session counts towards its list for 30 min

# ---------------------------------------------------------------- stimuli & lists
DATASET_REPO = os.environ.get("DATASET_REPO", "Enzo-py/partial-prompting-ds")
DATASET_REVISION = os.environ.get("DATASET_REVISION", "main")
DATASET_SPLIT = os.environ.get("DATASET_SPLIT", "train")


def load_stimuli():
    """Download the split's parquet files, write the images to STIM_DIR, return (rows, commit sha)."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi(token=HF_TOKEN)
    sha = api.dataset_info(DATASET_REPO, revision=DATASET_REVISION).sha
    files = [f for f in api.list_repo_files(DATASET_REPO, repo_type="dataset", revision=sha)
             if f.endswith(".parquet") and (f"/{DATASET_SPLIT}-" in f or f"/{DATASET_SPLIT}/" in f
                                            or f.split("/")[-1].startswith(DATASET_SPLIT))]
    if not files:
        raise RuntimeError(f"No parquet file for split '{DATASET_SPLIT}' in {DATASET_REPO}@{sha[:8]}")
    # The same parquet file uploaded at two paths (e.g. root and data/) would duplicate every row.
    seen, unique = set(), []
    for info in sorted(api.get_paths_info(DATASET_REPO, files, repo_type="dataset", revision=sha),
                       key=lambda i: i.path):
        oid = info.lfs.sha256 if info.lfs else info.blob_id
        if oid in seen:
            print(f"Skipping {info.path}: identical to an already selected file")
            continue
        seen.add(oid)
        unique.append(info.path)
    files = unique
    local = Path(snapshot_download(DATASET_REPO, repo_type="dataset", revision=sha,
                                   allow_patterns=files, token=HF_TOKEN))
    STIM_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in sorted(files):
        for r in pq.read_table(local / f).to_pylist():
            for col in ("sample_id", "prompt", "image"):
                if r.get(col) is None:
                    raise RuntimeError(f"Row without '{col}' in {f}")
            img = r.pop("image")
            data = img.get("bytes") if isinstance(img, dict) else None
            if not data:
                raise RuntimeError(f"{r['sample_id']}: image bytes are not embedded in the parquet file")
            ext = Path(img.get("path") or "").suffix.lower() or (".jpg" if data[:2] == b"\xff\xd8" else ".png")
            name = f"{r['sample_id']}{ext}"
            (STIM_DIR / name).write_bytes(data)
            r["file_name"] = name
            r.setdefault("role", "test")
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
LABEL_FIELDS = [k for k in STIMULI[0] if k not in ("sample_id", "prompt", "file_name")]

# Public fields sent to the browser. Taxonomy labels stay server-side so participants never see them.
PUBLIC_FIELDS = ("sample_id", "prompt", "file_name")


def build_lists(stimuli):
    """Group stimuli that must not be seen by the same person, then rotate through groups.

    Two stimuli are grouped when they belong to the same minimal pair (pair_ids) or reuse the same
    image (image_id). Controls are shown to everyone. Each list takes one member from every group, so
    every comparison inside a group is between-subjects. The number of lists is the LCM of group
    sizes, which keeps every member equally represented.
    """
    parent = {s["sample_id"]: s["sample_id"] for s in stimuli}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    by_key = defaultdict(list)
    for s in stimuli:
        if s["role"] == "control":
            continue
        for pid in s["pair_ids"]:
            by_key["pair:" + pid].append(s["sample_id"])
        by_key["img:" + s["image_id"]].append(s["sample_id"])
    for members in by_key.values():
        for m in members[1:]:
            union(members[0], m)

    groups = defaultdict(list)
    for s in stimuli:
        if s["role"] == "control":
            continue
        groups[find(s["sample_id"])].append(s["sample_id"])
    fixed = [s["sample_id"] for s in stimuli if s["role"] == "control"]
    rotating = []
    for members in groups.values():
        members.sort()
        if len(members) == 1:
            fixed += members
        else:
            rotating.append(members)
    n_lists = math.lcm(*[len(g) for g in rotating]) if rotating else 1
    lists = [sorted(fixed + [g[k % len(g)] for g in rotating]) for k in range(n_lists)]
    return lists, rotating, fixed


LISTS, GROUPS, FIXED = build_lists(STIMULI)
(ROOT / "lists.json").write_text(json.dumps(
    {"n_lists": len(LISTS), "always_shown": FIXED, "rotating_groups": GROUPS, "lists": LISTS}, indent=2))

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
    stimuli = [{k: BY_ID[i][k] for k in PUBLIC_FIELDS} for i in LISTS[list_id]]
    return {"session_id": session_id, "list_id": list_id, "study_version": STUDY_VERSION,
            "stimuli": stimuli, "prolific": bool(body.get("prolific_pid"))}


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

    redirect = (f"https://app.prolific.com/submissions/complete?cc={COMPLETION_CODE}"
                if COMPLETION_CODE and body.get("participant", {}).get("prolific_pid") else None)
    return {"ok": True, "completion_code": COMPLETION_CODE if body.get("participant", {}).get("prolific_pid") else None,
            "redirect": redirect}


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


def failed_attention_check(session):
    """Same rule as analysis/collect_responses.py: a rating above 2 on a control stimulus."""
    return any((t.get("rating") or 0) > 2 for t in session.get("trials", [])
               if BY_ID.get(t.get("sample_id"), {}).get("role") == "control")


def median(values):
    v = sorted(values)
    return (v[len(v) // 2] + v[(len(v) - 1) // 2]) / 2 if v else None


def mean(values):
    return sum(values) / len(values) if values else None


def summarize(sessions):
    """Progress per list and response statistics. Sessions that fail the attention check are
    counted in the progress but left out of the response statistics."""
    now = time.time()
    active = Counter(l for l, t in pending.values() if now - t < PENDING_TTL)
    done = Counter(s.get("list_id") for s in sessions)
    included = [s for s in sessions if not failed_attention_check(s)]

    trials = defaultdict(list)
    for s in included:
        for t in s.get("trials", []):
            if t.get("sample_id") in BY_ID and t.get("rating"):
                trials[t["sample_id"]].append(t)

    stimuli = []
    for st in STIMULI:
        ts = trials[st["sample_id"]]
        ratings = [t["rating"] for t in ts]
        stimuli.append({
            **{k: st.get(k) for k in ("sample_id", "prompt", "file_name", "case_id", "category",
                                      "category_name", "subcategory", "role", "expected_penalty")},
            "n": len(ts), "mean": mean(ratings),
            "dist": [ratings.count(r) for r in range(1, 6)],
            "unwanted_rate": mean([t.get("unwanted") == "yes" for t in ts]),
            "confidence": {c: sum(t.get("confidence") == c for t in ts) for c in ("low", "mid", "high")},
            "median_rt_s": (median([t["rt"] for t in ts if t.get("rt")]) or 0) / 1000 or None,
            "marks": [[m["x"], m["y"]] for t in ts for m in t.get("marks", [])],
        })

    by_cat = defaultdict(list)
    for st in STIMULI:
        if st["role"] != "control":
            by_cat[(st.get("category"), st.get("category_name"))] += trials[st["sample_id"]]
    categories = [{"category": c, "name": name, "n": len(ts), "mean": mean([t["rating"] for t in ts]),
                   "unwanted_rate": mean([t.get("unwanted") == "yes" for t in ts])}
                  for (c, name), ts in sorted(by_cat.items(), key=lambda kv: str(kv[0][0]))]

    by_pair = defaultdict(list)
    for st in STIMULI:
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
                      "prolific": bool(s.get("participant", {}).get("prolific_pid")),
                      "english": s.get("participant", {}).get("english"),
                      "t2i": s.get("participant", {}).get("t2i"),
                      "failed_check": failed_attention_check(s)}
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

    Controls are never used (they would reveal the attention check), nor are sessions that failed
    it. Returns None until enough other participants have rated this participant's stimuli.
    """
    others = defaultdict(list)
    for s in load_sessions().values():
        if s.get("session_id") == session.get("session_id") or failed_attention_check(s):
            continue
        for t in s.get("trials", []):
            if t.get("rating"):
                others[t.get("sample_id")].append(t["rating"])
    rows = []
    for t in session.get("trials", []):
        st = BY_ID.get(t.get("sample_id"))
        if st and st["role"] != "control" and t.get("rating") and len(others[st["sample_id"]]) >= FEEDBACK_MIN_OTHERS:
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
        "most_different": {**{k: st[k] for k in PUBLIC_FIELDS}, "rating": rating,
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
