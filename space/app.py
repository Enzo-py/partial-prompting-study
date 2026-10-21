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


app.mount("/stimuli", StaticFiles(directory=STIM_DIR), name="stimuli")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")
