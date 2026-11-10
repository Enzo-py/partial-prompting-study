"""Export the human study results in the dataset's own format (see the dataset README, Fields and Groups).

Usage:
    pip install huggingface_hub
    HF_TOKEN=... python analysis/export_human_results.py Enzo-py/partial-prompting-responses [--out human_results]
    python analysis/export_human_results.py path/to/responses      # a local copy (list*/<session>.json)

The admin page of the Space has an Export button that gives the same files as a zip. Both use
space/human_export.py, which describes the files and the rules (exclusions, mean_human_score,
details_human_score, ranking_human, details_ranking_human).
"""
import argparse
import json
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "space"))
import human_export  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("responses", help="responses repo id, or a local folder of session files")
ap.add_argument("--dataset", default="Enzo-py/partial-prompting-ds")
ap.add_argument("--revision", default=None, help="dataset revision (default: latest)")
ap.add_argument("--out", default="human_results")
ap.add_argument("--include-excluded", action="store_true", help="keep the sessions that fail the quality checks")
args = ap.parse_args()

read_jsonl = lambda p: [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
get = lambda f: read_jsonl(hf_hub_download(args.dataset, f, repo_type="dataset", revision=args.revision))

if Path(args.responses).is_dir():
    local = Path(args.responses)
else:
    local = Path(snapshot_download(args.responses, repo_type="dataset", allow_patterns="responses/**")) / "responses"
sessions = [json.loads(f.read_text()) for f in sorted(local.glob("list*/*.json"))]
if not sessions:
    raise SystemExit(f"No sessions in {args.responses}.")

files, note = human_export.export(sessions, get("data/metadata.jsonl"), get("groups/groups.jsonl"), args.include_excluded)
for path, rows in files.items():
    (Path(args.out) / path).parent.mkdir(parents=True, exist_ok=True)
    (Path(args.out) / path).write_text(human_export.jsonl(rows))
print(note)
print(f"-> {args.out}/")
