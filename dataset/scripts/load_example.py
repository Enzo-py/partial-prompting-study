"""Load the dataset locally (or from the Hub) and list the minimal pairs."""
import sys
from collections import defaultdict
from pathlib import Path
from datasets import load_dataset

src = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parent.parent)
ds = load_dataset(src, split="train")
print(ds)

pairs = defaultdict(list)
for row in ds:
    for pid, axis in zip(row["pair_ids"], row["pair_axes"]):
        pairs[pid].append((row["sample_id"], row["role"], axis, row["prompt"]))

for pid, members in sorted(pairs.items()):
    print(f"\n{pid}")
    for sid, role, axis, prompt in members:
        print(f"  {sid:4} {role:13} varies:{axis:6} “{prompt}”")
