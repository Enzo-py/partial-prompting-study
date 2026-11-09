# Partial Prompting Study

How do humans judge content that a text-to-image prompt does not mention? *"A chair on the grass"* says nothing about the sky, and *"an apple"* says nothing about a pear next to it, yet every generated image has to fill these gaps. This repository holds the human study that measures when people penalize unmentioned content: the taxonomy of cases, the annotation interface, and the analysis.

| Part | Where it lives | Folder here |
|---|---|---|
| Stimuli (images, prompts, minimal pairs) | Dataset [`Enzo-py/partial-prompting-ds`](https://huggingface.co/datasets/Enzo-py/partial-prompting-ds) | `dataset/` (card, taxonomy, scripts; no images) |
| Annotation interface | Space [`Enzo-py/partial-prompting-study`](https://huggingface.co/spaces/Enzo-py/partial-prompting-study) | `space/` |
| Responses | Private dataset `Enzo-py/partial-prompting-responses` | `analysis/` (scripts only) |

```
partial-prompting-study/
├── space/                    annotation Space (FastAPI + jsPsych), deployed automatically
├── dataset/
│   ├── hub/                  files synced to the stimuli dataset: card, taxonomy/, annotation/
│   ├── svg/                  vector sources of the placeholder stimuli
│   └── scripts/              push_to_hub.py, load_example.py
├── analysis/                 collect_responses.py, export_human_results.py
└── .github/workflows/        deploy-space.yml, sync-dataset-card.yml
```

Images stay on the Hugging Face dataset, not in Git.

## How deployment works

| You change | On push to `main` | Result |
|---|---|---|
| `space/**` | `deploy-space.yml` mirrors `space/` to the Space | The Space rebuilds (2–3 min) |
| `dataset/hub/**` | `sync-dataset-card.yml` uploads the card and taxonomy | Images and data on the Hub are untouched |
| Images or prompts on the dataset | nothing | Restart the Space (Settings → Restart) so it reloads the stimuli |

Both workflows can also be run by hand from the **Actions** tab (*Run workflow*).

### One-time setup

1. In this GitHub repository: **Settings → Secrets and variables → Actions → New repository secret**, name `HF_TOKEN`, value a Hugging Face token with write access to the Space and the dataset.
2. Optional, under the **Variables** tab: `HF_SPACE` and `HF_DATASET` if the Space or dataset ever move.
3. On the Space itself (Hugging Face → Space → Settings → Variables and secrets): `HF_TOKEN`, `RESPONSES_REPO`, `ADMIN_KEY` (opens the `/admin` dashboard) and optionally `PROLIFIC_COMPLETION_CODE`. See [`space/README.md`](space/README.md).

## Analysis

```bash
pip install huggingface_hub pandas pyarrow
python analysis/collect_responses.py Enzo-py/partial-prompting-responses --out responses.parquet
```

Downloads every session, flattens to one row per trial, and flags the quality checks (attention check, many answers under 1 s, identical ratings), excluding a session from 2 failed checks. Downloaded responses are git-ignored; they stay in the private repo.

To fill the dataset's post-study fields (`family_human` in `pairs/pairs.jsonl`, `ranking_human` in `groups/groups.jsonl`), click **Export results** on the Space's admin page, or from a terminal:

```bash
HF_TOKEN=... python analysis/export_human_results.py Enzo-py/partial-prompting-responses --out human_results
```

It writes both files in the dataset's format (every line and field kept, ready to replace the dataset's), plus `human/` with the numbers behind them: ratings per stimulus, the test-minus-reference difference with its 95% interval per pair, mean ranks per group, and each session's flags. Same exclusions as the Space. A pair is `sensitivity` when the interval lies below -0.5, `invariance` when it lies within ±0.5, else `open`; null under 10 ratings per member (`--margin`, `--min-ratings`, `--min-rankings`). A local folder of session files works in place of the repo id. The button and the script share `space/human_export.py`, so they give the same files.

## Taxonomy

25 cases in 7 categories (forced completion, atypical completion, implied content, optional addition, salience, control, generator-specific) and 7 cross-cutting axes. See [`dataset/hub/taxonomy/taxonomy.md`](dataset/hub/taxonomy/taxonomy.md).
