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
├── analysis/                 collect_responses.py
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

Downloads every session, flattens to one row per trial, and flags participants who failed the attention check. Downloaded responses are git-ignored; they stay in the private repo.

## Taxonomy

25 cases in 7 categories (forced completion, atypical completion, implied content, optional addition, salience, control, generator-specific) and 7 cross-cutting axes. See [`dataset/hub/taxonomy/taxonomy.md`](dataset/hub/taxonomy/taxonomy.md).
