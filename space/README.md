---
title: Image and Text Study
emoji: 🖼️
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: Human study on unmentioned content in text-to-image prompts
---

# Partial prompt study: annotation Space

Collection interface for the partial-prompt human study. Participants see a text and an image, give a 1–5 match rating, say whether something should not be there (and click on it), and report their confidence. Built with jsPsych; a small FastAPI backend loads the stimuli from the dataset [`Enzo-py/partial-prompting-ds`](https://huggingface.co/datasets/Enzo-py/partial-prompting-ds), assigns counterbalanced lists and saves responses to a **private dataset repo**.

## How it works

```
startup: download DATASET_REPO (parquet) → extract images to stimuli_cache/, build the lists

participant ──► Space (public URL, no login)
                 ├─ POST /api/start   → assigns the least-filled list, returns its stimuli (no labels)
                 ├─ jsPsych trials     → random order, response times per trial
                 └─ POST /api/submit  → writes data/listNN/<session>.json
                                          └─ CommitScheduler pushes data/ every 2 min
                                             → <RESPONSES_REPO>/responses/listNN/<session>.json
```

**Counterbalancing.** Stimuli that belong to the same minimal pair (`pair_ids`) or reuse the same image (`image_id`) are grouped, and each participant sees exactly one member of each group, so every comparison is between-subjects. Controls and stand-alone stimuli are shown to everyone. With the current 23 stimuli this gives **6 lists of 13 items** (`/api/status` shows the current numbers). Recruit participants in multiples of 6 for balanced cells. On startup the backend writes the lists to `lists.json`.

**Taxonomy labels stay server-side.** The browser only receives `sample_id`, `prompt` and the image. Labels are attached to the saved file on the server.

## Deploy

1. **Create the responses repo** (private): `hf repos create Enzo-py/partial-prompting-responses --repo-type dataset --private`. The Space also creates it on first save if it does not exist.
2. **Create a token** with write access to that repo (Settings → Access Tokens; a fine-grained token restricted to the repo is safest).
3. **Create the Space**: New Space → SDK **Docker** → blank → visibility **Public** (participants must reach it without an account).
4. In the Space's **Settings → Variables and secrets**, add:

   | Name | Type | Value |
   |---|---|---|
   | `HF_TOKEN` | Secret | the write token |
   | `DATASET_REPO` | Variable | optional, default `Enzo-py/partial-prompting-ds` |
   | `DATASET_REVISION` | Variable | optional, a branch, tag or commit to pin (default `main`) |
   | `DATASET_SPLIT` | Variable | optional, default `train` |
   | `RESPONSES_REPO` | Variable | `Enzo-py/partial-prompting-responses` |
   | `PROLIFIC_COMPLETION_CODE` | Secret | optional, the code from your Prolific study |
   | `ADMIN_KEY` | Secret | protects `/api/status` and enables the `/admin` dashboard |
   | `STUDY_VERSION` | Variable | optional, e.g. `pilot-v1` |

5. **Deploy.** Pushing to `main` on GitHub deploys this folder automatically (see the repository README). To deploy by hand from the repository root:
   ```bash
   hf upload Enzo-py/partial-prompting-study space . --repo-type space --delete="*"
   ```
6. Check `https://enzo-py-partial-prompting-study.hf.space/api/status?key=ADMIN_KEY`. It should show the dataset commit and stimulus count, and `"saving_to": "Enzo-py/partial-prompting-responses"`.

## Prolific

Study URL:
```
https://enzo-py-partial-prompting-study.hf.space/?PROLIFIC_PID={{%PROLIFIC_PID%}}&STUDY_ID={{%STUDY_ID%}}&SESSION_ID={{%SESSION_ID%}}
```
With `PROLIFIC_COMPLETION_CODE` set, Prolific participants see the code and are redirected at the end. Without Prolific parameters (colleagues, lab pilot), the page just thanks them.

## Using a new version of the dataset

The stimuli are read once, when the Space starts. After you push new images or prompts to the dataset, restart the Space (Settings → Restart). Every saved session records the exact dataset commit (`dataset.revision`), so responses from different versions stay distinguishable. For the real study, pin `DATASET_REVISION` to a tag or commit so the stimuli cannot change during collection.

**Dataset columns used.** Required: `sample_id` (unique), `prompt`, `image` (embedded in the parquet file). Optional: `role` (`control` = shown to everyone), `pair_ids` and `image_id` (used to build the lists). Every other column is copied as a label into the saved responses, server-side only.

## Before launching

- A free Space goes to sleep after a period without visits and takes a few seconds to wake up. Open the Space yourself just before you publish the study.
- Responses are written locally first and pushed every 2 minutes. A restart can lose at most the last 2 minutes; wait for the commit before restarting or redeploying during data collection.
- `S12` (attention check) reuses the `apple_red` image that also appears in `S18`/`S23`. Give the control its own image when the real stimuli are generated.
- The images are still placeholder illustrations.

## Saved data (one file per session)

```json
{
  "session_id": "…", "list_id": 5, "study_version": "pilot-v1",
  "dataset": {"repo": "Enzo-py/partial-prompting-ds", "revision": "<commit sha>", "split": "train"},
  "participant": {"prolific_pid": "…", "study_id": "…", "prolific_session_id": "…", "english": "fluent", "t2i": "sometimes"},
  "trials": [{
    "position": 1, "sample_id": "S09", "prompt": "an apple on a table",
    "rating": 4, "unwanted": "yes", "marks": [{"x": 0.147, "y": 0.180}], "confidence": "mid",
    "rt": 5310, "rt_first_rating": 1820, "rating_changes": 1,
    "case_id": "C7", "category": "C", "subcategory": "Semantic violation", "role": "test", "pair_ids": [], "...": "…"
  }],
  "total_time_ms": 240000, "client": {"user_agent": "…", "viewport": "…"}, "received_at": "…"
}
```
Click marks are normalized to the image (0–1 on each axis).

## Dashboard

`https://enzo-py-partial-prompting-study.hf.space/admin` shows the study while it runs: sessions completed and in progress, progress per list, exclusions, mean ratings by category, each minimal pair side by side, a table of every stimulus (rating distribution, "should not be there" rate, response time; click a row to see the image with every click mark) and the recent sessions.

The page asks for `ADMIN_KEY` and stays closed until that secret is set. It reads the sessions from `RESPONSES_REPO` plus the ones not pushed yet, refreshed at most once a minute (the Refresh button forces it). Sessions that fail the attention check count in the progress but not in the statistics, as in `collect_responses.py`. Prolific IDs are never sent to the page.

## Analysis

```bash
python analysis/collect_responses.py Enzo-py/partial-prompting-responses --out responses.parquet
```
Downloads every session, flattens to one row per trial, and flags participants who failed the attention check (rating above 2 on the control).

## Files

```
app.py               backend (dataset loading, lists, assignment, saving)
static/index.html    jsPsych experiment
static/admin.html    dashboard (/admin)
static/vendor/       jsPsych 7.3.4 (vendored, no CDN dependency)
scripts/             collect_responses.py
```

## Local test

```bash
pip install -r requirements.txt
uvicorn app:app --port 7860      # downloads the dataset; responses go to ./data without RESPONSES_REPO
```
