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
startup: download DATASET_REPO (imagefolder or parquet) → copy images to stimuli_cache/, build the lists

participant ──► Space (public URL, no login)
                 ├─ POST /api/start   → assigns the least-filled list, returns its stimuli (no labels)
                 ├─ jsPsych trials     → random order, response times per trial
                 └─ POST /api/submit  → writes data/listNN/<session>.json
                                          └─ CommitScheduler pushes data/ every 2 min
                                             → <RESPONSES_REPO>/responses/listNN/<session>.json
```

**Counterbalancing.** Two stimuli never go in the same list when they belong to the same minimal pair (`pair_ids`), reuse the same image (`image_id`) or share the same text, so every comparison is between-subjects and nobody sees a text twice. Controls are shown to everyone; a stimulus that conflicts with a control can never be shown and is left out. On startup the backend builds `LIST_COUNT` lists (default 5) of at most `LIST_SIZE` images (default 20, control included), as large as the conflicts allow, with each stimulus shown about equally often and the two members of a pair as evenly as possible. The search is deterministic: the same dataset always gives the same lists. They are written to `lists.json` (with how many lists each stimulus is in) and `/api/status` shows the sizes. Recruit participants in multiples of the number of lists.

**Session.** Consent, a two-question profile, instructions, one practice image (`static/practice.svg`, not saved), then the list in random order. Answering *Yes* to question 2 requires clicking at least one element. Leaving or reloading the page asks for confirmation, since it loses the answers. For each image the session stores its `position` (presentation order) and `rt` (time on the image, from display to *Next*).

**Quality filters.** A session is excluded from the statistics when it fails any of: the attention check (rating above 2 on the control), an image answered in under 1 s, or 99% or more identical ratings (`FAST_TRIAL_MS`, `SAME_ANSWER_SHARE` in `app.py`; `analysis/collect_responses.py` applies the same rules). Excluded sessions still count towards list progress.

**Prolific IDs are not saved.** The ID is used only to show the completion code. The session stores `source` (`prolific` or `direct`) and, if the `PID_SALT` secret is set, `pid_hash`: a salted hash that lets you find the session of a given Prolific submission (to reject or bonus it) by hashing its ID with the same salt, without the ID being in the data.

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
   | `STUDY_CONTACT` | Variable | optional, shown on the consent page, e.g. `Name, institution, email` |
   | `PID_SALT` | Secret | optional, any long random string; saves a hash of the Prolific ID instead of nothing |
   | `LIST_COUNT` / `LIST_SIZE` | Variable | optional, number of lists (default 5) and images per list (default 20, at most) |

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
With `PROLIFIC_COMPLETION_CODE` set, Prolific participants see the code at the end and a **Back to Prolific** button (no automatic redirect, so they have time to read their feedback). Without Prolific parameters (colleagues, lab pilot), the page just thanks them.

## Using a new version of the dataset

The stimuli are read once, when the Space starts. After you push new images or prompts to the dataset, restart the Space (Settings → Restart). Every saved session records the exact dataset commit (`dataset.revision`), so responses from different versions stay distinguishable. For the real study, pin `DATASET_REVISION` to a tag or commit so the stimuli cannot change during collection.

**Dataset columns used.** The dataset can be an imagefolder (`metadata.jsonl` with a `file_name` column next to the image files, the current layout) or parquet files with the images embedded. Required: `sample_id` (unique), `prompt` and the image. Optional: `role` (`control` = shown to everyone), `pair_ids` and `image_id` (used to build the lists). Every other column is copied as a label into the saved responses, server-side only.

## Before launching

- A free Space goes to sleep after a period without visits and takes a few seconds to wake up. Open the Space yourself just before you publish the study.
- Responses are written locally first and pushed every 2 minutes. A restart can lose at most the last 2 minutes; wait for the commit before restarting or redeploying during data collection.
- `S12` (attention check) reuses the `apple_red` image of `S18`/`S23`, and `S21` has the same text as `S12`. Under the list rules these three can never be shown, so pairs P-A1, P-C2, P-G1, P-G3, P-A′3 and P-E1 lose a member. Give the control its own image and text and they come back automatically.
- The images are still placeholder illustrations.

## Saved data (one file per session)

```json
{
  "session_id": "…", "list_id": 5, "study_version": "pilot-v1",
  "dataset": {"repo": "Enzo-py/partial-prompting-ds", "revision": "<commit sha>", "split": "train"},
  "participant": {"source": "prolific", "pid_hash": "3f9a…", "study_id": "…", "english": "fluent", "t2i": "sometimes"},
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

## Feedback to participants

After their answers are saved, participants see a short comparison with the other participants: how far their ratings were on average from the others' (in points), on how many images they were within 1 point of the others' average, and the image where they differed most (shown only for a gap of at least 1 point). It is requested separately (`POST /api/feedback`), so it can never delay or change the saved answers.

Safeguards: it is not announced beforehand, so it cannot push anyone to guess the "expected" answer; it is worded as a comparison, never as a score; controls are never used, so the attention check stays hidden; sessions that fail a quality filter are left out of the comparison; a stimulus counts only once at least 5 other participants rated it, and nothing is shown below 3 comparable stimuli (`FEEDBACK_MIN_OTHERS`, `FEEDBACK_MIN_ITEMS` in `app.py`). The first participants therefore see no feedback. The feedback is not saved.

## Dashboard

`https://enzo-py-partial-prompting-study.hf.space/admin` shows the study while it runs: sessions completed and in progress, progress per list, exclusions, mean ratings by category, each minimal pair side by side, a table of every stimulus (rating distribution, "should not be there" rate, response time; click a row to see the image with every click mark) and the recent sessions.

The page asks for `ADMIN_KEY` and stays closed until that secret is set. It reads the sessions from `RESPONSES_REPO` plus the ones not pushed yet, refreshed at most once a minute (the Refresh button forces it). Sessions that fail a quality filter count in the progress but not in the statistics, and the page shows which filter each one failed.

## Analysis

```bash
python analysis/collect_responses.py Enzo-py/partial-prompting-responses --out responses.parquet
```
Downloads every session, flattens to one row per trial, and adds one column per quality filter plus `excluded`.

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
