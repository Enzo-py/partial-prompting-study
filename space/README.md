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

Collection interface for the partial-prompt human study. Participants see a text and an image, give a 1–5 match rating, and say whether any part of the image is a problem or does not match the text (clicking on each such area). Built with jsPsych; a small FastAPI backend loads the stimuli from the dataset [`Enzo-py/partial-prompting-ds`](https://huggingface.co/datasets/Enzo-py/partial-prompting-ds), assigns counterbalanced lists and saves responses to a **private dataset repo**.

## How it works

```
startup: download DATASET_REPO (imagefolder or parquet) → write web copies of the images to stimuli_cache/ (WebP, at most 1024 px), build the lists

participant ──► Space (public URL, no login)
                 ├─ POST /api/start   → assigns the least-filled list, returns its stimuli (no labels)
                 ├─ jsPsych trials     → random order, response times per trial
                 └─ POST /api/submit  → writes data/listNN/<session>.json
                                          └─ CommitScheduler pushes data/ every 2 min
                                             → <RESPONSES_REPO>/responses/listNN/<session>.json
```

**Counterbalancing.** Only stimuli with `human_study` true are used. The attention checks are the controls whose image plainly contradicts the text (`role` = `control` with a `control_type`); other controls (such as the "every detail true" condition of F pairs) are study stimuli like the rest. Each list gets `CHECKS_PER_LIST` attention checks (default 3, rotating and balanced across lists) and up to `LIST_SIZE` study images (default 7, for a session of about 8 minutes). Two stimuli never go in the same list when they belong to the same minimal pair (`pair_ids`), reuse the same image (`image_id`) or share the same text, so every comparison is between-subjects and nobody sees a text twice. By default there are just enough lists to show every study stimulus at least once, and at least as many as the largest group of stimuli sharing a text, an image or a pair; `LIST_COUNT` forces a number. The search is deterministic: the same dataset always gives the same lists. They are written to `lists.json` (with how many lists each stimulus is in) and `/api/status` shows the sizes. With the current dataset: 24 lists of 10 images (3 checks + 7). Recruit participants in multiples of the number of lists.

**Session.** Consent, a two-question profile, instructions, one practice image ("A butterfly on a flower", `static/practice/`, not saved; a practice ranking with three images also opens the ranking part), then the list in random order. Answering *Yes* to question 2 requires marking at least one area. Question 2 is saved as `problem` (sessions before this change saved the older "should not be there" question as `unwanted`). Leaving or reloading the page asks for confirmation, since it loses the answers. The page shows at once: the images of step 1 load while the participant reads the consent and instructions, those of step 2 on the ranking instructions, and a loading screen appears only if an image is not ready when its turn comes. For each image the session stores its `position` (presentation order) and `rt` (time on the image, from display to *Next*).

**Rankings.** After the ratings, each participant ranks `RANKINGS_PER_LIST` groups (default 5). The groups come from the dataset, `groups/groups.jsonl`: one text and its images (`members`, any number from 2; a group with a single image is skipped), so editing that file and restarting the Space changes the rankings. Images that are only in a group (no `sample_id`) are shown only there. The question is preference (which image they would rather get from a generator), not match. They tap the images from favourite to least favourite, the last one included, shown in random order. Groups are spread evenly over the lists, never two with the same text in one list (15 groups today, each in 8 of the 24 lists). They come last so that seeing several images of one text cannot affect the between-subjects ratings. Each ranking saves `set_id` (the `group_id`), `shown` (screen order) and `order` (favourite first) as `image_id`s, `order_sample_ids` (the matching stimulus, null for group-only images), `rt` and `resets`.

**Progress.** Every page shows the same two-step bar, *1 · Rate images* then *2 · Rank images*, with the image or ranking number under it. The page adapts to phones: images fit the screen height, rankings use 2 columns on a phone and images up to 400 px wide on a large screen (3 per row, 2 × 2 for 4 images).

**Quality checks.** Each attention check rated as matching counts as one failed check: 3 or more when the named object is absent, 4 or more for a wrong colour, count, relation or binding, since "Partly" (3) is an honest answer there (`CHECK_FAIL_AT`). Checks are judged with the labels saved in the session, so later dataset changes do not affect past sessions. 25% or more of the images answered in under 1 s also counts as a failed check, and so do 99% or more identical ratings. A session is excluded from the statistics and the feedback once it fails 3 checks; fewer are a warning in the dashboard (`FAST_TRIAL_MS`, `FAST_SHARE`, `SAME_ANSWER_SHARE`, `EXCLUDE_AT` in `app.py`; `analysis/collect_responses.py` applies the same rules). Excluded sessions still count towards list progress.

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
   | `LIST_COUNT` / `LIST_SIZE` | Variable | optional, number of lists (default: enough to show every stimulus) and study images per list (default 7, at most) |
   | `CHECKS_PER_LIST` | Variable | optional, attention checks per list (default 3) |
   | `RANKINGS_PER_LIST` | Variable | optional, ranking questions after the ratings (default 5, 0 = none) |

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
- `S12` (attention check, "a banana") reuses the `apple_red` image of `S18`/`S23` and shares its text with `S21`, so those never appear in the same list as `S12`. That is handled by the lists, but a check with its own image and text would be cleaner.
- The images are still placeholder illustrations.

## Saved data (one file per session)

```json
{
  "session_id": "…", "list_id": 5, "study_version": "pilot-v1",
  "dataset": {"repo": "Enzo-py/partial-prompting-ds", "revision": "<commit sha>", "split": "train"},
  "participant": {"source": "prolific", "pid_hash": "3f9a…", "study_id": "…", "english": "fluent", "t2i": "sometimes"},
  "trials": [{
    "position": 1, "sample_id": "S09", "prompt": "an apple on a table",
    "rating": 4, "problem": "yes", "marks": [{"x": 0.147, "y": 0.180}],
    "rt": 5310, "rt_first_rating": 1820, "rating_changes": 1,
    "case_id": "C7", "category": "C", "subcategory": "Semantic violation", "role": "test", "pair_ids": [], "...": "…"
  }],
  "rankings": [{"set_id": "GR-SKYSCRAPER", "prompt": "a skyscraper", "position": 1,
                "shown": ["g13_skyscraper_city", "g32_park_far_skyscraper", "g14_skyscraper_desert"],
                "order": ["g32_park_far_skyscraper", "g13_skyscraper_city", "g14_skyscraper_desert"],
                "order_sample_ids": ["S201", "S106", "S107"], "rt": 9400, "resets": 0}],
  "total_time_ms": 240000, "client": {"user_agent": "…", "viewport": "…"}, "received_at": "…"
}
```
Click marks are normalized to the image (0–1 on each axis).

## Feedback to participants

After their answers are saved, participants see a short comparison with the other participants: how far their ratings were on average from the others' (in points), on how many images they were within 1 point of the others' average, and the image where they differed most (shown only for a gap of at least 1 point). It is requested separately (`POST /api/feedback`), so it can never delay or change the saved answers.

Safeguards: it is not announced beforehand, so it cannot push anyone to guess the "expected" answer; it is worded as a comparison, never as a score; controls are never used, so the attention check stays hidden; excluded sessions are left out of the comparison; a stimulus counts only once at least 5 other participants rated it, and nothing is shown below 3 comparable stimuli (`FEEDBACK_MIN_OTHERS`, `FEEDBACK_MIN_ITEMS` in `app.py`). The first participants therefore see no feedback. The feedback is not saved.

## Dashboard

`https://enzo-py-partial-prompting-study.hf.space/admin` shows the study while it runs: sessions completed and in progress, progress per list, exclusions, mean ratings by category, each minimal pair side by side, a table of every stimulus (rating distribution, "should not be there" rate, response time; click a row to see the image with every click mark) and the recent sessions.

The page asks for `ADMIN_KEY` and stays closed until that secret is set. It reads the sessions from `RESPONSES_REPO` plus the ones not pushed yet, refreshed at most once a minute (the Refresh button forces it). Excluded sessions count in the progress but not in the statistics; the page shows each session's failed checks, as a warning below 3.

## Analysis

```bash
python analysis/collect_responses.py Enzo-py/partial-prompting-responses --out responses.parquet
```
Downloads every session, flattens to one row per trial, and adds one column per quality check plus `n_failed_checks` and `excluded`. Rankings go to `rankings.parquet`, one row per ranked image.

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
