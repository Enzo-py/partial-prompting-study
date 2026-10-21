---
pretty_name: Partial Prompt Human Study
language:
  - en
license: cc-by-4.0
task_categories:
  - text-to-image
  - image-classification
tags:
  - text-to-image
  - evaluation
  - human-annotation
  - underspecification
  - pragmatics
  - image-text-alignment
size_categories:
  - n<1K
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/train-*.parquet
---

# Partial Prompt Human Study

**How do humans judge content that a text-to-image prompt does not mention?**

Prompts given to diffusion models are almost always partial: *"a chair on the grass"* says nothing about the sky, and *"an apple"* says nothing about whether a pear may sit next to it. Every generated image has to fill these gaps. Existing T2I alignment benchmarks (TIFA, DSG, Gecko, VQAScore, RichHF) derive their questions from the prompt alone, so unmentioned content is invisible to them by construction, and human annotation guidelines settle the question by convention instead of measuring it.

This dataset contains the stimuli for a human study that measures it. Each stimulus is an `(image, prompt)` pair labeled with a case from a taxonomy of unmentioned content, and most stimuli belong to a **minimal pair**: the same prompt with a test image and a reference image, or the same image under two prompts. The quantity of interest is how human ratings move between the two members of a pair.

> **Status: pilot.** The current images are placeholder SVG illustrations rendered to PNG, used to validate the protocol. They will be replaced by real generations and inpainted minimal pairs. Annotations are not included yet.

## Taxonomy

| Category | Name | Definition | Cases |
|---|---|---|---|
| A | Forced completion | The prompt leaves a dimension open and the image has to fill it with a typical value | A1–A3 |
| A′ | Atypical completion | The same forced dimension, filled with a valid but atypical value | A′1–A′3 |
| B | Implied content | Content the prompt never names but entails or excludes | B1–B3 |
| C | Optional addition | Content that nothing in the prompt requires | C1–C9 |
| D | Salience | Nothing is added or removed; what changes is what the image is about | D1–D2 |
| E | Control | Attention check | E1 |
| G | Generator-specific | Extra content typical of diffusion outputs | G1–G4 |

The full table of 25 cases (definition, example, minimal pair, theoretical anchor, predicted penalty) is in [`taxonomy/taxonomy.md`](taxonomy/taxonomy.md) and [`taxonomy/cases.json`](taxonomy/cases.json). Seven cross-cutting manipulations (exhaustivity, prompt specificity, prompt register, prompt type, animacy, dose, reference zero) are in [`taxonomy/axes.json`](taxonomy/axes.json).

Core hypothesis: the penalty grows along three axes. Forced completions are never penalized, optional typical additions are penalized little, and competing, extra-instance or incongruous additions are penalized strongly. A more detailed prompt is read as more exhaustive, which raises every penalty.

## Contents

```
data/                  train-00000-of-00001.parquet: 23 stimuli, images embedded (what the Hub loads)
raw_images/            the same stimuli as PNG (800×520) + metadata.jsonl, for local use
svg/                   vector sources of the placeholder images
taxonomy/              cases.json, axes.json, taxonomy.md, taxonomy_reference.html
annotation/            pilot_interface.html, response_schema.json, guidelines.md
```

Code (annotation Space, scripts, analysis) lives on GitHub in `partial-prompting-study`. The annotation interface runs as the Space [`Enzo-py/partial-prompting-study`](https://huggingface.co/spaces/Enzo-py/partial-prompting-study).

## Fields

| Field | Type | Description |
|---|---|---|
| `image` | Image | The stimulus (PNG, 800×520). In `raw_images/metadata.jsonl` this column is `file_name` |
| `sample_id` | str | `S01`–`S23`; `S01`–`S12` are the pilot samples |
| `prompt` | str | Text shown to the annotator |
| `case_id` | str | Taxonomy case, e.g. `C2` |
| `category` / `category_name` | str | `A`, `A′`, `B`, `C`, `D`, `E`, `G` and its name |
| `subcategory` | str | Case name |
| `theme` | str | Semantic domain (Food, Outdoor · furniture, Animals…) |
| `role` | str | `test`, `minimal_pair` (reference) or `control` |
| `pair_ids` | list[str] | Minimal pairs the stimulus belongs to; `P-*` pairs are defined by a case, `X-*` by a cross-cutting axis |
| `pair_axes` | list[str] | For each pair, what varies: `image` or `prompt` |
| `pair_roles` | list[str] | For each pair, this stimulus's role in it: `test`, `reference` or `control` (a stimulus can be the test in one pair and the reference in another) |
| `unmentioned_content` | str | What the image contains that the prompt does not state |
| `minimal_pair_description` | str | The reference condition defined by the taxonomy |
| `theoretical_anchor` | str | Linguistic or perceptual account the case tests |
| `expected_penalty_level` | float \| null | Predicted rating drop, 0 (none) to 3 (high); `null` = graded or open |
| `expected_penalty` | str | Prediction in words |
| `in_pilot` | bool | Whether the stimulus is in the 12-sample pilot |
| `image_id` | str | Shared by stimuli that reuse the same image under different prompts |
| `image_source` | str | `placeholder_svg_illustration` for now |

## Annotation protocol

Annotators first give a short profile (English level, how often they use image generators). For each stimulus, shown in random order, they answer:

1. **Match rating**, 1–5: *How well does the image match the text?*
2. **Unwanted content**, yes/no: *Is there anything in the image that should not be there, given the text?* If yes, they click on each unwanted element (coordinates in an 800×520 space, 400×260 in the pilot interface).
3. **Confidence**: not sure / fairly sure / very sure.
4. Optional free-text comment.

See [`annotation/guidelines.md`](annotation/guidelines.md) and the response format in [`annotation/response_schema.json`](annotation/response_schema.json). The pilot interface is a single self-contained HTML file.

## Usage

```python
from datasets import load_dataset
ds = load_dataset("Enzo-py/partial-prompting-ds", split="train")
pairs = ds.filter(lambda x: "P-C2" in x["pair_ids"])
```

## Related work

- Unmentioned content is ignored by question-based alignment metrics: TIFA, DSG, Gecko (Wiles et al., 2024), RichHF-18K (Liang et al., 2024).
- Hallucination in T2I: *Evaluating Hallucination in Text-to-Image Diffusion* (2024); Kasaei et al., *Hallucination as an Upper Bound* (2025).
- Underspecified prompts and diversity: GRADE (Rassin et al., 2024).
- Content implied by reasoning: Commonsense-T2I, PhyBench. Explicit negation: NegBench, Object-State-Bench.
- Lexical ambiguity: TAB (Mehrabi et al., 2023); homonym duplication (Rassin et al., 2022).
- Pragmatics: Grice's maxim of quantity; prototype theory (Rosch); scene grammar (Võ); order of mention and visual salience (Clarke et al., 2015).

## Citation

```bibtex
@misc{partialprompt2026,
  title  = {Partial Prompt Human Study},
  author = {TODO},
  year   = {2026},
  note   = {Pilot release}
}
```
