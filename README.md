# ProbeShift

**Measure how little must change before your classifier flips — one call on any predict function returns per-transform flip distances on benign transforms: the smallest brightness change, blur, shift, or rotation that changes the prediction, plus flip rates at fixed budgets and a fragility score.**

Aggregate robustness numbers hide the shape of a model's fragility. A model can hold 95% accuracy on a corruption suite and still flip on a 4% brightness change to *every* image it sees — a deployment fact no fixed-severity benchmark reports, because benchmarks fix the magnitude and count the damage. ProbeShift inverts that: it fixes the *transform* and measures the magnitude. Give it any callable that returns labels or class scores, and a batch of images; it sweeps fixed grids of six benign transforms — brightness, contrast, blur, noise, shift, rotation — and records, per example, the smallest magnitude that changes the prediction, in physical units: *this model flips under a +0.04 brightness change; that one survives brightness but flips at blur σ 2.3.*

ProbeShift is a measuring instrument, not an attack tool. It contains no gradients, no optimisation, and no target class: magnitudes are swept on fixed deterministic grids, never searched adversarially, and the output is a report, not an adversarial example.

## Install

ProbeShift is not on PyPI yet. Install the GitHub version:

```
pip install "git+https://github.com/habib-analyst/probeshift.git"
```

For development:

```
git clone https://github.com/habib-analyst/probeshift.git
cd probeshift
python -m pip install -e ".[dev]"
python -m pytest
```

Requires Python 3.10+. The only runtime dependency is NumPy. Your model is a callable — PyTorch, TensorFlow, scikit-learn, ONNX, a remote API behind a function, a hand-written rule: if it maps a batch of images to labels or scores, ProbeShift can probe it, and ProbeShift never imports your framework.

## Quickstart

```
import numpy as np

from probeshift import probe

def predict_fn(batch):            # (N, H, W[, C]) floats in [0, 1]
    ...                           # return labels (N,) or scores (N, C)

report = probe(predict_fn, images)

print(report.summary())
# ProbeShift report: 24 example(s), fragility score 0.188 ...
#   brightness flips 24/24 (100.0%), mean flip distance 0.03667 intensity
#   blur       flips 0/24 (0.0%), mean flip distance n/a sigma (px)
#   ...

card = report.cards[0]
print(card.label)                                   # original prediction
print(card.results["brightness"].flip_distance)     # e.g. 0.0375
print(card.results["brightness"].flipped_to)        # what it flipped to
print(card.most_fragile_transform)                  # e.g. "brightness"

report.to_json()   # full artifact: cards, summaries, budgets, config
```

`probe_image(predict_fn, image)` does the same for one `(H, W[, C])` image and returns its card directly.

## How it works

- **Grids, not search.** Each transform has a fixed magnitude grid in physical units (brightness Δ from 0.02 to 0.5; blur σ from 0.5 to 5 px; rotation 1° to 45°; shift 1 to 8 px; …), swept in ascending-distance order. The **flip distance** is the first grid magnitude whose prediction differs from the untouched input's. For scalar transforms the bracket around the flip is then refined by fixed-step bisection and the refined parameter is re-checked against the model — the card keeps both the refined distance and the raw `grid_distance`, so the measurement resolution is never hidden.
- **Distances are physical.** Brightness distance is added intensity; contrast distance is `|factor − 1|`; blur and noise distances are σ (pixels / intensity); shift distance is `max(|dx|, |dy|)` in pixels; rotation is degrees. A flip distance of 0.04 means something you can picture.
- **The noise field is fixed.** Noise magnitudes scale one fixed standard-normal field per example (seeded, recorded in the config), so "flips at σ 0.12" is a property of a fixed noise pattern growing, not of resampled luck — and two runs of a probe produce byte-identical reports.
- **Budgets and fragility.** Each transform summary reports flip rates at fixed budgets (default: 25% / 50% / 100% of the grid's range; overridable per transform), mean/median/min flip distances over the examples that flipped, and the report's **fragility score** is the mean flip rate at half of each grid's range — 0 for a model nothing flips, 1 for a model everything flips halfway through every grid.

## The report

Each `ExampleCard` carries the example's index and original label, and per transform a `FlipResult`: `flipped`, `flip_distance`, `grid_distance`, `flip_param` (the exact parameter that flipped it), and `flipped_to`. Each `TransformSummary` carries the counts, rates, and distance statistics. The report also records its full `config` — seed, grids, refinement settings, input shape — so a card can be re-derived or challenged, and `ProbeReport.from_dict(report.to_dict())` round-trips it: probe two model versions, diff the JSON, and see exactly which transform got more fragile.

## What ProbeShift is not

- **Not an adversarial attack tool.** CleverHans, the Adversarial Robustness Toolbox (ART), and Foolbox *generate* adversarial examples — gradient- or optimisation-driven perturbations, usually Lp-bounded, often targeted, produced through framework model wrappers. Foolbox even ships blur/contrast/noise "attacks" that line-search a minimal flipping magnitude. ProbeShift shares the *question* (how much change flips this?) and none of the machinery or intent: fixed grids of benign, physically meaningful transforms, no gradients, no optimisation, no target class, any black-box callable, and a measurement report as the product. If you need adversarial examples or a security evaluation, use those toolboxes — they are the right tools, and ProbeShift says so rather than competing with them.
- **Not a benchmark suite.** RobustBench and the ImageNet-C / CIFAR-C corruption suites fix datasets, models, and severity levels, and report aggregate accuracy (or mCE) — the right design for ranking published defences, and no use for *"how fragile is my model, on my images, in units I understand?"* ProbeShift has no leaderboard, no model zoo, and no fixed dataset; it measures your model on your inputs.
- **Not a counterfactual explainer.** DiCE and Alibi answer *"what feature changes would flip this one decision?"* — per-example, optimisation-based feature edits for explanation and recourse, primarily tabular. ProbeShift's transforms are whole-image physical changes swept over a population to produce aggregate fragility statistics; it explains no individual decision and proposes no recourse.
- **Not a robustness guarantee.** A large flip distance on these six transforms says nothing about other transforms, other distributions, or Lp adversaries. ProbeShift measures what it measures, and its reports say exactly what that was.

## Measured demo (synthetic, reproducible)

`examples/flip_distance_demo.py` builds 24 synthetic 16×16 blob images (seed 0) and two hand-built models that both score **100% clean accuracy** — a mean-intensity threshold and a peak-intensity threshold — then probes both:

| Transform | Model A (mean) | Model B (peak) |
|---|---|---|
| brightness | 24/24 flip, mean distance **0.037** | 24/24 flip, mean distance 0.166 |
| contrast | 0/24 | 15/24, mean 0.576 |
| blur | 0/24 | 12/24, mean σ **2.35** |
| noise | 11/24, mean σ 0.312 | 12/24, mean σ 0.093 |
| shift | 12/24, mean 6.2 px | 2/24, mean 8 px |
| rotation | 0/24 | 0/24 |
| **fragility score** | **0.188** | **0.410** |

Same accuracy, opposite instruments readings: Model A is exactly invariant to the mean-preserving transforms (blur, contrast) and flips if you breathe on the brightness; Model B tolerates 4.5× larger brightness changes but its peak erodes under blur and noise. One honest wrinkle the demo keeps: Model A's shift flips are real — at ≥ 6 px the blob is cropped out of frame and the mean genuinely falls — so "shift preserves the mean" holds only while the content stays in frame. The script prints whatever it measures and exits non-zero if its by-construction expectations do not hold. It is a mechanics demo, not a benchmark.

## License

MIT. Issues and pull requests are welcome on GitHub.
