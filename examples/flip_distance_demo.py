"""Mechanics demo: two models, same clean accuracy, opposite fragility.

Synthetic 16x16 images (seed 0) contain one Gaussian blob on a 0.2
background.  The classes differ only in blob amplitude:

* bright class: amplitude 0.5  -> peak 0.70, image mean ~0.310
* dark class:   amplitude 0.16 -> peak 0.36, image mean ~0.235

Two hand-built "models" separate them perfectly:

* **Model A** thresholds the image *mean* at 0.2725 (margin ~0.0375).
  It should flip under a brightness change of about 0.04, and be exactly
  invariant to blur and contrast, which preserve the mean.  Shift is
  *not* mean-preserving at large magnitudes: shifting by 6+ px crops
  blob mass out of frame (edge-replicated background flows in), so the
  bright class flips at large shifts only — an expectation corrected
  after the first measured run, which showed exactly that.
* **Model B** thresholds the image *maximum* at 0.51 (peak margin 0.19
  up / 0.15 down).  It should shrug off small brightness changes but flip
  under blur, which erodes the peak (at blur sigma 2.5 the bright peak
  falls to ~0.495).

ProbeShift measures both.  The script prints whatever it measures and
exits non-zero if the by-construction expectations above do not hold.
It is a mechanics demo on synthetic data, not a benchmark.
"""

import json
from pathlib import Path

import numpy as np

from probeshift import probe

SIZE = 16
BACKGROUND = 0.2
BLOB_SIGMA = 3.0
AMPLITUDES = {0: 0.16, 1: 0.5}  # class -> blob amplitude


def make_dataset(n_per_class: int = 12, seed: int = 0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    images, labels = [], []
    for label in (0, 1):
        for _ in range(n_per_class):
            cy, cx = rng.uniform(5, 10, size=2)
            blob = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2)
                          / (2 * BLOB_SIGMA ** 2))
            image = BACKGROUND + AMPLITUDES[label] * blob
            image = image + rng.normal(0.0, 0.01, size=image.shape)
            images.append(np.clip(image, 0.0, 1.0))
            labels.append(label)
    return np.stack(images), np.array(labels)


def model_a(batch):
    """Threshold the mean intensity."""
    return (batch.mean(axis=(1, 2)) > 0.2725).astype(int)


def model_b(batch):
    """Threshold the peak intensity."""
    return (batch.max(axis=(1, 2)) > 0.51).astype(int)


def main() -> int:
    images, labels = make_dataset()
    failures = []

    reports = {}
    for name, model in (("A (mean threshold)", model_a),
                        ("B (peak threshold)", model_b)):
        clean = model(images)
        accuracy = float(np.mean(clean == labels))
        report = probe(model, images)
        reports[name] = report
        print(f"\nModel {name}: clean accuracy {accuracy:.3f}")
        print(report.summary())

    a = reports["A (mean threshold)"]
    b = reports["B (peak threshold)"]

    a_bright = a.summary_for("brightness")
    if not (a_bright.flip_rate == 1.0
            and a_bright.mean_flip_distance is not None
            and a_bright.mean_flip_distance < 0.08):
        failures.append(
            f"model A brightness: expected flip rate 1.0 at mean distance "
            f"< 0.08, measured rate {a_bright.flip_rate} at "
            f"{a_bright.mean_flip_distance}")
    for transform in ("blur", "contrast"):
        summary = a.summary_for(transform)
        if summary.flip_rate != 0.0:
            failures.append(
                f"model A {transform}: expected no flips (mean-preserving), "
                f"measured flip rate {summary.flip_rate}")
    a_shift = a.summary_for("shift")
    if not (a_shift.min_flip_distance is not None
            and a_shift.min_flip_distance >= 6
            and a_shift.flip_rate <= 0.5):
        failures.append(
            f"model A shift: expected flips only at >= 6 px (blob cropped "
            f"out of frame) for at most the bright half, measured rate "
            f"{a_shift.flip_rate}, min distance {a_shift.min_flip_distance}")
    b_blur = b.summary_for("blur")
    if b_blur.flip_rate < 0.4:
        failures.append(
            f"model B blur: expected flip rate >= 0.4 (bright peaks erode), "
            f"measured {b_blur.flip_rate}")
    b_bright = b.summary_for("brightness")
    if not (b_bright.mean_flip_distance is not None
            and a_bright.mean_flip_distance is not None
            and b_bright.mean_flip_distance > a_bright.mean_flip_distance):
        failures.append(
            f"model B should tolerate larger brightness changes than A: "
            f"B mean {b_bright.mean_flip_distance} vs "
            f"A mean {a_bright.mean_flip_distance}")

    out = {
        name: {
            "fragility_score": report.fragility_score,
            "summaries": [s.to_dict() for s in report.summaries],
        }
        for name, report in reports.items()
    }
    out_path = Path(__file__).parent / "output" / "demo_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))

    if failures:
        print("\nDEMO EXPECTATIONS FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\nAll by-construction expectations held.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
