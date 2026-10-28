# Changelog

## 0.1.0 — 2026-10-28

First release.

- `probe`: deterministic flip-distance measurement over six benign
  transforms (brightness, contrast, blur, noise, shift, rotation) with
  fixed physical-unit magnitude grids. Accepts any callable returning
  labels `(N,)` or class scores `(N, C)`; no framework object, no
  gradients, no optimisation.
- Per-example `ExampleCard`s: original label plus, per transform, the
  flip distance, the grid distance, the flipping parameter, and the
  label flipped to. Scalar-transform flips are refined by bisection
  (grid value always recorded alongside).
- Per-transform `TransformSummary`: flip rate, flip rates at fixed
  budgets (25% / 50% / 100% of each grid's range by default), and
  mean / median / min flip distances over flipped examples.
- `ProbeReport`: cards + summaries + one fragility score (mean flip
  rate at half of each grid's range), the full probe config, text
  summaries, and JSON serialisation (`to_dict` / `from_dict` /
  `to_json`) for diffing across model versions.
- `probe_image`: single-image convenience wrapper returning one card.
- Synthetic two-model example: a mean-threshold model and a
  peak-threshold model with identical clean accuracy and opposite,
  measured fragility profiles.
