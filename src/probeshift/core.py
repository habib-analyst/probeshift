"""The probe engine: deterministic grid sweeps for first-flip magnitudes.

For each example and each transform, the transform's fixed magnitude grid
is swept in ascending-distance order and the model's prediction is recorded
at every step.  The *flip distance* is the smallest swept magnitude whose
prediction differs from the prediction on the untouched input.  For
scalar-parameter transforms the grid bracket around the flip can then be
refined by a fixed number of bisection steps (the grid value is always
kept alongside, so nothing about the measurement resolution is hidden).

Nothing here optimises anything: no gradients, no target class, no
adaptive search.  The same inputs and config always produce the same
report, byte for byte.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Union

import numpy as np

from .cards import ExampleCard, FlipResult, ProbeReport, TransformSummary
from .transforms import Transform, default_transforms, get_transform

PredictFn = Callable[[np.ndarray], object]

_BUDGET_FRACTIONS = (0.25, 0.5, 1.0)


def _validate_images(images: object, *, single: bool) -> np.ndarray:
    arr = np.asarray(images)
    if arr.ndim not in (2, 3, 4):
        raise ValueError(
            f"images must be a single image (H, W[, C]) or a batch "
            f"(N, H, W[, C]); got an array with ndim={arr.ndim}"
        )
    if single and arr.ndim == 4:
        raise ValueError("probe_image expects one image (H, W[, C]), "
                         "not a batch")
    if not single and arr.ndim == 2:
        raise ValueError("probe expects a batch (N, H, W[, C]); use "
                         "probe_image for a single (H, W[, C]) image")
    if arr.size == 0 or (not single and arr.shape[0] == 0):
        raise ValueError("images must contain at least one example")
    try:
        batch = np.array(arr, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"images must be numeric: {exc}") from exc
    if not np.all(np.isfinite(batch)):
        raise ValueError("images must be finite (no NaN or inf values)")
    if single:
        batch = batch[np.newaxis, ...]
    return batch


def _predict_labels(predict_fn: PredictFn, batch: np.ndarray) -> list:
    out = np.asarray(predict_fn(batch))
    n = batch.shape[0]
    if out.ndim == 1:
        if out.shape[0] != n:
            raise ValueError(
                f"predict_fn returned {out.shape[0]} labels for a batch "
                f"of {n} images"
            )
        return out.tolist()
    if out.ndim == 2 and out.shape[0] == n and out.shape[1] == 1:
        return out[:, 0].tolist()
    if out.ndim == 2 and out.shape[0] == n and out.shape[1] >= 2:
        return np.argmax(out, axis=1).tolist()
    raise ValueError(
        f"predict_fn must return labels of shape (N,) or class scores of "
        f"shape (N, C) for a batch of {n} images; got shape {out.shape}"
    )


def _resolve_transforms(
    transforms: Optional[Sequence[Union[str, Transform]]],
) -> List[Transform]:
    if transforms is None:
        return list(default_transforms())
    resolved: List[Transform] = []
    for item in transforms:
        if isinstance(item, Transform):
            resolved.append(item)
        elif isinstance(item, str):
            resolved.append(get_transform(item))
        else:
            raise ValueError(
                f"transforms entries must be names or Transform objects; "
                f"got {type(item).__name__}"
            )
    if not resolved:
        raise ValueError("at least one transform is required")
    names = [t.name for t in resolved]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate transforms: {names}")
    return resolved


def _resolve_budgets(
    budgets: Optional[Dict[str, Sequence[float]]],
    transform: Transform,
) -> List[float]:
    if budgets is None:
        return [round(f * transform.max_distance, 10)
                for f in _BUDGET_FRACTIONS]
    if transform.name not in budgets:
        return [round(f * transform.max_distance, 10)
                for f in _BUDGET_FRACTIONS]
    values = [float(b) for b in budgets[transform.name]]
    if any(b <= 0 for b in values):
        raise ValueError(
            f"budgets for {transform.name!r} must be positive distances"
        )
    return sorted(set(values))


def _refine_flip(
    predict_fn: PredictFn,
    image: np.ndarray,
    base_label: object,
    transform: Transform,
    flip_param: float,
    bracket_low: float,
    field: Optional[np.ndarray],
    iterations: int,
) -> float:
    """Bisect between a non-flipping parameter and the flipping one.

    Assumes the flip is locally monotone in the parameter on this side of
    the identity (documented in the README); the grid distance remains on
    the card regardless of what refinement returns.
    """
    low, high = bracket_low, flip_param
    for _ in range(iterations):
        mid = 0.5 * (low + high)
        transformed = transform.apply(image, mid, field)
        label = _predict_labels(predict_fn, transformed)[0]
        if label != base_label:
            high = mid
        else:
            low = mid
    return float(high)


def probe(
    predict_fn: PredictFn,
    images: object,
    *,
    transforms: Optional[Sequence[Union[str, Transform]]] = None,
    seed: int = 0,
    refine: bool = True,
    refine_iterations: int = 12,
    budgets: Optional[Dict[str, Sequence[float]]] = None,
) -> ProbeReport:
    """Probe a classifier's flip distances on benign transforms.

    Parameters
    ----------
    predict_fn:
        Any callable taking a batch of float images ``(N, H, W[, C])``
        with values in ``[0, 1]`` and returning either labels of shape
        ``(N,)`` or class scores/probabilities of shape ``(N, C)`` (the
        argmax is taken).  No framework object is required or imported.
    images:
        A batch ``(N, H, W)`` or ``(N, H, W, C)``.  Converted to float64;
        never mutated.
    transforms:
        Transform names (subset of the six built-ins) or ``Transform``
        objects; default is all six built-ins.
    seed:
        Seeds the fixed standard-normal field used by the noise transform
        (one field per transform position, drawn once per probe call, so
        noise magnitudes scale a single fixed field per example).
    refine:
        If true, refine each scalar transform's flip bracket by
        bisection (``refine_iterations`` steps).  The grid distance is
        always recorded next to the refined one.
    budgets:
        Optional per-transform flip-rate budgets, as absolute distances:
        ``{"brightness": [0.05, 0.1]}``.  Transforms not named here use
        budgets at 25% / 50% / 100% of their grid range.
    """
    if not callable(predict_fn):
        raise TypeError("predict_fn must be callable")
    if refine_iterations < 1:
        raise ValueError("refine_iterations must be >= 1")
    batch = _validate_images(images, single=False)
    resolved = _resolve_transforms(transforms)
    n = batch.shape[0]
    base_labels = _predict_labels(predict_fn, batch)

    cards = [ExampleCard(index=i, label=base_labels[i]) for i in range(n)]
    summaries: List[TransformSummary] = []
    mid_budget_rates: List[float] = []

    for position, transform in enumerate(resolved):
        field: Optional[np.ndarray] = None
        if transform.name == "noise":
            rng = np.random.default_rng(
                np.random.SeedSequence([int(seed), position])
            )
            field = rng.standard_normal(batch.shape)

        flipped = np.zeros(n, dtype=bool)
        flip_param: List[object] = [None] * n
        flipped_to: List[object] = [None] * n
        # Most recent non-flipping parameter on each side of the identity,
        # per example: the refinement bracket.  (+1 side, -1 side.)
        last_ok: Dict[int, List[Optional[float]]] = {
            i: [None, None] for i in range(n)
        }

        for param in transform.params:
            if bool(flipped.all()):
                break
            transformed = transform.apply(batch, param, field)
            labels = _predict_labels(predict_fn, transformed)
            for i in range(n):
                if flipped[i]:
                    continue
                if labels[i] != base_labels[i]:
                    flipped[i] = True
                    flip_param[i] = param
                    flipped_to[i] = labels[i]
                elif transform.kind == "scalar":
                    side = 0 if float(param) > transform.identity else 1
                    last_ok[i][side] = float(param)  # type: ignore[arg-type]

        distances: List[Optional[float]] = [None] * n
        for i in range(n):
            if not flipped[i]:
                continue
            param = flip_param[i]
            grid_distance = transform.distance(param)  # type: ignore[arg-type]
            distance = grid_distance
            if refine and transform.kind == "scalar":
                side = 0 if float(param) > transform.identity else 1  # type: ignore[arg-type]
                low = last_ok[i][side]
                bracket_low = transform.identity if low is None else low
                example_field = None if field is None else field[i:i + 1]
                refined = _refine_flip(
                    predict_fn, batch[i:i + 1], base_labels[i], transform,
                    float(param), bracket_low, example_field,  # type: ignore[arg-type]
                    refine_iterations,
                )
                distance = abs(refined - transform.identity)
                flip_param[i] = refined
            distances[i] = distance
            cards[i].results[transform.name] = FlipResult(
                transform=transform.name,
                flipped=True,
                flip_distance=float(distance),
                grid_distance=float(grid_distance),
                flip_param=(list(flip_param[i])  # type: ignore[arg-type]
                            if isinstance(flip_param[i], tuple)
                            else float(flip_param[i])),  # type: ignore[arg-type]
                flipped_to=flipped_to[i],
            )
        for i in range(n):
            if not flipped[i]:
                cards[i].results[transform.name] = FlipResult(
                    transform=transform.name, flipped=False,
                )

        flipped_distances = [d for d in distances if d is not None]
        budget_values = _resolve_budgets(budgets, transform)
        rate_entries = []
        for budget in budget_values:
            rate = float(np.mean([
                1.0 if (d is not None and d <= budget) else 0.0
                for d in distances
            ]))
            rate_entries.append({"budget": budget, "flip_rate": rate})
        summaries.append(TransformSummary(
            name=transform.name,
            unit=transform.unit,
            n_examples=n,
            n_flipped=int(flipped.sum()),
            flip_rate=float(flipped.mean()),
            mean_flip_distance=(float(np.mean(flipped_distances))
                                if flipped_distances else None),
            median_flip_distance=(float(np.median(flipped_distances))
                                  if flipped_distances else None),
            min_flip_distance=(float(np.min(flipped_distances))
                               if flipped_distances else None),
            flip_rate_at_budgets=rate_entries,
        ))
        mid_budget = round(0.5 * transform.max_distance, 10)
        rate = summaries[-1].rate_at(mid_budget)
        if rate is None:  # custom budgets may not include the mid budget
            rate = float(np.mean([
                1.0 if (d is not None and d <= mid_budget) else 0.0
                for d in distances
            ]))
        mid_budget_rates.append(rate)

    fragility = float(np.mean(mid_budget_rates)) if mid_budget_rates else 0.0
    config = {
        "seed": int(seed),
        "refine": bool(refine),
        "refine_iterations": int(refine_iterations),
        "n_examples": n,
        "input_shape": list(batch.shape),
        "transforms": {
            t.name: {"unit": t.unit, "grid_distances": list(t.distances())}
            for t in resolved
        },
        "budget_fractions": list(_BUDGET_FRACTIONS) if budgets is None
        else "custom",
        "fragility_definition": (
            "mean over transforms of the flip rate at half of each "
            "transform's grid range"
        ),
    }
    return ProbeReport(cards=cards, summaries=summaries,
                       fragility_score=fragility, config=config)


def probe_image(
    predict_fn: PredictFn,
    image: object,
    **kwargs,
) -> ExampleCard:
    """Probe one image ``(H, W[, C])`` and return its :class:`ExampleCard`.

    Accepts the same keyword arguments as :func:`probe`.  Note that the
    noise transform's fixed field is drawn for this single image, so a
    noise flip distance from ``probe_image`` need not match the one the
    same image gets inside a batch probe (where the field depends on the
    batch).  All other transforms are per-image exact.
    """
    batch = _validate_images(image, single=True)
    report = probe(predict_fn, batch, **kwargs)
    return report.cards[0]
