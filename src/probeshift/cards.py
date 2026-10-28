"""Report data structures: per-example flip cards and per-transform summaries.

A :class:`ProbeReport` is the artifact ProbeShift exists to produce: for
every probed example, the smallest benign-transform magnitude that flipped
the model's prediction (per transform), and for every transform, the
aggregate flip behaviour (flip rates at fixed budgets, flip-distance
statistics, and one overall fragility score).  Reports serialise to plain
JSON-safe dicts so they can be diffed across model versions, gated in CI,
or attached to a ticket.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

Label = Union[int, float, str]
ParamValue = Union[float, List[int]]


@dataclass
class FlipResult:
    """The outcome of probing one example with one transform."""

    transform: str
    flipped: bool
    flip_distance: Optional[float] = None     # refined (or grid) distance
    grid_distance: Optional[float] = None     # distance on the swept grid
    flip_param: Optional[ParamValue] = None   # parameter that first flipped
    flipped_to: Optional[Label] = None        # predicted label after the flip

    def to_dict(self) -> dict:
        return {
            "transform": self.transform,
            "flipped": self.flipped,
            "flip_distance": self.flip_distance,
            "grid_distance": self.grid_distance,
            "flip_param": self.flip_param,
            "flipped_to": self.flipped_to,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FlipResult":
        return cls(
            transform=data["transform"],
            flipped=bool(data["flipped"]),
            flip_distance=data["flip_distance"],
            grid_distance=data["grid_distance"],
            flip_param=data["flip_param"],
            flipped_to=data["flipped_to"],
        )


@dataclass
class ExampleCard:
    """One probed example: its original label and every transform's result."""

    index: int
    label: Label
    results: Dict[str, FlipResult] = field(default_factory=dict)

    @property
    def flipped_any(self) -> bool:
        return any(r.flipped for r in self.results.values())

    @property
    def min_flip_distance(self) -> Optional[float]:
        distances = [r.flip_distance for r in self.results.values()
                     if r.flipped and r.flip_distance is not None]
        return min(distances) if distances else None

    @property
    def most_fragile_transform(self) -> Optional[str]:
        best: Optional[FlipResult] = None
        for result in self.results.values():
            if result.flipped and result.flip_distance is not None:
                if best is None or result.flip_distance < (
                        best.flip_distance or float("inf")):
                    best = result
        return best.transform if best is not None else None

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "label": self.label,
            "results": {name: r.to_dict() for name, r in self.results.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExampleCard":
        return cls(
            index=int(data["index"]),
            label=data["label"],
            results={name: FlipResult.from_dict(r)
                     for name, r in data["results"].items()},
        )


@dataclass
class TransformSummary:
    """Aggregate flip behaviour for one transform over all probed examples."""

    name: str
    unit: str
    n_examples: int
    n_flipped: int
    flip_rate: float
    mean_flip_distance: Optional[float]
    median_flip_distance: Optional[float]
    min_flip_distance: Optional[float]
    flip_rate_at_budgets: List[Dict[str, float]] = field(default_factory=list)

    def rate_at(self, budget: float) -> Optional[float]:
        for entry in self.flip_rate_at_budgets:
            if entry["budget"] == budget:
                return entry["flip_rate"]
        return None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "unit": self.unit,
            "n_examples": self.n_examples,
            "n_flipped": self.n_flipped,
            "flip_rate": self.flip_rate,
            "mean_flip_distance": self.mean_flip_distance,
            "median_flip_distance": self.median_flip_distance,
            "min_flip_distance": self.min_flip_distance,
            "flip_rate_at_budgets": self.flip_rate_at_budgets,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TransformSummary":
        return cls(
            name=data["name"],
            unit=data["unit"],
            n_examples=int(data["n_examples"]),
            n_flipped=int(data["n_flipped"]),
            flip_rate=float(data["flip_rate"]),
            mean_flip_distance=data["mean_flip_distance"],
            median_flip_distance=data["median_flip_distance"],
            min_flip_distance=data["min_flip_distance"],
            flip_rate_at_budgets=list(data["flip_rate_at_budgets"]),
        )


@dataclass
class ProbeReport:
    """A complete probe run: cards, summaries, fragility, and the config."""

    cards: List[ExampleCard]
    summaries: List[TransformSummary]
    fragility_score: float
    config: dict = field(default_factory=dict)

    def summary_for(self, transform: str) -> TransformSummary:
        for summary in self.summaries:
            if summary.name == transform:
                return summary
        raise KeyError(f"no summary for transform {transform!r}")

    def summary(self) -> str:
        lines = [
            f"ProbeShift report: {len(self.cards)} example(s), "
            f"fragility score {self.fragility_score:.3f} "
            f"(mean flip rate at half of each grid's range; 0 = never "
            f"flips, 1 = always flips)",
        ]
        for s in self.summaries:
            mean = ("n/a" if s.mean_flip_distance is None
                    else f"{s.mean_flip_distance:.4g}")
            lines.append(
                f"  {s.name:<10} flips {s.n_flipped}/{s.n_examples} "
                f"({s.flip_rate:.1%}), mean flip distance {mean} {s.unit}"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "cards": [c.to_dict() for c in self.cards],
            "summaries": [s.to_dict() for s in self.summaries],
            "fragility_score": self.fragility_score,
            "config": self.config,
        }

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_dict(cls, data: dict) -> "ProbeReport":
        return cls(
            cards=[ExampleCard.from_dict(c) for c in data["cards"]],
            summaries=[TransformSummary.from_dict(s)
                       for s in data["summaries"]],
            fragility_score=float(data["fragility_score"]),
            config=dict(data["config"]),
        )
