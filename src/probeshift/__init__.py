"""ProbeShift — measure how little must change before a classifier flips.

ProbeShift sweeps fixed grids of *benign* transforms (brightness, contrast,
blur, noise, shift, rotation) over your images and records, per example and
per transform, the smallest magnitude that changes the model's prediction —
the flip distance — plus flip rates at fixed budgets and an overall
fragility score.  It is a measuring instrument, not an attack tool: no
gradients, no optimisation, no target class, and the model is any callable
returning labels or class scores.
"""

from .cards import ExampleCard, FlipResult, ProbeReport, TransformSummary
from .core import probe, probe_image
from .transforms import Transform, default_transforms, get_transform

__version__ = "0.1.0"

__all__ = [
    "probe",
    "probe_image",
    "ProbeReport",
    "ExampleCard",
    "FlipResult",
    "TransformSummary",
    "Transform",
    "default_transforms",
    "get_transform",
    "__version__",
]
