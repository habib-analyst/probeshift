"""Tests for ProbeShift.

The load-bearing tests are the analytic ones: toy models whose decision
boundaries are known exactly, so the measured flip distance must match the
hand-computed value — not just "a flip happened somewhere".
"""

import numpy as np
import pytest

from probeshift import (
    __version__,
    default_transforms,
    get_transform,
    probe,
    probe_image,
)
from probeshift.transforms import Transform


# ---------------------------------------------------------------------------
# Toy models with known boundaries
# ---------------------------------------------------------------------------

def mean_threshold_model(batch):
    """Class 1 iff the image's mean intensity exceeds 0.5."""
    return (batch.mean(axis=(1, 2)) > 0.5).astype(int)


def mean_threshold_probs(batch):
    labels = mean_threshold_model(batch)
    probs = np.full((batch.shape[0], 2), 0.1)
    probs[np.arange(batch.shape[0]), labels] = 0.9
    return probs


def constant_model(batch):
    """Always class 0, whatever the input."""
    probs = np.zeros((batch.shape[0], 3))
    probs[:, 0] = 1.0
    return probs


# ---------------------------------------------------------------------------
# Analytic flip distances
# ---------------------------------------------------------------------------

def test_brightness_flip_distance_matches_analytic_boundary():
    # Constant 0.44 image; the boundary is at added brightness +0.06.
    images = np.full((1, 12, 12), 0.44)
    report = probe(mean_threshold_model, images, transforms=["brightness"])
    result = report.cards[0].results["brightness"]
    assert result.flipped
    assert result.flipped_to == 1
    # Grid resolution: +0.05 leaves the mean at 0.49, +0.08 flips it.
    assert result.grid_distance == pytest.approx(0.08)
    # Refined by bisection: the true boundary, +0.06.
    assert result.flip_distance == pytest.approx(0.06, abs=0.005)


def test_brightness_flip_distance_without_refinement_is_grid_value():
    images = np.full((1, 12, 12), 0.44)
    report = probe(mean_threshold_model, images,
                   transforms=["brightness"], refine=False)
    result = report.cards[0].results["brightness"]
    assert result.flip_distance == pytest.approx(0.08)


def test_mean_preserving_transforms_never_flip_a_constant_image():
    # Contrast rescales around the mean, blur averages neighbours, shift
    # and rotation rearrange identical pixels: a constant image is exactly
    # invariant, so the mean-threshold model can never flip.
    images = np.full((2, 12, 12), 0.44)
    report = probe(mean_threshold_model, images,
                   transforms=["contrast", "blur", "shift", "rotation"])
    for card in report.cards:
        assert not card.flipped_any
    assert report.fragility_score == 0.0


def test_shift_flip_distance_matches_analytic_boundary():
    # Bright mass in columns 1-2 of a 10x10 image (kept off the border so
    # edge-replication fill plays no part).  The model fires when the
    # right half outweighs the left half.  Shifting right: at dx=3 the
    # halves tie (10 vs 10, not >), at dx=4 the right half wins (20 vs 0).
    image = np.zeros((10, 10))
    image[:, 1:3] = 1.0

    def half_balance_model(batch):
        left = batch[:, :, 0:5].sum(axis=(1, 2))
        right = batch[:, :, 5:10].sum(axis=(1, 2))
        return (right > left).astype(int)

    report = probe(half_balance_model, image[np.newaxis],
                   transforms=["shift"])
    result = report.cards[0].results["shift"]
    assert result.flipped
    assert result.flipped_to == 1
    assert result.flip_distance == pytest.approx(4.0)
    assert list(result.flip_param) == [4, 0]


# ---------------------------------------------------------------------------
# Robust dummy
# ---------------------------------------------------------------------------

def test_constant_predictor_reports_no_flips_anywhere():
    rng = np.random.default_rng(3)
    images = rng.uniform(0.0, 1.0, size=(4, 10, 10))
    report = probe(constant_model, images)
    assert report.fragility_score == 0.0
    for summary in report.summaries:
        assert summary.n_flipped == 0
        assert summary.flip_rate == 0.0
        assert summary.mean_flip_distance is None
        assert all(e["flip_rate"] == 0.0
                   for e in summary.flip_rate_at_budgets)
    for card in report.cards:
        assert not card.flipped_any
        assert card.min_flip_distance is None


# ---------------------------------------------------------------------------
# Determinism, equivalence, hygiene
# ---------------------------------------------------------------------------

def test_probe_is_deterministic_including_noise():
    rng = np.random.default_rng(7)
    images = rng.uniform(0.3, 0.7, size=(5, 9, 9))
    first = probe(mean_threshold_model, images)
    second = probe(mean_threshold_model, images)
    assert first.to_dict() == second.to_dict()


def test_labels_and_probabilities_inputs_agree():
    rng = np.random.default_rng(11)
    images = rng.uniform(0.3, 0.7, size=(5, 9, 9))
    by_label = probe(mean_threshold_model, images)
    by_probs = probe(mean_threshold_probs, images)
    assert by_label.to_dict() == by_probs.to_dict()


def test_input_batch_is_not_mutated():
    rng = np.random.default_rng(5)
    images = rng.uniform(0.0, 1.0, size=(3, 8, 8))
    snapshot = images.copy()
    probe(mean_threshold_model, images)
    assert np.array_equal(images, snapshot)


def test_probe_image_matches_batch_result_for_brightness():
    image = np.full((12, 12), 0.44)
    card = probe_image(mean_threshold_model, image,
                       transforms=["brightness"])
    batch = probe(mean_threshold_model, image[np.newaxis],
                  transforms=["brightness"])
    assert card.label == 0
    assert card.results["brightness"].flip_distance == pytest.approx(
        batch.cards[0].results["brightness"].flip_distance)


def test_fragility_orders_fragile_above_robust():
    images = np.full((3, 10, 10), 0.44)
    fragile = probe(mean_threshold_model, images,
                    transforms=["brightness"])
    robust = probe(constant_model, images, transforms=["brightness"])
    assert 0.0 <= robust.fragility_score <= fragile.fragility_score <= 1.0
    assert fragile.fragility_score > 0.0


def test_report_json_round_trip():
    rng = np.random.default_rng(13)
    images = rng.uniform(0.3, 0.7, size=(4, 8, 8))
    report = probe(mean_threshold_model, images)
    from probeshift import ProbeReport
    restored = ProbeReport.from_dict(report.to_dict())
    assert restored.to_dict() == report.to_dict()
    assert report.to_json()  # serialises without error


def test_summary_mentions_every_transform_and_fragility():
    images = np.full((2, 8, 8), 0.44)
    report = probe(mean_threshold_model, images)
    text = report.summary()
    for transform in default_transforms():
        assert transform.name in text
    assert "fragility" in text


def test_custom_budgets_are_used():
    images = np.full((1, 12, 12), 0.44)
    report = probe(mean_threshold_model, images,
                   transforms=["brightness"],
                   budgets={"brightness": [0.05, 0.1]})
    summary = report.summary_for("brightness")
    rates = {e["budget"]: e["flip_rate"]
             for e in summary.flip_rate_at_budgets}
    assert rates == {0.05: 0.0, 0.1: 1.0}  # refined distance is 0.06


# ---------------------------------------------------------------------------
# Transform primitives
# ---------------------------------------------------------------------------

def test_transform_grids_are_ordered_by_distance():
    for transform in default_transforms():
        distances = list(transform.distances())
        assert distances == sorted(distances), transform.name
        assert all(d > 0 for d in distances)


def test_identity_parameters_return_the_input():
    rng = np.random.default_rng(17)
    batch = rng.uniform(0.0, 1.0, size=(2, 8, 8))
    assert np.allclose(get_transform("brightness").apply(batch, 0.0), batch)
    assert np.allclose(get_transform("contrast").apply(batch, 1.0), batch)
    assert np.allclose(get_transform("blur").apply(batch, 0.0), batch)
    assert np.allclose(get_transform("rotation").apply(batch, 0.0), batch)
    assert np.allclose(get_transform("shift").apply(batch, (0, 0)), batch)
    field = rng.standard_normal(batch.shape)
    assert np.allclose(
        get_transform("noise").apply(batch, 0.0, field), batch)


def test_blur_preserves_a_constant_image():
    batch = np.full((2, 9, 9), 0.44)
    blurred = get_transform("blur").apply(batch, 3.0)
    assert np.allclose(blurred, batch, atol=1e-12)


def test_shift_moves_a_bright_pixel():
    batch = np.zeros((1, 6, 6))
    batch[0, 2, 2] = 1.0
    shifted = get_transform("shift").apply(batch, (2, 1))
    assert shifted[0, 3, 4] == 1.0
    assert shifted.sum() == 1.0


def test_noise_scales_the_fixed_field():
    batch = np.full((1, 4, 4), 0.5)
    field = np.ones((1, 4, 4))
    out = get_transform("noise").apply(batch, 0.1, field)
    assert np.allclose(out, 0.6)


def test_channel_images_are_supported():
    def channel_mean_model(batch):
        return (batch.mean(axis=(1, 2, 3)) > 0.5).astype(int)

    images = np.full((2, 10, 10, 3), 0.44)
    report = probe(channel_mean_model, images,
                   transforms=["brightness", "blur"])
    assert report.cards[0].results["brightness"].flipped
    assert not report.cards[1].results["blur"].flipped


def test_version_is_exposed():
    assert __version__ == "0.1.0"
    assert isinstance(Transform, type)


# ---------------------------------------------------------------------------
# Invalid inputs
# ---------------------------------------------------------------------------

def test_rejects_non_callable_predict_fn():
    with pytest.raises(TypeError):
        probe("not a function", np.zeros((1, 4, 4)))


def test_rejects_single_image_passed_to_probe():
    with pytest.raises(ValueError):
        probe(mean_threshold_model, np.zeros((8, 8)))


def test_rejects_batch_passed_to_probe_image():
    with pytest.raises(ValueError):
        probe_image(mean_threshold_model, np.zeros((2, 8, 8, 3)))


def test_rejects_nan_images():
    images = np.full((1, 6, 6), 0.5)
    images[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        probe(mean_threshold_model, images)


def test_rejects_empty_batch():
    with pytest.raises(ValueError):
        probe(mean_threshold_model, np.zeros((0, 6, 6)))


def test_rejects_unknown_transform_name():
    with pytest.raises(ValueError):
        probe(mean_threshold_model, np.zeros((1, 6, 6)),
              transforms=["posterize"])


def test_rejects_wrong_length_predictions():
    def bad_model(batch):
        return np.zeros(batch.shape[0] + 1, dtype=int)

    with pytest.raises(ValueError):
        probe(bad_model, np.zeros((2, 6, 6)))


def test_rejects_negative_budget():
    with pytest.raises(ValueError):
        probe(mean_threshold_model, np.zeros((1, 6, 6)),
              transforms=["brightness"], budgets={"brightness": [-0.1]})


def test_rejects_empty_transform_list():
    with pytest.raises(ValueError):
        probe(mean_threshold_model, np.zeros((1, 6, 6)), transforms=[])
