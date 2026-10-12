import math

import pytest

from rlinf.workers.sft.fsdp_sft_worker import gradient_diagnostics_due
from rlinf.workers.sft.fsdp_vla_sft_worker import (
    gradient_interaction_metrics,
    shared_gradient_interaction_metrics,
)


@pytest.mark.parametrize(
    ("global_step", "interval", "expected"),
    ((0, 500, False), (499, 500, False), (500, 500, True), (1000, 500, True)),
)
def test_gradient_diagnostics_skip_startup(
    global_step: int,
    interval: int,
    expected: bool,
) -> None:
    assert gradient_diagnostics_due(global_step, interval) is expected


@pytest.mark.parametrize(
    ("total_norm", "expected_cosine", "expected_interference"),
    [
        (7.0, 1.0, 1.0),
        (5.0, 0.0, 5.0 / 7.0),
        (1.0, -1.0, 1.0 / 7.0),
    ],
)
def test_gradient_interaction_metrics(
    total_norm: float,
    expected_cosine: float,
    expected_interference: float,
) -> None:
    metrics = gradient_interaction_metrics(
        action_norm=3.0,
        world_norm=4.0,
        total_norm=total_norm,
    )

    assert metrics["action_grad_norm"] == pytest.approx(3.0)
    assert metrics["weighted_tactile_world_grad_norm"] == pytest.approx(4.0)
    assert metrics["diagnostic_total_grad_norm"] == pytest.approx(total_norm)
    assert metrics["action_grad_norm_share"] == pytest.approx(3.0 / 7.0)
    assert metrics["weighted_tactile_world_grad_norm_share"] == pytest.approx(4.0 / 7.0)
    assert metrics["tactile_world_to_action_grad_ratio"] == pytest.approx(4.0 / 3.0)
    assert metrics["action_world_grad_cosine"] == pytest.approx(expected_cosine)
    assert metrics["action_world_grad_cosine_valid"] == 1.0
    assert metrics["gradient_norm_triangle_valid"] == 1.0
    assert metrics["gradient_interference_factor"] == pytest.approx(
        expected_interference
    )


def test_gradient_interaction_metrics_keep_zero_gradients_finite() -> None:
    metrics = gradient_interaction_metrics(0.0, 0.0, 0.0)

    assert all(math.isfinite(value) for value in metrics.values())
    assert metrics["action_world_grad_cosine"] == 0.0
    assert metrics["action_world_grad_cosine_valid"] == 0.0
    assert metrics["tactile_world_to_action_grad_ratio"] == 0.0
    assert metrics["tactile_world_to_action_grad_ratio_valid"] == 0.0


def test_gradient_interaction_metrics_flag_inconsistent_norms() -> None:
    metrics = gradient_interaction_metrics(3.0, 4.0, 10.0)

    assert metrics["gradient_norm_triangle_valid"] == 0.0
    assert metrics["action_world_grad_cosine_valid"] == 0.0
    assert metrics["action_world_grad_cosine"] == 1.0


def test_gradient_interaction_metrics_name_tactile_expert_branch() -> None:
    metrics = gradient_interaction_metrics(
        3.0,
        4.0,
        5.0,
        auxiliary_name="tactile_expert",
    )

    assert metrics["action_grad_norm"] == pytest.approx(3.0)
    assert metrics["weighted_tactile_expert_grad_norm"] == pytest.approx(4.0)
    assert metrics["tactile_expert_to_action_grad_ratio"] == pytest.approx(4.0 / 3.0)
    assert metrics["action_tactile_expert_grad_cosine"] == pytest.approx(0.0)
    assert metrics["action_tactile_expert_grad_cosine_valid"] == 1.0
    assert "weighted_tactile_world_grad_norm" not in metrics


def test_shared_gradient_metrics_exclude_branch_only_parameters() -> None:
    metrics = shared_gradient_interaction_metrics(
        action_squares={"shared": 9.0, "action_only": 16.0, "world_only": 0.0},
        world_squares={"shared": 16.0, "action_only": 0.0, "world_only": 9.0},
        total_squares={"shared": 25.0, "action_only": 16.0, "world_only": 9.0},
    )

    assert metrics["shared_action_grad_norm"] == pytest.approx(3.0)
    assert metrics["shared_weighted_tactile_world_grad_norm"] == pytest.approx(4.0)
    assert metrics["shared_action_world_grad_cosine"] == pytest.approx(0.0)
    assert metrics["shared_action_world_grad_cosine_valid"] == 1.0
    assert metrics["action_exclusive_grad_norm"] == pytest.approx(4.0)
    assert metrics["weighted_tactile_world_exclusive_grad_norm"] == pytest.approx(3.0)
    assert metrics["shared_action_grad_norm_fraction"] == pytest.approx(3.0 / 5.0)
    assert metrics["shared_weighted_tactile_world_grad_norm_fraction"] == pytest.approx(
        4.0 / 5.0
    )


def test_shared_gradient_metrics_name_tactile_expert_branch() -> None:
    metrics = shared_gradient_interaction_metrics(
        action_squares={"shared": 9.0, "action_only": 16.0},
        world_squares={"shared": 16.0, "tactile_only": 9.0},
        total_squares={
            "shared": 25.0,
            "action_only": 16.0,
            "tactile_only": 9.0,
        },
        auxiliary_name="tactile_expert",
    )

    assert metrics["shared_action_grad_norm"] == pytest.approx(3.0)
    assert metrics["shared_weighted_tactile_expert_grad_norm"] == pytest.approx(4.0)
    assert metrics["shared_action_tactile_expert_grad_cosine"] == pytest.approx(0.0)
    assert metrics["weighted_tactile_expert_exclusive_grad_norm"] == pytest.approx(3.0)
