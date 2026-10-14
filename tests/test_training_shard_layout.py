"""Regression for native FSDP2 per-parameter dim-0 shard imbalance."""

import pytest

from fakegpu.training_plan import TrainingPlanError, estimate_training_plan


CONFIG = {
    "framework": "fsdp2",
    "world_size": 2,
    "mixed_precision": "no",
    "sharding_strategy": "full_shard",
}
SHAPES = [[12, 8], [12], [3, 12], [3]]


def test_fsdp2_uneven_parameter_shapes_budget_largest_rank():
    plan = estimate_training_plan(
        CONFIG,
        parameter_bytes=1176,
        activation_bytes=0,
        parameter_element_bytes=8,
        parameter_shapes=SHAPES,
    )
    layout = plan["sharding"]["parameter_layout"]
    assert layout["parameter_bytes_by_rank"] == [640, 536]
    assert layout["budgeted_rank"] == 0
    storage = plan["rank_local_storage"]
    assert storage["parameter_bytes"] == storage["gradient_bytes"] == 640
    assert storage["optimizer_state_bytes"] == 1280
    assert storage["master_parameter_bytes"] == 0


@pytest.mark.parametrize(
    "config,shapes,match",
    [
        ({**CONFIG, "framework": "fsdp"}, SHAPES, "fsdp2"),
        ({**CONFIG, "sharding_strategy": "replicated"}, SHAPES, "full_shard"),
        (CONFIG, [[12, 8]], "parameter_bytes"),
        (CONFIG, [[0]], "positive"),
        (CONFIG, [[]], "nonempty"),
        (CONFIG, [[True]], "positive"),
    ],
)
def test_reject_unsupported_or_inconsistent_layouts(config, shapes, match):
    with pytest.raises(TrainingPlanError, match=match):
        estimate_training_plan(
            config,
            parameter_bytes=1176,
            activation_bytes=0,
            parameter_element_bytes=8,
            parameter_shapes=shapes,
        )


def test_dim0_empty_tail_ranks_are_not_fractional_elements():
    plan = estimate_training_plan(
        {**CONFIG, "world_size": 4},
        parameter_bytes=24,
        activation_bytes=0,
        parameter_element_bytes=8,
        parameter_shapes=[[3]],
    )
    assert plan["sharding"]["parameter_layout"]["parameter_bytes_by_rank"] == [
        8,
        8,
        8,
        0,
    ]
    assert plan["rank_local_storage"]["parameter_bytes"] == 8


def test_omitted_layout_retains_aggregate_model():
    plan = estimate_training_plan(
        CONFIG, parameter_bytes=1176, activation_bytes=0, parameter_element_bytes=8
    )
    assert plan["rank_local_storage"]["parameter_bytes"] == 588
    assert "parameter_layout" not in plan["sharding"]


def test_layout_cli_and_missing_shapes_error(tmp_path):
    import json
    from fakegpu.training_plan import main

    config, shapes, output = (
        tmp_path / name for name in ("config.json", "shapes.json", "report.json")
    )
    config.write_text(json.dumps(CONFIG))
    shapes.write_text(json.dumps({"shapes": SHAPES}))
    argv = [
        str(config),
        "--framework",
        "fsdp2",
        "--parameter-bytes",
        "1176",
        "--activation-bytes",
        "0",
        "--parameter-element-bytes",
        "8",
        "--parameter-shapes",
        str(shapes),
        "--json",
        str(output),
    ]
    assert main(argv) == 0
    assert (
        json.loads(output.read_text())["rank_local_storage"]["parameter_bytes"] == 640
    )
    shapes.write_text("{}")
    with pytest.raises(SystemExit) as error:
        main(argv)
    assert error.value.code == 2
