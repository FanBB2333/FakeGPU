from copy import deepcopy

import pytest

from fakegpu.finetune_run_estimator import estimate_finetuning_run
from test_finetune_run_estimator import native_run as native_run


@pytest.mark.parametrize("field", ["eval_strategy", "save_strategy"])
@pytest.mark.parametrize("strategy", ["steps", "epoch"])
def test_native_schedule_cannot_hide_recorded_periodic_events(
    native_run, field, strategy
):
    native_run["observation"]["observed_model"] = {
        "training_arguments": {field: strategy},
    }
    native_run["schedule"][field] = strategy
    assert (
        estimate_finetuning_run(native_run)["timing"]["native_pipeline"]["available"]
        is False
    )
    changed = deepcopy(native_run)
    changed["schedule"][field] = "no"
    with pytest.raises(ValueError, match="schedule.*recorded training arguments"):
        estimate_finetuning_run(changed)
