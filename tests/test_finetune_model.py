from functools import partial, wraps
from types import SimpleNamespace
import sys

import pytest

from fakegpu._finetune_model import actual_training_config


class Parameter:
    def __init__(self, count, dtype="torch.bfloat16", *, trainable=False):
        self.count = count
        self.dtype = dtype
        self.requires_grad = trainable
        self.device = SimpleNamespace(type="cuda")

    def numel(self):
        return self.count

    def element_size(self):
        return {"torch.bfloat16": 2, "torch.float32": 4, "torch.uint8": 1}[self.dtype]


class MetadataTensor(Parameter):
    def untyped_storage(self):
        return SimpleNamespace(
            data_ptr=lambda: id(self),
            nbytes=lambda: self.count * self.element_size(),
        )


class Params4bit(Parameter):
    __module__ = "bitsandbytes.nn.modules"

    def __init__(self):
        # 512 logical values occupy 128 BF16 storage elements, not 256 values.
        super().__init__(128)
        self.quant_state = SimpleNamespace(
            shape=(8, 64),
            quant_type="nf4",
            nested=True,
            blocksize=64,
            dtype="torch.bfloat16",
            absmax=MetadataTensor(8, "torch.uint8"),
            code=MetadataTensor(16, "torch.float32"),
            offset=MetadataTensor(1, "torch.float32"),
            state2=SimpleNamespace(
                nested=False,
                absmax=MetadataTensor(1, "torch.float32"),
                code=MetadataTensor(256, "torch.float32"),
            ),
        )
        self.quant_type = "nf4"
        self.quant_storage = "torch.bfloat16"
        self.blocksize = 64
        self.compress_statistics = True
        self.bnb_quantized = True


class Linear4bit:
    __module__ = "bitsandbytes.nn.modules"

    def __init__(self):
        self.weight = Params4bit()
        self.in_features = 64
        self.out_features = 8
        self.compute_dtype = "torch.bfloat16"


class Arguments(SimpleNamespace):
    def to_dict(self):
        return vars(self).copy()


@pytest.fixture
def trainer(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            backends=SimpleNamespace(
                cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)),
                cudnn=SimpleNamespace(allow_tf32=True),
            )
        ),
    )
    modules = [Linear4bit(), Linear4bit()]
    modules[1].weight.quant_state.code = modules[0].weight.quant_state.code
    modules[1].weight.quant_state.state2.code = modules[
        0
    ].weight.quant_state.state2.code
    parameters = [
        ("q_proj.base_layer.weight", modules[0].weight),
        ("v_proj.base_layer.weight", modules[1].weight),
        ("embedding.weight", Parameter(100)),
        ("norm.weight", Parameter(16, "torch.float32")),
        (
            "q_proj.lora_A.default.weight",
            Parameter(16, "torch.float32", trainable=True),
        ),
        (
            "q_proj.lora_B.default.weight",
            Parameter(16, "torch.float32", trainable=True),
        ),
    ]
    model = SimpleNamespace(
        is_quantized=True,
        is_loaded_in_4bit=True,
        is_loaded_in_8bit=False,
        is_gradient_checkpointing=False,
        quantization_method="bitsandbytes",
        hf_quantizer=SimpleNamespace(pre_quantized=False),
        config=SimpleNamespace(_attn_implementation="sdpa"),
        peft_config={
            "default": SimpleNamespace(
                peft_type="LORA",
                r=8,
                lora_alpha=16,
                lora_dropout=0.0,
                bias="none",
                target_modules={"q_proj", "v_proj"},
            )
        },
        named_parameters=lambda: iter(parameters),
        modules=lambda: iter(modules),
    )
    args = Arguments(
        world_size=1,
        bf16=True,
        fp16=False,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        num_train_epochs=1,
        max_steps=6,
        optim="adamw_torch",
        learning_rate=2e-5,
        lr_scheduler_type="constant",
        warmup_steps=0,
        warmup_ratio=0.0,
        weight_decay=0.0,
        max_grad_norm=1.0,
        dataloader_drop_last=False,
    )
    return (
        SimpleNamespace(model=model, args=args, data_collator=object()),
        modules,
        parameters,
    )


def _normalized(method="qlora"):
    return {
        "method": method,
        "distributed_config": {"activation_checkpointing": False},
        "native_execution_signature": "raw-configuration-hash",
        # Deliberately different from runtime to verify reconciliation.
        "quantization_bits": 4 if method == "qlora" else None,
        "quantization_type": "fp4" if method == "qlora" else None,
        "double_quantization": False if method == "qlora" else None,
        "quantization_storage": "uint8" if method == "qlora" else None,
        "quantization_compute_dtype": "fp32" if method == "qlora" else None,
    }


def test_packed_storage_uses_logical_shape_and_effective_runtime_settings(trainer):
    trainer, _, _ = trainer
    config, model = actual_training_config(trainer, _normalized())
    assert config["parameter_count"] == 2 * 512 + 100 + 16
    assert config["trainable_parameter_count"] == 32
    assert model["total_parameter_count"] == 1172
    assert model["quantization"]["stored_parameter_elements"] == 404
    assert model["quantization"]["logical_parameter_count"] == 1024
    assert model["quantization"]["packed_bytes"] == 512
    assert model["quantization"]["frozen_unquantized_dtype_counts"] == {
        "torch.bfloat16": 100,
        "torch.float32": 16,
    }
    assert config["parameter_dtype"] == "bf16"
    assert config["quantization_type"] == "nf4"
    assert config["double_quantization"] is True
    assert config["quantization_compute_dtype"] == "bf16"
    assert config["quantization_storage"] == "bf16"
    assert config["lora_target_modules"] == ["q_proj", "v_proj"]
    assert config["actual_model_signature"] == model["actual_model_signature"]
    assert config["actual_quantized_storage"] == {
        "quantized_parameters": 1024,
        "packed_weight_bytes": 512,
        "metadata_bytes": 1120,
        "unquantized_weight_bytes": 264,
    }
    assert "gradient_checkpointing_kwargs" not in config


def test_runtime_inventory_detects_quantization_without_model_flag(trainer):
    trainer, _, _ = trainer
    trainer.model.is_quantized = False
    trainer.model.is_loaded_in_4bit = False
    config, _ = actual_training_config(trainer, _normalized())
    assert config["method"] == "qlora"


@pytest.mark.parametrize(
    "attribute,value,match",
    [
        ("count", 256, "storage size"),
        ("quant_state", None, "not been quantized"),
        ("bnb_quantized", False, "not been quantized"),
        ("requires_grad", True, "must be frozen"),
        ("quant_type", "fp4", "quantization type"),
        ("compress_statistics", False, "double-quantization flags"),
        ("quant_storage", "torch.uint8", "packed storage dtype"),
        ("blocksize", 128, "block size"),
    ],
)
def test_corrupt_or_uninitialized_packed_parameters_rejected(
    trainer, attribute, value, match
):
    trainer, modules, _ = trainer
    setattr(modules[0].weight, attribute, value)
    with pytest.raises(ValueError, match=match):
        actual_training_config(trainer, _normalized())


def test_bitsandbytes_block_size_32_is_accepted(trainer):
    # bitsandbytes 0.50.2 quantize_4bit accepts block sizes 32 through 4096.
    trainer, modules, _ = trainer
    for module in modules:
        module.weight.blocksize = module.weight.quant_state.blocksize = 32
    _, details = actual_training_config(trainer, _normalized())
    assert {
        module["block_size"] for module in details["quantization"]["modules"].values()
    } == {32}


def test_logical_shape_must_match_linear_dimensions(trainer):
    trainer, modules, _ = trainer
    modules[0].weight.quant_state.shape = (16, 32)
    with pytest.raises(ValueError, match="logical shape"):
        actual_training_config(trainer, _normalized())


def test_mixed_quantization_types_cannot_claim_one_nf4_workload(trainer):
    trainer, modules, _ = trainer
    modules[0].weight.quant_state.quant_type = "fp4"
    modules[0].weight.quant_type = "fp4"
    with pytest.raises(ValueError, match="inconsistent quantization_type"):
        actual_training_config(trainer, _normalized())


@pytest.mark.parametrize(
    "attribute,value", [("is_loaded_in_8bit", True), ("quantization_method", "gptq")]
)
def test_unsupported_runtime_quantization_backend_rejected(trainer, attribute, value):
    trainer, _, _ = trainer
    setattr(trainer.model, attribute, value)
    with pytest.raises(ValueError, match="8-bit|backend"):
        actual_training_config(trainer, _normalized())


def test_prequantized_runtime_rejected_even_if_input_claims_unquantized(trainer):
    trainer, _, _ = trainer
    trainer.model.hf_quantizer.pre_quantized = True
    with pytest.raises(ValueError, match="unquantized weights"):
        actual_training_config(trainer, _normalized())


def test_actual_method_cannot_silently_disagree_with_requested_method(trainer):
    trainer, _, _ = trainer
    with pytest.raises(ValueError, match="fine-tuning method disagrees"):
        actual_training_config(trainer, _normalized("lora"))


@pytest.mark.parametrize("index,trainable", [(2, True), (4, False)])
def test_unfrozen_base_or_frozen_adapter_cannot_corrupt_parameter_counts(
    trainer, index, trainable
):
    trainer, _, parameters = trainer
    parameters[index][1].requires_grad = trainable
    with pytest.raises(ValueError, match="frozen base weights and trainable"):
        actual_training_config(trainer, _normalized())


def test_quantization_metadata_counts_shared_storage_only_once(trainer):
    trainer, modules, _ = trainer
    config, _ = actual_training_config(trainer, _normalized())
    # A view with the same storage must not be counted as another allocation.
    original = modules[0].weight.quant_state.code
    view = MetadataTensor(8, "torch.float32")
    view.untyped_storage = original.untyped_storage
    modules[1].weight.quant_state.code = view
    after, _ = actual_training_config(trainer, _normalized())
    assert config["actual_quantized_storage"] == after["actual_quantized_storage"]
    assert config["actual_model_signature"] == after["actual_model_signature"]


def test_cpu_quantization_metadata_rejected(trainer):
    trainer, modules, _ = trainer
    modules[0].weight.quant_state.absmax.device = SimpleNamespace(type="cpu")
    with pytest.raises(ValueError, match="metadata must reside on CUDA"):
        actual_training_config(trainer, _normalized())


@pytest.mark.parametrize(
    "option,value", [("use_dora", True), ("peft_type", "IA3"), ("bias", "all")]
)
def test_actual_adapter_features_cannot_bypass_config_validation(
    trainer, option, value
):
    trainer, _, _ = trainer
    setattr(trainer.model.peft_config["default"], option, value)
    with pytest.raises(ValueError, match="adapter"):
        actual_training_config(trainer, _normalized())


def test_actual_model_signature_binds_dtype_of_unquantized_frozen_tensors(trainer):
    trainer, _, parameters = trainer
    before, _ = actual_training_config(trainer, _normalized())
    parameters[3][1].dtype = "torch.bfloat16"
    after, _ = actual_training_config(trainer, _normalized())
    assert before["parameter_dtype"] == after["parameter_dtype"] == "bf16"
    assert before["actual_model_signature"] != after["actual_model_signature"]


def test_fp4_and_uint8_storage_are_inspected_from_actual_modules(trainer):
    trainer, modules, _ = trainer
    for module in modules:
        parameter = module.weight
        parameter.count = 256
        parameter.dtype = parameter.quant_storage = "torch.uint8"
        parameter.quant_type = parameter.quant_state.quant_type = "fp4"
        parameter.compress_statistics = parameter.quant_state.nested = False
    config, model = actual_training_config(trainer, _normalized())
    assert config["quantization_type"] == "fp4"
    assert config["quantization_storage"] == "uint8"
    assert config["double_quantization"] is False
    assert config["parameter_count"] == 1140
    assert model["quantization"]["packed_bytes"] == 512


def _unquantize(trainer, modules, parameters, *, full=False):
    for index in range(2):
        parameters[index] = (parameters[index][0], Parameter(512, trainable=full))
    modules.clear()
    trainer.model.is_quantized = False
    trainer.model.is_loaded_in_4bit = False
    if full:
        trainer.model.peft_config = {}
        for _, parameter in parameters:
            parameter.requires_grad = True


@pytest.mark.parametrize("full", [False, True])
def test_unquantized_inventory_keeps_existing_optional_signature_fields_absent(
    trainer, full
):
    trainer, modules, parameters = trainer
    _unquantize(trainer, modules, parameters, full=full)
    config, model = actual_training_config(
        trainer, _normalized("full" if full else "lora")
    )
    assert config["parameter_count"] == (1172 if full else 1140)
    assert config["trainable_parameter_count"] == (1172 if full else 32)
    assert config["parameter_dtype"] == "bf16"
    assert "actual_model_signature" not in config
    assert "gradient_checkpointing_kwargs" not in config
    assert "quantization" not in model


def test_factory_wrapped_checkpoint_function_reveals_actual_kwargs(trainer):
    trainer, modules, _ = trainer
    checkpoint = partial(lambda *args, **kwargs: None, use_reentrant=False)

    @wraps(checkpoint)
    def native_wrapper(*args, **kwargs):
        return checkpoint(*args, **kwargs)

    trainer.model.is_gradient_checkpointing = True
    modules[0].gradient_checkpointing = True
    modules[0]._gradient_checkpointing_func = native_wrapper
    normalized = dict(
        _normalized(), gradient_checkpointing_kwargs={"use_reentrant": True}
    )
    config, model = actual_training_config(trainer, normalized)
    assert config["gradient_checkpointing_kwargs"] == {"use_reentrant": False}
    assert model["gradient_checkpointing_kwargs"] == {"use_reentrant": False}


def test_inconsistent_checkpointing_modules_rejected(trainer):
    trainer, modules, _ = trainer
    trainer.model.is_gradient_checkpointing = True
    for index, module in enumerate(modules):
        module.gradient_checkpointing = True
        module._gradient_checkpointing_func = partial(
            lambda: None, use_reentrant=bool(index)
        )
    with pytest.raises(ValueError, match="inspectable and consistent"):
        actual_training_config(trainer, _normalized())


def test_disabled_actual_checkpointing_removes_stale_declared_kwargs(trainer):
    trainer, _, _ = trainer
    config, _ = actual_training_config(
        trainer,
        dict(_normalized(), gradient_checkpointing_kwargs={"use_reentrant": False}),
    )
    assert config["gradient_checkpointing"] is False
    assert "gradient_checkpointing_kwargs" not in config
