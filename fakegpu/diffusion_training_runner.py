"""Optional native CUDA runner for explicit cached-feature diffusion training."""

from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import uuid
import os
from .diffusion_training import (
    RUN_SCHEMA,
    TARGETS,
    _file,
    _digest,
    plan_diffusion_training,
    training_windows,
    validate_training_plan,
)
from .structured_io import write_json
from ._finetune_cache import capture_cache_regime
from ._finetune_allocator import capture_memory_allocator


def _update_ema(shadows, parameters, decay, *, torch):
    if len(shadows) != len(parameters):
        raise ValueError("EMA shadows must match every trainable parameter")
    with torch.no_grad():
        for shadow, parameter in zip(shadows, parameters):
            shadow.lerp_(parameter, 1 - decay)


def _save_ema(
    model, names, shadows, directory, original_directory, *, method, decay, updates
):
    from safetensors import safe_open
    from safetensors.torch import save_file

    if len(names) != len(shadows):
        raise ValueError("EMA export requires all named shadows")
    directory.mkdir()
    state = {name: value.cpu() for name, value in zip(names, shadows)}
    if method == "full":
        model.save_config(directory)
        save_file(
            state,
            str(directory / "diffusion_pytorch_model.safetensors"),
            metadata={"format": "pt"},
        )
    else:
        from peft import get_peft_model_state_dict

        state = get_peft_model_state_dict(model, state_dict=state)
        with safe_open(
            original_directory / "pytorch_lora_weights.safetensors",
            framework="pt",
            device="cpu",
        ) as source:
            metadata = source.metadata()
        save_file(
            state,
            str(directory / "pytorch_lora_weights.safetensors"),
            metadata=metadata,
        )
    write_json(
        directory / "fakegpu_ema.json",
        {
            "schema_version": "fakegpu.diffusion_ema.v1",
            "decay": decay,
            "optimizer_updates": updates,
            "parameters": names,
        },
    )


def _backward_window(model, scheduler, batches, *, torch, device, autocast=True):
    """Average all samples in a possibly partial accumulation window exactly once."""
    rows = sum(batch["latents"].shape[0] for batch in batches)
    losses = []
    for batch in batches:
        data = {k: v.to(device) for k, v in batch.items()}
        if any(
            not bool(torch.isfinite(v).all())
            for k, v in data.items()
            if k != "timesteps"
        ):
            raise ValueError("cached training tensors must be finite")
        steps = data["timesteps"]
        if (
            steps.min().item() < 0
            or steps.max().item() >= scheduler.config.num_train_timesteps
        ):
            raise ValueError("cached timesteps are outside the DDPM training schedule")
        added = (
            {"text_embeds": data["pooled_text_embeds"], "time_ids": data["time_ids"]}
            if "pooled_text_embeds" in data
            else None
        )
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=autocast):
            noisy = scheduler.add_noise(data["latents"], data["noise"], steps)
            output = model(
                noisy,
                steps,
                encoder_hidden_states=data["encoder_hidden_states"],
                added_cond_kwargs=added,
            ).sample
            loss = torch.nn.functional.mse_loss(
                output.float(), data["noise"].float(), reduction="mean"
            )
            loss = loss * (len(steps) / rows)
        if not bool(torch.isfinite(loss)):
            raise ValueError("native training loss is not finite")
        loss.backward()
        losses.append(float(loss.detach()))
    return sum(losses)


def run_diffusion_training(plan, *, output_dir, cache_regime="uncontrolled"):
    """Run all declared epochs on one real GPU, save native weights, retain a ledger.

    Setup includes optional imports and content hashing. Update times include
    CPU mmap reads, finite checks, data transfer, accumulation and optimizer work.
    The final model/adapter is saved; optimizer-state resume is not provided.
    """
    start = time.perf_counter()
    cache = capture_cache_regime(cache_regime)
    started = datetime.now(timezone.utc).isoformat()
    identity = {
        "uuid": str(uuid.uuid4()),
        "pid": os.getpid(),
        "started_at_utc": started,
    }
    plan = validate_training_plan(plan)
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("diffusion training requires a new or empty output directory")
    if (
        destination == Path(plan["model_dir"])
        or Path(plan["model_dir"]) in destination.parents
        or destination == Path(plan["dataset_path"]).parent
    ):
        raise ValueError("training outputs must be outside input directories")
    destination.mkdir(parents=True, exist_ok=True)
    s = plan["signature"]
    actual = plan_diffusion_training(
        plan["model_dir"],
        plan["dataset_path"],
        batch_size=s["batch_size"],
        gradient_accumulation_steps=s["gradient_accumulation_steps"],
        epochs=plan["epochs"],
        method=s["method"],
        lora_rank=s["lora_rank"] if s["method"] == "lora" else 8,
        gradient_checkpointing=s["gradient_checkpointing"],
        ema_decay=s["ema_decay"],
    )
    if actual != plan:
        raise ValueError("cached training inputs changed after planning")
    inputs = [Path(plan["dataset_path"])]
    for entry in plan["input_manifest"]["model"]:
        component = "scheduler" if entry["name"] == "scheduler_config.json" else "unet"
        inputs.append(Path(plan["model_dir"]) / component / entry["name"])

    def marker(path):
        stat = path.stat()
        return (
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        )

    input_markers = {path: marker(path) for path in inputs}
    import torch
    import diffusers
    from safetensors import safe_open
    from ._calibration_cuda import (
        _reject_simulated_cuda_environment,
        _real_cuda_environment,
    )

    _reject_simulated_cuda_environment(torch)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError(
            "cached diffusion training requires exactly one visible real CUDA GPU"
        )
    torch.cuda.set_device(0)
    allocator = capture_memory_allocator(torch)
    if allocator["backend"] != "native" or allocator["environment"][
        "PYTORCH_NO_CUDA_MEMORY_CACHING"
    ] not in (None, "", "0"):
        raise ValueError(
            "diffusion peak measurement requires PyTorch's native caching allocator"
        )
    device = torch.device("cuda:0")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    software = {
        name: importlib.metadata.version(name)
        for name in ("diffusers", "transformers", "peft", "safetensors")
    }
    environment = _real_cuda_environment(
        torch,
        framework="diffusers",
        framework_version=diffusers.__version__,
        software={k + "_version": v for k, v in software.items()},
    )
    implementation = _digest(
        {
            name: hashlib.sha256(
                Path(__file__).with_name(name).read_bytes()
            ).hexdigest()
            for name in ("diffusion_training.py", "diffusion_training_runner.py")
        }
    )
    torch.cuda.reset_peak_memory_stats()
    model, loading = diffusers.UNet2DConditionModel.from_pretrained(
        plan["model_dir"],
        subfolder="unet",
        local_files_only=True,
        use_safetensors=True,
        torch_dtype=torch.float32 if s["method"] == "full" else torch.bfloat16,
        output_loading_info=True,
    )
    if any(
        loading.get(name)
        for name in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    ):
        raise ValueError(f"native UNet checkpoint did not load exactly: {loading}")
    if sum(p.numel() for p in model.parameters()) != s["base_parameters"]:
        raise ValueError(
            "loaded UNet parameter count differs from checkpoint inspection"
        )
    if s["method"] == "lora":
        from peft import LoraConfig

        model.requires_grad_(False)
        model.add_adapter(
            LoraConfig(
                r=s["lora_rank"],
                lora_alpha=s["lora_rank"],
                target_modules=list(TARGETS),
                bias="none",
                lora_dropout=0.0,
            )
        )
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.float()
    model = model.to(device)
    model.train()
    processors = {
        type(p).__module__ + "." + type(p).__name__
        for p in model.attn_processors.values()
    }
    if processors != {"diffusers.models.attention_processor.AttnProcessor2_0"}:
        raise ValueError("cached training requires the native SDPA attention processor")
    if s["gradient_checkpointing"]:
        model.enable_gradient_checkpointing()
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    parameters = [p for _, p in named]
    trainable = sum(p.numel() for p in parameters)
    if trainable != s["trainable_parameters"] or any(
        p.dtype != torch.float32 for p in parameters
    ):
        raise ValueError("loaded trainable parameters differ from the plan")
    observed = {
        "base_parameters": s["base_parameters"],
        "trainable_parameters": trainable,
        "trainable_dtypes": {str(torch.float32): trainable},
        "attention_processors": sorted(processors),
        "gradient_checkpointing": model.is_gradient_checkpointing,
    }
    if observed["gradient_checkpointing"] != s["gradient_checkpointing"]:
        raise ValueError("native checkpointing differs from the plan")
    optimizer = torch.optim.AdamW(parameters, lr=2e-5, weight_decay=0.0, foreach=False)
    shadows = (
        [p.detach().clone() for p in parameters] if s["ema_decay"] is not None else None
    )
    scheduler = diffusers.DDPMScheduler.from_pretrained(
        plan["model_dir"], subfolder="scheduler", local_files_only=True
    )
    if scheduler.config.prediction_type != "epsilon":
        raise ValueError("native DDPM prediction type differs from the plan")
    torch.cuda.synchronize()
    boundary = time.perf_counter()
    setup = boundary - start
    updates = []
    with safe_open(plan["dataset_path"], framework="pt", device="cpu") as source:
        keys = sorted(source.keys())
        for window in training_windows(plan):
            optimizer.zero_grad(set_to_none=True)
            batches = []
            hashes = []
            for indices in window["row_indices"]:
                batch = {
                    key: source.get_slice(key)[indices[0] : indices[-1] + 1]
                    for key in keys
                }
                digest = hashlib.sha256()
                for key, tensor in sorted(batch.items()):
                    header = json.dumps(
                        {
                            "name": key,
                            "shape": list(tensor.shape),
                            "dtype": str(tensor.dtype),
                        },
                        sort_keys=True,
                    ).encode()
                    digest.update(len(header).to_bytes(8, "little"))
                    digest.update(header)
                    digest.update(tensor.numpy().tobytes())
                hashes.append(digest.hexdigest())
                batches.append(batch)
            loss = _backward_window(
                model, scheduler, batches, torch=torch, device=device
            )
            optimizer.step()
            if shadows is not None:
                _update_ema(shadows, parameters, s["ema_decay"], torch=torch)
            torch.cuda.synchronize()
            end = time.perf_counter()
            updates.append(
                {
                    **window,
                    "seconds": end - boundary,
                    "loss": loss,
                    "input_sha256": hashes,
                }
            )
            boundary = end
    model_dir = destination / "model"
    observed["gradient_bytes"] = sum(
        p.grad.numel() * p.grad.element_size() for p in parameters if p.grad is not None
    )
    observed["optimizer_moment_bytes"] = sum(
        value.numel() * value.element_size()
        for state in optimizer.state.values()
        for name, value in state.items()
        if name in {"exp_avg", "exp_avg_sq"}
    )
    observed["ema_bytes"] = (
        sum(value.numel() * value.element_size() for value in shadows)
        if shadows is not None
        else 0
    )
    if (
        observed["gradient_bytes"] != plan["static_memory"]["components"]["gradients"]
        or observed["optimizer_moment_bytes"]
        != plan["static_memory"]["components"]["adamw_moments"]
    ):
        raise ValueError(
            "native gradient/optimizer storage differs from the supported dense training contract"
        )
    if s["method"] == "lora":
        model.save_lora_adapter(str(model_dir), safe_serialization=True)
    else:
        model.save_pretrained(str(model_dir), safe_serialization=True)
    if shadows is not None:
        _save_ema(
            model,
            [name for name, _ in named],
            shadows,
            destination / "ema",
            model_dir,
            method=s["method"],
            decay=s["ema_decay"],
            updates=len(updates),
        )
    torch.cuda.synchronize()
    peaks = {
        field: int(
            getattr(
                torch.cuda,
                "max_memory_" + field.removeprefix("peak_").removesuffix("_bytes"),
            )()
        )
        for field in ("peak_allocated_bytes", "peak_reserved_bytes")
    }
    files = [
        {**_file(p), "path": str(p.relative_to(destination))}
        for p in sorted(destination.rglob("*"))
        if p.is_file()
    ]
    if any(marker(path) != identity for path, identity in input_markers.items()):
        raise ValueError("training input files changed during native execution")
    end = time.perf_counter()
    final = end - boundary
    return {
        "schema_version": RUN_SCHEMA,
        "plan": plan,
        "identity": identity,
        "environment": environment,
        "implementation_sha256": implementation,
        "cache_regime": cache,
        "memory_allocator": allocator,
        "observed_model": observed,
        "updates": updates,
        "timing": {
            "setup_seconds": setup,
            "finalization_seconds": final,
            "total_seconds": end - start,
        },
        **peaks,
        "saved_files": files,
        "output_dir": str(destination),
        "scope": "Cached latent/text-feature training only; optional imports/content hashing included; report writing and caller process startup excluded; final save excludes optimizer states.",
    }
