"""Resolve finite local framework inputs and inspect child package metadata."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess

from ._finetune_workflow_store import _file, _source_identity
from .structured_io import load_mapping


def _prepare_inputs(config_path, framework, *, workspace=None):
    """Resolve supported local sources before moving configs into the workspace."""
    source = Path(config_path).expanduser().resolve()
    workspace = (
        Path(workspace).expanduser().resolve() if workspace is not None else None
    )
    raw = load_mapping(source)
    config = deepcopy(raw)
    inputs = {str(source): _file(source)}

    def local(value, *, existing=True, record=True):
        if not isinstance(value, str) or not value or "://" in value:
            raise ValueError("calibration workflows require explicit local input paths")
        path = Path(value).expanduser()
        path = (source.parent / path if not path.is_absolute() else path).resolve()
        if existing and not path.exists():
            raise ValueError(f"local calibration input does not exist: {path}")
        if record and path.exists():
            if (
                workspace is not None
                and path.is_dir()
                and workspace.is_relative_to(path)
            ):
                raise ValueError(
                    f"workspace must be outside recorded input directories: {path}"
                )
            files = [path] if path.is_file() else sorted(path.rglob("*"))
            for item in files:
                if item.is_file():
                    inputs[str(item.resolve())] = _file(item)
                elif not item.is_dir():
                    raise ValueError(
                        f"calibration input is not an ordinary file: {item}"
                    )
        return str(path)

    model_field = "base_model" if framework == "axolotl" else "model_name_or_path"
    config[model_field] = local(config.get(model_field))
    for field in ("base_model_config", "tokenizer_config"):
        if config.get(field):
            config[field] = local(config[field])
    for field in ("cache_dir", "tokenized_path", "dataset_prepared_path"):
        if config.get(field):
            config[field] = local(config[field], existing=False, record=False)
    if framework == "huggingface":
        config["train_dataset"] = local(config.get("train_dataset"))
        if config.get("eval_dataset"):
            config["eval_dataset"] = local(config["eval_dataset"])
    elif framework == "llamafactory":
        directory = Path(local(config.get("dataset_dir", "data"), record=False))
        config["dataset_dir"] = str(directory)
        registry_path = directory / "dataset_info.json"
        registry = load_mapping(registry_path)
        inputs[str(registry_path)] = _file(registry_path)
        for field in ("dataset", "eval_dataset"):
            names = config.get(field, [])
            names = (
                [value.strip() for value in names.split(",")]
                if isinstance(names, str)
                else names
            )
            if not isinstance(names, list) or (field == "dataset" and not names):
                raise ValueError(
                    "LLaMA-Factory workflow requires registered local datasets"
                )
            for name in names:
                entry = registry.get(name)
                if (
                    not isinstance(entry, dict)
                    or not entry.get("file_name")
                    or any(
                        entry.get(key)
                        for key in (
                            "hf_hub_url",
                            "ms_hub_url",
                            "om_hub_url",
                            "script_url",
                        )
                    )
                ):
                    raise ValueError(
                        "LLaMA-Factory workflow requires local file_name registry entries"
                    )
                local(str(directory / entry["file_name"]))
        if config.get("media_dir"):
            raise ValueError(
                "calibration workflows support text data without media_dir"
            )
    else:

        def files(value):
            if isinstance(value, list):
                return [files(item) for item in value]
            if isinstance(value, dict):
                return {key: files(item) for key, item in value.items()}
            return local(value)

        for field in ("datasets", "test_datasets"):
            entries = config.get(field, [])
            if not isinstance(entries, list) or (field == "datasets" and not entries):
                raise ValueError("Axolotl workflow requires local dataset entries")
            for entry in entries:
                if not isinstance(entry, dict):
                    raise ValueError("Axolotl datasets must contain mappings")
                if entry.get("path") in {"json", "csv", "text", "parquet", "arrow"}:
                    if not entry.get("data_files"):
                        raise ValueError("local dataset loaders require data_files")
                else:
                    entry["path"] = local(entry.get("path"))
                if "data_files" in entry:
                    entry["data_files"] = files(entry["data_files"])
                template = entry.get("chat_template_jinja")
                if isinstance(template, str) and not any(
                    marker in template for marker in ("{{", "{%", "{#", "\n")
                ):
                    entry["chat_template_jinja"] = local(template)
        if config.get("pretraining_dataset"):
            raise ValueError(
                "calibration workflows currently support supervised datasets only"
            )
    if workspace is not None:
        # Native prepared-dataset keys need not include the raw file contents.
        # A fresh workspace binds these derived files to its immutable input
        # hashes; resume rejects changed inputs before another child starts.
        if framework == "axolotl":
            config["dataset_prepared_path"] = str(workspace / "prepared" / "axolotl")
        elif framework == "llamafactory" and config.get("tokenized_path"):
            config["tokenized_path"] = str(workspace / "prepared" / "llamafactory")
    return config, inputs


def _runtime_identity(executable, framework, environment):
    """Read child interpreter/package metadata without importing CUDA frameworks."""
    script = """import importlib.metadata as m, importlib.util, json, pathlib, sys
packages = ["torch", "transformers", "accelerate", "peft", "datasets", "trl", "bitsandbytes", "triton", sys.argv[1]]
versions = {}
for name in packages:
    try:
        versions[name] = m.version(name)
    except m.PackageNotFoundError:
        versions[name] = "not-installed"
spec = importlib.util.find_spec("fakegpu")
print(json.dumps({"python_version": sys.version, "prefix": sys.prefix, "versions": versions, "package_path": spec.origin if spec else None}))
"""
    result = subprocess.run(
        [executable, "-c", script, framework],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    identity = json.loads(result.stdout)
    if not identity["package_path"]:
        raise ValueError("native interpreter cannot locate the fakegpu package")
    root = Path(identity["package_path"]).parent
    identity["source"] = {
        path.relative_to(root).as_posix(): _file(path)
        for path in sorted(root.rglob("*.py"))
    }
    if identity["source"] != _source_identity():
        raise ValueError("native interpreter loads a different fakegpu implementation")
    return identity
