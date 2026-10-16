from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from dataclasses import field as dataclass_field
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse


CURRENT_COMPUTE_CAPABILITY_SOURCE = "https://developer.nvidia.com/cuda/gpus"
LEGACY_COMPUTE_CAPABILITY_SOURCE = "https://developer.nvidia.com/cuda/gpus/legacy"

_SUPPORTED_ARCHITECTURES = {
    "maxwell",
    "pascal",
    "volta",
    "turing",
    "ampere",
    "ada",
    "hopper",
    "blackwell",
}
_SUPPORTED_MEMORY_KINDS = {"dedicated", "unified", "synthetic"}
_SUPPORTED_PROFILE_STATUSES = {"measured", "reference", "synthetic"}
_SUPPORTED_SEGMENTS = {
    "consumer",
    "datacenter",
    "workstation",
    "embedded",
    "test",
    "custom",
}
_SUPPORTED_DATA_TYPES = {
    "fp32",
    "fp16",
    "bf16",
    "tf32",
    "int8",
    "int4",
    "fp8",
    "fp4",
}
_REQUIRED_INTEGER_FIELDS = (
    "compute_major",
    "compute_minor",
    "memory_bytes",
    "sm_count",
    "memory_bus_width_bits",
    "core_clock_mhz",
    "memory_clock_mhz",
    "l2_cache_bytes",
    "shared_mem_per_sm",
    "shared_mem_per_block",
    "shared_mem_per_block_optin",
    "regs_per_block",
    "regs_per_multiprocessor",
    "max_threads_per_multiprocessor",
    "max_blocks_per_multiprocessor",
    "typical_power_usage_mw",
    "max_power_limit_mw",
    "pci_device_id",
)
# Zero is an explicit unknown for public reference specifications. CUDA
# resource limits and memory capacity are still required by the runtime.
_REFERENCE_SPEC_FIELDS = (
    "sm_count",
    "memory_bus_width_bits",
    "core_clock_mhz",
    "memory_clock_mhz",
    "l2_cache_bytes",
    "typical_power_usage_mw",
    "max_power_limit_mw",
    "pci_device_id",
)
# Optional public specifications. Unlike the fields above, an absent key means
# unknown; an explicit 0 for NVLink or MIG means the product has none.
_OPTIONAL_INTEGER_FIELDS = (
    "base_clock_mhz",
    "pcie_generation",
    "pcie_lanes",
    "nvlink_generation",
    "nvlink_links",
    "mig_max_instances",
    "min_power_limit_mw",
)
_OPTIONAL_NUMBER_FIELDS = (
    "memory_bandwidth_gbps",
    "nvlink_bandwidth_gbps",
    "c2c_bandwidth_gbps",
)
# Dense peak rates per GPU in TFLOPS (TOPS for integers) at the boost clock.
# Tensor FP16/BF16/FP8 rates assume FP32 accumulation.
THROUGHPUT_FIELDS = (
    "fp64_tflops",
    "fp32_tflops",
    "fp16_tflops",
    "bf16_tflops",
    "tensor_fp64_tflops",
    "tensor_tf32_tflops",
    "tensor_fp16_tflops",
    "tensor_bf16_tflops",
    "tensor_fp8_tflops",
    "tensor_fp6_tflops",
    "tensor_fp4_tflops",
    "tensor_int8_tops",
    "tensor_int4_tops",
)
THROUGHPUT_BASES = {"throughput_published": "published", "throughput_derived": "derived"}
_MEMORY_TYPES = {
    "GDDR5", "GDDR5X", "GDDR6", "GDDR6X", "GDDR7",
    "HBM2", "HBM2e", "HBM3", "HBM3e",
    "LPDDR4", "LPDDR4X", "LPDDR5", "LPDDR5X", "DDR3", "DDR4",
}
_ECC_VALUES = {"supported", "unsupported"}
_FORM_FACTORS = {"sxm", "pcie", "superchip", "laptop", "embedded"}


class ProfileCatalogError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class GpuProfile:
    id: str
    name: str
    torch_name: str
    official_model: str
    architecture: str
    segment: str
    profile_path: str
    compute_major: int
    compute_minor: int
    memory_bytes: int
    sm_count: int
    memory_bus_width_bits: int
    core_clock_mhz: int
    memory_clock_mhz: int
    l2_cache_bytes: int
    shared_mem_per_sm: int
    shared_mem_per_block: int
    shared_mem_per_block_optin: int
    regs_per_block: int
    regs_per_multiprocessor: int
    max_threads_per_multiprocessor: int
    max_blocks_per_multiprocessor: int
    typical_power_usage_mw: int
    max_power_limit_mw: int
    pci_device_id: int
    supported_types: tuple[str, ...]
    memory_kind: str
    profile_status: str
    compute_capability_source: str
    spec_source: str
    notes: str
    base_clock_mhz: int | None = None
    memory_type: str = ""
    memory_bandwidth_gbps: float | None = None
    ecc: str = ""
    form_factor: str = ""
    pcie_generation: int | None = None
    pcie_lanes: int | None = None
    nvlink_generation: int | None = None
    nvlink_links: int | None = None
    nvlink_bandwidth_gbps: float | None = None
    c2c_bandwidth_gbps: float | None = None
    mig_max_instances: int | None = None
    min_power_limit_mw: int | None = None
    throughput: dict[str, float] = dataclass_field(default_factory=dict)
    throughput_basis: dict[str, str] = dataclass_field(default_factory=dict)
    throughput_source: str = ""

    @property
    def compute_capability(self) -> tuple[int, int]:
        return (self.compute_major, self.compute_minor)

    @property
    def compute_capability_text(self) -> str:
        return f"{self.compute_major}.{self.compute_minor}"

    @property
    def compiler_target(self) -> str:
        return f"sm_{self.compute_major}{self.compute_minor}"

    @property
    def unavailable_fields(self) -> tuple[str, ...]:
        """Public specifications not established for this reference profile."""

        return tuple(key for key in _REFERENCE_SPEC_FIELDS if getattr(self, key) == 0)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["compute_capability"] = self.compute_capability_text
        payload["compiler_target"] = self.compiler_target
        payload["supported_types"] = list(self.supported_types)
        payload["unavailable_fields"] = list(self.unavailable_fields)
        return payload


@dataclass(frozen=True, slots=True)
class CatalogValidation:
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


def architecture_for_compute_capability(major: int, minor: int) -> str:
    """Map a CUDA compute capability to its NVIDIA hardware architecture."""

    capability = (major, minor)
    if capability in {(10, 0), (10, 3), (11, 0), (12, 0), (12, 1)}:
        return "blackwell"
    if capability == (9, 0):
        return "hopper"
    if capability == (8, 9):
        return "ada"
    if capability in {(8, 0), (8, 6), (8, 7)}:
        return "ampere"
    if capability == (7, 5):
        return "turing"
    if capability in {(7, 0), (7, 2)}:
        return "volta"
    if capability in {(6, 0), (6, 1), (6, 2)}:
        return "pascal"
    if capability in {(5, 0), (5, 2), (5, 3)}:
        return "maxwell"
    return "unknown"


def profile_directory() -> Path:
    override = os.environ.get("FAKEGPU_PROFILE_DIR")
    candidates = []
    if override:
        candidates.append(Path(override))

    package_dir = Path(__file__).resolve().parent
    candidates.extend(
        (
            package_dir.parent / "profiles",
            package_dir / "_profiles",
        )
    )
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.rglob("*.yaml")):
            return candidate.resolve()
    searched = ", ".join(str(path) for path in candidates)
    raise ProfileCatalogError(f"FakeGPU profile directory not found; searched: {searched}")


def load_profiles(profile_dir: str | os.PathLike[str] | None = None) -> dict[str, GpuProfile]:
    resolved = Path(profile_dir).resolve() if profile_dir is not None else profile_directory()
    return dict(_load_profiles_cached(str(resolved)))


@lru_cache(maxsize=8)
def _load_profiles_cached(profile_dir: str) -> tuple[tuple[str, GpuProfile], ...]:
    root = Path(profile_dir)
    if not root.is_dir():
        raise ProfileCatalogError(f"profile directory does not exist: {root}")

    profiles: dict[str, GpuProfile] = {}
    for path in sorted(root.rglob("*.yaml")):
        profile = _profile_from_mapping(_parse_simple_yaml(path), path=path, root=root)
        if profile.id in profiles:
            raise ProfileCatalogError(f"duplicate profile id {profile.id!r}: {path}")
        if path.stem != profile.id:
            raise ProfileCatalogError(
                f"profile filename/id mismatch: {path.name!r} contains id {profile.id!r}"
            )
        profiles[profile.id] = profile

    if not profiles:
        raise ProfileCatalogError(f"no YAML profiles found in {root}")
    return tuple(sorted(profiles.items()))


def get_profile(
    profile_id: str,
    *,
    profiles: Mapping[str, GpuProfile] | None = None,
) -> GpuProfile:
    catalog = profiles if profiles is not None else load_profiles()
    normalized = profile_id.strip().lower()
    try:
        return catalog[normalized]
    except KeyError as exc:
        choices = ", ".join(sorted(catalog))
        raise ProfileCatalogError(
            f"unknown GPU profile {profile_id!r}; available profiles: {choices}"
        ) from exc


def official_compute_capabilities() -> dict[str, str]:
    """Return an isolated copy of the cached NVIDIA model snapshot."""

    return dict(_official_compute_capabilities_cached())


@lru_cache(maxsize=1)
def _official_compute_capabilities_cached() -> tuple[tuple[str, str], ...]:
    path = Path(__file__).resolve().parent / "data" / "nvidia_compute_capabilities.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProfileCatalogError(f"cannot read NVIDIA compute-capability snapshot: {path}") from exc

    models = payload.get("models")
    if not isinstance(models, dict):
        raise ProfileCatalogError(f"invalid NVIDIA compute-capability snapshot: {path}")
    return tuple((str(name), str(cc)) for name, cc in models.items())


def validate_catalog(
    profiles: Mapping[str, GpuProfile] | None = None,
    *,
    official_models: Mapping[str, str] | None = None,
) -> CatalogValidation:
    catalog = profiles if profiles is not None else load_profiles()
    model_cc = official_models if official_models is not None else official_compute_capabilities()
    errors: list[str] = []
    warnings: list[str] = []

    for profile in catalog.values():
        prefix = f"{profile.id}:"
        expected_arch = architecture_for_compute_capability(
            profile.compute_major,
            profile.compute_minor,
        )
        if expected_arch == "unknown":
            errors.append(
                f"{prefix} unsupported compute capability {profile.compute_capability_text}"
            )
        elif profile.architecture != expected_arch:
            errors.append(
                f"{prefix} architecture {profile.architecture!r} does not match "
                f"compute capability {profile.compute_capability_text} ({expected_arch})"
            )

        if profile.architecture not in _SUPPORTED_ARCHITECTURES:
            errors.append(f"{prefix} unknown architecture {profile.architecture!r}")
        if profile.segment not in _SUPPORTED_SEGMENTS:
            errors.append(f"{prefix} unknown segment {profile.segment!r}")
        if profile.memory_kind not in _SUPPORTED_MEMORY_KINDS:
            errors.append(f"{prefix} unknown memory_kind {profile.memory_kind!r}")
        if profile.profile_status not in _SUPPORTED_PROFILE_STATUSES:
            errors.append(f"{prefix} unknown profile_status {profile.profile_status!r}")
        if profile.memory_bytes <= 0:
            errors.append(f"{prefix} memory_bytes must be positive")
        if profile.sm_count == 0 and profile.profile_status != "reference":
            errors.append(f"{prefix} sm_count must be positive for non-reference profiles")
        for field in _REQUIRED_INTEGER_FIELDS:
            if getattr(profile, field) < 0:
                errors.append(f"{prefix} {field} must not be negative")
        if not profile.supported_types:
            errors.append(f"{prefix} supported_types must not be empty")
        elif "fp32" not in profile.supported_types:
            errors.append(f"{prefix} supported_types must include fp32")
        unknown_types = sorted(set(profile.supported_types) - _SUPPORTED_DATA_TYPES)
        if unknown_types:
            errors.append(
                f"{prefix} unknown supported_types: {', '.join(unknown_types)}"
            )
        if len(profile.supported_types) != len(set(profile.supported_types)):
            errors.append(f"{prefix} supported_types contains duplicates")
        if "fp8" in profile.supported_types and profile.architecture not in {
            "ada",
            "hopper",
            "blackwell",
        }:
            errors.append(
                f"{prefix} fp8 is not valid for architecture {profile.architecture}"
            )
        if "fp4" in profile.supported_types and profile.architecture != "blackwell":
            errors.append(
                f"{prefix} fp4 is not valid for architecture {profile.architecture}"
            )
        if profile.shared_mem_per_block > profile.shared_mem_per_block_optin:
            errors.append(
                f"{prefix} shared_mem_per_block exceeds shared_mem_per_block_optin"
            )
        if profile.shared_mem_per_block_optin > profile.shared_mem_per_sm:
            errors.append(
                f"{prefix} shared_mem_per_block_optin exceeds shared_mem_per_sm"
            )

        if profile.profile_status != "synthetic":
            official_cc = model_cc.get(profile.official_model)
            if official_cc is None:
                warnings.append(
                    f"{prefix} official model {profile.official_model!r} is not in the "
                    "cached NVIDIA compute-capability tables"
                )
            elif official_cc != profile.compute_capability_text:
                errors.append(
                    f"{prefix} compute capability {profile.compute_capability_text} "
                    f"does not match NVIDIA table value {official_cc}"
                )

        errors.extend(_optional_spec_errors(profile, prefix))

        for field in ("compute_capability_source", "spec_source"):
            value = getattr(profile, field)
            hostname = (urlparse(value).hostname or "").lower()
            official_host = any(
                hostname == domain or hostname.endswith(f".{domain}")
                for domain in ("nvidia.com", "nvidia.cn")
            )
            if profile.profile_status != "synthetic" and not official_host:
                warnings.append(f"{prefix} {field} is not an NVIDIA URL")

    return CatalogValidation(errors=tuple(errors), warnings=tuple(warnings))


def _optional_spec_errors(profile: GpuProfile, prefix: str) -> list[str]:
    errors = []
    if profile.memory_type and profile.memory_type not in _MEMORY_TYPES:
        errors.append(f"{prefix} unknown memory_type {profile.memory_type!r}")
    if profile.ecc and profile.ecc not in _ECC_VALUES:
        errors.append(f"{prefix} ecc must be supported or unsupported")
    if profile.form_factor and profile.form_factor not in _FORM_FACTORS:
        errors.append(f"{prefix} unknown form_factor {profile.form_factor!r}")
    numbers = {key: getattr(profile, key) for key in _OPTIONAL_INTEGER_FIELDS + _OPTIONAL_NUMBER_FIELDS}
    numbers.update(profile.throughput)
    for key, value in numbers.items():
        if value is not None and (value < 0 or value != value):
            errors.append(f"{prefix} {key} must not be negative")
    if profile.nvlink_generation == 0 and (profile.nvlink_links or profile.nvlink_bandwidth_gbps):
        errors.append(f"{prefix} nvlink_generation 0 (none) contradicts NVLink links or bandwidth")
    if profile.base_clock_mhz and profile.core_clock_mhz and profile.base_clock_mhz > profile.core_clock_mhz:
        errors.append(f"{prefix} base_clock_mhz exceeds the boost clock core_clock_mhz")
    if (
        profile.min_power_limit_mw and profile.max_power_limit_mw
        and profile.min_power_limit_mw > profile.max_power_limit_mw
    ):
        errors.append(f"{prefix} min_power_limit_mw exceeds max_power_limit_mw")
    if profile.throughput_basis and "published" in profile.throughput_basis.values() and not profile.throughput_source:
        errors.append(f"{prefix} published throughput needs throughput_source")
    return errors


def catalog_summary(
    profiles: Mapping[str, GpuProfile] | None = None,
) -> dict[str, Any]:
    catalog = profiles if profiles is not None else load_profiles()
    architecture_counts: dict[str, int] = {}
    segment_counts: dict[str, int] = {}
    capabilities: set[str] = set()
    for profile in catalog.values():
        architecture_counts[profile.architecture] = (
            architecture_counts.get(profile.architecture, 0) + 1
        )
        segment_counts[profile.segment] = segment_counts.get(profile.segment, 0) + 1
        capabilities.add(profile.compute_capability_text)
    return {
        "profile_count": len(catalog),
        "architectures": dict(sorted(architecture_counts.items())),
        "segments": dict(sorted(segment_counts.items())),
        "compute_capabilities": sorted(
            capabilities,
            key=lambda value: tuple(int(part) for part in value.split(".", 1)),
        ),
    }


def _profile_from_mapping(
    raw: Mapping[str, object],
    *,
    path: Path,
    root: Path,
) -> GpuProfile:
    def required_text(key: str) -> str:
        value = raw.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ProfileCatalogError(f"{path}: missing or invalid {key!r}")
        return value.strip()

    def optional_text(key: str, default: str = "") -> str:
        value = raw.get(key, default)
        if not isinstance(value, str):
            raise ProfileCatalogError(f"{path}: {key!r} must be a scalar")
        return value.strip()

    integers: dict[str, int] = {}
    for key in _REQUIRED_INTEGER_FIELDS:
        value = raw.get(key)
        if not isinstance(value, str):
            raise ProfileCatalogError(f"{path}: missing or invalid integer {key!r}")
        try:
            integers[key] = int(value, 0)
        except ValueError as exc:
            raise ProfileCatalogError(f"{path}: invalid integer {key}={value!r}") from exc

    optional: dict[str, Any] = {}
    for key in _OPTIONAL_INTEGER_FIELDS + _OPTIONAL_NUMBER_FIELDS + THROUGHPUT_FIELDS:
        value = raw.get(key)
        if value is None:
            continue
        if not isinstance(value, str):
            raise ProfileCatalogError(f"{path}: {key!r} must be a scalar")
        try:
            optional[key] = int(value, 0) if key in _OPTIONAL_INTEGER_FIELDS else float(value)
        except ValueError as exc:
            raise ProfileCatalogError(f"{path}: invalid number {key}={value!r}") from exc
    throughput = {key: optional.pop(key) for key in THROUGHPUT_FIELDS if key in optional}
    throughput_basis: dict[str, str] = {}
    for list_key, basis in THROUGHPUT_BASES.items():
        names = raw.get(list_key, [])
        if not isinstance(names, list):
            raise ProfileCatalogError(f"{path}: {list_key} must be a list")
        for name in names:
            if name not in throughput:
                raise ProfileCatalogError(f"{path}: {list_key} names {name!r} without a value")
            if name in throughput_basis:
                raise ProfileCatalogError(f"{path}: {name!r} is both published and derived")
            throughput_basis[name] = basis
    unlabeled = sorted(set(throughput) - set(throughput_basis))
    if unlabeled:
        raise ProfileCatalogError(
            f"{path}: throughput values need a throughput_published or throughput_derived entry: "
            + ", ".join(unlabeled)
        )

    supported_types_raw = raw.get("supported_types")
    if not isinstance(supported_types_raw, list) or not all(
        isinstance(item, str) for item in supported_types_raw
    ):
        raise ProfileCatalogError(f"{path}: supported_types must be a string list")

    try:
        relative_path = path.relative_to(root)
    except ValueError as exc:
        raise ProfileCatalogError(f"profile path is outside catalog root: {path}") from exc

    parts = relative_path.parts
    if len(parts) == 1:
        directory_architecture = ""
        segment = "custom"
    elif len(parts) == 3:
        directory_architecture = parts[0].lower()
        segment = parts[1].lower()
    else:
        raise ProfileCatalogError(
            f"{path}: expected <architecture>/<segment>/<profile>.yaml"
        )

    name = required_text("name")
    architecture = required_text("architecture").lower()
    if directory_architecture and directory_architecture != architecture:
        raise ProfileCatalogError(
            f"{path}: directory architecture {directory_architecture!r} does not "
            f"match profile architecture {architecture!r}"
        )

    return GpuProfile(
        id=required_text("id").lower(),
        name=name,
        torch_name=optional_text("torch_name", name),
        official_model=required_text("official_model"),
        architecture=architecture,
        segment=segment,
        profile_path=relative_path.as_posix(),
        supported_types=tuple(str(item).lower() for item in supported_types_raw),
        memory_kind=optional_text("memory_kind", "dedicated").lower(),
        profile_status=optional_text("profile_status", "reference").lower(),
        compute_capability_source=optional_text("compute_capability_source"),
        spec_source=optional_text("spec_source"),
        notes=optional_text("notes"),
        memory_type=optional_text("memory_type"),
        ecc=optional_text("ecc").lower(),
        form_factor=optional_text("form_factor").lower(),
        throughput=throughput,
        throughput_basis=throughput_basis,
        throughput_source=optional_text("throughput_source"),
        **integers,
        **optional,
    )


def _parse_simple_yaml(path: Path) -> dict[str, object]:
    scalars: dict[str, str] = {}
    lists: dict[str, list[str]] = {}
    current_list: str | None = None

    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("- "):
            if current_list is None:
                raise ProfileCatalogError(
                    f"{path}:{line_number}: list item without a preceding key"
                )
            lists.setdefault(current_list, []).append(line[2:].strip())
            continue
        if ":" not in line:
            raise ProfileCatalogError(f"{path}:{line_number}: missing ':'")
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value:
            scalars[key] = value
            current_list = None
        else:
            current_list = key
            lists.setdefault(key, [])

    result: dict[str, object] = dict(scalars)
    result.update(lists)
    return result
