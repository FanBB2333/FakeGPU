"""The ``--query-*`` field catalog and how one field is read from a state.

Split out of ``smi`` unchanged. Every queryable GPU, process, and runtime
field is declared here once — its path into the normalized inventory, its
CSV header, and how its value renders — so ``nvidia-smi``-style queries and
the query report stay in step with one schema.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from ._smi_state import NATIVE_ACTIVITY_COUNTERS


QUERY_SCHEMA_VERSION = "fakegpu.smi_query.v1"


@dataclass(frozen=True, slots=True)
class QueryField:
    # None: an nvidia-smi field FakeGPU does not model; it reads as missing.
    path: str | None
    unit: str | None = None
    divisor: float = 1.0
    precision: int = 0
    # The field nvidia-smi names in the CSV header when this one is an alias.
    header: str | None = None
    # CSV prints exactly `precision` decimals, as nvidia-smi prints watts
    # ("700.00 W" on an H100 SXM).
    fixed: bool = False
    # Extra text for --help-query-*, such as where a renamed value moved.
    note: str | None = None


GPU_QUERY_FIELDS: dict[str, QueryField] = {
    "timestamp": QueryField("timestamp"),
    "host": QueryField("host"),
    "count": QueryField("count"),
    "hostname": QueryField("host"),
    "index": QueryField("index"),
    "name": QueryField("name"),
    "uuid": QueryField("uuid"),
    "pci.bus_id": QueryField("pci_bus_id"),
    "topology.source": QueryField("topology.source"),
    "nvlink.active_links": QueryField(
        "topology.nvlink.active_links"
    ),
    "nvlink.peer_count": QueryField(
        "topology.nvlink.peer_count"
    ),
    "nvlink.bandwidth": QueryField(
        "topology.nvlink.aggregate_bandwidth_gbps",
        unit="GB/s",
        precision=1,
    ),
    "platform": QueryField("topology.platform"),
    "slot.type": QueryField("topology.slot.type"),
    "numa.node": QueryField("topology.slot.numa_node"),
    "pcie.link.gen.current": QueryField("topology.pcie.generation"),
    "pcie.link.gen.gpucurrent": QueryField("topology.pcie.generation"),
    "pcie.link.gen.max": QueryField("topology.pcie.max_generation"),
    "pcie.link.gen.gpumax": QueryField(
        "topology.pcie.gpu_max_generation"
    ),
    "pcie.link.width.current": QueryField("topology.pcie.width"),
    "pcie.link.width.max": QueryField("topology.pcie.max_width"),
    "host_link.bandwidth": QueryField(
        "topology.host_link.bandwidth_gbps",
        unit="GB/s",
        precision=1,
    ),
    "pcie.throughput.tx": QueryField(
        "transfer.links.pcie.tx_bytes_per_second",
        unit="KB/s",
        divisor=1024,
    ),
    "pcie.throughput.rx": QueryField(
        "transfer.links.pcie.rx_bytes_per_second",
        unit="KB/s",
        divisor=1024,
    ),
    "nvlink.throughput.tx": QueryField(
        "transfer.links.nvlink.tx_bytes_per_second",
        unit="KB/s",
        divisor=1024,
    ),
    "nvlink.throughput.rx": QueryField(
        "transfer.links.nvlink.rx_bytes_per_second",
        unit="KB/s",
        divisor=1024,
    ),
    "c2c.throughput.tx": QueryField(
        "transfer.links.c2c.tx_bytes_per_second",
        unit="KB/s",
        divisor=1024,
    ),
    "c2c.throughput.rx": QueryField(
        "transfer.links.c2c.rx_bytes_per_second",
        unit="KB/s",
        divisor=1024,
    ),
    "copy.h2d": QueryField(
        "transfer.copies.host_to_device.bytes",
        unit="MiB",
        divisor=2**20,
        precision=1,
    ),
    "copy.d2h": QueryField(
        "transfer.copies.device_to_host.bytes",
        unit="MiB",
        divisor=2**20,
        precision=1,
    ),
    "copy.peer": QueryField(
        "transfer.copies.peer_send.bytes",
        unit="MiB",
        divisor=2**20,
        precision=1,
    ),
    "copy.modeled_time": QueryField(
        "transfer.modeled_seconds",
        unit="s",
        precision=6,
    ),
    "health.status": QueryField("health.status"),
    "health.hardware": QueryField("health.hardware_health"),
    "health.max_severity": QueryField("health.max_severity"),
    "health.event_count": QueryField("health.event_count"),
    "health.event_types": QueryField("health.event_types_total"),
    "mig.mode": QueryField("mig.mode"),
    "mig.instance_count": QueryField("mig.instance_count"),
    "mig.allocated_memory": QueryField(
        "mig.allocated_memory_bytes",
        unit="MiB",
        divisor=2**20,
    ),
    "mig.unallocated_memory": QueryField(
        "mig.unallocated_memory_bytes",
        unit="MiB",
        divisor=2**20,
    ),
    "driver_version": QueryField("software.driver_version"),
    "cuda_version": QueryField("software.cuda_version"),
    "profile.id": QueryField("profile_id"),
    "profile.status": QueryField("profile.profile_status"),
    "architecture": QueryField("architecture"),
    "compute_cap": QueryField("compute_capability"),
    "compiler_target": QueryField("compiler_target"),
    "memory.total": QueryField(
        "memory.total_bytes",
        unit="MiB",
        divisor=2**20,
    ),
    "memory.used": QueryField(
        "memory.used_bytes",
        unit="MiB",
        divisor=2**20,
    ),
    "memory.free": QueryField(
        "memory.free_bytes",
        unit="MiB",
        divisor=2**20,
    ),
    "memory.tracked": QueryField(
        "memory.tracked_bytes",
        unit="MiB",
        divisor=2**20,
    ),
    # nvidia-smi's memory.reserved is "Total memory reserved by the NVIDIA
    # driver and firmware" (total = reserved + used + free), which FakeGPU
    # does not model. The caching allocator's reserved bytes, which this
    # field returned before, are memory.allocator_reserved.
    "memory.reserved": QueryField(
        None,
        unit="MiB",
        note=(
            "driver and firmware reserve; FakeGPU's allocator value moved "
            "to memory.allocator_reserved"
        ),
    ),
    "memory.allocator_reserved": QueryField(
        "memory.reserved_bytes",
        unit="MiB",
        divisor=2**20,
    ),
    "memory.peak": QueryField(
        "memory.reported_peak_bytes",
        unit="MiB",
        divisor=2**20,
    ),
    "memory.inactive_split": QueryField(
        "memory.inactive_split_bytes",
        unit="MiB",
        divisor=2**20,
    ),
    "memory.utilization": QueryField(
        "memory.utilization_percent",
        unit="%",
        precision=3,
    ),
    "memory.headroom": QueryField(
        "memory.headroom_bytes",
        unit="MiB",
        divisor=2**20,
    ),
    # nvidia-smi's default power ceiling. The catalog's reference ceiling,
    # not its typical workload power.
    "power.default_limit": QueryField(
        "profile.max_power_limit_mw",
        unit="W",
        divisor=1000,
        precision=2,
        fixed=True,
    ),
    "power.min_limit": QueryField(
        "profile.min_power_limit_mw",
        unit="W",
        divisor=1000,
        precision=2,
        fixed=True,
    ),
    "power.max_limit": QueryField(
        "profile.max_power_limit_mw",
        unit="W",
        divisor=1000,
        precision=2,
        fixed=True,
    ),
    "sm_count": QueryField("profile.sm_count"),
    "clocks.current.sm": QueryField(
        "profile.core_clock_mhz",
        unit="MHz",
    ),
    "clocks.current.memory": QueryField(
        "profile.memory_clock_mhz",
        unit="MHz",
    ),
    "memory.bus_width": QueryField(
        "profile.memory_bus_width_bits",
        unit="bit",
    ),
    "memory.kind": QueryField("profile.memory_kind"),
    "supported_types": QueryField("profile.supported_types"),
    "allocator.model": QueryField("allocator.model"),
    "allocator.segments": QueryField("allocator.segment_count"),
    "native.io_calls": QueryField("native_activity.io_calls"),
    "native.io_bytes": QueryField(
        "native_activity.io_bytes",
        unit="MiB",
        divisor=2**20,
    ),
    "native.kernel_launches": QueryField(
        "native_activity.kernel_launches"
    ),
    "native.gemm_calls": QueryField("native_activity.gemm_calls"),
    "native.gemm_flops": QueryField("native_activity.gemm_flops"),
    "native.compatibility_events": QueryField(
        "native_activity.compatibility_events"
    ),
    "native.unsupported_api_calls": QueryField(
        "native_activity.unsupported_api_calls"
    ),
    "processes": QueryField("process_count"),
    "fakegpu.version": QueryField("fakegpu.version"),
    "runtime": QueryField("fakegpu.runtime"),
    "runtime.backend": QueryField("fakegpu.backend"),
    "runtime.mode": QueryField("fakegpu.mode"),
    "runtime.stage": QueryField("stages"),
    "tracking.confidence": QueryField("tracking_confidence"),
    "state.status": QueryField("state.status"),
    "state.age": QueryField(
        "state.max_age_seconds",
        unit="s",
        precision=3,
    ),
}


PROCESS_QUERY_FIELDS: dict[str, QueryField] = {
    "timestamp": QueryField("timestamp"),
    "host": QueryField("host"),
    "gpu_index": QueryField("gpu_index"),
    "gpu_name": QueryField("gpu_name"),
    "gpu_bus_id": QueryField("pci_bus_id"),
    "gpu_uuid": QueryField("gpu_uuid"),
    "pci.bus_id": QueryField("pci_bus_id"),
    "profile.id": QueryField("profile_id"),
    "pid": QueryField("pid"),
    "process_name": QueryField("process_name"),
    "used_gpu_memory": QueryField(
        "memory.used_bytes",
        unit="MiB",
        divisor=2**20,
    ),
    "peak_gpu_memory": QueryField(
        "memory.reported_peak_bytes",
        unit="MiB",
        divisor=2**20,
    ),
    "tracked_gpu_memory": QueryField(
        "memory.tracked_bytes",
        unit="MiB",
        divisor=2**20,
    ),
    "reserved_gpu_memory": QueryField(
        "memory.reserved_bytes",
        unit="MiB",
        divisor=2**20,
    ),
    "stage": QueryField("stage"),
    "status": QueryField("status"),
    "state.age": QueryField(
        "age_seconds",
        unit="s",
        precision=3,
    ),
    "tracking_confidence": QueryField("tracking_confidence"),
    "allocator.model": QueryField("allocator_model"),
    "dispatch.calls": QueryField("dispatch_tracking.operator_calls"),
    "native.io_calls": QueryField("native_activity.io_calls"),
    "native.io_bytes": QueryField(
        "native_activity.io_bytes",
        unit="MiB",
        divisor=2**20,
    ),
    "native.kernel_launches": QueryField(
        "native_activity.kernel_launches"
    ),
    "native.gemm_calls": QueryField("native_activity.gemm_calls"),
    "native.gemm_flops": QueryField("native_activity.gemm_flops"),
    "native.compatibility_events": QueryField(
        "native_activity.compatibility_events"
    ),
    "native.unsupported_api_calls": QueryField(
        "native_activity.unsupported_api_calls"
    ),
    "fakegpu.version": QueryField("fakegpu.version"),
    "runtime.backend": QueryField("fakegpu.backend"),
}


RUNTIME_QUERY_FIELDS: dict[str, QueryField] = {
    "timestamp": QueryField("timestamp"),
    "host": QueryField("host"),
    "pid": QueryField("pid"),
    "process_name": QueryField("process_name"),
    "stage": QueryField("stage"),
    "status": QueryField("status"),
    "state.age": QueryField(
        "age_seconds",
        unit="s",
        precision=3,
    ),
    "tracking.confidence": QueryField("tracking_confidence"),
    "allocator.model": QueryField("allocator_model"),
    "fakegpu.version": QueryField("fakegpu.version"),
    "runtime": QueryField("fakegpu.runtime"),
    "runtime.backend": QueryField("fakegpu.backend"),
    "runtime.mode": QueryField("fakegpu.mode"),
    "policy.oom": QueryField("fakegpu.oom_policy"),
    "policy.unsupported_api": QueryField(
        "fakegpu.unsupported_api_policy"
    ),
    "distributed.mode": QueryField("fakegpu.distributed_mode"),
    "tracking.memory": QueryField(
        "fakegpu.memory_tracking_enabled"
    ),
    "tracking.dispatch": QueryField(
        "fakegpu.dispatch_memory_tracking_enabled"
    ),
    "catalog.profiles": QueryField(
        "fakegpu.profile_catalog.profile_count"
    ),
    "catalog.native_groups": QueryField(
        "fakegpu.native_capabilities.group_count"
    ),
    "catalog.native_apis": QueryField(
        "fakegpu.native_capabilities.explicit_api_count"
    ),
    "catalog.policy_apis": QueryField(
        "fakegpu.native_capabilities.policy_enforced_api_count"
    ),
    "software.python": QueryField("software.python_version"),
    "software.python_implementation": QueryField(
        "software.python_implementation"
    ),
    "software.pytorch": QueryField("software.torch_version"),
    "software.cuda": QueryField("software.cuda_version"),
    "software.cuda_source": QueryField(
        "software.cuda_version_source"
    ),
    "software.driver": QueryField("software.driver_version"),
    "software.platform": QueryField("software.platform"),
    "dispatch.calls": QueryField("dispatch_tracking.operator_calls"),
    "dispatch.outputs": QueryField(
        "dispatch_tracking.output_tensors"
    ),
    "dispatch.allocations": QueryField(
        "dispatch_tracking.new_allocations"
    ),
    "dispatch.aliases": QueryField(
        "dispatch_tracking.alias_outputs"
    ),
    "dispatch.inaccessible": QueryField(
        "dispatch_tracking.inaccessible_outputs"
    ),
    "publisher.interval": QueryField(
        "publisher.interval_seconds",
        unit="s",
        precision=3,
    ),
    "publisher.runtime_overhead": QueryField(
        "publisher.runtime_overhead_bytes",
        unit="MiB",
        divisor=2**20,
    ),
    "publisher.attempted_writes": QueryField(
        "publisher.health.attempted_writes"
    ),
    "publisher.successful_writes": QueryField(
        "publisher.health.successful_writes"
    ),
    "publisher.failed_writes": QueryField(
        "publisher.health.failed_writes"
    ),
    "publisher.last_duration": QueryField(
        "publisher.health.last_duration_us",
        unit="us",
    ),
    "publisher.max_duration": QueryField(
        "publisher.health.max_duration_us",
        unit="us",
    ),
    "publisher.state_size": QueryField(
        "publisher.health.last_serialized_bytes",
        unit="KiB",
        divisor=2**10,
        precision=3,
    ),
    "publisher.last_error": QueryField(
        "publisher.health.last_error"
    ),
    "publisher.detail_limit": QueryField(
        "publisher.limits.detail_entries"
    ),
    "publisher.state_size_limit": QueryField(
        "publisher.limits.max_state_bytes",
        unit="KiB",
        divisor=2**10,
        precision=3,
    ),
}


# The rest of nvidia-smi 590.48.01's --help-query-gpu fields, by CSV header
# unit. FakeGPU does not model them; nvidia-smi accepts every one and prints
# [N/A] where a GPU reports no value, so they read as missing rather than
# failing scripts that ask for them.
_NVIDIA_GPU_FIELDS: dict[str | None, tuple[str, ...]] = {
    None: (
        "driver_version", "vgpu_driver_capability.heterogenous_multivGPU",
        "count", "serial", "pci.domain", "pci.bus", "pci.device",
        "pci.baseClass", "pci.subClass", "pci.device_id", "pci.sub_device_id",
        "vgpu_device_capability.fractional_multiVgpu",
        "vgpu_device_capability.heterogeneous_timeSlice_profile",
        "vgpu_device_capability.heterogeneous_timeSlice_sizes",
        "vgpu_device_capability.homogeneous_placements",
        "vgpu_device_capability.mig_timeSlicing",
        "vgpu_device_capability.mig_timeSlicing_mode",
        "pcie.link.gen.hostmax", "display_mode", "display_attached",
        "display_active", "persistence_mode", "addressing_mode",
        "accounting.mode", "accounting.buffer_size", "driver_model.current",
        "driver_model.pending", "vbios_version", "inforom.img", "inforom.oem",
        "inforom.ecc", "inforom.pwr", "inforom.checksum_validation",
        "gpu_recovery_action", "gom.current", "gom.pending", "pstate",
        "clocks_event_reasons.supported", "clocks_event_reasons.active",
        "clocks_event_reasons.gpu_idle",
        "clocks_event_reasons.applications_clocks_setting",
        "clocks_event_reasons.sw_power_cap", "clocks_event_reasons.hw_slowdown",
        "clocks_event_reasons.hw_thermal_slowdown",
        "clocks_event_reasons.hw_power_brake_slowdown",
        "clocks_event_reasons.sw_thermal_slowdown",
        "clocks_event_reasons.sync_boost", "compute_mode",
        "encoder.stats.sessionCount", "encoder.stats.averageFps",
        "encoder.stats.averageLatency", "dramEncryption.mode.current",
        "dramEncryption.mode.pending", "ecc.mode.current", "ecc.mode.pending",
        *(
            f"ecc.errors.{kind}.{scope}.{location}"
            for kind in ("corrected", "uncorrected")
            for scope in ("volatile", "aggregate")
            for location in (
                "device_memory", "dram", "register_file", "l1_cache",
                "l2_cache", "texture_memory", "cbu", "sram", "total",
            )
        ),
        "ecc.errors.uncorrected.volatile.sram.parity",
        "ecc.errors.uncorrected.volatile.sram.secded",
        "ecc.errors.uncorrected.aggregate.sram.parity",
        "ecc.errors.uncorrected.aggregate.sram.secded",
        "ecc.errors.uncorrected.aggregate.sram.thresholdExceeded",
        "ecc.errors.uncorrected.aggregate.sram.l2",
        "ecc.errors.uncorrected.aggregate.sram.sm",
        "ecc.errors.uncorrected.aggregate.sram.mcu",
        "ecc.errors.uncorrected.aggregate.sram.pcie",
        "ecc.errors.uncorrected.aggregate.sram.other",
        "retired_pages.single_bit_ecc.count", "retired_pages.double_bit.count",
        "retired_pages.pending", "remapped_rows.correctable",
        "remapped_rows.uncorrectable", "remapped_rows.pending",
        "remapped_rows.failure", "remapped_rows.histogram.max",
        "remapped_rows.histogram.high", "remapped_rows.histogram.partial",
        "remapped_rows.histogram.low", "remapped_rows.histogram.none",
        "temperature.gpu", "temperature.gpu.tlimit", "temperature.memory",
        "power.management",
        "power_smoothing.delayed_power_smoothing_supported",
        "power_smoothing.curr_profile.primary_floor_act_window_multiplier",
        "power_smoothing.curr_profile.primary_floor_tar_window_multiplier",
        "power_smoothing.admin_override.primary_floor_act_window_multiplier",
        "power_smoothing.admin_override.primary_floor_tar_window_multiplier",
        "mig.mode.current", "mig.mode.pending", "gsp.mode.current",
        "gsp.mode.default", "c2c.mode", "fabric.state", "fabric.status",
        "fabric.cliqueId", "fabric.clusterUuid", "fabric.health.summary",
        "fabric.health.bandwidth", "fabric.health.route_recovery_in_progress",
        "fabric.health.route_unhealthy",
        "fabric.health.access_timeout_recovery",
        "fabric.health.incorrect_configuration",
        "platform.chassis_serial_number", "platform.slot_number",
        "platform.tray_index", "platform.host_id", "platform.peer_type",
        "platform.module_id", "platform.gpu_fabric_guid",
    ),
    "%": (
        "fan.speed", "utilization.gpu", "utilization.memory",
        "utilization.encoder", "utilization.decoder", "utilization.jpeg",
        "utilization.ofa", "edpp_multiplier",
    ),
    "us": (
        "clocks_event_reasons_counters.sw_power_cap",
        "clocks_event_reasons_counters.sync_boost",
        "clocks_event_reasons_counters.sw_thermal_slowdown",
        "clocks_event_reasons_counters.hw_thermal_slowdown",
        "clocks_event_reasons_counters.hw_power_brake_slowdown",
    ),
    "MiB": (
        "protected_memory.total", "protected_memory.used",
        "protected_memory.free",
    ),
    "W": (
        "power.draw", "power.draw.average", "power.draw.instant",
        "power.limit", "enforced.power.limit", "power.min_limit",
        "module.power.draw.average", "module.power.draw.instant",
        "module.power.limit", "module.enforced.power.limit",
        "module.power.default_limit", "module.power.min_limit",
        "module.power.max_limit", "power_smoothing.primary_power_floor",
        "power_smoothing.secondary_power_floor",
        "power_smoothing.min_primary_floor_activation_offset",
        "power_smoothing.min_primary_floor_activation_point",
        "power_smoothing.curr_profile.secondary_power_floor",
        "power_smoothing.curr_profile.primary_floor_act_offset",
        "power_smoothing.admin_override.secondary_power_floor",
        "power_smoothing.admin_override.primary_floor_act_offset",
    ),
    "ms": ("power_smoothing.window_multiplier",),
    "MHz": (
        "clocks.current.graphics", "clocks.current.video",
        "clocks.applications.graphics", "clocks.applications.memory",
        "clocks.default_applications.graphics",
        "clocks.default_applications.memory", "clocks.max.graphics",
        "clocks.max.sm", "clocks.max.memory",
    ),
}

# Alternative names from the same --help-query-gpu text. nvidia-smi heads the
# column with the first name, e.g. --query-gpu=clocks.sm prints
# "clocks.current.sm [MHz]".
_NVIDIA_GPU_ALIASES = {
    "gpu_name": "name",
    "gpu_serial": "serial",
    "gpu_uuid": "uuid",
    "gpu_bus_id": "pci.bus_id",
    "inforom.image": "inforom.img",
    "inforom.power": "inforom.pwr",
    "gpu_operation_mode.current": "gom.current",
    "gpu_operation_mode.pending": "gom.pending",
    **{
        f"clocks_throttle_reasons.{reason}": f"clocks_event_reasons.{reason}"
        for reason in (
            "supported", "active", "gpu_idle", "applications_clocks_setting",
            "sw_power_cap", "hw_slowdown", "hw_thermal_slowdown",
            "hw_power_brake_slowdown", "sw_thermal_slowdown", "sync_boost",
        )
    },
    "retired_pages.sbe": "retired_pages.single_bit_ecc.count",
    "retired_pages.dbe": "retired_pages.double_bit.count",
    "remapped_rows.sbe": "remapped_rows.correctable",
    "remapped_rows.dbe": "remapped_rows.uncorrectable",
    "gpu.power.draw.average": "power.draw.average",
    "gpu.power.draw.instant": "power.draw.instant",
    "gpu.power.requested_limit": "power.limit",
    "gpu.power.current_limit": "enforced.power.limit",
    "gpu.power.default_limit": "power.default_limit",
    "gpu.power.min_limit": "power.min_limit",
    "gpu.power.max_limit": "power.max_limit",
    "module.power.requested_limit": "module.power.limit",
    "module.power.current_limit": "module.enforced.power.limit",
    "clocks.gr": "clocks.current.graphics",
    "clocks.sm": "clocks.current.sm",
    "clocks.mem": "clocks.current.memory",
    "clocks.video": "clocks.current.video",
    "clocks.applications.gr": "clocks.applications.graphics",
    "clocks.applications.mem": "clocks.applications.memory",
    "clocks.default_applications.gr": "clocks.default_applications.graphics",
    "clocks.default_applications.mem": "clocks.default_applications.memory",
    "clocks.max.gr": "clocks.max.graphics",
    "clocks.max.mem": "clocks.max.memory",
}


def _add_nvidia_fields(
    fields: dict[str, QueryField],
    unmodeled: Mapping[str | None, tuple[str, ...]],
    aliases: Mapping[str, str],
) -> None:
    for unit, names in unmodeled.items():
        for name in names:
            fields.setdefault(name, QueryField(None, unit=unit))
    for alias, name in aliases.items():
        fields.setdefault(alias, replace(fields[name], header=name))


# nvidia-smi 580.178.04 --help-query-retired-pages and
# --help-query-remapped-rows. FakeGPU models neither page retirement nor row
# remapping, so their values print [N/A], as the same fields of --query-gpu
# do.
_DEVICE_IDENTITY_FIELDS: dict[str, QueryField] = {
    "timestamp": QueryField("timestamp"),
    "gpu_name": QueryField("name"),
    "gpu_bus_id": QueryField("pci_bus_id"),
    "gpu_serial": QueryField(None),
    "gpu_uuid": QueryField("uuid"),
}
# The driver 580.178.04 lists, for an Ampere GPU, one row per
# retirement cause with an [N/A] address.
RETIRED_PAGE_CAUSES = ("Single Bit ECC", "Double Bit ECC")
RETIRED_PAGES_QUERY_FIELDS: dict[str, QueryField] = {
    **_DEVICE_IDENTITY_FIELDS,
    "retired_pages.address": QueryField(None),
    "retired_pages.timestamp": QueryField(None),
    "retired_pages.cause": QueryField("retired_pages.cause"),
}
REMAPPED_ROWS_QUERY_FIELDS: dict[str, QueryField] = dict(_DEVICE_IDENTITY_FIELDS)
_add_nvidia_fields(
    REMAPPED_ROWS_QUERY_FIELDS,
    {
        None: (
            "remapped_rows.correctable", "remapped_rows.correctable_inactive",
            "remapped_rows.uncorrectable",
            "remapped_rows.uncorrectable_inactive", "remapped_rows.pending",
            "remapped_rows.failure", "remapped_rows.histogram.max",
            "remapped_rows.histogram.high", "remapped_rows.histogram.partial",
            "remapped_rows.histogram.low", "remapped_rows.histogram.none",
        )
    },
    {
        "remapped_rows.sbe": "remapped_rows.correctable",
        "remapped_rows.dbe": "remapped_rows.uncorrectable",
    },
)
# nvidia-smi 580.178.04 --help-query-accounted-apps, with the units of its
# CSV header. Accounting is not modeled, so the query lists no processes, as
# nvidia-smi does while accounting mode is disabled.
ACCOUNTED_APPS_QUERY_FIELDS: dict[str, QueryField] = dict(_DEVICE_IDENTITY_FIELDS)
_add_nvidia_fields(
    ACCOUNTED_APPS_QUERY_FIELDS,
    {
        None: ("vgpu_instance", "pid"),
        "%": ("gpu_utilization", "mem_utilization"),
        "MiB": ("max_memory_usage",),
        "ms": ("time",),
    },
    {"gpu_util": "gpu_utilization", "mem_util": "mem_utilization"},
)
_add_nvidia_fields(GPU_QUERY_FIELDS, _NVIDIA_GPU_FIELDS, _NVIDIA_GPU_ALIASES)
# nvidia-smi 590.48.01 --help-query-compute-apps.
_add_nvidia_fields(
    PROCESS_QUERY_FIELDS,
    {None: ("gpu_serial",)},
    {"name": "process_name", "used_memory": "used_gpu_memory"},
)


def _nested_value(record: Mapping[str, Any], path: str) -> Any:
    current: Any = record
    for part in path.split("."):
        if not isinstance(current, Mapping):
            return None
        current = current.get(part)
    return current


def _query_value(
    record: Mapping[str, Any],
    spec: QueryField,
) -> Any:
    if spec.path is None:
        return None
    value = _nested_value(record, spec.path)
    if value is None:
        return None
    # The profile catalog writes 0 for a specification it does not know.
    section, _, key = spec.path.partition(".")
    if section == "profile" and key in (
        _nested_value(record, "profile.unavailable_fields") or ()
    ):
        return None
    if isinstance(value, (list, tuple, set)):
        return ";".join(str(item) for item in value)
    if isinstance(value, Mapping):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    if (
        spec.divisor != 1.0
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    ):
        value = float(value) / spec.divisor
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
    ):
        if spec.precision:
            return round(float(value), spec.precision)
        if isinstance(value, float):
            return int(round(value))
    return value


def _query_timestamp() -> str:
    """The time of the query as nvidia-smi prints it, YYYY/MM/DD HH:MM:SS.msec."""
    seconds, nanoseconds = divmod(time.time_ns(), 10**9)
    return (
        time.strftime("%Y/%m/%d %H:%M:%S", time.localtime(seconds))
        + f".{nanoseconds // 10**6:03d}"
    )


def _query_csv_header(field: str, spec: QueryField) -> str:
    field = spec.header or field
    if spec.unit:
        return f"{field} [{spec.unit}]"
    return field


def _query_csv_value(
    value: Any,
    *,
    spec: QueryField,
    nounits: bool,
) -> str:
    if value is None:
        # nvidia-smi's CSV marker for a value the GPU does not report.
        return "[N/A]"
    if (
        spec.fixed
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    ):
        text = f"{float(value):.{spec.precision}f}"
    else:
        text = str(value)
    if spec.unit and not nounits:
        text += f" {spec.unit}"
    return text


def _has_native_activity(activity: Mapping[str, Any]) -> bool:
    return any(activity[key] for key in NATIVE_ACTIVITY_COUNTERS)
