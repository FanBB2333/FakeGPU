from __future__ import annotations

from ._api import env, library_dir, run
from ._runtime import (
    RuntimeInitResult,
    init,
    init_privateuse1,
    is_initialized,
    patch_torch,
)
from ._stage import stage
from ._version import __version__
from .capabilities import (
    audit_native_capability_sources,
    audit_native_exports,
    load_native_capabilities,
    native_capability_report,
)
from .calibration import (
    CalibrationError,
    build_workload_calibration_bundle,
    compare_memory_reports,
    verify_calibration_reports,
)
from .diffusion_estimator import (
    DiffusionEstimateError,
    estimate_diffusion_generation,
    inspect_diffusion_pipeline,
    load_diffusion_profiles,
)
from .flop_counter import MatmulFlopCounterMode
from .kernel_analysis import (
    KernelAnalysisError,
    analyze_kernel_file,
    analyze_ptx,
    analyze_sass,
)
from .memory_estimator import (
    WorkspaceCoverageError,
    analyze_graph_memory,
    estimate_module_memory,
    require_workspace_coverage,
)
from .performance_model import (
    PerformanceModelError,
    estimate_roofline,
    profile_roofline,
)
from .repository_analyzer import RepositoryAnalysisError, analyze_repository
from .llm_estimator import (
    estimate_decoder_inference,
    estimate_kv_cache_memory,
    inspect_safetensors_checkpoint,
)
from .operator_profiles import (
    OperatorProfileError,
    evaluate_operator_profile,
    load_operator_profiles,
    match_operator_profile,
)
from .topology import (
    Topology,
    TopologyError,
    simulate_collective,
    simulate_point_to_point,
)
from .trace_replay import TraceReplayError, replay_trace
from .training_plan import (
    TrainingPlanError,
    estimate_training_plan,
    normalize_training_config,
)
from .workspace_profiles import (
    load_workspace_profiles,
    workspace_profile_summary,
)
from .validation import load_validation_manifest, run_validation_manifest

__all__ = [
    "CalibrationError",
    "DiffusionEstimateError",
    "KernelAnalysisError",
    "MatmulFlopCounterMode",
    "OperatorProfileError",
    "PerformanceModelError",
    "RepositoryAnalysisError",
    "RuntimeInitResult",
    "Topology",
    "TopologyError",
    "TraceReplayError",
    "TrainingPlanError",
    "WorkspaceCoverageError",
    "__version__",
    "analyze_graph_memory",
    "analyze_kernel_file",
    "analyze_ptx",
    "analyze_repository",
    "analyze_sass",
    "audit_native_capability_sources",
    "audit_native_exports",
    "build_workload_calibration_bundle",
    "compare_memory_reports",
    "env",
    "estimate_decoder_inference",
    "estimate_diffusion_generation",
    "estimate_kv_cache_memory",
    "estimate_module_memory",
    "estimate_roofline",
    "estimate_training_plan",
    "evaluate_operator_profile",
    "init",
    "init_privateuse1",
    "inspect_safetensors_checkpoint",
    "inspect_diffusion_pipeline",
    "is_initialized",
    "library_dir",
    "load_native_capabilities",
    "load_operator_profiles",
    "load_diffusion_profiles",
    "load_validation_manifest",
    "load_workspace_profiles",
    "match_operator_profile",
    "native_capability_report",
    "normalize_training_config",
    "patch_torch",
    "profile_roofline",
    "replay_trace",
    "require_workspace_coverage",
    "run",
    "run_validation_manifest",
    "simulate_collective",
    "simulate_point_to_point",
    "stage",
    "workspace_profile_summary",
    "verify_calibration_reports",
]
