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
from .flop_counter import MatmulFlopCounterMode
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
from .llm_estimator import estimate_decoder_inference, inspect_safetensors_checkpoint
from .workspace_profiles import (
    load_workspace_profiles,
    workspace_profile_summary,
)
from .validation import load_validation_manifest, run_validation_manifest

__all__ = [
    "RuntimeInitResult",
    "MatmulFlopCounterMode",
    "PerformanceModelError",
    "RepositoryAnalysisError",
    "WorkspaceCoverageError",
    "__version__",
    "analyze_graph_memory",
    "analyze_repository",
    "audit_native_capability_sources",
    "audit_native_exports",
    "env",
    "estimate_module_memory",
    "estimate_roofline",
    "estimate_decoder_inference",
    "inspect_safetensors_checkpoint",
    "init",
    "init_privateuse1",
    "is_initialized",
    "library_dir",
    "load_workspace_profiles",
    "load_native_capabilities",
    "load_validation_manifest",
    "patch_torch",
    "profile_roofline",
    "native_capability_report",
    "require_workspace_coverage",
    "run",
    "run_validation_manifest",
    "stage",
    "workspace_profile_summary",
]
