from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from ._cli import (
    add_json_flag_argument,
    command_prog,
    usage_error,
)
from .workspace_profiles import (
    WorkspaceProfileError,
    _catalog_summary,
    load_workspace_profiles,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog=command_prog(__name__),
        description="Validate and inspect static backend-workspace profile catalogs.",
    )
    parser.add_argument(
        "--path",
        action="append",
        default=[],
        help="Additional JSON/YAML catalog; may be repeated.",
    )
    add_json_flag_argument(
        parser,
        help="Emit the validated catalog summary as JSON.",
    )
    args = parser.parse_args(argv)
    paths = [Path(value) for value in args.path]
    try:
        profiles = load_workspace_profiles(paths)
    except WorkspaceProfileError as exc:
        usage_error(parser, exc)
    summary = _catalog_summary(profiles)

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    print(f"Workspace profiles: {summary['profile_count']}")
    for profile in profiles:
        match = profile["match"]
        stack = []
        for key in (
            "profile_ids",
            "architectures",
            "compute_capabilities",
            "torch_versions",
            "cuda_versions",
        ):
            if key in match:
                stack.append(f"{key}={','.join(str(v) for v in match[key])}")
        selector = profile.get("operator") or profile.get("operator_regex")
        print(
            f"- {profile['id']}: {selector} | {profile['lifetime']} | "
            f"{'; '.join(stack) if stack else 'all stacks'}"
        )
    return 0
