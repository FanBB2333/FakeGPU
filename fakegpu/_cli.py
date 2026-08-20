"""Shared command-line conventions for FakeGPU entry points.

Every ``fakegpu <command>`` resolves to a module-level entry point listed in
:data:`BUILTIN_COMMANDS`, and ``fakegpu/__main__.py`` dispatches through that
registry.  Command modules build their ``prog=`` string with
:func:`command_prog` instead of repeating the name, so the registry is the
only place a command is named.

Conventions shared by the commands:

``--json``
    Commands that produce a report take an optional ``PATH``; ``--json``
    without a value writes the document to stdout (see
    :func:`add_json_path_argument`).  Commands whose entire output *is* the
    JSON document keep the boolean form (:func:`add_json_flag_argument`).

``--strict``
    Turns advisory findings into a non-zero status.  The status differs per
    command and is stated in that command's help text.

Exit codes
    ``0`` on success and ``2`` for unusable arguments or input, which is also
    argparse's own code for a parse error.  ``1`` reports a command-specific
    failure.  Commands that classify their outcome more finely (``preflight``
    grades a run, ``capabilities --strict`` reports an audit failure) document
    their codes in ``--help``.
"""

from __future__ import annotations

import importlib
from argparse import ArgumentParser, _ActionsContainer
from dataclasses import dataclass
from typing import Callable, NoReturn, Sequence


# ``ArgumentParser`` and the groups created by ``add_argument_group`` and
# ``add_mutually_exclusive_group`` share ``add_argument`` through this base
# class, so the flag factories below accept either.
ArgumentTarget = _ActionsContainer

EXIT_SUCCESS = 0
EXIT_FAILURE = 1
EXIT_USAGE = 2


@dataclass(frozen=True, slots=True)
class CommandSpec:
    module: str
    function: str


BUILTIN_COMMANDS: dict[str, CommandSpec] = {
    "demo": CommandSpec(".demo", "main"),
    "doctor": CommandSpec(".doctor", "main"),
    "preflight": CommandSpec(".preflight", "main"),
    "coordinator": CommandSpec(".distributed_cli", "coordinator_main"),
    "bandwidth": CommandSpec(".distributed_cli", "bandwidth_main"),
    "estimate-llm": CommandSpec(".llm_cli", "main"),
    "plan-serving": CommandSpec(".serving_plan", "main"),
    "estimate-diffusion": CommandSpec(".diffusion_estimator", "main"),
    "estimate-roofline": CommandSpec(".performance_model", "main"),
    "analyze-repo": CommandSpec(".repository_analyzer", "main"),
    "analyze-kernel": CommandSpec(".kernel_analysis", "main"),
    "calibrate": CommandSpec(".calibration", "main"),
    "plan-training": CommandSpec(".training_plan", "main"),
    "simulate-topology": CommandSpec(".topology", "main"),
    "replay-trace": CommandSpec(".trace_replay", "main"),
    "nvidia-smi": CommandSpec(".smi", "main"),
    "metrics": CommandSpec(".metrics", "main"),
    "workspace-profiles": CommandSpec(".workspace_cli", "main"),
    "validate": CommandSpec(".validation", "main"),
    "capabilities": CommandSpec(".capabilities", "main"),
}


def load_builtin_handler(
    command: str,
) -> Callable[[Sequence[str] | None], int] | None:
    spec = BUILTIN_COMMANDS.get(command)
    if spec is None:
        return None
    module = importlib.import_module(spec.module, package="fakegpu")
    return getattr(module, spec.function)


def command_name(module: str, function: str = "main") -> str:
    """Return the registered command served by ``module.function``.

    ``module`` accepts the caller's ``__name__``; only the final component is
    matched, so ``fakegpu.smi`` and ``.smi`` resolve alike.
    """

    tail = module.rpartition(".")[2]
    for command, spec in BUILTIN_COMMANDS.items():
        if spec.module.lstrip(".") == tail and spec.function == function:
            return command
    raise KeyError(f"no fakegpu command registered for {module}.{function}")


def command_prog(module: str, function: str = "main") -> str:
    """Return the ``prog=`` string for a registered command entry point."""

    return f"fakegpu {command_name(module, function)}"


def add_json_path_argument(
    parser: ArgumentTarget,
    *,
    help: str = "Write JSON to PATH, or stdout when PATH is omitted.",
) -> None:
    """Add the report-style ``--json [PATH]`` option.

    The value lands in ``args.json_path`` and is ready for
    :func:`fakegpu.structured_io.emit_json`, which treats ``"-"`` as stdout.
    """

    parser.add_argument(
        "--json",
        dest="json_path",
        nargs="?",
        const="-",
        help=help,
    )


def add_json_flag_argument(parser: ArgumentTarget, *, help: str) -> None:
    """Add the boolean ``--json`` switch used by stdout-only commands."""

    parser.add_argument("--json", action="store_true", help=help)


def add_strict_argument(
    parser: ArgumentTarget,
    *,
    help: str = "Return a non-zero status when validation finds a failure.",
) -> None:
    parser.add_argument("--strict", action="store_true", help=help)


def usage_error(parser: ArgumentParser, error: object) -> NoReturn:
    """Report unusable arguments or input and exit with :data:`EXIT_USAGE`."""

    parser.exit(EXIT_USAGE, f"{parser.prog}: {error}\n")
    raise AssertionError("parser.exit must not return")
