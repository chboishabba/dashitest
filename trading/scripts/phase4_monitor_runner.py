#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class RunnerTarget:
    name: str
    proposal_log: Path
    prices_csv: Path


def _parse_config(path: Path) -> Iterable[RunnerTarget]:
    if not path.exists():
        raise SystemExit(f"missing config: {path}")
    raw = json.loads(path.read_text())
    targets = raw.get("targets")
    if not targets or not isinstance(targets, list):
        raise SystemExit("config must contain a 'targets' list")
    for entry in targets:
        name = entry.get("name")
        proposal = entry.get("proposal_log")
        prices = entry.get("prices_csv")
        if not (name and proposal and prices):
            raise SystemExit("each target needs 'name', 'proposal_log', and 'prices_csv'")
        yield RunnerTarget(
            name=name,
            proposal_log=Path(proposal),
            prices_csv=Path(prices),
        )


def _load_profile_flags(profile_name: str, profile_path: Path) -> list[str]:
    if not profile_path.exists():
        raise SystemExit(f"profile config not found: {profile_path}")
    raw = json.loads(profile_path.read_text())
    profile = raw.get(profile_name)
    if profile is None:
        raise SystemExit(f"profile '{profile_name}' not defined in {profile_path}")
    if not isinstance(profile, dict):
        raise SystemExit(f"profile '{profile_name}' must be an object")
    flags: list[str] = []
    for key, value in profile.items():
        flag = f"--{key.replace('_', '-')}"
        flags.extend([flag, str(value)])
    return flags


def _build_command(targets: Iterable[RunnerTarget], extra_args: list[str]) -> list[str]:
    script_path = Path(__file__).with_name("phase4_density_monitor.py")
    cmd = [sys.executable, str(script_path)]
    cmd.extend(extra_args)
    for target in targets:
        if not target.proposal_log.exists():
            raise SystemExit(f"missing proposal log for {target.name}: {target.proposal_log}")
        if not target.prices_csv.exists():
            raise SystemExit(f"missing prices csv for {target.name}: {target.prices_csv}")
        cmd.extend(
            [
                "--target",
                f"{target.name}={target.proposal_log},{target.prices_csv}",
            ]
        )
    return cmd


def _prepare_env() -> dict[str, str]:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    parts = ["."] if "." not in existing.split(os.pathsep) else []
    if existing:
        parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the Phase-4 density monitor across multiple tapes.")
    ap.add_argument("--config", type=Path, required=True, help="JSON file listing targets.")
    ap.add_argument(
        "--profile",
        type=str,
        default="strict",
        help="Profile name from configs/phase4_monitor_profiles.json to apply before other args (set to 'none' to skip).",
    )
    ap.add_argument(
        "--profile-config",
        type=Path,
        default=Path("configs/phase4_monitor_profiles.json"),
        help="Profile config JSON containing named flag bundles.",
    )
    ap.add_argument(
        "--monitor-arg",
        action="append",
        default=[],
        help="Additional arguments to pass to phase4_density_monitor.py (provide as quoted string).",
    )
    args = ap.parse_args()

    targets = list(_parse_config(args.config))
    profile_args: list[str] = []
    profile_name = args.profile.strip() if args.profile else ""
    if profile_name and profile_name.lower() != "none":
        profile_args = _load_profile_flags(profile_name, args.profile_config)
        print(f"Applying monitor profile '{profile_name}' from {args.profile_config}")
    extra_args: list[str] = []
    for flag in args.monitor_arg:
        extra_args.extend(shlex.split(flag))

    cmd = _build_command(targets, profile_args + extra_args)
    env = _prepare_env()
    print(f"Running: {' '.join(shlex.quote(part) for part in cmd)}")
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        raise SystemExit(f"monitor runner failed ({result.returncode})")


if __name__ == "__main__":
    main()
