from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


NEVER_LEARNABLE = {
    # safety / veto
    "cvar_alpha",
    "veto_min_samples",
    "veto_buffer",
    "veto_cooldown",
    "hazard_threshold",
    "hazard_veto",
    # gate stability
    "tau_on",
    "tau_off",
    # hard eps / numerics
    "epsilon",
}

LEARNABLE = {
    # proposal scoring
    "score_weights",
    "instrument_weights",
    "size_weights",
    # option heads
    "opt_tenor_weights",
    "opt_mny_weights",
}


@dataclass(frozen=True)
class Bounds:
    low: float | None = None
    high: float | None = None

    def check(self, value: float, name: str) -> None:
        if self.low is not None and value < self.low:
            raise ValueError(f"{name}={value} < {self.low}")
        if self.high is not None and value > self.high:
            raise ValueError(f"{name}={value} > {self.high}")


NEVER_BOUNDS: dict[str, Bounds] = {
    "thresholds.tau_on": Bounds(0.0, 1.0),
    "thresholds.tau_off": Bounds(0.0, 1.0),
    "veto.cvar_alpha": Bounds(0.0, 1.0),
    "veto.min_samples": Bounds(1, None),
    "veto.buffer": Bounds(1, None),
    "veto.cooldown": Bounds(0, None),
    "veto.hazard_threshold": Bounds(0.0, None),
    "veto.hazard_veto.enabled": Bounds(0.0, 1.0),
    "veto.epsilon": Bounds(0.0, None),
}

LEARNABLE_BOUNDS: dict[str, Bounds] = {
    "score_weights": Bounds(0.0, 1.0),
    "instrument_weights": Bounds(0.0, 1.0),
    "size_weights": Bounds(0.0, 1.0),
    "opt_tenor_weights": Bounds(0.0, 1.0),
    "opt_mny_weights": Bounds(0.0, 1.0),
}

ALLOWED_PATHS = {
    "competition.ell_gain",
    "competition.hazard_penalty",
    "competition.tail_penalty",
    "competition.carry_gain",
    "competition.pnl_gain",
    "competition.score_margin_min",
    "competition.margin_gate.score_margin_min",
    "competition.margin_gate.enabled",
    "competition.score.logp_dir",
    "competition.score.logp_inst",
    "competition.score.logp_size",
    "competition.score.ell_margin.weight",
    "competition.score.hazard.weight",
    "competition.score.perp_carry.weight",
    "competition.score.tail_penalty.weight",
    "direction.deadzone",
    "direction.loss",
    "direction.class_weights.short",
    "direction.class_weights.flat",
    "direction.class_weights.long",
    "instrument.classes",
    "instrument.teacher_mix",
    "instrument.loss",
    "instrument.heuristics.perp.funding_weight",
    "instrument.heuristics.perp.basis_weight",
    "instrument.heuristics.option.iv_penalty",
    "instrument.heuristics.option.oi_penalty",
    "instrument.heuristics.option.expiry_penalty",
    "option.expiry_bins",
    "option.moneyness_bins",
    "option.loss",
    "option.temperature",
    "size.bins",
    "size.max_size",
    "size.teacher.ell_exp",
    "size.teacher.dir_exp",
    "size.teacher.hazard_exp",
    "size.teacher.tail_exp",
    "pnl.reweight_min",
    "pnl.reweight_max",
    "pnl.zscore_clip",
    "thresholds.tau_on",
    "thresholds.tau_off",
    "teachers.instrument.horizon_bars",
    "teachers.instrument.deadzone",
    "teachers.instrument.utility_temp",
    "teachers.instrument.teacher_mix",
    "teachers.instrument.funding_lambda",
    "teachers.instrument.option_iv_penalty",
    "teachers.instrument.option_slip",
    "veto.mode",
    "veto.alpha",
    "veto.min_samples",
    "veto.buffer",
    "veto.cooldown",
    "veto.hazard_threshold",
    "veto.hazard_veto.enabled",
    "veto.epsilon",
}

ALLOWED_LIST_PATHS = {
    "instrument.classes",
    "option.expiry_bins",
    "option.moneyness_bins",
    "size.bins",
}


def _parse_scalar(val: str) -> Any:
    if val.startswith('"') and val.endswith('"'):
        return val[1:-1]
    if val.startswith("'") and val.endswith("'"):
        return val[1:-1]
    if val.lower() in {"true", "false"}:
        return val.lower() == "true"
    if val.lower() in {"null", "none"}:
        return None
    try:
        if "." in val or "e" in val.lower():
            return float(val)
        return int(val)
    except ValueError:
        return val


def _parse_list(val: str) -> list[Any]:
    inner = val.strip()[1:-1].strip()
    if not inner:
        return []
    parts = [p.strip() for p in inner.split(",")]
    return [_parse_scalar(p) for p in parts]


def parse_simple_yaml(path: Path) -> dict[str, Any]:
    """
    Minimal YAML parser for mappings + inline lists.
    Ignores comments and empty lines.
    """
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(0, root)]
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if stripped.startswith("- "):
            item = stripped[2:].strip()
            if not stack:
                raise ValueError(f"List item with no parent: {raw}")
            parent = stack[-1][1]
            if "_list" not in parent:
                parent["_list"] = []
            if item.startswith("[") and item.endswith("]"):
                parent["_list"].append(_parse_list(item))
            else:
                parent["_list"].append(_parse_scalar(item))
            continue
        key_val = stripped.split(":", 1)
        if len(key_val) != 2:
            raise ValueError(f"Invalid line: {raw}")
        key = key_val[0].strip()
        val = key_val[1].strip()
        while stack and indent < stack[-1][0]:
            stack.pop()
        if not stack:
            raise ValueError(f"Bad indentation at: {raw}")
        parent = stack[-1][1]
        if val == "":
            node: dict[str, Any] = {}
            parent[key] = node
            stack.append((indent + 2, node))
        elif val.startswith("[") and val.endswith("]"):
            parent[key] = _parse_list(val)
        else:
            parent[key] = _parse_scalar(val)
    # fold any "_list" holders into actual lists
    def _fold(node: Any) -> Any:
        if isinstance(node, dict):
            if "_list" in node and len(node) == 1:
                return [_fold(x) for x in node["_list"]]
            return {k: _fold(v) for k, v in node.items() if k != "_list"}
        if isinstance(node, list):
            return [_fold(x) for x in node]
        return node

    return _fold(root)


def get_path(cfg: dict[str, Any], path: list[str], default: Any = None) -> Any:
    cur: Any = cfg
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def resolve_value(
    cli_value: Any,
    cfg: dict[str, Any],
    path: list[str],
    default: Any,
    name: str,
    resolved: dict[str, tuple[Any, str]],
) -> Any:
    if cli_value is not None:
        resolved[name] = (cli_value, "CLI")
        return cli_value
    yaml_val = get_path(cfg, path, None)
    if yaml_val is not None:
        resolved[name] = (yaml_val, "YAML")
        return yaml_val
    resolved[name] = (default, "DEFAULT")
    return default


def format_resolved(resolved: dict[str, tuple[Any, str]]) -> str:
    if not resolved:
        return ""
    keys = sorted(resolved.keys())
    width = max(len(k) for k in keys)
    lines = ["weights_config:"]
    for key in keys:
        value, source = resolved[key]
        lines.append(f"  {key.ljust(width)}: {value} ({source})")
    return "\n".join(lines)


def validate_never_learnable(cfg: dict[str, Any]) -> None:
    for path, bounds in NEVER_BOUNDS.items():
        keys = path.split(".")
        val = get_path(cfg, keys, None)
        if val is None:
            continue
        if isinstance(val, bool):
            num_val = float(val)
        else:
            num_val = float(val)
        bounds.check(num_val, path)


def validate_known_paths(cfg: dict[str, Any]) -> None:
    def _check(node: Any, prefix: str) -> None:
        if isinstance(node, dict):
            for key, val in node.items():
                path = f"{prefix}.{key}" if prefix else key
                if isinstance(val, dict):
                    if not any(p == path or p.startswith(f"{path}.") for p in ALLOWED_PATHS):
                        raise ValueError(f"Unknown config section: {path}")
                    _check(val, path)
                elif isinstance(val, list):
                    if path not in ALLOWED_LIST_PATHS and path not in ALLOWED_PATHS:
                        raise ValueError(f"Unknown list config: {path}")
                else:
                    if path not in ALLOWED_PATHS:
                        raise ValueError(f"Unknown config key: {path}")
        elif isinstance(node, list):
            if prefix not in ALLOWED_LIST_PATHS and prefix not in ALLOWED_PATHS:
                raise ValueError(f"Unknown list config: {prefix}")

    _check(cfg, "")


def _clamp_scalar(value: float, bounds: Bounds) -> float:
    if bounds.low is not None and value < bounds.low:
        return bounds.low
    if bounds.high is not None and value > bounds.high:
        return bounds.high
    return value


def clamp_learnable_vector(values: list[float], name: str) -> list[float]:
    bounds = LEARNABLE_BOUNDS.get(name, Bounds())
    return [_clamp_scalar(float(v), bounds) for v in values]


def normalize_vector(values: list[float]) -> list[float]:
    total = sum(float(v) for v in values)
    if total <= 0:
        return values
    return [float(v) / total for v in values]
