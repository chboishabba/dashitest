from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .contracts import StepStatus
from .schema import (
    ARROW_PROFILES,
    CANONICAL_ARROW_COLUMN,
    CANONICAL_CONE_MASK,
    CANONICAL_FEATURE_COLUMNS,
    ClosureEmbedding,
    FamilyClass,
)


EIGEN_LABELS = ("Earth", "Spoke", "Hub", "Clock")


@dataclass(frozen=True)
class SourceProjection:
    eigenspace: str
    prime: int | None
    hecke: str
    exponent: int | None
    mode: str
    score: float = 0.0


def safe_float(raw: object) -> float | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def build_closure_embedding(row: Mapping[str, object]) -> ClosureEmbedding:
    return ClosureEmbedding.from_mapping(row)


def canonical_projection(v: np.ndarray) -> np.ndarray:
    return np.asarray(v[: len(CANONICAL_FEATURE_COLUMNS)], dtype=float)


def canonical_arrow(v: np.ndarray) -> float:
    return float(np.asarray(v, dtype=float)[-1])


def canonical_mask() -> np.ndarray:
    return CANONICAL_CONE_MASK.copy()


def arrow_eps(profile: str) -> float:
    if profile not in ARROW_PROFILES:
        raise KeyError(f"unknown arrow profile: {profile}")
    return ARROW_PROFILES[profile]


def classify_step(structural_ok: bool, arrow_ok: bool) -> StepStatus:
    if structural_ok and arrow_ok:
        return StepStatus.INTERIOR
    if structural_ok and not arrow_ok:
        return StepStatus.ARROW_BOUNDARY
    if not structural_ok and arrow_ok:
        return StepStatus.STRUCTURAL_BOUNDARY
    return StepStatus.OUTSIDE


def classify_family(family_class: str) -> FamilyClass:
    mapping = {
        FamilyClass.INTERIOR_FAMILY.value: FamilyClass.INTERIOR_FAMILY,
        FamilyClass.ARROW_LADDER.value: FamilyClass.ARROW_LADDER,
        FamilyClass.SINGLE_ARROW_BREAK.value: FamilyClass.SINGLE_ARROW_BREAK,
        FamilyClass.MDL_TAIL_BOUNDARY.value: FamilyClass.MDL_TAIL_BOUNDARY,
        "mixed_hard_axis_outlier": FamilyClass.MDL_TAIL_BOUNDARY,
    }
    return mapping[family_class]


def normalize_counter(counter: Counter[str]) -> dict[str, float]:
    total = sum(counter.values())
    if total == 0:
        return {}
    return {key: value / total for key, value in counter.items()}


def dominant_label(distribution: Mapping[str, float]) -> tuple[str, float]:
    if not distribution:
        return "<missing>", 0.0
    items = sorted(distribution.items(), key=lambda item: (-item[1], item[0]))
    return items[0]


def ternary_signature(values: Sequence[float], eps: float) -> tuple[int, ...]:
    signature: list[int] = []
    for value in values:
        if value > eps:
            signature.append(1)
        elif value < -eps:
            signature.append(-1)
        else:
            signature.append(0)
    return tuple(signature)


def refined_signature_to_eigenspace(signature: Sequence[int]) -> str:
    if not signature:
        return "Earth"
    if signature[0] > 0:
        if len(signature) > 1 and signature[1] < 0:
            return "Spoke"
        return "Hub"
    if 0 in signature:
        return "Earth"
    return "Spoke"


def trace_eigen_mass(embedding: np.ndarray, eps: float = 1e-9) -> dict[str, float]:
    signature = ternary_signature(np.asarray(embedding, dtype=float), eps=eps)
    label = refined_signature_to_eigenspace(signature)
    weights = {eigen: 0.0 for eigen in EIGEN_LABELS}
    weights[label] = 1.0
    return weights


def rust_array_values(text: str, name: str) -> list[str]:
    pattern = rf"{re.escape(name)}:\s*\[[^\]]+\]\s*=\s*\[(.*?)\];"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return []
    raw = match.group(1)
    return [item.strip().strip('"') for item in raw.split(",") if item.strip()]


def rust_tuple_values(text: str, name: str) -> list[int]:
    pattern = rf"{re.escape(name)}:\s*\([^\)]*\)\s*=\s*\((.*?)\);"
    match = re.search(pattern, text)
    if not match:
        return []
    return [int(item.strip()) for item in match.group(1).split(",") if item.strip()]


def load_dasl_source_model(repo_root: Path) -> dict[str, object]:
    dasl_path = repo_root / "src" / "dasl.rs"
    sheaf_path = repo_root / "src" / "sheaf.rs"
    ipfs_path = repo_root / "src" / "ipfs.rs"
    if not dasl_path.exists():
        raise FileNotFoundError(f"missing DASL source file: {dasl_path}")
    if not sheaf_path.exists():
        raise FileNotFoundError(f"missing DASL source file: {sheaf_path}")
    dasl_text = dasl_path.read_text(encoding="utf-8")
    sheaf_text = sheaf_path.read_text(encoding="utf-8")

    monster_primes = [int(item) for item in rust_array_values(dasl_text, "MONSTER_PRIMES")]
    monster_exponents = [int(item) for item in rust_array_values(dasl_text, "MONSTER_EXPONENTS")]
    bott_names = rust_array_values(dasl_text, "BOTT_NAMES")
    attack_triple = rust_tuple_values(dasl_text, "ATTACK_TRIPLE")

    encoding_names: dict[str, str] = {}
    for variant, label in re.findall(r'Self::(\w+)\s*=>\s*"([^"]+)"', sheaf_text):
        encoding_names[variant] = label

    encoding_primes: dict[str, int] = {}
    for variant, prime_text in re.findall(r"Self::(\w+)\s*=>\s*(\d+)", sheaf_text):
        encoding_primes[variant] = int(prime_text)

    prime_to_eigenspace: dict[int, str] = {}
    for raw_primes, eigenspace in re.findall(r"([0-9\s|]+)=>\s*EigenSpace::(\w+)", sheaf_text):
        primes = [int(item.strip()) for item in raw_primes.split("|") if item.strip()]
        for prime in primes:
            prime_to_eigenspace[prime] = eigenspace

    entries: list[dict[str, object]] = []
    for index, prime in enumerate(monster_primes):
        bott_index = index % len(bott_names) if bott_names else None
        entries.append(
            {
                "prime": prime,
                "exponent": monster_exponents[index] if index < len(monster_exponents) else None,
                "eigenspace": prime_to_eigenspace.get(prime, "Earth"),
                "hecke": f"T_{prime}",
                "hecke_index": index,
                "bott_index": bott_index,
                "bott_name": bott_names[bott_index] if bott_index is not None else "<none>",
                "attack_triple": prime in attack_triple,
                "monster_basis": True,
                "encoding_name": encoding_names.get(str(prime), "<none>"),
            }
        )

    distribution = normalize_counter(Counter(str(entry["eigenspace"]) for entry in entries))
    return {
        "repo_root": str(repo_root),
        "files": {
            "dasl": str(dasl_path),
            "sheaf": str(sheaf_path),
            "ipfs": str(ipfs_path),
        },
        "monster_primes": monster_primes,
        "monster_exponents": monster_exponents,
        "attack_triple": attack_triple,
        "bott_names": bott_names,
        "entries": entries,
        "distribution": distribution,
    }


def source_projection_from_prime(
    source_model: Mapping[str, object],
    prime: int | None,
    *,
    mode: str = "canonical",
) -> SourceProjection:
    if prime is None:
        return SourceProjection(
            eigenspace="<missing>",
            prime=None,
            hecke="<none>",
            exponent=None,
            mode=mode,
        )
    for entry in source_model.get("entries", []):
        if int(entry.get("prime", -1)) != prime:
            continue
        return SourceProjection(
            eigenspace=str(entry.get("eigenspace", "Earth")),
            prime=prime,
            hecke=str(entry.get("hecke", f"T_{prime}")),
            exponent=int(entry["exponent"]) if entry.get("exponent") is not None else None,
            mode=mode,
            score=1.0,
        )
    return SourceProjection(
        eigenspace="<missing>",
        prime=prime,
        hecke=f"T_{prime}",
        exponent=None,
        mode=mode,
    )


def source_support_ok(
    source_model: Mapping[str, object],
    eigenspace: str,
) -> bool:
    distribution = source_model.get("distribution", {})
    if not isinstance(distribution, Mapping):
        return False
    return float(distribution.get(eigenspace, 0.0)) > 0.0


def source_support_basin_pred(source_model: Mapping[str, object]):
    def pred(x: np.ndarray) -> bool:
        eigenspace = dominant_label(trace_eigen_mass(x))[0]
        return source_support_ok(source_model, eigenspace)

    return pred


def source_eigen_fn(source_model: Mapping[str, object]):
    distribution = source_model.get("distribution", {})

    def eigen_fn(x: np.ndarray) -> dict[str, float]:
        trace_mass = trace_eigen_mass(x)
        if not isinstance(distribution, Mapping) or not distribution:
            return trace_mass
        dominant = dominant_label(trace_mass)[0]
        if float(distribution.get(dominant, 0.0)) > 0.0:
            return trace_mass
        return {str(key): float(value) for key, value in distribution.items()}

    return eigen_fn


def project_row_to_augmented_vector(row: Mapping[str, object]) -> np.ndarray:
    embedding = build_closure_embedding(row)
    return embedding.augmented_vector()


def default_source_repo_root() -> Path:
    return Path(__file__).resolve().parents[3] / "kant-zk-pastebin"


def default_agda_repo_root() -> Path:
    return default_source_repo_root()
