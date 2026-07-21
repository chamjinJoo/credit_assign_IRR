#!/usr/bin/env python3
"""Curate QMIX checkpoints for SMAX and Jumanji LBF evaluation."""

from __future__ import annotations

import argparse
import filecmp
import json
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EXPECTED_SEEDS = (1, 2, 31, 32, 42)

SMAX_CONSTRAINTS = {
    ("3m", 3): "vanilla",
    ("3m", 2): "hard",
    ("2s3z", 5): "vanilla",
    ("2s3z", 4): "medium",
    ("2s3z", 3): "hard",
    ("3s5z", 8): "vanilla",
    ("3s5z", 6): "hard",
    ("6h_vs_8z", 8): "vanilla",
    ("6h_vs_8z", 6): "hard",
    ("smacv2_5_units", 5): "vanilla",
    ("smacv2_5_units", 4): "medium",
    ("smacv2_5_units", 3): "hard",
    ("smacv2_10_units", 10): "vanilla",
    ("smacv2_10_units", 8): "medium",
    ("smacv2_10_units", 6): "hard",
}

LBF_SCENARIOS = {
    (2, 8, 2, 2, True): "2s-8x8-2p-2f-coop",
    (2, 10, 3, 3, False): "2s-10x10-3p-3f",
    (2, 15, 3, 3, False): "2s-15x15-3p-3f",
    (2, 15, 4, 3, False): "2s-15x15-4p-3f",
    (2, 15, 4, 5, False): "2s-15x15-4p-5f",
}

ALGORITHM_MODELS = {
    "jax_qmix": "qmix",
    "qmix": "qmix",
    "jax_pci_qmix": "pci-qmix",
    "pci_qmix": "pci-qmix",
}

ENVIRONMENTS = {
    "SMAXBatchWrapper": "smax",
    "JumanjiLBFBatchWrapper": "lbf",
}


@dataclass(frozen=True)
class Candidate:
    environment: str
    scenario: str
    constraint: str | None
    model: str
    config_model: str
    seed: int
    run_name: str
    env_dir: Path
    config_path: Path
    best_path: Path
    final_path: Path

    @property
    def complete(self) -> bool:
        return self.best_path.is_file() and self.final_path.is_file()

    @property
    def has_checkpoint(self) -> bool:
        return self.final_path.is_file() or self.best_path.is_file()

    @property
    def key(self) -> tuple[str, str, str | None, str, int]:
        return self.environment, self.scenario, self.constraint, self.model, self.seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--source", type=Path, default=root / "runs" / "checkpoints")
    parser.add_argument("--smax-destination", type=Path, default=root / "runs" / "checkpoints_smax")
    parser.add_argument(
        "--lbf-destination",
        type=Path,
        default=root / "runs" / "checkpoints_for_lbf",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=root / "runs" / "qmix_checkpoint_classification_report.json",
    )
    parser.add_argument("--apply", action="store_true", help="Copy the selected artifacts")
    return parser.parse_args()


def normalize_algorithm(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def checkpoint_tokens(path: Path) -> set[bytes]:
    wanted = {b"target_params", b"mixer", b"bridge_state"}
    found: set[bytes] = set()
    overlap = max(map(len, wanted)) - 1
    tail = b""
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            data = tail + chunk
            found.update(token for token in wanted if token in data)
            if found == wanted:
                break
            tail = data[-overlap:]
    return found


def checkpoint_model(path: Path) -> str:
    tokens = checkpoint_tokens(path)
    if b"target_params" not in tokens or b"mixer" not in tokens:
        raise ValueError(f"Not a QMIX checkpoint: {path}")
    return "pci-qmix" if b"bridge_state" in tokens else "qmix"


def get_smax_location(environment: dict[str, Any]) -> tuple[str, str]:
    key = (str(environment.get("map_name")), int(environment.get("allowed_target_window")))
    if key not in SMAX_CONSTRAINTS:
        raise ValueError(f"Unknown SMAX map/target-window pair: {key}")
    return key[0], SMAX_CONSTRAINTS[key]


def get_lbf_scenario(environment: dict[str, Any]) -> str:
    key = (
        int(environment.get("sight")),
        int(environment.get("field_size")),
        int(environment.get("num_agents")),
        int(environment.get("max_food")),
        bool(environment.get("force_coop")),
    )
    if key not in LBF_SCENARIOS:
        raise ValueError(f"Unknown LBF scenario: {key}")
    return LBF_SCENARIOS[key]


def discover(source: Path) -> tuple[list[Candidate], list[dict[str, Any]]]:
    candidates: list[Candidate] = []
    errors: list[dict[str, Any]] = []

    for config_path in sorted(source.glob("*/*/simulation_config.json")):
        try:
            payload = json.loads(config_path.read_text())
            simulation = payload.get("simulation_info", {})
            config = payload.get("config", {})
            environment_name = simulation.get("environment")
            environment_kind = ENVIRONMENTS.get(environment_name)
            config_model = ALGORITHM_MODELS.get(
                normalize_algorithm(config.get("algorithm", {}).get("type"))
            )
            if environment_kind is None or config_model is None:
                continue

            environment = config.get("environment", {})
            training = config.get("training", {})
            if environment_kind == "smax":
                scenario, constraint = get_smax_location(environment)
            else:
                scenario = get_lbf_scenario(environment)
                constraint = None

            env_dir = config_path.parent
            best_path = env_dir / "checkpoints" / "jax_checkpoint_best.pkl"
            final_path = env_dir / "checkpoints" / "jax_checkpoint_final.pkl"
            checkpoint_path = final_path if final_path.is_file() else best_path
            if not checkpoint_path.is_file():
                raise ValueError("Neither best nor final checkpoint exists")

            model = checkpoint_model(checkpoint_path)
            candidates.append(
                Candidate(
                    environment=environment_kind,
                    scenario=scenario,
                    constraint=constraint,
                    model=model,
                    config_model=config_model,
                    seed=int(training.get("seed")),
                    run_name=env_dir.parent.name,
                    env_dir=env_dir,
                    config_path=config_path,
                    best_path=best_path,
                    final_path=final_path,
                )
            )
        except Exception as exc:
            errors.append({"config_path": str(config_path), "error": str(exc)})

    return candidates, errors


def select_candidates(candidates: list[Candidate]) -> tuple[list[Candidate], list[Candidate]]:
    grouped: dict[tuple[str, str, str | None, str, int], list[Candidate]] = defaultdict(list)
    unexpected: list[Candidate] = []
    for candidate in candidates:
        if candidate.seed not in EXPECTED_SEEDS:
            unexpected.append(candidate)
            continue
        grouped[candidate.key].append(candidate)

    selected = []
    for group in grouped.values():
        completed = [candidate for candidate in group if candidate.complete]
        pool = completed or [candidate for candidate in group if candidate.has_checkpoint]
        selected.append(max(pool, key=lambda candidate: candidate.run_name))
    return sorted(selected, key=lambda candidate: candidate.key), unexpected


def destination_dir(candidate: Candidate, smax_root: Path, lbf_root: Path) -> Path:
    if candidate.environment == "smax":
        return smax_root / candidate.scenario / str(candidate.constraint) / candidate.model / candidate.run_name
    return lbf_root / candidate.scenario / candidate.model / candidate.run_name / candidate.env_dir.name


def selected_files(candidate: Candidate) -> list[tuple[Path, Path]]:
    files = [
        (candidate.config_path, Path("simulation_config.json")),
        (candidate.best_path, Path("checkpoints/jax_checkpoint_best.pkl")),
        (candidate.final_path, Path("checkpoints/jax_checkpoint_final.pkl")),
    ]
    if candidate.environment == "lbf":
        files.extend(
            [
                (candidate.env_dir / "README.txt", Path("README.txt")),
                (
                    candidate.env_dir / "results" / "training_summary.json",
                    Path("results/training_summary.json"),
                ),
            ]
        )
    return [(source, relative) for source, relative in files if source.is_file()]


def copy_candidate(candidate: Candidate, destination: Path, apply: bool) -> tuple[int, int]:
    copied = 0
    unchanged = 0
    for source, relative in selected_files(candidate):
        target = destination / relative
        if target.exists():
            if not filecmp.cmp(source, target, shallow=False):
                raise FileExistsError(f"Destination differs from source: {target}")
            unchanged += 1
            continue
        if apply:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        copied += 1
    return copied, unchanged


def expected_keys() -> set[tuple[str, str, str | None, str, int]]:
    keys = set()
    for (scenario, _), constraint in SMAX_CONSTRAINTS.items():
        for model in ("qmix", "pci-qmix"):
            for seed in EXPECTED_SEEDS:
                keys.add(("smax", scenario, constraint, model, seed))
    for scenario in LBF_SCENARIOS.values():
        for model in ("qmix", "pci-qmix"):
            for seed in EXPECTED_SEEDS:
                keys.add(("lbf", scenario, None, model, seed))
    return keys


def key_record(key: tuple[str, str, str | None, str, int]) -> dict[str, Any]:
    environment, scenario, constraint, model, seed = key
    return {
        "environment": environment,
        "scenario": scenario,
        "constraint": constraint,
        "model": model,
        "seed": seed,
    }


def candidate_record(candidate: Candidate, destination: Path) -> dict[str, Any]:
    return {
        **key_record(candidate.key),
        "run_name": candidate.run_name,
        "source": str(candidate.env_dir),
        "destination": str(destination),
        "config_model": candidate.config_model,
        "complete": candidate.complete,
        "has_best": candidate.best_path.is_file(),
        "has_final": candidate.final_path.is_file(),
    }


def print_missing(missing: list[tuple[str, str, str | None, str, int]]) -> None:
    grouped: dict[tuple[str, str, str | None, str], list[int]] = defaultdict(list)
    for environment, scenario, constraint, model, seed in missing:
        grouped[(environment, scenario, constraint, model)].append(seed)
    print("Missing expected checkpoints:")
    for key, seeds in sorted(grouped.items()):
        environment, scenario, constraint, model = key
        location = f"{scenario}/{constraint}" if constraint else scenario
        print(f"  {environment}: {location}/{model}: seeds {sorted(seeds)}")


def main() -> None:
    args = parse_args()
    candidates, errors = discover(args.source)
    selected, unexpected = select_candidates(candidates)

    copied_files = 0
    unchanged_files = 0
    copy_errors: list[dict[str, Any]] = []
    records = []
    for candidate in selected:
        destination = destination_dir(candidate, args.smax_destination, args.lbf_destination)
        records.append(candidate_record(candidate, destination))
        try:
            copied, unchanged = copy_candidate(candidate, destination, args.apply)
            copied_files += copied
            unchanged_files += unchanged
        except Exception as exc:
            copy_errors.append(
                {"source": str(candidate.env_dir), "destination": str(destination), "error": str(exc)}
            )

    selected_keys = {candidate.key for candidate in selected if candidate.complete}
    missing = sorted(expected_keys() - selected_keys)
    mismatches = [record for record in records if record["model"] != record["config_model"]]
    partial = [record for record in records if not record["complete"]]
    report = {
        "mode": "apply" if args.apply else "dry-run",
        "source": str(args.source.resolve()),
        "smax_destination": str(args.smax_destination.resolve()),
        "lbf_destination": str(args.lbf_destination.resolve()),
        "expected_seeds": list(EXPECTED_SEEDS),
        "source_candidates": len(candidates),
        "selected_runs": records,
        "unexpected_seed_candidates": [
            candidate_record(
                candidate,
                destination_dir(candidate, args.smax_destination, args.lbf_destination),
            )
            for candidate in unexpected
        ],
        "config_checkpoint_model_mismatches": mismatches,
        "partial_selected_runs": partial,
        "missing_complete_checkpoints": [key_record(key) for key in missing],
        "discovery_errors": errors,
        "copy_errors": copy_errors,
        "copied_files": copied_files,
        "unchanged_files": unchanged_files,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")

    selected_smax = sum(candidate.environment == "smax" for candidate in selected)
    selected_lbf = sum(candidate.environment == "lbf" for candidate in selected)
    print(f"Mode: {report['mode']}")
    print(f"Candidates: {len(candidates)}")
    print(f"Selected runs: {len(selected)} (SMAX={selected_smax}, LBF={selected_lbf})")
    print(f"Files: copied/planned={copied_files}, unchanged={unchanged_files}")
    print(f"Unexpected-seed candidates ignored: {len(unexpected)}")
    print(f"Config/checkpoint model mismatches: {len(mismatches)}")
    print(f"Selected runs without final: {len(partial)}")
    print(f"Discovery errors: {len(errors)}; copy errors: {len(copy_errors)}")
    print(f"Report: {args.report}")
    print_missing(missing)

    if errors or copy_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
