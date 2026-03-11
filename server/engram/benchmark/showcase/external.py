"""Supporting external-track metadata for the showcase benchmark."""

from __future__ import annotations

from pathlib import Path

from engram.benchmark.showcase.models import ExternalTrackResult


def collect_external_track_results(
    *,
    project_root: Path,
    locomo_dataset_path: str | None = None,
) -> list[ExternalTrackResult]:
    """Return appendix-track status for existing benchmark families."""
    scripts_dir = project_root / "scripts"
    results = [
        _script_track(
            name="retrieval_ab",
            script_path=scripts_dir / "benchmark_ab.py",
            recommended_command="uv run python scripts/benchmark_ab.py --verbose --seed 42",
            notes={
                "purpose": "Controlled retrieval A/B comparison on the synthetic corpus.",
            },
        ),
        _script_track(
            name="working_memory",
            script_path=scripts_dir / "benchmark_working_memory.py",
            recommended_command="uv run python scripts/benchmark_working_memory.py --verbose",
            notes={
                "purpose": "Working-memory continuity and bridge recall.",
            },
        ),
        _script_track(
            name="echo_chamber",
            script_path=scripts_dir / "benchmark_echo_chamber.py",
            recommended_command="uv run python scripts/benchmark_echo_chamber.py --queries 200",
            notes={
                "purpose": "Long-run drift, coverage, and surfaced-vs-used behavior.",
            },
        ),
    ]

    locomo_script = scripts_dir / "benchmark_locomo.py"
    dataset = Path(locomo_dataset_path).expanduser() if locomo_dataset_path else None
    locomo_available = locomo_script.exists() and dataset is not None and dataset.exists()
    results.append(
        ExternalTrackResult(
            name="locomo",
            available=locomo_available,
            executed=False,
            availability_reason=(
                None if locomo_available else "LoCoMo dataset path not provided or missing"
            ),
            summary_metrics={
                "purpose": "External multi-turn memory evaluation.",
            },
            artifact_path=str(locomo_script) if locomo_script.exists() else None,
            recommended_command=(
                f"uv run python scripts/benchmark_locomo.py --dataset-path {dataset}"
                if locomo_available
                else "uv run python scripts/benchmark_locomo.py --dataset-path <path>"
            ),
        )
    )
    return results


def _script_track(
    *,
    name: str,
    script_path: Path,
    recommended_command: str,
    notes: dict[str, str],
) -> ExternalTrackResult:
    return ExternalTrackResult(
        name=name,
        available=script_path.exists(),
        executed=False,
        availability_reason=None if script_path.exists() else "script missing",
        summary_metrics=notes,
        artifact_path=str(script_path) if script_path.exists() else None,
        recommended_command=recommended_command,
    )
