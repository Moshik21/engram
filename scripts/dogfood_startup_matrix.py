#!/usr/bin/env python3
"""Capture a multi-state Engram dogfood startup evidence bundle."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


@dataclass
class MatrixStep:
    name: str
    command: list[str] = field(default_factory=list)
    status: str = "pass"
    returncode: int | None = None
    duration_seconds: float = 0.0
    output_path: str | None = None
    detail: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo = Path(args.repo).expanduser().resolve()
    evidence_dir = Path(args.evidence_dir or default_evidence_dir()).expanduser().resolve()
    evidence_dir.mkdir(parents=True, exist_ok=True)

    steps: list[MatrixStep] = []
    context = {
        "repo": str(repo),
        "evidence_dir": str(evidence_dir),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "confirm_lifecycle": args.confirm_lifecycle,
    }

    stopped_by_matrix = False
    try:
        steps.extend(run_warmed_checks(repo, evidence_dir, prefix="01"))

        if not args.confirm_lifecycle:
            steps.append(
                MatrixStep(
                    name="lifecycle reset states",
                    status="skip",
                    detail="Skipped stop/start/stale-PID states; rerun with --confirm-lifecycle.",
                )
            )
        else:
            steps.append(
                run_command_step(
                    "stop runtime",
                    ["engramctl", "stop"],
                    repo,
                    evidence_dir,
                    "20-stop-runtime.log",
                )
            )
            stopped_by_matrix = True
            steps.append(
                run_command_step(
                    "stopped status",
                    ["engramctl", "status"],
                    repo,
                    evidence_dir,
                    "21-stopped-status.log",
                )
            )
            steps.append(
                run_validation_step(
                    "stopped validation",
                    repo,
                    evidence_dir,
                    "22-stopped-validation.json",
                    extra_args=["--skip-slow"],
                    expected="stopped",
                    allow_nonzero=True,
                )
            )
            start_step = run_command_step(
                "start runtime",
                ["engramctl", "start"],
                repo,
                evidence_dir,
                "30-start-runtime.log",
                timeout=args.start_timeout,
            )
            steps.append(start_step)
            if start_step.status == "pass":
                stopped_by_matrix = False
            steps.extend(run_warmed_checks(repo, evidence_dir, prefix="31"))
            if not args.skip_stale_pid:
                steps.append(run_stale_pid_simulation(repo, evidence_dir))
    finally:
        if stopped_by_matrix:
            steps.append(
                run_command_step(
                    "restore runtime after interruption",
                    ["engramctl", "start"],
                    repo,
                    evidence_dir,
                    "99-restore-runtime.log",
                    timeout=args.start_timeout,
                )
            )

    report = {
        "context": context,
        "steps": [asdict(step) for step in steps],
        "summary": summarize_steps(steps),
    }
    (evidence_dir / "matrix-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (evidence_dir / "matrix-report.md").write_text(render_markdown(report), encoding="utf-8")

    print(f"Evidence: {evidence_dir}")
    print(render_console_summary(report))
    return 1 if any(step.status == "fail" for step in steps) else 0


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Record exact command evidence for the Engram dogfood startup matrix. "
            "Stop/start and stale-PID states require --confirm-lifecycle."
        )
    )
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--evidence-dir")
    parser.add_argument("--confirm-lifecycle", action="store_true")
    parser.add_argument("--skip-stale-pid", action="store_true")
    parser.add_argument("--start-timeout", type=float, default=150.0)
    return parser.parse_args(argv)


def default_evidence_dir() -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    return Path("/tmp") / f"engram-dogfood-startup-{stamp}"


def run_warmed_checks(repo: Path, evidence_dir: Path, *, prefix: str) -> list[MatrixStep]:
    return [
        run_command_step(
            "status",
            ["engramctl", "status"],
            repo,
            evidence_dir,
            f"{prefix}-status.log",
        ),
        run_command_step(
            "storage",
            ["engramctl", "storage"],
            repo,
            evidence_dir,
            f"{prefix}-storage.log",
            timeout=60,
        ),
        run_command_step(
            "doctor",
            ["engramctl", "doctor", "--format", "json"],
            repo,
            evidence_dir,
            f"{prefix}-doctor.json",
            timeout=180,
        ),
        run_validation_step(
            "warmed validation",
            repo,
            evidence_dir,
            f"{prefix}-validation.json",
        ),
    ]


def run_validation_step(
    name: str,
    repo: Path,
    evidence_dir: Path,
    output_name: str,
    *,
    extra_args: list[str] | None = None,
    expected: str = "healthy",
    allow_nonzero: bool = False,
) -> MatrixStep:
    cmd = [
        sys.executable,
        str(repo / "scripts/dogfood_startup_validation.py"),
        "--json",
        *(extra_args or []),
    ]
    step = run_command_step(
        name,
        cmd,
        repo,
        evidence_dir,
        output_name,
        timeout=180,
        allow_nonzero=allow_nonzero,
    )
    payload = read_json_object(Path(step.output_path or ""))
    if payload:
        step.evidence["summary"] = validation_summary(payload)
    if expected == "healthy" and payload and has_validation_failures(payload):
        step.status = "fail"
        step.detail = "Validation JSON contains failing checks."
    elif expected == "stopped" and payload:
        if stopped_state_detected(payload):
            step.status = "pass"
            step.detail = "Stopped-state validation detected offline runtime as expected."
        else:
            step.status = "fail"
            step.detail = "Stopped-state validation did not detect offline runtime."
    return step


def run_stale_pid_simulation(repo: Path, evidence_dir: Path) -> MatrixStep:
    pid_file = Path.home() / ".engram/engram.pid"
    if pid_file.exists():
        return MatrixStep(
            name="stale PID simulation",
            status="skip",
            detail=f"Skipped because {pid_file} already exists.",
            evidence={"pid_file": str(pid_file)},
        )

    fake_pid = choose_fake_pid()
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(f"{fake_pid}\n", encoding="utf-8")
    try:
        step = run_validation_step(
            "stale PID simulation",
            repo,
            evidence_dir,
            "40-stale-pid-validation.json",
            extra_args=["--skip-slow"],
        )
        payload = read_json_object(Path(step.output_path or ""))
        supervisor = find_check(payload, "supervisor parity") if payload else None
        if supervisor and supervisor.get("status") == "warn" and "stale PID" in supervisor.get(
            "detail", ""
        ):
            step.status = "pass"
            step.detail = "Stale PID file was detected with specific remediation."
        else:
            step.status = "fail"
            step.detail = "Stale PID file was not detected as expected."
        step.evidence["simulated_pid_file"] = str(pid_file)
        step.evidence["fake_pid"] = fake_pid
        return step
    finally:
        try:
            if pid_file.read_text(encoding="utf-8").strip() == str(fake_pid):
                pid_file.unlink()
        except OSError:
            pass


def choose_fake_pid() -> int:
    pid = 999_999
    while process_exists(pid):
        pid += 1
    return pid


def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def run_command_step(
    name: str,
    cmd: list[str],
    cwd: Path,
    evidence_dir: Path,
    output_name: str,
    *,
    timeout: float = 60.0,
    allow_nonzero: bool = False,
) -> MatrixStep:
    output_path = evidence_dir / output_name
    started = time.monotonic()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        output = result.stdout
        returncode = result.returncode
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        returncode = None
        timed_out = True

    output_path.write_text(output, encoding="utf-8")
    duration = time.monotonic() - started
    status = "pass" if returncode == 0 or allow_nonzero else "fail"
    if timed_out:
        status = "fail"
    detail = "Command completed."
    if allow_nonzero and returncode not in (0, None):
        detail = "Command returned non-zero as expected for this state."
    if timed_out:
        detail = "Command timed out."
    return MatrixStep(
        name=name,
        command=cmd,
        status=status,
        returncode=returncode,
        duration_seconds=round(duration, 3),
        output_path=str(output_path),
        detail=detail,
        evidence={"output_excerpt": excerpt(strip_ansi(output))},
    )


def read_json_object(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def validation_summary(payload: dict[str, Any]) -> dict[str, int]:
    checks = payload.get("checks") or []
    return {
        status: sum(1 for check in checks if check.get("status") == status)
        for status in ("pass", "warn", "fail", "skip")
    }


def has_validation_failures(payload: dict[str, Any]) -> bool:
    return any(check.get("status") == "fail" for check in payload.get("checks") or [])


def stopped_state_detected(payload: dict[str, Any]) -> bool:
    health = find_check(payload, "HTTP health")
    status = find_check(payload, "engramctl status")
    return bool(
        health
        and health.get("status") == "fail"
        and status
        and status.get("status") == "fail"
    )


def find_check(payload: dict[str, Any], name: str) -> dict[str, Any] | None:
    for check in payload.get("checks") or []:
        if check.get("name") == name:
            return check
    return None


def summarize_steps(steps: list[MatrixStep]) -> dict[str, int]:
    return {
        status: sum(1 for step in steps if step.status == status)
        for status in ("pass", "warn", "fail", "skip")
    }


def render_console_summary(report: dict[str, Any]) -> str:
    summary = report["summary"]
    return (
        f"Summary: {summary['pass']} pass, {summary['warn']} warn, "
        f"{summary['fail']} fail, {summary['skip']} skip"
    )


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Engram Dogfood Startup Matrix Evidence",
        "",
        f"- Generated: `{report['context']['generated_at']}`",
        f"- Repo: `{report['context']['repo']}`",
        f"- Evidence dir: `{report['context']['evidence_dir']}`",
        f"- Lifecycle reset states: `{report['context']['confirm_lifecycle']}`",
        f"- {render_console_summary(report)}",
        "",
        "| Step | Status | Command | Evidence | Detail |",
        "| --- | --- | --- | --- | --- |",
    ]
    for step in report["steps"]:
        command = shell_join(step["command"]) if step["command"] else ""
        evidence = step.get("output_path") or ""
        lines.append(
            f"| {step['name']} | `{step['status']}` | `{command}` | "
            f"`{evidence}` | {step['detail']} |"
        )
    lines.append("")
    return "\n".join(lines)


def shell_join(cmd: list[str]) -> str:
    return " ".join(quote_arg(part) for part in cmd)


def quote_arg(value: str) -> str:
    if re.search(r"[^A-Za-z0-9_./:=+-]", value):
        return "'" + value.replace("'", "'\"'\"'") + "'"
    return value


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def excerpt(text: str, *, max_lines: int = 20, max_chars: int = 3000) -> str:
    lines = text.strip().splitlines()
    if len(lines) > max_lines:
        lines = [*lines[: max_lines // 2], "...", *lines[-max_lines // 2 :]]
    out = "\n".join(lines)
    if len(out) > max_chars:
        out = out[: max_chars - 20] + "\n...<truncated>"
    return out


if __name__ == "__main__":
    raise SystemExit(main())
