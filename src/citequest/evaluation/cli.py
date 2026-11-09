"""Offline recorded evaluation: no agent execution, model or external tracing."""

# ruff: noqa: T201 -- bounded, non-sensitive machine-readable process status
from __future__ import annotations

import argparse
import json
from pathlib import Path
from uuid import UUID

from .recorded import Protector, evaluate_records, validate_cases


def bounded(path: Path, maximum: int) -> bytes:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum:
        raise ValueError("bounded_regular_evaluation_input_required")
    return path.read_bytes()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="Read only explicitly selected tasks owned by this owner")
    export.add_argument("--business-dsn-file", type=Path, required=True)
    export.add_argument("--owner-id", type=UUID, required=True)
    export.add_argument("--task-id", type=UUID, action="append", required=True)
    evaluate = commands.add_parser("evaluate", help="Evaluate signed recorded traces against explicit protected gold")
    evaluate.add_argument("--recorded", type=Path, required=True)
    evaluate.add_argument("--cases", type=Path, required=True)
    for command in (export, evaluate):
        command.add_argument("--hmac-key-file", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise FileExistsError("fresh_evaluation_output_required")
        protector = Protector(bounded(args.hmac_key_file, 4096))
        if args.command == "export":
            from .exporter import export_recorded
            value = export_recorded(args.business_dsn_file, args.owner_id, args.task_id, protector)
        else:
            records = protector.verify(json.loads(bounded(args.recorded, 32 * 1024**2)))
            cases = validate_cases(json.loads(bounded(args.cases, 4 * 1024**2)), protector.key_id)
            value = evaluate_records(records, cases)
        encoded = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf8") as output:
            output.write(encoded)
        print(json.dumps({"status": "exported" if args.command == "export" else value["status"],
            "records": len(value.get("records", value.get("cases", []))), "judge_model_requests": 0}))
        return 0
    except Exception as error:
        # DSNs, raw DB rows, args, quotes, package errors and local paths stay out.
        print(json.dumps({"status": "unavailable", "error_type": type(error).__name__, "judge_model_requests": 0}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
