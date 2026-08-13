"""Command-line installation health check."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pcketlm.core.supervisor import build_supervisor_report, save_supervisor_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check this PocketLM installation.")
    parser.add_argument("--output", type=Path, help="Write the sanitized report to this path.")
    parser.add_argument("--json", action="store_true", help="Print the complete sanitized report.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_supervisor_report()
    path = save_supervisor_report(report, output=args.output)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        summary = report["summary"]
        print(f"PocketLM installation: {report['status']}")
        print(f"Installation ID: {report['installation_id']}")
        print(f"Runnable models: {summary['runnable_model_count']}")
        print(f"Report: {path}")
    return 1 if report["status"] == "blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())
