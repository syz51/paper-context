from __future__ import annotations

import argparse
import json
import sys

import uvicorn

from paper_context.config import get_settings
from paper_context.logging import configure_logging
from paper_context.worker.runner import run_synthetic_job_verification, run_worker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="paper-context")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("serve")

    worker_parser = subparsers.add_parser("worker")
    worker_parser.add_argument("--once", action="store_true")

    subparsers.add_parser("verify-synthetic-job")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    configure_logging(settings.log_level)

    if args.command == "serve":
        uvicorn.run(
            "paper_context.api.app:create_app",
            host=settings.runtime.app_host,
            port=settings.runtime.app_port,
            factory=True,
        )
        return 0

    if args.command == "worker":
        run_worker(once=args.once)
        return 0

    if args.command == "verify-synthetic-job":
        report = run_synthetic_job_verification()
        sys.stdout.write(json.dumps(report, default=str, indent=2) + "\n")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
