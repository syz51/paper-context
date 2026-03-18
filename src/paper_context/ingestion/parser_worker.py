from __future__ import annotations

import sys
from pathlib import Path

from .parser_isolation import run_parser_worker


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) not in {3, 4}:
        raise SystemExit(
            "usage: python -m paper_context.ingestion.parser_worker "
            "<parser> <filename> <output_dir> [source_path]"
        )
    parser_name, filename, output_dir = args[:3]
    if len(args) == 4:
        sys.stdout.buffer.write(
            run_parser_worker(
                parser_name,
                filename,
                Path(output_dir),
                source_path=Path(args[3]),
            )
        )
        return 0
    content = sys.stdin.buffer.read()
    sys.stdout.buffer.write(run_parser_worker(parser_name, filename, Path(output_dir), content))
    return 0


if __name__ == "__main__":  # pragma: no cover - module entrypoint
    raise SystemExit(main())
