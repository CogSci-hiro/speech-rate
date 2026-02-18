"""Compatibility wrapper for the packaged rate CLI entrypoint."""

from __future__ import annotations

from rate.cli.main import main


if __name__ == "__main__":
    raise SystemExit(main())
