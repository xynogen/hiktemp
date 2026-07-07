set shell := ["bash", "-cu"]

# ── default ────────────────────────────────────────────────────────────────────

[private]
default:
    @just --list

# ── env ────────────────────────────────────────────────────────────────────────

# Install all dependencies via uv
[group('env')]
install:
    uv sync --extra dev

# Build wheel + sdist, then upload to PyPI
[group('env')]
publish:
    uv build
    uv run --with twine -- twine upload dist/*

# ── test ───────────────────────────────────────────────────────────────────────

# Test operations (unit|tidy)
[group('test')]
test action:
    #!/usr/bin/env bash
    set -euo pipefail
    case "{{action}}" in
      unit)
        uv run --extra dev -- pytest tests/ -v
        ;;
      tidy)
        find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
        rm -rf .pytest_cache dist build hiktemp.egg-info
        ;;
      *)
        echo "Unknown action: {{action}}"
        echo "Usage: just test <unit|tidy>"
        exit 1
        ;;
    esac

# Format operations — fix rewrites, check is dry-run for CI (fix|check)
[group('format')]
format action="fix":
    #!/usr/bin/env bash
    set -euo pipefail
    case "{{action}}" in
      fix)
        uv run --with ruff -- ruff format hiktemp/ tests/
        uv run --with ruff -- ruff check --fix hiktemp/ tests/
        ;;
      check)
        uv run --with ruff -- ruff format --check hiktemp/ tests/
        uv run --with ruff -- ruff check hiktemp/ tests/
        ;;
      *)
        echo "Unknown action: {{action}}"
        echo "Usage: just format <fix|check>"
        exit 1
        ;;
    esac

# Lint operations — check reports issues, fix auto-corrects (check|fix)
[group('format')]
lint action="check":
    #!/usr/bin/env bash
    set -euo pipefail
    case "{{action}}" in
      check)
        uv run --with ruff -- ruff check hiktemp/ tests/
        ;;
      fix)
        uv run --with ruff -- ruff check --fix hiktemp/ tests/
        ;;
      *)
        echo "Unknown action: {{action}}"
        echo "Usage: just lint <check|fix>"
        exit 1
        ;;
    esac
