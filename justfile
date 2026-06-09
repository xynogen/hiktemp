set shell := ["bash", "-cu"]

# ── default ────────────────────────────────────────────────────────────────────

[private]
default:
    @just --list

# ── env ────────────────────────────────────────────────────────────────────────

# Create venv and install all dependencies (including dev extras)
[group('env')]
install:
    python3 -m venv venv
    venv/bin/pip install -e ".[dev]"

# ── test ───────────────────────────────────────────────────────────────────────

# Test operations (unit|tidy)
[group('test')]
test action:
    #!/usr/bin/env bash
    set -euo pipefail
    case "{{action}}" in
      unit)
        venv/bin/python -m pytest tests/ -v
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
        venv/bin/ruff format hiktemp/ tests/
        venv/bin/ruff check --fix hiktemp/ tests/
        ;;
      check)
        venv/bin/ruff format --check hiktemp/ tests/
        venv/bin/ruff check hiktemp/ tests/
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
        venv/bin/ruff check hiktemp/ tests/
        ;;
      fix)
        venv/bin/ruff check --fix hiktemp/ tests/
        ;;
      *)
        echo "Unknown action: {{action}}"
        echo "Usage: just lint <check|fix>"
        exit 1
        ;;
    esac
