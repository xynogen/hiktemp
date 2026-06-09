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

# Test operations (unit|lint|tidy)
[group('test')]
test action:
    #!/usr/bin/env bash
    set -euo pipefail
    case "{{action}}" in
      unit)
        venv/bin/python -m pytest tests/ -v
        ;;
      lint)
        venv/bin/python -m ruff check hiktemp/ tests/
        ;;
      tidy)
        find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
        rm -rf .pytest_cache dist build hiktemp.egg-info
        ;;
      *)
        echo "Unknown action: {{action}}"
        echo "Usage: just test <unit|lint|tidy>"
        exit 1
        ;;
    esac
