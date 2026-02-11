"""Startup health checks for the NIFTY trading system.

Validates config, database, data directory, and API credentials at launch
so operators know immediately if something is misconfigured.
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _PROJECT_ROOT / "data"


def check_config() -> tuple[bool, str]:
    """Check that config.yaml loads and passes schema validation."""
    try:
        from core.config import load_config
        cfg = load_config()
        if not cfg:
            return False, "config.yaml loaded but is empty"
        return True, "config.yaml loaded successfully"
    except Exception as e:
        return False, f"config.yaml failed to load: {e}"


def check_database() -> tuple[bool, str]:
    """Check that SQLite database is accessible and writable."""
    db_path = _DATA_DIR / "nifty_sentiment.db"
    try:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        # Test write access by opening in WAL mode
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("SELECT 1")
        conn.close()
        return True, f"Database accessible at {db_path}"
    except Exception as e:
        return False, f"Database check failed: {e}"


def check_data_dir() -> tuple[bool, str]:
    """Check that data/ directory exists and is writable."""
    try:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        # Test write access
        test_file = _DATA_DIR / ".health_check_test"
        test_file.write_text("ok")
        test_file.unlink()
        return True, f"data/ directory writable at {_DATA_DIR}"
    except Exception as e:
        return False, f"data/ directory check failed: {e}"


def check_kite_credentials() -> tuple[bool, str]:
    """Check that Kite API credentials are present (not validated against API)."""
    from dotenv import load_dotenv
    load_dotenv(_PROJECT_ROOT / ".env", override=True)

    api_key = os.environ.get("KITE_API_KEY", "")
    access_token = os.environ.get("KITE_ACCESS_TOKEN", "")

    if not api_key and not access_token:
        return False, "KITE_API_KEY and KITE_ACCESS_TOKEN not set (Kite features will be unavailable)"
    if not api_key:
        return False, "KITE_API_KEY not set"
    if not access_token:
        return False, "KITE_ACCESS_TOKEN not set (regenerate daily via utils/kite_auth.py)"
    return True, "Kite credentials present"


def check_anthropic_key() -> tuple[bool, str]:
    """Check that ANTHROPIC_API_KEY is set."""
    from dotenv import load_dotenv
    load_dotenv(_PROJECT_ROOT / ".env", override=True)

    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        return False, "ANTHROPIC_API_KEY not set (Claude analysis features will be unavailable)"
    return True, "ANTHROPIC_API_KEY present"


def run_startup_checks() -> list[dict]:
    """Run all health checks and return results.

    Returns:
        List of dicts with keys: name, ok (bool), message (str), critical (bool).
    """
    checks = [
        ("Config", check_config, True),
        ("Database", check_database, True),
        ("Data Directory", check_data_dir, True),
        ("Kite Credentials", check_kite_credentials, False),
        ("Anthropic API Key", check_anthropic_key, False),
    ]

    results = []
    for name, fn, critical in checks:
        try:
            ok, message = fn()
        except Exception as e:
            ok, message = False, f"Check raised unexpected error: {e}"

        results.append({
            "name": name,
            "ok": ok,
            "message": message,
            "critical": critical,
        })

        level = logging.INFO if ok else (logging.ERROR if critical else logging.WARNING)
        logger.log(level, "Health check [%s]: %s — %s", name, "PASS" if ok else "FAIL", message)

    return results
