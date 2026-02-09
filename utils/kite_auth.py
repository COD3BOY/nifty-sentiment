"""Kite Connect OAuth flow helper."""

import logging
import webbrowser

from core.config import get_env

logger = logging.getLogger(__name__)


def get_login_url() -> str:
    """Get the Kite Connect login URL for OAuth."""
    api_key = get_env("KITE_API_KEY")
    if not api_key:
        raise ValueError("KITE_API_KEY not set in .env")
    return f"https://kite.zerodha.com/connect/login?v=3&api_key={api_key}"


def generate_session(request_token: str) -> str:
    """Exchange request_token for access_token after OAuth redirect."""
    from kiteconnect import KiteConnect

    api_key = get_env("KITE_API_KEY")
    api_secret = get_env("KITE_API_SECRET")

    if not api_key or not api_secret:
        raise ValueError("KITE_API_KEY and KITE_API_SECRET must be set in .env")

    kite = KiteConnect(api_key=api_key)
    data = kite.generate_session(request_token, api_secret=api_secret)
    access_token = data["access_token"]
    logger.info("Successfully generated Kite access token")
    return access_token


def open_login() -> None:
    """Open Kite login in the default browser."""
    url = get_login_url()
    print("Opening Kite login in browser...")
    print("After login, copy the request_token from the redirect URL.")
    webbrowser.open(url)
