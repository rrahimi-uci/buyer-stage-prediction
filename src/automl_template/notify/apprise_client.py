"""Apprise wrapper. Replaces ``bs_lambda_send_sns_notification``.

If ``APPRISE_URL`` is empty (the safe demo default) notifications are logged only.
"""

from __future__ import annotations

import logging

from automl_template.config import Settings, get_settings

logger = logging.getLogger("automl_template.notify")


def notify(level: str, message: str, settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    title = f"[tabular-automl-template] {level.upper()}"
    if not settings.apprise_url:
        logger.info("%s %s", title, message)
        return
    import apprise

    ap = apprise.Apprise()
    ap.add(settings.apprise_url)
    ap.notify(title=title, body=message)


__all__ = ["notify"]
