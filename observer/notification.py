import logging
from typing import Any

import requests

from configuration.types import (
    Notification,
    NotificationDiscord,
    NotificationGeneric,
    NotificationSlack,
    NotificationTelegram,
)

from .message import Message

LOGGER = logging.getLogger(__name__)


def log_message(notification: Notification, message: Message):
    LOGGER.log(message.level.value, message.message)

    lvl_msg = f"{message.level.name} {message.message}"

    notify_discord(notification.discord, lvl_msg)
    notify_slack(notification.slack, lvl_msg)
    notify_telegram(notification.telegram, lvl_msg)
    notify_generic(notification.generic, message)


def notify(
    url: str, method: str, headers: dict[str, str], json: dict[str, Any]
) -> requests.Response | None:
    try:
        return requests.request(
            url=url,
            method=method,
            headers=headers,
            json=json,
        )
    except Exception:
        pass


def notify_discord(config: NotificationDiscord, message: str) -> None:
    for u in config.webhook_url:
        notify(
            u,
            "POST",
            headers={"Content-Type": "application/json"},
            json={"content": message},
        )


def notify_slack(config: NotificationSlack, message: str) -> None:
    for u in config.webhook_url:
        notify(
            u,
            "POST",
            headers={"Content-Type": "application/json"},
            json={"text": message},
        )


def notify_telegram(config: NotificationTelegram, message: str) -> None:
    for t in config.bot:
        notify(
            f"https://api.telegram.org/bot{t.bot_token}/sendMessage",
            "POST",
            headers={"Content-Type": "application/json"},
            json={"chat_id": t.chat_id, "text": message},
        )


def notify_generic(config: NotificationGeneric, issue: "Message") -> None:
    for u in config.webhook_url:
        notify(
            u,
            "POST",
            headers={"Content-Type": "application/json"},
            json={"level": issue.level.value, "message": issue.message},
        )
