from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

import requests

from configuration.config import ChainId, Protocol
from configuration.types import (
    NotificationDiscord,
    NotificationGeneric,
    NotificationSlack,
    NotificationTelegram,
)

from .message import Message, MessageLevel


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


# 2026-05-13: Discord groups consecutive webhook messages from the same
# author into a single continuous bubble. When 5 alerts fire in 30s, the
# bodies blur together. Prepend a visible ASCII divider so the
# `[LEVEL] network:X round:Y` header of each alert is clearly delineated
# from the previous alert's body.
_DISCORD_ALERT_DIVIDER = "═══════════════════════════════════════"


def notify_discord(config: NotificationDiscord, message: Message) -> None:
    for u in config.webhook_url:
        content = (
            f"{_DISCORD_ALERT_DIVIDER}\n"
            f"{message.build_str(with_log=True)}"
        )
        notify(
            u,
            "POST",
            headers={"Content-Type": "application/json"},
            json={"content": content},
        )


LEVEL_COLORS = defaultdict(
    lambda: 0x95A5A6,
    {
        MessageLevel.DEBUG: 0x3498DB,  # Blue
        MessageLevel.INFO: 0x2ECC71,  # Green
        MessageLevel.WARNING: 0xF1C40F,  # Yellow
        MessageLevel.ERROR: 0xE74C3C,  # Red
        MessageLevel.CRITICAL: 0x992D22,  # Dark Red
    },
)


def get_icon_url(network: int) -> str:
    network_name = ChainId.id_to_name(network)
    return (
        "https://raw.githubusercontent.com/flare-foundation/fsp-observer/main"
        f"/assets/{network_name}.png"
    )


def notify_discord_embed(config: NotificationDiscord, message: Message) -> None:
    color = LEVEL_COLORS[message.level]

    fields = []
    if message.protocol is not None:
        fields.append(
            {
                "name": "Protocol",
                "value": Protocol.id_to_name(message.protocol).upper(),
                "inline": True,
            }
        )

    if message.round is not None:
        fields.append(
            {
                "name": "Round",
                "value": str(message.round.id),
                "inline": True,
            }
        )

    embed = {
        "author": {
            "name": f"fsp-observer @ {ChainId.id_to_name(message.network)}",
            "url": "https://github.com/flare-foundation/fsp-observer",
        },
        "title": f"{message.level.name.title()}",
        "description": f"{message.message}",
        "color": color,
        "fields": fields,
        "timestamp": datetime.now(UTC).isoformat(),
        "footer": {
            "text": "Flare Network",
            "icon_url": get_icon_url(ChainId.FLARE),
        },
        "thumbnail": {
            "url": get_icon_url(message.network),
            "height": 32,
            "width": 32,
        },
    }

    for u in config.webhook_url:
        notify(
            u,
            "POST",
            headers={"Content-Type": "application/json"},
            json={"embeds": [embed]},
        )


def notify_slack(config: NotificationSlack, message: Message) -> None:
    for u in config.webhook_url:
        notify(
            u,
            "POST",
            headers={"Content-Type": "application/json"},
            json={"text": message.build_str(with_log=True)},
        )


def notify_slack_embed(config: NotificationSlack, message: Message) -> None:
    color = LEVEL_COLORS[message.level]

    fields = []
    if message.protocol is not None:
        fields.append(
            {
                "title": "Protocol",
                "value": Protocol.id_to_name(message.protocol).upper(),
                "short": True,
            }
        )

    if message.round is not None:
        fields.append(
            {
                "title": "Round",
                "value": str(message.round.id),
                "short": True,
            }
        )

    attachment = {
        "color": f"#{color:06X}",
        "author_name": f"fsp-observer @ {ChainId.id_to_name(message.network)}",
        "author_link": "https://github.com/flare-foundation/fsp-observer",
        "title": f"{message.level.name.title()}",
        "text": f"{message.message}",
        "fields": fields,
        "footer": "Flare Network",
        "footer_icon": get_icon_url(ChainId.FLARE),
        "thumb_url": get_icon_url(message.network),
        "ts": int(datetime.now(UTC).timestamp()),
    }

    for u in config.webhook_url:
        notify(
            u,
            "POST",
            headers={"Content-Type": "application/json"},
            json={"attachments": [attachment]},
        )


def notify_telegram(config: NotificationTelegram, message: Message) -> None:
    for t in config.bot:
        notify(
            f"https://api.telegram.org/bot{t.bot_token}/sendMessage",
            "POST",
            headers={"Content-Type": "application/json"},
            json={"chat_id": t.chat_id, "text": message.build_str(with_log=True)},
        )


def notify_generic(config: NotificationGeneric, issue: Message) -> None:
    for u in config.webhook_url:
        notify(
            u,
            "POST",
            headers={"Content-Type": "application/json"},
            json={"level": issue.level.value, "message": issue.message},
        )
