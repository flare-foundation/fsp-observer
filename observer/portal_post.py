"""FlareWatch: mirror fsp-observer alerts into the portal `posts` table.

Additive to the Discord/Slack/Telegram dispatch in observer.log_message().
The portal `posts` table is the operator-facing source of truth that the
Activity page renders from; Discord stays the public-facing forum.

Hard rules:
  * Best-effort. Every failure here is swallowed — a portal/DB problem must
    never block, delay, or crash the observer's existing alerting path.
  * DB target is the shared portal Postgres. PORTAL_DATABASE_URL must point
    at the /flarewatch_portal database (NOT /neondb — see the 2026-05-20
    db-split incident). The URL path is hard-checked before any INSERT.
  * INSERT + NOTIFY. The portal's live Activity SSE stream refreshes when a
    NOTIFY lands on the `portal_events` channel. There is NO INSERT trigger
    on `posts` (verified against the portal migrations) — the portal fires
    that NOTIFY from app code (lib/webhooks/post-writer.ts), so this module
    must too, or a mirrored row only surfaces on a full page reload. The
    NOTIFY runs in the same transaction as the INSERT (delivered on commit)
    and matches the portal's own writer: channel `portal_events`, topic
    `posts`, data {post_id, post_type, source}.
"""
import json
import logging
import os
import time
import uuid
from urllib.parse import urlparse

from configuration.config import ChainId, Protocol

from .message import Message, MessageLevel

LOGGER = logging.getLogger(__name__)

# severity -> attention_score (the portal sorts the Activity stream by this).
_ATTENTION_SCORE = {
    "critical": 0.95,
    "urgent": 0.7,
    "attention": 0.4,
    "info": 0.1,
}

# Fallback severity: observer MessageLevel -> portal severity. Used only when
# no fine-grained rule below matches the headline.
_LEVEL_SEVERITY = {
    MessageLevel.CRITICAL: "critical",
    MessageLevel.ERROR: "urgent",
    MessageLevel.WARNING: "attention",
    MessageLevel.INFO: "info",
    MessageLevel.DEBUG: "info",
}

# Fine-grained classification, ordered — first headline match wins. Each rule
# fixes BOTH the post_type and the severity. The severity here intentionally
# OVERRIDES the level-based fallback so that "node failing its job" alerts
# (missed submit, reveal offence, fast-update miss, low gas balance) land
# `critical` even though the observer tags them ERROR. headline = first
# non-empty line of message.message (the summary line passed to
# alert_text.build_alert()).
_RULES: list[tuple[str, str, str]] = [
    # (lowercased headline substring, post_type, severity)
    ("didn't submit a fast update", "validator_fast_update_missed", "critical"),
    ("didnt submit a fast update", "validator_fast_update_missed", "critical"),
    ("reveal didn't match", "validator_fsp_reveal_offence", "critical"),
    ("reveal didnt match", "validator_fsp_reveal_offence", "critical"),
    ("causing reveal offence", "validator_fsp_reveal_offence", "critical"),
    ("causing a reveal offence", "validator_fsp_reveal_offence", "critical"),
    ("no submit1 transaction", "validator_fsp_missed_submit", "critical"),
    ("no submit2 transaction", "validator_fsp_missed_submit", "critical"),
    ("no submitsignatures transaction", "validator_fsp_missed_signatures", "critical"),
    ("voter not registered", "validator_voter_registration_timeout", "critical"),
    ("observer crashed", "validator_observer_crashed", "critical"),
    ("balance", "validator_gas_balance_low", "critical"),
    ("signature doesn't match finalization", "validator_fsp_signature_mismatch", "urgent"),
    ("signature doesnt match finalization", "validator_fsp_signature_mismatch", "urgent"),
    ("sent after grace period", "validator_fsp_grace_period_missed", "urgent"),
    ("correct time interval", "validator_fsp_late_submit", "urgent"),
    ("commit hash unexpected length", "validator_fsp_client_bug", "urgent"),
    ("bit vote length", "validator_fsp_client_bug", "urgent"),
    ("does not use submit1", "validator_fsp_client_bug", "urgent"),
    ("didn't confirm consensus request", "validator_fdc_consensus_miss", "urgent"),
    ("didnt confirm consensus request", "validator_fdc_consensus_miss", "urgent"),
    ("minimal condition for ftso anchor feeds", "validator_ftso_anchor_feeds_low", "urgent"),
    ("minimal condition for fdc participation", "validator_fdc_participation_low", "urgent"),
    ("minimal condition for staking", "validator_node_uptime_low", "urgent"),
    ("address mismatch", "validator_address_mismatch", "urgent"),
    ("feeds, expected", "validator_fsp_value_anomaly", "attention"),
    ("'none' on indices", "validator_fsp_value_anomaly", "attention"),
    ("out of range", "validator_fsp_value_anomaly", "attention"),
    ("unclaimed reward", "validator_unclaimed_rewards", "info"),
]


def _headline(message: Message) -> str:
    body = message.message or ""
    for line in body.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return body.strip()


def classify(message: Message) -> tuple[str, str]:
    """Return (post_type, severity) for a Message.

    Fine-grained when the headline matches a known alert; otherwise a coarse
    per-protocol post_type with the severity mapped from MessageLevel.
    """
    headline = _headline(message).lower()
    for needle, post_type, severity in _RULES:
        if needle in headline:
            return post_type, severity
    # Fallback: coarse type by protocol, severity from level.
    if message.protocol is not None:
        proto = Protocol.id_to_name(message.protocol).replace(" ", "_")
        post_type = f"validator_fsp_{proto}_alert"
    else:
        post_type = "validator_fsp_alert"
    return post_type, _LEVEL_SEVERITY.get(message.level, "info")


def _db_url() -> str | None:
    """Return PORTAL_DATABASE_URL iff set and pointed at /flarewatch_portal."""
    db_url = os.environ.get("PORTAL_DATABASE_URL", "").strip()
    if not db_url:
        return None  # not provisioned — no-op, same as an unconfigured webhook
    path = (urlparse(db_url).path or "").rstrip("/")
    if path.rsplit("/", 1)[-1] != "flarewatch_portal":
        LOGGER.error(
            "portal mirror skipped: PORTAL_DATABASE_URL path is %r, "
            "expected /flarewatch_portal",
            path,
        )
        return None
    return db_url


def mirror_to_portal(config, message: Message) -> None:
    """INSERT one posts row mirroring this Message. Best-effort; never raises."""
    try:
        db_url = _db_url()
        if db_url is None:
            return

        try:
            import psycopg
            from psycopg.types.json import Jsonb
        except ImportError:
            LOGGER.warning("portal mirror skipped: psycopg not installed")
            return

        post_type, severity = classify(message)
        headline = _headline(message)
        node_label = os.environ.get("PORTAL_NODE_LABEL", "sgb-01")
        validator_id = os.environ.get("PORTAL_VALIDATOR_ID", "flarewatch-sgb-1")
        node_id = os.environ.get("PORTAL_NODE_ID", "primary")

        network = (
            ChainId.id_to_name(message.network)
            if message.network is not None
            else None
        )
        protocol = (
            Protocol.id_to_name(message.protocol)
            if message.protocol is not None
            else None
        )
        round_id = message.round.id if message.round is not None else None

        title = f"{node_label} — {headline}"[:200]
        context = [
            part
            for part in (
                protocol,
                f"round {round_id}" if round_id is not None else None,
                message.level.name,
            )
            if part
        ]
        summary = " · ".join(context) if context else headline
        now_ms = int(time.time() * 1000)
        post_id = str(uuid.uuid4())
        payload = {
            "level": message.level.name,
            "network": network,
            "protocol": protocol,
            "round": round_id,
            "headline": headline,
            "body": (message.message or "")[:6000],
            "producer": "fsp-observer",
        }

        with psycopg.connect(
            db_url,
            connect_timeout=5,
            application_name="fsp-observer-portal-mirror",
        ) as conn:
            conn.execute(
                """
                INSERT INTO posts (
                    post_id, validator_id, node_id, post_type, severity,
                    status, title, summary, payload, source,
                    created_at_ms, updated_at_ms, attention_score
                ) VALUES (
                    %s, %s, %s, %s, %s,
                    'open', %s, %s, %s, 'validator_chain',
                    %s, %s, %s
                )
                """,
                (
                    post_id,
                    validator_id,
                    node_id,
                    post_type,
                    severity,
                    title,
                    summary,
                    Jsonb(payload),
                    now_ms,
                    now_ms,
                    _ATTENTION_SCORE.get(severity, 0.1),
                ),
            )
            # Wake the portal's live Activity SSE stream. No INSERT trigger
            # exists on `posts`; the portal NOTIFYs from app code, so we do
            # too. Same connection/transaction as the INSERT — Postgres
            # delivers the NOTIFY atomically on the commit below.
            conn.execute(
                "SELECT pg_notify('portal_events', %s)",
                (
                    json.dumps(
                        {
                            "topic": "posts",
                            "validatorId": validator_id,
                            "data": {
                                "post_id": post_id,
                                "post_type": post_type,
                                "source": "validator_chain",
                            },
                            "atMs": now_ms,
                        }
                    ),
                ),
            )
            conn.commit()
        LOGGER.info(
            "portal mirror ok: post_type=%s severity=%s", post_type, severity
        )
    except Exception:
        LOGGER.exception("portal mirror failed (non-fatal)")
