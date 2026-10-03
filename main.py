import asyncio
import logging
import traceback

import dotenv

from configuration.config import ChainId, get_config
from configuration.types import Configuration
from observer.message import Message, MessageLevel
from observer.observer import log_message, observer_loop

LOGGER = logging.getLogger()


def _classify_exception(e: BaseException) -> str:
    """Bucket an exception into one of: connection-reset, timeout,
    parse-or-config, import-error, unknown.

    Used to pick the right diagnosis + operator-actions text in the
    crash alert per 2026-05-13 alert-body refinement directive.
    """
    name = type(e).__name__
    if isinstance(e, (ConnectionResetError, ConnectionAbortedError, BrokenPipeError)):
        return "connection-reset"
    if isinstance(e, (ConnectionError, OSError)) and "Connection reset" in str(e):
        return "connection-reset"
    if isinstance(e, (TimeoutError, asyncio.TimeoutError)):
        return "timeout"
    if isinstance(e, (ValueError, KeyError, TypeError, AttributeError)):
        return "parse-or-config"
    if isinstance(e, (ImportError, ModuleNotFoundError)):
        return "import-error"
    # Best-effort name-based fallback for non-stdlib exceptions
    if "ConnectionReset" in name or "Reset" in name:
        return "connection-reset"
    if "Timeout" in name:
        return "timeout"
    return "unknown"


def _diagnosis_for(exc_class: str) -> str:
    if exc_class == "connection-reset":
        return (
            "The observer's connection to its upstream RPC endpoint was reset.\n"
            "Most likely the RPC node (go-flare via cloudflared, or the public\n"
            "fallback) closed the TCP connection mid-request: rate-limited,\n"
            "restarted, or proxied through a stale Cloudflare edge. The observer\n"
            "process exits on this error; supervisor (docker compose restart\n"
            "policy or systemd) brings it back, but the gap shows up as missed\n"
            "epoch participation."
        )
    if exc_class == "timeout":
        return (
            "An async operation timed out. The observer was waiting on an RPC\n"
            "response or an internal task that never completed. Usually the\n"
            "underlying RPC endpoint is slow or unreachable; less commonly an\n"
            "internal deadlock."
        )
    if exc_class == "parse-or-config":
        return (
            "An exception was raised while parsing data or accessing config\n"
            "state. Usually means the upstream RPC response shape changed\n"
            "(go-flare upgrade with breaking JSON-RPC change), OR the\n"
            "configuration file is missing an expected field, OR a contract\n"
            "ABI mismatch with the upstream Flare protocol."
        )
    if exc_class == "import-error":
        return (
            "Python import failed at observer-loop startup. Deployment\n"
            "configuration drift: a required package is missing from the\n"
            "container's Python environment, OR a code-level rename of an\n"
            "internal module wasn't propagated to all call sites."
        )
    return (
        "Unclassified exception. Treat as a real crash and investigate via\n"
        "the traceback in EVIDENCE."
    )


def _actions_for(exc_class: str) -> str:
    if exc_class == "connection-reset":
        return (
            "If supervisor already restarted the observer (check\n"
            "  docker ps --filter name=fsp-observer\n"
            "  docker logs --tail 50 fsp-observer\n"
            "and see if there's recent activity): the observer is back up;\n"
            "verify next epoch is being signed by checking the validator's\n"
            "vote-power utilization on Flare Explorer.\n"
            "\n"
            "If the observer is NOT back up: restart it manually:\n"
            "  cd /opt/flare/observer && docker compose up -d\n"
            "\n"
            "If the connection reset is recurring (same error every few minutes):\n"
            "  - Check upstream RPC: curl -s http://127.0.0.1:9653/ext/C/rpc \\\n"
            "      -d '{\"jsonrpc\":\"2.0\",\"method\":\"eth_blockNumber\",\"params\":[],\"id\":1}' \\\n"
            "      -H 'Content-Type: application/json'\n"
            "  - If local RPC is slow/dead: investigate go-flare; see\n"
            "    docs/runbooks/flr-rpc-heartbeat-deploy.md.\n"
            "  - If local RPC is fine but observer keeps disconnecting: check\n"
            "    if the observer is hitting a rate-limited public RPC instead\n"
            "    of the local node (observer/configuration/*.env)."
        )
    if exc_class == "timeout":
        return (
            "Restart the observer:\n"
            "  cd /opt/flare/observer && docker compose up -d\n"
            "\n"
            "If the timeout recurs: same RPC investigation path as\n"
            "connection-reset above. Verify upstream RPC latency from\n"
            "validator-host:\n"
            "  time curl -s http://127.0.0.1:9653/ext/C/rpc \\\n"
            "    -d '{\"jsonrpc\":\"2.0\",\"method\":\"eth_blockNumber\",\"params\":[],\"id\":1}'"
        )
    if exc_class == "parse-or-config":
        return (
            "The crash is likely a code-level bug or upstream-shape change,\n"
            "not a transient network event. Restarting won't fix it.\n"
            "\n"
            "Read the full traceback in journalctl:\n"
            "  journalctl -u fsp-observer-* -n 200 --no-pager\n"
            "OR\n"
            "  docker logs --tail 200 fsp-observer\n"
            "\n"
            "Identify the failing call site. Then:\n"
            "  - If go-flare was recently upgraded: cross-reference the\n"
            "    upstream go-flare changelog for breaking JSON-RPC changes.\n"
            "  - If config drift: diff the on-box config files against the\n"
            "    repo's configuration/ tree.\n"
            "  - If ABI change: check Flare Foundation announcements + open\n"
            "    a triage parking entry."
        )
    if exc_class == "import-error":
        return (
            "The container's Python environment is incomplete. Rebuild:\n"
            "  cd /opt/flare/observer && docker compose build --no-cache\n"
            "  docker compose up -d\n"
            "\n"
            "If the import error persists post-rebuild: pyproject.toml /\n"
            "requirements.txt is out of sync with what main.py imports.\n"
            "Check git log on observer/ for recent rename or refactor that\n"
            "wasn't fully propagated."
        )
    return (
        "Read the full traceback in journalctl OR docker logs:\n"
        "  docker logs --tail 200 fsp-observer\n"
        "\n"
        "Restart the observer if appropriate:\n"
        "  cd /opt/flare/observer && docker compose up -d\n"
        "\n"
        "If unfamiliar exception type: search upstream\n"
        "flare-foundation/fsp-observer issues + Flare Foundation Discord."
    )


def _truncate_traceback(tb: str, head: int = 12, tail: int = 12) -> str:
    """Keep the first `head` and last `tail` lines of a traceback; replace
    the middle with a placeholder. Discord embeds cap at 2000 chars and the
    body has more sections than just the trace."""
    lines = tb.splitlines()
    if len(lines) <= head + tail + 1:
        return tb
    return "\n".join(
        lines[:head]
        + [f"  ... ({len(lines) - head - tail} lines truncated; full in journalctl) ..."]
        + lines[-tail:]
    )


def main(config: Configuration):
    try:
        asyncio.run(observer_loop(config))
    except Exception as e:
        # Capture the traceback BEFORE building the alert body
        tb = traceback.format_exc()
        network_name = ChainId.id_to_name(config.chain_id)

        exc_class = _classify_exception(e)
        diagnosis = _diagnosis_for(exc_class)
        actions = _actions_for(exc_class)
        tb_truncated = _truncate_traceback(tb)

        body = (
            f"observer crashed on network:{network_name} (class: {exc_class})\n"
            f"\n"
            f"DIAGNOSIS\n"
            f"{diagnosis}\n"
            f"\n"
            f"EVIDENCE\n"
            f"exception: {type(e).__name__}: {e}\n"
            f"\n"
            f"traceback (truncated; full in journalctl/docker logs):\n"
            f"{tb_truncated}\n"
            f"\n"
            f"OPERATOR ACTIONS\n"
            f"{actions}"
        )

        mb = Message.builder().add(network=config.chain_id)
        message = mb.build(MessageLevel.CRITICAL, body)
        log_message(config, message)
        LOGGER.exception(e)
        LOGGER.error(traceback.format_exc())


if __name__ == "__main__":
    dotenv.load_dotenv()
    config = get_config()
    main(config)
