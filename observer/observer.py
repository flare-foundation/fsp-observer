import asyncio
import hashlib
import logging
import os
import time
from collections import defaultdict, deque
from collections.abc import Awaitable, Sequence
from functools import partial
from itertools import chain
from typing import Any, Self

import requests
from eth_abi.abi import encode
from eth_account._utils.signing import to_standard_v
from eth_account.messages import _hash_eip191_message, encode_defunct
from eth_keys.datatypes import Signature as EthSignature
from eth_typing import ChecksumAddress
from eth_utils.address import to_checksum_address
from py_flare_common.b58 import flare_b58_encode_check
from py_flare_common.fsp.epoch.epoch import RewardEpoch
from py_flare_common.fsp.messaging import (
    parse_submit1_tx,
    parse_submit2_tx,
    parse_submit_signature_tx,
)
from py_flare_common.fsp.messaging.types import Signature as SSignature
from py_flare_common.ftso.median import FtsoMedian
from web3 import AsyncWeb3
from web3._utils.events import get_event_data
from web3.middleware import ExtraDataToPOAMiddleware
from web3.types import TxData

from configuration.config import Protocol
from configuration.types import (
    Configuration,
    un_prefix_0x,
)
from observer.contract_manager import ContractManager
from observer.fast_updates_manager import FastUpdate, FastUpdatesManager
from observer.reward_epoch_manager import (
    RewardManager,
    SigningPolicy,
)
from observer.signing_policy_manager import SigningPolicyManager
from observer.types import (
    AttestationRequest,
    FastUpdateFeeds,
    FastUpdateFeedsSubmitted,
    ProtocolMessageRelayed,
    RandomAcquisitionStarted,
    SigningPolicyInitialized,
    VotePowerBlockSelected,
    VoterPreRegistered,
    VoterRegistered,
    VoterRegistrationInfo,
    VoterRemoved,
)
from observer.validation.minimal_conditions import MinimalConditions
from observer.validation.validation import extract_round_for_entity, validate_round

from . import metrics
from .message import Message, MessageLevel
from .notification import (
    notify_discord,
    notify_discord_embed,
    notify_generic,
    notify_slack,
    notify_slack_embed,
    notify_telegram,
)
from .portal_post import mirror_to_portal
from .voting_round import (
    VotingRoundManager,
    WTxData,
)

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s\t%(levelname)s\t%(name)s\t%(message)s",
    level="INFO",
)
logging.getLogger("web3").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


def node_id_to_representation(node_id):
    decoded = bytes.fromhex(node_id)
    return f"NodeID-{flare_b58_encode_check(decoded).decode()}"


class Signature(EthSignature):
    @classmethod
    def from_vrs(cls, s: SSignature) -> Self:
        return cls(
            vrs=(
                to_standard_v(int(s.v, 16)),
                int(s.r, 16),
                int(s.s, 16),
            )
        )

    @classmethod
    def from_dict(cls, dict: dict[str, Any]):
        return Signature(
            vrs=(
                to_standard_v(dict["v"]),
                int.from_bytes((dict["r"]), "big"),
                int.from_bytes((dict["s"]), "big"),
            )
        )

    def recover_addr_from_msg(self, sp_hash: str) -> str:
        return self.recover_public_key_from_msg_hash(
            _hash_eip191_message(encode_defunct(hexstr=sp_hash))
        ).to_checksum_address()


def calculate_update_from_tx(config: Configuration, w: AsyncWeb3, tx: TxData):
    submission = w.eth.contract(
        abi=config.contracts.Submission.abi, address=config.contracts.Submission.address
    )
    fast_updates = w.eth.contract(
        abi=config.contracts.FastUpdater.abi,
        address=config.contracts.FastUpdater.address,
    )
    assert "input" in tx
    proxy_input = submission.decode_function_input(tx["input"])[1]["_data"].hex()
    updates = fast_updates.decode_function_input("0x470e91df" + proxy_input)[1][
        "_updates"
    ]

    cred = updates["sortitionCredential"]
    signed_message = (
        hashlib.sha256(
            encode(
                ["uint256", "(uint256,(uint256,uint256),uint256,uint256)", "bytes"],
                [
                    updates["sortitionBlock"],
                    (
                        cred["replicate"],
                        (cred["gamma"]["x"], cred["gamma"]["y"]),
                        cred["c"],
                        cred["s"],
                    ),
                    updates["deltas"],
                ],
            )
        )
        .digest()
        .hex()
    )

    signing_policy_address = un_prefix_0x(
        Signature.from_dict(updates["signature"]).recover_addr_from_msg(signed_message)
    )

    assert "from" in tx
    address = tx["from"]

    array = "".join(f"{i:08b}" for i in updates["deltas"])
    assert len(array) % 2 == 0
    signed_array = [
        -int(array[u + 1]) if array[u] == "1" else int(array[u + 1])
        for u in range(0, len(array), 2)
    ]

    return signing_policy_address, address, signed_array


async def get_block_production(w: AsyncWeb3, lookback: int = 1_000_000) -> float:
    latest_block = await w.eth.get_block("latest")
    assert "timestamp" in latest_block
    assert "number" in latest_block
    to_compare = min(lookback, int(latest_block["number"]) - 1)
    comparison_block = await w.eth.get_block(int(latest_block["number"]) - to_compare)
    assert "timestamp" in comparison_block
    time_delta = latest_block["timestamp"] - comparison_block["timestamp"]
    block_production = time_delta / to_compare
    return block_production


def calculate_maximum_exponent(block_production: float, config: Configuration) -> int:
    blocks_in_epoch = int(
        config.epoch.reward_epoch_factory.duration() / block_production
    )
    max_exponent = blocks_in_epoch // 100
    return max_exponent


# a reward epoch's signing policy is initialized in the 2h before the epoch starts
# (newSigningPolicyInitializationStartSeconds is 7200 on every chain): random
# acquisition, then voter registration, then the signing policy event all land in that
# window, independent of the reward epoch length (3.5d on mainnets, 6h on testnets)
SIGNING_POLICY_INITIALIZATION_S = 2 * 60 * 60
# scan a bit wider on both ends so boundary events aren't clipped by block estimation
REGISTRATION_SCAN_BUFFER_S = 30 * 60


async def find_block_at_timestamp(
    w: AsyncWeb3,
    from_block_id: int,
    from_ts: int,
    target_ts: int,
    block_production: float,
) -> int:
    # walk back from a known (block, ts) pair using the chain's block production rate
    # to estimate the block at target_ts, then refine until within tolerance; using the
    # measured rate (instead of assuming 1s/block) keeps this accurate on every chain
    block_id = from_block_id - int((from_ts - target_ts) / block_production)
    block = await w.eth.get_block(block_id)
    assert "timestamp" in block
    d = block["timestamp"] - target_ts
    while abs(d) > 600:
        block_id -= int(d / block_production) or (1 if d > 0 else -1)
        block = await w.eth.get_block(block_id)
        assert "timestamp" in block
        d = block["timestamp"] - target_ts
    return block_id


async def find_voter_registration_blocks(
    w: AsyncWeb3,
    current_block_id: int,
    reward_epoch: RewardEpoch,
    block_production: float,
) -> tuple[int, int]:
    current_ts = int(time.time())

    # everything from random acquisition to the signing policy event happens in
    # [start - 2h, start], scanned with a buffer on each side
    target_start_ts = (
        reward_epoch.start_s
        - SIGNING_POLICY_INITIALIZATION_S
        - REGISTRATION_SCAN_BUFFER_S
    )
    # the epoch we observe has already started, so cap the end at the current block
    target_end_ts = min(reward_epoch.start_s + REGISTRATION_SCAN_BUFFER_S, current_ts)

    start_block_id = await find_block_at_timestamp(
        w, current_block_id, current_ts, target_start_ts, block_production
    )
    end_block_id = await find_block_at_timestamp(
        w, current_block_id, current_ts, target_end_ts, block_production
    )

    return (start_block_id, end_block_id)


async def get_logs_chunked(
    w: AsyncWeb3,
    params: dict[str, Any],
    start_block: int,
    end_block: int,
    max_block_range: int,
) -> list:
    # some rpc providers cap the number of blocks per get_logs request, so we split
    # the range into chunks of at most max_block_range blocks
    logs = []
    chunk_start = start_block
    while chunk_start <= end_block:
        chunk_end = min(chunk_start + max_block_range - 1, end_block)
        logs.extend(
            await w.eth.get_logs(
                {**params, "fromBlock": chunk_start, "toBlock": chunk_end}
            )
        )
        chunk_start = chunk_end + 1
    return logs


async def get_signing_policy_events(
    w: AsyncWeb3,
    config: Configuration,
    reward_epoch: RewardEpoch,
    start_block: int,
    end_block: int,
) -> SigningPolicy:
    # reads logs for given blocks for the informations about the signing policy

    builder = SigningPolicy.builder().for_epoch(reward_epoch)

    contracts = [
        config.contracts.VoterRegistry,
        config.contracts.FlareSystemsCalculator,
        config.contracts.Relay,
        config.contracts.FlareSystemsManager,
    ]

    event_names = {
        # relay
        "SigningPolicyInitialized",
        # flare systems calculator
        "VoterRegistrationInfo",
        # flare systems manager
        "RandomAcquisitionStarted",
        "VotePowerBlockSelected",
        "VoterRegistered",
        "VoterRemoved",
    }
    event_signatures = {
        e.signature: e
        for c in contracts
        for e in c.events_by_signature.values()
        if e.name in event_names
    }

    block_logs = await get_logs_chunked(
        w,
        {"address": [contract.address for contract in contracts]},
        start_block,
        end_block,
        config.max_block_range,
    )

    _relay_patch_sps = await get_logs_chunked(
        w,
        {
            "address": [
                to_checksum_address("0x92a6E1127262106611e1e129BB64B6D8654273F7"),
                to_checksum_address("0x97702e350CaEda540935d92aAf213307e9069784"),
                to_checksum_address("0x57a4c3676d08Aa5d15410b5A6A80fBcEF72f3F45"),
                to_checksum_address("0x67a916E175a2aF01369294739AA60dDdE1Fad189"),
            ],
            "topics": [
                "0x"
                + config.contracts.Relay.events["SigningPolicyInitialized"].signature
            ],
        },
        start_block,
        end_block,
        config.max_block_range,
    )
    block_logs.extend(_relay_patch_sps)

    # NOTE:(@janezicmatej) VoterRegistry and FlareSystemsCalculator were redeployed on
    # flare and songbird on 2026-07-17; the contract registry only returns the new
    # addresses, so registrations emitted by the legacy deployments have to be indexed
    # in addition (abis for both variants are already loaded, see Contracts)
    _legacy_registration_logs = await get_logs_chunked(
        w,
        {
            "address": [
                # flare: legacy VoterRegistry
                to_checksum_address("0x2580101692366e2f331e891180d9ffdF861Fce83"),
                # flare: legacy FlareSystemsCalculator
                to_checksum_address("0x67c4B11c710D35a279A41cff5eb089Fe72748CF8"),
                # songbird: legacy VoterRegistry
                to_checksum_address("0x31B9EC65C731c7D973a33Ef3FC83B653f540dC8D"),
                # songbird: legacy FlareSystemsCalculator
                to_checksum_address("0x126FAeEc75601dA3354c0b5Cc0b60C85fCbC3A5e"),
            ],
        },
        start_block,
        end_block,
        config.max_block_range,
    )
    block_logs.extend(_legacy_registration_logs)

    # patched logs come from separate queries and get appended out of order; restore
    # chain order so the SigningPolicyInitialized early-break below doesn't skip them
    block_logs.sort(key=lambda log: (log["blockNumber"], log["logIndex"]))

    for log in block_logs:
        sig = log["topics"][0]

        if sig.hex() not in event_signatures:
            continue

        event = event_signatures[sig.hex()]
        data = get_event_data(w.eth.codec, event.abi, log)

        match event.name:
            case "VoterRegistered":
                e = VoterRegistered.from_dict(data["args"])
            case "VoterRemoved":
                e = VoterRemoved.from_dict(data["args"])
            case "VoterRegistrationInfo":
                e = VoterRegistrationInfo.from_dict(data["args"])
            case "SigningPolicyInitialized":
                e = SigningPolicyInitialized.from_dict(data["args"])
            case "VotePowerBlockSelected":
                e = VotePowerBlockSelected.from_dict(data["args"])
            case "RandomAcquisitionStarted":
                e = RandomAcquisitionStarted.from_dict(data["args"])
            case x:
                raise ValueError(f"Unexpected event {x}")
        builder.add(e)

        # signing policy initialized is the last event that gets emitted
        if event.name == "SigningPolicyInitialized":
            break

    return builder.build()


# FlareWatch S40 (2026-05-07): Operator-deferral-aware notification filter.
# Reads FLAREWATCH_PROTOCOLS_ACTIVE from env (comma-separated lowercase
# protocol names matching Protocol.id_to_name() output: "ftso", "fdc",
# "fast updates", "staking"). Messages tagged with a protocol NOT in the
# whitelist get logged locally but are NOT dispatched to notification
# channels. Default (env unset or empty): all protocols active = original
# upstream behavior. Untagged messages (e.g. observer crashes) always
# dispatch — only protocol-tagged operational alerts are filterable.
def _flarewatch_should_dispatch(message: Message) -> bool:
    raw = os.environ.get("FLAREWATCH_PROTOCOLS_ACTIVE", "").strip()
    if not raw:
        return True
    if message.protocol is None:
        return True
    active = {p.strip().lower() for p in raw.split(",") if p.strip()}
    protocol_name = Protocol.id_to_name(message.protocol)
    return protocol_name in active


# FlareWatch 2026-05-13: TTL-based dedup cache for log_message dispatch.
# The observer's outer loop occasionally double-dispatches the same alert
# (same level/network/round/protocol/headline) from multiple call sites —
# validation_msgs + event_messages + tx_messages can converge on overlapping
# content during catch-up. This in-memory cache squashes the duplicate
# without affecting alerts for distinct rounds or distinct content.
#
# Key: (level, network, round_id, protocol_id, headline_120). round_id is
# IN the key, so different rounds DO fire separate alerts (a 5-round
# streak still produces 5 alerts — desired). TTL is short (300s) so
# transient state doesn't persist across long observer pauses.
_LOG_MESSAGE_DEDUP_CACHE: dict[tuple, float] = {}
_LOG_MESSAGE_DEDUP_TTL_SECONDS = 300


def _log_message_dedup_key(message: Message) -> tuple:
    return (
        message.level,
        message.network,
        message.round.id if message.round is not None else None,
        message.protocol,
        # Headline only (first line, capped) — the body content can shift
        # across catch-up scans even when alert semantics are the same.
        (message.message.split("\n", 1)[0] if message.message else "")[:120],
    )


def _log_message_should_dedup(message: Message) -> bool:
    import time as _time
    now = _time.time()
    # Evict stale entries opportunistically
    stale = [k for k, ts in _LOG_MESSAGE_DEDUP_CACHE.items()
             if now - ts > _LOG_MESSAGE_DEDUP_TTL_SECONDS]
    for k in stale:
        _LOG_MESSAGE_DEDUP_CACHE.pop(k, None)
    key = _log_message_dedup_key(message)
    if key in _LOG_MESSAGE_DEDUP_CACHE:
        return True
    _LOG_MESSAGE_DEDUP_CACHE[key] = now
    return False


# FlareWatch 2026-05-25: ongoing-issue coalescer. Sits BELOW the 5min
# dedup cache (which catches same-round duplicates from multiple
# call sites). This layer catches the same logical issue repeating
# across DIFFERENT rounds — the pattern that flooded the operator
# inbox during the 2026-05-24 XRP RPC fallback storm (one consensus-
# miss alert per round per attestation_type, ~12 alerts per voting
# cycle while GetBlock was quota-exhausted).
#
# Cadence per operator directive: first occurrence immediate;
# subsequent occurrences within 1h window suppressed + counted;
# next occurrence after window includes "+N suppressed in last 1h"
# preamble in the message body. Operator silence on the suppressed
# run is acceptable per feedback_rolling_subscription_reminders
# (silence = aware + still working the cause).
#
# Key DROPS round.id — distinct rounds of the same logical issue
# coalesce. The headline is captured INCLUDING the [data:<source>]
# prefix so per-chain consensus misses coalesce per-chain (one
# XRP-flood preamble, one DOGE-flood preamble) instead of fusing
# all chains into one bucket.
_COALESCE_CACHE: dict[tuple, dict] = {}  # key -> {first_at, last_at, count}
_COALESCE_WINDOW_SECONDS = 3600


def _coalesce_key(message: Message) -> tuple:
    return (
        message.level,
        message.network,
        message.protocol,
        (message.message.split("\n", 1)[0] if message.message else "")[:120],
    )


def _coalesce_decision(message: Message) -> tuple[bool, int]:
    """Returns (should_dispatch, suppressed_count_since_last_fire).

    - First occurrence (no state OR window expired): dispatch=True,
      suppressed_count = prior cycle's count (0 on first-ever).
    - Within window: dispatch=False, suppressed_count incremented.
    """
    import time as _time
    now = _time.time()
    key = _coalesce_key(message)
    state = _COALESCE_CACHE.get(key)
    if state is None:
        _COALESCE_CACHE[key] = {"first_at": now, "last_at": now, "count": 0}
        return True, 0
    if now - state["first_at"] > _COALESCE_WINDOW_SECONDS:
        prior_count = state["count"]
        _COALESCE_CACHE[key] = {"first_at": now, "last_at": now, "count": 0}
        return True, prior_count
    state["last_at"] = now
    state["count"] += 1
    return False, 0


def _coalesce_apply_preamble(message: Message, suppressed: int) -> Message:
    """Returns a new Message with the '+N suppressed' preamble injected.

    The preamble lands at the TOP of the message body so Discord
    embeds (which truncate at 4096 chars) keep it visible even if
    later body sections are clipped.
    """
    if suppressed <= 0:
        return message
    window_min = _COALESCE_WINDOW_SECONDS // 60
    preamble = (
        f"[STILL ONGOING] +{suppressed} similar alert(s) suppressed in "
        f"last ~{window_min} min; firing this update.\n\n"
    )
    import copy as _copy
    new_msg = _copy.copy(message)
    new_msg.message = preamble + message.message
    return new_msg


def log_message(config: Configuration, message: Message):
    # Always log to stdout/journald (no dedup at the logger layer — operator
    # may want full historical record on box). Dedup ONLY at the
    # downstream notification dispatch.
    LOGGER.log(message.level.value, message.message)

    n = config.notification
    # TODO:(@janezicmatej) this should be done eariler in the message lifecycle
    message.network = config.chain_id

    # FlareWatch S40: gate notifications by FLAREWATCH_PROTOCOLS_ACTIVE
    if not _flarewatch_should_dispatch(message):
        return

    # FlareWatch 2026-05-13: dedup the SAME alert content within a 5-min
    # window across all dispatch call sites.
    if _log_message_should_dedup(message):
        LOGGER.debug("log_message dedup'd; skipping notify for %s",
                     _log_message_dedup_key(message))
        return

    # FlareWatch 2026-05-25: ongoing-issue coalesce. After the 5-min
    # same-round dedup, collapse same-logical-issue alerts across
    # different rounds within a 1h window. First occurrence fires;
    # subsequent are suppressed + counted; next post-window fire
    # includes "+N suppressed" preamble.
    should_dispatch, suppressed = _coalesce_decision(message)
    if not should_dispatch:
        LOGGER.debug("log_message coalesced (window-suppressed); key=%s",
                     _coalesce_key(message))
        return
    if suppressed > 0:
        message = _coalesce_apply_preamble(message, suppressed)

    notify_discord(n.discord, message)
    notify_discord_embed(n.discord_embed, message)
    notify_slack(n.slack, message)
    notify_slack_embed(n.slack_embed, message)
    notify_telegram(n.telegram, message)
    notify_generic(n.generic, message)

    # FlareWatch: mirror this alert into the portal `posts` table (the
    # operator-facing source of truth the Activity page renders from).
    # Additive to the dispatch above and runs LAST — mirror_to_portal is
    # best-effort and never raises, so a portal/DB problem cannot affect
    # the Discord path. Inherits the 300s dedup gate above for free, so
    # one row lands per state, not per polling cycle.
    mirror_to_portal(config, message)


async def wait_until_registered(
    w: AsyncWeb3,
    config: Configuration,
    tia: ChecksumAddress,
    signing_policy: SigningPolicy,
    block_production: float,
) -> SigningPolicy:
    # the observer can only follow an entity that is part of the active signing
    # policy; while it isn't, wait for upcoming registration windows and rescan
    # signing policy events until the entity shows up in one
    vef = config.epoch.voting_epoch_factory
    ref = config.epoch.reward_epoch_factory

    while tia not in signing_policy.entity_mapper.by_identity_address:
        next_epoch = signing_policy.reward_epoch.next
        log_message(
            config,
            Message.builder()
            .add(network=config.chain_id)
            .build(
                MessageLevel.WARNING,
                (
                    f"Entity {tia} not registered for reward epoch"
                    f" {signing_policy.reward_epoch.id}, waiting for registration"
                    f" window of reward epoch {next_epoch.id}"
                ),
            ),
        )

        # voter registration for next_epoch runs in the 2h window before it
        # starts, so sleep until that window opens
        window_open_ts = next_epoch.start_s - SIGNING_POLICY_INITIALIZATION_S
        while (remaining := window_open_ts - int(time.time())) > 0:
            LOGGER.info(
                f"Waiting for reward epoch {next_epoch.id} registration window:"
                f" opens in {remaining}s"
            )
            metrics.VOTING_ROUND.set(vef.from_timestamp(int(time.time())).id)
            metrics.REWARD_EPOCH.set(ref.from_timestamp(int(time.time())).id)
            await asyncio.sleep(min(remaining, 3600))

        # with the window open, rescan until the signing policy for next_epoch is
        # initialized; it lands before the epoch starts unless the epoch is
        # extended
        while True:
            block = await w.eth.get_block("latest")
            assert "number" in block
            assert "timestamp" in block
            target_start_ts = window_open_ts - REGISTRATION_SCAN_BUFFER_S
            lower_block_id = await find_block_at_timestamp(
                w,
                block["number"],
                block["timestamp"],
                target_start_ts,
                block_production,
            )
            try:
                signing_policy = await get_signing_policy_events(
                    w, config, next_epoch, lower_block_id, block["number"]
                )
                break
            except AssertionError:
                LOGGER.info(
                    f"Signing policy for reward epoch {next_epoch.id}"
                    " not initialized yet, retrying in 60s"
                )
                await asyncio.sleep(60)

        nb_entities = len(signing_policy.entity_mapper.by_identity_address)
        LOGGER.info(
            f"Signing policy loaded: reward_epoch={signing_policy.reward_epoch.id}"
            f" | entities={nb_entities}"
            f" | starts_at_round={signing_policy.start_voting_round}"
        )

    log_message(
        config,
        Message.builder()
        .add(network=config.chain_id)
        .build(
            MessageLevel.INFO,
            (
                f"Entity {tia} registered in signing policy for reward epoch"
                f" {signing_policy.reward_epoch.id}"
            ),
        ),
    )

    # the new signing policy only takes effect at its start voting round; wait for
    # it before handing control back to the main loop
    start_ts = vef.make_epoch(signing_policy.start_voting_round).start_s
    while (remaining := start_ts - int(time.time())) > 0:
        LOGGER.info(
            f"Waiting for reward epoch {signing_policy.reward_epoch.id} to start:"
            f" {remaining}s"
        )
        await asyncio.sleep(min(remaining, 600))

    return signing_policy


async def cron(
    check_functions: Sequence[Awaitable[Sequence[Message]]],
) -> Sequence[Message]:
    results = await asyncio.gather(*check_functions)

    return list(chain.from_iterable(results))


def _record_submit_metrics(
    protocol: str, extracted, *, include_submit1: bool = True
) -> None:
    phases = []
    if include_submit1:
        phases.append(("submit1", extracted.submit_1))
    phases.append(("submit2", extracted.submit_2))
    phases.append(("signatures", extracted.submit_signatures))

    for phase, ext in phases:
        if ext.extracted is not None:
            metrics.SUBMIT_OK.labels(
                identity_address=metrics.identity_address,
                protocol=protocol,
                phase=phase,
            ).inc()
        elif ext.late:
            metrics.SUBMIT_LATE.labels(
                identity_address=metrics.identity_address,
                protocol=protocol,
                phase=phase,
            ).inc()
        elif ext.early:
            metrics.SUBMIT_EARLY.labels(
                identity_address=metrics.identity_address,
                protocol=protocol,
                phase=phase,
            ).inc()
        else:
            metrics.SUBMIT_MISSING.labels(
                identity_address=metrics.identity_address,
                protocol=protocol,
                phase=phase,
            ).inc()


async def observer_loop(config: Configuration) -> None:
    logging.getLogger().setLevel(config.log_level)
    w = AsyncWeb3(
        AsyncWeb3.AsyncHTTPProvider(config.rpc_url),
        middleware=[ExtraDataToPOAMiddleware],
    )

    # reasignments for quick access
    ve = config.epoch.voting_epoch
    # re = config.epoch.reward_epoch
    vef = config.epoch.voting_epoch_factory
    ref = config.epoch.reward_epoch_factory

    # set up target address from config (must come before any labeled metric calls)
    tia = w.to_checksum_address(config.identity_address)
    metrics.setup(tia)

    # get current voting round and reward epoch
    block = await w.eth.get_block("latest")
    assert "timestamp" in block
    assert "number" in block
    reward_epoch = ref.from_timestamp(block["timestamp"])
    voting_epoch = vef.from_timestamp(block["timestamp"])

    metrics.VOTING_ROUND.set(voting_epoch.id)
    metrics.REWARD_EPOCH.set(reward_epoch.id)
    metrics.REGISTERED_CURRENT_EPOCH.labels(identity_address=tia).set(0)
    metrics.REGISTERED_NEXT_EPOCH.labels(identity_address=tia).set(0)

    LOGGER.info(
        f"Block #{block['number']}"
        f" | reward_epoch={reward_epoch.id}"
        f" | voting_epoch={voting_epoch.id}"
    )

    # block production rate differs per chain, so measure it and reuse it both to
    # locate the registration blocks and to size the fast updates exponent window
    # FlareWatch: pass the configured lookback (pruned local RPC nodes don't
    # retain the upstream default 1M blocks of history).
    block_production = await get_block_production(w, config.block_production_lookback)
    maximum_exponent = calculate_maximum_exponent(block_production, config)

    LOGGER.debug(
        f"Block production: {block_production:.3f}s/block"
        f" | max_exponent={maximum_exponent}"
    )

    # we first fill signing policy for current reward epoch

    # the signing policy is initialized in the 2h before the reward epoch; find the
    # block range spanning that window to read the events that build it
    lower_block_id, end_block_id = await find_voter_registration_blocks(
        w, block["number"], reward_epoch, block_production
    )

    # get informations for events that build the current signing policy
    signing_policy = await get_signing_policy_events(
        w,
        config,
        reward_epoch,
        lower_block_id,
        end_block_id,
    )
    nb_entities = len(signing_policy.entity_mapper.by_identity_address)
    LOGGER.info(
        f"Signing policy loaded: reward_epoch={reward_epoch.id}"
        f" | entities={nb_entities}"
        f" | starts_at_round={signing_policy.start_voting_round}"
    )
    # started before the potential registration wait below so the observer stays
    # observable while it waits
    if config.metrics.enabled:
        metrics.start_metrics_server(config.metrics.port, config.metrics.address)

    if tia not in signing_policy.entity_mapper.by_identity_address:
        LOGGER.warning(f"Entity {tia} NOT found in current signing policy!")
        signing_policy = await wait_until_registered(
            w, config, tia, signing_policy, block_production
        )
        reward_epoch = signing_policy.reward_epoch

        # the wait can span multiple reward epochs, refresh chain related state
        block = await w.eth.get_block("latest")
        assert "timestamp" in block
        assert "number" in block
        voting_epoch = vef.from_timestamp(block["timestamp"])
        metrics.VOTING_ROUND.set(voting_epoch.id)
        metrics.REWARD_EPOCH.set(reward_epoch.id)

    spb = SigningPolicy.builder().for_epoch(reward_epoch.next)

    # print("Signing policy created for reward epoch", current_rid)
    # print("Reward Epoch object created", reward_epoch_info)
    # print("Current Reward Epoch status", reward_epoch_info.status(config))

    metrics.REGISTERED_CURRENT_EPOCH.labels(identity_address=tia).set(1)

    _init_entity = signing_policy.entity_mapper.by_identity_address[tia]
    _init_node_ids = [node_id_to_representation(n.node_id) for n in _init_entity.nodes]
    metrics.initialize_labels(node_ids=_init_node_ids)

    LOGGER.info(
        f"Entity found in signing policy:"
        f" submit={_init_entity.submit_address}"
        f" | nodes={_init_node_ids}"
    )
    await _init_entity.check_addresses(config, w)
    _unclaimed_init = await RewardManager().get_unclaimed_rewards(
        _init_entity, config, w
    )
    for m in _unclaimed_init:
        log_message(config, m)

    LOGGER.info(f"Connecting to RPC: {config.rpc_url}")

    # TODO:(matej) log version and initial voting round, maybe signing policy info
    log_message(
        config,
        Message.builder()
        .add(network=config.chain_id)
        .build(
            MessageLevel.INFO,
            f"Initialized observer for identity_address={tia}",
        ),
    )

    LOGGER.info(
        f"Observer ready: identity={tia}"
        f" | reward_epoch={reward_epoch.id}"
        f" | voting_epoch={voting_epoch.id}"
    )

    cron_time = time.time()

    # wait until next voting epoch
    block_number = block["number"]
    while True:
        latest_block = await w.eth.block_number
        if block_number == latest_block:
            time.sleep(2)
            continue

        block_number += 1
        block_data = await w.eth.get_block(block_number)

        assert "timestamp" in block_data

        _ve = vef.from_timestamp(block_data["timestamp"])
        if _ve == voting_epoch.next:
            voting_epoch = voting_epoch.next
            break

    vrm = VotingRoundManager(voting_epoch.previous.id)

    # set up contracts and events (from config)
    cm = ContractManager(config.contracts)
    contracts = cm.get_contracts_list()
    event_signatures = cm.get_events()

    entity = signing_policy.entity_mapper.by_identity_address[tia]
    fum = FastUpdatesManager(
        block_number, FastUpdate(reward_epoch.id, entity.signing_policy_address, [])
    )
    spm = SigningPolicyManager(signing_policy, signing_policy)
    rm = RewardManager()

    # check transactions for submit transactions
    target_function_signatures = {
        config.contracts.Submission.functions[
            "submitSignatures"
        ].signature: "submitSignatures",
        config.contracts.Submission.functions["submit1"].signature: "submit1",
        config.contracts.Submission.functions["submit2"].signature: "submit2",
    }

    minimal_conditions = (
        MinimalConditions()
        .for_network(config.chain_id)
        .for_reward_epoch(reward_epoch.id)
        .set_false_positive_threshold(config.false_positive_threshold)
    )
    last_minimal_conditions_check = int(time.time())
    last_ping = int(time.time())
    last_registration_check = int(time.time())

    uptime_validation_frequency = 60
    node_connections = defaultdict(
        partial(
            deque[bool],
            maxlen=minimal_conditions.time_period.value // uptime_validation_frequency,
        )
    )
    uptime_validations = 0

    medians: deque[list[FtsoMedian | None]] = deque(
        maxlen=minimal_conditions.time_period.value // 90
    )
    entity_votes: deque[list[int | None]] = deque(
        maxlen=minimal_conditions.time_period.value // 90
    )

    signatures: deque[bool] = deque(maxlen=minimal_conditions.time_period.value // 90)

    voter_registration_started: bool = False
    voter_registration_started_ts: int = 0
    registered: bool = False

    nr_of_feeds: int = 0
    fast_update_re: int = 0

    messages: list[Message] = []
    min_cond_messages: list[Message] = []

    while True:
        latest_block = await w.eth.block_number
        if block_number == latest_block:
            time.sleep(2)
            continue

        for block in range(block_number, latest_block):
            LOGGER.debug(f"processing {block}")
            block_data = await w.eth.get_block(block, full_transactions=True)
            assert "transactions" in block_data
            assert "timestamp" in block_data
            block_ts = block_data["timestamp"]

            voting_epoch = vef.from_timestamp(block_ts)
            metrics.VOTING_ROUND.set(voting_epoch.id)

            LOGGER.debug(
                f"Block #{block}"
                f" | voting_epoch={voting_epoch.id}"
                f" | txs={len(block_data['transactions'])}"
            )

            if (
                spb.signing_policy_initialized is not None
                and spb.signing_policy_initialized.start_voting_round_id
                == voting_epoch.id
            ):
                # TODO:(matej) this could fail if the observer is started during
                # last two hours of the reward epoch
                voter_registration_started = False
                registered = False
                spm.previous_policy = signing_policy
                old_epoch_id = signing_policy.reward_epoch.id
                signing_policy = spb.build()
                spm.current_policy = signing_policy
                metrics.REWARD_EPOCH.set(signing_policy.reward_epoch.id)
                metrics.REGISTERED_CURRENT_EPOCH.labels(
                    identity_address=metrics.identity_address
                ).set(
                    1 if tia in signing_policy.entity_mapper.by_identity_address else 0
                )
                metrics.REGISTERED_NEXT_EPOCH.labels(
                    identity_address=metrics.identity_address
                ).set(0)

                spb = SigningPolicy.builder().for_epoch(
                    signing_policy.reward_epoch.next
                )

                nb_new = len(signing_policy.entity_mapper.by_identity_address)
                LOGGER.info(
                    f"Epoch transition: {old_epoch_id}"
                    f" → {signing_policy.reward_epoch.id}"
                    f" | entities={nb_new}"
                    f" | starts_at_round={signing_policy.start_voting_round}"
                )

                minimal_conditions.reward_epoch_id = signing_policy.reward_epoch.id

                entity = signing_policy.entity_mapper.by_identity_address[tia]
                unclaimed_rewards = await rm.get_unclaimed_rewards(entity, config, w)
                for m in unclaimed_rewards:
                    log_message(config, m)

            block_logs = await w.eth.get_logs(
                {
                    "address": [contract.address for contract in contracts],
                    "fromBlock": block,
                    "toBlock": block,
                }
            )
            _relay_patch_sps = await w.eth.get_logs(
                {
                    "address": [
                        to_checksum_address(
                            "0x92a6E1127262106611e1e129BB64B6D8654273F7"
                        ),
                        to_checksum_address(
                            "0x97702e350CaEda540935d92aAf213307e9069784"
                        ),
                        to_checksum_address(
                            "0x57a4c3676d08Aa5d15410b5A6A80fBcEF72f3F45"
                        ),
                        to_checksum_address(
                            "0x67a916E175a2aF01369294739AA60dDdE1Fad189"
                        ),
                    ],
                    "fromBlock": block,
                    "toBlock": block,
                    "topics": [
                        "0x"
                        + config.contracts.Relay.events[
                            "SigningPolicyInitialized"
                        ].signature
                    ],
                }
            )
            block_logs.extend(_relay_patch_sps)

            tx_messages = []
            event_messages = []
            for log in block_logs:
                sig = log["topics"][0]

                if sig.hex() in event_signatures:
                    event = event_signatures[sig.hex()]
                    data = get_event_data(w.eth.codec, event.abi, log)
                    match event.name:
                        case "ProtocolMessageRelayed":
                            e = ProtocolMessageRelayed.from_dict(
                                data["args"], block_data
                            )
                            voting_round = vrm.get(ve(e.voting_round_id))
                            if e.protocol_id == 100:
                                voting_round.ftso.finalization = e
                            if e.protocol_id == 200:
                                voting_round.fdc.finalization = e

                            # this had to be sent to Relay
                            # so we can check if the Relay address changed
                            entity = signing_policy.entity_mapper.by_identity_address[
                                tia
                            ]
                            tx = await w.eth.get_transaction(data["transactionHash"])
                            assert "to" in tx
                            assert "from" in tx
                            if tx["from"] == entity.identity_address:
                                relay_address = tx["to"]
                                event_messages.extend(
                                    cm.check_relay_address(relay_address)
                                )

                        case "AttestationRequest":
                            e = AttestationRequest.from_dict(data, voting_epoch)
                            vrm.get(e.voting_epoch_id).fdc.requests.agg.append(e)

                        case "SigningPolicyInitialized":
                            e = SigningPolicyInitialized.from_dict(data["args"])
                            spb.add(e)
                            LOGGER.info(
                                f"SigningPolicyInitialized:"
                                f" reward_epoch={e.reward_epoch_id}"
                                f" | starts_at_round={e.start_voting_round_id}"
                                f" | voters={len(e.voters)}"
                            )
                        case "VoterRegistered":
                            e = VoterRegistered.from_dict(data["args"])
                            spb.add(e)
                            if registered:
                                continue
                            entity = signing_policy.entity_mapper.by_identity_address[
                                tia
                            ]
                            if (
                                entity.signing_policy_address
                                == e.signing_policy_address
                            ):
                                registered = True
                                metrics.REGISTERED_NEXT_EPOCH.labels(
                                    identity_address=metrics.identity_address
                                ).set(1)
                                LOGGER.info(
                                    f"VoterRegistered: our entity registered"
                                    f" for epoch {e.reward_epoch_id}"
                                )
                        case "VoterRemoved":
                            e = VoterRemoved.from_dict(data["args"])
                            spb.add(e)
                        case "VoterRegistrationInfo":
                            e = VoterRegistrationInfo.from_dict(data["args"])
                            spb.add(e)
                        case "VotePowerBlockSelected":
                            e = VotePowerBlockSelected.from_dict(data["args"])
                            spb.add(e)
                            LOGGER.info(
                                f"VotePowerBlockSelected:"
                                f" epoch={e.reward_epoch_id}"
                                f" | vote_power_block=#{e.vote_power_block}"
                                f" | registration window open"
                            )
                            if registered:
                                continue
                            voter_registration_started = True
                            voter_registration_started_ts = int(time.time())
                        case "RandomAcquisitionStarted":
                            e = RandomAcquisitionStarted.from_dict(data["args"])
                            spb.add(e)
                        case "FastUpdateFeedsSubmitted":
                            e = FastUpdateFeedsSubmitted.from_dict(data)
                            tx = await w.eth.get_transaction(e.transaction_hash)
                            spa, address, update_array = calculate_update_from_tx(
                                config, w, tx
                            )
                            entity = signing_policy.entity_mapper.by_identity_address[
                                tia
                            ]
                            if un_prefix_0x(entity.signing_policy_address) == spa:
                                fum.last_update = FastUpdate(
                                    signing_policy.reward_epoch.id,
                                    address,
                                    update_array,
                                )
                                fum.last_update_block = int(data["blockNumber"])
                                # We check update array when we receive a new one
                                event_messages.extend(
                                    fum.check_update_length(nr_of_feeds, fast_update_re)
                                )
                                fum.address_list.add(address)
                                LOGGER.info(
                                    f"FastUpdateFeedsSubmitted:"
                                    f" our entity at block #{fum.last_update_block}"
                                    f" | feeds={len(update_array)}"
                                )
                        case "FastUpdateFeeds":
                            e = FastUpdateFeeds.from_dict(data)
                            nr_of_feeds, fast_update_re = (
                                len(e.feeds),
                                ref.from_voting_epoch(
                                    vef.make_epoch(e.voting_round_id)
                                ).id,
                            )
                        case "VoterPreRegistered":
                            e = VoterPreRegistered.from_dict(data)
                            entity = signing_policy.entity_mapper.by_identity_address[
                                tia
                            ]
                            if tia == e.voter:
                                registered = True
                                metrics.REGISTERED_NEXT_EPOCH.labels(
                                    identity_address=metrics.identity_address
                                ).set(1)

            for tx in block_data["transactions"]:
                assert not isinstance(tx, bytes)
                wtx = WTxData.from_tx_data(tx, block_data)

                called_function_sig = wtx.input[:4].hex()
                input = wtx.input[4:].hex()
                sender_address = wtx.from_address
                entity = signing_policy.entity_mapper.by_omni.get(sender_address)
                if entity is None:
                    continue
                target_entity = signing_policy.entity_mapper.by_identity_address[tia]

                if called_function_sig in target_function_signatures:
                    # check if the Submission address is correct
                    if entity == target_entity:
                        tx_messages.extend(cm.check_submission_address(wtx.to_address))
                    mode = target_function_signatures[called_function_sig]
                    match mode:
                        case "submit1":
                            try:
                                parsed = parse_submit1_tx(input)
                                if parsed.ftso is not None:
                                    vrm.get(
                                        ve(parsed.ftso.voting_round_id)
                                    ).ftso.insert_submit_1(entity, parsed.ftso, wtx)
                                    if entity == target_entity:
                                        LOGGER.info(
                                            f"submit1 FTSO: our entity at"
                                            f" block #{block}"
                                            f" | round="
                                            f"{parsed.ftso.voting_round_id}"
                                        )
                                if parsed.fdc is not None:
                                    vrm.get(
                                        ve(parsed.fdc.voting_round_id)
                                    ).fdc.insert_submit_1(entity, parsed.fdc, wtx)
                            except Exception:
                                pass

                        case "submit2":
                            try:
                                parsed = parse_submit2_tx(input)
                                if parsed.ftso is not None:
                                    vrm.get(
                                        ve(parsed.ftso.voting_round_id)
                                    ).ftso.insert_submit_2(entity, parsed.ftso, wtx)
                                    if entity == target_entity:
                                        LOGGER.info(
                                            f"submit2 FTSO: our entity at"
                                            f" block #{block}"
                                            f" | round="
                                            f"{parsed.ftso.voting_round_id}"
                                        )
                                if parsed.fdc is not None:
                                    vrm.get(
                                        ve(parsed.fdc.voting_round_id)
                                    ).fdc.insert_submit_2(entity, parsed.fdc, wtx)
                                    if entity == target_entity:
                                        LOGGER.info(
                                            f"submit2 FDC: our entity at"
                                            f" block #{block}"
                                            f" | round="
                                            f"{parsed.fdc.voting_round_id}"
                                        )
                            except Exception:
                                pass

                        case "submitSignatures":
                            try:
                                parsed = parse_submit_signature_tx(input)
                                if parsed.ftso is not None:
                                    vrm.get(
                                        ve(parsed.ftso.voting_round_id)
                                    ).ftso.insert_submit_signatures(
                                        entity, parsed.ftso, wtx
                                    )
                                    if entity == target_entity:
                                        LOGGER.info(
                                            f"submitSignatures FTSO:"
                                            f" our entity at block #{block}"
                                            f" | round="
                                            f"{parsed.ftso.voting_round_id}"
                                        )
                                if parsed.fdc is not None:
                                    vr = vrm.get(ve(parsed.fdc.voting_round_id))
                                    vr.fdc.insert_submit_signatures(
                                        entity, parsed.fdc, wtx
                                    )

                                    # NOTE:(matej) this is currently the easiest
                                    # way to get consensus bitvote
                                    vr.fdc.consensus_bitvote[
                                        parsed.fdc.payload.unsigned_message
                                    ] += 1

                                    if entity == target_entity:
                                        LOGGER.info(
                                            f"submitSignatures FDC:"
                                            f" our entity at block #{block}"
                                            f" | round="
                                            f"{parsed.fdc.voting_round_id}"
                                        )

                            except Exception:
                                pass

            messages.clear()
            messages.extend(tx_messages)
            messages.extend(event_messages)
            entity = signing_policy.entity_mapper.by_identity_address[tia]

            # perform all minimal condition checks here
            if int(time.time() - last_minimal_conditions_check) > 60:
                metrics.FAST_UPDATE_BLOCKS_SINCE_LAST.labels(
                    identity_address=metrics.identity_address
                ).set(block - fum.last_update_block)
                min_cond_messages.clear()

                min_cond_messages.extend(
                    minimal_conditions.calculate_ftso_block_latency_feeds(
                        maximum_exponent,
                        entity,
                        signing_policy,
                        fum.last_update_block,
                        block,
                    )
                )
                min_cond_messages.extend(
                    minimal_conditions.calculate_ftso_anchor_feeds(
                        medians, entity_votes
                    )
                )
                min_cond_messages.extend(
                    minimal_conditions.calculate_staking(
                        uptime_validations, node_connections
                    )
                )
                min_cond_messages.extend(
                    minimal_conditions.calculate_fdc_participation(signatures)
                )

                for m in min_cond_messages:
                    log_message(config, m)

                last_minimal_conditions_check = int(time.time())

            node_ids = [
                node_id_to_representation(node.node_id) for node in entity.nodes
            ]
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "platform.getCurrentValidators",
                "params": {"nodeIDs": node_ids},
            }
            if (
                int(time.time() - last_ping) > uptime_validation_frequency
                and len(node_ids) > 0
            ):
                try:
                    response = requests.post(
                        config.p_chain_rpc_url, json=payload, timeout=10
                    )
                    response.raise_for_status()
                    result = response.json()
                    if "error" in result:
                        LOGGER.warning("Error calling API: check params")
                        continue

                    if (
                        uptime_validations
                        < minimal_conditions.time_period.value
                        // uptime_validation_frequency
                    ):
                        uptime_validations += 1
                    for node in result["result"]["validators"]:
                        node_connections[node["nodeID"]].append(node["connected"])
                        history = node_connections[node["nodeID"]]
                        metrics.NODE_UPTIME_RATIO.labels(
                            identity_address=metrics.identity_address,
                            node_id=node["nodeID"],
                        ).set(sum(history) / len(history) if history else 0)
                except requests.RequestException as e:
                    LOGGER.warning(f"Error calling API: {e}")
                last_ping = int(time.time())

            if int(time.time()) - cron_time > 60 * 60:
                cron_time = int(time.time())
                check_functions = [
                    entity.check_addresses(config, w),
                    fum.check_addresses(config, w),
                ]
                messages.extend(await cron(check_functions))

            rounds = vrm.finalize(block_data)
            # prepare new data for anchor feeds
            if len(rounds) > 0:
                medians.extend([round.ftso.medians for round in rounds])
                for round in rounds:
                    extracted_ftso = extract_round_for_entity(
                        round.ftso, entity, round.voting_epoch
                    ).submit_2.extracted
                    if extracted_ftso is not None:
                        votes = extracted_ftso.parsed_payload.payload.values
                        entity_votes.append(votes)
                    else:
                        entity_votes.append([])
            for r in rounds:
                ftso_fin = "YES" if r.ftso.finalization else "NO"
                fdc_fin = "YES" if r.fdc.finalization else "NO"
                nb_medians = len(r.ftso.medians) if r.ftso.medians else 0
                validation_msgs = validate_round(r, signing_policy, entity, config)
                if validation_msgs:
                    LOGGER.warning(
                        f"Round {r.voting_epoch.id}:"
                        f" {len(validation_msgs)} issue(s)"
                        f" | FTSO={ftso_fin} FDC={fdc_fin}"
                    )
                else:
                    LOGGER.info(
                        f"Round {r.voting_epoch.id}: OK"
                        f" | FTSO={ftso_fin} FDC={fdc_fin}"
                        f" medians={nb_medians}"
                    )
                messages.extend(validation_msgs)
                _record_submit_metrics(
                    "ftso", extract_round_for_entity(r.ftso, entity, r.voting_epoch)
                )
                _record_submit_metrics(
                    "fdc",
                    extract_round_for_entity(r.fdc, entity, r.voting_epoch),
                    include_submit1=False,
                )

            # prepare new data for FDC participation
            signatures.extend([round.submitted_signatures for round in rounds])

            # reporting on registration and preregistration once per minute
            interval = 60
            if int(time.time() - voter_registration_started_ts) > 15 * 60:
                interval = 10
            if (
                int(time.time() - last_registration_check) > interval
                and voter_registration_started
                and not registered
            ):
                mb = Message.builder().add(network=config.chain_id)
                if int(time.time() - voter_registration_started_ts) > 60:
                    level = MessageLevel.CRITICAL
                    message = mb.build(
                        level,
                        (
                            "Voter not registered after "
                            f"{int(time.time() - voter_registration_started_ts) // 60}"
                            " minutes"
                        ),
                    )
                    messages.append(message)

            for m in messages:
                log_message(config, m)

        block_number = latest_block
