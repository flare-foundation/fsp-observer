from prometheus_client import Counter, Gauge, start_http_server

# Identity address set once at startup via setup()
_ia: str = ""


def setup(identity_address: str) -> None:
    global _ia
    _ia = identity_address


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

VOTING_ROUND = Gauge(
    "flare_fsp_voting_round_current",
    "Current voting round ID",
)

REWARD_EPOCH = Gauge(
    "flare_fsp_reward_epoch_current",
    "Current reward epoch ID",
)

REGISTERED_CURRENT_EPOCH = Gauge(
    "flare_fsp_registered_current_epoch",
    "Whether the entity is in the active signing policy for the current epoch (0 or 1)",
    ["identity_address"],
)

REGISTERED_NEXT_EPOCH = Gauge(
    "flare_fsp_registered_next_epoch",
    "Whether the entity has registered for the next epoch (0 or 1)",
    ["identity_address"],
)

# ---------------------------------------------------------------------------
# Submissions — Counters per protocol (ftso, fdc) and phase
# (submit1, submit2, signatures)
# ---------------------------------------------------------------------------

SUBMIT_OK = Counter(
    "flare_fsp_submit_ok_total",
    "Total rounds where submission was present and valid",
    ["identity_address", "protocol", "phase"],
)

SUBMIT_MISSING = Counter(
    "flare_fsp_submit_missing_total",
    "Total rounds where submission was absent",
    ["identity_address", "protocol", "phase"],
)

SUBMIT_LATE = Counter(
    "flare_fsp_submit_late_total",
    "Total rounds where submission was sent after the allowed window",
    ["identity_address", "protocol", "phase"],
)

SUBMIT_EARLY = Counter(
    "flare_fsp_submit_early_total",
    "Total rounds where submission was sent before the allowed window",
    ["identity_address", "protocol", "phase"],
)

# ---------------------------------------------------------------------------
# Minimal conditions
# ---------------------------------------------------------------------------

FTSO_ANCHOR_FEEDS_SUCCESS_RATE = Gauge(
    "flare_fsp_ftso_anchor_feeds_success_rate_bips",
    "FTSO anchor feeds success rate in bips (0-10000) over the last 2 hours",
    ["identity_address"],
)

FAST_UPDATE_BLOCKS_SINCE_LAST = Gauge(
    "flare_fsp_fast_update_blocks_since_last",
    "Number of blocks elapsed since the last fast update submission",
    ["identity_address"],
)

NODE_UPTIME_RATIO = Gauge(
    "flare_fsp_node_uptime_ratio",
    "Node uptime ratio over the sliding window (0.0 to 1.0)",
    ["identity_address", "node_id"],
)

FDC_PARTICIPATION_RATE = Gauge(
    "flare_fsp_fdc_participation_rate_bips",
    "FDC participation rate in bips (0-10000) over the last 2 hours",
    ["identity_address"],
)

# ---------------------------------------------------------------------------
# Balance
# ---------------------------------------------------------------------------

ADDRESS_BALANCE = Gauge(
    "flare_fsp_address_balance_wei",
    "Address balance in wei",
    ["identity_address", "address", "role"],
)

# ---------------------------------------------------------------------------
# Validation issues
# ---------------------------------------------------------------------------

REVEAL_OFFENCE = Counter(
    "flare_fsp_reveal_offence_total",
    "Total rounds where a reveal offence occurred"
    " (missing reveal after commit, or hash mismatch)",
    ["identity_address", "protocol"],
)

SIGNATURE_GRACE_PERIOD_MISSED = Counter(
    "flare_fsp_signature_grace_period_missed_total",
    "Total rounds where submitSignatures was sent after the grace period deadline",
    ["identity_address", "protocol"],
)

SIGNATURE_MISMATCH = Counter(
    "flare_fsp_signature_mismatch_total",
    "Total rounds where submitSignatures signature did not match finalization",
    ["identity_address", "protocol"],
)

# ---------------------------------------------------------------------------
# Contract address issues
# ---------------------------------------------------------------------------

CONTRACT_ADDRESS_WRONG = Counter(
    "flare_fsp_contract_address_wrong_total",
    "Total times a wrong contract address was detected (submission or relay)",
    ["identity_address", "contract"],
)

# ---------------------------------------------------------------------------
# Unclaimed rewards
# ---------------------------------------------------------------------------

UNCLAIMED_REWARDS = Gauge(
    "flare_fsp_unclaimed_rewards_wei",
    "Unclaimed reward amount in wei per address/epoch/claim_type",
    ["identity_address", "address", "reward_epoch", "claim_type"],
)


def initialize_labels(node_ids: list[str] | None = None) -> None:
    """Pre-initialize all label combinations so time series appear immediately at 0."""
    for protocol, phases in [
        ("ftso", ["submit1", "submit2", "signatures"]),
        ("fdc", ["submit2", "signatures"]),
    ]:
        for phase in phases:
            SUBMIT_OK.labels(identity_address=_ia, protocol=protocol, phase=phase)
            SUBMIT_MISSING.labels(identity_address=_ia, protocol=protocol, phase=phase)
            SUBMIT_LATE.labels(identity_address=_ia, protocol=protocol, phase=phase)
            SUBMIT_EARLY.labels(identity_address=_ia, protocol=protocol, phase=phase)

        REVEAL_OFFENCE.labels(identity_address=_ia, protocol=protocol)
        SIGNATURE_GRACE_PERIOD_MISSED.labels(identity_address=_ia, protocol=protocol)
        SIGNATURE_MISMATCH.labels(identity_address=_ia, protocol=protocol)

    for contract in ["submission", "relay"]:
        CONTRACT_ADDRESS_WRONG.labels(identity_address=_ia, contract=contract)

    FAST_UPDATE_BLOCKS_SINCE_LAST.labels(identity_address=_ia).set(0)

    if node_ids:
        for node_id in node_ids:
            NODE_UPTIME_RATIO.labels(identity_address=_ia, node_id=node_id).set(0)


def start_metrics_server(port: int, address: str = "0.0.0.0") -> None:
    start_http_server(port, addr=address)
