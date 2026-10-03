from prometheus_client import Counter, Gauge, start_http_server

# Identity address set once at startup via setup()
identity_address: str = ""


def setup(ia: str) -> None:
    global identity_address
    identity_address = ia


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
# FDC consensus / submit2 quality (logger-only events promoted to Counters)
# ---------------------------------------------------------------------------
#
# These three counters back ERROR-class log messages emitted from
# observer/validation/fdc.py that were previously logger-only — no
# Prometheus surface meant downstream consumers (FlareWatch agent's C9
# cluster surface, F1-derived alarm playbooks) had no way to track
# rate or count.
#
# Cardinality note (FDC_SUBMIT2_CONSENSUS_MISS):
# attestation_type and source_id are both 32-byte fields per
# py_flare_common, so theoretically unbounded. In practice the Flare
# protocol uses a small known set (~7 attestation types × ~6 source
# ids = ~42 max combos per identity). Acceptable as Prometheus labels;
# a future protocol expansion that pushes cardinality high enough to
# matter would need to revisit (move to JSON payload field). Counter
# is NOT pre-initialized via initialize_labels — labels are added on
# first emission so we don't enumerate combos we don't actually see.

FDC_SUBMIT1_UNEXPECTED = Counter(
    "flare_fsp_fdc_submit1_unexpected_total",
    "Total rounds where an FDC submit1 transaction was found "
    "(FDC protocol does not use submit1; presence indicates misconfig)",
    ["identity_address"],
)

FDC_SUBMIT2_BIT_VOTE_LENGTH_MISMATCH = Counter(
    "flare_fsp_fdc_submit2_bit_vote_length_mismatch_total",
    "Total rounds where the FDC submit2 bit-vote length did not match "
    "the number of consensus requests in the round",
    ["identity_address"],
)

FDC_SUBMIT2_CONSENSUS_MISS = Counter(
    "flare_fsp_fdc_submit2_consensus_miss_total",
    "Total per-(attestation_type, source_id) consensus requests where "
    "submit2 did not confirm a request that was part of consensus. "
    "Operators may have intentionally opted out of certain "
    "(attestation_type, source_id) combos via downstream policy; "
    "this counter records the raw observation regardless.",
    ["identity_address", "attestation_type", "source_id"],
)

# ---------------------------------------------------------------------------
# Unclaimed rewards
# ---------------------------------------------------------------------------

UNCLAIMED_REWARDS = Gauge(
    "flare_fsp_unclaimed_rewards_wei",
    "Unclaimed reward amount in wei per address/epoch/claim_type",
    ["identity_address", "address", "reward_epoch", "claim_type"],
)

UNCLAIMED_VALIDATOR_REWARDS = Gauge(
    "flare_fsp_unclaimed_validator_rewards_wei",
    "Unclaimed validator (P-chain stake) reward amount in wei per address",
    ["identity_address", "address"],
)


def initialize_labels(node_ids: list[str] | None = None) -> None:
    """Pre-initialize all label combinations so time series appear immediately at 0."""
    assert identity_address, "metrics.setup() must be called before initialize_labels()"
    for protocol, phases in [
        ("ftso", ["submit1", "submit2", "signatures"]),
        ("fdc", ["submit2", "signatures"]),
    ]:
        for phase in phases:
            SUBMIT_OK.labels(
                identity_address=identity_address, protocol=protocol, phase=phase
            )
            SUBMIT_MISSING.labels(
                identity_address=identity_address, protocol=protocol, phase=phase
            )
            SUBMIT_LATE.labels(
                identity_address=identity_address, protocol=protocol, phase=phase
            )
            SUBMIT_EARLY.labels(
                identity_address=identity_address, protocol=protocol, phase=phase
            )

        REVEAL_OFFENCE.labels(identity_address=identity_address, protocol=protocol)
        SIGNATURE_GRACE_PERIOD_MISSED.labels(
            identity_address=identity_address, protocol=protocol
        )
        SIGNATURE_MISMATCH.labels(identity_address=identity_address, protocol=protocol)

    for contract in ["submission", "relay"]:
        CONTRACT_ADDRESS_WRONG.labels(
            identity_address=identity_address, contract=contract
        )

    FAST_UPDATE_BLOCKS_SINCE_LAST.labels(identity_address=identity_address).set(0)

    if node_ids:
        for node_id in node_ids:
            NODE_UPTIME_RATIO.labels(
                identity_address=identity_address, node_id=node_id
            ).set(0)


def start_metrics_server(port: int, address: str = "0.0.0.0") -> None:
    start_http_server(port, addr=address)
