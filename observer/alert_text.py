"""4-section diagnostic alert bodies for fsp-observer ERROR alerts.

Per 2026-05-13 operator directive aligning to agent-side TrainingNotifier
format. Every per-round ERROR alert from observer/validation/* should use
build_alert() to construct the message body. Diagnostic + action text lives
here so future refinements happen in one place instead of scattered through
validation/*.

Body shape (rendered by Discord embed as the description):

    <summary line>

    DIAGNOSIS
    <2-3 paragraphs distinguishing real-outage from false-positive>

    EVIDENCE
      key1  value1
      key2  value2
      ...

    OPERATOR ACTIONS
    <if-real-outage branch>
    <if-false-positive branch>

Per-alert body sizes target ~1500-2000 chars; Discord embed description
limit is 4096 chars. Multi-round streaks (5+ consecutive alerts) stay
under Discord's per-channel rate-limit even with verbose bodies.
"""
from typing import Any


def build_alert(
    summary: str,
    diagnosis: str,
    evidence: dict[str, Any],
    actions: str,
) -> str:
    """Build a 4-section diagnostic alert body.

    summary   - First line; becomes the Discord embed title prefix
                (alongside the network/round/protocol prefix added by
                Message.build_str()).
    diagnosis - Multi-line paragraph(s) explaining what likely happened.
                Should distinguish real-outage from false-positive when
                applicable.
    evidence  - dict of key->value rendered as 'key:value' lines under the
                EVIDENCE header. Keys are left-padded for column alignment.
    actions   - Multi-line paragraph(s) describing what the operator should
                do. Usually has two branches ('If REAL outage' vs 'If
                FALSE POSITIVE'); concrete commands inline.
    """
    if evidence:
        key_width = max(len(k) for k in evidence.keys())
        evidence_lines = "\n".join(
            f"  {k:<{key_width}}  {v}" for k, v in evidence.items()
        )
    else:
        evidence_lines = "  (no additional evidence captured)"

    return (
        f"{summary}\n"
        f"\n"
        f"DIAGNOSIS\n"
        f"{diagnosis.strip()}\n"
        f"\n"
        f"EVIDENCE\n"
        f"{evidence_lines}\n"
        f"\n"
        f"OPERATOR ACTIONS\n"
        f"{actions.strip()}"
    )


# ──────────────────────────────────────────────────────────────────────
# Diagnostic + action templates per alert class.
#
# Each entry is a dict with `diagnosis` + `actions` strings. Call site
# pulls these and passes them to build_alert() along with the
# call-site-specific summary + evidence.
# ──────────────────────────────────────────────────────────────────────

FTSO_NO_SUBMIT1 = {
    "diagnosis": (
        "The observer scanned this voting round and saw no submit1 (commit) "
        "transaction from this entity. submit1 is the FTSO COMMIT phase; "
        "missing this means the validator does not participate in this "
        "round's vote at all (and the corresponding submit2 reveal cannot "
        "match, causing a reveal offence). Two common causes:\n"
        "  (a) REAL outage. The FTSO client failed to commit. Investigate "
        "immediately — submit1 missing is more serious than submitSignatures "
        "missing because it affects the underlying vote.\n"
        "  (b) FALSE POSITIVE. The observer was offline during this round "
        "and is reporting a historical gap during catch-up. Correlates with "
        "a recent fsp-observer container restart."
    ),
    "actions": (
        "If you JUST restarted fsp-observer and the round start_unix is "
        "BEFORE the restart timestamp: FALSE POSITIVE. Alerts will stop "
        "firing within ~2-3 rounds. Spot-check one round on Flare Explorer "
        "for a submit1 tx from submit_addr to confirm.\n"
        "\n"
        "If observer was up the whole time (REAL miss):\n"
        "  docker logs --tail 50 flare-systems-deployment-ftso-client-1\n"
        "  Check gas balance of submit_addr on Flare Explorer.\n"
        "  Verify FSP entity registration still active via "
        "EntityManager.getVoterAddresses(identity).\n"
        "  Check the ftso-client config for the upstream RPC URL — if it's "
        "still pointing at 172.17.0.1, it may have hit the same "
        "post-compose-restart routing issue that broke fsp-observer.\n"
        "\n"
        "Single missed round per ~24h is acceptable. 3+ consecutive misses "
        "= real outage; check ftso-client + signing key health."
    ),
}

FTSO_NO_SUBMIT_SIGNATURES = {
    "diagnosis": (
        "The observer scanned this voting round and saw no submitSignatures "
        "transaction from this entity. Two common causes:\n"
        "  (a) REAL outage. The FTSO client failed to submit. Investigate "
        "if this is a NEW streak of 3+ consecutive rounds.\n"
        "  (b) FALSE POSITIVE. The observer was offline during this round "
        "and is reporting a historical gap during catch-up. Correlates with "
        "a recent fsp-observer container restart."
    ),
    "actions": (
        "If you JUST restarted fsp-observer and the round start_unix is "
        "BEFORE the restart timestamp: FALSE POSITIVE. Alerts will stop "
        "firing within ~2-3 rounds. Spot-check one round on Flare Explorer "
        "for a submitSignatures tx from submit_sigs_to to confirm the "
        "actual submission landed.\n"
        "\n"
        "If observer was up the whole time (REAL miss):\n"
        "  docker logs --tail 50 flare-systems-deployment-ftso-client-1\n"
        "  Check gas balance of submit_sigs_to on Flare Explorer for the "
        "round window.\n"
        "  Verify FSP entity registration still active via "
        "EntityManager.getVoterAddresses(identity).\n"
        "\n"
        "Single missed round per ~24h is acceptable (network reorg or "
        "transient RPC blip). 3+ consecutive misses indicates a real "
        "outage; escalate per docs/runbooks/staking-key-emergency-"
        "rotation.md if signing-key compromise is suspected (cross-check "
        "staking-dir tripwire HC.io status)."
    ),
}

FTSO_SIGNATURE_MISMATCH = {
    "diagnosis": (
        "The observer found a submitSignatures transaction in this round "
        "BUT the recovered signer address does not match the entity's "
        "signing_policy_address. This is a higher-severity finding than "
        "missing signatures because it suggests SOMEONE is submitting "
        "signatures on this entity's behalf with a key that doesn't match "
        "the registered signing-policy. Possible causes:\n"
        "  (a) KEY ROTATION DRIFT. signing_policy was rotated on-chain but "
        "the off-chain client still signs with the old key (or vice versa).\n"
        "  (b) CONFIG ERROR. The ftso-client is loading the wrong signing "
        "key file.\n"
        "  (c) HOSTILE PATTERN. Someone else is submitting signatures using "
        "a key they control, against this entity's identity. Investigate "
        "as potential compromise."
    ),
    "actions": (
        "Verify which key the ftso-client is actually using:\n"
        "  docker exec flare-systems-deployment-ftso-client-1 env | grep -i KEY\n"
        "  (or check the client's config file for SIGNING_POLICY_KEY path)\n"
        "\n"
        "Verify on-chain signing-policy address matches the registered key:\n"
        "  EntityManager.getVoterAddresses(identity) -> signing_policy\n"
        "  Compare against the ftso-client's actual signing key derived address.\n"
        "\n"
        "If MISMATCH: rotate the signing-policy key on-chain to match the "
        "client's current key OR roll back the client's key to the "
        "registered one. Coordinate via docs/runbooks/audit-signer-key-"
        "rotation.md (similar pattern; FSP-specific rotation procedure in "
        "dev.flare.network/run-node/flare-entity).\n"
        "\n"
        "If HOSTILE SUSPECTED: cross-check staking-dir tripwire HC.io "
        "status; if also fired, escalate immediately to docs/runbooks/"
        "staking-key-emergency-rotation.md."
    ),
}

FDC_NO_SUBMIT_SIGNATURES = {
    "diagnosis": (
        "The observer scanned this voting round and saw no submitSignatures "
        "transaction from this entity for the FDC protocol. FDC handles "
        "external chain data attestations; missing signatures here means "
        "this validator's attestation isn't counted toward consensus. Two "
        "common causes:\n"
        "  (a) REAL outage. The FDC client failed to submit. Investigate "
        "if this is a NEW streak of 3+ consecutive rounds.\n"
        "  (b) FALSE POSITIVE. The observer was offline during this round "
        "and is reporting a historical gap during catch-up. Correlates "
        "with a recent fsp-observer container restart."
    ),
    "actions": (
        "If you JUST restarted fsp-observer and the round start_unix is "
        "BEFORE the restart timestamp: FALSE POSITIVE. Alerts will stop "
        "firing within ~2-3 rounds.\n"
        "\n"
        "If observer was up the whole time (REAL miss):\n"
        "  docker logs --tail 50 flare-systems-deployment-fdc-client-1\n"
        "  Check gas balance of submit_sigs_to on Flare Explorer.\n"
        "  Check verifier-{btc,doge,xrp,evm}-* container health (FDC "
        "depends on per-chain verifiers being up).\n"
        "  Verify FSP entity registration still active.\n"
        "\n"
        "Single missed round per ~24h is acceptable. 3+ consecutive misses "
        "= real outage; the verifier-chain dependency is the most common "
        "cause."
    ),
}

FTSO_SUBMIT1_LATE = {
    "diagnosis": (
        "submit1 (commit) was sent AFTER the protocol's correct time window. "
        "The transaction is on-chain but late, which usually means the FTSO "
        "client's clock or scheduling drifted, OR the RPC the client uses "
        "was slow to relay. Late submissions earn reduced or zero reward "
        "weight depending on how late and on the validator's grace-period "
        "config. Not a key compromise; not a reveal offence."
    ),
    "actions": (
        "If this is a one-off:\n"
        "  Check the ftso-client logs for timing-related warnings:\n"
        "    docker logs --tail 100 flare-systems-deployment-ftso-client-1 | "
        "grep -i 'submit1\\|deadline'\n"
        "\n"
        "If this happens repeatedly:\n"
        "  Check clock drift on validator-host: chronyc tracking\n"
        "  Check upstream RPC latency: time curl ... eth_blockNumber\n"
        "  Check ftso-client CPU/mem pressure: docker stats "
        "flare-systems-deployment-ftso-client-1"
    ),
}

FTSO_SUBMIT1_HASH_LENGTH = {
    "diagnosis": (
        "submit1 transaction was found but its commit-hash payload is the "
        "wrong length (expected 32 bytes). This is a client-side bug or "
        "config error, not a network issue. The reveal phase (submit2) will "
        "fail to match this commit, causing a REVEAL OFFENCE — reward loss "
        "for this round."
    ),
    "actions": (
        "Inspect the submit1 transaction payload on Flare Explorer for the "
        "voting_epoch above; compare the actual payload length to expected "
        "32 bytes.\n"
        "\n"
        "If the ftso-client was recently updated: roll back the container "
        "to the previous known-good version.\n"
        "  docker images | grep ftso-scaling\n"
        "  Edit /opt/flare/docker-compose.yml to pin the previous version.\n"
        "  cd /opt/flare && docker compose up -d flare-systems-deployment-"
        "ftso-client-1\n"
        "\n"
        "If no recent update: file an issue against the ftso-scaling repo. "
        "Cross-check whether other validators are seeing the same bug "
        "(Flare Foundation Discord)."
    ),
}

FTSO_SUBMIT2_MISSING_OR_OUT_OF_WINDOW = {
    "diagnosis": (
        "submit2 (reveal) was missing OR sent outside the correct time "
        "window. If submit1 was also missing this round, the validator did "
        "not participate (less severe). If submit1 EXISTS but submit2 is "
        "missing, this is a REVEAL OFFENCE — full reward loss for the "
        "round and a penalty on signal weight in future rounds. Causes:\n"
        "  (a) REAL outage. ftso-client crashed between commit and reveal.\n"
        "  (b) Late reveal due to clock drift / RPC slowness.\n"
        "  (c) FALSE POSITIVE from observer catch-up if recent restart."
    ),
    "actions": (
        "If JUST restarted fsp-observer and round is historical: FALSE "
        "POSITIVE. Confirm via Flare Explorer for the round.\n"
        "\n"
        "If REAL miss:\n"
        "  docker logs --tail 100 flare-systems-deployment-ftso-client-1\n"
        "  Look for crashes / panics / lost connection between the commit "
        "and reveal phases (~45s apart).\n"
        "  Cross-check clock: chronyc tracking\n"
        "  Check submit account gas balance on Flare Explorer.\n"
        "\n"
        "If REVEAL OFFENCE confirmed: single offence is recoverable; 3+ in "
        "a rolling window = real operational issue, investigate ftso-client "
        "stability."
    ),
}

FTSO_COMMIT_REVEAL_MISMATCH = {
    "diagnosis": (
        "Both submit1 (commit) and submit2 (reveal) were on-chain, but the "
        "commit hash does not derive from the reveal values. This is a "
        "REVEAL OFFENCE — full reward loss for the round and penalty on "
        "signal weight. Causes:\n"
        "  (a) ftso-client bug — produced inconsistent commit/reveal "
        "between the two phases.\n"
        "  (b) Configuration changed mid-round (highly unusual).\n"
        "  (c) Submit account or signing key was reused by another "
        "process between commit and reveal (would be HOSTILE)."
    ),
    "actions": (
        "Inspect both transactions:\n"
        "  Find submit1 + submit2 tx hashes on Flare Explorer for the "
        "voting_epoch; compare submit1.commit_hash to "
        "keccak(submit_addr, epoch_id, rnd, reveal_values).\n"
        "\n"
        "Investigate ftso-client state across the commit/reveal window:\n"
        "  docker logs --tail 200 flare-systems-deployment-ftso-client-1 | "
        "less\n"
        "\n"
        "If HOSTILE suspected (concurrent submission from another source): "
        "cross-check staking-dir tripwire + audit-log chain integrity. "
        "Escalate per docs/runbooks/staking-key-emergency-rotation.md if "
        "any other suspicious indicators."
    ),
}

FDC_FOUND_SUBMIT1 = {
    "diagnosis": (
        "FDC protocol does NOT use submit1 (commit phase). The observer "
        "saw a submit1 transaction from this entity in an FDC round, "
        "which means either:\n"
        "  (a) The fdc-client is misconfigured and sending submit1 calls "
        "to the FDC contract (wastes gas; ignored by protocol).\n"
        "  (b) A custom or forked fdc-client is calling submit1 by "
        "mistake.\n"
        "Not a signing-key concern; not a reveal offence. Wasted-gas "
        "concern only."
    ),
    "actions": (
        "Check fdc-client logs for unexpected submit1 calls:\n"
        "  docker logs --tail 100 flare-systems-deployment-fdc-client-1 | "
        "grep -i submit1\n"
        "\n"
        "If recent fdc-client upgrade: roll back. If not: file an issue "
        "against fdc-client repo; cross-check Flare Foundation Discord "
        "for other validators seeing same symptom."
    ),
}

FDC_SUBMIT2_MISSING_OR_OUT_OF_WINDOW = {
    "diagnosis": (
        "FDC submit2 (bit-vote) was missing OR outside the correct time "
        "window. submit2 is how the validator confirms which attestation "
        "requests it agrees to attest. Missing here means the validator "
        "is not contributing its bit-vote to consensus. Causes:\n"
        "  (a) REAL outage. fdc-client crashed OR the underlying verifiers "
        "(btc/doge/xrp/evm) were unreachable when bit-vote was due.\n"
        "  (b) Late submission due to verifier slowness.\n"
        "  (c) FALSE POSITIVE from observer catch-up after restart."
    ),
    "actions": (
        "If FALSE POSITIVE (recent observer restart, round is historical): "
        "alerts will stop within ~2-3 rounds.\n"
        "\n"
        "If REAL miss:\n"
        "  docker logs --tail 50 flare-systems-deployment-fdc-client-1\n"
        "  docker ps --filter name=verifier- (all 4 chains must be up)\n"
        "  curl on each verifier-{btc,doge,xrp,evm}-verifier health URL "
        "(see runbook for the URLs).\n"
        "\n"
        "The most common real cause is one of the verifier chains being "
        "slow OR the underlying chain node (node-mainnet-btc / "
        "node-mainnet-doge / etc.) being out of sync."
    ),
}

FDC_SUBMIT2_BITVOTE_LENGTH = {
    "diagnosis": (
        "FDC submit2 was on-chain but the bit-vote vector length does not "
        "match the number of attestation requests in this round. This is "
        "a CLIENT BUG, not a config or network issue. The bit-vote is "
        "discarded by the protocol; the validator earns no FDC reward "
        "this round."
    ),
    "actions": (
        "Inspect fdc-client logs for vector-length errors:\n"
        "  docker logs --tail 100 flare-systems-deployment-fdc-client-1 | "
        "grep -iE 'bit.?vote|vector|length'\n"
        "\n"
        "If recent fdc-client update: roll back to previous known-good "
        "version.\n"
        "  Edit /opt/flare/docker-compose.yml to pin a previous version "
        "of ghcr.io/flare-foundation/fdc-client.\n"
        "  cd /opt/flare && docker compose up -d "
        "flare-systems-deployment-fdc-client-1\n"
        "\n"
        "Otherwise: file an issue against fdc-client repo with the round "
        "ID + the actual-vs-expected vector lengths."
    ),
}

FDC_SUBMIT2_CONSENSUS_MISS = {
    "diagnosis": (
        "FDC submit2 was on-chain but DID NOT confirm a specific request "
        "that was part of the consensus bit-vote. This means the "
        "validator's verifier for this chain disagreed with the rest of "
        "the network on whether to attest this request. Causes:\n"
        "  (a) Underlying verifier (btc/doge/xrp/evm) was out of sync or "
        "slow; missed the consensus window.\n"
        "  (b) Verifier had a bug that didn't recognize a valid request.\n"
        "  (c) Network had a split-brain event (rare but possible)."
    ),
    "actions": (
        "Identify which verifier protocol failed (attestation_type / "
        "source_id in the alert text). Then check that verifier's logs:\n"
        "  docker logs --tail 100 verifier-<chain>-verifier\n"
        "  docker logs --tail 50 verifier-<chain>-server\n"
        "  docker logs --tail 50 verifier-<chain>-indexer (or index-blocks)\n"
        "\n"
        "Check that verifier's underlying chain node is up to head:\n"
        "  docker logs --tail 30 node-mainnet-<chain> (for btc/doge)\n"
        "  XRP / EVM verifiers use external chains; check their connectivity.\n"
        "\n"
        "Single consensus-miss is acceptable. Repeated misses on the same "
        "chain = verifier substrate issue; consider verifier resync."
    ),
}

FDC_REVEAL_OFFENCE_NO_SIGS = {
    "diagnosis": (
        "FDC submit2 was on-chain and dominated the consensus bit-vote, "
        "but no submitSignatures transaction was sent. This is a REVEAL "
        "OFFENCE in FDC — full reward loss for the round and signal-weight "
        "penalty. Either the fdc-client crashed between submit2 and "
        "signatures phase, OR signing-policy address has no gas / signing "
        "key issue."
    ),
    "actions": (
        "Investigate fdc-client state between submit2 and signature phases:\n"
        "  docker logs --tail 200 flare-systems-deployment-fdc-client-1\n"
        "  Check for crashes / OOM / hung HTTP calls between the phases.\n"
        "\n"
        "Check signing-policy address gas balance on Flare Explorer.\n"
        "\n"
        "Cross-check FTSO equivalent — if both FTSO and FDC dropped "
        "signatures in the same window, the underlying issue is shared "
        "infrastructure (likely RPC connection). If FDC-only, the FDC "
        "client substrate is the suspect."
    ),
}

FDC_SIGNATURE_MISMATCH = {
    "diagnosis": (
        "FDC submitSignatures was on-chain but the recovered signer "
        "address does NOT match the entity's signing_policy_address. Same "
        "shape as the FTSO signature mismatch alert. Possible causes:\n"
        "  (a) KEY ROTATION DRIFT. signing_policy was rotated on-chain "
        "but the fdc-client still signs with the old key (or vice versa).\n"
        "  (b) CONFIG ERROR. fdc-client loaded the wrong signing key.\n"
        "  (c) HOSTILE PATTERN. Someone else submitted signatures using a "
        "key they control. Investigate as potential compromise."
    ),
    "actions": (
        "Verify which key the fdc-client is using:\n"
        "  docker exec flare-systems-deployment-fdc-client-1 env | "
        "grep -i KEY\n"
        "\n"
        "Verify on-chain signing-policy:\n"
        "  EntityManager.getVoterAddresses(identity) -> signing_policy\n"
        "\n"
        "If MISMATCH: rotate signing-policy key on-chain to match or roll "
        "back the client. See docs/runbooks/audit-signer-key-rotation.md "
        "for the rotation procedure pattern.\n"
        "\n"
        "If HOSTILE SUSPECTED: cross-check staking-dir tripwire; if also "
        "fired, escalate to docs/runbooks/staking-key-emergency-rotation.md."
    ),
}

GRACE_PERIOD_LATE_SIGNATURES = {
    "diagnosis": (
        "submitSignatures was on-chain but landed AFTER the grace-period "
        "deadline (60s past the round's end). The signature is still valid "
        "but earns reduced or zero reward weight for this round. Usually "
        "means the client OR the upstream RPC was slow during the signing "
        "window. Not a key or signing issue."
    ),
    "actions": (
        "Check the client logs around the round timestamp:\n"
        "  docker logs --tail 100 flare-systems-deployment-ftso-client-1 "
        "(or fdc-client-1)\n"
        "\n"
        "Most common cause: upstream RPC was slow. Cross-check the "
        "flr-rpc-heartbeat HC.io status for the same window — if it was "
        "yellow/red, the RPC was the bottleneck.\n"
        "\n"
        "Single occurrence: noise. Repeated occurrences: RPC capacity or "
        "client tuning issue; investigate ftso-client / fdc-client config."
    ),
}

FAST_UPDATE_MISSED = {
    "diagnosis": (
        "The validator's fast-updates client has not submitted an update "
        "within the expected block-count window. Fast-updates is the "
        "high-frequency price-feed protocol that runs between FTSO rounds. "
        "Missing here means the validator's feed-value-provider OR the "
        "fast-updates client is degraded. At CRITICAL severity, the "
        "false-positive probability is <= 100ppb (effectively zero); this "
        "is a real outage."
    ),
    "actions": (
        "Check fast-updates client logs:\n"
        "  docker logs --tail 50 flare-systems-deployment-fast-updates-1\n"
        "\n"
        "Check ftso-fvp health (provides the feed values fast-updates submits):\n"
        "  docker ps | grep ftso-fvp\n"
        "  curl -s http://127.0.0.1:3101/api-doc | head -1\n"
        "\n"
        "Check submission account gas balance for the fast-updates address.\n"
        "\n"
        "Check upstream RPC latency — fast-updates is sensitive to chain "
        "latency. Cross-reference flr-rpc-heartbeat HC.io status."
    ),
}

LOW_BALANCE = {
    "diagnosis": (
        "An entity operational address has dropped BELOW 5 NAT (FLR or "
        "SGB) and is at risk of running out of gas for upcoming "
        "protocol submissions. Each round's submit1 / submit2 / "
        "submitSignatures costs gas; account exhaustion stops "
        "participation entirely. Threshold is operator-tuned via "
        "config.fee_threshold; this ERROR fires at the hard 5 NAT floor."
    ),
    "actions": (
        "Transfer additional NAT to the depleted address. Funding source:\n"
        "  - Treasury wallet (per cross-repo C11)\n"
        "  - Operator's identity address (cold storage; coordinate via "
        "Ledger session)\n"
        "\n"
        "After topping up: verify via Flare Explorer the new balance is "
        "comfortably above threshold. Future runway: 5 NAT typically "
        "covers ~1-2 weeks of submissions; aim for >50 NAT for several "
        "months of headroom.\n"
        "\n"
        "If THIS specific address is regularly depleting fast: check "
        "tx gas-price config in the relevant client. May be over-paying."
    ),
}

MINIMAL_CONDITIONS_NULL_VALUES = {
    "diagnosis": (
        "submit2 (reveal) was on-chain but contained NULL feed values at "
        "specific indices. The FTSO protocol allows null reveals (the feed "
        "value provider couldn't compute a value for that feed) but null "
        "feeds contribute zero weight to that feed's median. Repeated "
        "nulls on the same feed = the validator's feed-value-provider is "
        "missing data for that feed."
    ),
    "actions": (
        "Identify which feed indices had nulls (in the alert message).\n"
        "\n"
        "Check the ftso-fvp (Feed Value Provider) for those feeds:\n"
        "  docker logs --tail 100 ftso-fvp | grep -iE 'null|error|fail'\n"
        "  curl -s http://127.0.0.1:3101/api-doc\n"
        "\n"
        "Most common cause: an upstream price source (Binance, Coinbase, "
        "etc.) was rate-limited or unreachable. The ftso-fvp emits null "
        "rather than guessing. Acceptable for occasional feeds; investigate "
        "if persistent on critical feeds (FLR, BTC, ETH)."
    ),
}

MINIMAL_CONDITIONS_OUT_OF_RANGE = {
    "diagnosis": (
        "submit2 (reveal) had feed values that fell outside the +/-0.5% "
        "minimum-conditions band around the network median (per FIP-10). "
        "Out-of-range values earn reduced or zero reward weight for those "
        "feeds. Either the ftso-fvp is producing biased values OR a "
        "specific feed's upstream price source disagreed with the network "
        "consensus."
    ),
    "actions": (
        "Identify the failing feed indices (in the alert message).\n"
        "\n"
        "Check the ftso-fvp config for those feeds:\n"
        "  cat /opt/flare/feeds.json | jq\n"
        "\n"
        "Check upstream price sources for divergence from the network "
        "median around the affected voting_epoch on Flare Explorer.\n"
        "\n"
        "If a particular feed is repeatedly out of range: tune the "
        "ftso-fvp's source-weighting OR exclude the offending source. "
        "Single-round misses are noise; persistent = config issue."
    ),
}

ADDRESS_MISMATCH = {
    "diagnosis": (
        "An entity address registered on-chain does not match what this "
        "observer is configured to track for this entity. This is a "
        "CONFIGURATION DRIFT alert: the on-chain EntityManager state has "
        "diverged from the observer's local config (or vice versa). Common "
        "causes:\n"
        "  (a) The operator changed addresses on-chain (rotated a "
        "submit/signing-policy address) without updating fsp-observer's "
        "configuration.\n"
        "  (b) The observer was deployed with stale config; the entity "
        "has since updated its registration."
    ),
    "actions": (
        "Check on-chain EntityManager state for this identity:\n"
        "  EntityManager.getVoterAddresses(identity)\n"
        "  Cross-reference against /opt/flare/observer/.env (or "
        "observer config file) for SUBMIT_ADDRESS, SIGNING_POLICY_ADDRESS, "
        "DELEGATION_ADDRESS, etc.\n"
        "\n"
        "If on-chain is correct and config is stale: update the .env "
        "and restart fsp-observer.\n"
        "\n"
        "If config is correct and on-chain has unexpected entries: "
        "INVESTIGATE — someone (else) registered an address against this "
        "entity. Cross-check the audit log for the registration tx."
    ),
}
