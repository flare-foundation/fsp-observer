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
