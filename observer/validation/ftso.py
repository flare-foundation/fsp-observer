from collections.abc import Sequence

from py_flare_common.fsp.messaging import parse_generic_tx
from py_flare_common.fsp.messaging.byte_parser import ByteParser
from py_flare_common.fsp.messaging.types import (
    FtsoSubmit1,
    FtsoSubmit2,
    SubmitSignatures,
)
from py_flare_common.ftso.commit import commit_hash

from ..message import Message, MessageBuilder, MessageLevel
from ..reward_epoch_manager import Entity
from ..types import ProtocolMessageRelayed
from ..voting_round import VotingRound, WParsedPayload
from .signature import Signature
from .types import ValidateFn


# NOTE:(matej) stupid type cast
def _check_type(f: ValidateFn[FtsoSubmit1, FtsoSubmit2, SubmitSignatures]):
    return f


@_check_type
def check_submit_1(
    submit_1: WParsedPayload[FtsoSubmit1] | None,
    message_builder: MessageBuilder,
    **_,
) -> Sequence[Message]:
    issues = []
    mb = message_builder

    # NOTE:(matej) In ftso protocol submit1 is used for sending the commit hash
    # this means that the messsage must exist and its payload should be 32 bytes.
    # we perform the following checks:
    # - submit1 doesn't exist -> error
    # - submit1 exists but commit hash length isn't 32 -> error

    if submit_1 is None:
        issues.append(mb.build(MessageLevel.ERROR, "no submit1 transaction"))

    if submit_1 is not None:
        hash_len = len(submit_1.parsed_payload.payload.commit_hash)
        if hash_len != 32:
            issues.append(
                mb.build(
                    MessageLevel.ERROR,
                    f"submit1 commit hash unexpeted length ({hash_len}), expected 32",
                )
            )

    return issues


@_check_type
def check_submit_2(
    submit_1: WParsedPayload[FtsoSubmit1] | None,
    submit_2: WParsedPayload[FtsoSubmit2] | None,
    message_builder: MessageBuilder,
    entity: Entity,
    round: VotingRound,
    **_,
) -> Sequence[Message]:
    issues = []
    mb = message_builder

    # NOTE:(matej) In ftso protocol submit2 is used for sending the reveal
    # this means that the messsage must exist and its hash must match submit1.
    # Additionally decoded ftso values must have values that aren't null and
    # are in range of minimal conditions
    # we perform the following checks:
    # - submit1 doesn't exist and submit2 doesn't exist -> error
    # - submit1 exists and submit2 doesn't -> reveal offence
    # - both exist but reveal hash doesn't match commit hash -> reveal offence
    # - ftso values have null values -> warning
    # - ftso value have values that aren't in range of minimal conditions -> warning
    # - ftso values have incorrect length -> warning

    if submit_1 is None and submit_2 is None:
        issues.append(mb.build(MessageLevel.ERROR, "no submit2 transaction"))

    if submit_1 is not None and submit_2 is None:
        issues.append(
            mb.build(
                MessageLevel.CRITICAL, "no submit2 transaction, causing reveal offence"
            )
        )

    if submit_1 is not None and submit_2 is not None:
        # TODO:(matej) should just build back from parsed message
        bp = ByteParser(parse_generic_tx(submit_2.wtx_data.input).ftso.payload)
        rnd = bp.uint256()
        feed_v = bp.drain()

        hashed = commit_hash(entity.submit_address, round.voting_epoch.id, rnd, feed_v)

        if submit_1.parsed_payload.payload.commit_hash.hex() != hashed:
            issues.append(
                mb.build(
                    MessageLevel.CRITICAL,
                    "commit hash and reveal didn't match, causing reveal offence",
                ),
            )

    if submit_2 is not None:
        medians = round.ftso.medians
        values = submit_2.parsed_payload.payload.values

        if len(values) != len(medians):
            issues.append(
                mb.build(
                    MessageLevel.WARNING,
                    (
                        f"submit2 had values for {len(values)} feeds, "
                        f"expected {len(medians)}"
                    ),
                )
            )

        else:
            none_indices = []
            minimal_condition_indices = []

            for i, (v, m) in enumerate(zip(values, medians)):
                if v is None:
                    none_indices.append(str(i))
                    continue

                # as per https://proposals.flare.network/FIP/FIP_10.html
                mcb_low = m.value * 0.995
                mcb_high = m.value * 1.005

                if not (mcb_low <= v <= mcb_high):
                    minimal_condition_indices.append(str(i))

            if none_indices:
                ind = ", ".join(none_indices)
                issues.append(
                    mb.build(
                        MessageLevel.WARNING,
                        f"submit2 had 'None' on indices {ind}",
                    )
                )

            # TODO:(matej) change this to a sampling array instead
            # if minimal_condition_indices:
            #     ind = ", ".join(minimal_condition_indices)
            #     issues.append(
            #         mb.build(
            #             MessageLevel.WARNING,
            #             f"submit2 values missed minimal conditions on indices {ind}",
            #         )
            #     )

    return issues


@_check_type
def check_submit_signatures(
    submit_signatures: WParsedPayload[SubmitSignatures] | None,
    finalization: ProtocolMessageRelayed | None,
    message_builder: MessageBuilder,
    entity: Entity,
    round: VotingRound,
    **_,
) -> Sequence[Message]:
    issues = []
    mb = message_builder

    # NOTE:(matej) In ftso protocol submitSignatures is used for sending the signature
    # of finalization struct (ProtocolMessageRelayed event on chain). This means that
    # the message must exist and the signature must match the finalization. Additionally
    # the signature must be deposited before the end of grace period or finalization on
    # chain (whichever is later)
    # we perform the following checks:
    # - submitSignatures doesn't exist -> error
    # - submitSignature was sent after the deadline -> warning
    # - signature doesn't match finalization -> error

    if submit_signatures is None:
        issues.append(
            mb.build(MessageLevel.ERROR, "no submitSignatures transaction"),
        )

    if submit_signatures is not None:
        deadline = max(
            round.voting_epoch.next.start_s + 60,
            (finalization and finalization.timestamp) or 0,
        )

        if submit_signatures.wtx_data.timestamp > deadline:
            issues.append(
                mb.build(
                    MessageLevel.WARNING,
                    "no submitSignatures during grace period, causing loss of rewards",
                )
            )

    if submit_signatures is not None and finalization is not None:
        s = Signature.from_parsed_signature(
            submit_signatures.parsed_payload.payload.signature
        )
        addr = s.recover_public_key_from_msg_hash(
            finalization.to_message()
        ).to_checksum_address()

        if addr != entity.signing_policy_address:
            issues.append(
                mb.build(
                    MessageLevel.ERROR,
                    "submitSignatures signature doesn't match finalization",
                ),
            )

    return issues


@_check_type
def check_tx_timing(
    message_builder: MessageBuilder,
    entity: Entity,
    round: VotingRound,
    **_,
) -> Sequence[Message]:
    # NOTE: (miha) The protocol accepts the latest transaction sent in the correct time
    # interval. We check if there are any transactions sent before or after this
    # interval and send a warning for those before and a warning for those after
    issues = []
    mb = message_builder

    epoch = round.voting_epoch
    next = epoch.next
    rd = next.reveal_deadline()

    _submit_1 = round.ftso.submit_1.by_identity[entity.identity_address]
    submit_1_before, submit_1_after = _submit_1.extract_outside_of_bounds(
        range(epoch.start_s, epoch.end_s)
    )
    if len(submit_1_before) > 0:
        issues.append(
            mb.build(
                MessageLevel.WARNING,
                f"submit 1 transactions sent before correct time interval: {submit_1_before}",  # noqa: E501
            )
        )
    if len(submit_1_after) > 0:
        issues.append(
            mb.build(
                MessageLevel.WARNING,
                f"submit 1 transactions sent after correct time interval: {submit_1_after}",  # noqa: E501
            )
        )

    _submit_2 = round.ftso.submit_2.by_identity[entity.identity_address]
    submit_2_before, submit_2_after = _submit_2.extract_outside_of_bounds(
        range(next.start_s, rd)
    )
    if len(submit_2_before) > 0:
        issues.append(
            mb.build(
                MessageLevel.WARNING,
                f"submit 2 transactions sent before correct time interval: {submit_2_before}",  # noqa: E501
            )
        )
    if len(submit_2_after) > 0:
        issues.append(
            mb.build(
                MessageLevel.WARNING,
                f"submit 2 transactions sent after correct time interval: {submit_2_after}",  # noqa: E501
            )
        )

    _submit_signatures = round.ftso.submit_signatures.by_identity[
        entity.identity_address
    ]
    submit_signatures_before, submit_signatures_after = (
        _submit_signatures.extract_outside_of_bounds(range(rd, next.end_s))
    )
    if len(submit_signatures_before) > 0:
        issues.append(
            mb.build(
                MessageLevel.WARNING,
                f"submit signatures transactions sent before correct time interval: {submit_signatures_before}",  # noqa: E501
            )
        )
    if len(submit_signatures_after) > 0:
        issues.append(
            mb.build(
                MessageLevel.WARNING,
                f"submit signatures transactions sent after correct time interval: {submit_signatures_after}",  # noqa: E501
            )
        )

    return issues
