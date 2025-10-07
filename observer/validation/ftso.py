from collections.abc import Sequence
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from observer.validation.validation import ExtractedEntityVotingRound


# NOTE:(matej) stupid type cast
def _check_type(f: ValidateFn[FtsoSubmit1, FtsoSubmit2, SubmitSignatures]):
    return f


@_check_type
def check_submit_1(
    submit_1: WParsedPayload[FtsoSubmit1] | None,
    message_builder: MessageBuilder,
    extracted_round: "ExtractedEntityVotingRound[FtsoSubmit1, FtsoSubmit2, SubmitSignatures]",  # noqa: E501
    **_,
) -> Sequence[Message]:
    issues = []
    mb = message_builder

    # NOTE:(matej) In ftso protocol submit1 is used for sending the commit hash
    # this means that the messsage must exist and its payload should be 32 bytes.
    # we perform the following checks:
    # - submit1 doesn't exist -> error
    # - submit1 exists but commit hash length isn't 32 -> error
    # - submit1 exists but was sent before or after submission window -> warning

    if late := extracted_round.submit_1.late:
        tx_hashes = ", ".join([tx.wtx_data.hash.to_0x_hex() for tx in late])
        # TODO: (@miha.gyergyek.aflabs) Eventually configurable
        # level = MessageLevel.WARNING
        level = MessageLevel.ERROR
        issues.append(
            mb.build(
                level,
                (f"submit1 transactions sent after correct time interval: {tx_hashes}"),
            )
        )

    if submit_1 is None and not late:
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
    extracted_round: "ExtractedEntityVotingRound[FtsoSubmit1, FtsoSubmit2, SubmitSignatures]",  # noqa: E501
    **_,
) -> Sequence[Message]:
    issues = []
    mb = message_builder

    # NOTE:(matej) In ftso protocol submit2 is used for sending the reveal
    # this means that the messsage must exist and its hash must match submit1.
    # Additionally decoded ftso values must have values that aren't null and
    # are in range of minimal conditions
    # we perform the following checks:
    # - submit1 doesn't exist and submit2 doesn't exist in the right interval -> error
    # - submit1 exists and submit2 doesn't in the right interval -> reveal offence
    # - both exist but reveal hash doesn't match commit hash -> reveal offence
    # - ftso values have null values -> warning
    # - ftso value have values that aren't in range of minimal conditions -> warning
    # - ftso values have incorrect length -> warning

    early = extracted_round.submit_2.early
    late = extracted_round.submit_2.late
    # TODO: (miha.gyergyek.aflabs) this should be configurable
    level = MessageLevel.ERROR
    message: str = ""
    if early and late:
        early_tx_hashes = ", ".join([tx.wtx_data.hash.to_0x_hex() for tx in early])
        late_tx_hashes = ", ".join([tx.wtx_data.hash.to_0x_hex() for tx in late])
        message = (
            "submit2 transactions sent outside correct time interval. "
            f"Before: {early_tx_hashes}, "
            f"after: {late_tx_hashes}"
        )
    elif early and not late:
        early_tx_hashes = ", ".join([tx.wtx_data.hash.to_0x_hex() for tx in early])
        message = (
            f"submit2 transactions sent before correct time interval: {early_tx_hashes}"
        )
    elif late and not early:
        late_tx_hashes = ", ".join([tx.wtx_data.hash.to_0x_hex() for tx in late])
        message = (
            f"submit2 transactions sent after correct time interval: {late_tx_hashes}"
        )
    elif not early and not late:
        message = "no submit2 transaction"

    if submit_2 is None:
        if submit_1 is not None:
            level = MessageLevel.CRITICAL
            message += ". This caused a reveal offence"
        issues.append(mb.build(level, message))

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

        if early or late:
            issues.append(mb.build(level, message))

    return issues


@_check_type
def check_submit_signatures(
    submit_signatures: WParsedPayload[SubmitSignatures] | None,
    finalization: ProtocolMessageRelayed | None,
    message_builder: MessageBuilder,
    entity: Entity,
    round: VotingRound,
    extracted_round: "ExtractedEntityVotingRound[FtsoSubmit1, FtsoSubmit2, SubmitSignatures]",  # noqa: E501
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
    # - submitSignatures doesn't exist in the correct interval -> error
    # - submitSignature was sent after the deadline -> warning
    # - signature doesn't match finalization -> error

    early = extracted_round.submit_signatures.early
    # TODO: (miha.gyergyek.aflabs) this should be configurable
    level = MessageLevel.ERROR
    message: str = ""
    if early := extracted_round.submit_signatures.early:
        tx_hashes = ", ".join([tx.wtx_data.hash.to_0x_hex() for tx in early])
        message = (
            "submit signatures transactions sent before correct time "
            f"interval: {tx_hashes}"
        )

    if submit_signatures is None:
        if not early:
            issues.append(
                mb.build(MessageLevel.ERROR, "no submitSignatures transaction"),
            )
        else:
            issues.append(mb.build(level, message))

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

        if early:
            issues.append(mb.build(level, message))

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
