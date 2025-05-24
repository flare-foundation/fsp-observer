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
    # - submit1 exists but commit hash length isn't 32 -> warning

    if submit_1 is None:
        issues.append(mb.build(MessageLevel.ERROR, "no submit1 transaction"))

    if submit_1 is not None:
        hash_len = len(submit_1.parsed_payload.payload.commit_hash)
        if hash_len != 32:
            issues.append(
                mb.build(
                    MessageLevel.WARNING,
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
    # TODO:(matej)
    # - ftso value have values that aren't in range of minimal conditions -> warning
    # - ftso values have inocrrect length -> warning

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
        indices = [
            str(i)
            for i, v in enumerate(submit_2.parsed_payload.payload.values)
            if v is None
        ]

        if indices:
            issues.append(
                mb.build(
                    MessageLevel.WARNING,
                    f"submit 2 had 'None' on indices {', '.join(indices)}",
                )
            )

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
    # - submitSignature was sent after the deadline -> error
    # - signature doesn't match finalization -> error

    if submit_signatures is None:
        issues.append(
            mb.build(MessageLevel.ERROR, "no submitSignatures transaction"),
        )

    if submit_signatures is not None:
        deadline = max(
            round.voting_epoch.next.start_s + 55,
            (finalization and finalization.timestamp) or 0,
        )

        if submit_signatures.wtx_data.timestamp > deadline:
            issues.append(
                mb.build(
                    MessageLevel.ERROR,
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
                    "submit signatures signature doesn't match finalization",
                ),
            )

    return issues
