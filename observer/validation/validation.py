from collections.abc import Sequence

from attrs import frozen
from py_flare_common.fsp.epoch.epoch import VotingEpoch
from py_flare_common.fsp.messaging.types import (
    FdcSubmit1,
    FdcSubmit2,
    FtsoSubmit1,
    FtsoSubmit2,
    SubmitSignatures,
)

from configuration.types import Configuration

from ..message import Message
from ..reward_epoch_manager import Entity
from ..voting_round import VotingRound, VotingRoundProtocol, WParsedPayload
from . import fdc, ftso
from .types import ValidateFn, ValidateFnKwargs


@frozen
class ExtractedEntityVotingRound[S1, S2, SS]:
    submit_1: WParsedPayload[S1] | None
    submit_2: WParsedPayload[S2] | None
    submit_signatures: WParsedPayload[SS] | None


def extract_round_for_entity[S1, S2, S3](
    r: VotingRoundProtocol[S1, S2, S3], e: Entity, epoch: VotingEpoch
) -> ExtractedEntityVotingRound[S1, S2, S3]:
    next = epoch.next
    rd = next.reveal_deadline()

    _submit_1 = r.submit_1.by_identity[e.identity_address]
    submit_1 = _submit_1.extract_latest(range(epoch.start_s, epoch.end_s))

    _submit_2 = r.submit_2.by_identity[e.identity_address]
    submit_2 = _submit_2.extract_latest(range(next.start_s, rd))

    _submit_signatures = r.submit_signatures.by_identity[e.identity_address]
    submit_signatures = _submit_signatures.extract_latest(range(rd, next.end_s))

    return ExtractedEntityVotingRound(
        submit_1=submit_1, submit_2=submit_2, submit_signatures=submit_signatures
    )


def validate_round(
    round: VotingRound, entity: Entity, config: Configuration
) -> Sequence[Message]:
    issues = []

    mb_ftso = Message.builder().add(
        network=config.chain_id,
        round=round.voting_epoch,
        protocol=100,
    )
    extracted_ftso = extract_round_for_entity(round.ftso, entity, round.voting_epoch)

    ftso_validations: Sequence[
        ValidateFn[FtsoSubmit1, FtsoSubmit2, SubmitSignatures]
    ] = (
        ftso.check_submit_1,
        ftso.check_submit_2,
        ftso.check_submit_signatures,
    )

    ftso_kwargs: ValidateFnKwargs[FtsoSubmit1, FtsoSubmit2, SubmitSignatures] = {
        "submit_1": extracted_ftso.submit_1,
        "submit_2": extracted_ftso.submit_2,
        "submit_signatures": extracted_ftso.submit_signatures,
        "finalization": round.ftso.finalization,
        "extracted_round": extracted_ftso,
        "message_builder": mb_ftso,
        "entity": entity,
        "round": round,
    }

    for validator in ftso_validations:
        issues.extend(validator(**ftso_kwargs))

    mb_fdc = Message.builder().add(
        network=config.chain_id,
        round=round.voting_epoch,
        protocol=200,
    )
    extracted_fdc = extract_round_for_entity(round.fdc, entity, round.voting_epoch)

    fdc_validations: Sequence[ValidateFn[FdcSubmit1, FdcSubmit2, SubmitSignatures]] = (
        fdc.check_submit_1,
        fdc.check_submit_2,
        fdc.check_submit_signatures,
    )

    fdc_kwargs: ValidateFnKwargs[FdcSubmit1, FdcSubmit2, SubmitSignatures] = {
        "submit_1": extracted_fdc.submit_1,
        "submit_2": extracted_fdc.submit_2,
        "submit_signatures": extracted_fdc.submit_signatures,
        "finalization": round.fdc.finalization,
        "extracted_round": extracted_fdc,
        "message_builder": mb_fdc,
        "entity": entity,
        "round": round,
    }

    for validator in fdc_validations:
        issues.extend(validator(**fdc_kwargs))

    return issues
