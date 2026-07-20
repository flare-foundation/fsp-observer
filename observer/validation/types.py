from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, TypedDict, Unpack

if TYPE_CHECKING:
    from configuration.types import Configuration

    from ..message import Message, MessageBuilder
    from ..reward_epoch_manager import Entity
    from ..types import ProtocolMessageRelayed
    from ..voting_round import VotingRound, WParsedPayload
    from .validation import ExtractedEntityVotingRound


class ValidateFnKwargs[S1, S2, SS](TypedDict):
    submit_1: WParsedPayload[S1] | None
    submit_2: WParsedPayload[S2] | None
    submit_signatures: WParsedPayload[SS] | None
    finalization: ProtocolMessageRelayed | None
    extracted_round: ExtractedEntityVotingRound[S1, S2, SS]
    message_builder: MessageBuilder
    entity: Entity
    round: VotingRound
    config: Configuration


class ValidateFn[S1, S2, SS](Protocol):
    def __call__(
        self,
        **kwargs: Unpack[ValidateFnKwargs[S1, S2, SS]],
    ) -> Sequence[Message]: ...
