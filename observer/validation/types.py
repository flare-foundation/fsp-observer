from collections.abc import Sequence
from typing import TYPE_CHECKING, Optional, Protocol, TypedDict, Unpack

if TYPE_CHECKING:
    from ..message import Message, MessageBuilder
    from ..reward_epoch_manager import Entity
    from ..types import ProtocolMessageRelayed
    from ..voting_round import VotingRound, WParsedPayload
    from .validation import ExtractedEntityVotingRound


class ValidateFnKwargs[S1, S2, SS](TypedDict):
    submit_1: Optional["WParsedPayload[S1]"]
    submit_2: Optional["WParsedPayload[S2]"]
    submit_signatures: Optional["WParsedPayload[SS]"]
    finalization: Optional["ProtocolMessageRelayed"]
    extracted_round: "ExtractedEntityVotingRound[S1, S2, SS]"
    message_builder: "MessageBuilder"
    entity: "Entity"
    round: "VotingRound"


class ValidateFn[S1, S2, SS](Protocol):
    def __call__(
        self,
        **kwargs: Unpack[ValidateFnKwargs[S1, S2, SS]],
    ) -> Sequence["Message"]: ...
