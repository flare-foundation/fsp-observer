import copy
import enum
import io
from typing import Self

from attrs import define, frozen
from py_flare_common.fsp.epoch.epoch import VotingEpoch

from configuration.config import ChainId, Protocols


class MessageLevel(enum.Enum):
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


@frozen
class Message:
    level: MessageLevel
    message: str

    @classmethod
    def builder(cls) -> "MessageBuilder":
        return MessageBuilder()


@define
class MessageBuilder:
    level: MessageLevel | None = None
    message: str | None = None

    network: int | None = None
    round: VotingEpoch | None = None
    protocol: int | str | None = None

    def copy(self) -> Self:
        return copy.copy(self)

    def _build(self) -> Message:
        assert self.level is not None
        assert self.message is not None

        s = io.StringIO()

        if self.network is not None:
            network = ChainId.id_to_name(self.network)
            s.write(f"network:{network} ")

        if self.round is not None:
            s.write(f"round:{self.round.id} ")

        if self.protocol is not None:
            protocol = Protocols.id_to_name(self.protocol)
            s.write(f"protocol:{protocol} ")

        s.write(self.message)

        s.seek(0)
        return Message(level=self.level, message=s.read())

    def build(self, level: MessageLevel, message: str) -> Message:
        return self.copy().add(level=level, message=message)._build()

    def add(
        self,
        *,
        level: MessageLevel | None = None,
        network: int | None = None,
        round: VotingEpoch | None = None,
        protocol: int | str | None = None,
        message: str | None = None,
    ) -> Self:
        if level is not None:
            self.level = level

        if network is not None:
            self.network = network

        if round is not None:
            self.round = round

        if protocol is not None:
            self.protocol = protocol

        if message is not None:
            self.message = message

        return self
