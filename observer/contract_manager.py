from collections.abc import Sequence

from attrs import define

from configuration.types import Contract, Contracts
from observer.message import Message, MessageLevel


@define
class ContractManager:
    contracts: Contracts

    def get_contracts_list(self) -> list[Contract]:
        return [getattr(self.contracts, c.name) for c in self.contracts.__attrs_attrs__]  # type: ignore

    def get_events(self):
        contracts = self.get_contracts_list()
        return {e.signature: e for c in contracts for e in c.events.values()}

    def check_submission_address(self, address) -> Sequence[Message]:
        mb = Message.builder()
        messages = []
        if address != self.contracts.Submission.address:
            messages.append(
                mb.build(MessageLevel.CRITICAL, "Incorrect Submmission address")
            )
        return messages

    def check_relay_address(self, address) -> Sequence[Message]:
        mb = Message.builder()
        messages = []
        if address != self.contracts.Relay.address:
            messages.append(mb.build(MessageLevel.CRITICAL, "Incorrect Relay address"))
        return messages
