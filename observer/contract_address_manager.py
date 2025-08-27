from attrs import define
from eth_typing import ChecksumAddress

from observer.message import Message, MessageLevel


@define
class ContractAddressManager:
    submission: ChecksumAddress
    relay: ChecksumAddress

    def check_submission_address(self, address) -> list[Message]:
        mb = Message.builder()
        messages = []
        if address != self.submission:
            messages.append(
                mb.build(MessageLevel.CRITICAL, "Incorrect Submmission address")
            )
        return messages

    def check_relay_address(self, address) -> list[Message]:
        mb = Message.builder()
        messages = []
        if address != self.relay:
            messages.append(mb.build(MessageLevel.CRITICAL, "Incorrect Relay address"))
        return messages
