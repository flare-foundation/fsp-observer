from collections.abc import Sequence

from attrs import define, field, frozen
from eth_typing import ChecksumAddress
from web3 import AsyncWeb3

from configuration.types import Configuration
from observer.address import AddressChecker

from .message import Message, MessageLevel


@frozen
class FastUpdate:
    reward_epoch_id: int
    address: ChecksumAddress
    update_array: list[int]


@define
class FastUpdatesManager:
    last_update_block: int
    last_update: FastUpdate
    address_list: set[ChecksumAddress] = field(factory=set)

    async def check_addresses(
        self, config: Configuration, w: AsyncWeb3
    ) -> Sequence[Message]:
        addrs = [("fast updates address", address) for address in self.address_list]

        return await AddressChecker.check_addresses(addrs, config, w)

    def check_update_length(
        self, nr_of_feeds: int, fast_update_re: int
    ) -> Sequence[Message]:
        mb = Message.builder()
        messages = []
        level = MessageLevel.WARNING

        if self.last_update.reward_epoch_id != fast_update_re or not nr_of_feeds:
            return messages

        # round up to the next multiple of 8 as update array is encoded in integer
        # number of bytes
        expected = nr_of_feeds + 7 - (nr_of_feeds - 1) % 8
        submitted = len(self.last_update.update_array)

        if expected != submitted:
            messages.append(
                mb.build(
                    level,
                    (
                        f"incorrect number of sent feeds, should be {expected} but got "
                        f"{submitted}"
                    ),
                )
            )

        return messages
