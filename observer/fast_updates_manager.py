from collections import deque
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
    fast_updates: deque[FastUpdate] = field(factory=deque)
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
        fus = list(
            filter(
                lambda x: x.reward_epoch_id >= fast_update_re
                and x.address in self.address_list,
                self.fast_updates,
            )
        )
        if len(fus) > 0:
            fu = fus[-1]
        else:
            return messages
        rounded_nr_feeds = nr_of_feeds
        # update arrays are whole bytes, so we need to round the number of feeds
        # up to the nearest multiple of 8 if it is not one already
        if nr_of_feeds % 8 != 0:
            rounded_nr_feeds = nr_of_feeds + (8 - nr_of_feeds % 8)
        if rounded_nr_feeds > 0 and len(fu.update_array) != rounded_nr_feeds:
            messages.append(
                mb.build(
                    level,
                    (
                        "Incorrect length of last update array, should be"
                        f" {rounded_nr_feeds} but got {len(fu.update_array)}"
                    ),
                )
            )
        return messages
