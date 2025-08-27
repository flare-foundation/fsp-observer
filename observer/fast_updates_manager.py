from collections import deque
from collections.abc import Sequence

from attrs import define, field, frozen
from eth_typing import ChecksumAddress
from web3 import AsyncWeb3

from configuration.types import Configuration

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
        mb = Message.builder()
        messages = []

        addrs = (("fast updates address", address) for address in self.address_list)

        for name, addr in addrs:
            balance = await w.eth.get_balance(addr, "latest")
            if balance < config.fee_threshold * 1e18:
                level = MessageLevel.WARNING
                if balance <= 5e18:
                    level = MessageLevel.ERROR

                messages.append(
                    mb.build(
                        level,
                        f"low balance for {name} address ({balance / 1e18:.4f} NAT)",
                    )
                )

        return messages

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
        if nr_of_feeds > 0 and len(fu.update_array) != nr_of_feeds:
            messages.append(
                mb.build(
                    level,
                    f"Incorrect length of last update array, should be {nr_of_feeds} but got {len(fu.update_array)}",  # noqa: E501
                )
            )
        return messages
