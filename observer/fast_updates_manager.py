from collections.abc import Sequence

from attrs import define, field, frozen
from eth_typing import ChecksumAddress
from web3 import AsyncWeb3

from configuration.types import Configuration
from observer.reward_epoch_manager import Entity

from .message import Message, MessageLevel


@frozen
class FastUpdate:
    reward_epoch_id: int
    entity: Entity
    update_array: list[int]


@define
class FastUpdatesManager:
    fast_updates: list[FastUpdate] = field(factory=list)
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
