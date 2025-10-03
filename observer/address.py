from collections.abc import Sequence

from eth_typing import ChecksumAddress
from web3 import AsyncWeb3

from configuration.types import Configuration
from observer.message import Message, MessageLevel


class AddressChecker:
    @classmethod
    async def check_addresses(
        cls,
        address_list: list[tuple[str, ChecksumAddress]],
        config: Configuration,
        w: AsyncWeb3,
    ) -> Sequence[Message]:
        mb = Message.builder()
        messages = []

        for name, addr in address_list:
            balance = await w.eth.get_balance(addr, "latest")
            if balance < config.fee_threshold * 1e18:
                level = MessageLevel.WARNING
                if balance <= 5e18:
                    level = MessageLevel.ERROR

                messages.append(
                    mb.build(
                        level,
                        f"low balance for {name} {addr} ({balance / 1e18:.4f} NAT)",
                    )
                )

        return messages
