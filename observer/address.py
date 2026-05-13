from collections.abc import Sequence

from eth_typing import ChecksumAddress
from web3 import AsyncWeb3

from configuration.types import Configuration
from observer import metrics
from observer.alert_text import LOW_BALANCE, build_alert
from observer.message import Message, MessageLevel


class AddressChecker:
    @classmethod
    async def check_addresses(
        cls,
        address_list: list[tuple[str, ChecksumAddress]],
        config: Configuration,
        w: AsyncWeb3,
    ) -> Sequence[Message]:
        mb = Message.builder().add(network=config.chain_id)
        messages = []

        for name, addr in address_list:
            balance = await w.eth.get_balance(addr, "latest")
            metrics.ADDRESS_BALANCE.labels(
                identity_address=metrics.identity_address, address=addr, role=name
            ).set(balance)
            if balance < config.fee_threshold * 1e18:
                level = MessageLevel.WARNING
                if balance <= 5e18:
                    level = MessageLevel.ERROR

                balance_nat = balance / 1e18
                messages.append(
                    mb.build(
                        level,
                        build_alert(
                            summary=(
                                f"low balance for {name} {addr} "
                                f"({balance_nat:.4f} NAT)"
                            ),
                            diagnosis=LOW_BALANCE["diagnosis"],
                            evidence={
                                "role": name,
                                "address": addr,
                                "balance_nat": f"{balance_nat:.4f}",
                                "threshold_nat": f"{config.fee_threshold:.4f}",
                                "severity": (
                                    "ERROR" if balance <= 5e18 else "WARNING"
                                ),
                            },
                            actions=LOW_BALANCE["actions"],
                        ),
                    )
                )

        return messages
