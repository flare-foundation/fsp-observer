from collections import deque
from collections.abc import Sequence
from enum import Enum
from typing import Self

from attrs import frozen
from py_flare_common.ftso.median import FtsoMedian

from configuration.config import Protocol
from observer.fast_updates_manager import FastUpdatesManager
from observer.message import Message, MessageLevel
from observer.reward_epoch_manager import Entity
from observer.signing_policy_manager import SigningPolicyManager


class Interval(Enum):
    LAST_2_HOURS = 2 * 60 * 60
    LAST_4_HOURS = 4 * 60 * 60
    LAST_6_HOURS = 6 * 60 * 60


@frozen
class MinimalConditionsConfig:
    ftso_median_band_bips = 50
    ftoo_median_threshold_bips = 8000
    threshold = 0.8


class MinimalConditions:
    reward_epoch_id: int | None = None

    time_period: Interval = Interval.LAST_2_HOURS

    network: int | None = None

    def for_network(self, network: int) -> Self:
        self.network = network
        return self

    def for_reward_epoch(self, rid: int) -> Self:
        self.reward_epoch_id = rid
        return self

    def set_time_interval(self, interval: Interval) -> Self:
        self.time_period = interval
        return self

    def calculate_ftso_anchor_feeds(
        self, medians: deque[list[FtsoMedian]], votes: deque[list[int | None]]
    ) -> Sequence[Message]:
        mb = Message.builder().add(network=self.network, protocol=Protocol.FTSO)
        messages = []

        total, total_hit = 0, 0

        for median_list, vote_list in zip(medians, votes):
            for i in range(len(median_list)):
                total += 1

                if len(vote_list) <= i or vote_list[i] is None:
                    continue

                median = median_list[i]
                vote = vote_list[i]

                assert vote is not None

                band = MinimalConditionsConfig.ftso_median_band_bips
                low = median.value * (10_000 - band) / 10_000
                high = median.value * (10_000 + band) / 10_000

                if low <= vote <= high:
                    total_hit += 1

        if not total:
            return messages

        success_rate_bips = (total_hit * 10000) // total

        if success_rate_bips < MinimalConditionsConfig.ftoo_median_threshold_bips:
            messages.append(
                mb.add(network=config.chain_id).build(
                    MessageLevel.WARNING,
                    (
                        "not meeting minimal condition for FTSO anchor feeds in past "
                        f"two hours - success rate: {success_rate_bips / 100:.2f}%"
                    ),
                )
            )

        return messages

    def calculate_ftso_block_latency_feeds(
        self, entity: Entity, spm: SigningPolicyManager, fum: FastUpdatesManager
    ) -> Sequence[Message]:
        mb = Message.builder().add(network=self.network, protocol=Protocol.FAST_UPDATES)
        messages = []
        previous_total_active_weight = sum(
            [e.normalized_weight for e in spm.previous_policy.entities]
        )
        previous_normalized_weight = (
            spm.previous_policy.entity_mapper.by_identity_address[
                entity.identity_address
            ].normalized_weight
        )
        previous_number_of_updates = len(
            list(
                filter(
                    lambda x: x.reward_epoch_id == spm.previous_policy.reward_epoch.id,
                    fum.fast_updates,
                )
            )
        )
        previous_expected_updates = (
            MinimalConditionsConfig.threshold
            * previous_number_of_updates
            * previous_normalized_weight
            / previous_total_active_weight
        )

        total_active_weight = sum(
            [e.normalized_weight for e in spm.current_policy.entities]
        )
        normalized_weight = spm.current_policy.entity_mapper.by_identity_address[
            entity.identity_address
        ].normalized_weight
        number_of_updates = len(
            list(
                filter(
                    lambda x: x.reward_epoch_id == spm.current_policy.reward_epoch.id,
                    fum.fast_updates,
                )
            )
        )
        expected_updates = (
            MinimalConditionsConfig.threshold
            * number_of_updates
            * normalized_weight
            / total_active_weight
        )

        previous_actual_updates = len(
            list(
                filter(
                    lambda x: x.address in fum.address_list
                    and x.reward_epoch_id == spm.previous_policy.reward_epoch.id,
                    fum.fast_updates,
                )
            )
        )
        actual_updates = len(
            list(
                filter(
                    lambda x: x.address in fum.address_list
                    and x.reward_epoch_id == spm.current_policy.reward_epoch.id,
                    fum.fast_updates,
                )
            )
        )
        if not (
            (previous_actual_updates + actual_updates)
            >= (previous_expected_updates + expected_updates)
        ):
            messages.append(
                mb.add(network=config.chain_id).build(
                    MessageLevel.WARNING,
                    (
                        "Not meeting minimal condition for fast updates "
                        "in the latest interval"
                    ),
                )
            )
        return messages

    def calculate_staking(
        self, uptime_checks: int, node_connections: dict[str, deque]
    ) -> Sequence[Message]:
        mb = Message.builder().add(network=self.network, protocol=Protocol.STAKING)
        messages = []
        for node in node_connections:
            if (
                node_connections[node].count(True) / uptime_checks
                < MinimalConditionsConfig.threshold
            ):
                messages.append(
                    mb.add(network=config.chain_id).build(
                        MessageLevel.WARNING,
                        (
                            f"Node {node} not meeting minimal condition for "
                            "staking in the latest interval"
                        ),
                    )
                )

        return messages

    def calculate_fdc_participation(self, signatures: deque[bool]) -> Sequence[Message]:
        mb = Message.builder().add(network=self.network, protocol=Protocol.FDC)
        messages = []
        if (
            len(signatures) > 0
            and signatures.count(True) / len(signatures)
            < MinimalConditionsConfig.threshold
        ):
            messages.append(
                mb.add(network=config.chain_id).build(
                    MessageLevel.WARNING,
                    (
                        "Not meeting minimal condition for FDC participation "
                        "in the latest interval"
                    ),
                )
            )

        return messages
