from collections import deque
from collections.abc import Sequence
from enum import Enum
from typing import Self

from py_flare_common.ftso.median import FtsoMedian

from observer.fast_updates_manager import FastUpdatesManager
from observer.message import Message, MessageLevel
from observer.reward_epoch_manager import Entity
from observer.signing_policy_manager import SigningPolicyManager


class Interval(Enum):
    LAST_2_HOURS = 2 * 60 * 60
    LAST_4_HOURS = 4 * 60 * 60
    LAST_6_HOURS = 6 * 60 * 60


class MinimalConditions:
    reward_epoch_id: int | None = None

    time_period: Interval = Interval.LAST_2_HOURS

    def for_reward_epoch(self, rid: int) -> Self:
        self.reward_epoch_id = rid
        return self

    def set_time_interval(self, interval: Interval) -> Self:
        self.time_period = interval
        return self

    def calculate_ftso_anchor_feeds(
        self, medians: deque[list[FtsoMedian]], votes: deque[list[int | None]]
    ) -> Sequence[Message]:
        mb = Message.builder()
        messages = []
        total_rounds = len(medians)
        if total_rounds == 0:
            return messages
        for i in range(len(medians[0])):
            rounds_in_interval = 0
            for median_list, vote_list in zip(medians, votes):
                if len(vote_list) < i + 1 or vote_list[i] is None:
                    continue
                vote = vote_list[i]
                assert vote
                if 0.995 * median_list[i].value <= vote <= 1.005 * median_list[i].value:
                    rounds_in_interval += 1
            if rounds_in_interval / total_rounds < 0.8:
                messages.append(
                    mb.build(
                        MessageLevel.WARNING,
                        f"Not meeting minimal condition for FTSO anchor feed in the latest interval, feed index: {i}",  # noqa: E501
                    )
                )

        return messages

    def calculate_ftso_block_latency_feeds(
        self, entity: Entity, spm: SigningPolicyManager, fum: FastUpdatesManager
    ) -> Sequence[Message]:
        mb = Message.builder()
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
            0.8
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
            0.8 * number_of_updates * normalized_weight / total_active_weight
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
                mb.build(
                    MessageLevel.WARNING,
                    "Not meeting minimal condition for fast updates in the latest interval",  # noqa: E501
                )
            )
        return messages

    def calculate_staking(
        self, uptime_checks: int, node_connections: dict[str, deque]
    ) -> Sequence[Message]:
        mb = Message.builder()
        messages = []
        for node in node_connections:
            if node_connections[node].count(True) / uptime_checks < 0.8:
                messages.append(
                    mb.build(
                        MessageLevel.WARNING,
                        f"Node {node} not meeting minimal condition for staking in the latest interval",  # noqa: E501
                    )
                )

        return messages

    def calculate_fdc_participation(self, signatures: deque[bool]) -> Sequence[Message]:
        mb = Message.builder()
        messages = []
        if len(signatures) > 0 and signatures.count(True) / len(signatures) < 0.8:
            messages.append(
                mb.build(
                    MessageLevel.WARNING,
                    "Not meeting minimal condition for FDC participation in the latest interval",  # noqa: E501
                )
            )

        return messages
