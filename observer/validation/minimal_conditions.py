from enum import Enum
from typing import Self

from py_flare_common.ftso.median import FtsoMedian

from configuration.types import Configuration
from observer.fast_updates_manager import FastUpdatesManager
from observer.message import Message, MessageLevel
from observer.reward_epoch_manager import Entity


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

    def calculate_ftso_anchor_feeds(
        self, medians: list[list[FtsoMedian]], votes: list[list[int | None]]
    ) -> list[Message]:
        mb = Message.builder()
        messages = []
        total_rounds = len(medians)
        if total_rounds > 0:
            for i in range(len(medians[0])):
                rounds_in_interval = 0
                for median_list, vote_list in zip(medians, votes):
                    print(vote_list)
                    if vote_list[i] is None:
                        continue
                    vote = vote_list[i]
                    assert vote
                    if (
                        0.995 * median_list[i].value
                        <= vote
                        <= 1.005 * median_list[i].value
                    ):
                        rounds_in_interval += 1
                if rounds_in_interval / total_rounds >= 0.8:
                    pass
                else:
                    messages.append(
                        mb.build(
                            MessageLevel.WARNING,
                            f"Not meeting minimal condition for FTSO anchor feeds in reward epoch {self.reward_epoch_id}, feed: {i}",  # noqa: E501
                        )
                    )

        return messages

    def calculate_ftso_block_latency_feeds(
        self, entity: Entity, weights: list[int], fum: FastUpdatesManager
    ) -> list[Message]:
        mb = Message.builder()
        messages = []
        total_active_weight = sum(weights)
        normalized_weight = entity.normalized_weight
        number_of_updates = len(
            list(
                filter(
                    lambda x: x.reward_epoch_id == self.reward_epoch_id,
                    fum.fast_updates,
                )
            )
        )
        expected_updates = (
            0.8 * number_of_updates * normalized_weight / total_active_weight
        )

        def filter1(x):
            return x.entity == entity

        def filter2(x):
            return x.reward_epoch_id == self.reward_epoch_id

        filters = (filter1, filter2)
        actual_updates = len(
            list(filter(lambda x: all(f(x) for f in filters), fum.fast_updates))
        )

        if (
            actual_updates >= expected_updates
            or normalized_weight < 0.02 * total_active_weight
        ):
            return messages
        else:
            messages.append(
                mb.build(
                    MessageLevel.WARNING,
                    f"Not meeting minimal condition for fast updates in reward epoch {self.reward_epoch_id}",  # noqa: E501
                )
            )
            return messages

    def calculate_staking(
        self, uptime_checks: int, node_connections: dict[str, int]
    ) -> list[Message]:
        mb = Message.builder()
        messages = []
        for node in node_connections:
            if node_connections[node] / uptime_checks < 0.8:
                messages.append(
                    mb.build(
                        MessageLevel.WARNING,
                        f"Node {node} not meeting minimal condition for staking in reward epoch {self.reward_epoch_id}",  # noqa: E501
                    )
                )

        return messages

    def calculate_fdc_participation(
        self, signatures: list[bool], config: Configuration
    ) -> list[Message]:
        mb = Message.builder()
        messages = []
        assert self.reward_epoch_id
        number_of_epochs = (
            config.epoch.voting_epoch_factory.now_id()
            - config.epoch.reward_epoch(self.reward_epoch_id).to_first_voting_epoch().id
        )
        if sum(signatures) / number_of_epochs < 0.8:
            messages.append(
                mb.build(
                    MessageLevel.WARNING,
                    f"Not meeting minimal condition for FDC participation in reward epoch {self.reward_epoch_id}",  # noqa: E501
                )
            )

        return messages
