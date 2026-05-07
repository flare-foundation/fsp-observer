from collections import deque
from collections.abc import Sequence
from enum import Enum
from typing import Self

from attrs import frozen
from py_flare_common.ftso.median import FtsoMedian

from configuration.config import Protocol
from observer import metrics
from observer.message import Message, MessageLevel
from observer.reward_epoch_manager import Entity, SigningPolicy


class Interval(Enum):
    LAST_2_HOURS = 2 * 60 * 60
    LAST_4_HOURS = 4 * 60 * 60
    LAST_6_HOURS = 6 * 60 * 60


@frozen
class MinimalConditionsConfig:
    ftso_median_band_bips = 50
    ftso_median_threshold_bips = 8000
    fast_updates_updates_per_block = 1.5
    staking_threshold_bips = 8000
    fdc_participation_threshold_bips = 8000


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
        metrics.FTSO_ANCHOR_FEEDS_SUCCESS_RATE.labels(
            identity_address=metrics.identity_address
        ).set(success_rate_bips)

        if success_rate_bips < MinimalConditionsConfig.ftso_median_threshold_bips:
            messages.append(
                mb.build(
                    MessageLevel.WARNING,
                    (
                        "not meeting minimal condition for FTSO anchor feeds in past "
                        f"two hours - success rate: {success_rate_bips / 100:.2f}%"
                    ),
                )
            )

        return messages

    def calculate_ftso_block_latency_feeds(
        self,
        max_exponent: int,
        entity: Entity,
        sp: SigningPolicy,
        last_update: int,
        current_block: int,
    ) -> Sequence[Message]:
        mb = Message.builder().add(network=self.network, protocol=Protocol.FAST_UPDATES)
        messages = []

        # NOTE:(janezicmatej) we can always use the active signing policy to do the
        # calculation even if previous update happened in the previous reward epoch

        total_weight = sum(e.normalized_weight for e in sp.entities)
        weight = entity.normalized_weight

        per_block = MinimalConditionsConfig.fast_updates_updates_per_block
        n_blocks = current_block - last_update
        if n_blocks >= max_exponent:
            n_blocks = max_exponent
        probability_ppb = int(
            1_000_000_000 * (1 - per_block * weight / total_weight) ** (n_blocks)
        )
        if n_blocks < max_exponent:
            if probability_ppb <= 100:
                level = MessageLevel.CRITICAL
                messages.append(
                    mb.build(
                        level,
                        f"didn't submit a fast update in {n_blocks} blocks",
                    )
                )
            return messages

        level = MessageLevel.WARNING
        if probability_ppb <= 100:
            level = MessageLevel.CRITICAL

        # FlareWatch patch (S40, 2026-05-07): suppress WARNING-level noise for
        # small-share validators where high probability_ppb is expected statistical
        # luck, not a real anomaly. CRITICAL still fires unconditionally.
        # Threshold: 5% (50_000_000 ppb). Rationale: at our 5/65505 normalized
        # weight on Songbird epoch 396, P(no submission in max_exponent blocks)
        # is ~79%, which is mathematically correct but operationally noise.
        # Whales (weight/total >= 0.01) drop probability_ppb below threshold
        # within ~max_exponent/2 blocks, preserving alert sensitivity for them.
        if level == MessageLevel.WARNING and probability_ppb > 50_000_000:
            return messages

        messages.append(
            mb.build(
                level,
                f"didn't submit a fast update in {n_blocks} blocks "
                f"(false positive probability: {probability_ppb / 10_000_000:.5f})%",
            )
        )

        return messages

    def calculate_staking(
        self, uptime_checks: int, node_connections: dict[str, deque[bool]]
    ) -> Sequence[Message]:
        mb = Message.builder().add(network=self.network, protocol=Protocol.STAKING)
        messages = []
        for node in node_connections:
            if (
                node_connections[node].count(True) * 10_000
            ) // uptime_checks < MinimalConditionsConfig.staking_threshold_bips:
                messages.append(
                    mb.build(
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
        if len(signatures) > 0:
            participation_bips = (signatures.count(True) * 10_000) // len(signatures)
            metrics.FDC_PARTICIPATION_RATE.labels(
                identity_address=metrics.identity_address
            ).set(participation_bips)
        if (
            len(signatures) > 0
            and participation_bips
            < MinimalConditionsConfig.fdc_participation_threshold_bips
        ):
            messages.append(
                mb.build(
                    MessageLevel.WARNING,
                    (
                        "Not meeting minimal condition for FDC participation "
                        "in the latest interval"
                    ),
                )
            )

        return messages
