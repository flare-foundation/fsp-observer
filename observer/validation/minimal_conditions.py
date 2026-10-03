import time
from collections import deque
from collections.abc import Sequence
from enum import Enum
from typing import Self

from attrs import frozen
from py_flare_common.ftso.median import FtsoMedian

from configuration.config import Protocol
from observer import metrics
from observer.alert_text import FAST_UPDATE_MISSED, build_alert
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
    # FlareWatch: the anchor-feed alert is trend-aware — re-alert on a
    # further drop of at least this many bips, else re-remind at most once
    # per this many seconds while the rate sits flat below threshold.
    ftso_anchor_drop_margin_bips = 200
    ftso_anchor_reminder_seconds = 6 * 60 * 60


class MinimalConditions:
    reward_epoch_id: int | None = None

    time_period: Interval = Interval.LAST_2_HOURS

    network: int | None = None

    # FlareWatch: FTSO anchor-feed alert state — trend-aware + deduped.
    # The raw check re-emitted the WARNING every cycle the 2h trailing rate
    # sat below threshold, and the embedded % defeated observer.py's dedup.
    _anchor_alerted: bool = False
    _anchor_last_bips: int | None = None
    _anchor_alerted_at: float | None = None

    # optional: fast update miss notifications with a false positive probability (in %)
    # above this are suppressed; None means no suppression (default, original behavior)
    false_positive_threshold: float | None = None

    def for_network(self, network: int) -> Self:
        self.network = network
        return self

    def for_reward_epoch(self, rid: int) -> Self:
        self.reward_epoch_id = rid
        return self

    def set_time_interval(self, interval: Interval) -> Self:
        self.time_period = interval
        return self

    def set_false_positive_threshold(self, threshold: float | None) -> Self:
        self.false_positive_threshold = threshold
        return self

    def calculate_ftso_anchor_feeds(
        self, medians: deque[list[FtsoMedian | None]], votes: deque[list[int | None]]
    ) -> Sequence[Message]:
        mb = Message.builder().add(network=self.network, protocol=Protocol.FTSO)
        messages = []

        total, total_hit = 0, 0

        for median_list, vote_list in zip(medians, votes, strict=False):
            for i in range(len(median_list)):
                total += 1

                if len(vote_list) <= i or vote_list[i] is None:
                    continue

                median = median_list[i]

                # feed had no consensus median this round, nothing to compare against
                if median is None:
                    continue

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

        # FlareWatch: trend-aware, deduped anchor-feed alerting. The raw
        # check re-emitted this WARNING every cycle while the 2h trailing
        # rate sat below threshold. Now: alert once on the breach, again
        # only on a meaningful further drop or a periodic reminder while it
        # sits flat, stay quiet while it recovers on its own, and emit one
        # INFO when it climbs back to the minimal condition.
        threshold = MinimalConditionsConfig.ftso_median_threshold_bips
        prev = self._anchor_last_bips
        self._anchor_last_bips = success_rate_bips
        pct = success_rate_bips / 100

        if success_rate_bips >= threshold:
            if self._anchor_alerted:
                self._anchor_alerted = False
                self._anchor_alerted_at = None
                messages.append(
                    mb.build(
                        MessageLevel.INFO,
                        (
                            "FTSO anchor feeds recovered - success rate back "
                            f"to {pct:.2f}% (>= minimal condition)"
                        ),
                    )
                )
            return messages

        # success_rate_bips < threshold
        now = time.monotonic()
        drop_margin = MinimalConditionsConfig.ftso_anchor_drop_margin_bips
        reminder_s = MinimalConditionsConfig.ftso_anchor_reminder_seconds
        climbing = prev is not None and success_rate_bips > prev
        worsening = prev is not None and success_rate_bips <= prev - drop_margin
        stale = (
            self._anchor_alerted_at is not None
            and now - self._anchor_alerted_at >= reminder_s
        )

        text: str | None = None
        if not self._anchor_alerted:
            text = (
                "not meeting minimal condition for FTSO anchor feeds in past "
                f"two hours - success rate: {pct:.2f}%"
            )
        elif worsening:
            text = (
                "still not meeting minimal condition for FTSO anchor feeds - "
                f"rate degrading to {pct:.2f}% (was {prev / 100:.2f}%)"
            )
        elif stale and not climbing:
            text = (
                "still not meeting minimal condition for FTSO anchor feeds - "
                f"success rate {pct:.2f}% (unchanged)"
            )

        if text is not None:
            self._anchor_alerted = True
            self._anchor_alerted_at = now
            messages.append(mb.build(MessageLevel.WARNING, text))

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

        # probability that missing this many blocks is just bad luck rather than a real
        # miss (a false positive); it drops as more blocks are missed
        probability_pct = 100 * (1 - per_block * weight / total_weight) ** (n_blocks)

        # below the exponent cap only report when we are near certain it is a real miss
        # (<= 0.00001%), otherwise stay silent; once the cap is reached always report,
        # escalating from warning to critical at the same confidence
        if n_blocks < max_exponent:
            if probability_pct <= 0.00001:
                messages.append(
                    mb.build(
                        MessageLevel.CRITICAL,
                        # FlareWatch: upstream logic, our build_alert formatting —
                        # stable summary (no embedded %) keeps observer.py's dedup
                        # working; the probability lives in evidence instead.
                        build_alert(
                            summary=(
                                f"didn't submit a fast update in {n_blocks} blocks"
                            ),
                            diagnosis=FAST_UPDATE_MISSED["diagnosis"],
                            evidence={
                                "identity": entity.identity_address,
                                "n_blocks_missed": n_blocks,
                                "current_block": current_block,
                                "last_update_block": last_update,
                                "probability_pct": round(probability_pct, 5),
                            },
                            actions=FAST_UPDATE_MISSED["actions"],
                        ),
                    )
                )
        else:
            level = MessageLevel.WARNING
            if probability_pct <= 0.00001:
                level = MessageLevel.CRITICAL
            # NOTE: our old S40 hardcoded 5% WARNING suppression is superseded by
            # upstream's configurable false_positive_threshold (set
            # FALSE_POSITIVE_THRESHOLD=5 in the deploy env for identical behavior).
            messages.append(
                mb.build(
                    level,
                    build_alert(
                        summary=(
                            f"didn't submit a fast update in {n_blocks} blocks"
                        ),
                        diagnosis=FAST_UPDATE_MISSED["diagnosis"],
                        evidence={
                            "identity": entity.identity_address,
                            "n_blocks_missed": n_blocks,
                            "current_block": current_block,
                            "last_update_block": last_update,
                            "probability_pct": round(probability_pct, 5),
                            "max_exponent": max_exponent,
                        },
                        actions=FAST_UPDATE_MISSED["actions"],
                    ),
                )
            )

        # additional operator-configurable suppression on top of the above: when set,
        # drop the notification if its false positive probability is above the
        # threshold; this mainly quiets low weight entities, whose probability stays
        # high longer
        if (
            self.false_positive_threshold is not None
            and probability_pct > self.false_positive_threshold
        ):
            return []

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
