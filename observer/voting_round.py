from collections import defaultdict
from collections.abc import Sequence
from typing import Self

from attrs import define, field, frozen
from eth_typing import ChecksumAddress
from hexbytes import HexBytes
from py_flare_common.fsp.epoch.epoch import VotingEpoch
from py_flare_common.fsp.messaging.byte_parser import ByteParser
from py_flare_common.fsp.messaging.parse import parse_generic_tx
from py_flare_common.fsp.messaging.types import (
    FdcSubmit1,
    FdcSubmit2,
    FtsoSubmit1,
    FtsoSubmit2,
    ParsedPayload,
    SubmitSignatures,
)
from py_flare_common.ftso.commit import commit_hash
from py_flare_common.ftso.median import FtsoMedian, FtsoVote, calculate_median
from web3.types import BlockData, TxData

from .reward_epoch_manager import Entity, SigningPolicy
from .types import AttestationRequest, ProtocolMessageRelayed


@frozen
class WTxData:
    wrapped: TxData
    hash: HexBytes
    to_address: ChecksumAddress | None
    input: HexBytes
    block_number: int
    timestamp: int
    transaction_index: int
    from_address: ChecksumAddress
    value: int

    def is_first_or_second(self) -> bool:
        return (
            True
            if self.transaction_index == 0 or self.transaction_index == 1
            else False
        )

    @classmethod
    def from_tx_data(cls, tx_data: TxData, block_data: BlockData) -> Self:
        assert "hash" in tx_data
        assert "input" in tx_data
        assert "blockNumber" in tx_data
        assert "transactionIndex" in tx_data
        assert "from" in tx_data
        assert "value" in tx_data

        assert "timestamp" in block_data

        return cls(
            wrapped=tx_data,
            hash=tx_data["hash"],
            to_address=tx_data.get("to"),
            input=tx_data["input"],
            block_number=tx_data["blockNumber"],
            transaction_index=tx_data["transactionIndex"],
            from_address=tx_data["from"],
            value=tx_data["value"],
            timestamp=block_data["timestamp"],
        )


@frozen
class WParsedPayload[T]:
    parsed_payload: ParsedPayload[T]
    wtx_data: WTxData


@define
class WParsedPayloadList[T]:
    agg: list[WParsedPayload[T]] = field(factory=list)

    def extract_latest(self, r: range) -> WParsedPayload[T] | None:
        latest: WParsedPayload[T] | None = None

        for wpp in self.agg:
            wtx = wpp.wtx_data

            if not (r.start <= wtx.timestamp < r.stop):
                continue

            if latest is None or wtx.timestamp > latest.wtx_data.timestamp:
                latest = wpp

        return latest

    def extract_outside_of_bounds(
        self, r: range
    ) -> tuple[Sequence[WParsedPayload[T]], Sequence[WParsedPayload[T]]]:
        before: Sequence[WParsedPayload[T]] = []
        after: Sequence[WParsedPayload[T]] = []

        for wpp in self.agg:
            wtx = wpp.wtx_data

            if wtx.timestamp < r.start:
                before.append(wpp)

            if wtx.timestamp >= r.stop:
                after.append(wpp)

        return before, after


@define
class ParsedPayloadMapper[T]:
    by_identity: dict[ChecksumAddress, WParsedPayloadList[T]] = field(
        factory=lambda: defaultdict(WParsedPayloadList)
    )

    def insert(self, r: Entity, wpp: WParsedPayload[T]):
        self.by_identity[r.identity_address].agg.append(wpp)


@define
class VotingRoundProtocol[S1, S2, SS]:
    submit_1: ParsedPayloadMapper[S1] = field(factory=ParsedPayloadMapper)
    submit_2: ParsedPayloadMapper[S2] = field(factory=ParsedPayloadMapper)
    submit_signatures: ParsedPayloadMapper[SS] = field(factory=ParsedPayloadMapper)

    finalization: ProtocolMessageRelayed | None = None

    def insert_submit_1(self, e: Entity, pp: ParsedPayload[S1], wtx: WTxData) -> None:
        self.submit_1.insert(e, WParsedPayload(pp, wtx))

    def insert_submit_2(self, e: Entity, pp: ParsedPayload[S2], wtx: WTxData) -> None:
        self.submit_2.insert(e, WParsedPayload(pp, wtx))

    def insert_submit_signatures(
        self, e: Entity, pp: ParsedPayload[SS], wtx: WTxData
    ) -> None:
        self.submit_signatures.insert(e, WParsedPayload(pp, wtx))


@define
class FtsoVotingRoundProtocol(
    VotingRoundProtocol[FtsoSubmit1, FtsoSubmit2, SubmitSignatures]
):
    medians: list[FtsoMedian] = field(factory=list)

    def calculate_medians(
        self,
        epoch: VotingEpoch,
        signing_policy: SigningPolicy,
    ):
        next = epoch.next
        rd = next.reveal_deadline()

        # TODO:(matej) hacky way to determine nr of feeds, should read from
        # events when available
        # type: nr_of_feeds: times_found
        number_of_feeds: dict[int, int] = defaultdict(int)

        votes_to_consider: list[
            tuple[Entity, WParsedPayload[FtsoSubmit1], WParsedPayload[FtsoSubmit2]]
        ] = []

        for entity in signing_policy.entities:
            _submit_1 = self.submit_1.by_identity[entity.identity_address]
            submit_1 = _submit_1.extract_latest(range(epoch.start_s, epoch.end_s))

            _submit_2 = self.submit_2.by_identity[entity.identity_address]
            submit_2 = _submit_2.extract_latest(range(next.start_s, rd))

            if submit_1 is None or submit_2 is None:
                continue

            bp = ByteParser(parse_generic_tx(submit_2.wtx_data.input).ftso.payload)
            rnd = bp.uint256()
            feed_v = bp.drain()

            hashed = commit_hash(entity.submit_address, epoch.id, rnd, feed_v)

            if hashed != submit_1.parsed_payload.payload.commit_hash.hex():
                continue

            number_of_feeds[len(submit_2.parsed_payload.payload.values)] += 1

            votes_to_consider.append((entity, submit_1, submit_2))

        nr_feed = max(number_of_feeds.items(), key=lambda x: x[1])[0]

        for i in range(nr_feed):
            ftso_votes = []

            for e, _, s_2 in votes_to_consider:
                values = s_2.parsed_payload.payload.values

                if i >= len(values):
                    continue

                value = values[i]

                if value is None:
                    continue

                ftso_votes.append(
                    FtsoVote(
                        value=value,
                        weight=e.w_nat_capped_weight,
                    )
                )

            ftso_votes.sort(key=lambda x: x.value)

            median = calculate_median(ftso_votes)
            assert median is not None

            self.medians.append(median)


@define
class AttestationRequestMapper:
    agg: list[AttestationRequest] = field(factory=list)

    def sorted(self) -> list[AttestationRequest]:
        ret = []
        seen = set()

        for ar in sorted(self.agg, key=lambda x: (x.block, x.log_index)):
            if ar.data in seen:
                continue

            seen.add(ar.data)
            ret.append(ar)

        return list(reversed(ret))


@define
class FdcVotingRoundProtocol(
    VotingRoundProtocol[FdcSubmit1, FdcSubmit2, SubmitSignatures]
):
    requests: AttestationRequestMapper = field(factory=AttestationRequestMapper)
    consensus_bitvote: dict[bytes, int] = field(factory=lambda: defaultdict(int))


@define
class VotingRound:
    # epoch corresponding to the round
    voting_epoch: VotingEpoch

    ftso: FtsoVotingRoundProtocol = field(factory=FtsoVotingRoundProtocol)
    fdc: FdcVotingRoundProtocol = field(factory=FdcVotingRoundProtocol)
    submitted_signatures: bool = False


@define
class VotingRoundManager:
    finalized: int
    rounds: dict[VotingEpoch, VotingRound] = field(factory=dict)

    def get(self, v: VotingEpoch) -> VotingRound:
        if v not in self.rounds:
            self.rounds[v] = VotingRound(v)
        return self.rounds[v]

    def finalize(self, block: BlockData) -> list[VotingRound]:
        assert "timestamp" in block
        keys = list(self.rounds.keys())

        rounds = []
        for k in keys:
            if k.id <= self.finalized:
                self.rounds.pop(k, None)
                continue

            # need to wait until end of epoch to collect all signatures
            round_completed = k.next.end_s < block["timestamp"]

            if round_completed:
                self.finalized = max(self.finalized, k.id)
                rounds.append(self.rounds.pop(k))

        return rounds
