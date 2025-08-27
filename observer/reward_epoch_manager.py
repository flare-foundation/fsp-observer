from collections.abc import Sequence
from typing import Self

from attrs import define, field, frozen
from eth_typing import ChecksumAddress
from py_flare_common.fsp.epoch.epoch import RewardEpoch
from web3 import AsyncWeb3

from configuration.types import Configuration

from .message import Message, MessageLevel
from .types import (
    RandomAcquisitionStarted,
    SigningPolicyInitialized,
    VotePowerBlockSelected,
    VoterRegistered,
    VoterRegistrationInfo,
    VoterRemoved,
)


@frozen
class Node:
    node_id: str
    weight: int


@frozen
class Entity:
    identity_address: ChecksumAddress
    submit_address: ChecksumAddress
    submit_signatures_address: ChecksumAddress
    signing_policy_address: ChecksumAddress
    delegation_address: ChecksumAddress

    public_key: str
    nodes: list[Node]

    delegation_fee_bips: int

    w_nat_weight: int
    w_nat_capped_weight: int

    # used for internal calculation, (capped + stake) ** 3/4
    registration_weight: int

    # this is emitted in signing policy initialized event
    normalized_weight: int

    async def check_addresses(
        self, config: Configuration, w: AsyncWeb3
    ) -> Sequence[Message]:
        mb = Message.builder()
        messages = []

        addrs = (
            ("submit", self.submit_address),
            ("submit signatures", self.submit_signatures_address),
            ("signing policy", self.signing_policy_address),
        )

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


@frozen
class EntityMapper:
    by_identity_address: dict[ChecksumAddress, Entity] = field(factory=dict)
    by_submit_address: dict[ChecksumAddress, Entity] = field(factory=dict)
    by_submit_signatures_address: dict[ChecksumAddress, Entity] = field(factory=dict)
    by_signing_policy_address: dict[ChecksumAddress, Entity] = field(factory=dict)
    by_delegation_address: dict[ChecksumAddress, Entity] = field(factory=dict)
    by_omni: dict[ChecksumAddress, Entity] = field(factory=dict)

    def insert(self, e: Entity):
        self.by_identity_address[e.identity_address] = e
        self.by_submit_address[e.submit_address] = e
        self.by_submit_signatures_address[e.submit_signatures_address] = e
        self.by_signing_policy_address[e.signing_policy_address] = e
        self.by_delegation_address[e.delegation_address] = e

        self.by_omni[e.identity_address] = e
        self.by_omni[e.submit_address] = e
        self.by_omni[e.submit_signatures_address] = e
        self.by_omni[e.signing_policy_address] = e
        self.by_omni[e.delegation_address] = e


@frozen
class SigningPolicy:
    reward_epoch: RewardEpoch

    vote_power_block: int
    start_voting_round: int

    threshold: int
    seed: int
    signing_policy_bytes: str

    entities: list[Entity]
    entity_mapper: EntityMapper

    @classmethod
    def builder(cls) -> "SigningPolicyBuilder":
        return SigningPolicyBuilder()


@define
class SigningPolicyBuilder:
    reward_epoch: RewardEpoch | None = None

    random_acquisation_started: RandomAcquisitionStarted | None = None
    vote_power_block_selected: VotePowerBlockSelected | None = None

    voter_registered: list[VoterRegistered] = field(factory=list)
    voter_registration_info: list[VoterRegistrationInfo] = field(factory=list)
    voter_removed: list[VoterRemoved] = field(factory=list)

    signing_policy_initialized: SigningPolicyInitialized | None = None

    def for_epoch(self, r: RewardEpoch) -> Self:
        self.reward_epoch = r
        return self

    def add(
        self,
        event: RandomAcquisitionStarted
        | VotePowerBlockSelected
        | VoterRegistered
        | VoterRegistrationInfo
        | VoterRemoved
        | SigningPolicyInitialized,
    ) -> Self:
        if isinstance(event, RandomAcquisitionStarted):
            assert self.random_acquisation_started is None
            self.random_acquisation_started = event

        if isinstance(event, VotePowerBlockSelected):
            assert self.vote_power_block_selected is None
            self.vote_power_block_selected = event

        if isinstance(event, VoterRegistered):
            self.voter_registered.append(event)

        if isinstance(event, VoterRegistrationInfo):
            self.voter_registration_info.append(event)

        if isinstance(event, VoterRemoved):
            self.voter_removed.append(event)

        if isinstance(event, SigningPolicyInitialized):
            assert self.signing_policy_initialized is None
            self.signing_policy_initialized = event

        return self

    # def status(self, config: Configuration) -> str | None:
    #     ts_now = int(time.time())
    #     next_expected_ts = config.epoch.reward_epoch(self.id + 1).start_s
    #
    #     # current reads
    #     ras = self.random_acquisition_started
    #     vpbs = self.vote_power_block_selected
    #     sp = self.signing_policy
    #
    #     if not ras:
    #         return "collecting offers"
    #
    #     if ras and vpbs is None:
    #         return "selecting snapshot"
    #
    #     if vpbs is not None and sp is None:
    #         return "voter registration"
    #
    #     if sp is not None:
    #         svrs = config.epoch.voting_epoch(sp.start_voting_round_id).start_s
    #         if svrs > ts_now:
    #             return "ready for start"
    #
    #         # here svrs < ts_now
    #         if next_expected_ts > ts_now:
    #             return "active"
    #
    #         if next_expected_ts < ts_now:
    #             return "extended"

    def build(self) -> SigningPolicy:
        assert self.reward_epoch is not None
        rid = self.reward_epoch.id

        assert self.random_acquisation_started is not None
        assert self.random_acquisation_started.reward_epoch_id == rid

        assert self.vote_power_block_selected is not None
        assert self.vote_power_block_selected.reward_epoch_id == rid

        assert self.signing_policy_initialized is not None
        assert self.signing_policy_initialized.reward_epoch_id == rid

        assert len(self.voter_registered) == len(self.voter_registration_info)

        spa = {v.signing_policy_address: v.voter for v in self.voter_registered}
        vres = {v.voter: v for v in self.voter_registered}
        vries = {v.voter: v for v in self.voter_registration_info}

        entities: list[Entity] = []
        mapper = EntityMapper()

        for i, voter in enumerate(self.signing_policy_initialized.voters):
            weight = self.signing_policy_initialized.weights[i]
            vre = vres[spa[voter]]
            vrie = vries[spa[voter]]

            nodes = []
            for n, w in zip(vrie.node_ids, vrie.node_weights):
                nodes.append(Node(n, w))

            entity = Entity(
                identity_address=vre.voter,
                submit_address=vre.submit_address,
                submit_signatures_address=vre.submit_signatures_address,
                signing_policy_address=vre.signing_policy_address,
                delegation_address=vrie.delegation_address,
                public_key=vre.public_key,
                nodes=nodes,
                delegation_fee_bips=vrie.delegation_fee_bips,
                w_nat_weight=vrie.w_nat_weight,
                w_nat_capped_weight=vrie.w_nat_capped_weight,
                registration_weight=vre.registration_weight,
                normalized_weight=weight,
            )

            entities.append(entity)
            mapper.insert(entity)

        return SigningPolicy(
            reward_epoch=self.reward_epoch,
            vote_power_block=self.vote_power_block_selected.vote_power_block,
            start_voting_round=self.signing_policy_initialized.start_voting_round_id,
            threshold=self.signing_policy_initialized.threshold,
            seed=self.signing_policy_initialized.seed,
            signing_policy_bytes=self.signing_policy_initialized.signing_policy_bytes,
            entities=entities,
            entity_mapper=mapper,
        )


@define
class RewardManager:
    async def get_unclaimed_rewards(
        self, entity: Entity, config: Configuration, w: AsyncWeb3
    ) -> Sequence[Message]:
        mb = Message.builder()
        messages = []
        addresses = [
            entity.identity_address,
            entity.delegation_address,
            entity.signing_policy_address,
            entity.submit_address,
            entity.submit_signatures_address,
        ]
        claimable_func = w.eth.contract(
            abi=config.contracts.RewardManager.abi,
            address=config.contracts.RewardManager.address,
        ).functions["getRewardEpochIdsWithClaimableRewards"]
        min_re, max_re = await claimable_func().call()
        for address in addresses:
            for re in range(min_re, max_re + 1):
                for claim_type in range(4):
                    unclaimed_func = w.eth.contract(
                        abi=config.contracts.RewardManager.abi,
                        address=config.contracts.RewardManager.address,
                    ).functions["getUnclaimedRewardState"]
                    state = await unclaimed_func(address, re, claim_type).call()
                    # we are looking for claims that are not initialised
                    # and with non-zero amount
                    if not state[0] and state[1] > 0:
                        messages.append(
                            mb.build(
                                MessageLevel.WARNING,
                                f"Unclaimed rewards in reward epoch {re}, claim type {claim_type}, amount {state[1]}",  # noqa: E501
                            )
                        )
        return messages
