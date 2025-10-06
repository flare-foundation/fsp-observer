from typing import Any, Self

from attrs import frozen
from eth_account.messages import _hash_eip191_message, encode_defunct
from eth_typing import ChecksumAddress
from eth_utils.crypto import keccak
from hexbytes import HexBytes
from py_flare_common.fdc.attestation_source import AttestationSource
from py_flare_common.fdc.attestation_type import AttestationType
from py_flare_common.fsp.epoch.epoch import VotingEpoch
from web3.types import BlockData, EventData


@frozen
class ProtocolMessageRelayed:
    protocol_id: int
    voting_round_id: int
    is_secure_random: bool
    merkle_root: str
    timestamp: int

    def to_message(self) -> bytes:
        message = (
            self.protocol_id.to_bytes(1, "big")
            + self.voting_round_id.to_bytes(4, "big")
            + self.is_secure_random.to_bytes(1, "big")
            + bytes.fromhex(self.merkle_root)
        )

        return _hash_eip191_message(encode_defunct(keccak(message)))

    @classmethod
    def from_dict(cls, d: dict[str, Any], block_data: BlockData) -> Self:
        assert "timestamp" in block_data

        return cls(
            protocol_id=int(d["protocolId"]),
            voting_round_id=int(d["votingRoundId"]),
            is_secure_random=d["isSecureRandom"],
            merkle_root=d["merkleRoot"].hex(),
            timestamp=block_data["timestamp"],
        )


@frozen
class SigningPolicyInitialized:
    reward_epoch_id: int
    start_voting_round_id: int
    threshold: int
    seed: int
    voters: list[ChecksumAddress]
    weights: list[int]
    signing_policy_bytes: str
    timestamp: int

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        return cls(
            reward_epoch_id=int(d["rewardEpochId"]),
            start_voting_round_id=int(d["startVotingRoundId"]),
            threshold=int(d["threshold"]),
            seed=int(d["seed"]),
            voters=d["voters"],
            weights=[int(w) for w in d["weights"]],
            signing_policy_bytes=d["signingPolicyBytes"],
            timestamp=int(d["timestamp"]),
        )


@frozen
class VoterRegistered:
    reward_epoch_id: int
    voter: ChecksumAddress
    signing_policy_address: ChecksumAddress
    submit_address: ChecksumAddress
    submit_signatures_address: ChecksumAddress
    public_key: str
    registration_weight: int

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        return cls(
            reward_epoch_id=int(d["rewardEpochId"]),
            voter=d["voter"],
            signing_policy_address=d["signingPolicyAddress"],
            submit_address=d["submitAddress"],
            submit_signatures_address=d["submitSignaturesAddress"],
            public_key=d["publicKeyPart1"].hex() + d["publicKeyPart2"].hex(),
            registration_weight=int(d["registrationWeight"]),
        )


@frozen
class VoterRemoved:
    reward_epoch_id: int
    voter: ChecksumAddress

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        return cls(
            reward_epoch_id=int(d["rewardEpochId"]),
            voter=d["voter"],
        )


@frozen
class VoterRegistrationInfo:
    reward_epoch_id: int
    voter: ChecksumAddress
    delegation_address: ChecksumAddress
    delegation_fee_bips: int
    w_nat_weight: int
    w_nat_capped_weight: int
    node_ids: list[str]
    node_weights: list[int]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        return cls(
            reward_epoch_id=int(d["rewardEpochId"]),
            voter=d["voter"],
            delegation_address=d["delegationAddress"],
            delegation_fee_bips=int(d["delegationFeeBIPS"]),
            w_nat_weight=int(d["wNatWeight"]),
            w_nat_capped_weight=int(d["wNatCappedWeight"]),
            node_ids=[n.hex() for n in d["nodeIds"]],
            node_weights=[int(w) for w in d["nodeWeights"]],
        )


@frozen
class VotePowerBlockSelected:
    reward_epoch_id: int
    vote_power_block: int
    timestamp: int

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        return cls(
            reward_epoch_id=int(d["rewardEpochId"]),
            vote_power_block=int(d["votePowerBlock"]),
            timestamp=int(d["timestamp"]),
        )


@frozen
class RandomAcquisitionStarted:
    reward_epoch_id: int
    timestamp: int

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        return cls(
            reward_epoch_id=int(d["rewardEpochId"]),
            timestamp=int(d["timestamp"]),
        )


@frozen
class AttestationRequest:
    log_index: int
    block: int
    voting_epoch_id: VotingEpoch
    data: bytes

    @property
    def attestation_type(self) -> AttestationType:
        return AttestationType(self.data[0:32])

    @property
    def source_id(self) -> AttestationSource:
        return AttestationSource(self.data[32:64])

    @classmethod
    def from_dict(cls, data: EventData, voting_epoch: VotingEpoch) -> Self:
        d = data["args"]

        return cls(
            log_index=data["logIndex"],
            block=data["blockNumber"],
            voting_epoch_id=voting_epoch,
            data=d["data"],
        )


@frozen
class FastUpdateFeedsSubmitted:
    voting_round_id: int
    emitter_address: ChecksumAddress
    transaction_hash: HexBytes
    signing_policy_address: ChecksumAddress

    @classmethod
    def from_dict(cls, data: EventData):
        d = data["args"]
        address = data["address"]
        tx_hash = data["transactionHash"]

        return cls(
            voting_round_id=int(d["votingRoundId"]),
            emitter_address=address,
            transaction_hash=tx_hash,
            signing_policy_address=d["signingPolicyAddress"],
        )


@frozen
class FastUpdateFeeds:
    voting_round_id: int
    emitter_address: ChecksumAddress
    transaction_hash: HexBytes
    feeds: list[int]
    decimals: list[int]

    @classmethod
    def from_dict(cls, data: EventData):
        d = data["args"]
        address = data["address"]
        tx_hash = data["transactionHash"]

        return cls(
            voting_round_id=int(d["votingEpochId"]),
            emitter_address=address,
            transaction_hash=tx_hash,
            feeds=[int(v) for v in d["feeds"]],
            decimals=[int(v) for v in d["decimals"]],
        )


@frozen
class VoterPreRegistered:
    block: int
    emitter_address: ChecksumAddress
    transaction_hash: HexBytes
    voter: ChecksumAddress
    reward_epoch_id: int

    @classmethod
    def from_dict(cls, data: EventData):
        d = data["args"]
        block = data["blockNumber"]
        address = data["address"]
        tx_hash = data["transactionHash"]

        return cls(
            block=int(block),
            emitter_address=address,
            transaction_hash=tx_hash,
            voter=d["voter"],
            reward_epoch_id=int(d["rewardEpochId"]),
        )
