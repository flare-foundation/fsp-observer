import asyncio
import hashlib
import logging
import time
from collections import defaultdict, deque
from collections.abc import Awaitable, Sequence
from functools import partial
from itertools import chain
from typing import Any, Self

import requests
from eth_abi.abi import encode
from eth_account._utils.signing import to_standard_v
from eth_account.messages import _hash_eip191_message, encode_defunct
from eth_keys.datatypes import Signature as EthSignature
from py_flare_common.b58 import flare_b58_encode_check
from py_flare_common.fsp.epoch.epoch import RewardEpoch
from py_flare_common.fsp.messaging import (
    parse_submit1_tx,
    parse_submit2_tx,
    parse_submit_signature_tx,
)
from py_flare_common.fsp.messaging.types import Signature as SSignature
from py_flare_common.ftso.median import FtsoMedian
from web3 import AsyncWeb3
from web3._utils.events import get_event_data
from web3.middleware import ExtraDataToPOAMiddleware
from web3.types import TxData

from configuration.types import (
    Configuration,
    un_prefix_0x,
)
from observer.contract_manager import ContractManager
from observer.fast_updates_manager import FastUpdate, FastUpdatesManager
from observer.reward_epoch_manager import (
    RewardManager,
    SigningPolicy,
)
from observer.signing_policy_manager import SigningPolicyManager
from observer.types import (
    AttestationRequest,
    FastUpdateFeeds,
    FastUpdateFeedsSubmitted,
    ProtocolMessageRelayed,
    RandomAcquisitionStarted,
    SigningPolicyInitialized,
    VotePowerBlockSelected,
    VoterPreRegistered,
    VoterRegistered,
    VoterRegistrationInfo,
    VoterRemoved,
)
from observer.validation.minimal_conditions import MinimalConditions
from observer.validation.validation import extract_round_for_entity, validate_round

from .message import Message, MessageLevel
from .notification import (
    notify_discord,
    notify_discord_embed,
    notify_generic,
    notify_slack,
    notify_telegram,
)
from .voting_round import (
    VotingRoundManager,
    WTxData,
)

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s\t%(levelname)s\t%(name)s\t%(message)s",
    level="INFO",
)


def node_id_to_representation(node_id):
    decoded = bytes.fromhex(node_id)
    return f"NodeID-{flare_b58_encode_check(decoded).decode()}"


class Signature(EthSignature):
    @classmethod
    def from_vrs(cls, s: SSignature) -> Self:
        return cls(
            vrs=(
                to_standard_v(int(s.v, 16)),
                int(s.r, 16),
                int(s.s, 16),
            )
        )

    @classmethod
    def from_dict(cls, dict: dict[str, Any]):
        return Signature(
            vrs=(
                to_standard_v(dict["v"]),
                int.from_bytes((dict["r"]), "big"),
                int.from_bytes((dict["s"]), "big"),
            )
        )

    def recover_addr_from_msg(self, sp_hash: str) -> str:
        return self.recover_public_key_from_msg_hash(
            _hash_eip191_message(encode_defunct(hexstr=sp_hash))
        ).to_checksum_address()


def calculate_update_from_tx(config: Configuration, w: AsyncWeb3, tx: TxData):
    submission = w.eth.contract(
        abi=config.contracts.Submission.abi, address=config.contracts.Submission.address
    )
    fast_updates = w.eth.contract(
        abi=config.contracts.FastUpdater.abi,
        address=config.contracts.FastUpdater.address,
    )
    assert "input" in tx
    proxy_input = submission.decode_function_input(tx["input"])[1]["_data"].hex()
    updates = fast_updates.decode_function_input("0x470e91df" + proxy_input)[1][
        "_updates"
    ]

    cred = updates["sortitionCredential"]
    signed_message = (
        hashlib.sha256(
            encode(
                ["uint256", "(uint256,(uint256,uint256),uint256,uint256)", "bytes"],
                [
                    updates["sortitionBlock"],
                    (
                        cred["replicate"],
                        (cred["gamma"]["x"], cred["gamma"]["y"]),
                        cred["c"],
                        cred["s"],
                    ),
                    updates["deltas"],
                ],
            )
        )
        .digest()
        .hex()
    )

    signing_policy_address = un_prefix_0x(
        Signature.from_dict(updates["signature"]).recover_addr_from_msg(signed_message)
    )

    assert "from" in tx
    address = tx["from"]

    array = "".join(f"{i:08b}" for i in updates["deltas"])
    assert len(array) % 2 == 0
    signed_array = [
        -int(array[u + 1]) if array[u] == "1" else int(array[u + 1])
        for u in range(0, len(array), 2)
    ]

    return signing_policy_address, address, signed_array


async def get_block_production(w: AsyncWeb3) -> float:
    latest_block = await w.eth.get_block("latest")
    assert "timestamp" in latest_block
    assert "number" in latest_block
    to_compare = min(1_000_000, int(latest_block["number"]) - 1)
    comparison_block = await w.eth.get_block(int(latest_block["number"]) - to_compare)
    assert "timestamp" in comparison_block
    time_delta = latest_block["timestamp"] - comparison_block["timestamp"]
    block_production = time_delta / to_compare
    return block_production


def calculate_maximum_exponent(block_production: float, config: Configuration) -> int:
    blocks_in_epoch = int(
        config.epoch.reward_epoch_factory.duration() / block_production
    )
    max_exponent = blocks_in_epoch // 100
    return max_exponent


async def find_voter_registration_blocks(
    w: AsyncWeb3,
    current_block_id: int,
    reward_epoch: RewardEpoch,
) -> tuple[int, int]:
    # there are roughly 3600 blocks in an hour
    avg_block_time = 3600 / 3600
    current_ts = int(time.time())

    # find timestamp that is more than 2h30min (=9000s) before start_of_epoch_ts
    target_start_ts = reward_epoch.start_s - 9000
    start_diff = current_ts - target_start_ts

    start_block_id = current_block_id - int(start_diff / avg_block_time)
    block = await w.eth.get_block(start_block_id)
    assert "timestamp" in block
    d = block["timestamp"] - target_start_ts
    while abs(d) > 600:
        start_block_id -= 100 * (d // abs(d))
        block = await w.eth.get_block(start_block_id)
        assert "timestamp" in block
        d = block["timestamp"] - target_start_ts

    # end timestamp is 1h (=3600s) before start_of_epoch_ts
    target_end_ts = reward_epoch.start_s - 3600
    end_diff = current_ts - target_end_ts
    end_block_id = current_block_id - int(end_diff / avg_block_time)

    block = await w.eth.get_block(end_block_id)
    assert "timestamp" in block
    d = block["timestamp"] - target_end_ts
    while abs(d) > 600:
        end_block_id -= 100 * (d // abs(d))
        block = await w.eth.get_block(end_block_id)
        assert "timestamp" in block
        d = block["timestamp"] - target_end_ts

    return (start_block_id, end_block_id)


async def get_signing_policy_events(
    w: AsyncWeb3,
    config: Configuration,
    reward_epoch: RewardEpoch,
    start_block: int,
    end_block: int,
) -> SigningPolicy:
    # reads logs for given blocks for the informations about the signing policy

    builder = SigningPolicy.builder().for_epoch(reward_epoch)

    contracts = [
        config.contracts.VoterRegistry,
        config.contracts.FlareSystemsCalculator,
        config.contracts.Relay,
        config.contracts.FlareSystemsManager,
    ]

    event_names = {
        # relay
        "SigningPolicyInitialized",
        # flare systems calculator
        "VoterRegistrationInfo",
        # flare systems manager
        "RandomAcquisitionStarted",
        "VotePowerBlockSelected",
        "VoterRegistered",
        "VoterRemoved",
    }
    event_signatures = {
        e.signature: e
        for c in contracts
        for e in c.events.values()
        if e.name in event_names
    }

    block_logs = await w.eth.get_logs(
        {
            "address": [contract.address for contract in contracts],
            "fromBlock": start_block,
            "toBlock": end_block,
        }
    )

    for log in block_logs:
        sig = log["topics"][0]

        if sig.hex() not in event_signatures:
            continue

        event = event_signatures[sig.hex()]
        data = get_event_data(w.eth.codec, event.abi, log)

        match event.name:
            case "VoterRegistered":
                e = VoterRegistered.from_dict(data["args"])
            case "VoterRemoved":
                e = VoterRemoved.from_dict(data["args"])
            case "VoterRegistrationInfo":
                e = VoterRegistrationInfo.from_dict(data["args"])
            case "SigningPolicyInitialized":
                e = SigningPolicyInitialized.from_dict(data["args"])
            case "VotePowerBlockSelected":
                e = VotePowerBlockSelected.from_dict(data["args"])
            case "RandomAcquisitionStarted":
                e = RandomAcquisitionStarted.from_dict(data["args"])
            case x:
                raise ValueError(f"Unexpected event {x}")
        builder.add(e)

        # signing policy initialized is the last event that gets emitted
        if event.name == "SigningPolicyInitialized":
            break

    return builder.build()


def log_message(config: Configuration, message: Message):
    LOGGER.log(message.level.value, message.message)

    n = config.notification

    notify_discord(n.discord, message)
    notify_discord_embed(n.discord_embed, message)
    notify_slack(n.slack, message)
    notify_telegram(n.telegram, message)
    notify_generic(n.generic, message)


async def cron(
    check_functions: Sequence[Awaitable[Sequence[Message]]],
) -> Sequence[Message]:
    results = await asyncio.gather(*check_functions)

    return list(chain.from_iterable(results))


async def observer_loop(config: Configuration) -> None:
    w = AsyncWeb3(
        AsyncWeb3.AsyncHTTPProvider(config.rpc_url),
        middleware=[ExtraDataToPOAMiddleware],
    )

    # reasignments for quick access
    ve = config.epoch.voting_epoch
    # re = config.epoch.reward_epoch
    vef = config.epoch.voting_epoch_factory
    ref = config.epoch.reward_epoch_factory

    # get current voting round and reward epoch
    block = await w.eth.get_block("latest")
    assert "timestamp" in block
    assert "number" in block
    reward_epoch = ref.from_timestamp(block["timestamp"])
    voting_epoch = vef.from_timestamp(block["timestamp"])

    # we first fill signing policy for current reward epoch

    # voter registration period is 2h before the reward epoch and lasts 30min
    # find block that has timestamp approx. 2h30min before the reward epoch
    # and block that has timestamp approx. 1h before the reward epoch
    lower_block_id, end_block_id = await find_voter_registration_blocks(
        w, block["number"], reward_epoch
    )

    # get informations for events that build the current signing policy
    signing_policy = await get_signing_policy_events(
        w,
        config,
        reward_epoch,
        lower_block_id,
        end_block_id,
    )
    spb = SigningPolicy.builder().for_epoch(reward_epoch.next)

    block_production = await get_block_production(w)
    maximum_exponent = calculate_maximum_exponent(block_production, config)

    # print("Signing policy created for reward epoch", current_rid)
    # print("Reward Epoch object created", reward_epoch_info)
    # print("Current Reward Epoch status", reward_epoch_info.status(config))

    # set up target address from config
    tia = w.to_checksum_address(config.identity_address)
    # TODO:(matej) log version and initial voting round, maybe signing policy info
    log_message(
        config,
        Message.builder()
        .add(network=config.chain_id)
        .build(
            MessageLevel.INFO,
            f"Initialized observer for identity_address={tia}",
        ),
    )

    cron_time = time.time()

    # wait until next voting epoch
    block_number = block["number"]
    while True:
        latest_block = await w.eth.block_number
        if block_number == latest_block:
            time.sleep(2)
            continue

        block_number += 1
        block_data = await w.eth.get_block(block_number)

        assert "timestamp" in block_data

        _ve = vef.from_timestamp(block_data["timestamp"])
        if _ve == voting_epoch.next:
            voting_epoch = voting_epoch.next
            break

    vrm = VotingRoundManager(voting_epoch.previous.id)

    # set up contracts and events (from config)
    cm = ContractManager(config.contracts)
    contracts = cm.get_contracts_list()
    event_signatures = cm.get_events()

    entity = signing_policy.entity_mapper.by_identity_address[tia]
    fum = FastUpdatesManager(
        block_number, FastUpdate(reward_epoch.id, entity.signing_policy_address, [])
    )
    spm = SigningPolicyManager(signing_policy, signing_policy)
    rm = RewardManager()

    # check transactions for submit transactions
    target_function_signatures = {
        config.contracts.Submission.functions[
            "submitSignatures"
        ].signature: "submitSignatures",
        config.contracts.Submission.functions["submit1"].signature: "submit1",
        config.contracts.Submission.functions["submit2"].signature: "submit2",
    }

    minimal_conditions = (
        MinimalConditions()
        .for_network(config.chain_id)
        .for_reward_epoch(reward_epoch.id)
    )
    last_minimal_conditions_check = int(time.time())
    last_ping = int(time.time())
    last_registration_check = int(time.time())

    uptime_validation_frequency = 60
    node_connections = defaultdict(
        partial(
            deque[bool],
            maxlen=minimal_conditions.time_period.value // uptime_validation_frequency,
        )
    )
    uptime_validations = 0

    medians: deque[list[FtsoMedian]] = deque(
        maxlen=minimal_conditions.time_period.value // 90
    )
    entity_votes: deque[list[int | None]] = deque(
        maxlen=minimal_conditions.time_period.value // 90
    )

    signatures: deque[bool] = deque(maxlen=minimal_conditions.time_period.value // 90)

    voter_registration_started: bool = False
    voter_registration_started_ts: int = 0
    registered: bool = False

    nr_of_feeds: int = 0
    fast_update_re: int = 0

    messages: list[Message] = []
    min_cond_messages: list[Message] = []

    while True:
        latest_block = await w.eth.block_number
        if block_number == latest_block:
            time.sleep(2)
            continue

        for block in range(block_number, latest_block):
            LOGGER.debug(f"processing {block}")
            block_data = await w.eth.get_block(block, full_transactions=True)
            assert "transactions" in block_data
            assert "timestamp" in block_data
            block_ts = block_data["timestamp"]

            voting_epoch = vef.from_timestamp(block_ts)

            if (
                spb.signing_policy_initialized is not None
                and spb.signing_policy_initialized.start_voting_round_id
                == voting_epoch.id
            ):
                # TODO:(matej) this could fail if the observer is started during
                # last two hours of the reward epoch
                voter_registration_started = False
                registered = False
                spm.previous_policy = signing_policy
                signing_policy = spb.build()
                spm.current_policy = signing_policy

                spb = SigningPolicy.builder().for_epoch(
                    signing_policy.reward_epoch.next
                )

                minimal_conditions.reward_epoch_id = signing_policy.reward_epoch.id

                entity = signing_policy.entity_mapper.by_identity_address[tia]
                unclaimed_rewards = await rm.get_unclaimed_rewards(entity, config, w)
                for m in unclaimed_rewards:
                    log_message(config, m)

            block_logs = await w.eth.get_logs(
                {
                    "address": [contract.address for contract in contracts],
                    "fromBlock": block,
                    "toBlock": block,
                }
            )

            tx_messages = []
            event_messages = []
            for log in block_logs:
                sig = log["topics"][0]

                if sig.hex() in event_signatures:
                    event = event_signatures[sig.hex()]
                    data = get_event_data(w.eth.codec, event.abi, log)
                    match event.name:
                        case "ProtocolMessageRelayed":
                            e = ProtocolMessageRelayed.from_dict(
                                data["args"], block_data
                            )
                            voting_round = vrm.get(ve(e.voting_round_id))
                            if e.protocol_id == 100:
                                voting_round.ftso.finalization = e
                            if e.protocol_id == 200:
                                voting_round.fdc.finalization = e

                            # this had to be sent to Relay
                            # so we can check if the Relay address changed
                            entity = signing_policy.entity_mapper.by_identity_address[
                                tia
                            ]
                            tx = await w.eth.get_transaction(data["transactionHash"])
                            assert "to" in tx
                            assert "from" in tx
                            if tx["from"] == entity.identity_address:
                                relay_address = tx["to"]
                                event_messages.extend(
                                    cm.check_relay_address(relay_address)
                                )

                        case "AttestationRequest":
                            e = AttestationRequest.from_dict(data, voting_epoch)
                            vrm.get(e.voting_epoch_id).fdc.requests.agg.append(e)

                        case "SigningPolicyInitialized":
                            e = SigningPolicyInitialized.from_dict(data["args"])
                            spb.add(e)
                        case "VoterRegistered":
                            e = VoterRegistered.from_dict(data["args"])
                            spb.add(e)
                            if registered:
                                continue
                            entity = signing_policy.entity_mapper.by_identity_address[
                                tia
                            ]
                            if (
                                entity.signing_policy_address
                                == e.signing_policy_address
                            ):
                                registered = True
                        case "VoterRemoved":
                            e = VoterRemoved.from_dict(data["args"])
                            spb.add(e)
                        case "VoterRegistrationInfo":
                            e = VoterRegistrationInfo.from_dict(data["args"])
                            spb.add(e)
                        case "VotePowerBlockSelected":
                            e = VotePowerBlockSelected.from_dict(data["args"])
                            spb.add(e)
                            if registered:
                                continue
                            voter_registration_started = True
                            voter_registration_started_ts = int(time.time())
                        case "RandomAcquisitionStarted":
                            e = RandomAcquisitionStarted.from_dict(data["args"])
                            spb.add(e)
                        case "FastUpdateFeedsSubmitted":
                            e = FastUpdateFeedsSubmitted.from_dict(data)
                            tx = await w.eth.get_transaction(e.transaction_hash)
                            spa, address, update_array = calculate_update_from_tx(
                                config, w, tx
                            )
                            entity = signing_policy.entity_mapper.by_identity_address[
                                tia
                            ]
                            if un_prefix_0x(entity.signing_policy_address) == spa:
                                fum.last_update = FastUpdate(
                                    signing_policy.reward_epoch.id,
                                    address,
                                    update_array,
                                )
                                fum.last_update_block = int(data["blockNumber"])
                                # We check update array when we receive a new one
                                event_messages.extend(
                                    fum.check_update_length(nr_of_feeds, fast_update_re)
                                )
                                fum.address_list.add(address)
                        case "FastUpdateFeeds":
                            e = FastUpdateFeeds.from_dict(data)
                            nr_of_feeds, fast_update_re = (
                                len(e.feeds),
                                ref.from_voting_epoch(
                                    vef.make_epoch(e.voting_round_id)
                                ).id,
                            )
                        case "VoterPreRegistered":
                            e = VoterPreRegistered.from_dict(data)
                            entity = signing_policy.entity_mapper.by_identity_address[
                                tia
                            ]
                            if tia == e.voter:
                                registered = True

            for tx in block_data["transactions"]:
                assert not isinstance(tx, bytes)
                wtx = WTxData.from_tx_data(tx, block_data)

                called_function_sig = wtx.input[:4].hex()
                input = wtx.input[4:].hex()
                sender_address = wtx.from_address
                entity = signing_policy.entity_mapper.by_omni.get(sender_address)
                if entity is None:
                    continue
                target_entity = signing_policy.entity_mapper.by_identity_address[tia]

                if called_function_sig in target_function_signatures:
                    # check if the Submission address is correct
                    if entity == target_entity:
                        tx_messages.extend(cm.check_submission_address(wtx.to_address))
                    mode = target_function_signatures[called_function_sig]
                    match mode:
                        case "submit1":
                            try:
                                parsed = parse_submit1_tx(input)
                                if parsed.ftso is not None:
                                    vrm.get(
                                        ve(parsed.ftso.voting_round_id)
                                    ).ftso.insert_submit_1(entity, parsed.ftso, wtx)
                                if parsed.fdc is not None:
                                    vrm.get(
                                        ve(parsed.fdc.voting_round_id)
                                    ).fdc.insert_submit_1(entity, parsed.fdc, wtx)
                            except Exception:
                                pass

                        case "submit2":
                            try:
                                parsed = parse_submit2_tx(input)
                                if parsed.ftso is not None:
                                    vrm.get(
                                        ve(parsed.ftso.voting_round_id)
                                    ).ftso.insert_submit_2(entity, parsed.ftso, wtx)
                                if parsed.fdc is not None:
                                    vrm.get(
                                        ve(parsed.fdc.voting_round_id)
                                    ).fdc.insert_submit_2(entity, parsed.fdc, wtx)
                            except Exception:
                                pass

                        case "submitSignatures":
                            try:
                                parsed = parse_submit_signature_tx(input)
                                if parsed.ftso is not None:
                                    vrm.get(
                                        ve(parsed.ftso.voting_round_id)
                                    ).ftso.insert_submit_signatures(
                                        entity, parsed.ftso, wtx
                                    )
                                if parsed.fdc is not None:
                                    vr = vrm.get(ve(parsed.fdc.voting_round_id))
                                    vr.fdc.insert_submit_signatures(
                                        entity, parsed.fdc, wtx
                                    )

                                    # NOTE:(matej) this is currently the easiest
                                    # way to get consensus bitvote
                                    vr.fdc.consensus_bitvote[
                                        parsed.fdc.payload.unsigned_message
                                    ] += 1

                            except Exception:
                                pass

            messages.clear()
            messages.extend(tx_messages)
            messages.extend(event_messages)
            entity = signing_policy.entity_mapper.by_identity_address[tia]

            # perform all minimal condition checks here
            if int(time.time() - last_minimal_conditions_check) > 60:
                min_cond_messages.clear()

                min_cond_messages.extend(
                    minimal_conditions.calculate_ftso_block_latency_feeds(
                        maximum_exponent,
                        entity,
                        signing_policy,
                        fum.last_update_block,
                        block,
                    )
                )
                min_cond_messages.extend(
                    minimal_conditions.calculate_ftso_anchor_feeds(
                        medians, entity_votes
                    )
                )
                min_cond_messages.extend(
                    minimal_conditions.calculate_staking(
                        uptime_validations, node_connections
                    )
                )
                min_cond_messages.extend(
                    minimal_conditions.calculate_fdc_participation(signatures)
                )

                for m in min_cond_messages:
                    log_message(config, m)

                last_minimal_conditions_check = int(time.time())

            node_ids = [
                node_id_to_representation(node.node_id) for node in entity.nodes
            ]
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "platform.getCurrentValidators",
                "params": {"nodeIDs": node_ids},
            }
            if (
                int(time.time() - last_ping) > uptime_validation_frequency
                and len(node_ids) > 0
            ):
                try:
                    response = requests.post(
                        config.p_chain_rpc_url, json=payload, timeout=10
                    )
                    response.raise_for_status()
                    result = response.json()
                    if "error" in result:
                        LOGGER.warning("Error calling API: check params")
                        continue

                    if (
                        uptime_validations
                        < minimal_conditions.time_period.value
                        // uptime_validation_frequency
                    ):
                        uptime_validations += 1
                    for node in result["result"]["validators"]:
                        node_connections[node["nodeID"]].append(node["connected"])
                except requests.RequestException as e:
                    LOGGER.warning(f"Error calling API: {e}")
                last_ping = int(time.time())

            if int(time.time()) - cron_time > 60 * 60:
                cron_time = int(time.time())
                check_functions = [
                    entity.check_addresses(config, w),
                    fum.check_addresses(config, w),
                ]
                messages.extend(await cron(check_functions))

            rounds = vrm.finalize(block_data)
            # prepare new data for anchor feeds
            if len(rounds) > 0:
                medians.extend([round.ftso.medians for round in rounds])
                for round in rounds:
                    extracted_ftso = extract_round_for_entity(
                        round.ftso, entity, round.voting_epoch
                    ).submit_2.extracted
                    if extracted_ftso is not None:
                        votes = extracted_ftso.parsed_payload.payload.values
                        entity_votes.append(votes)
                    else:
                        entity_votes.append([])
            for r in rounds:
                LOGGER.info(f"processed round {r.voting_epoch.id}")
                messages.extend(validate_round(r, signing_policy, entity, config))

            # prepare new data for FDC participation
            signatures.extend([round.submitted_signatures for round in rounds])

            # reporting on registration and preregistration once per minute
            interval = 60
            if int(time.time() - voter_registration_started_ts) > 15 * 60:
                interval = 10
            if (
                int(time.time() - last_registration_check) > interval
                and voter_registration_started
                and not registered
            ):
                mb = Message.builder().add(network=config.chain_id)
                if int(time.time() - voter_registration_started_ts) > 60:
                    level = MessageLevel.CRITICAL
                    message = mb.build(
                        level,
                        (
                            "Voter not registered after "
                            f"{int(time.time() - voter_registration_started_ts) // 60}"
                            " minutes"
                        ),
                    )
                    messages.append(message)

            for m in messages:
                log_message(config, m)

        block_number = latest_block
