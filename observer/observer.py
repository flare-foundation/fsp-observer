import logging
import time
from typing import Self

from eth_account._utils.signing import to_standard_v
from eth_keys.datatypes import Signature as EthSignature
from py_flare_common.fsp.epoch.epoch import RewardEpoch
from py_flare_common.fsp.messaging import (
    parse_submit1_tx,
    parse_submit2_tx,
    parse_submit_signature_tx,
)
from py_flare_common.fsp.messaging.types import Signature as SSignature
from web3 import AsyncWeb3
from web3._utils.events import get_event_data
from web3.middleware import ExtraDataToPOAMiddleware

from configuration.types import (
    Configuration,
)
from observer.reward_epoch_manager import (
    SigningPolicy,
)
from observer.types import (
    AttestationRequest,
    ProtocolMessageRelayed,
    RandomAcquisitionStarted,
    SigningPolicyInitialized,
    VotePowerBlockSelected,
    VoterRegistered,
    VoterRegistrationInfo,
    VoterRemoved,
)
from observer.validation.validation import validate_round

from .message import Message, MessageLevel
from .notification import notify_discord, notify_generic, notify_slack, notify_telegram
from .voting_round import (
    VotingRoundManager,
    WTxData,
)

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s\t%(levelname)s\t%(name)s\t%(message)s",
    level="INFO",
)


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

    if n.discord is not None:
        notify_discord(n.discord, message.level.name + " " + message.message)

    if n.slack is not None:
        notify_slack(n.slack, message.level.name + " " + message.message)

    if n.telegram is not None:
        notify_telegram(n.telegram, message.level.name + " " + message.message)

    if n.generic is not None:
        notify_generic(n.generic, message)


async def observer_loop(config: Configuration) -> None:
    w = AsyncWeb3(
        AsyncWeb3.AsyncHTTPProvider(config.rpc_url),
        middleware=[ExtraDataToPOAMiddleware],
    )

    # log_issue(
    #     config,
    #     Issue(
    #         IssueLevel.INFO,
    #         MessageBuilder()
    #         .add_network(config.chain_id)
    #         .add_protocol(100)
    #         .add_round(VotingEpoch(12, None))
    #         .build_with_message("testing message" + str(config.notification)),
    #     ),
    # )
    # return

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
    spb = SigningPolicy.builder()

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
    # target_voter = signing_policy.entity_mapper.by_identity_address[tia]
    # notify_discord(
    #     config,
    #     f"flare-observer initialized\n\n"
    #     f"chain: {config.chain}\n"
    #     f"submit address: {target_voter.submit_address}\n"
    #     f"submit signatures address: {target_voter.submit_signatures_address}\n",
    #     # f"this address has voting power of: {signing_policy.voter_weight(tia)}\n\n"
    #     # f"starting in voting round: {voting_round.next.id} "
    #     # f"(current: {voting_round.id})\n"
    #     # f"current reward epoch: {current_rid}",
    # )

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
    # TODO: (nejc) set this up with a function on class
    # or contracts = attrs.asdict(config.contracts) <- this doesn't work
    contracts = [
        config.contracts.Relay,
        config.contracts.VoterRegistry,
        config.contracts.FlareSystemsManager,
        config.contracts.FlareSystemsCalculator,
        config.contracts.FdcHub,
    ]
    event_signatures = {e.signature: e for c in contracts for e in c.events.values()}

    # start listener
    # print("Listener started from block number", block_number)
    # check transactions for submit transactions
    target_function_signatures = {
        config.contracts.Submission.functions[
            "submitSignatures"
        ].signature: "submitSignatures",
        config.contracts.Submission.functions["submit1"].signature: "submit1",
        config.contracts.Submission.functions["submit2"].signature: "submit2",
    }

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
                and spb.signing_policy_initialized.start_voting_round_id == voting_epoch
            ):
                # TODO:(matej) this could fail if the observer is started during
                # last two hours of the reward epoch
                signing_policy = spb.build()
                spb = SigningPolicy.builder().for_epoch(
                    signing_policy.reward_epoch.next
                )

            block_logs = await w.eth.get_logs(
                {
                    "address": [contract.address for contract in contracts],
                    "fromBlock": block,
                    "toBlock": block,
                }
            )

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

                        case "AttestationRequest":
                            e = AttestationRequest.from_dict(data, voting_epoch)
                            vrm.get(e.voting_epoch_id).fdc.requests.agg.append(e)

                        case "SigningPolicyInitialized":
                            e = SigningPolicyInitialized.from_dict(data["args"])
                            spb.add(e)
                        case "VoterRegistered":
                            e = VoterRegistered.from_dict(data["args"])
                            spb.add(e)
                        case "VoterRemoved":
                            e = VoterRemoved.from_dict(data["args"])
                            spb.add(e)
                        case "VoterRegistrationInfo":
                            e = VoterRegistrationInfo.from_dict(data["args"])
                            spb.add(e)
                        case "VotePowerBlockSelected":
                            e = VotePowerBlockSelected.from_dict(data["args"])
                            spb.add(e)
                        case "RandomAcquisitionStarted":
                            e = RandomAcquisitionStarted.from_dict(data["args"])
                            spb.add(e)

            for tx in block_data["transactions"]:
                assert not isinstance(tx, bytes)
                wtx = WTxData.from_tx_data(tx, block_data)

                called_function_sig = wtx.input[:4].hex()
                input = wtx.input[4:].hex()
                sender_address = wtx.from_address
                entity = signing_policy.entity_mapper.by_omni.get(sender_address)
                if entity is None:
                    continue

                if called_function_sig in target_function_signatures:
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

                                    # NOTE:(matej) this is currently the easies way to
                                    # get consensus bitvote
                                    vr.fdc.consensus_bitvote[
                                        parsed.fdc.payload.unsigned_message
                                    ] += 1

                            except Exception:
                                pass

            rounds = vrm.finalize(block_data)
            for r in rounds:
                entity = signing_policy.entity_mapper.by_identity_address[tia]
                for message in validate_round(r, entity, config):
                    log_message(config, message)

        block_number = latest_block
