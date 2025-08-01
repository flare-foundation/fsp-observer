import json
from typing import Callable, Self

from attrs import field, frozen
from eth_typing import ABI, ABIEvent, ABIFunction, ChecksumAddress
from eth_utils.address import to_checksum_address
from py_flare_common.fsp.epoch.epoch import RewardEpoch, VotingEpoch
from py_flare_common.fsp.epoch.factory import RewardEpochFactory, VotingEpochFactory
from web3 import Web3


def un_prefix_0x(to_unprefixed: str) -> str:
    return to_unprefixed.removeprefix("0x")


def abi_from_file_location(file_location):
    return json.load(open(file_location))["abi"]


def event_signature(event_abi: ABIEvent) -> str:
    params = ""
    for index, input in enumerate(event_abi["inputs"]):
        if index > 0:
            params += ","

        if input["type"] == "tuple[]":
            params += "("
            for index2, tuple_component in enumerate(input["components"]):
                if index2 > 0:
                    params += ","

                params += tuple_component["type"]

            params += ")[]"

        elif input["type"] == "tuple":
            params += "("
            for index2, tuple_component in enumerate(input["components"]):
                if index2 > 0:
                    params += ","

                params += tuple_component["type"]

            params += ")"

        else:
            params += input["type"]

    return un_prefix_0x(Web3.keccak(text=event_abi["name"] + "(" + params + ")").hex())  # type: ignore


def function_signature(function_name: str) -> str:
    return Web3.keccak(text=function_name).hex()[:8]


@frozen
class Event:
    name: str
    abi: ABIEvent
    contract: "Contract"
    signature: str = field(init=False)

    def __str__(self) -> str:
        return f"Event: {self.name}, signature: {self.signature}"

    def __repr__(self) -> str:
        return f"Event: {self.name}, signature: {self.signature}"

    def __attrs_post_init__(self):
        object.__setattr__(self, "signature", event_signature(self.abi))


@frozen
class Function:
    name: str
    abi: ABIFunction
    contract: "Contract"
    signature: str = field(init=False)

    def to_full_name(self):
        assert "inputs" in self.abi
        inputs = [i["type"] for i in self.abi["inputs"]]  # type: ignore
        return f"{self.name}({','.join(inputs)})"

    def __str__(self) -> str:
        return f"Function: {self.to_full_name()}, signature: {self.signature}"

    def __repr__(self) -> str:
        return f"Function: {self.to_full_name()}, signature: {self.signature}"

    def __attrs_post_init__(self):
        object.__setattr__(self, "signature", function_signature(self.to_full_name()))


@frozen
class Contract:
    name: str
    address: ChecksumAddress
    abi: ABI = field(converter=abi_from_file_location)
    events: dict[str, Event] = field(init=False)
    functions: dict[str, Function] = field(init=False)

    def __str__(self) -> str:
        target_str = (
            f"Contract: {self.name}, addr.: {self.address}, "
            f"events: {self.events}, functions: {self.functions}"
        )
        return target_str

    def __repr__(self) -> str:
        target_str = (
            f"Contract: {self.name}, addr.: {self.address}, "
            f"events: {self.events}, functions: {self.functions}"
        )
        return target_str

    def __attrs_post_init__(self):
        events = {}
        functions = {}
        for entry in self.abi:
            assert "type" in entry
            if entry["type"] == "event":
                assert "name" in entry
                events[entry["name"]] = Event(entry["name"], entry, self)
            elif entry["type"] == "function":
                assert "name" in entry
                functions[entry["name"]] = Function(entry["name"], entry, self)
        object.__setattr__(self, "events", events)
        object.__setattr__(self, "functions", functions)


# NOTE:(matej) FlareContractRegistry smart contract always provides an up to date
# mapper ({name:address}) for all official Flare contracts. It is deployed on all 4
# chains on the SAME address and is guaranteed to never be redeployed. This is why we
# can hardcode it here
FLARE_CONTRACT_REGISTRY_ADDRESS = to_checksum_address(
    "0xaD67FE66660Fb8dFE9d6b1b4240d8650e30F6019"
)
FLARE_CONTRACT_REGISTRY_ABI = json.load(
    open("configuration/artifacts/FlareContractRegistry.json")
)


@frozen
class Contracts:
    VoterRegistry: Contract
    FlareSystemsCalculator: Contract
    FlareSystemsManager: Contract
    Relay: Contract
    Submission: Contract
    FdcHub: Contract
    FastUpdater: Contract

    @classmethod
    def get_contracts(cls, w: Web3) -> Self:
        attr_names = [a.name for a in cls.__attrs_attrs__]  # type: ignore

        registry = w.eth.contract(
            address=FLARE_CONTRACT_REGISTRY_ADDRESS,
            abi=FLARE_CONTRACT_REGISTRY_ABI,
        )

        kwargs = {}
        for name in attr_names:
            address = to_checksum_address(
                registry.functions.getContractAddressByName(name).call()
            )

            kwargs[name] = Contract(
                name,
                address,
                f"configuration/artifacts/{name}.json",
            )

        return cls(**kwargs)


@frozen
class Epoch:
    voting_epoch: Callable[[int], VotingEpoch]
    reward_epoch: Callable[[int], RewardEpoch]
    voting_epoch_factory: VotingEpochFactory
    reward_epoch_factory: RewardEpochFactory


@frozen
class NotificationDiscord:
    webhook_url: list[str]


@frozen
class NotificationSlack:
    webhook_url: list[str]


@frozen
class TelegramBot:
    bot_token: str
    chat_id: str


@frozen
class NotificationTelegram:
    bot: list[TelegramBot]


@frozen
class NotificationGeneric:
    webhook_url: list[str]


@frozen
class Notification:
    discord: NotificationDiscord
    slack: NotificationSlack
    telegram: NotificationTelegram
    generic: NotificationGeneric


@frozen
class Configuration:
    identity_address: ChecksumAddress
    chain_id: int
    contracts: Contracts
    rpc_url: str
    p_chain_rpc_url: str
    epoch: Epoch
    notification: Notification
    fee_threshold: int
