import os

from eth_utils.address import to_checksum_address
from py_flare_common.fsp.epoch.timing import coston, coston2, flare, songbird
from web3 import Web3

from .types import (
    Configuration,
    Contracts,
    Epoch,
    Notification,
    NotificationDiscord,
    NotificationGeneric,
    NotificationSlack,
    NotificationTelegram,
    TelegramBot,
)


class ChainId:
    COSTON = 16
    SONGBIRD = 19
    COSTON2 = 114
    FLARE = 14

    @classmethod
    def id_to_name(cls, chain_id):
        match chain_id:
            case cls.COSTON:
                return "coston"
            case cls.SONGBIRD:
                return "songbird"
            case cls.COSTON2:
                return "coston2"
            case cls.FLARE:
                return "flare"
            case _:
                raise ValueError(f"Unknown chain ({chain_id=})")

    @classmethod
    def all(cls):
        return [cls.COSTON, cls.SONGBIRD, cls.COSTON2, cls.FLARE]
    
class Protocols:
    FTSO = 100
    FDC = 200
    # NOTE:(lightftso) is this ok? Fast updates doesnt have a protocol id
    FAST_UPDATES = "fu"
    STAKING = "staking"

    @classmethod
    def id_to_name(cls, protocol):
        match protocol:
            case cls.FTSO:
                return "ftso"
            case cls.FDC:
                return "fdc"
            case "fu":
                return "fast updates"
            case "staking":
                return "staking"
            case _:
                raise ValueError(f"Unknown protocol ({protocol=})")

    @classmethod
    def all(cls):
        return [cls.FTSO, cls.FDC]


class ConfigError(Exception):
    pass


def get_epoch(chain_id: int) -> Epoch:
    match chain_id:
        case ChainId.COSTON:
            module = coston
        case ChainId.SONGBIRD:
            module = songbird
        case ChainId.COSTON2:
            module = coston2
        case ChainId.FLARE:
            module = flare
        case _:
            raise ValueError(f"Unknown chain ({chain_id=})")

    return Epoch(
        voting_epoch=module.voting_epoch,
        voting_epoch_factory=module.voting_epoch_factory,
        reward_epoch=module.reward_epoch,
        reward_epoch_factory=module.reward_epoch_factory,
    )


def get_notification_config() -> Notification:
    discord_webhook = os.environ.get("NOTIFICATION_DISCORD_WEBHOOK")
    discord = []
    if discord_webhook is not None:
        discord.extend(discord_webhook.split(","))

    slack_webhook = os.environ.get("NOTIFICATION_SLACK_WEBHOOK")
    slack = []
    if slack_webhook is not None:
        slack.extend(slack_webhook.split(","))

    telegram_bot_token = os.environ.get("NOTIFICATION_TELEGRAM_BOT_TOKEN")
    telegram_chat_id = os.environ.get("NOTIFICATION_TELEGRAM_CHAT_ID")
    telegram = []
    if telegram_bot_token is not None and telegram_chat_id is not None:
        bot_tokens = telegram_bot_token.split(",")
        chat_ids = telegram_chat_id.split(",")
        telegram = [TelegramBot(t, c) for t, c in zip(bot_tokens, chat_ids)]

    generic_webhook = os.environ.get("NOTIFICATION_GENERIC_WEBHOOK")
    generic = []
    if generic_webhook is not None:
        generic.extend(generic_webhook.split(","))

    return Notification(
        discord=NotificationDiscord(discord),
        slack=NotificationSlack(slack),
        telegram=NotificationTelegram(telegram),
        generic=NotificationGeneric(generic),
    )


def get_config() -> Configuration:
    rpc_base_url = os.environ.get("RPC_BASE_URL")
    if rpc_base_url is None:
        raise ConfigError("RPC_BASE_URL environment variable must be set.")

    rpc_url = rpc_base_url + "/ext/bc/C/rpc"
    p_chain_rpc_url = rpc_base_url + "/ext/bc/P"

    w = Web3(Web3.HTTPProvider(rpc_url))
    if not w.is_connected():
        raise ConfigError(f"Unable to connect to rpc with provided {rpc_url=}")

    chain_id = w.eth.chain_id
    if chain_id not in ChainId.all():
        raise ConfigError(f"Detected unknown chain ({chain_id=})")

    identity_address = os.environ.get("IDENTITY_ADDRESS")
    if identity_address is None:
        raise ConfigError("IDENTITY_ADDRESS environment variable must be set.")

    _fee_threshold = os.environ.get("FEE_THRESHOLD", "25")
    fee_threshold = int(_fee_threshold)

    config = Configuration(
        rpc_url=rpc_url,
        p_chain_rpc_url=p_chain_rpc_url,
        identity_address=to_checksum_address(identity_address),
        chain_id=chain_id,
        contracts=Contracts.get_contracts(w),
        epoch=get_epoch(chain_id),
        notification=get_notification_config(),
        fee_threshold=fee_threshold,
    )

    return config
