import os

from eth_utils.address import to_checksum_address
from py_flare_common.fsp.epoch.timing import coston, coston2, flare, songbird
from web3 import Web3

from .types import (
    Configuration,
    Contracts,
    Epoch,
    MetricsConfig,
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


class ProtocolId(int):
    pass


class Protocol:
    FTSO = ProtocolId(100)
    FDC = ProtocolId(200)
    # NOTE:(janezicmatej) FSP only uses 1 byte for protocol id so we can use numbers
    # larger than 256 without worry
    FAST_UPDATES = ProtocolId(300)
    STAKING = ProtocolId(400)

    @classmethod
    def id_to_name(cls, protocol: ProtocolId):
        match protocol:
            case cls.FTSO:
                return "ftso"
            case cls.FDC:
                return "fdc"
            case cls.FAST_UPDATES:
                return "fast updates"
            case cls.STAKING:
                return "staking"
            case _:
                raise ValueError(f"Unknown protocol ({protocol=})")


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

    discord_embed_webhook = os.environ.get("NOTIFICATION_DISCORD_EMBED_WEBHOOK")
    discord_embed = []
    if discord_embed_webhook is not None:
        discord_embed.extend(discord_embed_webhook.split(","))

    slack_webhook = os.environ.get("NOTIFICATION_SLACK_WEBHOOK")
    slack = []
    if slack_webhook is not None:
        slack.extend(slack_webhook.split(","))

    slack_embed_webhook = os.environ.get("NOTIFICATION_SLACK_EMBED_WEBHOOK")
    slack_embed = []
    if slack_embed_webhook is not None:
        slack_embed.extend(slack_embed_webhook.split(","))

    telegram_bot_token = os.environ.get("NOTIFICATION_TELEGRAM_BOT_TOKEN")
    telegram_chat_id = os.environ.get("NOTIFICATION_TELEGRAM_CHAT_ID")
    telegram = []
    if telegram_bot_token is not None and telegram_chat_id is not None:
        bot_tokens = telegram_bot_token.split(",")
        chat_ids = telegram_chat_id.split(",")
        telegram = [
            TelegramBot(t, c) for t, c in zip(bot_tokens, chat_ids, strict=False)
        ]

    generic_webhook = os.environ.get("NOTIFICATION_GENERIC_WEBHOOK")
    generic = []
    if generic_webhook is not None:
        generic.extend(generic_webhook.split(","))

    return Notification(
        discord=NotificationDiscord(discord),
        discord_embed=NotificationDiscord(discord_embed),
        slack=NotificationSlack(slack),
        slack_embed=NotificationSlack(slack_embed),
        telegram=NotificationTelegram(telegram),
        generic=NotificationGeneric(generic),
    )


def get_metrics_config() -> MetricsConfig:
    enabled = os.environ.get("METRICS_ENABLED", "false").lower() == "true"
    port = int(os.environ.get("METRICS_PORT", "8000"))
    address = os.environ.get("METRICS_ADDRESS", "0.0.0.0")
    return MetricsConfig(enabled=enabled, port=port, address=address)


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

    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()

    # some rpc providers cap the number of blocks per get_logs request, so we chunk
    # ranged log queries to stay within this limit
    max_block_range = int(os.environ.get("MAX_BLOCK_RANGE", "1000"))

    # optionally suppress fast update miss notifications whose false positive
    # probability (in %) is above this; unset = no suppression (mainly affects
    # low weight entities, whose probability stays high longer)
    _false_positive_threshold = os.environ.get("FALSE_POSITIVE_THRESHOLD")
    false_positive_threshold = (
        float(_false_positive_threshold)
        if _false_positive_threshold is not None
        else None
    )

    # optionally suppress the ftso "missing feed" warning (entity submitted no value for
    # some feeds); useful on networks with feeds that no provider serves
    suppress_ftso_missing_feed = (
        os.environ.get("SUPPRESS_FTSO_MISSING_FEED", "false").lower() == "true"
    )

    config = Configuration(
        rpc_url=rpc_url,
        p_chain_rpc_url=p_chain_rpc_url,
        identity_address=to_checksum_address(identity_address),
        chain_id=chain_id,
        contracts=Contracts.get_contracts(w),
        epoch=get_epoch(chain_id),
        notification=get_notification_config(),
        fee_threshold=fee_threshold,
        metrics=get_metrics_config(),
        log_level=log_level,
        max_block_range=max_block_range,
        false_positive_threshold=false_positive_threshold,
        suppress_ftso_missing_feed=suppress_ftso_missing_feed,
    )

    return config
