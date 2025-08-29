import asyncio

import dotenv

from configuration.config import ConfigError, get_config, get_notification_config
from configuration.types import Configuration
from observer.message import Message, MessageLevel
from observer.notification import log_message
from observer.observer import observer_loop


def main(config: Configuration):
    asyncio.run(observer_loop(config))


if __name__ == "__main__":
    dotenv.load_dotenv()
    try:
        config = get_config()
        main(config)
    except Exception as ex:
        log_message(
            get_notification_config(),
            Message.builder()
            .build(
                MessageLevel.CRITICAL,
                repr(ex),
            ),
        )
    