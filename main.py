import asyncio

import dotenv

from configuration.config import get_config
from configuration.types import Configuration
from observer.message import Message, MessageLevel
from observer.observer import log_message, observer_loop


def main(config: Configuration):
    try:
        asyncio.run(observer_loop(config))
    except Exception as e:
        mb = Message.builder()
        level = MessageLevel.CRITICAL
        message = mb.build(
            level,
            (f"Observer crashed. Reason: {e}"),
        )
        log_message(config, message)


if __name__ == "__main__":
    dotenv.load_dotenv()
    config = get_config()
    main(config)
