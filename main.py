import asyncio
import logging
import traceback

import dotenv

from configuration.config import get_config
from configuration.types import Configuration
from observer.message import Message, MessageLevel
from observer.observer import log_message, observer_loop

LOGGER = logging.getLogger()


def main(config: Configuration):
    try:
        asyncio.run(observer_loop(config))
    except Exception as e:
        mb = Message.builder().add(network=config.chain_id)
        message = mb.build(
            MessageLevel.CRITICAL,
            (f"observer crashed (traceback in logs) - {e}"),
        )
        log_message(config, message)
        LOGGER.exception(e)
        LOGGER.error(traceback.format_exc())


if __name__ == "__main__":
    dotenv.load_dotenv()
    config = get_config()
    main(config)
