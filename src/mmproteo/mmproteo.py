#!/usr/bin/python3
import os
import sys
from typing import Optional, NoReturn

from mmproteo.utils import commands, log
from mmproteo.utils.config import Config

APPLICATION_NAME: str = "MMProteo"
_HERE: str = "."


def create_logger(config: Config) -> log.Logger:
    if config.dummy_logger:
        return log.DummyLogger(send_welcome=True,
                               fail_early=config.fail_early,
                               terminate_process=config.terminate_process,
                               verbose=config.verbose)

    if config.log_to_stdout:
        log_to_std = sys.stdout
    else:
        log_to_std = sys.stderr

    if config.log_file is not None and len(config.log_file) == 0:
        log_dir = None  # this disables file logging
    else:
        log_dir = config.storage_dir

    logger = log.create_logger(name=APPLICATION_NAME,
                               log_dir=log_dir,
                               filename=config.log_file,
                               verbose=config.verbose,
                               log_to_std=log_to_std,
                               fail_early=config.fail_early)
    return logger


def main(config: Config = None, logger: Optional[log.Logger] = None) \
        -> Optional[NoReturn]:
    if config is None:
        config = Config()
        if logger is not None:
            config.set_logger(logger)
        config.parse_arguments()

    try:
        config.validate_arguments()

        if logger is not None:
            config.set_logger(logger)
            config.check()
        else:
            config.check()
            logger = create_logger(config)
            config.set_logger(logger)

        if config.storage_dir is not None:
            os.chdir(config.storage_dir)

        config.storage_dir = _HERE

        commands.DISPATCHER.dispatch_commands(config=config, logger=logger)
    except log.LoggedErrorException:
        pass
    except KeyboardInterrupt:
        msg = "Received KeyboardInterrupt - Shutting down"
        if logger is not None:
            logger.info(msg)
        else:
            print(msg)
    return None


if __name__ == '__main__':
    main()
