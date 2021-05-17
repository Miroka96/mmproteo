import logging
import os
import sys
from typing import List, NoReturn, Optional, TextIO

LOG_FORMAT: str = '%(asctime)s - %(name)s: %(message)s'
DEFAULT_LOG_SUFFIX: str = '.log'


class LoggedWarningException(Exception):
    def __init__(self, msg: str):
        super().__init__(msg)


class LoggedErrorException(Exception):
    def __init__(self, msg: str):
        super().__init__(msg)


class Logger:
    def __init__(self,
                 logger: Optional[logging.Logger],
                 fail_early: bool = True,
                 terminate_process: bool = False,
                 verbose: Optional[bool] = None):
        assert logger is None or verbose is None, "cannot use 'logger' and 'verbose' parameter simultaneously. " \
                                                  "'verbose' is only considered if 'logger' is None"
        assert logger is not None or verbose is not None, "either 'logger' or 'verbose' must be given to define " \
                                                          "the log level"
        self.logger = logger
        self.fail_early = fail_early
        self.terminate_process = terminate_process
        self._verbose = verbose

    def __print(self, msg: str) -> None:
        print(msg)

    def info(self, msg: str = "") -> None:
        if self.logger is not None:
            self.logger.info(msg)
        else:
            self.__print("INFO: " + msg)

    def debug(self, msg: str = "") -> None:
        if self.logger is not None:
            self.logger.debug("DEBUG: " + msg)
        elif self._verbose:
            self.__print("DEBUG: " + msg)

    def error(self, msg: str = "") -> NoReturn:
        if self.logger is not None:
            self.logger.info("ERROR: " + msg)
        else:
            self.__print("ERROR: " + msg)
        if self.terminate_process:
            sys.exit(1)
        raise LoggedErrorException(msg)

    def warning(self, msg: str = "") -> Optional[NoReturn]:
        if self.logger is not None:
            self.logger.info("WARNING: " + msg)
        else:
            self.__print("WARNING: " + msg)
        if self.fail_early:
            if self.terminate_process:
                self.info("Shutting down because of fail-early configuration")
                sys.exit(1)
            raise LoggedWarningException(msg)
        return None

    def assert_true(self, condition: bool, error_msg: str, as_warning: bool = False) \
            -> Optional[NoReturn]:
        if not condition:
            if as_warning:
                self.warning(error_msg)
            else:
                self.error(error_msg)
        return None

    def is_verbose(self) -> bool:
        if self.logger is not None:
            return self.logger.getEffectiveLevel() == logging.DEBUG
        elif self._verbose is not None:
            return self._verbose
        assert False, "either a logger or the verbose flag must be set"


class DummyLogger(Logger):
    def __init__(self,
                 send_welcome: bool = True,
                 fail_early: bool = False,
                 terminate_process: bool = False,
                 verbose: bool = False):
        super().__init__(logger=None,
                         fail_early=fail_early,
                         terminate_process=terminate_process,
                         verbose=verbose)
        if send_welcome:
            self.info("Printing to Stdout")


class TestLogger(Logger):
    def __init__(self,
                 fail_early: bool = False,
                 terminate_process: bool = False,
                 verbose: bool = False):
        super().__init__(logger=None,
                         fail_early=fail_early,
                         terminate_process=terminate_process,
                         verbose=verbose)
        self.msg_buffer: List[str] = []

    def __print(self, msg: str) -> None:
        self.msg_buffer.append(msg)


def create_logger(name: str,
                  level: Optional[int] = None,
                  verbose: Optional[bool] = None,
                  filename: Optional[str] = None,
                  log_dir: Optional[str] = ".",
                  log_to_std: Optional[TextIO] = sys.stderr,
                  fail_early: bool = False) -> Logger:
    """
    Create a new logger instance.

    :param name:        the name of the new logger
    :param level:       the log level as defined in :mod:`logging`.
                        Cannot be used together with :paramref:`create_logger.verbose`
    :param verbose:     whether to log debug messages. Cannot be used together with :paramref:`create_logger.level`
    :param filename:    the name of the file in the :paramref:`create_logger.log_dir` to which the log is written.
                        If it is None, the log filename is constructed as ':paramref:`create_logger.name`.log'.
    :param log_dir:     the path of the directory in which the log file is created.
                        If it is None, file logging is disabled.
    :param log_to_std:  an IO handler (e.g. :attr:`sys.stderr` or :attr:`sys.stdout`), usually used for command line
                        log output. If it is None, command line logging is disabled.
    :param fail_early:  whether to fail already on logged warnings.
    :return:            a fully configured Logger instance. In case of occurred errors, the logger will be a
                        :class:`DummyLogger`.
    """
    if verbose is not None:
        assert level is None, "level and verbose are exclusive parameters"
        if verbose:
            level = logging.DEBUG
        else:
            level = logging.INFO
    else:
        assert level is not None, "either level or verbose must be given as parameter"

    assert name is not None, "the 'name' parameter is required as logger name"
    assert log_dir is not None or log_to_std is not None, \
        "either 'log_dir' or 'log_to_std' must be given to specify a log target"

    try:
        if filename is None:
            filename = name + DEFAULT_LOG_SUFFIX
        if log_dir is not None:
            filename = os.path.join(log_dir, filename)
        logger: logging.Logger = logging.getLogger(name)
        logger.setLevel(level)

        formatter = logging.Formatter(LOG_FORMAT)

        target_names = list()

        if log_dir is not None:
            # log to file
            dir_name = os.path.dirname(filename)

            from mmproteo.utils.utils import \
                ensure_dir_exists  # prevent circular import
            ensure_dir_exists(dir_name)
            file_logging_handler = logging.FileHandler(filename)
            file_logging_handler.setFormatter(formatter)
            file_logging_handler.setLevel(level)
            logger.addHandler(file_logging_handler)

            target_names.append(f"file '{filename}'")

        if log_to_std is not None:
            std_logging_handler = logging.StreamHandler(log_to_std)
            std_logging_handler.flush = sys.stdout.flush  # type: ignore
            std_logging_handler.setFormatter(formatter)
            std_logging_handler.setLevel(level)
            logger.addHandler(std_logging_handler)

            target_names.append(log_to_std.name)

        logger.info("Logging to " + " and to ".join(
            [target for target in target_names]))

        return Logger(logger, fail_early=fail_early)
    except Exception as e:
        dummy_logger: Logger = DummyLogger(send_welcome=True,
                                           fail_early=fail_early)
        dummy_logger.debug(str(e))
        dummy_logger.warning("Failed to create logger " + str(name))
        dummy_logger.info("Returned print-only dummy logger")

        return dummy_logger


DEFAULT_LOGGER: Logger = DummyLogger(send_welcome=False)
