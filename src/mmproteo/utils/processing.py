import gc
import multiprocessing
import signal
from typing import Callable, Any, Optional, Sequence, Tuple, Iterable, List, NoReturn, Union

from mmproteo.utils import log, utils
from mmproteo.utils.config import Config


class _IndexedItemProcessor:
    def __init__(self, item_processor: Callable[[Any], Union[Optional[Any], NoReturn]]):
        self.item_processor = item_processor

    def __call__(self, indexed_item: Tuple[int, Optional[Any]]) -> Tuple[int, Optional[Any]]:
        index, item = indexed_item
        if item is None:
            response = None
        else:
            try:
                response = self.item_processor(item)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                response = e
        gc.collect()
        return index, response


class ItemProcessor:
    def __init__(self,
                 items: Iterable[Optional[Any]],
                 item_processor: Callable[[Any], Union[Optional[Any], NoReturn]],
                 action_name: str,
                 action_name_past_form: Optional[str] = None,
                 subject_name: str = "file",
                 max_num_items: Optional[int] = None,
                 keep_null_values: bool = Config.default_keep_null_values,
                 count_null_results: bool = Config.default_count_null_results,
                 count_failed_items: bool = Config.default_count_failed_files,
                 thread_count: int = Config.default_thread_count,
                 logger: log.Logger = log.DEFAULT_LOGGER):

        if max_num_items == 0:
            max_num_items = None

        self.items: Sequence[Any] = list(items)
        del items
        self.indexed_item_processor = _IndexedItemProcessor(item_processor)
        self.action_name = action_name
        self.subject_name = subject_name
        self.keep_null_values = keep_null_values
        self.count_null_results = count_null_results
        self.count_failed_items = count_failed_items
        self.max_num_items = max_num_items
        self.logger = logger

        if thread_count == 0:
            thread_count = multiprocessing.cpu_count()
        if thread_count > len(self.items):
            thread_count = len(self.items)
        if max_num_items is not None and thread_count > max_num_items:
            thread_count = max_num_items
        if thread_count > 1:
            logger.debug(f"Processing items with {thread_count} subprocesses")
            original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
            self.process_pool: Optional[multiprocessing.pool.Pool] = multiprocessing.Pool(processes=thread_count)
            signal.signal(signal.SIGINT, original_sigint_handler)
        else:
            self.process_pool = None

        if action_name_past_form is None:
            if action_name.endswith("e"):
                action_name_past_form = action_name + "d"
            else:
                action_name_past_form = action_name + "ed"
        self.action_name_past_form = action_name_past_form
        self.processing_results: List[Optional[Any]] = list()

    def __drop_null_items(self) -> None:
        non_null_items: Sequence[Any] = [item for item in self.items if item is not None]
        self.items_to_process_count = len(non_null_items)

        if not self.keep_null_values:
            self.items = non_null_items

    def __limit_number_of_items_to_process(self) -> None:
        if self.max_num_items is not None:
            self.items_to_process_count = min(self.items_to_process_count, self.max_num_items)
            if self.items_to_process_count < self.max_num_items:
                # limit doesn't get triggered
                self.max_num_items = None

    def __process_indexed_item_batch_in_parallel(self, indexed_item_batch: Iterable[Tuple[int, Optional[Any]]]) \
            -> Iterable[Tuple[int, Optional[Any]]]:
        assert self.process_pool is not None, \
            "specify a thread count to create a process pool to use this function"
        try:
            indexed_results: Iterable[Tuple[int, Optional[Any]]] = self.process_pool.imap_unordered(
                self.indexed_item_processor,
                indexed_item_batch)
        except KeyboardInterrupt:
            self.logger.info("Terminating workers")
            self.process_pool.terminate()
            self.process_pool.join()
            raise
        else:
            self.process_pool.close()
            self.process_pool.join()
            return indexed_results

    def __process_indexed_item_batch(self, indexed_item_batch: Iterable[Tuple[int, Optional[Any]]]) -> None:
        if self.process_pool is None:
            indexed_results: Iterable[Tuple[int, Optional[Any]]] \
                = [self.indexed_item_processor(indexed_item) for indexed_item in indexed_item_batch]
        else:
            indexed_results = self.__process_indexed_item_batch_in_parallel(indexed_item_batch)
        results: List[Optional[Any]] = [indexed_item[1] for indexed_item in sorted(indexed_results)]
        self.processing_results += results

    def __get_processing_results(self,
                                 keep_null_items: bool = False,
                                 keep_exceptions_as_nulls: bool = False) -> List[Any]:
        items = self.processing_results
        if not keep_null_items:
            items = [item for item in items if item is not None]
        if keep_exceptions_as_nulls:
            items = [None if isinstance(item, Exception) else item for item in items]
        else:
            items = [item for item in items if not isinstance(item, Exception)]
        return items

    def get_exceptions(self) -> List[Exception]:
        exceptions = [item for item in self.processing_results if isinstance(item, Exception)]
        return exceptions

    def count_successfully_processed_items(self) -> int:
        return len(self.__get_processing_results(keep_null_items=self.count_null_results,
                                                 keep_exceptions_as_nulls=self.count_failed_items))

    def __process_items(self) -> None:
        indexed_items = list(enumerate(self.items))
        if self.max_num_items is None:
            self.__process_indexed_item_batch(indexed_items)
        else:
            while self.items_to_process_count > 0:
                processed_items_count = len(self.processing_results)
                current_item_batch = indexed_items[processed_items_count:
                                                   processed_items_count + self.items_to_process_count]
                if len(current_item_batch) == 0:
                    self.items_to_process_count = 0
                    break

                self.__process_indexed_item_batch(current_item_batch)

                self.items_to_process_count = self.max_num_items - self.count_successfully_processed_items()

    def __evaluate_results_textually(self) -> None:
        successfully_processed_items_count = self.count_successfully_processed_items()
        if successfully_processed_items_count > 0:
            self.logger.info(
                f"Successfully {self.action_name_past_form} {successfully_processed_items_count} {self.subject_name}"
                f"{utils.get_plural_s(successfully_processed_items_count)}")
        else:
            self.logger.info(f"No {self.subject_name}s were {self.action_name_past_form}")
        exceptions = self.get_exceptions()
        self.logger.info(
            f"Encountered {len(exceptions)} exception{utils.get_plural_s(len(exceptions))} during processing")
        for exception in exceptions:
            self.logger.debug(str(exception))

    def process(self) -> Iterable[Optional[Any]]:
        self.__drop_null_items()
        if self.items_to_process_count == 0:
            self.logger.warning(f"No {self.subject_name}s available to {self.action_name}")
            return [None for _ in self.items]  # type: ignore

        self.__limit_number_of_items_to_process()
        self.logger.debug(f"Trying to {self.action_name} {self.items_to_process_count} {self.subject_name}"
                          f"{utils.get_plural_s(self.items_to_process_count)}")

        self.__process_items()
        self.__evaluate_results_textually()

        return self.__get_processing_results(keep_null_items=self.keep_null_values,
                                             keep_exceptions_as_nulls=self.keep_null_values)
