import os
import subprocess
import time

try:
    from subprocess import DEVNULL  # Python 3.
except ImportError:
    DEVNULL = open(os.devnull, 'wb')
from typing import Any, Hashable, Iterable, List, Optional, Union, Dict, NoReturn, Callable

import numpy as np
import pandas as pd

from mmproteo.utils import log


def concat_set_of_options(options: Iterable[str], option_quote: str = '"', separator: str = ", ") -> str:
    if type(options) != list:
        options = sorted(options)
    return separator.join([option_quote + option + option_quote for option in options])


def deduplicate_list(lst: List[Hashable]) -> List[Hashable]:
    already_inserted = set()
    deduplicated_list = []
    for e in lst:
        if e not in already_inserted:
            already_inserted.add(e)
        deduplicated_list.append(e)
    return deduplicated_list


def _denumpyfy(element: Any) -> Any:
    if type(element) == np.int64:
        return int(element)
    if type(element) == np.float64:
        return float(element)
    if type(element) in [int, str, float]:
        return element
    if type(element) == dict:
        return {k: _denumpyfy(v) for k, v in element.items()}
    if type(element) == list:
        return [_denumpyfy(v) for v in element]
    if type(element) == set:
        return {_denumpyfy(v) for v in element}
    raise NotImplementedError(type(element))


def denumpyfy(element: Union[np.int64, np.float64, int, str, float, dict, list, set]) \
        -> Union[int, str, float, dict, list, set]:
    return _denumpyfy(element)


def ensure_dir_exists(directory: str, logger: log.Logger = log.DEFAULT_LOGGER) -> None:
    """Ensure the existence of a given directory (path) by creating it/them if they do not exist yet."""
    if len(directory) == 0:
        return
    try:
        os.makedirs(directory, exist_ok=True)
    except FileExistsError:
        logger.warning("'%s' already exists and is not a directory")


def flatten_single_element_containers(elem: Union[Iterable, Any]) -> Union[Iterable, Any]:
    try:
        # skip one-element lists or sets,...
        while type(elem) == list or type(elem) == set or type(elem) == tuple:
            if len(elem) == 1:
                elem = next(iter(elem))
                continue
            non_null_elements = [e for e in elem if e is not None]
            if len(non_null_elements) == 0:
                elem = None
                break
            if len(non_null_elements) == 1:
                elem = non_null_elements[0]
                continue
            break
    except Exception:
        pass
    return elem


def flatten_dict(input_dict: dict,
                 result_dict: dict = None,
                 overwrite: bool = False,
                 clean_keys: bool = True,
                 space_filler: str = "_",
                 concat_keys: bool = True,
                 concatenator: str = "__") -> Dict[Union[str, Any], Any]:
    """
    Transforms a hierarchical dictionary structure into a flat dictionary by recursively applying
    :func:`flatten_single_element_containers` on dictionary values until there are no more child dictionaries.

    :param input_dict:   the python dictionary that should be flattened
    :param result_dict:  the output dictionary to which the flattened keys will be added
    :param overwrite:    whether keys from inner dictionaries overwrite keys from outer dictionaries
    :param clean_keys:   If true, all dictionary keys will be converted to strings. Spaces in keys will be replaced by
                         the :paramref:`space_filler`. All non-alphanumeric characters that are not part of the
                         :paramref:`space_filler` or :paramref:`concatenator` will be removed.
    :param space_filler: a string that will be used to replace spaces in keys
    :param concat_keys:  Keys of dictionaries in dictionaries will be concatenated with their parent keys using the
                         :paramref:`concatenator` as separator. Side effect is that all keys become strings.
    :param concatenator: a string that will be used to separate concatenated keys
    :return:             a dictionary without dictionaries as values. It contains the flattened keys and their values.
                         If :paramref:`result_dict` was given, the result object is the given dictionary.
    """
    if result_dict is None:
        result_dict = dict()

    dict_queue = [("", input_dict)]

    while len(dict_queue) > 0:
        key_prefix, item = dict_queue[0]
        dict_queue = dict_queue[1:]

        for key, value in item.items():
            if clean_keys:
                key = str(key)
                key = key.replace(" ", space_filler)
                key = "".join([c for c in key if c.isalnum() or c in space_filler or c in concatenator])
            value = flatten_single_element_containers(value)

            if type(value) == dict:
                new_prefix = str(key) + concatenator
                if concat_keys:
                    new_prefix = key_prefix + new_prefix
                dict_queue.append((new_prefix, value))
                continue

            if concat_keys:
                new_key = key_prefix + str(key)
            else:
                new_key = key

            if not overwrite and new_key not in result_dict:
                result_dict[new_key] = value

    return result_dict


def get_plural_s(count: int) -> str:
    if count != 1:
        return "s"
    else:
        return ""


def list_of_dicts_to_dict(items: List[Dict], dict_key: str) -> Optional[Dict]:
    """
    Transform :paramref:`items` into one common dictionary by using the common key :paramref:`dict_key` as identifier.

    :param items:    a list of dictionaries
    :param dict_key: a key that is present in all :paramref:`items`

    :return:         a dictionary that contains all occurring values of :paramref:`dict_key` as keys and the originating
                     dictionaries as values. If :paramref:`dict_key` does not occur in all given :paramref:`items` this
                     method returns None
    """
    all_have_key = True

    for item in items:
        if dict_key not in item.keys():
            all_have_key = False
            break

    if all_have_key:
        items_dict = dict()
        for item in items:
            key = item[dict_key]
            items_dict[key] = item
        return items_dict

    return None


def format_command_template(command_template: str, formatter: Callable[[str], str]) -> List[str]:
    parts = command_template.split(" ")
    parts = [formatter(part) for part in parts]
    return parts


def stop_docker_container(container_name: str,
                          docker_stop_container_command_template: str = "docker stop {container_name}",
                          logger: log.Logger = log.DEFAULT_LOGGER) -> str:
    stop_command = format_command_template(docker_stop_container_command_template,
                                           lambda s: s.format(container_name=container_name))
    subprocess.run(stop_command)
    stop_command_str = " ".join(stop_command)

    while True:
        status = get_docker_container_status(container_name=container_name)
        if status is None:
            break
        logger.info(f"Waiting for container '{container_name}' to leave status '{status}'.")
        time.sleep(1)
    return stop_command_str


def get_docker_container_status(container_name: str,
                                docker_inspect_container_command_template: str =
                                "docker container inspect -f {{{{.State.Status}}}} {container_name}") -> Optional[str]:
    check_command = format_command_template(docker_inspect_container_command_template,
                                            lambda s: s.format(container_name=container_name))
    process_result = subprocess.run(check_command, stdout=subprocess.PIPE, stderr=DEVNULL)
    if process_result.returncode != 0:
        return None
    return process_result.stdout.decode("utf-8")[:-1]


def is_docker_container_running(container_name: str) -> bool:
    return get_docker_container_status(container_name=container_name) == "running"


def merge_column_values(df: Optional[pd.DataFrame], columns: Iterable[str]) -> List[str]:
    if df is None:
        return list()

    values = set()

    for column in columns:
        if column in df.columns:
            values.update(df[column].dropna())

    if None in values:
        values.remove(None)

    return list(values)


def list_files_in_directory(directory_path: str) -> List[str]:
    paths_in_dir = [os.path.join(directory_path, element) for element in os.listdir(directory_path)]
    file_paths = [path for path in paths_in_dir if os.path.isfile(path)]
    return file_paths
