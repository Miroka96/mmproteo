import glob
import os
import random
from typing import Dict, List, Optional, Tuple, Iterable, Set

from mmproteo.utils import visualization, utils, log


def _get_values_at_wildcard_path_position(wildcard_path: str, path_position: int) -> Set[str]:
    return {path.split(os.path.sep)[path_position] for path in glob.glob(wildcard_path)}


def _create_placeholder_path(wildcard_path: str, path_position: int, placeholder_name: str = 'value') -> str:
    parts = wildcard_path.split(os.path.sep)
    parts[path_position] = f"{{{placeholder_name}}}"
    return os.path.sep.join(parts)


def _assign_values_randomly_to_splits(values: Iterable[str], splits: Dict[str, float]) -> Dict[str, List[str]]:
    prefixed_splits: Dict[Optional[str], float] = splits  # type: ignore
    prefixed_splits[None] = 0

    sorted_splits: List[Tuple[Optional[str], float]] = sorted(prefixed_splits.items(), key=lambda tupl: tupl[1])
    assert sorted_splits[0][0] is None, "the added zero should also be the zeroth element"

    shuffled_values = list(values)
    random.shuffle(shuffled_values)

    assigned_values = {
        category: shuffled_values[
                            int(sorted_splits[i][1] * len(shuffled_values)):
                            int(split * len(shuffled_values))
                            ]
        for i, (category, split) in enumerate(sorted_splits[1:])
    }
    return assigned_values  # type: ignore


def _find_file_paths_per_category_value(
        assigned_values: Dict[str, List[str]],
        wildcard_path_with_placeholder: str,
        placeholder_name: str = 'value'
) -> Dict[str, List[str]]:
    return {
        category: utils.flatten_single_list(
            [
                glob.glob(wildcard_path_with_placeholder.format(**{placeholder_name: value})) for value in values
            ]
        ) for category, values in assigned_values.items()
    }


def _store_assigned_file_paths_as_json(
        assigned_file_paths: Dict[str, List[str]],
        output_file: str) -> str:
    with open(output_file, 'w') as file:
        file.write(visualization.pretty_print_json(assigned_file_paths))

    return output_file


def assign_wildcard_paths_to_splits_grouped_by_path_position_value(
        wildcard_path: str,
        path_position: int,
        splits: Dict[str, float],
        paths_dump_file: Optional[str] = None,
        skip_existing: bool = True,
        logger: log.Logger = log.DEFAULT_LOGGER
) -> Dict[str, List[str]]:
    """
    Collect and assign paths from a given wildcard path and split the paths grouped by values at a given path
    position.

    :param wildcard_path:   a path containing '*'s to match several files/directories simultaneously
    :param path_position:   a path-separator separated position index in the given wildcard path. Negative index
                            values matches the path snippets count reversely
    :param splits:          a dictionary containing the categories, to which the paths should be assigned, as keys.
                            The corresponding float values represent the upper bound of the share for this category.
                            The float values should be accumulative.
    :param paths_dump_file: a file path to use for storing/loading the (random) path assignment (optional)
    :param skip_existing:   whether to try loading path assignments from
                        :paramref:`split_collected_paths_by_path_position_value.paths_dump_file`
    :param logger:          the logger instance to use
    :return: a dictionary with the category values of the splits as keys and lists of assigned file paths as values
    """
    if skip_existing and paths_dump_file is not None and os.path.exists(paths_dump_file):
        file_paths = utils.load_json(paths_dump_file)
        logger.info(f"found file paths dump '{paths_dump_file}'")
        return file_paths

    categorical_values = _get_values_at_wildcard_path_position(wildcard_path, path_position)
    assigned_values = _assign_values_randomly_to_splits(categorical_values, splits=splits)
    logger.debug("assigned values:")
    visualization.print_list_length_in_dict(assigned_values, print_func=logger.debug)

    file_paths = _find_file_paths_per_category_value(
        assigned_values=assigned_values,
        wildcard_path_with_placeholder=_create_placeholder_path(
            wildcard_path=wildcard_path,
            path_position=path_position,
            placeholder_name='value'
        ),
        placeholder_name='value',
    )
    if paths_dump_file is not None:
        _store_assigned_file_paths_as_json(file_paths, paths_dump_file)
        logger.info(f"dumped file paths into '{paths_dump_file}'")
    return file_paths
