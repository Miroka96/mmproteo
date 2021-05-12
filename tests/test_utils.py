from typing import Optional, NoReturn, Dict

from mmproteo.utils import utils


def test_flatten_single_element_containers() -> Optional[NoReturn]:
    a: Dict = dict()
    assert utils.flatten_element_containers(a) == a
    assert utils.flatten_element_containers([a]) == a

    b = [1, 2]
    assert utils.flatten_element_containers(b) == b
    assert utils.flatten_element_containers([[b]]) == b

    c = "c"
    assert utils.flatten_element_containers(c) == c
    assert utils.flatten_element_containers([{c}]) == c

    return None


def test_flatten_dict_without_concatenation() -> Optional[NoReturn]:
    a = {3: 4, 4: [4], 5: {5}}
    b = {0: 0, 1: [1], 2: {2}, 3: 3, "a": a}

    res = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
    assert utils.flatten_dict(b, concat_keys=False, clean_keys=False) == res
    return None


def test_flatten_dict_without_concatenation_with_cleaning() \
        -> Optional[NoReturn]:
    a = {3: 4, 4: [4], 5: {5}}
    b = {0: 0, 1: [1], 2: {2}, 3: 3, "a": a}

    res = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5}
    assert utils.flatten_dict(b, concat_keys=False, clean_keys=True) == res
    return None


def test_flatten_dict_with_concatenation() -> Optional[NoReturn]:
    a = {3: 4, 4: [4], 5: {5}}
    b = {0: 0, 1: [1], 2: {2}, 3: 3, "a": a}

    res = {"0": 0, "1": 1, "2": 2, "3": 3, "a__3": 4, "a__4": 4, "a__5": 5}
    assert utils.flatten_dict(b, concat_keys=True, clean_keys=False) == res
    assert utils.flatten_dict(b, concat_keys=True, clean_keys=True) == res
    return None
