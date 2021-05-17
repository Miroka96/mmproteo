from typing import Optional, NoReturn, Dict

from mmproteo.utils import utils


def test_flatten_single_element_containers() -> Optional[NoReturn]:
    a: Dict = dict()
    assert utils.flatten_element_containers(a) == dict()
    assert utils.flatten_element_containers([a]) == {0: dict()}

    b = [1, 2]
    assert utils.flatten_element_containers(b) == {0: 1, 1: 2}
    assert utils.flatten_element_containers([b]) == {0: {0: 1, 1: 2}}
    assert utils.flatten_element_containers([[b]]) == {0: {0: {0: 1, 1: 2}}}

    c = "c"
    assert utils.flatten_element_containers(c) == c
    assert utils.flatten_element_containers({c}) == {0: c}
    assert utils.flatten_element_containers([{c}]) == {0: {0: c}}

    return None


def test_flatten_dict_without_concatenation() -> Optional[NoReturn]:
    assert utils.flatten_dict({0: 0, 1: {0: 1}}, concat_keys=False, clean_keys=False) == {0: 0}
    assert utils.flatten_dict({0: 0, 1: [1]}, concat_keys=False, clean_keys=False) == {0: 0}
    assert utils.flatten_dict({0: 0, 1: {1}}, concat_keys=False, clean_keys=False) == {0: 0}
    assert utils.flatten_dict({0: 0, 1: (1,)}, concat_keys=False, clean_keys=False) == {0: 0}

    a = {3: 4, 4: [4], 5: {5}}
    b = {1: [1], 2: {2}, 3: 3, 10: 0, "a": a}

    res = {0: 1, 3: 3, 10: 0}
    assert utils.flatten_dict(b, concat_keys=False, clean_keys=False) == res
    return None


def test_flatten_dict_without_concatenation_with_cleaning() -> Optional[NoReturn]:
    assert utils.flatten_dict({0: 0, 1: {0: 1}}, concat_keys=False, clean_keys=True) == {'0': 0}
    assert utils.flatten_dict({0: 0, 1: [1]}, concat_keys=False, clean_keys=True) == {'0': 0}
    assert utils.flatten_dict({0: 0, 1: {1}}, concat_keys=False, clean_keys=True) == {'0': 0}
    assert utils.flatten_dict({0: 0, 1: (1,)}, concat_keys=False, clean_keys=True) == {'0': 0}
    assert utils.flatten_dict({0: 0, 1: (1, 2)}, concat_keys=False, clean_keys=True) == {'0': 0, '1': 2}

    a = {3: 4, 4: [4], 5: {5}}
    b = {0: 0, 1: [1], 2: {2}, 3: 3, "a": a}

    res = {'0': 0, '3': 3}
    assert utils.flatten_dict(b, concat_keys=False, clean_keys=True) == res
    return None


def test_flatten_dict_with_concatenation() -> Optional[NoReturn]:
    a = {3: 4, 4: [4], 5: {5}}
    b = {0: 0, 1: [1], 2: {2}, 3: 3, "a": a}

    res = {"0": 0, "1__0": 1, "2__0": 2, "3": 3, "a__3": 4, "a__4__0": 4, "a__5__0": 5}
    assert utils.flatten_dict(b, concat_keys=True, clean_keys=False) == res
    assert utils.flatten_dict(b, concat_keys=True, clean_keys=True) == res
    return None
