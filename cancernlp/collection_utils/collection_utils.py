"""Extra utilities for working with collections and iterators."""
import functools
import re
from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, TypeVar, Union

K = TypeVar("K")
K2 = TypeVar("K2")
T = TypeVar("T")

# keys can either be functions that take an object and return a key value, or a string,
# in which case the string is an object attribute as accepted by get_by_path
KeyType = Union[str, Callable[[T], K]]
KeyType2 = Union[str, Callable[[T], K2]]


def group_by(l: Iterable[T], key: KeyType) -> Dict[K, List[T]]:
    """
    Group elements of `l` by the return value of `key` called with `l` as an argument
    """
    if isinstance(key, str):
        key = functools.partial(get_by_path, path=key)
    grouped = defaultdict(list)
    for _entry in l:
        grouped[key(_entry)].append(_entry)
    return dict(grouped)


def _get_component(ob, component, raise_error=True):
    if hasattr(ob, component):
        return getattr(ob, component)

    try:
        return ob[component]
    except (KeyError, IndexError, TypeError):
        pass

    # try again with int index
    try:
        component = int(component)
        return ob[component]
    except (KeyError, IndexError, TypeError, ValueError):
        pass

    if raise_error:
        raise KeyError(component)

    return None


class _NoDefault:
    pass


def get_by_path(ob: Any, path: str, default=_NoDefault):
    """
    Return nested property / dict / list elements by dotted path.

    Example:
    >>> ob = {"hello": ["a", "b", {"c": "d"}]}
    >>> get_by_path(ob, "hello[2].c")
    Out[1]: 'd'
    >>> get_by_path(ob, "hello[2].keys")
    Out[2]: dict_keys(['c'])
    """

    path_components = [_p for _p in re.split(r"[.\[\]]+", path) if _p]
    cur_ob = ob
    try:
        for _component in path_components:
            cur_ob = _get_component(cur_ob, _component)
    except KeyError:
        if default is _NoDefault:
            raise
        cur_ob = default

    return cur_ob
