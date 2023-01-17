from enum import IntEnum, Enum
from typing import Union

from .decorators import *
from .widgets import *
from .concurrent import asynchronous


def enum2int(enum: Union[Enum, IntEnum]) -> int:
    """
    PyQt5 uses IntEnum like object for settings, for example SortOrder while
    PyQt6 uses Enum. PyQt5's IntEnum also does not support value attribute.
    This function transform both settings objects to int.

    Parameters
    ----------
    enum
        IntEnum like object or Enum object with Qt's settings

    Returns
    -------
    Settings transformed to int
    """
    return int(enum) if isinstance(enum, int) else enum.value
