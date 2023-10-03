"""
This module contains defintions of constants used within FUNMAN
"""
import sys
from typing import Literal, Union

BIG_NUMBER: float = 1.0e6
NEG_INFINITY: Union[float, str] = -sys.float_info.max
POS_INFINITY: Union[float, str] = sys.float_info.max

LABEL_ANY = "any"
LABEL_ALL = "all"

LABEL_TRUE: Literal["true"] = "true"
LABEL_FALSE: Literal["false"] = "false"
LABEL_UNKNOWN: Literal["unknown"] = "unknown"
LABEL_DROPPED: Literal["dropped"] = "dropped"
Label = Literal["true", "false", "unknown", "dropped"]
