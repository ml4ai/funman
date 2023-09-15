"""
This module contains defintions of constants used within FUNMAN
"""
import sys
from typing import Union

BIG_NUMBER: float = 1.0e6
NEG_INFINITY: Union[float, str] = -sys.float_info.max
POS_INFINITY: Union[float, str] = sys.float_info.max
