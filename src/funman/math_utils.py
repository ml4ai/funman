"""
This submodule contains helper functions for math operations used in FUNMAN
"""
from typing import Literal, Union
from funman.constants import NEG_INFINITY, POS_INFINITY

Number = Union[int, float, Literal["-inf", "inf"]]

# lhs < rhs
def lt(lhs: Number, rhs: Number) -> bool:
    """
    Less than comparison

    Parameters
    ----------
    lhs : Number
        The left hand side
    rhs : Number
        The right hand side

    Returns
    -------
    bool
        lhs < rhs
    """
    # inf < ?
    if lhs == POS_INFINITY:
        return False

    # ? < -inf
    if rhs == NEG_INFINITY:
        return False

    # -inf < ?
    if lhs == NEG_INFINITY:
        # -inf < (> -int)
        return True

    # ? < inf
    if rhs == POS_INFINITY:
        # (< inf) < inf
        return True

    # there is no case where
    # - lhs = POS_INFINITY
    # - lhs = NEG_INFINITY
    # - rhs = POS_INFINITY
    # - rhs = NEG_INFINITY
    # so just use the float check
    return lhs < rhs

# lhs > rhs
def gt(lhs: Number, rhs: Number) -> bool:
    """
    Greater than comparison

    Parameters
    ----------
    lhs : Number
        The left hand side
    rhs : Number
        The right hand side

    Returns
    -------
    bool
        lhs > rhs
    """
    # ? > inf
    if rhs == POS_INFINITY:
        return False

    # -inf > ?
    if lhs == NEG_INFINITY:
        return False

    # ? > -inf
    if rhs == NEG_INFINITY:
        # (> -int) > -inf
        return True

    # inf > ?
    if lhs == POS_INFINITY:
        # inf > (< inf)
        return True

    # there is no case where
    # - lhs = POS_INFINITY
    # - lhs = NEG_INFINITY
    # - rhs = POS_INFINITY
    # - rhs = NEG_INFINITY
    # so just use the float check
    return lhs > rhs

# lhs >= rhs
def gte(lhs: Number, rhs: Number) -> bool:
    """
    Greater than or equal comparison

    Parameters
    ----------
    lhs : Number
        The left hand side
    rhs : Number
        The right hand side

    Returns
    -------
    bool
        lhs >= rhs
    """
    return not lt(lhs, rhs)

# lhs <= rhs
def lte(lhs: Number, rhs: Number) -> bool:
    """
    Less than or equal comparison

    Parameters
    ----------
    lhs : Number
        The left hand side
    rhs : Number
        The right hand side

    Returns
    -------
    bool
        lhs <= rhs
    """
    return not gt(lhs, rhs)