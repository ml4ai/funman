from funman.constants import NEG_INFINITY, POS_INFINITY


# lhs < rhs
def lt(lhs, rhs):
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
def gt(lhs, rhs):
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
def gte(lhs, rhs):
    return not lt(lhs, rhs)


# lhs <= rhs
def lte(lhs, rhs):
    return not gt(lhs, rhs)


def minus(lhs, rhs):
    if lhs == rhs:
        return 0
    # inf - (< inf)
    if lhs == POS_INFINITY:
        return POS_INFINITY
    # (-inf) - (> -inf)
    if lhs == NEG_INFINITY:
        return NEG_INFINITY
    if rhs == POS_INFINITY:
        return NEG_INFINITY
    if rhs == NEG_INFINITY:
        return POS_INFINITY
    return lhs - rhs


def plus(numer, denom):
    if numer == NEG_INFINITY:
        if denom == POS_INFINITY:
            return 0
        else:
            return NEG_INFINITY
    if denom == NEG_INFINITY:
        if numer == POS_INFINITY:
            return 0
        else:
            return NEG_INFINITY
    if numer == POS_INFINITY:
        if denom == NEG_INFINITY:
            return 0
        else:
            return POS_INFINITY
    if denom == POS_INFINITY:
        if numer == NEG_INFINITY:
            return 0
        else:
            return POS_INFINITY
    return numer + denom


def div(numer: float, denom: float) -> float:
    if numer == NEG_INFINITY:
        if denom == POS_INFINITY:
            return NEG_INFINITY
        else:
            return POS_INFINITY
    if denom == NEG_INFINITY:
        if numer == POS_INFINITY:
            return NEG_INFINITY
        else:
            return POS_INFINITY
    if numer == POS_INFINITY:
        if denom == NEG_INFINITY:
            return NEG_INFINITY
        else:
            return POS_INFINITY
    if denom == POS_INFINITY:
        if numer == NEG_INFINITY:
            return NEG_INFINITY
        else:
            return POS_INFINITY
    if float(denom) == 0.0:
        raise ZeroDivisionError()
    return numer / denom
