from unicodedata import name


class Model(object):
    def __init__(self, formula) -> None:
        self.formula = formula


class Parameter(object):
    def __init__(self, name, symbol=None) -> None:
        self.name = name

        # if the symbol is None, then need to get the symbol from a solver
        self.symbol = symbol

    def __eq__(self, other):
        if not isinstance(other, Parameter):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return self.symbol.symbol_name() == other.symbol.symbol_name()

    def __hash__(self):
        # necessary for instances to behave sanely in dicts and sets.
        return hash(self.symbol)
