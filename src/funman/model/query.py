class Query(object):
    def __init__(self) -> None:
        pass


class QueryTrue(Query):
    pass


class QueryEncoded(Query):
    def __init__(self, fnode) -> None:
        super().__init__()
        self.formula = fnode


class QueryLE(Query):
    def __init__(self, variable, ub, at_end=False) -> None:
        super().__init__()
        self.variable = variable
        self.ub = ub
        self.at_end = at_end
