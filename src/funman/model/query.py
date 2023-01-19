"""
This module defines all Query classes.  Queries are combined with Model objects in Scenarios to determine whether the model satisfies the query.
"""
from abc import ABC
from typing import Callable

from pydantic import BaseModel
from pysmt.formula import FNode


class Query(ABC, BaseModel):
    """
    Abstract base class for queries.
    """

    pass


class QueryFunction(Query):
    """
    This query uses a Python function passed to the constructor to evaluate a query on the results of a scenario.
    """

    def __init__(self, function: Callable) -> None:
        super().__init__()
        self.function = function


class QueryTrue(Query):
    """
    Query that represents logical true.  I.e., this query does not place any additional constraints upon a model.
    """

    pass


class QueryEncoded(Query):
    """
    Class to contain a formula that is already encoded by a pysmt FNode.
    """

    def __init__(self, fnode: FNode) -> None:
        super().__init__()
        self.formula = fnode


class QueryLE(Query):
    """
    Class to represent a query of the form: var <= ub, where var is a variable, and ub is a constant upper bound.
    """

    def __init__(self, variable: str, ub: float, at_end: bool = False) -> None:
        """
        Create a QueryLE object.

        Parameters
        ----------
        variable : str
            model variable name
        ub : float
            upper bound constant
        at_end : bool, optional
            apply the constraint to the last timepoint of a scenario only, by default False
        """
        super().__init__()
        self.variable = variable
        self.ub = ub
        self.at_end = at_end
