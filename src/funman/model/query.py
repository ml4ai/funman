"""
This module defines all Query classes.  Queries are combined with Model objects in Scenarios to determine whether the model satisfies the query.
"""
from abc import ABC
from typing import List

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

    function: str


class QueryTrue(Query):
    """
    Query that represents logical true.  I.e., this query does not place any additional constraints upon a model.
    """

    pass


class QueryEncoded(Query):
    """
    Class to contain a formula that is already encoded by a pysmt FNode.
    """

    class Config:
        arbitrary_types_allowed = True
        underscore_attrs_are_private = True

    _formula: FNode


class QueryLE(Query):
    """
    Class to represent a query of the form: var <= ub, where var is a variable, and ub is a constant upper bound.

    Parameters
    ----------
    variable : str
        model variable name
    ub : float
        upper bound constant
    at_end : bool, optional
        apply the constraint to the last timepoint of a scenario only, by default False
    """

    variable: str
    ub: float
    at_end: bool = False


class QueryGE(Query):
    """
    Class to represent a query of the form: var >= lb, where var is a variable, and lb is a constant lower bound.

    Parameters
    ----------
    variable : str
        model variable name
    lb : float
        lower bound constant
    at_end : bool, optional
        apply the constraint to the last timepoint of a scenario only, by default False
    """

    variable: str
    lb: float
    at_end: bool = False


class QueryEquals(Query):
    """
    Class to represent a query of the form: var == value, where var is a variable, and value is a constant.

    Parameters
    ----------
    variable : str
        model variable name
    value : float
        value
    at_end : bool, optional
        apply the constraint to the last timepoint of a scenario only, by default False
    """

    variable: str
    value: float
    at_end: bool = False


class QueryAnd(Query):
    """
    Conjunction of queries.

    Parameters
    ----------
    queries : List[Query]
        queries to conjoin.
    """

    queries: List[Query]
