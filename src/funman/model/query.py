"""
This module defines all Query classes.  Queries are combined with Model objects in Scenarios to determine whether the model satisfies the query.
"""
from typing import List, Optional, Union

from pydantic import BaseModel, ConfigDict
from pysmt.formula import FNode

from funman.model.model import Model
from funman.representation.symbol import ModelSymbol


class Query(BaseModel):
    """
    Abstract base class for queries.
    """

    model: Optional[Model] = None


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

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

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

    variable: Union[str, ModelSymbol]
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

    variable: Union[str, ModelSymbol]
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

    variable: Union[str, ModelSymbol]
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

    queries: List[Union[QueryLE, QueryGE, QueryEquals, Query]]
