"""
This module defines the Funman class, the primary entry point for FUNMAN
analysis.
"""
import logging
import threading
from typing import Callable, Optional, Union

import multiprocess as mp
from pydantic import BaseModel, Field, validator

from funman.utils.handlers import (
    NoopResultHandler,
    ResultCombinedHandler,
    ResultHandler,
    WaitAction,
)

l = logging.getLogger(__file__)
l.setLevel(logging.ERROR)


class FUNMANConfig(BaseModel):
    """
    Base definition of a configuration object
    """

    class Config:
        underscore_attrs_are_private = True
        arbitrary_types_allowed = True

    tolerance: float = 1e-8
    """Algorithm-specific tolerance for approximation, used by BoxSearch"""

    queue_timeout: int = 1
    """Multiprocessing queue timeout, used by BoxSearch"""
    number_of_processes: int = 1  # mp.cpu_count()
    """Number of BoxSearch processes"""
    _handler: Union[
        ResultCombinedHandler, NoopResultHandler, ResultHandler
    ] = NoopResultHandler()
    wait_timeout: int = None
    """Timeout for BoxSearch procesess to wait for boxes to evaluate"""
    _wait_action: WaitAction = None
    wait_action_timeout: float = 0.05
    """Time to sleep proceses waiting for work"""
    _read_cache: ResultHandler = None
    # episode_type: =None,
    _search: str = None
    """Name of search algorithm to use"""
    solver: str = "dreal"  # "z3"
    """Name of pysmt solver to use"""
    num_steps: int = 2
    """Number of timesteps to encode"""
    step_size: int = 1
    """Step size for encoding"""
    num_initial_boxes: int = 1
    """Number of initial boxes for BoxSearch"""
    initial_state_tolerance = 0.0
    """Factor used to relax initial state values bounds"""
    save_smtlib: bool = False
    """Whether to save each smt invocation as an SMTLib file"""
    dreal_precision: float = 1
    """Precision delta for dreal solver"""
    dreal_log_level: str = "off"
    """Constraint noise term to relax constraints"""
    constraint_noise: float = 0.0
    """Use MCTS in dreal"""
    dreal_mcts = True

    @validator("solver")
    def import_dreal(cls, v):
        if v == "dreal":
            try:
                import funman_dreal
            except:
                raise Exception(
                    "The funman_dreal package failed to import. Do you have it installed?"
                )
            else:
                funman_dreal.ensure_dreal_in_pysmt()
        return v


class Funman(object):
    """
    The Funman FUNctional Model ANalysis class is the main entry point for
    performing analysis on models.  The main entry point Funman.solve() performs analysis of an funman.scenario.AnalysisScenario, subject to a set of configuration options in the funman.search.SearchConfig class.
    """

    def solve(
        self,
        problem: "AnalysisScenario",
        config: FUNMANConfig = FUNMANConfig(),
        haltEvent: Optional[threading.Event] = None,
        resultsCallback: Optional[Callable[["ParameterSpace"], None]] = None,
    ) -> "AnalysisScenarioResult":
        """
        This method is the main entry point for Funman analysis.  Its inputs
        describe an AnalysisScenario and SearchConfig that setup the problem and
        analysis technique.  The return value is an AnalysisScenarioResult,
        which comprises all relevant output pertaining to the AnalysisScenario.

        Parameters
        ----------
        problem : AnalysisScenario
            The problem is a description of the analysis to be performed, and
            typically describes a Model and a Query.
        config : SearchConfig, optional
            The configuration for the search algorithm applied to analyze the
            problem, by default SearchConfig()

        Returns
        -------
        AnalysisScenarioResult
            The resulting data, statistics, and other relevant information
            produced by the analysis.
        """
        return problem.solve(
            config, haltEvent=haltEvent, resultsCallback=resultsCallback
        )
