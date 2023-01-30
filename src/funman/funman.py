"""
This module defines the Funman class, the primary entry point for FUNMAN
analysis.
"""
import logging

import multiprocess as mp
from pydantic import BaseModel, validator

from funman.utils.handlers import NoopResultHandler, ResultHandler, WaitAction

l = logging.getLogger(__file__)
l.setLevel(logging.ERROR)


class FUNMANConfig(BaseModel):
    """
    Base definition of a configuration object
    """

    class Config:
        underscore_attrs_are_private = True
        arbitrary_types_allowed = True

    tolerance: float = 0.1

    queue_timeout: int = 1
    number_of_processes: int = mp.cpu_count()
    _handler: ResultHandler = NoopResultHandler()
    wait_timeout: int = None
    _wait_action: WaitAction = None
    wait_action_timeout: float = 0.05
    _read_cache: ResultHandler = None
    # episode_type: =None,
    _search: str = None
    solver: str = "z3"
    max_steps: int = 2
    step_size: int = 1
    num_initial_boxes: int = 1

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
        return problem.solve(config)
