"""
This module defines the Funman class, the primary entry point for FUNMAN
analysis.
"""
import logging
import threading
from typing import Callable, Optional

from .config import FUNMANConfig

l = logging.getLogger(__file__)
l.setLevel(logging.ERROR)


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
        try:
            if config.profile:
                import cProfile

                with cProfile.Profile() as pr:
                    result = problem.solve(
                        config,
                        haltEvent=haltEvent,
                        resultsCallback=resultsCallback,
                    )
                    pr.dump_stats("profile.stats")
            else:
                result = problem.solve(
                    config,
                    haltEvent=haltEvent,
                    resultsCallback=resultsCallback,
                )
            return result
        except Exception as e:
            l.exception(f"funman.solve() exiting due to exception: {e}")
