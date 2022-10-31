"""
The funman package implements multiple simulator model analysis methods.  Current methods include:

- Simulation: running original simulator and querying the results.

- Parameter Synthesis: Generating feasible values for model parameters.
"""

import funman
from funman._version import __version__

from funman.config import Config
from funman.scenario import (
    AnalysisScenario,
    AnalysisScenarioResult,
)



import logging

l = logging.getLogger(__file__)
l.setLevel(logging.ERROR)

class Funman(object):
    """
    The Funman FUNctional Model ANalysis class is the main entry point for
    performing analysis on models.  The main entry point Funman.solve() performs
    analysis of an funman.scenario.AnalysisScenario, subject to a set of configuration
    options in the funman.search.SearchConfig class.
    """

    def solve(
        self, problem: AnalysisScenario, config: Config = None
    ) -> AnalysisScenarioResult:
        """
        This method is the main entry point for Funman analysis.  Its inputs
        describe an AnalysisScenario and SearchConfig that setup the problem and
        analysis technique.  The return value is an AnalysisScenarioResult,
        which comprises all relevant output pertaining to the AnalysisScenario.

        Parameters
        ----------
        problem : AnalysisScenario
            The problem is a description of the analysis to be performed, and typically describes a Model and a Query. 
        config : SearchConfig, optional
            The configuration for the search algorithm applied to analyze the problem, by default SearchConfig()

        Returns
        -------
        AnalysisScenarioResult
            The resulting data, statistics, and other relevant information produced by the analysis.
        """    
        return problem.solve(config)


def main():
    return funman.Funman()

if __name__ == "main":
    main()
    