from asyncio.subprocess import Process
from multiprocessing import Queue
from funman.constants import BIG_NUMBER, POS_INFINITY, NEG_INFINITY
from funman.scenario import ParameterSynthesisScenario
from funman.scenario import (
    AnalysisScenario,
    AnalysisScenarioResult,
    ParameterSynthesisScenarioResult,
)
from funman.search import Box, BoxSearch, SearchConfig


import logging

l = logging.getLogger(__file__)
l.setLevel(logging.ERROR)


class Funman(object):
    def __init__(self) -> None:
        self.scenario_handlers = {
            ParameterSynthesisScenario: self.synthesize_parameters
        }

    def solve(
        self, problem: AnalysisScenario, config: SearchConfig = SearchConfig()
    ) -> AnalysisScenarioResult:
        return self.scenario_handlers[type(problem)](problem, config=config)

    def synthesize_parameters(
        self, problem: ParameterSynthesisScenario, config: SearchConfig = SearchConfig()
    ) -> ParameterSynthesisScenarioResult:
        """ """

        search = BoxSearch()
        result = search.search(problem, config=config)

        
        result.plot(
            title=f"Residual = {result.statistics.current_residual}, |boxes|={result.statistics.num_unknown}",
        )

        # FIXME the parameter space will be added the to object below, in addition to the problem.
        #       Initially, the parameter space can be a set of boxes
        return ParameterSynthesisScenarioResult(problem, result)
