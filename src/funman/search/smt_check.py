from typing import Optional

from pysmt.logics import QF_NRA
from pysmt.shortcuts import And, Solver

from funman.search_episode import SearchEpisode
from funman.utils.search_utils import SearchConfig

from .search import Search


class SMTCheck(Search):
    def search(
        self, problem, config: Optional[SearchConfig] = None
    ) -> SearchEpisode:
        episode = SearchEpisode(config=config, problem=problem)
        result = self.expand(problem, episode)
        episode.model = result
        return result

    def expand(self, problem, episode):
        with Solver(name=episode.config.solver, logic=QF_NRA) as s:
            s.add_assertion(
                And(
                    problem.model_encoding.formula,
                    problem.query_encoding.formula,
                )
            )
            result = s.solve()
            if result:
                result = s.get_model()
        return result
