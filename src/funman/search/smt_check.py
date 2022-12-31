from typing import Optional

from pysmt.logics import QF_NRA
from pysmt.shortcuts import And, Solver

# import funman.search as search
from . import search


class SMTCheck(search.Search):
    def search(
        self, problem, config: Optional[search.SearchConfig] = None
    ) -> search.SearchEpisode:
        episode = search.SearchEpisode(config=config, problem=problem)
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
