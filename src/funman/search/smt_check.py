from typing import Optional

from pysmt.logics import QF_NRA
from pysmt.shortcuts import And, Solver

# import funman.search as search
from .search import Search, SearchEpisode


class SMTCheck(Search):
    def search(
        self, problem, config: Optional["FUNMANConfig"] = None
    ) -> "SearchEpisode":
        episode = SearchEpisode(config=config, problem=problem)
        result = self.expand(problem, episode)
        episode._model = result
        return result

    def expand(self, problem, episode):
        with Solver(name=episode.config.solver, logic=QF_NRA) as s:
            s.add_assertion(
                And(
                    problem._model_encoding.formula,
                    problem._query_encoding.formula,
                )
            )
            result = s.solve()
            if result:
                result = s.get_model()
        return result
