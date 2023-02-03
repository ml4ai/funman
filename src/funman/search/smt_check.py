from typing import Optional

from pysmt.logics import QF_NRA
from pysmt.shortcuts import And, Solver

from funman.utils.smtlib_utils import smtlibscript_from_formula_list

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
            formula = And(
                problem._model_encoding.formula,
                problem._query_encoding.formula,
            )
            s.add_assertion(formula)
            self.store_smtlib(formula)
            result = s.solve()
            if result:
                result = s.get_model()
        return result

    def store_smtlib(self, formula, filename="dbg.smt2"):
        with open(filename, "w") as f:
            smtlibscript_from_formula_list(
                [formula],
                logic=QF_NRA,
            ).serialize(f, daggify=False)
