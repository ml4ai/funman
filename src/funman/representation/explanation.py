from typing import List, Dict, Callable

from pydantic import BaseModel

from funman.search.box_search import BoxSearchEpisode

from .assumption import Assumption

from pysmt.formula import FNode
from pysmt.logics import QF_NRA
from pysmt.shortcuts import Solver

class Explanation(BaseModel):
    _expression: FNode



class BoxExplanation(Explanation):
    unsat_assumptions: List[Assumption] = []

    def check_assumptions(self, episode:BoxSearchEpisode, my_solver: Callable) -> List[Assumption]:
        # FIXME use step size from options
        assumption_symbols:Dict[Assumption, FNode] = episode.problem._encodings[1]._encoder.encode_assumptions(episode.problem._assumptions, None)
        with my_solver() as solver:
            solver.add_assertion(self._expression)
            solver.push(1)

            self.unsat_assumptions = [a for a, symbol in assumption_symbols.items() if not self.satisfies_assumption(symbol, solver)]
        return self.unsat_assumptions
    
    def satisfies_assumption(self, assumption: FNode, solver: Solver)-> bool:
        solver.push(1)
        solver.add_assertion(assumption)
        is_sat = solver.solve()
        solver.pop(1)
        return is_sat


class ParameterSpaceExplanation(Explanation):
    true_explanations: List[BoxExplanation] = []
    false_explanations: List[BoxExplanation] = []
