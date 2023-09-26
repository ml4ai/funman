from typing import Any, List, Dict, Callable, Optional

from pydantic import BaseModel


from .assumption import Assumption

from pysmt.formula import FNode
from pysmt.logics import QF_NRA
from pysmt.shortcuts import Solver

class Explanation(BaseModel):
    _expression: FNode

    def explain(self) -> Dict[str, Any]:
        return {
            "description": "The expression is implied by this scenario and is unsatisfiable",
            "expression" : self._expression.serialize()
        }

class BoxExplanation(Explanation):
    unsat_assumptions: List[Assumption] = []
    expression: Optional[str] = None

    def check_assumptions(self, episode:"BoxSearchEpisode", my_solver: Callable) -> List[Assumption]:
        self.expression = self._expression.serialize()
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

    def explain(self) -> Dict[str, Any]:
        unsat_constraints = [a.constraint.model_dump() for a in self.unsat_assumptions]
        expl = {
            "description": "The scenario implies that the constraints are unsatisfiable",
            "unsat_constraints": unsat_constraints
                }
        if self.expression is not None:
            expl["expression"] = self.expression
        return expl

class ParameterSpaceExplanation(Explanation):
    true_explanations: List[BoxExplanation] = []
    false_explanations: List[BoxExplanation] = []
