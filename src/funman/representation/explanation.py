from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel
from pysmt.formula import FNode

from .assumption import Assumption


class Explanation(BaseModel):
    _expression: FNode

    def explain(self) -> Dict[str, Any]:
        return {
            "description": "The expression is implied by this scenario and is unsatisfiable",
            "expression": self._expression.serialize(),
        }


class BoxExplanation(Explanation):
    relevant_assumptions: List[Assumption] = []
    expression: Optional[str] = None

    def check_assumptions(
        self,
        episode: "BoxSearchEpisode",
        my_solver: Callable,
        options: "EncodingOptions",
    ) -> List[Assumption]:
        """
        Find the assumptions that are unit clauses in the expression (unsat core).

        Parameters
        ----------
        episode : BoxSearchEpisode
            _description_
        my_solver : Callable
            _description_

        Returns
        -------
        List[Assumption]
            _description_
        """
        self.expression = self._expression.serialize()
        # FIXME use step size from options
        assumption_symbols: Dict[str, Assumption] = {
            str(a): a for a in episode.problem._assumptions
        }
        # with my_solver() as solver:
        #     solver.add_assertion(self._expression)
        #     solver.push(1)
        expression_symbols = [
            str(v) for v in self._expression.get_free_variables()
        ]
        self.relevant_assumptions = [
            a
            for symbol, a in assumption_symbols.items()
            if symbol in expression_symbols
        ]
        return self.relevant_assumptions

    # def satisfies_assumption(self, assumption: FNode)-> bool:
    #     # solver.push(1)
    #     # solver.add_assertion(assumption)
    #     # is_sat = solver.solve()
    #     # solver.pop(1)
    #     is_sat =
    #     return is_sat

    def explain(self) -> Dict[str, Any]:
        relevant_constraints = [
            a.constraint.model_dump() for a in self.relevant_assumptions
        ]
        expl = {"relevant_constraints": relevant_constraints}
        if self.expression is not None:
            expl["expression"] = self.expression
        return expl


class ParameterSpaceExplanation(Explanation):
    true_explanations: List[BoxExplanation] = []
    false_explanations: List[BoxExplanation] = []
