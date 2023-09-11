import json
import logging
import sys
import threading
from functools import partial
from typing import Callable, Optional, Tuple

from pysmt.formula import FNode
from pysmt.logics import QF_NRA
from pysmt.shortcuts import REAL, And, Equals, Real, Solver, Symbol

from funman.representation.representation import (
    LABEL_FALSE,
    LABEL_TRUE,
    Box,
    Interval,
    ParameterSpace,
    Point,
)
from funman.scenario.scenario import AnalysisScenario
from funman.utils.smtlib_utils import smtlibscript_from_formula_list
from pysmt.solvers.solver import Model as pysmtModel

# import funman.search as search
from .search import Search, SearchEpisode

# from funman.utils.sympy_utils import sympy_to_pysmt, to_sympy


l = logging.getLogger(__file__)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
l.setLevel(logging.INFO)


class SMTCheck(Search):
    def search(
        self,
        problem,
        config: Optional["FUNMANConfig"] = None,
        haltEvent: Optional[threading.Event] = None,
        resultsCallback: Optional[Callable[["ParameterSpace"], None]] = None,
    ) -> "SearchEpisode":
        parameter_space = ParameterSpace(
            num_dimensions=problem.num_dimensions()
        )
        models = {}
        consistent = {}
        for (
            structural_configuration
        ) in problem._smt_encoder._timed_model_elements["configurations"]:
            l.debug(f"Solving configuration: {structural_configuration}")
            problem._encode_timed(
                structural_configuration["num_steps"],
                problem._smt_encoder._timed_model_elements["step_sizes"].index(
                    structural_configuration["step_size"]
                ),
                config,
            )
            episode = SearchEpisode(
                config=config,
                problem=problem,
                structural_configuration=structural_configuration,
            )
            # self._initialize_encoding(solver, episode, box_timepoint, box)
            result = self.expand(
                problem,
                episode,
                parameter_space,
                list(range(structural_configuration["num_steps"] + 1)),
            )

            if result is not None and isinstance(result, pysmtModel):
                result_dict = result.to_dict() if result else None
                l.debug(f"Result: {json.dumps(result_dict, indent=4)}")
                if result_dict is not None:
                    parameter_values = {
                        k: v
                        for k, v in result_dict.items()
                        # if k in [p.name for p in problem.parameters]
                    }
                    for k, v in structural_configuration.items():
                        parameter_values[k] = v
                    point = Point(values=parameter_values, label=LABEL_TRUE)
                    if config.normalize:
                        denormalized_point = point.denormalize(problem)
                        point = denormalized_point
                    models[point] = result
                    consistent[point] = result_dict
                    parameter_space.true_points.append(point)
            elif result is not  None and isinstance(result, str):
                box = Box(bounds={p.name: Interval(lb=p.lb, ub=p.ub) for p in problem.parameters}, label=LABEL_FALSE, explanation=result)
                parameter_space.false_boxes.append(box)
            if resultsCallback:
                resultsCallback(parameter_space)

        return parameter_space, models, consistent

    def build_formula(
        self, episode: SearchEpisode, timepoints
    ) -> Tuple[FNode, FNode]:
        formula: FNode = And(
            episode.problem._model_encoding.encoding(
                episode.problem._model_encoding._encoder.encode_model_layer,
                layers=timepoints,
            ),
            episode.problem._query_encoding.encoding(
                partial(
                    episode.problem._query_encoding._encoder.encode_query_layer,
                    episode.problem.query,
                    episode.problem,
                    episode.config,
                ),
                layers=timepoints,
            ),
            episode.problem._smt_encoder.box_to_smt(
                episode._initial_box().project(
                    episode.problem.model_parameters()
                )
            ),
        )
        simplified_formula = None
        if (
            episode.config.simplify_query
            and episode.config.substitute_subformulas
        ):
            simplified_formula = formula.substitute(
                episode.problem._smt_encoder._timed_model_elements[
                    "time_step_substitutions"
                ][0]
            ).simplify()
        return formula, simplified_formula

    def solve_formula(self, s: Solver, formula: FNode, episode):
        s.push(1)
        s.add_assertion(formula)
        if episode.config.save_smtlib:
            self.store_smtlib(
                formula,
                filename=f"dbg_steps{episode.structural_configuration['num_steps']}_ssize{episode.structural_configuration['step_size']}.smt2",
            )
        result = s.solve()
        if result:
            result = s.get_model()
        else:
            result = s.get_unsat_core()
        s.pop(1)
        return result

    def expand(self, problem, episode, parameter_space, timepoints):
        if episode.config.solver == "dreal":
            opts = {
                "dreal_precision": episode.config.dreal_precision,
                "dreal_log_level": episode.config.dreal_log_level,
                "dreal_mcts": episode.config.dreal_mcts,
            }
        else:
            opts = {}
        with Solver(
            name=episode.config.solver,
            logic=QF_NRA,
            solver_options=opts,
        ) as s:
            formula, simplified_formula = self.build_formula(
                episode, timepoints
            )

            if simplified_formula is not None:
                # If using a simplified formula, we need to solve it and use its values in the original formula to get the values of all variables
                result = self.solve_formula(s, simplified_formula, episode)
                if result is not None and isinstance(result, pysmtModel):
                    assigned_vars = result.to_dict()
                    substitution = {
                        Symbol(p, REAL): Real(v)
                        for p, v in assigned_vars.items()
                    }
                    result_assignment = And(
                        [
                            Equals(Symbol(p, REAL), Real(v))
                            for p, v in assigned_vars.items()
                        ]
                        + [
                            Equals(Symbol(p.name, REAL), Real(0.0))
                            for p in episode.problem.model_parameters()
                            if p.is_unbound() and p.name not in assigned_vars
                        ]
                    )
                    formula_w_params = And(
                        formula.substitute(substitution), result_assignment
                    )
                    result = self.solve_formula(s, formula_w_params, episode)
                elif result is not None and isinstance(result, str):
                    # Unsat core
                    pass
            else:
                result = self.solve_formula(s, formula, episode)

        return result

    def store_smtlib(self, formula, filename="dbg.smt2"):
        with open(filename, "w") as f:
            smtlibscript_from_formula_list(
                [formula],
                logic=QF_NRA,
            ).serialize(f, daggify=False)
