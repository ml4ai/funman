import json
import logging
import sys
import threading
from typing import Callable, Optional

from pysmt.logics import QF_NRA
from pysmt.shortcuts import And, Solver

from funman.representation.representation import (
    LABEL_TRUE,
    ParameterSpace,
    Point,
)
from funman.utils.smtlib_utils import smtlibscript_from_formula_list

# import funman.search as search
from .search import Search, SearchEpisode

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
            l.info(f"Solving configuration: {structural_configuration}")
            problem._encode_timed(
                structural_configuration["num_steps"],
                structural_configuration["step_size"],
                config,
            )
            episode = SearchEpisode(
                config=config,
                problem=problem,
                structural_configuration=structural_configuration,
            )

            result = self.expand(problem, episode, parameter_space)

            result_dict = result.to_dict() if result else None
            l.info(f"Result: {json.dumps(result_dict, indent=4)}")
            if result_dict:
                parameter_values = {
                    k: v
                    for k, v in result_dict.items()
                    if k in [p.name for p in problem.parameters]
                }
                for k, v in structural_configuration.items():
                    parameter_values[k] = v
                point = Point(values=parameter_values, label=LABEL_TRUE)
                models[point] = result
                consistent[point] = result_dict
                parameter_space.true_points.append(point)
            resultsCallback(parameter_space)

        return parameter_space, models, consistent

    def expand(self, problem, episode, parameter_space):
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
            formula = And(
                problem._model_encoding._formula,
                problem._query_encoding._formula,
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
