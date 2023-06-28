import threading
from typing import Callable, Optional
from funman.representation.representation import (
    LABEL_TRUE,
    ParameterSpace,
    Point,
)

from pysmt.logics import QF_NRA
from pysmt.shortcuts import And, Solver

from funman.utils.smtlib_utils import smtlibscript_from_formula_list

# import funman.search as search
from .search import Search, SearchEpisode


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
        for (
            structural_configuration
        ) in problem._smt_encoder._timed_model_elements["configurations"]:
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
            parameter_values = {
                k: v
                for k, v in result_dict.items()
                if k in [p.name for p in problem.parameters]
            }
            point = Point(values=parameter_values, label=LABEL_TRUE)
            models[point] = result
            parameter_space.true_points.append(point)
            resultsCallback(parameter_space)

        return parameter_space, models

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
