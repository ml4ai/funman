import json
import logging
import os
import unittest

from funman_demo.handlers import RealtimeResultPlotter, ResultCacheWriter

from funman import Funman
from funman.funman import FUNMANConfig
from funman.model import QueryLE
from funman.model.bilayer import BilayerDynamics, BilayerModel
from funman.model.query import QueryTrue
from funman.representation.representation import Parameter
from funman.scenario import (
    ConsistencyScenario,
    ConsistencyScenarioResult,
    ParameterSynthesisScenario,
    ParameterSynthesisScenarioResult,
)
from funman.utils.handlers import ResultCombinedHandler

from pysmt.shortcuts import (
    GE,
    GT,
    LE,
    LT,
    REAL,
    TRUE,
    And,
    Equals,
    Minus,
    Or,
    Real,
    Symbol,
    Times,
)

RESOURCES = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../resources"
)


class TestBucky(unittest.TestCase):
    def make_monotonic_constraints(self, values, steps):
        constraints = []
        for (name, direction) in values:
            for i in range(steps):
                vi = Symbol(f"{name}_{i}", REAL)
                vj = Symbol(f"{name}_{i+1}", REAL)
                if direction == "increase":
                    constraint = LT(vi, vj)
                else:
                    constraint = LT(vj, vi)
                constraints.append(constraint)
        return And(constraints)

    def setup_use_case_bilayer_common(self, config):
        bilayer_path = os.path.join(
            RESOURCES,
            "bilayer",
            "Bucky_SEIIIRRD_BiLayer_v3.json"
        )
        with open(bilayer_path, "r") as f:
            bilayer_src = json.load(f)

        infected_threshold = 5
        init_values = {
            "S": 99.43,
            "E": 0.4,
            "I_asym": 0.1,
            "I_mild": 0.05,
            "I_crit": 0.02,
            "R": 0,
            "R_hosp": 0,
            "D": 0,
        }

        extra_constraints = And(
            self.make_monotonic_constraints([
                ("S", "decrease"),
                ("R", "increase"),
                ("R_hosp", "increase")
            ], config.num_steps),
            GE(Symbol(f"S_{config.num_steps}", REAL), Real(0)),
            GE(Symbol(f"E_{config.num_steps}", REAL), Real(0)),
            GE(Symbol(f"I_asym_{config.num_steps}", REAL), Real(0)),
            GE(Symbol(f"I_mild_{config.num_steps}", REAL), Real(0)),
            GE(Symbol(f"I_crit_{config.num_steps}", REAL), Real(0)),
            GE(Symbol(f"R_{config.num_steps}", REAL), Real(0)),
            GE(Symbol(f"R_hosp_{config.num_steps}", REAL), Real(0)),
            GE(Symbol(f"D_{config.num_steps}", REAL), Real(0)),
        )

        identical_parameters = [
            ["beta_1", "beta_2"],
            ["gamma_1", "gamma_2"],
        ]

        beta = [0.24, 0.26]
        gamma = [0.469, 0.471]
        gamma_h = [0.3648, 0.3648]
        delta_1 = [0.24, 0.26]
        delta_2 = [0.009, 0.01]
        delta_3 = [0.001, 0.002]
        delta_4 = [0.09, 0.1]
        sigma = [0.016, 0.018]
        theta = [0.1012, 0.1012]

        model = BilayerModel(
            bilayer=BilayerDynamics(json_graph=bilayer_src),
            init_values=init_values,
            identical_parameters=identical_parameters,
            parameter_bounds={
                "beta_1": beta,
                "beta_2": beta,
                "gamma_1": gamma,
                "gamma_2": gamma,
                "gamma_h": gamma_h,
                "delta_1": delta_1,
                "delta_2": delta_2,
                "delta_3": delta_3,
                "delta_4": delta_4,
                "sigma": sigma,
                "theta": theta,
            },
        )

        #model._extra_constraints = extra_constraints

        query = QueryTrue()
        # query = QueryLE(variable="I_crit", ub=infected_threshold)
        return model, query

    def setup_use_case_bilayer_consistency(self, config):
        model, query = self.setup_use_case_bilayer_common(config)

        scenario = ConsistencyScenario(model=model, query=query)
        return scenario

    def setup_use_case_bilayer_parameter_synthesis(self, config):
        model, query = self.setup_use_case_bilayer_common(config)

        def make_parameter(name):
            [lb, ub] = model.parameter_bounds[name]
            return Parameter(name=name, lb=lb, ub=ub)

        scenario = ParameterSynthesisScenario(
            parameters=[
                make_parameter("beta_1"),
                #make_parameter("beta_2"),
                make_parameter("gamma_1"),
                #make_parameter("gamma_2"),
                #make_parameter("gamma_h"),
                #make_parameter("sigma"),
            ],
            model=model,
            query=query,
        )

        return scenario

    def test_use_case_bilayer_parameter_synthesis(self):
        funman = Funman()
        config = FUNMANConfig(
            tolerance=1e-8,
            num_steps=1,
            step_size=10,
            solver="dreal",
            dreal_mcts=False,
            number_of_processes=1,
            log_level=logging.INFO,
        )
        scenario = self.setup_use_case_bilayer_parameter_synthesis(config)
        # FIXME arguments with form _* do not get assigned when using pydantic
        # config._handler = ResultCombinedHandler(
        #     [
        #         ResultCacheWriter(f"bucky_box_search.json"),
        #         RealtimeResultPlotter(
        #             scenario.parameters,
        #             plot_points=True,
        #             title=f"Feasible Regions (beta)",
        #             realtime_save_path=f"bucky_box_search.png",
        #         ),
        #     ]
        # )

        result: ParameterSynthesisScenarioResult = funman.solve(
            scenario, config=config
        )
        assert len(result.parameter_space.true_boxes) > 0
        assert len(result.parameter_space.false_boxes) > 0

    def ttest_use_case_bilayer_consistency(self):
        funman = Funman()
        config = FUNMANConfig(
            tolerance=1e-8,
            #num_steps=10,
            #step_size=10,
            solver="dreal",
            dreal_mcts=False,
            log_level=logging.INFO,
        )
        scenario = self.setup_use_case_bilayer_consistency(config)

        # Show that region in parameter space is sat (i.e., there exists a true point)
        result_sat: ConsistencyScenarioResult = funman.solve(
            scenario, config=config
        )
        df = result_sat.dataframe()
        print(f"\n{df}")
        parameters = result_sat._parameters()

        # assert abs(df["I"][2] - 2.24) < 0.01
        # beta = result_sat._parameters()["beta"]
        # assert abs(beta - 0.00005) < 0.00001

        # # Show that region in parameter space is unsat/false
        # scenario.model.parameter_bounds["beta"] = [
        #     0.000067 * 1.5,
        #     0.000067 * 1.75,
        # ]
        # result_unsat: ConsistencyScenarioResult = funman.solve(scenario)
        # assert not result_unsat.consistent


if __name__ == "__main__":
    unittest.main()
