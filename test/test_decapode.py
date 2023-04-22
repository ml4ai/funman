import json
import os
import unittest

from funman_demo.handlers import RealtimeResultPlotter, ResultCacheWriter
from pysmt.shortcuts import GE, LE, And, Real, Symbol
from pysmt.typing import REAL

from funman import Funman
from funman.funman import FUNMANConfig
from funman.model import QueryLE
from funman.model.decapode import DecapodeDynamics, DecapodeModel
from funman.representation.representation import Parameter
from funman.scenario import (
    ConsistencyScenario,
    ConsistencyScenarioResult,
    ParameterSynthesisScenario,
    ParameterSynthesisScenarioResult,
)
from funman.utils.handlers import ResultCombinedHandler

RESOURCES = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../resources"
)


class TestUseCases(unittest.TestCase):
    def setup_use_case_decapode_common(self):
        """
        Setup a Decapode model that has parameters:
        R^Mo(Other("*")): gas constant
        m_Mo(Other("‾")): mean molecular mass
        T_n: global mean reference temperature

        The query states that the geopotential does not exceed a threshold over the entire range of altitude z.

        Returns
        -------
        DecapodeModel
            Model of H(z), where H(0) = 1
        Query
            for all z. H(z) <= geopotential_threshold (5000)
        """
        # read in the decapode file
        decapode_path = os.path.join(
            RESOURCES, "decapode", "hydrostatic_3_6.json"
        )
        with open(decapode_path, "r") as f:
            decapode_src = json.load(f)
        # set a threshold geopotential to be used in query
        geopotential_threshold = 5000
        # set boundary condition
        init_values = {"H": 1}

        # set range of parameter space to search over
        scale_factor = 0.75
        lb = 1.67e-27 * (1 - scale_factor)
        ub = 1.67e-27 * (1 + scale_factor)

        # create instance of decapode model that can be queried
        model = DecapodeModel(
            decapode=DecapodeDynamics(json_graph=decapode_src),
            init_values=init_values,
            parameter_bounds={
                'R^Mo(Other("*"))': [8.31, 8.31],
                "T_n": [286, 286],
                'm_Mo(Other("‾"))': [lb, ub],
                "g": [9.8, 9.8],
            },
        )
        # set up query that checks whether the geopotential stays under a given threshold
        query = QueryLE(variable="H", ub=geopotential_threshold)

        return model, query

    def setup_use_case_decapode_parameter_synthesis(self):
        """
        Create a ParameterSynthesisScenario to compute values of m_bar that satisfy the query.

        Returns
        -------
        _type_
            _description_
        """
        model, query = self.setup_use_case_decapode_common()
        [lb, ub] = model.parameter_bounds['m_Mo(Other("‾"))']
        scenario = ParameterSynthesisScenario(
            parameters=[Parameter(name='m_Mo(Other("‾"))', lb=lb, ub=ub)],
            model=model,
            query=query,
        )

        return scenario

    @unittest.expectedFailure
    def test_use_case_decapode_parameter_synthesis(self):
        """
        Use case for Parameter Synthesis.
        Case 2:  Regression: find m-bar values that set H(z=1000) = 500mb
                    Test: m0 is in ps(m-bar).true

        Case 4:  Sensitivity: Variance in H(z)=500mb due to m-bar
                    Test: | Var(H(z)|z=500mb) - V0 | <= epsilon
        """
        try:
            scenario = self.setup_use_case_decapode_parameter_synthesis()
            funman = Funman()
            result: ParameterSynthesisScenarioResult = funman.solve(
                scenario,
                config=FUNMANConfig(
                    tolerance=1e-8,
                    number_of_processes=1,
                    _handler=ResultCombinedHandler(
                        [
                            ResultCacheWriter(f"box_search.json"),
                            RealtimeResultPlotter(
                                scenario.parameters,
                                plot_points=True,
                                title=f"Feasible Regions (beta)",
                                realtime_save_path=f"box_search.png",
                            ),
                        ]
                    ),
                ),
            )
            # Finding true and false boxes for Regression use case
            assert len(result.parameter_space.true_boxes) > 0
            assert len(result.parameter_space.false_boxes) > 0

            # Analysis of Parameter Synthesis: Sensitivity Analysis Use Case
            # Grid sampling over m-bar and calculate the altitude (z) at which geopotential is 500mb.  Report the variance over Var(H(z| z=500mb, m-bar)).  How sensitive is the altitude of a reference geopotential to the choice of m-bar?
            # Find variance of true boxes
            sensitivity_analysis_result = variance(
                result.parameter_space.true_boxes
            )
        except Exception as e:
            print(f"Could not solve scenario because: {e}")
            assert False

    def setup_use_case_decapode_consistency(self):
        model, query = self.setup_use_case_decapode_common()

        scenario = ConsistencyScenario(model=model, query=query)
        return scenario

    @unittest.expectedFailure
    def test_use_case_decapode_consistency(self):
        """
        Check that for a given m_bar, that the geopotential at z= 500mb is a given constant H500.
        Case 1: Consistency: assert |H(z=1000) - H0| <= epsilon in formulation and test whether its consistent.
                Test: is satisfiable
        Case 3:  Projection: for m-bar = m0, calculate H(z=1000)
            Test: |H(z=1000) - H0| <= epsilon


        query = QueryAnd(queries=[QueryLE(variable="H", ub=H0+epsilon, at_end=True), QueryGE(variable="H", lb=H0-epsilon, at_end=True)]), requires that last value of z is 1000.

        """
        scenario = self.setup_use_case_decapode_consistency()

        funman = Funman()

        # Show that region in parameter space is sat (i.e., there exists a true point)
        try:
            result_sat: ConsistencyScenarioResult = funman.solve(scenario)

            df = result_sat.dataframe()

            assert abs(df["H"][-1] - 500) < epsilon

            # Show that region in parameter space is unsat/false
            scenario.model.parameter_bounds['m_Mo(Other("‾"))'] = [
                1.67e-27 * 1.5,
                1.67e-27 * 1.75,
            ]
            result_unsat: ConsistencyScenarioResult = funman.solve(scenario)
            assert not result_unsat.consistent
        except Exception as e:
            print(f"Could not solve scenario because: {e}")
            assert False


if __name__ == "__main__":
    unittest.main()
