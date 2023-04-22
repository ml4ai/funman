"""
The tests in this module represent use cases for space weather model analysis with FUNMAN.  The test class includes common configuration functions and four test cases, as described in: https://ml4ai.github.io/funman/sw_use_cases.html
"""

import json
import os
import unittest

from funman_demo.handlers import RealtimeResultPlotter, ResultCacheWriter

from funman import Funman
from funman.funman import FUNMANConfig
from funman.model import QueryLE
from funman.model.decapode import DecapodeDynamics, DecapodeModel
from funman.model.query import Query, QueryAnd, QueryGE, QueryTrue
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

GEOPOTENTIAL_THRESHOLD = 500


class TestUseCases(unittest.TestCase):
    def setup_use_case_decapode_common(self):
        """
        Setup a Decapode model that has treats the constant m_Mo(Other("‾")) (i.e., the mean molecular mass) as a parameter.  The model uses the structural_parameter_bounds to define a number of steps and step size so that the extreme value of the z variable is 1000, where queries will evaluate H(z).

        Returns
        -------
        DecapodeModel
            Model of H(z), with boundary H(0) = 1
        """
        # read in the decapode file
        decapode_path = os.path.join(
            RESOURCES, "decapode", "hydrostatic_3_6.json"
        )
        with open(decapode_path, "r") as f:
            decapode_src = json.load(f)

        # set range of parameter space to search over
        scale_factor = 0.75
        lb = 1.67e-27 * (1 - scale_factor)
        ub = 1.67e-27 * (1 + scale_factor)

        # create instance of decapode model that can be queried
        model = DecapodeModel(
            decapode=DecapodeDynamics(json_graph=decapode_src),
            init_values={"H": 1},  # set boundary condition
            parameter_bounds={
                'R^Mo(Other("*"))': [8.31, 8.31],
                "T_n": [286, 286],
                'm_Mo(Other("‾"))': [lb, ub],
                "g": [9.8, 9.8],
            },
            structural_parameter_bounds={
                "num_steps": [100, 100],
                "step_size": [10, 10],
            },
        )

        return model

    def setup_use_case_decapode_parameter_synthesis(self, query: Query):
        """
        Create a ParameterSynthesisScenario to compute values of m_Mo(Other("‾")) that satisfy the query.

        Returns
        -------
        ParameterSynthesisScenario
            test case scenario definition
        """
        model = self.setup_use_case_decapode_common()
        [lb, ub] = model.parameter_bounds['m_Mo(Other("‾"))']
        scenario = ParameterSynthesisScenario(
            parameters=[Parameter(name='m_Mo(Other("‾"))', lb=lb, ub=ub)],
            model=model,
            query=query,
        )

        return scenario

    @unittest.expectedFailure
    def test_use_case_decapode_sensitivity_analysis(self):
        """
        Use case for Regression with Parameter Synthesis. Find the values for mean molecular mass where the geopotential is 500mb at an altitude of 100 (i.e.  H(z=1000) = 500mb)
        """
        try:
            query = QueryAnd(
                QueryEquals("H", GEOPOTENTIAL_THRESHOLD, at_end=True)
            )
            scenario = self.setup_use_case_decapode_parameter_synthesis(query)
            result: ParameterSynthesisScenarioResult = Funman().solve(
                scenario, config=FUNMANConfig(number_of_processes=1)
            )

            assert len(result.parameter_space.true_boxes) > 0

            print(
                f"The geopotential will be 500mb at an alitude of 1000m if the mean molecular mass is in the intervals: {result.parameter_space.true_boxes}"
            )

        except Exception as e:
            print(f"Could not solve scenario because: {e}")
            assert False

    @unittest.expectedFailure
    def test_use_case_decapode_sensitivity_analysis(self):
        """
        Use case for Sensitivity Analysis with Parameter Synthesis. Find the variance in geopotential over feasible values for the mean molecular mass.
        """
        try:
            scenario = self.setup_use_case_decapode_parameter_synthesis(
                QueryTrue()
            )
            result: ParameterSynthesisScenarioResult = Funman().solve(
                scenario, config=FUNMANConfig(number_of_processes=1)
            )

            assert len(result.parameter_space.true_boxes) > 0

            # Extract several point values for the mean molecular mass that are feasible
            points = result.parameter_space.sample_true_boxes()

            # Calculate the distribution of geopotential H over altitude z for each point
            dataframe = result.true_point_timeseries(points)

            # Calculate the variance at an altitude of 1000m
            sensitivity = dataframe.loc[dataframe.z == 1000].var()

            print(
                f"The variance geopotential at an alitude of 1000m due to the mean molecular mass is: {sensitivity.H}"
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
