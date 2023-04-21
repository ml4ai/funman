Use Cases

.. _test decapode: https://github.com/ml4ai/funman/tree/main/test/test_decapode.py

The following use cases reside in `test decapode`_.  
=========

Check Consistency of DECAPODE Model:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This use case involves FUNMAN reasoning about the hydrostatic equation within the TIE-GCM space weather model, represented as a DECAPODE model file.  It checks whether the DECAPODE model is consistent with some expected output behavior (e.g., from a reliable data source or simulation).
It first constructs an instance of the DecapodeModel class using the provided DECAPODE file.  This class constructs a model from the DECAPODE file that can be asked to answer a query with that model.  In the example, the provided DECAPODE file corresponds to the hydrostatic equation within the TIE-GCM model.  The query asks whether there are values of the parameters m_bar, step_size, and num_steps such that the geopotential does not exceed a specified threshold.  The test will succeed if there are satisfying values.  This Consistency use case is set up similarly to the Projection use case below.

Projection (find outputs for given input) of DECAPODE Model:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This use case involves FUNMAN reasoning about the hydrostatic equation within the TIE-GCM space weather model, represented as a DECAPODE model file.  It checks the output of the model given some input parameter values. 
It first constructs an instance of the DecapodeModel class using the provided DECAPODE file.  This class constructs a model from the DECAPODE file that can be asked to answer a query with that model.  In the example, the provided DECAPODE file corresponds to the hydrostatic equation within the TIE-GCM model.  The query asks to find the expected geopotential value given the fixed parameter values of m_bar, step_size, and num_steps.  The test will return the expected simulation values for the geopotential.


.. code-block:: py
    def test_use_case_decapode_consistency(self):
        """
        Check that for a given m_bar, that the geopotential at z= 500mb is a given constant H500.
        Consistency: assert |H(z=1000) - H0| <= epsilon in formulation and test whether its consistent.
                Test: is satisfiable
        Projection: for m-bar = m0, calculate H(z=1000)
            Test: |H(z=1000) - H0| <= epsilon


        query = QueryAnd(queries=[QueryLE(variable="H", ub=H0+epsilon, at_end=True), QueryGE(variable="H", lb=H0-epsilon, at_end=True)]), requires that last value of z is 1000.

        """
        scenario = self.setup_use_case_decapode_consistency()

        funman = Funman()

        # Show that region in parameter space is sat (i.e., there exists a true point)
        try:
            result_sat: ConsistencyScenarioResult = funman.solve(scenario)

            df = result_sat.dataframe()


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


Regression (find inputs for given output) of DECAPODE Model:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


This use case involves FUNMAN reasoning about the hydrostatic equation within the TIE-GCM space weather model, represented as a DECAPODE model file.  It checks whether the DECAPODE model is consistent with some expected output behavior (e.g., from a reliable data source or simulation).  If it is, the query will return all parameter values that allow the model to be consistent.
It first constructs an instance of the DecapodeModel class using the provided DECAPODE file.  This class constructs a model from the DECAPODE file that can be asked to answer a query with that model.  In the example, the provided DECAPODE file corresponds to the hydrostatic equation within the TIE-GCM model.  The query asks to find all values of m_bar, step_size, and num_steps such that the geopotential does not exceed a specified threshold.  The test will return ranges and point values of m_bar, num_steps, and step_size that jointly satisfy the query.


.. code-block:: py
    def test_use_case_decapode_parameter_synthesis(self):
        """
        Use case for Parameter Synthesis.
        Regression: find m-bar values that set H(z=1000) = 500mb
                    Test: m0 is in ps(m-bar).true

        Sensitivity: Variance in H(z)=500mb due to m-bar
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
            assert len(result.parameter_space.true_boxes) > 0
            assert len(result.parameter_space.false_boxes) > 0

            # Analysis of Parameter Synthesis:
            # Grid sampling over m-bar and calculate the altitude (z) at which geopotential is 500mb.  Report the variance over Var(H(z| z=500mb, m-bar)).  How sensitive is the altitude of a reference geopotential to the choice of m-bar?
        except Exception as e:
            print(f"Could not solve scenario because: {e}")
            assert False


Sensitivity Analysis (capture the relationship between input and output behavior) of DECAPODE Model:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This use case involves FUNMAN reasoning about the hydrostatic equation within the TIE-GCM space weather model, represented as a DECAPODE model file.  It finds information about the relationship between the inputs and outputs of the model.
It first constructs an instance of the DecapodeModel class using the provided DECAPODE file.  This class constructs a model from the DECAPODE file that can be asked to answer a query with that model.  In the example, the provided DECAPODE file corresponds to the hydrostatic equation within the TIE-GCM model.  The query fixes the parameters step_size and num_steps, then gives a range around a specified value of the parameter m_bar.  The test will return the range of the output values for the geopotential.  By comparing the results of this test to those of the projection test above, we can see how perturbations in the parameter values can impact the output.
This use case follows the same initial setup as the Regression use case above, but is followed by an analysis of how sensitive the geopotential is to the parameter m_bar.




