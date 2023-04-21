Use Cases

.. _test decapode: https://github.com/ml4ai/funman/tree/main/test/test_decapode.py

The following use cases reside in `test decapode`_.  
=========

Check Consistency of DECAPODE Model:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This use case involves FUNMAN reasoning about the hydrostatic equation within the TIE-GCM space weather model, represented as a DECAPODE model file.  It checks whether the DECAPODE model is consistent with some expected output behavior (e.g., from a reliable data source or simulation).
It first constructs an instance of the DecapodeModel class using the provided DECAPODE file.  This class constructs a model from the DECAPODE file that can be asked to answer a query with that model.  In the example, the provided DECAPODE file corresponds to the hydrostatic equation within the TIE-GCM model.  The query asks whether there are values of the parameters m_bar, step_size, and num_steps such that the geopotential does not exceed a specified threshold.  The test will succeed if there are satisfying values.  This consistency use case is set up similarly to the Projection use case below.

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




Compare Bilayer Model to Simulator:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This use case involves the simulator and FUNMAN reasoning about the CHIME
SIR bilayer model.  See test `test_use_case_bilayer_consistency` in the `test use cases`_.

It first uses a SimulationScenario to execute the input simulator
function and evaluate the input query function using the simulation results.
In the example below this results in the run_CHIME_SIR simulator function and
evaluating whether or not the number of infected crosses the provided threshold with a custom QueryFunction referencing the `does_not_cross_threshold` function.

It then constructs an instance of the ConsistencyScenario class to evaluate whether a BilayerModel will satisfy the given query. The query asks whether the
number of infected at any time point exceeds a specified threshold.

Once each of these steps is executed the results are compared. The test will
succeed if the SimulatorScenario and ConsistencyScenario agree on the response to the query.

.. code-block:: py

    def compare_against_CHIME_Sim(
        self, bilayer_path, init_values, infected_threshold
    ):
        # query the simulator
        def does_not_cross_threshold(sim_results):
            i = sim_results[2]
            return all(i_t <= infected_threshold for i_t in i)

        query = QueryLE("I", infected_threshold)

        funman = Funman()

        sim_result: SimulationScenarioResult = funman.solve(
            SimulationScenario(
                model=SimulatorModel(run_CHIME_SIR),
                query=QueryFunction(does_not_cross_threshold),
            )
        )

        consistency_result: ConsistencyScenarioResult = funman.solve(
            ConsistencyScenario(
                model=BilayerModel(
                    BilayerDynamics.from_json(bilayer_path),
                    init_values=init_values,
                ),
                query=query,
            )
        )

        # assert the both queries returned the same result
        return sim_result.query_satisfied == consistency_result.query_satisfied

    def test_use_case_bilayer_consistency(self):
        """
        This test compares a BilayerModel against a SimulatorModel to
        determine whether their response to a query is identical.
        """
        bilayer_path = os.path.join(
            RESOURCES, "bilayer", "CHIME_SIR_dynamics_BiLayer.json"
        )
        infected_threshold = 130
        init_values = {"S": 9998, "I": 1, "R": 1}
        assert self.compare_against_CHIME_Sim(
            bilayer_path, init_values, infected_threshold
        )

Parameter Synthesis
-------------------

See tests `test_use_case_simple_parameter_synthesis` and `test_use_case_bilayer_parameter_synthesis` in the `test use cases`_.

The base set of types used during Parameter Synthesis include:

- a list of Parameters representing variables to be assigned
- a Model to be encoded as an SMTLib formula 
- a Scenario container representing a set of parameters and model
- a SearchConfig to configure search behavior
- the Funman interface that runs analysis using scenarios and configuration data

In the following example two parameters, x and y, are constructed. A model is 
also constructed that says 0.0 < x < 5.0 and 10.0 < y < 12.0. These parameters
and model are used to define a scenario that will use BoxSearch to synthesize
the parameters. The Funman interface and a search configuration are also 
defined. All that remains is to have Funman solve the scenario using the defined
configuration.

.. code-block:: py
    
    def test_use_case_simple_parameter_synthesis(self):
        x = Symbol("x", REAL)
        y = Symbol("y", REAL)

        formula = And(
            LE(x, Real(5.0)),
            GE(x, Real(0.0)),
            LE(y, Real(12.0)),
            GE(y, Real(10.0)),
        )

        funman = Funman()
        result: ParameterSynthesisScenarioResult = funman.solve(
            ParameterSynthesisScenario(
                [
                    Parameter(name="x", symbol=x),
                    Parameter(name="y", symbol=y),
                ],
                EncodedModel(formula),
            )
        )
        assert result

As an additional parameter synthesis example, the following test case demonstrates how to perform parameter synthesis for a bilayer model.  The configuration differs from the example above by introducing bilayer-specific constraints on the initial conditions (`init_values` assignments), parameter bounds (`parameter_bounds` intervals) and a model query.

.. code-block:: py

    def test_use_case_bilayer_parameter_synthesis(self):
        bilayer_path = os.path.join(
            RESOURCES, "bilayer", "CHIME_SIR_dynamics_BiLayer.json"
        )
        infected_threshold = 3
        init_values = {"S": 9998, "I": 1, "R": 1}

        lb = 0.000067 * (1 - 0.5)
        ub = 0.000067 * (1 + 0.5)

        funman = Funman()
        result: ParameterSynthesisScenarioResult = funman.solve(
            ParameterSynthesisScenario(
                parameters=[Parameter(name="beta", lb=lb, ub=ub)],
                model=BilayerModel(
                    BilayerDynamics.from_json(bilayer_path),
                    init_values=init_values,
                    parameter_bounds={
                        "beta": [lb, ub],
                        "gamma": [1.0 / 14.0, 1.0 / 14.0],
                    },
                ),
                query=QueryLE("I", infected_threshold),
            ),
            config=SearchConfig(tolerance=1e-8),
        )
        assert len(result.parameter_space.true_boxes) > 0 
        assert len(result.parameter_space.false_boxes) > 0 



.. _future-cases:

Future Cases
------------

Compare Translated FN to Simulator:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This use case involves the simulator and FUNMAN reasoning about the CHIME
SIR model.

It first runs the query_simulator function which executes the input simulator
function and evaluates the input query function using the simulation results.
In the example below this results in the run_CHIME_SIR simulator function and
evaluating whether or not the number of infected crosses the provided threshold.

It then constructs an instance of the QueryableGromet class using the provided
GroMEt file. This class constructs a model from the GroMEt file that can be
asked to answer a query with that model. In the example below the provided
GroMET file corresponds to the CHIME_SIR simulator. The query asks whether the
number of infected at any time point exceeds a specified threshold.

Once each of these steps is executed the results are compared. The test will
succeed if the simulator and QueryableGromet class agree on the response to the
query.

.. code-block::

    def compare_against_CHIME_FN(gromet_path, infected_threshold):
        # query the simulator
        def does_not_cross_threshold(sim_results):
            i = sim_results[2]
            return all(i_t <= infected_threshold for i_t in i)
        q_sim = does_not_cross_threshold(run_CHIME_SIR())

        # query the gromet file
        gromet = QueryableGromet.from_gromet_file(gromet_path)
        q_gromet = gromet.query(f"(forall ((t Int)) (<= (I t) {infected_threshold}))")

        # assert the both queries returned the same result
        return  q_sim == q_gromet

    # example call
    gromet = "CHIME_SIR_while_loop--Gromet-FN-auto.json"
    infected_threshold = 130
    assert compare_against_CHIME_FN(gromet, infected_threshold)


Compare Constant and Time-dependent Betas:
------------------------------------------

This use case involves two formulations of the CHIME model:
  - the original model where Beta is a epoch-dependent constant over three
    epochs (i.e., a triple of parameters)
  - a modified variant of the original model using a single constant Beta over
    the entire simulation (akin to a single epoch).

These two formulations of the CHIME model are read in from GroMEt files into
instances of the QueryableGromet class. These models are asked to synthesize a
parameter space based on a query. Note that this synthesis step is stubbed in
this example and a more representative example of parameter synthesis can be
found below. Once these parameter spaces are synthesized the example then
compares the models by determining that the respective spaces of feasible
parameter values intersect.

.. code-block::

    def test_parameter_synthesis():
    ############################ Prepare Models ####################################
    # read in the gromet files
    # GROMET_FILE_1 is the original GroMEt extracted from the simulator
    # It sets N_p = 3 and n_days = 20, resulting in three epochs of 0, 20, and 20 days
    gromet_org = QueryableGromet.from_gromet_file(GROMET_FILE_1)
    # GROMET_FILE_2 modifes sets N_p = 2 and n_days = 40, resulting in one epoch of 40 days
    gromet_sub = QueryableGromet.from_gromet_file(GROMET_FILE_2)
    # Scenario query threshold
    infected_threshold = 130

    ############################ Evaluate Models ###################################
    # some test query
    query f"(forall ((t Int)) (<= (I t) {infected_threshold}))"
    # get parameter space for the original (3 epochs)
    ps_b1_b2_b3 = gromet_org.synthesize_parameters(query)
    # get parameter space for the constant beta variant
    ps_bc = gromet_sub.synthesize_parameters(query)

    ############################ Compare Models ####################################
    # construct special parameter space where parameters are equal
    ps_eq = ParameterSpace.construct_all_equal(ps_b1_b2_b3)
    # intersect the original parameter space with the ps_eq to get the
    # valid parameter space where (b1 == b2 == b3)
    ps = ParameterSpace.intersect(ps_b1_b2_b3, ps_eq)
    # assert the parameters spaces for the original and the constant beta
    # variant are the same
    assert(ParameterSpace.compare(ps_bc, ps))
