Use Cases
=========

.. _test use cases: https://github.com/ml4ai/funman/tree/main/test/test_use_cases.py

The following use cases reside in `test use cases`_.  The use cases listed under :ref:`Future Cases (below) <future-cases>` are use cases identified in previous versions that are work in progress.

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


Compare Bi-Layer Model to Bi-Layer Simulator:
---------------------------------------------

This use case compares the simulator and FUNMAN reasoning about the CHIME
SIR model.

It first runs the query_simulator function which executes the input simulator
function and evaluates the input query function using the simulation results.
In the example below this results in the run_CHIME_SIR_BL simulator function and
evaluating whether or not the number of infected crosses the provided threshold.

It then constructs an instance of the QueryableBilayer class using the provided
bilayer file. This class constructs a model from the bilayer file that can be
asked to answer a query with that model. In the example below the provided
bilayer file corresponds to the CHIME_SIR simulator. The query asks whether the
number of infected at any time point exceeds a specified threshold.

Once each of these steps is executed the results are compared. The test will
succeed if the simulator and QueryableBilayer class agree on the response to the
query.

.. code-block::

    def compare_against_CHIME_bilayer(bilayer_file, infected_threshold):
        # query the simulator
        def does_not_cross_threshold(sim_results):
            i = sim_results[1]
            return (i <= infected_threshold)
        q_sim = does_not_cross_threshold(run_CHIME_SIR_BL())
        print("q_sim", q_sim)

        # query the bilayer file
        bilayer = QueryableBilayer.from_file(bilayer_file)
        q_bilayer = bilayer.query(f"(i <= infected_threshold)")
        print("q_bilayer", q_bilayer)

        # assert the both queries returned the same result
        return  q_sim == q_bilayer

    # example call
    bilayer_file = "CHIME_SIR_dynamics_BiLayer.json"
    infected_threshold = 130
    assert compare_against_CHIME_bilayer(bilayer_file, infected_threshold)

