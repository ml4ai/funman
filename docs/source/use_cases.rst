Use Cases
=========

Compare Translated FN to Simulator:
-----------------------------------

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
        q_sim = query_simulator(run_CHIME_SIR, does_not_cross_threshold)

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
        q_sim = query_simulator(run_CHIME_SIR_BL, does_not_cross_threshold)
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