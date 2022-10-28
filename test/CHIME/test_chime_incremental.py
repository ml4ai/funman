from cmath import inf
from contextlib import contextmanager
from dis import dis
from pysmt.simplifier import BddSimplifier
from pysmt.shortcuts import (
    get_model,
    And,
    Solver,
    TRUE,
    reset_env as pysmt_reset_env,
    is_sat,
    Symbol,
    BOOL,
    Implies,
    Not,
    Equals,
    substitute,
    get_env,
    TRUE,
)
from pysmt.typing import INT, REAL, BOOL
from pysmt.logics import QF_NRA, QF_LRA, QF_UFLIRA, QF_UFNRA


import unittest
import os
from funman.examples.chime import CHIME
from timeit import default_timer

RESOURCES = os.path.join("resources")


@contextmanager
def elapsed_timer():
    start = default_timer()
    elapser = lambda: default_timer() - start
    try:
        yield elapser
    finally:
        elapser = None


def reset_env():
    env = pysmt_reset_env()
    env.enable_infix_notation = True
    return env


class TestHandcoded(unittest.TestCase):
    def run_assumption_solver(
        self, formulas, solver_name=None, logic=None, solver_options=None
    ):
        # with steps
        with elapsed_timer() as elapsed:
            with Solver(
                name=solver_name, logic=logic, solver_options=solver_options
            ) as solver:
                assumptions = [
                    Symbol(f"time_{t}", BOOL) for t in range(len(formulas))
                ]
                assumption_steps = [
                    Implies(assumption, formula)
                    for (assumption, formula) in zip(assumptions, formulas)
                ]

                # solver.add_assertion(phi)

                past_assumptions = []
                for step in range(len(assumptions)):
                    # print(f"Step: {step}/{len(assumptions)-1}")
                    solver.add_assertion(assumption_steps[step])
                    current_step_assumption = [assumptions[step]]
                    past_step_assumptions = [
                        a for a in assumptions[:step]
                    ]  # [Not(a) for a in assumptions[:step]]
                    step_assumptions = (
                        current_step_assumption + past_step_assumptions
                    )
                    # enabled_steps = [a for a in assumptions[: step + 1]]
                    # disabled_steps = [Not(a) for a in assumptions[step + 1 :]]
                    # step_assumptions = enabled_steps  # + disabled_steps
                    if not solver.solve(assumptions=step_assumptions):
                        raise Exception("unsat")
                    model = solver.get_model()
                    # solver.add_assertion(assumptions[step])
                    model_assertions = [
                        (Equals(v, model[v]))
                        for v in assumption_steps[step].get_free_variables()
                        if not model[v].is_bool_constant()
                    ]
                    for a in model_assertions:
                        if a:
                            solver.add_assertion(a)
                            # past_assumptions.append(a)
                    # print(elapsed())
                    pass
                # if not solver.solve():
                #     raise Exception("unsat")
                # model = solver.get_model()
                model = None
                elapsed = elapsed()
                solver.exit()
        return model, elapsed

    def run_decomposed_solver(
        self, formulas, solver_name=None, logic=None, solver_options=None
    ):
        past_assignments = {}
        env = get_env()

        with elapsed_timer() as elapsed:
            # with Solver(
            #     name=solver_name, logic=logic, solver_options=solver_options
            # ) as solver:
            for step in range(len(formulas)):
                # with elapsed_timer() as elapsed:
                # with Solver(
                #     name=solver_name, logic=logic, solver_options=solver_options
                # ) as solver:
                relevant_vars = set(
                    map(
                        lambda x: env.formula_manager.normalize(x),
                        formulas[step].get_free_variables(),
                    )
                )
                past_relevant_vars = {
                    k: v
                    for k, v in past_assignments.items()
                    if k in relevant_vars
                }

                # past_relevant_var_ids = {k: k._node_id for k, v in past_assignments.items() if k in relevant_vars}
                # encoding_var_ids = {v: v._node_id for formula_step in formulas for v in formula_step.get_free_variables()}
                # pre_env_var_ids = {k: v._node_id for k, v in env.formula_manager.symbols.items()}

                # env used by substitute does not have the past_relevant_vars defined
                # need to add the vars to ensure that the assignments can be made
                step_formula = env.substituter.substitute(
                    env.formula_manager.normalize(formulas[step]),
                    past_relevant_vars,
                ).simplify()
                # past_assignment_fn = And([Equals(k, v) for k, v in past_assignments.items()])
                # sub_env_var_ids = {k: v._node_id for k, v in env.formula_manager.symbols.items()}

                # solver.add_assertion(step_formula)
                # if not solver.solve():
                #         raise Exception("unsat")
                # model = solver.get_model()
                # model = get_model(And(step_formula, past_assignment_fn))
                model = None
                if step_formula != TRUE():
                    model = get_model(step_formula, solver_name=solver_name)
                    if not model:
                        raise Exception("unsat")

                # post_env_var_ids = {k: v._node_id for k, v in env.formula_manager.symbols.items()}
                # model_var_ids = {k: v._node_id for k, v in model.environment.formula_manager.symbols.items()}

                if model:
                    model_assertions = {
                        env.formula_manager.normalize(
                            var
                        ): env.formula_manager.normalize(model.get_value(var))
                        # env.formula_manager.normalize(var): env.formula_manager.normalize(val)
                        # for var, val in model
                        for var in step_formula.get_free_variables()
                        if not var.is_bool_constant()
                        and not var in past_relevant_vars
                    }
                    # model_assn_var_ids = {k: k._node_id for k in model_assertions.keys()}
                    past_assignments.update(model_assertions)
                    model = None
        elapsed = elapsed()
        # solver.exit()
        return model, elapsed

    def run_incremental_solver(
        self, formulas, solver_name=None, logic=None, solver_options=None
    ):
        # with steps
        with elapsed_timer() as elapsed:
            with Solver(
                name=solver_name, logic=logic, solver_options=solver_options
            ) as solver:
                for i, phi in enumerate(formulas):
                    if i > 0:
                        solver.push()
                    solver.add_assertion(phi)

                    # if not solver.solve():
                    #     raise Exception("unsat")

                if not solver.solve():
                    raise Exception("unsat")
                # model = solver.get_model()
                model = None
                elapsed = elapsed()
                solver.exit()
        return model, elapsed

    def run_get_model(
        self, formulas, solver_name=None, logic=None, solver_options=None
    ):
        # just model
        with elapsed_timer() as elapsed:
            model = get_model(
                And(formulas),
                solver_name=solver_name,
                logic=logic,
                # solver_options=solver_options,
            )
            elapsed = elapsed()
        return model, elapsed

    def test_simple_chime_propositional(self):

        min_num_timepoints = 60
        max_num_timepoints = 61

        solver_idx = 1
        solver_names = ["msat", "z3", "cvc"]
        solver_options = {
            "z3": {
                "nlsat.check_lemmas": True,
                # "precision": 10,
                "push_to_real": True,
                "elim_to_real": True,
                "algebraic_number_evaluator": False
                # "dot_proof_file": "z3_proof.dot"
                # "add_bound_upper": 1010,
                # "add_bound_lower": 0,
                # "propagate_values.max_rounds": 10
                # "arith.nl.nra": True,
                # "arith.nl.grobner": False,
                # "arith.nl.horner": False,
                # "arith.nl.order": False,
                # "arith.nl.tangents": False,
                # "threads": 2
            },
            "msat": {},
            "cvc": {},
        }

        # query = CHIME.make_chime_query(infected, num_timepoints)
        print(f"steps\torig\tinc")
        chime = CHIME()
        vars, (parameters, init, dynamics, query) = chime.make_model()
        for num_timepoints in range(min_num_timepoints, max_num_timepoints):

            reset_env()
            phi = chime.encode_time_horizon(
                parameters, init, dynamics, query, num_timepoints
            )
            model, elapsed = self.run_get_model(
                phi,
                solver_name="z3",
                logic=QF_UFLIRA,
                solver_options=solver_options,
            )
            elapsed = 0
            # reset_env()
            # asm_model, asm_elapsed = self.run_assumption_solver(
            #     phi_stratified,
            #     solver_name="z3",
            #     logic=QF_UFLIRA,
            #     solver_options=solver_options,
            # )

            reset_env()
            phi_stratified = chime.encode_time_horizon_layered(
                parameters, init, dynamics, query, num_timepoints
            )
            asm_model, asm_elapsed = self.run_decomposed_solver(
                phi_stratified,
                solver_name=solver_names[solver_idx],
                logic=QF_UFLIRA,
                solver_options=solver_options[solver_names[solver_idx]],
            )
            reset_env()
            inc_model, inc_elapsed = self.run_incremental_solver(
                phi_stratified,
                solver_name=solver_names[solver_idx],
                logic=QF_UFLIRA,
                solver_options=solver_options[solver_names[solver_idx]],
            )
            inc_elapsed = 0
            print(
                f"{num_timepoints}\t{float('{:.2f}'.format(elapsed))}\t{float('{:.2f}'.format(inc_elapsed))}\t{float('{:.2f}'.format(asm_elapsed))}"
            )


if __name__ == "__main__":
    unittest.main()
