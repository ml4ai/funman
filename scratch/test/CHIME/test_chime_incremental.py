import math
import os
import unittest
from cmath import inf
from contextlib import contextmanager
from timeit import default_timer

from funman_demo.example.chime import CHIME
from pysmt.logics import QF_LRA, QF_NRA, QF_UFLIRA, QF_UFNRA
from pysmt.shortcuts import (
    BOOL,
    TRUE,
    And,
    Equals,
    Implies,
    Not,
    Real,
    Solver,
    Symbol,
    get_env,
    get_model,
    is_sat,
)
from pysmt.shortcuts import reset_env as pysmt_reset_env
from pysmt.shortcuts import substitute, write_smtlib
from pysmt.simplifier import BddSimplifier
from pysmt.typing import BOOL, INT, REAL

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
                    if not solver.solve(assumptions=step_assumptions):
                        raise Exception("unsat")
                    # model = solver.get_model()
                    # model_assertions = [
                    #     (Equals(v, model[v]))
                    #     for v in assumption_steps[step].get_free_variables()
                    #     if not model[v].is_bool_constant()
                    # ]
                    # for a in model_assertions:
                    #     if a:
                    #         solver.add_assertion(a)

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
                print(f"{step}/{len(formulas)}")
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
                        ): env.formula_manager.normalize(
                            Real(
                                self.approx_rational_helper(
                                    model.get_value(var)
                                )
                            )
                        )
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

    def approx_rational_helper(self, x, max_denominator=1e6):
        float_value = float(x.constant_value())
        lhs = math.floor(float_value)
        rhs = float_value - lhs
        (numerator, denominator) = self.approx_rational(
            rhs, max_denominator=max_denominator
        )
        numerator = numerator + (denominator * lhs)
        return (numerator, denominator)

    def approx_rational(self, x, max_denominator=1e6):
        a, b = 0, 1
        c, d = 1, 1
        while b <= max_denominator and d <= max_denominator:
            mediant = float(a + c) / (b + d)
            if x == mediant:
                if b + d <= max_denominator:
                    return a + c, b + d
                elif d > b:
                    return c, d
                else:
                    return a, b
            elif x > mediant:
                a, b = a + c, b + d
            else:
                c, d = a + c, b + d

        if b > max_denominator:
            return c, d
        else:
            return a, b

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

                    if not solver.solve():
                        raise Exception("unsat")

                if not solver.solve():
                    raise Exception("unsat")
                # model = solver.get_model()
                model = None
                solver.exit()
            elapsed = elapsed()
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

    @unittest.skip("Time consuming ...")
    def test_simple_chime_propositional(self):
        min_num_timepoints = 30
        max_num_timepoints = 30

        solver_idx = 1
        solver_names = ["msat", "z3", "cvc"]
        solver_options = {
            "z3": {
                "nlsat.check_lemmas": True,
                # "precision": 10,
                # "push_to_real": True,
                # "elim_to_real": True,
                # "algebraic_number_evaluator": False
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
        vars, (parameters, init, dynamics, query) = chime.make_model(
            assign_betas=False
        )
        for num_timepoints in range(
            min_num_timepoints, max_num_timepoints + 1
        ):
            reset_env()
            phi = chime.encode_time_horizon(
                parameters, init, dynamics, query, num_timepoints
            )
            write_smtlib(And(phi), f"chime_flat_{num_timepoints}.smt2")
            model, elapsed = self.run_get_model(
                phi,
                solver_name=solver_names[solver_idx],
                logic=QF_UFLIRA,
                solver_options=solver_options[solver_names[solver_idx]],
            )
            elapsed = 0

            phi_stratified = chime.encode_time_horizon_layered(
                parameters, init, dynamics, query, num_timepoints
            )

            reset_env()
            asm_model, asm_elapsed = self.run_assumption_solver(
                phi_stratified,
                solver_name=solver_names[solver_idx],
                logic=QF_UFLIRA,
                solver_options=solver_options[solver_names[solver_idx]],
            )

            # reset_env()
            # asm_model, asm_elapsed = self.run_decomposed_solver(
            #     phi_stratified,
            #     solver_name=solver_names[solver_idx],
            #     logic=QF_UFLIRA,
            #     solver_options=solver_options[solver_names[solver_idx]],
            # )
            asm_elapsed = 0
            # reset_env()
            # inc_model, inc_elapsed = self.run_incremental_solver(
            #     phi_stratified,
            #     solver_name=solver_names[solver_idx],
            #     logic=QF_UFLIRA,
            #     solver_options=solver_options[solver_names[solver_idx]],
            # )

            # reset_env()
            phi_stratified = chime.encode_time_horizon_layered(
                parameters, init, dynamics, query, num_timepoints
            )
            # asm_model, asm_elapsed = self.run_decomposed_solver(
            #     phi_stratified,
            #     solver_name=solver_names[solver_idx],
            #     logic=QF_UFLIRA,
            #     solver_options=solver_options[solver_names[solver_idx]],
            # )
            asm_elapsed = 0
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
