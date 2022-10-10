from cmath import inf
from contextlib import contextmanager
from dis import dis
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
    def encode_time_horizon(self, parameters, init, dynamics, horizon):
        dynamics_t = (
            And([d_t for d_t in dynamics[0:horizon]]) if horizon > 0 else TRUE()
        )
        return And(parameters, init, dynamics_t)

    def run_assumption_solver(
        self, formulas, solver_name=None, logic=None, solver_options=None
    ):
        # with steps
        with elapsed_timer() as elapsed:
            with Solver(
                name=solver_name, logic=logic, solver_options=solver_options
            ) as solver:
                assumptions = [Symbol(f"time_{t}", BOOL) for t in range(len(formulas))]
                assumption_steps = [
                    Implies(assumption, formula)
                    for (assumption, formula) in zip(assumptions, formulas)
                ]

                # solver.add_assertion(phi)

                for step in range(len(assumptions)):
                    print(f"Step: {step}/{len(assumptions)-1}")
                    solver.add_assertion(assumption_steps[step])
                    step_assumptions = [Not(a) for a in assumptions[:step]] + [
                        assumptions[step]
                    ]
                    # enabled_steps = [a for a in assumptions[: step + 1]]
                    # disabled_steps = [Not(a) for a in assumptions[step + 1 :]]
                    # step_assumptions = enabled_steps  # + disabled_steps
                    if not solver.solve(assumptions=step_assumptions):
                        raise Exception("unsat")
                    model = solver.get_model()
                    # solver.add_assertion(assumptions[step])
                    model_assertions = [
                        (
                            Equals(v, model[v])
                            if not model[v].is_bool_constant()
                            else None
                            # else Not(
                            #     v
                            # )  # Negate the step assertion, which is the only bool (could be risky)
                        )
                        for v in assumption_steps[step].get_free_variables()
                    ]
                    for a in model_assertions:
                        if a:
                            solver.add_assertion(a)
                    # print(elapsed())

                # if not solver.solve():
                #     raise Exception("unsat")
                # model = solver.get_model()
                model = None
                elapsed = elapsed()
                solver.exit()
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

        min_num_timepoints = 1
        max_num_timepoints = 40

        solver_options = {
            # "nlsat.check_lemmas": True,
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
        }

        # query = CHIME.make_chime_query(infected, num_timepoints)
        print(f"steps\torig\tinc")
        for num_timepoints in range(min_num_timepoints, max_num_timepoints):
            vars = CHIME.make_chime_variables(num_timepoints)
            parameters, init, dynamics, bounds = CHIME.make_chime_model(
                *vars,
                num_timepoints,
            )
            phi = self.encode_time_horizon(parameters, init, dynamics, num_timepoints)
            phi_stratified = [And(init, parameters), *dynamics[0:num_timepoints]]
            reset_env()
            model, elapsed = self.run_get_model(
                phi, solver_name="z3", logic=QF_UFLIRA, solver_options=solver_options
            )
            reset_env()
            asm_model, asm_elapsed = self.run_assumption_solver(
                phi_stratified,
                solver_name="z3",
                logic=QF_UFLIRA,
                solver_options=solver_options,
            )

            # reset_env()
            # inc_model, inc_elapsed = self.run_incremental_solver(
            #     phi_stratified,
            #     solver_name="z3",
            #     logic=QF_UFLIRA,
            #     solver_options=solver_options,
            # )
            inc_elapsed = 0
            print(
                f"{num_timepoints}\t{float('{:.2f}'.format(elapsed))}\t{float('{:.2f}'.format(inc_elapsed))}\t{float('{:.2f}'.format(asm_elapsed))}"
            )


if __name__ == "__main__":
    unittest.main()
