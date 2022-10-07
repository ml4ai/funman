from pysmt.shortcuts import (
    get_model,
    And,
    Symbol,
    FunctionType,
    Function,
    Equals,
    Int,
    Real,
    substitute,
    TRUE,
    FALSE,
    Iff,
    Plus,
    ForAll,
    LT,
    simplify,
    GT,
    LE,
    GE,
    Solver,
    Ite,
    reset_env as pysmt_reset_env
)
from pysmt.logics import QF_NRA, QF_LRA, QF_UFLIRA, QF_UFNRA
from pysmt.typing import INT, REAL, BOOL

import random
import time
import unittest

import logging
l = logging.getLogger(__file__)
l.setLevel(logging.ERROR)

from contextlib import contextmanager
from timeit import default_timer

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



class TestIncrementalSolver(unittest.TestCase):

    def make_linear_formulas(self, timesteps, type, cons):
        formulas = []
        x = [Symbol(f'x_{i}', type) for i in range(timesteps + 1)]
        y = [Symbol(f'y_{i}', type) for i in range(timesteps)]
        initial = And([
            Equals(x[0], cons(0))
        ])
        formulas.append(initial)

        for i in range(timesteps):
            formulas.append(And([
                Equals(x[i + 1], y[i] + 1),
                Equals(y[i], x[i] + 1)
            ]))
        return formulas

    def make_product_formulas(self, timesteps, type, cons):
        formulas = []
        x = [Symbol(f'x_{i}', type) for i in range(timesteps + 1)]
        y = [Symbol(f'y_{i}', type) for i in range(timesteps + 1)]
        initial = And([
            Equals(x[0], cons(2)),
            Equals(y[0], cons(3))
        ])
        formulas.append(initial)

        for i in range(timesteps):
            formulas.append(And([
                Equals(x[i + 1], x[i] * y[i]),
                Equals(y[i + 1], x[i] + 1)
            ]))
        return formulas

    def make_polynomial_formulas(self, timesteps, type, cons):
        formulas = []
        x = [Symbol(f'x_{i}', type) for i in range(timesteps + 1)]
        y = [Symbol(f'y_{i}', type) for i in range(timesteps + 1)]
        initial = And([
            Equals(x[0], cons(2)),
            Equals(y[0], cons(3))
        ])
        formulas.append(initial)

        for i in range(timesteps):
            formulas.append(And([
                Equals(x[i + 1], x[i] * x[i]),
                Equals(y[i + 1], x[i] + 1)
            ]))
        return formulas

    def run_get_model(self, formulas, solver_name=None, logic=None):
        # just model
        with elapsed_timer() as elapsed:
            model = get_model(And(formulas), solver_name=solver_name, logic=logic)
            elapsed = elapsed()
        return model, elapsed

    def run_solver(self, formulas, solver_name=None, logic=None):
        # without steps
        with elapsed_timer() as elapsed:
            with Solver(name=solver_name, logic=logic) as solver:
                solver.add_assertion(And(formulas))
                if not solver.solve():
                    raise Exception("unsat")
                model = solver.get_model()
                elapsed = elapsed()
                solver.exit()
        return model, elapsed

    def run_solver_with_single_push(self, formulas, solver_name=None, logic=None):
        # without steps
        with elapsed_timer() as elapsed:
            with Solver(name=solver_name, logic=logic) as solver:
                solver.push()
                solver.add_assertion(And(formulas))
                if not solver.solve():
                    raise Exception("unsat")
                model = solver.get_model()
                elapsed = elapsed()
                solver.exit()
        return model, elapsed

    def run_incremental_solver_2(self, formulas, solver_name=None, logic=None):
        # with steps
        with elapsed_timer() as elapsed:
            with Solver(name=solver_name, logic=logic) as solver:
                for phi in formulas:
                    solver.push()
                    solver.add_assertion(phi)
                    if not solver.solve():
                        raise Exception("unsat")
                    model = solver.get_model()
                    solver.pop()
                    solver.add_assertion(And([Equals(v, model[v]) for v in phi.get_free_variables()]))
                if not solver.solve():
                    raise Exception("unsat")
                model = solver.get_model()
                elapsed = elapsed()
                solver.exit()
        return model, elapsed

    def run_incremental_solver(self, formulas, solver_name=None, logic=None):
        # with steps
        with elapsed_timer() as elapsed:
            with Solver(name=solver_name, logic=logic) as solver:
                for phi in formulas:
                    solver.push()
                    solver.add_assertion(phi)
                    # if not solver.solve():
                    #     raise Exception("unsat")
                if not solver.solve():
                    raise Exception("unsat")
                model = solver.get_model()
                elapsed = elapsed()
                solver.exit()
        return model, elapsed

    def run_n_times(self, n, f, make_formulas, solver_name=None, logic=None):
        sum_e = 0
        for i in range(n):
            reset_env()
            (m, e) = f(make_formulas(), solver_name, logic)
            sum_e += e
        return sum_e / n

    def run_control(self, formulas, solver_name=None, logic=None):
        with elapsed_timer() as elapsed:
            time.sleep(0.01)
            elapsed = elapsed()
        return None, elapsed

    def test_incremental_solver_timings(self):
        solver_name='z3'
        logic = QF_UFLIRA
        # force Z3 to load
        reset_env()
        self.run_get_model(self.make_product_formulas(10, REAL, Real), solver_name=solver_name, logic=logic)
        reset_env()
        self.run_incremental_solver(self.make_product_formulas(10, REAL, Real), solver_name=solver_name, logic=logic)

        timesteps = 100
        for i in range(timesteps):
            t = i + 1
            make_real_formulas = lambda: self.make_product_formulas(t, REAL, Real)
            n = 1
            ljust = 25
            elapsed0 = self.run_n_times(n, self.run_get_model, make_real_formulas, solver_name=solver_name, logic=logic)
            # elapsed1 = self.run_n_times(n, self.run_solver, make_real_formulas, solver_name=solver_name, logic=logic)
            elapsed2 = self.run_n_times(n, self.run_incremental_solver, make_real_formulas, solver_name=solver_name, logic=logic)
            print(f"Timestep {t}")
            print(f"{'  get_model:'.ljust(ljust)}{elapsed0:.3f}")
            # print(f"{'  solver:'.ljust(ljust)}{elapsed1:.3f}")
            print(f"{'  incremental solver:'.ljust(ljust)}{elapsed2:.3f}")

        # elapsed0 = self.run_n_times(n, self.run_get_model, make_int_formulas, solver_name=solver_name, logic=logic)
        # elapsed1 = self.run_n_times(n, self.run_solver, make_int_formulas, solver_name=solver_name, logic=logic)
        # elapsed2 = self.run_n_times(n, self.run_incremental_solver, make_int_formulas, solver_name=solver_name, logic=logic)
        # print("INT")
        # print(f"{'  get_model:'.ljust(ljust)}{elapsed0:.3f}")
        # print(f"{'  solver:'.ljust(ljust)}{elapsed1:.3f}")
        # print(f"{'  incremental solver:'.ljust(ljust)}{elapsed2:.3f}")

if __name__ == "__main__":
    unittest.main()
