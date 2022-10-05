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
    Solver
)
from pysmt.logics import QF_UFLIRA
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



class TestStepSolver(unittest.TestCase):

    def make_formulas(self, timesteps):
        formulas = []
        x = [Symbol(f'x_{i}', REAL) for i in range(timesteps + 1)]
        y = [Symbol(f'y_{i}', REAL) for i in range(timesteps)]
        a = [Symbol(f'a_{i}', REAL) for i in range(timesteps + 1)]
        b = [Symbol(f'b_{i}', REAL) for i in range(timesteps)]

        initial = And([
            Equals(x[0], Real(0)),
            Equals(a[0], Real(0))
        ])
        formulas.append(initial)

        for i in range(timesteps):
            formulas.append(And([
                Equals(b[i], a[i] + 1.0),
                Equals(x[i + 1], y[i] + 1.0),
                Equals(a[i + 1], b[i] + 1.0),
                Equals(y[i], x[i] + 1.0)
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
        return model, elapsed

    def run_incremental_solver(self, formulas, solver_name=None, logic=None):
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
        return model, elapsed

    def run_incremental_solver_with_substitution(self, formulas, solver_name=None, logic=None):
        # with steps and substitution
        with elapsed_timer() as elapsed:
            with Solver(name=solver_name, logic=logic) as solver:
                substitutions = {}
                for phi in formulas:
                    solver.push()
                    sub_phi = phi.substitute(substitutions).simplify()
                    solver.add_assertion(sub_phi)
                    if not solver.solve():
                        raise Exception("unsat")
                    model = solver.get_model()
                    solver.pop()
                    for v in sub_phi.get_free_variables():
                        substitutions[v] = model[v]
                solver.add_assertion(And([Equals(k, v) for (k, v) in substitutions.items()]))
                if not solver.solve():
                    raise Exception("unsat")
                model = solver.get_model()
            elapsed = elapsed()
        return model, elapsed

    def run_n_times(self, n, f, formulas, solver_name=None, logic=None):
        sum_e = 0
        for i in range(n):
            (m, e) = f(formulas, solver_name, logic)
            sum_e += e
        return sum_e / n

    def run_control(self, formulas, solver_name=None, logic=None):
        with elapsed_timer() as elapsed:
            time.sleep(0.01)
            elapsed = elapsed()
        return None, elapsed

    def test_step_solver(self):
        formulas = self.make_formulas(100)

        solver_name='z3'
        logic = QF_UFLIRA

        get_model(And(formulas), solver_name=solver_name, logic=logic)

        n = 20
        elapsed0 = self.run_n_times(n, self.run_get_model, formulas, solver_name=solver_name, logic=logic)
        elapsed1 = self.run_n_times(n, self.run_solver, formulas, solver_name=solver_name, logic=logic)
        elapsed2 = self.run_n_times(n, self.run_incremental_solver, formulas, solver_name=solver_name, logic=logic)
        #elapsed3 = self.run_n_times(n, self.run_incremental_solver_with_substitution, formulas, solver_name=solver_name, logic=logic)

        elapsedc = self.run_n_times(n, self.run_control, formulas, solver_name=solver_name, logic=logic)

        ljust = 25
        print(f"{'get_model:'.ljust(ljust)}{elapsed0:.3f}")
        print(f"{'solver:'.ljust(ljust)}{elapsed1:.3f}")
        print(f"{'incremental solver:'.ljust(ljust)}{elapsed2:.3f}")
        # print(f"{'incremental solver with substitution:'.ljust(ljust)}{elapsed3:.3f}")
        print(f"\n{'control (0.010):'.ljust(ljust)}{elapsedc:.3f}")

if __name__ == "__main__":
    unittest.main()
