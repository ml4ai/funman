import os
import unittest
from cmath import inf

from funman_demo.example.chime import CHIME
from pysmt.shortcuts import (
    GE,
    LE,
    LT,
    And,
    Equals,
    ForAll,
    Function,
    FunctionType,
    Iff,
    Int,
    Plus,
    Real,
    Symbol,
    get_model,
    is_sat,
    write_smtlib,
)
from pysmt.typing import BOOL, INT, REAL

RESOURCES = os.path.join("resources")


class TestHandcoded(unittest.TestCase):
    def test_toy(self):
        s1 = Symbol("infected")
        model = get_model(And(s1))
        print(model)

    def test_simple_chime_firstorder(self):
        num_timepoints = 12
        # timepoints = [Symbol(t, INT) for t in range(num_timepoints)]

        # Function Definition for Time indexed symbols
        # I -> R
        const_t = FunctionType(REAL, [INT])

        # Function symbol for I(t)
        infected_sym = Symbol("I", const_t)

        # Declare I(0) function
        i_0 = Function(infected_sym, [Int(0)])
        # i_1 = Function(infected_sym, [Int(1)])

        # Assert I(0) = 1.0
        init = Equals(i_0, Real(1.0))

        # Dynamics for I: T(I(t), I(t+1))
        # (I->R) x (I->R) -> B
        trans_func_type = FunctionType(BOOL, [REAL, REAL])
        # T: (I->R) x (I->R) -> B
        trans_func_sym = Symbol("T", trans_func_type)
        # t: -> I
        t_sym = Symbol("t", INT)
        # t+1: -> I
        tp1_sym = Symbol("t+1", INT)
        # I(t)
        i_t = Function(infected_sym, [t_sym])
        # I(t+1)
        i_tp1 = Function(infected_sym, [tp1_sym])
        # T(I(t), I(t+1))
        trans_func = Function(trans_func_sym, [i_t, i_tp1])

        transf_func_eq = Iff(
            trans_func,
            And(
                Equals(tp1_sym, Plus(t_sym, Int(1))),
                Equals(i_tp1, Plus(i_t, Real(1.0))),
            ),
        )

        subs = {
            t_sym: Int(0),
            tp1_sym: Int(1),
        }

        # Forall t, t+1.
        #   T(I(t), I(t+1)) &&
        #   t+1 = t + 1     &&
        #   T(I(t), I(t+1)) <-> I(t+1) = I(t) + 1.0
        trans_func_def = ForAll(
            [t_sym, tp1_sym], And(trans_func, transf_func_eq)
        )
        # substitute(transf_func_eq, subs))

        # I(1) = 2.0
        query = Equals(Function(infected_sym, [Int(1)]), Real(2.0))

        # phi = init
        phi = And(init, trans_func_def, query)

        # Solve phi
        model = get_model(phi)
        print(model)

    def test_simple_chime_propositional(self):
        num_timepoints = 12
        chime = CHIME()
        vars, (parameters, init, dynamics, query) = chime.make_model()
        (
            susceptible,
            infected,
            recovered,
            susceptible_n,
            infected_n,
            recovered_n,
            scale,
            betas,
            gamma,
            n,
            delta,
        ) = vars

        # Check that dynamics (minus query) is consistent
        # phi1 = And(parameters, init, dynamics)
        # phi2 = And(parameters, init, dynamics, query)
        # write_smtlib(phi2, f"chime_{num_timepoints}.smt2")
        # res = is_sat(phi)
        phi = chime.encode_time_horizon(
            parameters, init, dynamics, query, num_timepoints
        )
        model = get_model(phi)

        infected_values = [float(model.get_py_value(i)) for i in infected]
        print(f"infected = {infected_values}")
        #  [1.0, 0.9964285714285714, 0.9928653098123178, 0.989310235435031, 0.9857633681757312, 0.9822247275110826, 0.9786943325178128, 0.9751722018751358, 0.9716583538671775, 0.9681528063854044]
        if model:
            print("Model is consistent")
            # print("*" * 80)
            # print(model)
            # print("*" * 80)
            # phi = init & dynamics & query
            # phi = And(parameters, init, dynamics, bounds, query).simplify()
        else:
            print("Model is inconsistent")


if __name__ == "__main__":
    unittest.main()
