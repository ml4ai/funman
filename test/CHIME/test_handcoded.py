from pysmt.shortcuts import \
    get_model, And, Symbol, FunctionType, Function, Equals, Int, Real, \
    Iff, Plus, ForAll, LT
from pysmt.typing import INT, REAL, BOOL

import unittest
import os

RESOURCES = os.path.join("resources")

class TestHandcoded(unittest.TestCase):
    def test_toy(self):
        s1 = Symbol("infected")
        model = get_model(And(s1))
        print(model)

    def test_simple_chime_firstorder(self):
        num_timepoints = 10
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
            And(Equals(tp1_sym, Plus(t_sym, Int(1))), 
                Equals(i_tp1, Plus(i_t, Real(1.0))))
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
            [t_sym, tp1_sym], 
            And(trans_func, transf_func_eq)) 
            # substitute(transf_func_eq, subs))
        
        # I(1) = 2.0
        query = Equals(Function(infected_sym, [Int(1)]), Real(2.0))


        # phi = init
        phi = And(init, trans_func_def, query)

        # Solve phi
        model = get_model(phi)
        print(model)

    def test_simple_chime_propositional(self):
        num_timepoints = 10
        threshold = 110.0

        susceptible = [Symbol(f"s_{t}", REAL) for t in range(num_timepoints)]
        infected = [Symbol(f"i_{t}", REAL) for t in range(num_timepoints)]
        recovered = [Symbol(f"r_{t}", REAL) for t in range(num_timepoints)]

        susceptible_n = [Symbol(f"s_n_{t}", REAL) for t in range(num_timepoints)]
        infected_n = [Symbol(f"i_n_{t}", REAL) for t in range(num_timepoints)]
        recovered_n = [Symbol(f"r_n_{t}", REAL) for t in range(num_timepoints)]
        
        scale = [Symbol(f"scale_{t}", REAL) for t in range(num_timepoints)]
        
        beta = Symbol(f"beta", REAL)
        gamma = Symbol(f"gamma", REAL)
        n = Symbol(f"n", REAL)

        # Params
        parameters = And([
            Equals(beta, Real(0.01)),
            Equals(gamma, Real(0.01))
        ])

        # initial population
        # s_n = 1000  ## main_s_n_exp
        # i_n = 1  ## main_i_n_exp
        # r_n = 1  ## main_r_n_exp
        init = And([
            Equals(susceptible[0], Real(1000.0)),
            Equals(infected[0], Real(1.0)),
            Equals(recovered[0], Real(1.0)),
            Equals(
                n, 
                susceptible[0] + infected[0] + recovered[0]
            )

        ])

        dynamics = And([
            And([
                # r_n = gamma * i + r  # Update to the amount of individuals that are recovered ## sir_r_n_exp
                Equals(
                    recovered_n[t+1], 
                    gamma * infected[t] + recovered[t] 
                ),

                # s_n = (-beta * s * i) + s  # Update to the amount of individuals that are susceptible ## sir_s_n_exp
                Equals(
                    susceptible_n[t+1], 
                    (-beta * susceptible[t] * infected[t]) + susceptible[t]
                ),

                # i_n = (beta * s * i - gamma * i) + i  # Update to the amount of individuals that are infectious ## sir_i_n_exp
                Equals(
                    infected_n[t+1], 
                    (beta * susceptible[t] * infected[t] - gamma + infected[t]) + infected[t]
                ),

                # scale = n / (s_n + i_n + r_n)  # A scaling factor to compute updated disease variables ## sir_scale_exp
                Equals(
                    scale[t+1],
                    n / (susceptible_n[t+1] + infected_n[t+1] + recovered_n[t+1])
                ),

                # s = s_n * scale  ## sir_s_exp
                Equals(
                    susceptible[t+1],
                    susceptible_n[t+1] * scale[t+1]
                ),

                # i = i_n * scale  ## sir_i_exp
                Equals(
                    infected[t+1],
                    infected_n[t+1] * scale[t+1]
                ),
                
                # r = r_n * scale  ## sir_r_exp
                Equals(
                    recovered[t+1],
                    recovered_n[t+1] * scale[t+1]
                ),
            ])
            for t in range(num_timepoints-1)
        ])

        
        # I_t <= 100
        query = And([
            LT(infected[t], Real(threshold)) 
            for t in range(num_timepoints)
        ])

        # Check that dynamics (minus query) is consistent
        phi = And(parameters, init, dynamics)
        model = get_model(phi)
        

        infected_values = [float(model.get_py_value(i)) for i in infected]
        print(f"infected = {infected_values}")

        if model:
            print("Model is consistent")
            print("*"*80)
            print(model)
            print("*"*80)
            # phi = init & dynamics & query
            phi = And(parameters, init, dynamics, query)            

            # Solve phi
            model = get_model(phi)
            if model:
                print("Model & Query is consistent")
                print("*"*80)
                print(model)
                print("*"*80)

                infected_values = [model.get_py_value(i) for i in infected]
                print(f"infected = {infected_values}")
            else:
                print("Model & Query is inconsistent")
        else:
            print("Model is inconsistent")

if __name__ == '__main__':
    unittest.main()