from pysmt.shortcuts import (
    get_model,
    And,
    Symbol,
    FunctionType,
    Function,
    Equals,
    Int,
    Real,
    Iff,
    Plus,
    ForAll,
    LT,
    LE,
    GE,
)
from pysmt.typing import REAL


class CHIME(object):
    def make_chime_variables(num_timepoints):
        susceptible = [Symbol(f"s_{t}", REAL) for t in range(num_timepoints + 1)]
        infected = [Symbol(f"i_{t}", REAL) for t in range(num_timepoints + 1)]
        recovered = [Symbol(f"r_{t}", REAL) for t in range(num_timepoints + 1)]

        susceptible_n = [Symbol(f"s_n_{t+1}", REAL) for t in range(num_timepoints)]
        infected_n = [Symbol(f"i_n_{t+1}", REAL) for t in range(num_timepoints)]
        recovered_n = [Symbol(f"r_n_{t+1}", REAL) for t in range(num_timepoints)]

        scale = [Symbol(f"scale_{t+1}", REAL) for t in range(num_timepoints)]

        beta = Symbol(f"beta", REAL)
        gamma = Symbol(f"gamma", REAL)
        delta = Symbol(f"delta", REAL)
        n = Symbol(f"n", REAL)
        return (
            susceptible,
            infected,
            recovered,
            susceptible_n,
            infected_n,
            recovered_n,
            scale,
            beta,
            gamma,
            n,
            delta,
        )

    def make_chime_model(
        susceptible,
        infected,
        recovered,
        susceptible_n,
        infected_n,
        recovered_n,
        scale,
        beta,
        gamma,
        n,
        delta,
        num_timepoints,
    ):

        # Params
        parameters = And(
            [
                Equals(beta, Real(6.7e-05)),
                Equals(gamma, Real(0.07)),
                Equals(delta, Real(0.0)),
            ]
        )

        # initial population
        # s_n = 1000  ## main_s_n_exp
        # i_n = 1  ## main_i_n_exp
        # r_n = 1  ## main_r_n_exp
        init = And(
            [
                Equals(susceptible[0], Real(1000.0)),
                Equals(infected[0], Real(1.0)),
                Equals(recovered[0], Real(1.0)),
                Equals(n, susceptible[0] + infected[0] + recovered[0]),
            ]
        )

        dynamics = [
            And(
                [
                    # r_n = gamma * i + r  # Update to the amount of individuals that are recovered ## sir_r_n_exp
                    Equals(recovered_n[t], gamma * infected[t] + recovered[t]),
                    # s_n = (-beta * s * i) + s  # Update to the amount of individuals that are susceptible ## sir_s_n_exp
                    Equals(
                        susceptible_n[t],
                        (-beta * infected[t]) + susceptible[t],
                    ),
                    # Equals(
                    #     susceptible_n[t],
                    #     (-beta * susceptible[t] * infected[t]) + susceptible[t],
                    # ),
                    # i_n = (beta * s * i - gamma * i) + i  # Update to the amount of individuals that are infectious ## sir_i_n_exp
                    Equals(
                        infected_n[t],
                        (beta * infected[t] - gamma * infected[t]) + infected[t],
                    ),
                    # Equals(
                    #     infected_n[t],
                    #     (beta * susceptible[t] * infected[t] - gamma * infected[t])
                    #     + infected[t],
                    # ),
                    # scale = n / (s_n + i_n + r_n)  # A scaling factor to compute updated disease variables ## sir_scale_exp
                    Equals(
                        scale[t],
                        n / (susceptible_n[t] + infected_n[t] + recovered_n[t]),
                    ),
                    # s = s_n * scale  ## sir_s_exp
                    Equals(susceptible[t + 1], susceptible_n[t] * scale[t]),
                    # i = i_n * scale  ## sir_i_exp
                    Equals(infected[t + 1], infected_n[t] * scale[t]),
                    # r = r_n * scale  ## sir_r_exp
                    Equals(recovered[t + 1], recovered_n[t] * scale[t]),
                ]
            )
            for t in range(num_timepoints)
        ]

        bounds = [
            And(
                [
                    LE(recovered_n[t], n),
                    GE(recovered_n[t], Real(0.0)),
                    LE(susceptible_n[t], n),
                    GE(susceptible_n[t], Real(0.0)),
                    LE(infected_n[t], n),
                    GE(infected_n[t], Real(0.0)),
                    LE(recovered[t + 1], n),
                    GE(recovered[t + 1], Real(0.0)),
                    LE(susceptible[t + 1], n),
                    GE(susceptible[t + 1], Real(0.0)),
                    LE(infected[t + 1], n),
                    GE(infected[t + 1], Real(0.0)),
                ]
            )
            for t in range(num_timepoints)
        ]

        return parameters, init, dynamics, bounds

    def make_chime_query(infected, num_timepoints):
        threshold = 100

        # I_t <= 100
        query = [LT(infected[t], Real(threshold)) for t in range(num_timepoints)]
        return query
