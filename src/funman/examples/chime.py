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
from pysmt.typing import INT, REAL, BOOL


class CHIME(object):
    def make_chime_variables(num_timepoints):
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
        num_timepoints,
    ):

        # Params
        parameters = And(
            [
                Equals(beta, Real(6.7857e-05)),
                Equals(gamma, Real(0.071428571)),
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

        dynamics = And(
            [
                And(
                    [
                        # r_n = gamma * i + r  # Update to the amount of individuals that are recovered ## sir_r_n_exp
                        Equals(recovered_n[t + 1], gamma * infected[t] + recovered[t]),
                        # s_n = (-beta * s * i) + s  # Update to the amount of individuals that are susceptible ## sir_s_n_exp
                        Equals(
                            susceptible_n[t + 1],
                            (-beta * susceptible[t] * infected[t]) + susceptible[t],
                        ),
                        # i_n = (beta * s * i - gamma * i) + i  # Update to the amount of individuals that are infectious ## sir_i_n_exp
                        Equals(
                            infected_n[t + 1],
                            (beta * susceptible[t] * infected[t] - gamma * infected[t])
                            + infected[t],
                        ),
                        # scale = n / (s_n + i_n + r_n)  # A scaling factor to compute updated disease variables ## sir_scale_exp
                        Equals(
                            scale[t + 1],
                            n
                            / (
                                susceptible_n[t + 1]
                                + infected_n[t + 1]
                                + recovered_n[t + 1]
                            ),
                        ),
                        # s = s_n * scale  ## sir_s_exp
                        Equals(susceptible[t + 1], susceptible_n[t + 1] * scale[t + 1]),
                        # i = i_n * scale  ## sir_i_exp
                        Equals(infected[t + 1], infected_n[t + 1] * scale[t + 1]),
                        # r = r_n * scale  ## sir_r_exp
                        Equals(recovered[t + 1], recovered_n[t + 1] * scale[t + 1]),
                    ]
                )
                for t in range(num_timepoints - 1)
            ]
        )

        bounds = And(
            [
                And(
                    [
                        LE(recovered_n[t], n),
                        GE(recovered_n[t], Real(0.0)),
                        LE(susceptible_n[t], n),
                        GE(susceptible_n[t], Real(0.0)),
                        LE(infected_n[t], n),
                        GE(infected_n[t], Real(0.0)),
                        LE(recovered[t], n),
                        GE(recovered[t], Real(0.0)),
                        LE(susceptible[t], n),
                        GE(susceptible[t], Real(0.0)),
                        LE(infected[t], n),
                        GE(infected[t], Real(0.0)),
                    ]
                )
                for t in range(num_timepoints)
            ]
        )
        return parameters, init, dynamics, bounds

    def make_chime_query(infected, num_timepoints):
        threshold = 100

        # I_t <= 100
        query = And([LT(infected[t], Real(threshold)) for t in range(num_timepoints)])
        return query
