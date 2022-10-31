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
    Times,
)
from pysmt.typing import REAL


class CHIME(object):
    def make_model(
        self,
        epochs=[(0, 20), (21, 60)],
        population_size=1002,
        infectious_days=14.0,
        infected_threshold=0.1,
        linearize=False
    ):
        num_timepoints = epochs[-1][-1]  # Last timepoint of last epoch
        vars = self.make_chime_variables(num_timepoints, epochs)
        self.linearize = linearize
        parameters, init, dynamics = self.make_chime_model(
            *vars, num_timepoints, epochs, population_size, infectious_days
        )
        query = self.make_chime_query(
            vars[1], vars[9], num_timepoints, infected_threshold
        )
        return vars, (parameters, init, dynamics, query)

    def make_chime_variables(self, num_timepoints, epochs):
        susceptible = [
            Symbol(f"s_{t}", REAL) for t in range(num_timepoints + 1)
        ]
        infected = [Symbol(f"i_{t}", REAL) for t in range(num_timepoints + 1)]
        recovered = [Symbol(f"r_{t}", REAL) for t in range(num_timepoints + 1)]

        susceptible_n = [
            Symbol(f"s_n_{t+1}", REAL) for t in range(num_timepoints)
        ]
        infected_n = [Symbol(f"i_n_{t+1}", REAL) for t in range(num_timepoints)]
        recovered_n = [
            Symbol(f"r_n_{t+1}", REAL) for t in range(num_timepoints)
        ]

        scale = [Symbol(f"scale_{t+1}", REAL) for t in range(num_timepoints)]

        betas = [Symbol(f"beta_{b}", REAL) for b in range(len(epochs))]
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
            betas,
            gamma,
            n,
            delta,
        )

    def make_dynamics_s1(self, *vars):

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
            num_timepoints,
            epochs,
            population_size,
            infectious_days,
            epoch_idx,
            t,
        ) = vars
        return [
            # r_n = gamma * i + r  # Update to the amount of individuals that are recovered ## sir_r_n_exp
            Equals(recovered_n[t], gamma * infected[t] + recovered[t]),
            # s_n = (-beta * s * i) + s  # Update to the amount of individuals that are susceptible ## sir_s_n_exp
            (
                Equals(
                    susceptible_n[t],
                    (-betas[epoch_idx] * infected[t]) + susceptible[t],
                )
                if self.linearize 
                else
                    Equals(
                        susceptible_n[t],
                        (-betas[epoch_idx] * susceptible[t] * infected[t]) + susceptible[t],
                    )),
            # i_n = (beta * s * i - gamma * i) + i  # Update to the amount of individuals that are infectious ## sir_i_n_exp
            (
                Equals(
                    infected_n[t],
                    (betas[epoch_idx] * infected[t] - gamma * infected[t])
                    + infected[t],
                )
                if self.linearize else
                    Equals(
                        infected_n[t],
                        (betas[epoch_idx] *  susceptible[t] * infected[t] - gamma * infected[t])
                        + infected[t],
                    )),
            LE(recovered_n[t], n),
            GE(recovered_n[t], Real(0.0)),
            LE(susceptible_n[t], n),
            GE(susceptible_n[t], Real(0.0)),
            LE(infected_n[t], n),
            GE(infected_n[t], Real(0.0)),
        ]

    def make_dynamics_s2(self, *vars):
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
            num_timepoints,
            epochs,
            population_size,
            infectious_days,
            epoch_idx,
            t,
        ) = vars
        return [
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
            LE(scale[t], Real(1.0)),
            GE(scale[t], Real(0.0)),
        ]

    def make_dynamics_s3(self, *vars):
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
            num_timepoints,
            epochs,
            population_size,
            infectious_days,
            epoch_idx,
            t,
        ) = vars
        return [
            # s = s_n * scale  ## sir_s_exp
            Equals(susceptible[t + 1], susceptible_n[t] * scale[t]),
            # i = i_n * scale  ## sir_i_exp
            Equals(infected[t + 1], infected_n[t] * scale[t]),
            # r = r_n * scale  ## sir_r_exp
            Equals(recovered[t + 1], recovered_n[t] * scale[t]),
            LE(recovered[t + 1], n),
            GE(recovered[t + 1], Real(0.0)),
            LE(susceptible[t + 1], n),
            GE(susceptible[t + 1], Real(0.0)),
            LE(infected[t + 1], n),
            GE(infected[t + 1], Real(0.0)),
        ]

    def make_chime_model(self, *vars):
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
            num_timepoints,
            epochs,
            population_size,
            infectious_days,
        ) = vars
        # Params
        parameters = [
            Equals(gamma, Real(1.0 / infectious_days)),
            Equals(delta, Real(0.0)),
        ] + [Equals(b, Real(6.7e-05)) for b in betas]

        # initial population
        # s_n = 1000  ## main_s_n_exp
        # i_n = 1  ## main_i_n_exp
        # r_n = 1  ## main_r_n_exp
        init = And(
            [
                Equals(susceptible[0], Real(population_size - 2)),
                Equals(infected[0], Real(1.0)),
                Equals(recovered[0], Real(1.0)),
                Equals(n, susceptible[0] + infected[0] + recovered[0]),
            ]
        )

        dynamics = [
            (
                self.make_dynamics_s1(*vars, epoch_idx, t),
                self.make_dynamics_s2(*vars, epoch_idx, t),
                self.make_dynamics_s3(*vars, epoch_idx, t),
            )
            for epoch_idx, epoch in enumerate(epochs)
            for t in range(*epoch)
        ]

        return parameters, init, dynamics

    def make_chime_query(self, infected, n, num_timepoints, threshold):

        # I_t <= n * threshold, threshold is proportion (0, 1]
        query = [
            LT(infected[t], Times(n, Real(threshold)))
            for t in range(num_timepoints + 1)
        ]
        return query

    def encode_time_horizon(self, parameters, init, dynamics, query, horizon):
        dynamics_t = (
            And([And([And(s_t) for s_t in d_t]) for d_t in dynamics[0:horizon]])
            if horizon > 0
            else TRUE()
        )
        query_t = (
            And([q_t for q_t in query[0:horizon]]) if horizon > 0 else TRUE()
        )
        return And(And(parameters), init, dynamics_t, query_t)

    def encode_time_horizon_layered(
        self, parameters, init, dynamics, query, num_timepoints
    ):
        tmp = [
            [
                query[t],
                And(dynamics[t][0]),
                And(dynamics[t][1]),
                And(dynamics[t][2]),
            ]
            for t in range(num_timepoints - 1)
        ]
        tmp.append([query[num_timepoints]])
        layered = []
        for t in tmp:
            for s in t:
                layered.append(s)
        return [And(init, And(parameters)), *layered]
