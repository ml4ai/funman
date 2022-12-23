import logging
from typing import Dict, List, Union

import pysmt
from pysmt.shortcuts import (
    FALSE,
    GE,
    GT,
    LE,
    LT,
    TRUE,
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
    Times,
    get_model,
    simplify,
    substitute,
)
from pysmt.typing import BOOL, INT, REAL

from funman.model import Model, Parameter, QueryLE, QueryTrue
from funman.model.bilayer import (
    Bilayer,
    BilayerEdge,
    BilayerFluxNode,
    BilayerMeasurement,
    BilayerModel,
    BilayerNode,
    BilayerStateNode,
)
from funman.search.representation import Box
from funman.translate import Encoder, Encoding, EncodingOptions

l = logging.Logger(__name__)


class BilayerEncodingOptions(EncodingOptions):
    def __init__(self, step_size=1, max_steps=2) -> None:
        super().__init__(max_steps)
        self.step_size = step_size
        self.max_steps = max_steps


class BilayerEncoder(Encoder):
    def __init__(
        self, config: BilayerEncodingOptions = BilayerEncodingOptions()
    ) -> None:
        super().__init__(config)

    def encode_model(self, model: BilayerModel):
        state_timepoints = range(
            0,
            self.config.max_steps + 1,
            self.config.step_size,
        )

        if len(list(state_timepoints)) == 0:
            raise Exception(
                f"Could not identify timepoints from step_size = {self.config.step_size} and max_steps = {self.config.max_steps}"
            )

        transition_timepoints = range(
            0, self.config.max_steps, self.config.step_size
        )

        init = And(
            [
                Equals(
                    self._encode_bilayer_state_node(node, 0),
                    Real(model.init_values[node.parameter]),
                )
                for idx, node in model.bilayer.state.items()
            ]
        )

        encoding = self._encode_bilayer(model.bilayer, state_timepoints)

        if model.parameter_bounds:
            parameters = [
                Parameter(
                    node.parameter,
                    lb=model.parameter_bounds[node.parameter][0],
                    ub=model.parameter_bounds[node.parameter][1],
                )
                for _, node in model.bilayer.flux.items()
                if node.parameter in model.parameter_bounds
                and model.parameter_bounds[node.parameter]
            ] + [
                Parameter(
                    node.parameter,
                    lb=model.parameter_bounds[node.parameter][0],
                    ub=model.parameter_bounds[node.parameter][1],
                )
                for _, node in model.measurements.flux.items()
                if node.parameter in model.parameter_bounds
                and model.parameter_bounds[node.parameter]
            ]

            timed_parameters = [
                p.timed_copy(timepoint)
                for p in parameters
                for timepoint in transition_timepoints
            ]
            parameter_box = Box(timed_parameters)
            parameter_constraints = parameter_box.to_smt(
                closed_upper_bound=True
            )
        else:
            parameter_constraints = TRUE()

        measurements = self._encode_measurements(
            model.measurements, state_timepoints
        )

        ## Assume that all parameters are constant
        parameter_constraints = And(
            parameter_constraints,
            self._set_parameters_constant(
                [v.parameter for v in model.bilayer.flux.values()], encoding
            ),
            self._set_parameters_constant(
                [v.parameter for v in model.measurements.flux.values()],
                measurements,
            ),
        )

        formula = And(init, parameter_constraints, encoding, measurements)
        symbols = self._symbols(formula)
        return Encoding(formula=formula, symbols=symbols)

    def _encode_measurements(
        self, measurements: BilayerMeasurement, timepoints
    ):
        ans = And(
            [
                self._encode_measurements_timepoint(
                    measurements, timepoints[i]
                )
                for i in range(len(timepoints))
            ]
        )
        return ans

    def _encode_measurements_timepoint(self, measurements, t):
        observable_defs = And(
            [
                Equals(
                    self._encode_bilayer_state_node(o, t),
                    self._observable_defn(measurements, o, t),
                )
                for o in measurements.observable.values()
            ]
        )
        return observable_defs

    def _encode_bilayer(self, bilayer, timepoints):
        ans = simplify(
            And(
                [
                    self._encode_bilayer_timepoint(
                        bilayer, timepoints[i], timepoints[i + 1]
                    )
                    for i in range(len(timepoints) - 1)
                ]
            )
        )
        return ans

    def _encode_bilayer_timepoint(self, bilayer, timepoint, next_timepoint):
        ## Calculate time step size
        time_step_size = next_timepoint - timepoint
        eqns = (
            []
        )  ## List of SMT equations for a given timepoint. These will be
        ## joined by an "And" command and returned

        for t in bilayer.tangent:  ## Loop over tangents (derivatives)
            derivative_expr = 0
            ## Get tangent variable and translate it to SMT form tanvar_smt
            tanvar = bilayer.tangent[t].parameter
            tanvar_smt = self._encode_bilayer_state_node(
                bilayer.tangent[t], timepoint
            )
            state_var_next_step = bilayer.state[t].parameter
            state_var_smt = self._encode_bilayer_state_node(
                bilayer.state[t], timepoint
            )
            state_var_next_step_smt = self._encode_bilayer_state_node(
                bilayer.state[t], next_timepoint
            )

            relevant_output_edges = [
                (val, val.src.index)
                for val in bilayer.output_edges
                if val.tgt.index == bilayer.tangent[t].index
            ]
            for flux_sign_index in relevant_output_edges:
                flux_term = bilayer.flux[flux_sign_index[1]]
                output_edge = bilayer.output_edges[flux_sign_index[1]]
                expr = self._encode_bilayer_flux_node(flux_term, timepoint)
                ## Check which state vars go to that param
                relevant_input_edges = [
                    self._encode_bilayer_state_node(
                        bilayer.state[val2.src.index], timepoint
                    )
                    for val2 in bilayer.input_edges
                    if val2.tgt.index == flux_sign_index[1]
                ]
                for state_var in relevant_input_edges:
                    expr = Times(expr, state_var)
                if (
                    self._encode_bilayer_edge(flux_sign_index[0], timepoint)
                    == "positive"
                ):
                    derivative_expr += expr
                elif (
                    self._encode_bilayer_edge(flux_sign_index[0], timepoint)
                    == "negative"
                ):
                    derivative_expr -= expr
            # Assemble into equation of the form f(t + delta t) approximately =
            # f(t) + (delta t) f'(t)
            eqn = simplify(
                Equals(
                    state_var_next_step_smt,
                    Plus(state_var_smt, time_step_size * derivative_expr),
                )
            )
            # print(eqn)
            eqns.append(eqn)
        return And(eqns)

    def _encode_bilayer_node(self, node, timepoint):
        if not isinstance(node, BilayerNode):
            raise Exception("Node is not a BilayerNode")
        param = node.parameter
        ans = Symbol(f"{param}_{timepoint}", REAL)
        return ans

    def _encode_bilayer_state_node(self, node, timepoint):
        if not isinstance(node, BilayerStateNode):
            raise Exception("Node is not a BilayerStateNode")
        return self._encode_bilayer_node(node, timepoint)

    def _encode_bilayer_flux_node(self, node, timepoint):
        if not isinstance(node, BilayerFluxNode):
            raise Exception("Node is not a BilayerFluxNode")
        return self._encode_bilayer_node(node, timepoint)

    def _encode_bilayer_edge(self, edge, timepoint):
        if not isinstance(edge, BilayerEdge):
            raise Exception("Edge is not a BilayerEdge")
        return edge.get_label()

    def _observable_defn(self, measurements, obs, t):
        # flux * incoming1 * incoming2 ...
        obs_in_edges = measurements.node_incoming_edges[obs]
        result = Real(0.0)
        for src in obs_in_edges:
            # src is a flux
            f_t = self._encode_bilayer_flux_node(src, t)
            src_srcs = [
                self._encode_bilayer_edge(s, t)
                for s in measurements.node_incoming_edges[src]
            ]
            result = Plus(result, Times([f_t] + src_srcs)).simplify()
        # flux = next([measurements.output_edges])
        return result

    def _set_parameters_constant(self, parameters, formula):
        params = {
            parameter: Symbol(parameter, REAL) for parameter in parameters
        }
        # print(formula)
        symbols = self._symbols(formula)
        all_equal = And(
            [
                And([Equals(params[p], s) for t, s in symbols[p].items()])
                for p in params
            ]
        )
        return all_equal

    def _split_symbol(self, symbol):
        if "_" in symbol.symbol_name():
            return symbol.symbol_name().rsplit("_", 1)
        else:
            return symbol.symbol_name(), None

    def symbol_values(self, model_encoding, pysmtModel):
        vars = model_encoding.symbols
        vals = {}
        for var in vars:
            vals[var] = {}
            for t in vars[var]:
                try:
                    symbol = vars[var][t]
                    vals[var][t] = float(pysmtModel.get_py_value(symbol))
                except OverflowError as e:
                    l.warn(e)
        return vals

    def parameter_values(
        self, model, pysmtModel: pysmt.solvers.solver.Model
    ) -> Dict[str, List[Union[float, None]]]:
        try:
            parameters = {
                node.parameter: pysmtModel[Symbol(node.parameter, REAL)]
                for _, node in model.bilayer.flux.items()
            }
            return parameters
        except OverflowError as e:
            l.warn(e)
            return {}

    def symbol_timeseries(
        self, model_encoding, pysmtModel: pysmt.solvers.solver.Model
    ) -> Dict[str, List[Union[float, None]]]:
        """
        Generate a symbol (str) to timeseries (list) of values

        Parameters
        ----------
        pysmtModel : pysmt.solvers.solver.Model
            variable assignment
        """
        series = self.symbol_values(model_encoding, pysmtModel)
        a_series = {}  # timeseries as array/list
        max_t = max(
            [max([int(k) for k in tps.keys()]) for _, tps in series.items()]
        )
        a_series["index"] = list(range(0, max_t + 1))
        for var, tps in series.items():

            vals = [None] * (int(max_t) + 1)
            for t, v in tps.items():
                vals[int(t)] = v
            a_series[var] = vals
        return a_series
