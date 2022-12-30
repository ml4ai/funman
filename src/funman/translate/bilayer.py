"""
This module encodes bilayer models into a SMTLib formula.  

"""
import logging
from typing import Dict, List, Union

import pysmt
from pysmt.shortcuts import (
    TRUE,
    And,
    Equals,
    Plus,
    Real,
    Symbol,
    Times,
    simplify,
)
from pysmt.typing import REAL

from funman.model import Parameter
from funman.model.bilayer import (
    BilayerEdge,
    BilayerFluxNode,
    BilayerMeasurement,
    BilayerModel,
    BilayerNode,
    BilayerStateNode,
)
from funman.model.model import Model
from funman.search.representation import Box
from funman.translate import Encoder, Encoding, EncodingOptions

l = logging.Logger(__name__)


class BilayerEncodingOptions(EncodingOptions):
    """
    The BilayerEncodingOptions are:

    * step_size: the number of time units separating encoding steps

    * max_steps: the number of encoding steps

    """

    def __init__(self, step_size=1, max_steps=2) -> None:
        super().__init__(max_steps)
        self.step_size = step_size
        self.max_steps = max_steps


class BilayerEncoder(Encoder):
    """
    The BilayerEncoder compiles a BilayerModel into a SMTLib formula.  The
    formula defines a series of steps that update a set of variables each step,
    as defined by a Bilayer model.
    """

    def __init__(
        self, config: BilayerEncodingOptions = BilayerEncodingOptions()
    ) -> None:
        super().__init__(config)

    def encode_model(self, model: Model):
        """
        Encode the model as an SMTLib formula.

        Parameters
        ----------
        model : Model
            model to encode

        Returns
        -------
        FNode
            formula encoding the model

        Raises
        ------
        Exception
            cannot identify encoding timepoints
        Exception
            cannot encode model type
        """
        if isinstance(model, BilayerModel):
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
                    [v.parameter for v in model.bilayer.flux.values()],
                    encoding,
                ),
                self._set_parameters_constant(
                    [v.parameter for v in model.measurements.flux.values()],
                    measurements,
                ),
            )

            formula = And(init, parameter_constraints, encoding, measurements)
            symbols = self._symbols(formula)
            return Encoding(formula=formula, symbols=symbols)
        else:
            raise Exception(
                f"BilayerEncoder cannot encode model of type: {type(model)}"
            )

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

    def symbol_values(
        self, model_encoding: Encoding, pysmtModel: pysmt.solvers.solver.Model
    ) -> Dict[str, Dict[str, float]]:
        """
         Get the value assigned to each symbol in the pysmtModel.

        Parameters
        ----------
        model_encoding : Encoding
            encoding using the symbols
        pysmtModel : pysmt.solvers.solver.Model
            assignment to symbols

        Returns
        -------
        Dict[str, Dict[str, float]]
            mapping from symbol and timepoint to value
        """

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
        self, model: Model, pysmtModel: pysmt.solvers.solver.Model
    ) -> Dict[str, List[Union[float, None]]]:
        """
        Gather values assigned to model parameters.

        Parameters
        ----------
        model : Model
            model encoded by self
        pysmtModel : pysmt.solvers.solver.Model
            the assignment to symbols

        Returns
        -------
        Dict[str, List[Union[float, None]]]
            mapping from parameter symbol name to value
        """
        try:
            parameters = {
                node.parameter: pysmtModel[Symbol(node.parameter, REAL)]
                for _, node in model.bilayer.flux.items()
            }
            return parameters
        except OverflowError as e:
            l.warn(e)
            return {}
