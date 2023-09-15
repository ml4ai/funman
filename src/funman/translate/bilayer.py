"""
This module encodes bilayer models into a SMTLib formula.  

"""
import logging
from functools import reduce
from typing import Dict, List, Tuple

import pysmt
from pydantic import ConfigDict
from pysmt.formula import FNode
from pysmt.shortcuts import (
    GE,
    GT,
    LE,
    LT,
    TRUE,
    And,
    Equals,
    Minus,
    Plus,
    Real,
    Symbol,
    Times,
    simplify,
)
from pysmt.typing import REAL

from funman.model.bilayer import (
    BilayerEdge,
    BilayerFluxNode,
    BilayerMeasurement,
    BilayerModel,
    BilayerNode,
    BilayerStateNode,
)
from funman.model.model import Model
from funman.representation import ModelParameter
from funman.representation.representation import Box, Interval
from funman.translate import Encoder, Encoding, EncodingOptions
from funman.translate.simplifier import FUNMANSimplifier
from funman.utils.sympy_utils import to_sympy

l = logging.Logger(__name__)


# class BilayerEncodingOptions(EncodingOptions):
#     """
#     The BilayerEncodingOptions are:

#     * step_size: the number of time units separating encoding steps

#     * max_steps: the number of encoding steps

#     """

#     def __init__(self, step_size=1, max_steps=2) -> None:
#         super().__init__(max_steps)
#         self.step_size = step_size
#         self.max_steps = max_steps


class BilayerEncoder(Encoder):
    """
    The BilayerEncoder compiles a BilayerModel into a SMTLib formula.  The
    formula defines a series of steps that update a set of variables each step,
    as defined by a Bilayer model.
    """

    model_config = ConfigDict()

    def _encode_next_step(
        self,
        scenario: "AnalysisScenario",
        step: int,
        next_step: int,
        time_dependent_parameters=None,
        substitutions=None,
    ) -> Tuple[FNode, Dict[FNode, FNode]]:
        transition, substitutions = self._encode_bilayer_timepoint(
            scenario,
            step,
            next_step,
            time_dependent_parameters=time_dependent_parameters,
            substitutions=substitutions,
        )

        if scenario.model.measurements:
            measurements = self._encode_measurements(
                scenario.model.measurements, [step + next_step]
            )
        else:
            measurements = TRUE()

        return And(transition, measurements).simplify(), substitutions

    def _encode_untimed_constraints(
        self, scenario: "AnalysisScenario"
    ) -> FNode:
        super_untimed_constraints = Encoder._encode_untimed_constraints(
            self, scenario
        )
        untimed_constraints = []

        # Encode that all of the identical parameters are equal
        untimed_constraints.append(
            And(
                [
                    Equals(Symbol(var1, REAL), Symbol(var2, REAL))
                    for group in scenario.model.identical_parameters
                    for var1 in group
                    for var2 in group
                    if var1 != var2
                ]
            ).simplify()
        )
        return And(
            And(untimed_constraints).simplify(), super_untimed_constraints
        )

    def _get_timed_symbols(self, model: Model) -> List[str]:
        timed_symbols = []
        # All state nodes correspond to timed symbols
        for idx, node in model.bilayer._state.items():
            timed_symbols.append(node.parameter)
        return timed_symbols

    def encode_model(self, model: Model, time_dependent_parameters=False):
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
                (self.config.num_steps * self.config.step_size) + 1,
                self.config.step_size,
            )

            if len(list(state_timepoints)) == 0:
                raise Exception(
                    f"Could not identify timepoints from step_size = {self.config.step_size} and num_steps = {self.config.num_steps}"
                )

            transition_timepoints = range(
                0,
                self.config.num_steps * self.config.step_size,
                self.config.step_size,
            )

            # All state nodes correspond to timed symbols
            for idx, node in model.bilayer._state.items():
                self._timed_symbols.add(node.parameter)

            # All flux nodes correspond to untimed symbols
            for _, node in model.bilayer._flux.items():
                self._untimed_symbols.add(node.parameter)

            init = self._define_init(scenario)

            encoding = self._encode_bilayer(
                model.bilayer,
                state_timepoints,
                time_dependent_parameters=time_dependent_parameters,
            )

            if model.parameter_bounds:
                parameters = [
                    ModelParameter(
                        name=node.parameter,
                        lb=model.parameter_bounds[node.parameter][0],
                        ub=model.parameter_bounds[node.parameter][1],
                    )
                    for _, node in model.bilayer._flux.items()
                    if node.parameter in model.parameter_bounds
                    and model.parameter_bounds[node.parameter]
                ]
                if model.measurements:
                    parameters += [
                        ModelParameter(
                            name=node.parameter,
                            lb=model.parameter_bounds[node.parameter][0],
                            ub=model.parameter_bounds[node.parameter][1],
                        )
                        for _, node in model.measurements._flux.items()
                        if node.parameter in model.parameter_bounds
                        and model.parameter_bounds[node.parameter]
                    ]
                if time_dependent_parameters:
                    timed_parameters = [
                        p.timed_copy(timepoint)
                        for p in parameters
                        for timepoint in transition_timepoints
                    ]
                    parameter_box = Box(
                        bounds={
                            p.name: Interval(lb=p.lb, ub=p.ub)
                            for p in timed_parameters
                        }
                    )
                else:
                    parameter_box = Box(
                        bounds={
                            p.name: Interval(lb=p.lb, ub=p.ub)
                            for p in parameters
                        }
                    )
                parameter_constraints = self.box_to_smt(
                    parameter_box, closed_upper_bound=True
                )
            else:
                parameter_constraints = TRUE()

            if model.measurements:
                measurements = self._encode_measurements(
                    model.measurements, state_timepoints
                )
            else:
                measurements = TRUE()

            if time_dependent_parameters:
                ## Assume that all parameters are constant
                parameter_constraints = And(
                    parameter_constraints,
                    self._set_parameters_constant(
                        [v.parameter for v in model.bilayer._flux.values()],
                        encoding,
                    ),
                )
                if model.measurements:
                    parameter_constraints = And(
                        parameter_constraints,
                        self._set_parameters_constant(
                            [
                                v.parameter
                                for v in model.measurements._flux.values()
                            ],
                            measurements,
                        ),
                    )
            else:
                pass

            # Encode that all of the identical parameters are equal
            identical_parameters = And(
                [
                    Equals(Symbol(var1, REAL), Symbol(var2, REAL))
                    for group in model.identical_parameters
                    for var1 in group
                    for var2 in group
                    if var1 != var2
                ]
            ).simplify()

            formula = And(
                init,
                parameter_constraints,
                encoding,
                measurements,
                identical_parameters,
                (
                    model._extra_constraints
                    if model._extra_constraints
                    else TRUE()
                ),
            )
            symbols = self._symbols(formula)
            return Encoding(_formula=formula, _symbols=symbols)
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

    def can_encode():
        """
        Return boolean indicating if the scenario can be encoded with the FUNMANConfig
        """
        encodable = True
        reasons = []
        if self.config.substitute_subformulas:
            if not all(
                v in self._scenario.model.init_values
                for v in self._scenario.model._state_var_names()
            ):
                encodable = False
                reasons.append(
                    "Cannot use configuration option 'substitute_subformulas=True' if there is no init_values specified for model."
                )
        if len(reasons) > 0:
            l.error(reasons)

        return encodable

    def _encode_bilayer(
        self, scenario, timepoints, time_dependent_parameters=False
    ):
        ans = simplify(
            And(
                [
                    self._encode_bilayer_timepoint(
                        scenario,
                        timepoints[i],
                        timepoints[i + 1],
                        time_dependent_parameters=time_dependent_parameters,
                    )[0]
                    for i in range(len(timepoints) - 1)
                ]
            )
        )
        return ans

    def _encode_bilayer_timepoint(
        self,
        scenario,
        timepoint,
        next_timepoint,
        time_dependent_parameters=False,
        substitutions={},
    ):
        bilayer = scenario.model.bilayer
        ## Calculate time step size
        time_step_size = next_timepoint - timepoint
        eqns = (
            []
        )  ## List of SMT equations for a given timepoint. These will be
        ## joined by an "And" command and returned

        for t in bilayer._tangent:  ## Loop over _tangents (derivatives)
            pos_derivative_expr_terms = []
            neg_derivative_expr_terms = []
            ## Get _tangent variable and translate it to SMT form tanvar_smt
            tanvar = bilayer._tangent[t].parameter
            tanvar_smt = self._encode_bilayer_state_node(
                bilayer._tangent[t],
                timepoint=timepoint,
            )
            state_var_next_step = bilayer._state[t].parameter
            state_var_smt = self._encode_bilayer_state_node(
                bilayer._state[t], timepoint=timepoint
            )
            state_var_next_step_smt = self._encode_bilayer_state_node(
                bilayer._state[t], timepoint=next_timepoint
            )

            relevant_output_edges = [
                (val, val.src.index)
                for val in bilayer._output_edges
                if val.tgt.index == bilayer._tangent[t].index
            ]
            for flux_sign_index in relevant_output_edges:
                flux_term = bilayer._flux[flux_sign_index[1]]
                output_edge = bilayer._output_edges[flux_sign_index[1]]
                expr = self._encode_bilayer_flux_node(
                    flux_term,
                    timepoint=(
                        timepoint if time_dependent_parameters else None
                    ),
                )
                ## Check which state vars go to that param
                relevant_input_edges = [
                    self._encode_bilayer_state_node(
                        bilayer._state[val2.src.index], timepoint=timepoint
                    )
                    for val2 in bilayer._input_edges
                    if val2.tgt.index == flux_sign_index[1]
                ]
                for state_var in relevant_input_edges:
                    expr = Times(expr, state_var)
                if (
                    self._encode_bilayer_edge(flux_sign_index[0], timepoint)
                    == "positive"
                ):
                    pos_derivative_expr_terms.append(expr)
                elif (
                    self._encode_bilayer_edge(flux_sign_index[0], timepoint)
                    == "negative"
                ):
                    neg_derivative_expr_terms.append(expr)
            # Assemble into equation of the form f(t + delta t) approximately =
            # f(t) + (delta t) f'(t)
            pos_terms = (
                reduce(lambda a, b: Plus(a, b), pos_derivative_expr_terms)
                if len(pos_derivative_expr_terms) > 0
                else Real(0.0)
            )
            neg_terms = (
                reduce(lambda a, b: Plus(a, b), neg_derivative_expr_terms)
                if len(neg_derivative_expr_terms) > 0
                else Real(0.0)
            )
            # noise = Symbol(f"noise_{state_var_next_step_smt}", REAL)
            # self._timed_symbols.add(f"{noise}".rsplit("_", 1)[0])
            if self.config.constraint_noise != 0.0:
                eqn = simplify(
                    And(
                        LE(
                            state_var_next_step_smt,
                            Plus(
                                state_var_smt,
                                Times(
                                    Real(time_step_size),
                                    Minus(pos_terms, neg_terms),
                                ),
                                Real(self.config.constraint_noise),
                            ),
                        ),
                        GE(
                            state_var_next_step_smt,
                            Plus(
                                state_var_smt,
                                Times(
                                    Real(time_step_size),
                                    Minus(pos_terms, neg_terms),
                                ),
                                Real(-self.config.constraint_noise),
                            ),
                        ),
                    )
                )
            else:
                rhs = Plus(
                    state_var_smt,
                    Times(
                        Real(time_step_size),
                        Minus(pos_terms, neg_terms),
                    ),
                ).simplify()
                if self.config.substitute_subformulas:
                    rhs = FUNMANSimplifier.sympy_simplify(
                        to_sympy(
                            rhs, [str(s) for s in rhs.get_free_variables()]
                        ),
                        parameters=scenario.parameters,
                        substitutions=substitutions,
                        threshold=self.config.series_approximation_threshold,
                        taylor_series_order=self.config.taylor_series_order,
                    )

                eqn = Equals(state_var_next_step_smt, rhs)
                substitutions[state_var_next_step_smt] = rhs

            # print(eqn)
            eqns.append(eqn)
        return And(eqns), substitutions

    def _encode_bilayer_node(self, node, timepoint=None):
        if not isinstance(node, BilayerNode):
            raise Exception("Node is not a BilayerNode")
        param = node.parameter
        if timepoint is not None:
            ans = Symbol(f"{param}_{timepoint}", REAL)
        else:
            ans = Symbol(f"{param}", REAL)
        return ans

    def _encode_bilayer_state_node(self, node, timepoint=None):
        if not isinstance(node, BilayerStateNode):
            raise Exception("Node is not a BilayerStateNode")
        return self._encode_bilayer_node(node, timepoint=timepoint)

    def _encode_bilayer_flux_node(self, node, timepoint=None):
        if not isinstance(node, BilayerFluxNode):
            raise Exception("Node is not a BilayerFluxNode")
        return self._encode_bilayer_node(node, timepoint=timepoint)

    def _encode_bilayer_edge(self, edge, timepoint=None):
        if not isinstance(edge, BilayerEdge):
            raise Exception("Edge is not a BilayerEdge")
        return edge._get_label()

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
        # flux = next([measurements._output_edges])
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
