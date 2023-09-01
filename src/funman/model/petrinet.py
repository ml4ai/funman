from typing import Dict, List, Union
from funman.funman import FUNMANConfig

import graphviz
import sympy
from pydantic import ConfigDict, BaseModel
from pysmt.shortcuts import REAL, Div, Plus, Real, Symbol

from funman.representation.representation import ModelParameter
from funman.translate.petrinet import PetrinetEncoder
from funman.utils.sympy_utils import substitute, to_sympy

from .generated_models.petrinet import Model as GeneratedPetrinet
from .generated_models.petrinet import State, Transition
from .model import Model


class AbstractPetriNetModel(Model):
    def _num_flow_from_state_to_transition(
        self, state_id: Union[str, int], transition_id: Union[str, int]
    ) -> int:
        return len(
            [
                edge
                for edge in self._input_edges()
                if self._edge_source(edge) == state_id
                and self._edge_target(edge) == transition_id
            ]
        )

    def _num_flow_from_transition_to_state(
        self, state_id: Union[str, int], transition_id: Union[str, int]
    ) -> int:
        return len(
            [
                edge
                for edge in self._output_edges()
                if self._edge_source(edge) == transition_id
                and self._edge_target(edge) == state_id
            ]
        )

    def _num_flow_from_transition(self, transition_id: Union[str, int]) -> int:
        return len(
            [
                edge
                for edge in self._output_edges()
                if self._edge_source(edge) == transition_id
            ]
        )

    def _num_flow_into_transition(self, transition_id: Union[str, int]) -> int:
        return len(
            [
                edge
                for edge in self._input_edges()
                if self._edge_target(edge) == transition_id
            ]
        )

    def _flow_into_state_via_transition(
        self, state_id: Union[str, int], transition_id: Union[str, int]
    ) -> float:
        num_flow_to_transition = self._num_flow_into_transition(transition_id)
        num_inflow = self._num_flow_from_transition_to_state(
            state_id, transition_id
        )
        num_transition_outputs = self._num_flow_from_transition(transition_id)
        if num_transition_outputs > 0:
            return (
                num_inflow / num_transition_outputs
            ) * num_flow_to_transition
        else:
            return 0

    def to_dot(self, values={}):
        """
        Create a dot object for visualizing the graph.

        Returns
        -------
        graphviz.Digraph
            The graph represented by self.
        """
        dot = graphviz.Digraph(
            name=f"petrinet",
            graph_attr={},
        )

        state_vars = self._state_vars()
        state_var_names = self._state_var_names()
        transitions = self._transitions()
        variable_values = self._variable_values()

        # Don't substitute initial state values for state variables, only parameters
        variable_values = {
            k: (None if k in state_var_names else v)
            for k, v in variable_values.items()
        }

        for _, var in enumerate(state_vars):
            state_var_id = self._state_var_id(var)
            state_var_name = self._state_var_name(var)
            for transition in transitions:
                transition_id = self._transition_id(transition)
                transition_parameters = self._transition_rate(transition)
                # transition_parameter_value = [
                #     self._parameter_values()[t] for t in transition_parameters
                # ]
                transition_parameter_value = [
                    substitute(t, variable_values)
                    for t in transition_parameters
                ]
                transition_name = f"{transition_id}({transition_parameters}) = {transition_parameter_value}"
                dot.node(transition_name, _attributes={"shape": "box"})
                # state var to transition
                for edge in self._input_edges():
                    if (
                        self._edge_source(edge) == state_var_id
                        and self._edge_target(edge) == transition_id
                    ):
                        dot.edge(state_var_name, transition_name)
                # transition to state var
                for edge in self._output_edges():
                    if (
                        self._edge_source(edge) == transition_id
                        and self._edge_target(edge) == state_var_id
                    ):
                        flow = self._flow_into_state_via_transition(
                            state_var_id, transition_id
                        ) / self._num_flow_from_transition_to_state(
                            state_var_id, transition_id
                        )
                        dot.edge(
                            transition_name, state_var_name, label=f"{flow}"
                        )

        return dot

    def calculate_normalization_constant(self, scenario: "AnalysisScenario") -> float:
        vars = self._state_var_names()
        values = {
            v: self._get_init_value(v, scenario)
            for v in vars
        }
        if all(v.is_constant() for v in values.values()):
            return float(sum(v.constant_value() for v in values.values()))
        else:
            raise Exception(f"Cannot calculate the normalization constant for {type(self)} because the initial state variables are not constants. Try setting the 'normalize' in the configuration to constant.")


class GeneratedPetriNetModel(AbstractPetriNetModel):
    model_config = ConfigDict(arbitrary_types_allowed=True
    )

    petrinet: GeneratedPetrinet
    _transition_rates_cache: Dict[str, Union[sympy.Expr, str]] = {}

    def default_encoder(
        self, config: "FUNMANConfig", scenario: "AnalysisScenario"
    ) -> "Encoder":
        """
        Return the default Encoder for the model

        Returns
        -------
        Encoder
            SMT encoder for model
        """
        return PetrinetEncoder(
            config=config,
            scenario=scenario,
        )

    def _time_var(self):
        if (
            hasattr(self.petrinet, "semantics")
            and hasattr(self.petrinet.semantics, "ode")
            and hasattr(self.petrinet.semantics.ode, "time")
        ):
            return self.petrinet.semantics.ode.time
        else:
            return None

    def _time_var_id(self, time_var):
        if time_var:
            return f"timer_{time_var.id}"
        else:
            return None

    def _symbols(self):
        symbols = self._state_var_names() + self._parameter_names()
        if self._time_var():
            symbols += [self._time_var().id]
        return symbols

    def _get_init_value(self, var: str, scenario: "AnalysisScenario"):
        value = Model._get_init_value(self, var, scenario)
        if value is None:
            if hasattr(self.petrinet.semantics, "ode"):
                initials = self.petrinet.semantics.ode.initials
                value = next(i.expression for i in initials if i.target == var)
            else:
                value = f"{var}0"

        try:  # to cast to float
            value = float(value)
        except:
            pass

        if isinstance(value, float):
            value = Real(value)
        if isinstance(value, int):
            value = Real(float(value))
        elif isinstance(value, str):
            value = Symbol(value, REAL)

        if scenario.normalization_constant:
            value = Div(value, Real(scenario.normalization_constant))

        return value

    def _parameter_lb(self, param_name: str):
        return next(
            (
                p.distribution.parameters["minimum"]
                if p.distribution
                else p.value
            )
            for p in self.petrinet.semantics.ode.parameters
            if p.id == param_name
        )

    def _parameter_ub(self, param_name: str):
        return next(
            (
                p.distribution.parameters["maximum"]
                if p.distribution
                else p.value
            )
            for p in self.petrinet.semantics.ode.parameters
            if p.id == param_name
        )

    def _state_vars(self) -> List[State]:
        return self.petrinet.model.states

    def _state_var_names(self) -> List[str]:
        return [self._state_var_name(s) for s in self._state_vars()]

    def _transitions(self) -> List[Transition]:
        return self.petrinet.model.transitions

    def _state_var_name(self, state_var: State) -> str:
        return state_var.id

    def _input_edges(self):
        return [(i, t.id) for t in self._transitions() for i in t.input]

    def _edge_source(self, edge):
        return edge[0]

    def _edge_target(self, edge):
        return edge[1]

    def _output_edges(self):
        return [(t.id, o) for t in self._transitions() for o in t.output]

    def _transition_rate(self, transition, sympify=False):
        if hasattr(self.petrinet.semantics, "ode"):
            if transition.id not in self._transition_rates_cache:
                t_rates = [
                    (
                        to_sympy(r.expression, self._symbols())
                        if not any(
                            p == r.expression for p in self._parameter_names()
                        )
                        else r.expression
                    )
                    for r in self.petrinet.semantics.ode.rates
                    if r.target == transition.id
                ]
                time_var = self._time_var()
                if time_var:
                    t_rates = [
                        t.subs(
                            {
                                self._time_var().id: f"timer_{self._time_var().id}"
                            }
                        )
                        for t in t_rates
                    ]
                self._transition_rates_cache[transition.id] = t_rates
            return self._transition_rates_cache[transition.id]
        else:
            return transition.id

    def _transition_id(self, transition):
        return transition.id

    def _state_var_id(self, state_var):
        return self._state_var_name(state_var)

    def _parameter_names(self):
        if hasattr(self.petrinet.semantics, "ode"):
            return [p.id for p in self.petrinet.semantics.ode.parameters]
        else:
            # Create a parameter for each transition and initial state variable
            return [t.id for t in self.petrinet.model.transitions] + [
                f"{s.id}0" for s in self.petrinet.model.states
            ]

    def _parameter_values(self):
        if hasattr(self.petrinet.semantics, "ode"):
            return {
                p.id: p.value for p in self.petrinet.semantics.ode.parameters
            }
        else:
            return {}

    def _variable_values(self):
        values = {}
        if hasattr(self.petrinet.semantics, "ode"):
            for p in self.petrinet.semantics.ode.parameters:
                values[p.id] = p.value
            for p in self.petrinet.semantics.ode.initials:
                values[p.target] = p.expression
                try:
                    values[p.target] = float(values[p.target])
                except:
                    pass

        return values


class PetrinetDynamics(BaseModel):
    json_graph: Dict[
        str,
        List[
            Dict[
                str, Union[Dict[str, Union[str, bool, float]], int, str, float]
            ]
        ],
    ]

    # def __init__(self, **kwargs):
    #     super().__init__(**kwargs)
    #     self.json_graph = kwargs["json_graph"]
    #     self._initialize_from_json()


class PetrinetModel(AbstractPetriNetModel):
    petrinet: PetrinetDynamics

    def default_encoder(self, config: "FUNMANConfig") -> "Encoder":
        """
        Return the default Encoder for the model

        Returns
        -------
        Encoder
            SMT encoder for model
        """
        return PetrinetEncoder(
            config=config,
            model=self,
        )

    def _state_vars(self):
        return self.petrinet.json_graph["S"]

    def _state_var_names(self):
        return [self._state_var_name(s) for s in self.petrinet.json_graph["S"]]

    def _state_var_name(self, state_var: Dict) -> str:
        return state_var["sname"]

    def _transitions(self):
        return self.petrinet.json_graph["T"]

    def _input_edges(self):
        return self.petrinet.json_graph["I"]

    def _output_edges(self):
        return self.petrinet.json_graph["O"]

    def _edge_source(self, edge):
        return edge["is"] if "is" in edge else edge["ot"]

    def _edge_target(self, edge):
        return edge["it"] if "it" in edge else edge["os"]

    def _transition_rate(self, transition):
        return self._encode_state_var(transition["tprop"]["parameter_name"])

    def _transition_id(self, transition):
        return next(
            i + 1
            for i, s in enumerate(self._transitions())
            if s["tname"] == transition["tname"]
        )

    def _state_var_id(self, state_var):
        return next(
            i + 1
            for i, s in enumerate(self._state_vars())
            if s["sname"] == state_var["sname"]
        )

    def _parameter_names(self):
        return [t["tprop"]["parameter_name"] for t in self._transitions()]

    def _parameter_values(self):
        return {
            t["tprop"]["parameter_name"]: t["tprop"]["parameter_value"]
            for t in self._transitions()
        }
