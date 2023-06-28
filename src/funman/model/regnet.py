from typing import Any, Dict, List, Union

import graphviz
from pydantic import BaseModel

from funman.representation.representation import Parameter
from funman.translate.regnet import RegnetEncoder

from .generated_models.regnet import Edge as GeneratedRegnetEdge
from .generated_models.regnet import Model as GeneratedRegnet
from .generated_models.regnet import Parameter as GeneratedRegnetParameter
from .generated_models.regnet import Vertice as GeneratedRegnetVertice
from .model import Model


class AbstractRegnetModel(Model):
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
        return RegnetEncoder(
            config=config,
            scenario=scenario,
        )

    def _state_var(self, var_id: str):
        state_vars = self._state_vars()
        try:
            state_var = next(
                s for s in state_vars if self._vertice_id(s) == var_id
            )
            return state_var
        except StopIteration as e:
            raise Exception(f"Could not find state var with id {var_id}: {e}")

    def _parameter_names(self):
        transition_parameters = [
            self._transition_rate_constant(e)
            for e in self._transitions()
            if isinstance(self._transition_rate_constant(e), str)
        ]
        declared_parameters = [
            self._parameter_id(t) for t in self._declared_parameters()
        ]
        return declared_parameters + transition_parameters


class GeneratedRegnetModel(AbstractRegnetModel):
    regnet: GeneratedRegnet

    def _transitions(self) -> List[GeneratedRegnetEdge]:
        return self.regnet.model.edges

    def _state_vars(self) -> List[GeneratedRegnetVertice]:
        return self.regnet.model.vertices

    def _state_var_names(self) -> List[str]:
        return [s.id for s in self._state_vars()]

    def _transition_source(self, transition: GeneratedRegnetEdge):
        return transition.source

    def _transition_target(self, transition: GeneratedRegnetEdge):
        return transition.target

    def _transition_sign(self, transition: GeneratedRegnetEdge):
        return transition.sign

    def _vertice_id(self, vertice: GeneratedRegnetVertice):
        return vertice.id

    def _vertice_sign(self, vertice: GeneratedRegnetVertice):
        return vertice.sign

    def _vertice_rate_constant(self, vertex: GeneratedRegnetVertice):
        return vertex.rate_constant.__root__

    def _parameter_id(self, parameter: GeneratedRegnetParameter):
        return parameter.id

    def _declared_parameters(self) -> List[GeneratedRegnetParameter]:
        return self.regnet.model.parameters

    def _parameter_values(self):
        return {
            self._parameter_id(t): t.value for t in self.regnet.model.parameters
        }

    def _transition_rate_constant(self, transitition: GeneratedRegnetEdge):
        return (
            transitition.properties.rate_constant.__root__
            if transitition.properties and transitition.properties.rate_constant
            else transitition.id
        )

    def _get_init_value(self, var: str):
        value = Model._get_init_value(self, var)
        if value is None:
            state_var = next(s for s in self._state_vars() if s.id == var)
            value = state_var.initial.__root__
        return value


class RegnetDynamics(BaseModel):
    json_graph: Dict[str, Union[str, Dict[str, List[Dict[str, Any]]]]]


class RegnetModel(AbstractRegnetModel):
    regnet: RegnetDynamics

    def _get_init_value(self, var: str):
        value = Model._get_init_value(self, var)
        if value is None:
            state_var = next(s for s in self._state_vars() if s["id"] == var)
            if "initial" in state_var:
                value = state_var["initial"]
        return value

    def _state_vars(self):
        return self.regnet.json_graph["model"]["vertices"]

    def _state_var_names(self):
        return [s["id"] for s in self._state_vars()]

    def _vertice_id(self, vertice):
        return vertice["id"]

    def _vertice_sign(self, vertice):
        return vertice["sign"]

    def _vertice_rate_constant(self, vertex):
        return vertex["rate_constant"]

    def _transitions(self):
        return self.regnet.json_graph["model"]["edges"]

    def _transition_source(self, transition):
        return transition["source"]

    def _transition_target(self, transition):
        return transition["target"]

    def _transition_sign(self, transition):
        return transition["sign"]

    def _transition_rate_constant(self, transition):
        return (
            transition["properties"]["rate_constant"]
            if "rate_constant" in transition["properties"]
            else transition["id"]
        )

    def _parameter_id(self, parameter):
        return parameter["id"]

    def _declared_parameters(self):
        return self.regnet.json_graph["model"]["parameters"]

    def _input_edges(self):
        return self.regnet.json_graph["I"]

    def _output_edges(self):
        return self.regnet.json_graph["O"]

    def _parameter_values(self):
        return {
            t["id"]: t["value"]
            for t in self.regnet.json_graph["model"]["parameters"]
        }

    def _num_flow_from_transition_to_state(
        self, state_index: int, transition_index: int
    ) -> int:
        return len(
            [
                edge
                for edge in self._output_edges()
                if edge["os"] == state_index and edge["ot"] == transition_index
            ]
        )

    def _num_flow_from_state_to_transition(
        self, state_index: int, transition_index: int
    ) -> int:
        return len(
            [
                edge
                for edge in self._input_edges()
                if edge["is"] == state_index and edge["it"] == transition_index
            ]
        )

    def _num_flow_from_transition(self, transition_index: int) -> int:
        return len(
            [
                edge
                for edge in self._output_edges()
                if edge["ot"] == transition_index
            ]
        )

    def _num_flow_into_transition(self, transition_index: int) -> int:
        return len(
            [
                edge
                for edge in self._input_edges()
                if edge["it"] == transition_index
            ]
        )

    def _flow_into_state_via_transition(
        self, state_index: int, transition_index: int
    ) -> float:
        num_flow_to_transition = self._num_flow_into_transition(
            transition_index
        )
        num_inflow = self._num_flow_from_transition_to_state(
            state_index, transition_index
        )
        num_transition_outputs = self._num_flow_from_transition(
            transition_index
        )
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
            name=f"regnet",
            graph_attr={},
        )

        state_vars = self._state_vars()
        transitions = self._transitions()

        for v_index, var in enumerate(state_vars):
            state_var_name = var["sname"]
            for t_index, transition in enumerate(transitions):
                transition_name = f"{transition['tname']}({transition['tprop']['parameter_name']}) = {transition['tprop']['parameter_value']}"
                dot.node(transition_name, _attributes={"shape": "box"})
                # state var to transition
                for edge in self._input_edges():
                    if edge["is"] == v_index + 1 and edge["it"] == t_index + 1:
                        dot.edge(state_var_name, transition_name)
                # transition to state var
                for edge in self._output_edges():
                    if edge["os"] == v_index + 1 and edge["ot"] == t_index + 1:
                        flow = self._flow_into_state_via_transition(
                            v_index + 1, t_index + 1
                        ) / self._num_flow_from_transition_to_state(
                            v_index + 1, t_index + 1
                        )
                        dot.edge(
                            transition_name, state_var_name, label=f"{flow}"
                        )

        return dot
