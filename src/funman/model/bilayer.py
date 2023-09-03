from abc import ABC, abstractmethod
from typing import Dict, List, Literal, Optional, Union

import graphviz
from pydantic import ConfigDict, BaseModel, validator
from pysmt.formula import FNode
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

from funman.model import Model
from funman.representation.representation import ModelParameter


class BilayerMetadata(BaseModel):
    """
    Metadata for a BilayerNode
    """

    ref: Optional[str] = None
    type: Optional[Literal["float", "int"]] = None
    initial_value: Optional[Union[float, int]] = None
    lb: Optional[Union[float, int]] = None
    ub: Optional[Union[float, int]] = None


class BilayerNode(BaseModel):
    """
    Node in a BilayerGraph.
    """

    index: int
    parameter: str
    metadata: Optional[BilayerMetadata] = None

    def to_dot(self, dot, values={}):
        label = values[self.parameter] if self.parameter in values else ""
        return dot.node(
            self.parameter, _attributes={"label": f"{self.parameter}: {label}"}
        )

    def __hash__(self):
        return self.index


class BilayerStateNode(BilayerNode):
    """
    BilayerNode representing a state variable.
    """

    pass


class BilayerFluxNode(BilayerNode):
    """
    BilayerNode representing a flux.
    """

    pass


class BilayerEdge(BaseModel):
    src: BilayerNode
    tgt: BilayerNode

    def _get_label(self):
        return ""

    def to_dot(self, dot):
        """
        Create a dot object for visualizing the edge.


        Parameters
        ----------
        dot : graphviz.Graph
            Graph to add the edge.
        """
        dot.edge(self.src.parameter, self.tgt.parameter)


class BilayerPositiveEdge(BilayerEdge):
    """
    Class representing a positive influence between a FluxNode and a StateNode.
    """

    def _get_label(self):
        """
        Edge label

        Returns
        -------
        str
            Label of edge
        """
        return "positive"


class BilayerNegativeEdge(BilayerEdge):
    """
    Class representing a positive influence between a FluxNode and a StateNode.
    """

    def _get_label(self):
        """
        Edge label

        Returns
        -------
        str
            Label of edge
        """
        return "negative"

    def to_dot(self, dot):
        """
        Create a dot object for visualizing the edge.

        Parameters
        ----------
        dot : graphviz.Graph
            Graph to add the edge.
        """
        dot.edge(self.src.parameter, self.tgt.parameter, style="dashed")


class BilayerGraph(ABC, BaseModel):
    """
    Abstract representation of a Bilayer graph.
    """

    model_config = ConfigDict()

    json_graph: Dict
    _node_incoming_edges: Dict[BilayerNode, Dict[BilayerNode, BilayerEdge]] = {}
    _node_outgoing_edges: Dict[BilayerNode, Dict[BilayerNode, BilayerEdge]] = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.json_graph = kwargs["json_graph"]
        self._initialize_from_json()

    @abstractmethod
    def _initialize_from_json(self):
        pass

    def _get_json_node(self, node_dict, node_type, node_list, node_name):
        for indx, i in enumerate(node_list):
            node_dict[indx + 1] = node_type(
                index=indx + 1,
                parameter=i[node_name],
                metadata=i.get("metadata"),
            )

    def _get_json_edge(
        self, edge_type, edge_list, src, src_nodes, tgt, tgt_nodes
    ):
        edges = []
        for json_edge in edge_list:
            edge = edge_type(
                src=src_nodes[json_edge[src]], tgt=tgt_nodes[json_edge[tgt]]
            )
            edges.append(edge)

            if edge.tgt not in self._node_incoming_edges:
                self._node_incoming_edges[edge.tgt] = {}
            self._node_incoming_edges[edge.tgt][edge.src] = edge
            if edge.src not in self._node_outgoing_edges:
                self._node_outgoing_edges[edge.src] = {}
            self._node_outgoing_edges[edge.src][edge.tgt] = edge

        return edges

    def _incoming_edges(self, node: "BilayerNode"):
        return self._node_incoming_edges[node]


class BilayerDynamics(BilayerGraph):
    """
    The BilayerDynamics class represents a state update (dynamics) model for a set of variables.  The graph consists of:

    * state nodes (current state),

    * tangent nodes (next state), and

    * flux nodes (causal relationships).
    """

    model_config = ConfigDict()

    _tangent: Dict[
        int, BilayerStateNode
    ] = {}  # Output layer variables, defined in Qout
    _flux: Dict[
        int, BilayerFluxNode
    ] = {}  # Functions, defined in Box, one param per flux
    _state: Dict[
        int, BilayerStateNode
    ] = {}  # Input layer variables, defined in Qin
    _input_edges: BilayerEdge = []  # Input to flux, defined in Win
    _output_edges: BilayerEdge = []  # Flux to Output, defined in Wa,Wn

    # def initialize(self):
    #     if self.json_graph:
    #         self.initialize_from_json()
    #     else:
    #         raise Exception(
    #             f"Cannot initilize BilayerDynamics without self.json_graph"
    #         )

    # @staticmethod
    # def from_json(bilayer_src: Union[str, Dict]):
    #     bilayer = BilayerDynamics(json_graph=bilayer_src)
    #     bilayer.initialize_from_json()
    #     return bilayer

    def _initialize_from_json(self):
        """
        Create a BilayerDynamics object from a JSON formatted bilayer graph.

        Parameters
        ----------
        bilayer_src : str (filename) or dictionary
            The source bilayer representation in a file or dictionary.

        Returns
        -------
        BilayerDynamics
            The BilayerDynamics object corresponding to the bilayer_src.
        """

        # if isinstance(self.json_graph, dict):
        data = self.json_graph
        # else:
        #     with open(self.json_graph, "r") as f:
        #         data = json.load(f)

        # Get the input state variable nodes
        self._get_json_to_statenodes(data)

        # Get the output state variable nodes (tangent)
        self._get_json_to_tangents(data)

        # Get the flux nodes
        self._get_json_to_flux(data)

        # Get the input edges
        self._get_json_to_input_edges(data)

        # Get the output edges
        self._get_json_to_output_edges(data)

    def _get_json_to_statenodes(self, data):
        self._get_json_node(
            self._state, BilayerStateNode, data["Qin"], "variable"
        )

    def _get_json_to_tangents(self, data):
        self._get_json_node(
            self._tangent, BilayerStateNode, data["Qout"], "tanvar"
        )

    def _get_json_to_flux(self, data):
        self._get_json_node(
            self._flux, BilayerFluxNode, data["Box"], "parameter"
        )

    def _get_json_to_input_edges(self, data):
        self._input_edges += self._get_json_edge(
            BilayerEdge, data["Win"], "arg", self._state, "call", self._flux
        )

    def _get_json_to_output_edges(self, data):
        self._output_edges += self._get_json_edge(
            BilayerPositiveEdge,
            data["Wa"],
            "influx",
            self._flux,
            "infusion",
            self._tangent,
        )
        self._output_edges += self._get_json_edge(
            BilayerNegativeEdge,
            data["Wn"],
            "efflux",
            self._flux,
            "effusion",
            self._tangent,
        )

    def to_dot(self, values={}):
        """
        Create a dot object for visualizing the graph.

        Returns
        -------
        graphviz.Digraph
            The graph represented by self.
        """
        dot = graphviz.Digraph(
            name=f"bilayer",
            graph_attr={
                #    'label': self.name,
                "shape": "box"
            },
        )
        for n in (
            list(self._tangent.values())
            + list(self._flux.values())
            + list(self._state.values())
        ):
            n.to_dot(dot, values=values)
        for e in self._input_edges + self._output_edges:
            e.to_dot(dot)
        return dot

    def _state_var_names(self) -> List[str]:
        return [v.parameter for v in self._state.values()]


class BilayerMeasurement(BilayerGraph, BaseModel):
    """
    The BilayerMeasurement class represents measurements taken on the BilayerNode state nodes of a BilayerDynamics object.  The graph consists of:

    * state nodes (current state),

    * observation nodes, and

    * flux nodes (causal relationships).
    """

    state: Dict[int, BilayerStateNode] = {}
    flux: Dict[
        int, BilayerFluxNode
    ] = {}  # Functions, defined in observable, one param per flux
    observable: Dict[int, BilayerStateNode] = {}
    input_edges: BilayerEdge = []  # Input to observable, defined in Win
    output_edges: BilayerEdge = []  # Flux to Output, defined in Wa,Wn

    def _state_var_names(self) -> List[str]:
        return [v.parameter for v in model.bilayer._state.values()]

    def _initialize_from_json(self):
        """
        Create a BilayerMeasurement object from a JSON formatted bilayer graph.

        Parameters
        ----------
        bilayer_src : str (filename) or dictionary
            The source bilayer representation in a file or dictionary.

        Returns
        -------
        BilayerMeasurement
            The BilayerMeasurement object corresponding to the bilayer_src.
        """
        # measurement = BilayerMeasurement()
        # if isinstance(src, dict):
        #     data = src
        # else:
        #     with open(src, "r") as f:
        #         data = json.load(f)
        data = self.json_graph
        # TODO extract measurment graph
        # blm = BilayerMeasurement()

        # Get the input state variable nodes
        self._get_json_node(
            self.state, BilayerStateNode, data["state"], "variable"
        )

        # Get the output state variable nodes (tangent)
        self._get_json_node(
            self.observable, BilayerStateNode, data["observable"], "observable"
        )

        # Get the flux nodes
        self._get_json_node(
            self.flux, BilayerFluxNode, data["rate"], "parameter"
        )

        # Get the input edges
        self.input_edges += self._get_json_edge(
            BilayerEdge,
            data["Din"],
            "variable",
            self.state,
            "parameter",
            self.flux,
        )

        # Get the output edges
        self.output_edges += self._get_json_edge(
            BilayerPositiveEdge,
            data["Dout"],
            "parameter",
            self.flux,
            "observable",
            self.observable,
        )

        # return blm

    def to_dot(self, values={}):
        """
        Create a dot object for visualizing the graph.

        Returns
        -------
        graphviz.Digraph
            The graph represented by self.
        """
        dot = graphviz.Digraph(
            name=f"bilayer_measurement",
            graph_attr={
                #    'label': self.name,
                "shape": "box"
            },
        )
        for n in (
            list(self.state.values())
            + list(self.flux.values())
            + list(self.observable.values())
        ):
            n.to_dot(dot, values=values)
        for e in self.input_edges + self.output_edges:
            e.to_dot(dot)
        return dot


class BilayerModel(Model):
    """
    A BilayerModel is a complete specification of a Model that uses a BilayerDynamics graph to represent dynamics. It includes the attributes:

    * bilayer: the BilayerDynamics graph

    * measurements: the BilayerMeasurement graph (used to derive additional variables from the state nodes)

    * init_values: a dict mapping from state variables and flux parameters to initial value

    * identical_parameters: a list of lists of flux parameters that have identical values

    * parameter_bounds: a list of lower and upper bounds on parameters

    """

    model_config = ConfigDict()

    bilayer: BilayerDynamics
    measurements: BilayerMeasurement = None
    identical_parameters: List[List[str]] = []

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
        from funman.translate import BilayerEncoder

        return BilayerEncoder(config=config, scenario=scenario)

    def _state_var_names(self):
        return self.bilayer._state_var_names()

    def _parameter_names(self):
        param_names = [node.parameter for _, node in self.bilayer._flux.items()]
        if self.measurements:
            param_names += [
                node.parameter for _, node in self.measurements._flux.items()
            ]
        return param_names

    def _parameters(self) -> List[ModelParameter]:
        params = [
            ModelParameter(
                name=node.parameter,
                lb=self.parameter_bounds[node.parameter][0],
                ub=self.parameter_bounds[node.parameter][1],
            )
            for _, node in self.bilayer._flux.items()
            if self.parameter_bounds
            and node.parameter in self.parameter_bounds
            and self.parameter_bounds[node.parameter]
        ]
        if self.measurements:
            params += [
                ModelParameter(
                    name=node.parameter,
                    lb=self.parameter_bounds[node.parameter][0],
                    ub=self.parameter_bounds[node.parameter][1],
                )
                for _, node in self.measurements._flux.items()
                if node.parameter in self.parameter_bounds
                and self.parameter_bounds[node.parameter]
            ]
        return params

    def _parameter_lb(self, p):
        return self.parameter_bounds[p][0]

    def _parameter_ub(self, p):
        return self.parameter_bounds[p][1]

    def calculate_normalization_constant(
        self, scenario: "AnalysisScenario", config: "FUNMANConfig"
    ) -> float:
        vars = self._state_var_names()
        values = {v: self._get_init_value(v, scenario, config) for v in vars}
        if all(v.is_constant() for v in values.values()):
            return float(sum(v.constant_value() for v in values.values()))
        else:
            raise Exception(
                f"Cannot calculate the normalization constant for {type(self)} because the initial state variables are not constants. Try setting the 'normalization_constant' in the configuration to constant."
            )
