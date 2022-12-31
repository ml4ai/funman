import json
from abc import ABC, abstractmethod
from typing import Dict, List, Union

import graphviz
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


class BilayerGraph(ABC):
    """
    Abstract representation of a Bilayer graph.
    """

    def __init__(self):
        self.node_incoming_edges = {}
        self.node_outgoing_edges = {}

    def _get_json_node(self, node_dict, node_type, node_list, node_name):
        for indx, i in enumerate(node_list):
            node_dict[indx + 1] = node_type(indx + 1, i[node_name])

    def _get_json_edge(
        self, edge_type, edge_list, src, src_nodes, tgt, tgt_nodes
    ):
        edges = []
        for json_edge in edge_list:
            edge = edge_type(
                src_nodes[json_edge[src]], tgt_nodes[json_edge[tgt]]
            )
            edges.append(edge)

            if edge.tgt not in self.node_incoming_edges:
                self.node_incoming_edges[edge.tgt] = {}
            self.node_incoming_edges[edge.tgt][edge.src] = edge
            if edge.src not in self.node_outgoing_edges:
                self.node_outgoing_edges[edge.src] = {}
            self.node_outgoing_edges[edge.src][edge.tgt] = edge

        return edges

    def _incoming_edges(self, node: "BilayerNode"):
        return self.node_incoming_edges[node]


class BilayerMeasurement(BilayerGraph):
    """
    The BilayerMeasurement class represents measurements taken on the BilayerNode state nodes of a BilayerDynamics object.  The graph consists of:

    * state nodes (current state),

    * observation nodes, and

    * flux nodes (causal relationships).
    """

    def __init__(self) -> None:
        super().__init__()
        self.state: Dict[int, BilayerStateNode] = {}
        self.flux: Dict[
            int, BilayerFluxNode
        ] = {}  # Functions, defined in observable, one param per flux
        self.observable: Dict[int, BilayerStateNode] = {}
        self.input_edges: BilayerEdge = (
            []
        )  # Input to observable, defined in Win
        self.output_edges: BilayerEdge = []  # Flux to Output, defined in Wa,Wn

    @staticmethod
    def from_json(src: Union[str, Dict]):
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
        measurement = BilayerMeasurement()
        if isinstance(src, dict):
            data = src
        else:
            with open(src, "r") as f:
                data = json.load(f)

        # TODO extract measurment graph
        blm = BilayerMeasurement()

        # Get the input state variable nodes
        blm._get_json_node(
            blm.state, BilayerStateNode, data["state"], "variable"
        )

        # Get the output state variable nodes (tangent)
        blm._get_json_node(
            blm.observable, BilayerStateNode, data["observable"], "observable"
        )

        # Get the flux nodes
        blm._get_json_node(
            blm.flux, BilayerFluxNode, data["rate"], "parameter"
        )

        # Get the input edges
        blm.input_edges += blm._get_json_edge(
            BilayerEdge,
            data["Din"],
            "variable",
            blm.state,
            "parameter",
            blm.flux,
        )

        # Get the output edges
        blm.output_edges += blm._get_json_edge(
            BilayerPositiveEdge,
            data["Dout"],
            "parameter",
            blm.flux,
            "observable",
            blm.observable,
        )

        return blm

    def to_dot(self):
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
            n.to_dot(dot)
        for e in self.input_edges + self.output_edges:
            e.to_dot(dot)
        return dot


class BilayerNode(object):
    """
    Node in a BilayerGraph.
    """

    def __init__(self, index, parameter):
        self.index = index
        self.parameter = parameter

    def to_dot(self, dot):
        return dot.node(self.parameter)


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


class BilayerEdge(object):
    def __init__(self, src, tgt):
        self.src = src
        self.tgt = tgt

    @abstractmethod
    def get_label(self):
        pass

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

    def get_label(self):
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

    def get_label(self):
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


class BilayerModel(Model):
    """
    A BilayerModel is a complete specification of a Model that uses a BilayerDynamics graph to represent dynamics. It includes the attributes:

    * bilayer: the BilayerDynamics graph

    * measurements: the BilayerMeasurement graph (used to derive additional variables from the state nodes)

    * init_values: a dict mapping from state variables and flux parameters to initial value

    * identical_parameters: a list of lists of flux parameters that have identical values

    * parameter_bounds: a list of lower and upper bounds on parameters

    """

    def __init__(
        self,
        bilayer: "BilayerDynamics",
        measurements: BilayerMeasurement = None,
        init_values: Dict[str, float] = None,
        identical_parameters: List[List[str]] = [],
        parameter_bounds: Dict[str, List[float]] = None,
    ) -> None:
        super().__init__(
            init_values=init_values, parameter_bounds=parameter_bounds
        )
        self.bilayer = bilayer
        self.measurements = measurements
        self.identical_parameters = identical_parameters

    def default_encoder(self) -> "Encoder":
        """
        Return the default Encoder for the model

        Returns
        -------
        Encoder
            SMT encoder for model
        """
        from funman.translate import BilayerEncoder, BilayerEncodingOptions

        return BilayerEncoder(config=BilayerEncodingOptions())


class BilayerDynamics(BilayerGraph):
    """
    The BilayerDynamics class represents a state update (dynamics) model for a set of variables.  The graph consists of:

    * state nodes (current state),

    * tangent nodes (next state), and

    * flux nodes (causal relationships).
    """

    def __init__(self):
        super().__init__()
        self.tangent: Dict[
            int, BilayerStateNode
        ] = {}  # Output layer variables, defined in Qout
        self.flux: Dict[
            int, BilayerFluxNode
        ] = {}  # Functions, defined in Box, one param per flux
        self.state: Dict[
            int, BilayerStateNode
        ] = {}  # Input layer variables, defined in Qin
        self.input_edges: BilayerEdge = []  # Input to flux, defined in Win
        self.output_edges: BilayerEdge = []  # Flux to Output, defined in Wa,Wn

    @staticmethod
    def from_json(bilayer_src: Union[str, Dict]):
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
        bilayer = BilayerDynamics()

        if isinstance(bilayer_src, dict):
            data = bilayer_src
        else:
            with open(bilayer_src, "r") as f:
                data = json.load(f)

        # Get the input state variable nodes
        bilayer._get_json_to_statenodes(data)

        # Get the output state variable nodes (tangent)
        bilayer._get_json_to_tangents(data)

        # Get the flux nodes
        bilayer._get_json_to_flux(data)

        # Get the input edges
        bilayer._get_json_to_input_edges(data)

        # Get the output edges
        bilayer._get_json_to_output_edges(data)

        return bilayer

    def _get_json_to_statenodes(self, data):
        self._get_json_node(
            self.state, BilayerStateNode, data["Qin"], "variable"
        )

    def _get_json_to_tangents(self, data):
        self._get_json_node(
            self.tangent, BilayerStateNode, data["Qout"], "tanvar"
        )

    def _get_json_to_flux(self, data):
        self._get_json_node(
            self.flux, BilayerFluxNode, data["Box"], "parameter"
        )

    def _get_json_to_input_edges(self, data):
        self.input_edges += self._get_json_edge(
            BilayerEdge, data["Win"], "arg", self.state, "call", self.flux
        )

    def _get_json_to_output_edges(self, data):
        self.output_edges += self._get_json_edge(
            BilayerPositiveEdge,
            data["Wa"],
            "influx",
            self.flux,
            "infusion",
            self.tangent,
        )
        self.output_edges += self._get_json_edge(
            BilayerNegativeEdge,
            data["Wn"],
            "efflux",
            self.flux,
            "effusion",
            self.tangent,
        )

    def to_dot(self):
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
            list(self.tangent.values())
            + list(self.flux.values())
            + list(self.state.values())
        ):
            n.to_dot(dot)
        for e in self.input_edges + self.output_edges:
            e.to_dot(dot)
        return dot
