import json
from typing import Dict

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


class BilayerGraph(object):
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

    def from_json(src):
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
    def __init__(self, index, parameter):
        self.index = index
        self.parameter = parameter

    def to_dot(self, dot):
        return dot.node(self.parameter)


class BilayerStateNode(BilayerNode):
    pass


class BilayerFluxNode(BilayerNode):
    pass


class BilayerEdge(object):
    def __init__(self, src, tgt):
        self.src = src
        self.tgt = tgt

    def get_label(self):
        pass

    def to_dot(self, dot):
        dot.edge(self.src.parameter, self.tgt.parameter)


class BilayerPositiveEdge(BilayerEdge):
    def get_label(self):
        return "positive"


class BilayerNegativeEdge(BilayerEdge):
    def get_label(self):
        return "negative"

    def to_dot(self, dot):
        dot.edge(self.src.parameter, self.tgt.parameter, style="dashed")


class BilayerModel(Model):
    def __init__(
        self,
        bilayer,
        measurements=None,
        init_values=None,
        identical_parameters=[],
        parameter_bounds=None,
    ) -> None:
        super().__init__(
            init_values=init_values, parameter_bounds=parameter_bounds
        )
        self.bilayer = bilayer
        self.measurements = measurements
        self.identical_parameters = identical_parameters


class Bilayer(BilayerGraph):
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

    def from_json(bilayer_src):
        bilayer = Bilayer()

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
