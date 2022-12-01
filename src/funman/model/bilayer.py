import json
import graphviz
from typing import Dict, List, Union
from funman.model import Model
from pysmt.shortcuts import (
    get_model,
    And,
    Symbol,
    FunctionType,
    Function,
    Equals,
    Int,
    Real,
    substitute,
    TRUE,
    FALSE,
    Iff,
    Plus,
    Times,
    ForAll,
    simplify,
    LT,
    LE,
    GT,
    GE,
)
from pysmt.typing import INT, REAL, BOOL


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
        blm._get_json_node(blm.flux, BilayerFluxNode, data["rate"], "parameter")

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

    def to_smtlib(self, timepoint):
        param = self.parameter
        ans = Symbol(f"{param}_{timepoint}", REAL)
        return ans


class BilayerStateNode(BilayerNode):
    pass


class BilayerFluxNode(BilayerNode):
    pass


class BilayerEdge(object):
    def __init__(self, src, tgt):
        self.src = src
        self.tgt = tgt

    def to_smtlib(self, timepoint):
        pass

    def to_dot(self, dot):
        dot.edge(self.src.parameter, self.tgt.parameter)


class BilayerPositiveEdge(BilayerEdge):
    def to_smtlib(self, timepoint):
        return "positive"


class BilayerNegativeEdge(BilayerEdge):
    def to_smtlib(self, timepoint):
        return "negative"

    def to_dot(self, dot):
        dot.edge(self.src.parameter, self.tgt.parameter, style="dashed")


class BilayerModel(Model):
    def __init__(
        self,
        bilayer,
        measurements=None,
        init_values=None,
        parameter_bounds=None,
    ) -> None:
        super().__init__(init_values=init_values, parameter_bounds=parameter_bounds)
        self.bilayer = bilayer
        self.measurements = measurements
       


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

    def to_smtlib(self, timepoints):
        #        ans = simplify(And([self.to_smtlib_timepoint(t) for t in timepoints]))
        ans = simplify(
            And(
                [
                    self.to_smtlib_timepoint(timepoints[i], timepoints[i + 1])
                    for i in range(len(timepoints) - 1)
                ]
            )
        )
        # print(ans)
        return ans

    def to_smtlib_timepoint(
        self, timepoint, next_timepoint
    ):  ## TODO remove prints
        ## Calculate time step size
        time_step_size = next_timepoint - timepoint
        # print("timestep size:", time_step_size)
        eqns = (
            []
        )  ## List of SMT equations for a given timepoint. These will be joined by an "And" command and returned
        for t in self.tangent:  ## Loop over tangents (derivatives)
            derivative_expr = 0
            ## Get tangent variable and translate it to SMT form tanvar_smt
            tanvar = self.tangent[t].parameter
            tanvar_smt = self.tangent[t].to_smtlib(timepoint)
            state_var_next_step = self.state[t].parameter
            state_var_smt = self.state[t].to_smtlib(timepoint)
            state_var_next_step_smt = self.state[t].to_smtlib(next_timepoint)
            #            state_var_next_step_smt = self.state[t].to_smtlib(timepoint + 1)
            relevant_output_edges = [
                (val, val.src.index)
                for val in self.output_edges
                if val.tgt.index == self.tangent[t].index
            ]
            for flux_sign_index in relevant_output_edges:
                flux_term = self.flux[flux_sign_index[1]]
                output_edge = self.output_edges[flux_sign_index[1]]
                expr = flux_term.to_smtlib(timepoint)
                ## Check which state vars go to that param
                relevant_input_edges = [
                    self.state[val2.src.index].to_smtlib(timepoint)
                    for val2 in self.input_edges
                    if val2.tgt.index == flux_sign_index[1]
                ]
                for state_var in relevant_input_edges:
                    expr = Times(expr, state_var)
                if flux_sign_index[0].to_smtlib(timepoint) == "positive":
                    derivative_expr += expr
                elif flux_sign_index[0].to_smtlib(timepoint) == "negative":
                    derivative_expr -= expr
            ## Assemble into equation of the form f(t + delta t) approximately = f(t) + (delta t) f'(t)
            eqn = simplify(
                Equals(
                    state_var_next_step_smt,
                    Plus(state_var_smt, time_step_size * derivative_expr),
                )
            )
            # print(eqn)
            eqns.append(eqn)
        return And(eqns)
