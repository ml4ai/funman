from typing import List

import graphviz
from pydantic import BaseModel

from funman.representation import Parameter
from funman.translate import EnsembleEncoder

from .model import Model


class EnsembleModel(Model):
    models: List[Model]

    def default_encoder(self, config: "FUNMANConfig") -> "Encoder":
        """
        Return the default Encoder for the model

        Returns
        -------
        Encoder
            SMT encoder for model
        """
        return EnsembleEncoder(
            config=config,
            model=self,
        )

    def _get_init_value(self, var: str):
        return self.init_values[var]

    def _state_vars(self):
        return map(lambda m: m._state_vars(), self.models)

    def _state_var_names(self):
        return map(lambda m: m._state_var_names(), self.models)

    def _parameter_names(self):
        return map(lambda m: m._parameter_names(), self.models)

    def _parameter_values(self):
        return map(lambda m: m._parameter_values(), self.models)

    def _parameters(self) -> List[Parameter]:
        return map(lambda m: m._parameters(), self.models)

    def to_dot(self, values={}):
        """
        Create a dot object for visualizing the graph.

        Returns
        -------
        graphviz.Digraph
            The graph represented by self.
        """
        dot = graphviz.Digraph(
            name=f"ensemble",
            graph_attr={},
        )

        for m in self.models:
            dot.subgraph(m.to_dot(), values=values)

        return dot
