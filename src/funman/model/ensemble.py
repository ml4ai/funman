from typing import Dict, List, Tuple

import graphviz
from pydantic import BaseModel

from funman.representation import Parameter
from funman.translate import EnsembleEncoder

from .model import Model


class EnsembleModel(Model):
    models: List[Model]
    _model_name_map: Dict[str, Model] = None
    _var_name_map: Dict[str, Tuple[str, Model]] = None
    _parameter_name_map: Dict[str, Tuple[str, Model]] = None
    _parameter_map: Dict[str, Parameter] = None

    class Config:
        underscore_attrs_are_private = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.models = kwargs["models"]
        self._initialize_mappings()

    def default_encoder(self, config: "FUNMANConfig") -> "Encoder":
        """
        Return the default Encoder for the model

        Returns
        -------
        Encoder
            SMT encoder for model
        """
        from funman.translate import EnsembleEncoder

        return EnsembleEncoder(
            config=config,
            model=self,
        )

    def _get_init_value(self, var: str):
        (m_name, orig_var) = self._var_name_map[var]
        return self._model_name_map[m_name].init_values[orig_var]

    def _state_vars(self):
        return map(lambda m: m._state_vars(), self.models)

    def _initialize_mappings(self):
        self._model_name_map = {m.name: m for m in self.models}
        model_vars = {m.name: m._state_var_names() for m in self.models}
        self._var_name_map = {
            f"model_{m_name}_{v}": (m_name, v)
            for m_name, _ in self._model_name_map.items()
            for v in model_vars[m_name]
        }
        model_parameters = {m.name: m._parameter_names() for m in self.models}
        self._parameter_name_map = {
            f"model_{m_name}_{p}": (m_name, p)
            for m_name, _ in self._model_name_map.items()
            for p in model_parameters[m_name]
        }
        self._parameter_map = {
            p_name: Parameter(
                name=p_name,
                lb=self._model_name_map[m_name].parameter_bounds[p][0],
                ub=self._model_name_map[m_name].parameter_bounds[p][1],
            )
            for p_name, (m_name, p) in self._parameter_name_map.items()
            if p in self._model_name_map[m_name].parameter_bounds
        }

    def _state_var_names(self):
        return list(self._var_name_map.keys())

    def _parameter_names(self):
        return list(self._parameter_name_map.keys())

    def _parameter_values(self):
        return map(lambda m: m._parameter_values(), self.models)

    def _parameters(self) -> List[Parameter]:
        return list(self._parameter_map.values())

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
