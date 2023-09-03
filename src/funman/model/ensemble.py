from typing import Dict, List, Tuple

import graphviz
from pysmt.shortcuts import REAL, Div, Real, Symbol

from funman.representation import ModelParameter

from .model import Model
from pydantic import ConfigDict


class EnsembleModel(Model):
    models: List[Model]
    _model_name_map: Dict[str, Model] = None
    _var_name_map: Dict[str, Tuple[str, Model]] = None
    _parameter_name_map: Dict[str, Tuple[str, Model]] = None
    _parameter_map: Dict[str, ModelParameter] = None
    model_config = ConfigDict()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.models = kwargs["models"]
        self._initialize_mappings()

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
        from funman.translate import EnsembleEncoder

        return EnsembleEncoder(config=config, scenario=scenario)

    def calculate_normalization_constant(
        self, scenario: "AnalysisScenario", config: "FUNMANConfig"
    ) -> float:
        return max(
            m.calculate_normalization_constant(scenario, config)
            for m in self.models
        )

    def _get_init_value(
        self, var: str, scenario: "AnalysisScenario", config: "FUNMANConfig"
    ):
        (m_name, orig_var) = self._var_name_map[var]
        value = self._model_name_map[m_name].init_values[orig_var]
        if isinstance(value, str):
            value = Symbol(value, REAL)
        else:
            value = Real(value)

        if scenario.normalization_constant:
            norm = Real(scenario.normalization_constant)
            value = Div(value, norm)
        return value

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
            p_name: ModelParameter(
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
        return {
            p_name: self._model_name_map[m_name[0]]._parameter_values()[
                m_name[1]
            ]
            for p_name, m_name in self._parameter_name_map.items()
        }

    def _parameters(self) -> List[ModelParameter]:
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
