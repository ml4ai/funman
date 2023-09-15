import random
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
from matplotlib import pyplot as plt
from pydantic import BaseModel

from funman.funman import FUNMANConfig
from funman.model.bilayer import BilayerModel
from funman.model.decapode import DecapodeModel
from funman.model.encoded import EncodedModel
from funman.model.ensemble import EnsembleModel
from funman.model.petrinet import GeneratedPetriNetModel, PetrinetModel
from funman.model.query import QueryAnd, QueryFunction, QueryLE, QueryTrue
from funman.model.regnet import GeneratedRegnetModel, RegnetModel
from funman.representation import ModelParameter
from funman.representation.representation import (
    LABEL_ANY,
    LABEL_TRUE,
    LabeledParameter,
    ModelParameter,
    ParameterSpace,
    Point,
    StructureParameter,
)
from funman.scenario.consistency import (
    ConsistencyScenario,
    ConsistencyScenarioResult,
)
from funman.scenario.parameter_synthesis import (
    ParameterSynthesisScenario,
    ParameterSynthesisScenarioResult,
)
from funman.scenario.scenario import AnalysisScenario


class FunmanWorkRequest(BaseModel):
    query: Optional[Union[QueryAnd, QueryLE, QueryFunction, QueryTrue]] = None
    parameters: Optional[List[LabeledParameter]] = None
    config: Optional[FUNMANConfig] = None
    structure_parameters: Optional[List[LabeledParameter]] = None


class FunmanWorkUnit(BaseModel):
    """
    Fields
    ------
    id : The UUID assigned to the request
    request : A copy of the request associated with this response
    """

    id: str
    progress: float = 0.0
    model: Union[
        RegnetModel,
        PetrinetModel,
        DecapodeModel,
        BilayerModel,
        GeneratedRegnetModel,
        GeneratedPetriNetModel,
    ]
    request: FunmanWorkRequest

    def to_scenario(
        self,
    ) -> Union[ConsistencyScenario, ParameterSynthesisScenario]:
        query = (
            self.request.query
            if self.request.query is not None
            else QueryTrue()
        )
        parameters = []
        if (
            hasattr(self.request, "parameters")
            and self.request.parameters is not None
        ):
            for data in self.request.parameters:
                parameters.append(
                    ModelParameter(
                        name=data.name,
                        ub=data.ub,
                        lb=data.lb,
                        label=data.label,
                    )
                )
        if (
            hasattr(self.request, "structure_parameters")
            and self.request.structure_parameters is not None
        ):
            for data in self.request.structure_parameters:
                parameters.append(
                    StructureParameter(
                        name=data.name,
                        ub=data.ub,
                        lb=data.lb,
                        label=data.label,
                    )
                )

        if (
            not hasattr(self.request, "parameters")
            or self.request.parameters is None
            or all(p.label == LABEL_ANY for p in self.request.parameters)
        ):
            return ConsistencyScenario(
                model=self.model,
                query=query,
                parameters=parameters,
            )

        if isinstance(self.model, EnsembleModel):
            raise Exception(
                "TODO handle EnsembleModel for ParameterSynthesisScenario"
            )

        return ParameterSynthesisScenario(
            model=self.model, query=query, parameters=parameters
        )


class FunmanResults(BaseModel):
    class Config:
        underscore_attrs_are_private = True

    _finalized: bool = False

    id: str
    progress: float = 0.0
    model: Union[
        GeneratedRegnetModel,
        GeneratedPetriNetModel,
        RegnetModel,
        PetrinetModel,
        DecapodeModel,
        BilayerModel,
        EncodedModel,
    ]
    request: FunmanWorkRequest
    done: bool = False
    error: bool = False
    parameter_space: Optional[ParameterSpace] = None

    def is_final(self):
        return self._finalized

    def finalize_result(
        self,
        result: Union[
            ConsistencyScenarioResult, ParameterSynthesisScenarioResult
        ],
    ):
        if self._finalized:
            raise Exception("FunmanResults was already finalized")
        self._finalized = True
        ps = None
        if isinstance(result, ConsistencyScenarioResult):
            ps = result.parameter_space
        if isinstance(result, ParameterSynthesisScenarioResult):
            ps = result.parameter_space

        if ps is None:
            raise Exception("No ParameterSpace for result")

        self.parameter_space = ps
        self.done = True
        self.progress = 1.0

    def finalize_result_as_error(
        self,
    ):
        if self._finalized:
            raise Exception("FunmanResults was already finalized")
        self._finalized = True
        self.parameter_space = None
        self.error = True
        self.done = True
        self.progress = 1.0

    def dataframe(
        self, points: List[Point], interpolate="linear", max_time=None
    ):
        """
        Extract a timeseries as a Pandas dataframe.

        Parameters
        ----------
        interpolate : str, optional
            interpolate between time points, by default "linear"

        Returns
        -------
        pandas.DataFrame
            the timeseries

        Raises
        ------
        Exception
            fails if scenario is not consistent
        """
        scenario = FunmanWorkUnit(
            id=self.id, model=self.model, request=self.request
        ).to_scenario()
        to_plot = scenario.model._state_var_names()
        time_var = scenario.model._time_var()
        if time_var:
            to_plot += ["timer_t"]

        all_df = pd.DataFrame()
        for i, point in enumerate(points):
            timeseries = self.symbol_timeseries(point, to_plot)
            df = pd.DataFrame.from_dict(timeseries)
            df["id"] = i
            # if max_time:
            # if time_var:
            #     df = df.at[max_time, :] = None
            # df = df.reindex(range(max_time+1), fill_value=None)

            if interpolate:
                df = df.interpolate(method=interpolate)
            if time_var and any("timer_t" in x for x in df.columns):
                df = (
                    df.rename(columns={"timer_t": "time"})
                    .set_index("time", drop=True)
                    .drop(columns=["index"])
                )

            df = df.reindex(sorted(df.columns), axis=1)

            all_df = pd.concat([all_df, df])

        return all_df

    def symbol_timeseries(
        self, point: Point, variables: List[str]
    ) -> Dict[str, List[Union[float, None]]]:
        """
        Generate a symbol (str) to timeseries (list) of values

        Parameters
        ----------
        pysmtModel : pysmt.solvers.solver.Model
            variable assignment
        """
        series = self.symbol_values(point, variables)
        a_series = {}  # timeseries as array/list
        max_t = max(
            [
                max([int(k) for k in tps.keys() if k.isdigit()] + [0])
                for _, tps in series.items()
            ]
        )
        a_series["index"] = list(range(0, max_t + 1))
        for var, tps in series.items():
            vals = [None] * (int(max_t) + 1)
            for t, v in tps.items():
                if t.isdigit():
                    vals[int(t)] = v
            a_series[var] = vals
        return a_series

    def symbol_values(
        self, point: Point, variables: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
         Get the value assigned to each symbol in the pysmtModel.

        Parameters
        ----------
        model_encoding : Encoding
            encoding using the symbols
        pysmtModel : pysmt.solvers.solver.Model
            assignment to symbols

        Returns
        -------
        Dict[str, Dict[str, float]]
            mapping from symbol and timepoint to value
        """

        vars = self._symbols(point, variables)
        vals = {}
        for var in vars:
            vals[var] = {}
            for t in vars[var]:
                try:
                    value = point.values[vars[var][t]]
                    vals[var][t] = float(value)
                except OverflowError as e:
                    l.warning(e)
        return vals

    def _symbols(
        self, point: Point, variables: List[str]
    ) -> Dict[str, Dict[str, str]]:
        symbols = {}
        # vars.sort(key=lambda x: x.symbol_name())
        for var in point.values:
            if any(v in var for v in variables):
                var_name, timepoint = self._split_symbol(var)
                if timepoint:
                    if var_name not in symbols:
                        symbols[var_name] = {}
                    symbols[var_name][timepoint] = var
        return symbols

    def _split_symbol(self, symbol: str) -> Tuple[str, str]:
        s, t = symbol.rsplit("_", 1)
        return s, t

    def plot_trajectories(self, variable: str, num: int = 200):
        fig, ax = plt.subplots()
        len_tps = len(self.parameter_space.true_points)
        len_fps = len(self.parameter_space.false_points)
        num_tp_samples = min(len_tps, num)
        num_fp_samples = min(len_fps, num)

        tps = random.sample(self.parameter_space.true_points, num_tp_samples)
        fps = random.sample(self.parameter_space.false_points, num_fp_samples)
        if len(tps) > 0:
            tps_df = self.dataframe(tps)
            # tps_df = tps_df[tps_df[variable] != 0.0]
            tps_df.groupby("id")[variable].plot(c="green", alpha=0.2, ax=ax)
        if len(fps) > 0:
            fps_df = self.dataframe(fps)
            # fps_df = fps_df[fps_df[variable] != 0.0]
            fps_df.groupby("id")[variable].plot(c="red", alpha=0.2, ax=ax)

        return ax

    def plot(
        self,
        points: List[Point],
        variables=None,
        log_y=False,
        max_time=None,
        **kwargs
    ):
        """
        Plot the results in a matplotlib plot.

        Raises
        ------
        Exception
            failure if scenario is not consistent.
        """

        df = self.dataframe(points, max_time=max_time)

        if variables is not None:
            ax = df[variables].plot(marker="o", **kwargs)
        else:
            ax = df.plot(marker="o", **kwargs)

        if log_y:
            ax.set_yscale("symlog")
            plt.ylim(bottom=0)
        # plt.show(block=False)
        return ax
