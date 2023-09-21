import json
import os
from time import sleep
from typing import Dict, List

import matplotlib.pyplot as plt
from IPython.display import Image, display

from funman.representation import ModelParameter
from funman.search import Box, ParameterSpace, Point
from funman.utils.handlers import ResultHandler, WaitAction

from .box_plotter import BoxPlotter


class NotebookImageRefresher(WaitAction):
    def __init__(self, image_path, *, sleep_for=1) -> None:
        self.image_path = image_path
        self.sleep_for = sleep_for
        self.image = None
        self.handle = None

    def run(self):
        if not os.path.exists(self.image_path):
            return
        if self.image is None:
            self.image = Image(filename=self.image_path)
        if self.handle is None:
            self.handle = display(self.image, clear=True, display_id=True)
        self.image.reload()
        self.handle.update(self.image)
        sleep(self.sleep_for)


class ResultCacheWriter(ResultHandler):
    def __init__(self, write_path) -> None:
        self.write_path = write_path
        self.parameter_space = ParameterSpace()
        self.f = None

    def open(self) -> None:
        self.f = open(self.write_path, "w")

    def process(self, result: dict) -> None:
        self.parameter_space.append_result(result)
        self.f.seek(0)
        json.dump(self.parameter_space.model_dump(), self.f, indent=2)
        self.f.flush()

    def close(self) -> None:
        if self.f is not None:
            self.f.close()


class RealtimeResultPlotter(ResultHandler):
    def __init__(
        self,
        parameters: List[ModelParameter],
        plot_bounds: Box = None,
        title: str = "Feasible Regions",
        color_map: Dict[str, str] = {
            "true": "g",
            "false": "r",
            "dropped": "p",
            "unknown": "b",
        },
        shape_map: Dict[str, str] = {"true": "x", "false": "o"},
        plot_points=False,
        realtime_save_path=None,
        dpi=300,
    ) -> None:
        self.plot_points = plot_points
        self.realtime_save_path = realtime_save_path
        self.dpi = dpi
        self.plotter = BoxPlotter(
            parameters=parameters,
            plot_bounds=plot_bounds,
            title=title,
            color_map=color_map,
            shape_map=shape_map,
        )

    def open(self) -> None:
        self.plotter.initialize_figure()

    def process(self, result: dict) -> None:
        inst = ParameterSpace.decode_labeled_object(result)
        label = inst.label
        if isinstance(inst, Box):
            if label == "unknown":
                self.plotter.plot_add_patch(
                    inst, color=self.plotter.color_map[label]
                )
            else:
                self.plotter.plot_add_box(
                    inst, color=self.plotter.color_map[label]
                )
        elif isinstance(inst, Point):
            if self.plot_points:
                self.plotter.plot_add_point(
                    inst,
                    color=self.plotter.color_map[label],
                    shape=self.plotter.shape_map[label],
                )
        else:
            print(f"Skipping invalid object type: {type(inst)}")

        if self.realtime_save_path is not None:
            plt.savefig(self.realtime_save_path, dpi=self.dpi)

    def close(self) -> None:
        pass
