from imp import reload
import json
from time import sleep
from typing import Dict, List

import matplotlib.pyplot as plt
from funman.model import Parameter

from funman.search_utils import Box, Point, ResultHandler, decode_labeled_object, WaitAction

from .box_plotter import BoxPlotter

from IPython.display import display, Image

import os

class NotebookImageRefresher(WaitAction):
    def __init__(self, image_path, *, sleep_for = 1) -> None:
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
            self.handle = display(self.image, clear = True, display_id = True)
        self.image.reload()
        self.handle.update(self.image)
        sleep(self.sleep_for)
    

class ResultCacheWriter(ResultHandler):
    def __init__(self, write_path) -> None:
        self.write_path = write_path
        self.f = None

    def open(self) -> None:
        self.f = open(self.write_path, 'w')

    def process(self, result: dict) -> None:
        data = json.dumps(result)
        self.f.write(data)
        self.f.write("\n")
        self.f.flush()

    def close(self) -> None:
        if self.f is not None:
            self.f.close()

class RealtimeResultPlotter(ResultHandler):
    def __init__(
            self,
            parameters: List[Parameter],
            plot_bounds: Box = None,
            title: str = "Feasible Regions",
            color_map: Dict[str, str] = {"true": "g", "false": "r", "unknown": "b"},
            shape_map: Dict[str, str] = {"true": "x", "false": "o"},
            plot_points = False,
            realtime_save_path = None
        ) -> None:
        self.plot_points = plot_points
        self.realtime_save_path = realtime_save_path
        self.plotter = BoxPlotter(
            parameters=parameters,
            plot_bounds=plot_bounds,
            title=title,
            color_map=color_map,
            shape_map=shape_map
        )

    def open(self) -> None:
        self.plotter.initialize_figure()

    def process(self, result: dict) -> None:
        ((inst, label), typ) = decode_labeled_object(result)
        if typ is Box:
            self.plotter.plot_add_box(
                inst,
                color=self.plotter.color_map[label]
            )
        elif typ is Point:
            if self.plot_points:
                self.plotter.plot_add_point(
                    inst,
                    color=self.plotter.color_map[label],
                    shape=self.plotter.shape_map[label]
                )
        else:
            print(f"Skipping invalid object type: {typ}")

        if self.realtime_save_path is not None:
            plt.savefig(self.realtime_save_path)

    def close(self) -> None:
        pass
