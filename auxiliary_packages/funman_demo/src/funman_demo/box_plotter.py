import logging
from multiprocessing import Queue
from queue import Empty
from typing import Dict, List

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
from matplotlib.lines import Line2D

from funman.representation import Parameter
from funman.representation.representation import Box, Interval, Point
from funman.search import SearchEpisode

l = logging.getLogger(__file__)
l.setLevel(logging.INFO)


class BoxPlotter(object):
    def __init__(
        self,
        parameters: List[Parameter],
        plot_bounds: Box = None,
        title: str = "Feasible Regions",
        color_map: Dict[str, str] = {
            "true": "g",
            "false": "r",
            "unknown": "b",
        },
        shape_map: Dict[str, str] = {"true": "x", "false": "o"},
        alpha=0.2,
    ) -> None:
        self.parameters = parameters
        # assert (
        #     len(self.parameters) <= 2 and len(self.parameters) > 0,
        #     f"Plotting {len(self.parameters)} parameteres is not supported, must be 1 or 2",
        # )
        self.px = self.parameters[0]
        self.py = self.parameters[1] if len(self.parameters) > 1 else None
        self.plot_bounds = plot_bounds if plot_bounds else self.plotBox()
        self.title = title
        self.color_map = color_map
        self.shape_map = shape_map
        clear_output(wait=True)
        self.custom_lines = [
            Line2D([0], [0], color="g", lw=4, alpha=alpha),
            Line2D([0], [0], color="r", lw=4, alpha=alpha),
        ]
        # plt.ion()

    def initialize_figure(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        (self.data,) = self.ax.plot([], [])
        plt.title(self.title)
        plt.legend(self.custom_lines, ["true", "false"])

        plt.xlabel(self.px.name)
        # plt.xlim(
        #     [
        #         self.plot_bounds.bounds[self.px].lb,
        #         self.plot_bounds.bounds[self.px].ub,
        #     ]
        # )
        if len(self.parameters) > 1:
            plt.ylabel(self.py.name)
            # plt.ylim(
            #     [
            #         self.plot_bounds.bounds[self.py].lb,
            #         self.plot_bounds.bounds[self.py].ub,
            #     ]
            # )

        # plt.show(block=False)
        # plt.pause(0.1)

    def run(self, rval: Queue, episode: SearchEpisode):
        try:
            while True:
                try:
                    # if self.real_time_plotting and self.write_region_to_cache is not None:
                    #     time.sleep(0.1)
                    region = episode.get_box_to_plot()
                except Empty:
                    break
                else:
                    try:
                        # if self.plot_bounds.intersects(box["box"]) and box["box"].finite():
                        if self.write_region_to_cache is not None:
                            self.write_region_to_cache(region)

                        if "box" in region and region["box"].finite():
                            if region["label"] == "unknown":
                                if self.real_time_plotting:
                                    self.plot_add_patch(
                                        region["box"],
                                        color=self.color_map[region["label"]],
                                    )
                            else:
                                if self.real_time_plotting:
                                    self.plot_add_box(
                                        region,
                                        color=self.color_map[region["label"]],
                                    )
                        elif "point" in region:
                            l.debug(f"{region['label']}: {region['point']}")
                            if self.real_time_plotting:
                                self.plot_add_point(
                                    region["point"],
                                    color=self.color_map[region["label"]],
                                    shape=self.shape_map[region["label"]],
                                )
                    except Exception as e:
                        print(e)
        finally:
            episode.close()
            if self.out_cache is not None:
                self.out_cache.close()
            rval.put({"true_boxes": [], "false_boxes": []})

    def plot_list_of_boxes(self):
        while True:
            try:
                box = self.episode.get_box_to_plot()
            except Empty:
                break
            else:
                try:
                    if (
                        self.plot_bounds.intersects(box["box"])
                        and box["box"].finite()
                    ):
                        self.plot_add_box(
                            box, color=self.color_map[box["label"]]
                        )
                except Exception as e:
                    pass

    def plot_add_box(self, box: Box, color="r"):
        if self.py:
            self.plot2DBox(box, self.px, self.py, color=color)
        else:
            self.plot1DBox(box, self.px, color=color)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        # plt.show(block=False)

    def plot_add_patch(self, box: Box, color="r"):
        lb_y = box.bounds[self.py.name].lb if self.py else -0.05
        width_y = box.bounds[self.py.name].width() if self.py else 1e-1
        rect = patches.Rectangle(
            (box.bounds[self.px.name].lb, lb_y),
            box.bounds[self.px.name].width(),
            width_y,
            linewidth=1,
            edgecolor=color,
            facecolor="none",
        )

        # Add the patch to the Axes
        self.ax.add_patch(rect)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        # plt.show(block=False)

    def plot_add_point(self, point: Point, color="r", shape="x", alpha=0.2):
        yval = point.values[self.py.name] if self.py else 0.0
        plt.scatter(
            point.values[self.px.name],
            yval,
            color=color,
            marker=shape,
            alpha=alpha,
        )
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        # plt.show(block=False)

    def plotBox(self, interval: Interval = Interval(lb=-2000, ub=2000)):
        box = Box(bounds={p.name: interval for p in self.parameters})
        return box

    def plot(
        self,
        plot_bounds: Box = None,
        title="Feasible Regions",
    ):
        raise DeprecationWarning(
            "This has been superseded by the run() function, do not use!"
        )
        if not plot_bounds:
            plot_bounds = self.plotBox()

        clear_output(wait=True)

        if self.num_parameters == 1:
            self.plot1D(plot_bounds=plot_bounds)
        elif self.num_parameters == 2:
            self.plot2D(plot_bounds=plot_bounds)
        else:
            raise Exception(
                f"Plotting for {self.num_parameters} >= 2 is not supported."
            )

        custom_lines = [
            Line2D([0], [0], color="g", lw=4),
            Line2D([0], [0], color="r", lw=4),
        ]
        plt.title(title)
        plt.legend(custom_lines, ["true", "false"])
        plt.show(block=False)

    def plot1DBox(self, i: Box, p1: Parameter, color="g"):
        x_values = [i.bounds[p1].lb, i.bounds[p1].ub]
        plt.plot(x_values, np.zeros(len(x_values)), color, linestyle="-")

    def plot2DBox(self, i, p1: Parameter, p2: Parameter, color="g", alpha=0.2):
        x_limits = i.bounds[p1.name]
        y_limits = i.bounds[p2.name]
        if abs(float(x_limits.lb)) < 100 and abs(float(x_limits.ub)) < 100:
            x = np.linspace(float(x_limits.lb), float(x_limits.ub), 1000)
            plt.fill_between(
                x, y_limits.lb, y_limits.ub, color=color, alpha=alpha
            )
        plt.show(block=False)

    def plot2DBox_temp(
        a, color="g", alpha=0.2
    ):  ## temp DMI 11/2/22 - clean up later
        plt.ion()
        a_params = list(a.bounds.keys())
        x_limits = a.bounds[a_params[0]]  # first box, first parameter
        y_limits = a.bounds[a_params[1]]  # first box, second parameter
        if abs(float(x_limits.lb)) < 100 and abs(float(x_limits.ub)) < 100:
            x = np.linspace(float(x_limits.lb), float(x_limits.ub), 1000)
            plt.fill_between(
                x, y_limits.lb, y_limits.ub, color=color, alpha=alpha
            )
            plt.pause(1)
        plt.show()

    def plot2DBoxByHeight_temp(
        x_limits, height, color="g", alpha=0.2
    ):  ## temp DMI 11/2/22 - clean up later
        plt.ion()
        if abs(float(x_limits[0])) < 100 and abs(float(x_limits[1])) < 100:
            x = np.linspace(float(x_limits[0]), float(x_limits[1]), 1000)
            plt.fill_between(x, 0, height, color=color, alpha=alpha)
            plt.pause(1)
        plt.show()

    def plot2DBoxes_temp(
        list_of_boxes, color="g", alpha=0.2
    ):  ## temp DMI 11/2/22 - clean up later
        for a in list_of_boxes:
            a_params = list(a.bounds.keys())
            x_limits = a.bounds[a_params[0]]  # first box, first parameter
            y_limits = a.bounds[a_params[1]]  # first box, second parameter
            if abs(float(x_limits.lb)) < 100 and abs(float(x_limits.ub)) < 100:
                x = np.linspace(float(x_limits.lb), float(x_limits.ub), 1000)
                plt.fill_between(
                    x, y_limits.lb, y_limits.ub, color=color, alpha=alpha
                )
        plt.show()
        plt.pause(5)
        plt.close()

    def plot2DBoxesByHeight_temp(
        list_of_x_limits, heights, color="g", alpha=0.2
    ):  ## temp DMI 11/2/22 - clean up later
        for i in range(len(list_of_x_limits)):
            x_limits = list_of_x_limits[i]
            height = heights[i]
            if abs(float(x_limits[0])) < 100 and abs(float(x_limits[1])) < 100:
                x = np.linspace(float(x_limits[0]), float(x_limits[1]), 1000)
                plt.fill_between(x, 0, height, color=color, alpha=alpha)
        plt.show()
        plt.pause(5)
        plt.close()

    def plot2DBoxList(b, color="g", alpha=0.2):
        box = list(b.bounds.values())
        x_limits = box[0]
        y_limits = box[1]
        # if abs(float(x_limits.lb)) < 100 and abs(float(x_limits.ub)) < 100:
        x = np.linspace(float(x_limits.lb), float(x_limits.ub), 1000)
        plt.fill_between(x, y_limits.lb, y_limits.ub, color=color, alpha=alpha)

    def plot1D(
        self,
        true_boxes: Queue = None,
        false_boxes: Queue = None,
        plot_bounds: Box = None,
    ):
        p1 = self.problem.parameters[0]

        t_boxes = true_boxes if true_boxes else self.true_boxes
        f_boxes = false_boxes if false_boxes else self.false_boxes

        for b in t_boxes:
            if b and plot_bounds.intersects(b) and b.finite():
                self.plot1DBox(b, p1, color="g")
        for b in f_boxes:
            if b and plot_bounds.intersects(b) and b.finite():
                self.plot1DBox(b, p1, color="r")

        plt.xlabel(p1.name)
        plt.xlim([plot_bounds.bounds[p1].lb, plot_bounds.bounds[p1].ub])

    def plot2D(
        self,
        true_boxes: Queue = None,
        false_boxes: Queue = None,
        plot_bounds: Box = None,
    ):
        p1 = self.problem.parameters[0]
        p2 = self.problem.parameters[1]

        t_boxes = true_boxes if true_boxes else self.true_boxes
        f_boxes = false_boxes if false_boxes else self.false_boxes

        for b in t_boxes:
            if b and plot_bounds.intersects(b) and b.finite():
                self.plot2DBox(b, p1, p2, color="g")
        for b in f_boxes:
            if b and plot_bounds.intersects(b) and b.finite():
                self.plot2DBox(b, p1, p2, color="r")

        plt.xlabel(p1.name)
        plt.ylabel(p2.name)
        plt.xlim([plot_bounds.bounds[p1].lb, plot_bounds.bounds[p1].ub])
        plt.ylim([plot_bounds.bounds[p2].lb, plot_bounds.bounds[p2].ub])
