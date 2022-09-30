from typing import Dict, List, Union
from datetime import datetime
from funman.search_utils import Box, Interval, SearchConfig, SearchStatistics
from pysmt.shortcuts import get_model, And, Not
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from IPython.display import clear_output
from multiprocessing import Queue, Process, Value
from queue import Empty
import logging

from funman.model import Parameter

l = logging.getLogger(__file__)
l.setLevel(logging.ERROR)


class SearchEpisode(object):
    def __init__(self) -> None:
        self.statistics = SearchStatistics()


class BoxSearchEpisode(SearchEpisode):
    def __init__(
        self, config: SearchConfig, problem
    ) -> None:
        super(BoxSearchEpisode, self).__init__()
        self.unknown_boxes = Queue()
        self.true_boxes = []
        self.false_boxes = []

        self.config = config
        self.problem = problem
        self.num_parameters = len(self.problem.parameters)
        self.iteration = Value("i", 0)
        self.internal_process_id = 0

    def initial_box(self) -> Box:
        return Box(self.problem.parameters)

    def on_start(self):
        self.statistics.last_time.value = str(datetime.now())

    def close(self):
        self.unknown_boxes.close()
        self.statistics.close()

    def on_iteration(self):
        if self.internal_process_id == 1 and self.iteration.value % 10 == 0:
            self.plot(
                title=f"Residual = {self.statistics.current_residual}, |boxes|={self.statistics.num_unknown}",
            )
            pass
        self.iteration.value = self.iteration.value + 1

    def add_unknown(self, box: Union[Box, List[Box]]):
        if isinstance(box, list):
            for b in box:
                if b.width() > self.config.tolerance:
                    self.unknown_boxes.put(b, timeout=self.config.queue_timeout)
                    self.statistics.num_unknown.value += 1
        else:
            if box.width() > self.config.tolerance:
                self.unknown_boxes.put(box)
                with self.statistics.num_unknown.get_lock():
                    self.statistics.num_unknown.value += 1

    def add_false(self, box: Box):
        self.false_boxes.append(box)
        with self.statistics.num_false.get_lock():
            self.statistics.num_false.value += 1
        self.statistics.iteration_operation.put("f")

    def add_true(self, box: Box):
        self.true_boxes.append(box)
        with self.statistics.num_true.get_lock():
            self.statistics.num_true.value += 1
        self.statistics.iteration_operation.put("t")

    def get_unknown(self):
        box = self.unknown_boxes.get(timeout=self.config.queue_timeout)
        self.statistics.num_unknown.value = self.statistics.num_unknown.value - 1
        self.statistics.current_residual.value = box.width()
        self.statistics.residuals.put(box.width())
        this_time = datetime.now()
        # FIXME self.statistics.iteration_time.put(this_time - self.statistics.last_time.value)
        # FIXME self.statistics.last_time[:] = str(this_time)
        return box

    def plotBox(self, interval: Interval = Interval(-20, 20)):
        box = Box(self.problem.parameters)
        for p, _ in box.bounds.items():
            box.bounds[p] = interval
        return box

    def plot(
        self,
        plot_bounds: Box = None,
        title="Feasible Regions",
    ):
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

    def plot2DBox(self, i: Box, p1: Parameter, p2: Parameter, color="g"):
        x_limits = i.bounds[p1]
        y_limits = i.bounds[p2]
        x = np.linspace(x_limits.lb, x_limits.ub, 1000)
        plt.fill_between(x, y_limits.lb, y_limits.ub, color=color)

    def plot1D(
        self,
        true_boxes: Queue = None,
        false_boxes: Queue = None,
        plot_bounds: Box = None,
    ):
        p1 = self.problem.parameters[0]

        t_boxes = true_boxes if true_boxes else self.true_boxes
        f_boxes = false_boxes if false_boxes else self.false_boxes
        unknown_boxes = unknown_boxes if unknown_boxes else self.unknown_boxes

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


class BoxSearch(object):
    def __init__(self) -> None:
    
        self.episodes = []

    def split(self, box: Box, episode: BoxSearchEpisode):
        b1, b2 = box.split()
        episode.add_unknown([b1, b2])
        episode.statistics.iteration_operation.put("s")

    def initialize(self, episode: BoxSearchEpisode):
        initial_boxes = Queue()
        initial_boxes.put(episode.initial_box())
        num_boxes = 1
        while num_boxes < episode.config.number_of_processes:
            b1, b2 = initial_boxes.get().split()
            initial_boxes.put(b1)
            initial_boxes.put(b2)
            num_boxes += 1
        for i in range(num_boxes):
            b = initial_boxes.get()
            episode.add_unknown(b)

    def expand(self, rval: Queue, episode: BoxSearchEpisode) -> None:
        """
        A single search process will evaluate and expand the boxes in the episode.unknown_boxes queue.  The processes exit when the queue is empty.  For each box, the algorithm checks whether the box contains a false (infeasible) point.  If it contains a false point, then it checks if the box contains a true point.  If a box contains both a false and true point, then the box is split into two boxes and both are added to the unknown_boxes queue.  If a box contains no false points, then it is a true_box (all points are feasible).  If a box contains no true points, then it is a false_box (all points are infeasible).  
        
        The return value is pushed onto the rval queue to end the process's work within the method.  The return value is a Dict[str, List[Box]] type that maps the "true_boxes" and "false_boxes" to a list of boxes in each set.  Each box in these sets is unique by design.  

        Parameters
        ----------
        rval : Queue
            Return value shared queue
        episode : BoxSearchEpisode
            Shared search data and statistics.
        """
        if episode.internal_process_id == 0:
            self.initialize(episode)

        while True:
            try:
                box = episode.get_unknown()
            except Empty:
                break
            else:
                # Check whether box intersects f (false region)
                phi = And(box.to_smt(), Not(episode.problem.model.formula))
                res = get_model(phi)
                if res:
                    # box intersects f (false region)

                    # Check whether box intersects t (true region)
                    phi1 = And(box.to_smt(), episode.problem.model.formula)
                    res1 = get_model(phi1)
                    if res1:
                        # box intersects both t and f, so it must be split
                        self.split(box, episode)
                    else:
                        # box is a subset of f (intersects f but not t)
                        episode.add_false(box)  # TODO consider merging lists of boxes

                else:
                    # box does not intersect f, so it is in t (true region)
                    episode.add_true(box)
                episode.on_iteration()
        episode.close()
        rval.put({"true_boxes": episode.true_boxes, "false_boxes": episode.false_boxes})


    def search(
        self, problem, config: SearchConfig = SearchConfig()
    ) -> BoxSearchEpisode:
        """
        The BoxSearch.search() creates a BoxSearchEpisode object that stores the
        search progress.  This method is the entry point to the search that
        spawns several processes to parallelize the evaluation of boxes in the
        BoxSearch.expand() method.  It treats the zeroth process as a special
        process that is allowed to initialize the search and plot the progress
        of the search.

        Parameters
        ----------
        problem : ParameterSynthesisScenario
            Model and parameters to synthesize
        config : SearchConfig, optional
            BoxSearch configuration, by default SearchConfig()

        Returns
        -------
        BoxSearchEpisode
            Final search results (parameter space) and statistics.
        """    

        episode = BoxSearchEpisode(config, problem)
        self.episodes.append(episode)
        episode.on_start()

        processes = []

        rval = Queue()

        # creating processes
        for w in range(episode.config.number_of_processes):
            p = Process(
                target=self.expand,
                args=(
                    rval,
                    episode,
                ),
            )
            processes.append(p)
            p.start()
            episode.internal_process_id += 1

        results = [rval.get() for p in processes]
        rval.close()

        for p in processes:
            p.terminate()

        # completing process
        for p in processes:
            p.join()

        episode.false_boxes = [b for r in results for b in r["false_boxes"]]
        episode.true_boxes = [b for r in results for b in r["true_boxes"]]


        return episode
