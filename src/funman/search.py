
from functools import total_ordering
from inspect import Parameter
import queue
from typing import Dict, List

from funman.scenario import ParameterSynthesisScenario, ParameterSynthesisScenarioResult
from funman.constants import NEG_INFINITY, POS_INFINITY, BIG_NUMBER

from pysmt.shortcuts import get_model, And, LT, LE, GE, TRUE, Not, Real
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from IPython.display import clear_output


import logging
l = logging.getLogger(__file__)
l.setLevel(logging.ERROR)


class Interval(object):
    def __init__(self, lb: float, ub: float) -> None:
        self.lb = lb
        self.ub = ub

    def width(self):
        if self.lb == NEG_INFINITY or self.ub == POS_INFINITY:
            return BIG_NUMBER
        else:
            return self.ub - self.lb

    def __lt__(self, other):
        if isinstance(other, Interval):
            return self.width() < other.width()
        else:
            raise Exception(f"Cannot compare __lt__() Interval to {type(other)}")
        
    def __eq__(self, other):
        if isinstance(other, Interval):
            return self.lb == other.lb and self.ub == other.ub
        else:
            raise Exception(f"Cannot compare __eq__() Interval to {type(other)}")

    def __repr__(self):
        return f"[{self.lb}, {self.ub}]"

    def __str__(self):
        return self.__repr__()

    def contains(self, other: "Interval") -> bool:
        lhs = (self.lb == NEG_INFINITY or other.lb != NEG_INFINITY) if (self.lb == NEG_INFINITY or other.lb == NEG_INFINITY) else self.lb <= other.lb
        rhs = (other.ub != POS_INFINITY or self.ub == POS_INFINITY) if (self.ub == POS_INFINITY or other.ub == POS_INFINITY) else other.ub <= self.ub
        return lhs and rhs


    def midpoint(self):
        if self.lb == NEG_INFINITY and self.ub == POS_INFINITY:
            return 0
        elif self.lb == NEG_INFINITY:
            return self.ub - BIG_NUMBER
        if self.ub == POS_INFINITY:
            return self.lb + BIG_NUMBER
        else:
            return ((self.ub - self.lb)/2) + self.lb
    
    def to_smt(self, p: Parameter):
        return And(
                    (GE(p.symbol, Real(self.lb)) if self.lb != NEG_INFINITY else TRUE()), 
                    (LT(p.symbol, Real(self.ub)) if self.ub != POS_INFINITY else TRUE())
                    ).simplify()
                
@total_ordering
class Box(object):

    def __init__(self, parameters) -> None:
        self.bounds = { p: Interval(NEG_INFINITY, POS_INFINITY) for p in parameters}
    
    def to_smt(self):
        return And([interval.to_smt(p) for p, interval in self.bounds.items()])

    def _copy(self):
        c = Box(list(self.bounds.keys()))
        for p, b in self.bounds.items():
            c.bounds[p] = Interval(b.lb, b.ub)
        return c

    def __lt__(self, other):
        if isinstance(other, Box):
            return self.width() > other.width()
        else:
            raise Exception(f"Cannot compare __lt__() Box to {type(other)}")
    
    def __eq__(self, other):
        if isinstance(other, Box):
            return all([self.bounds[p] == other.bounds[p] for p in self.bounds.keys()])
        else:
            raise Exception(f"Cannot compare __eq__() Box to {type(other)}")
    
    def __repr__(self):
        return f"{self.bounds}"

    def __str__(self):
        return self.__repr__()

    def contains(self, other: "Box") -> bool:
        return all([interval.contains(other.bounds[p]) for p, interval in self.bounds.items()])

    def _get_max_width_parameter(self):
        widths = [bounds.width() for _, bounds in self.bounds.items()]
        max_width = max(widths)
        param = list(self.bounds.keys())[widths.index(max_width)]
        return param, max_width

    def width(self) -> float:
        _, width = self._get_max_width_parameter()
        return width

    def split(self):
        p, _ = self._get_max_width_parameter()
        mid = self.bounds[p].midpoint()

        b1 = self._copy()
        b2 = self._copy()

        # b1 is lower half
        b1.bounds[p] = Interval(b1.bounds[p].lb, mid)

        # b2 is upper half
        b2.bounds[p] = Interval(mid, b2.bounds[p].ub)

        return [b1, b2]


class BoxSearch(object):
    def __init__(self, problem: ParameterSynthesisScenario) -> None:
        self.problem = problem
        self.num_parameters = len(self.problem.parameters)
        

    def initial_box(self) -> Box:
         return Box(self.problem.parameters)

    def search(
        self,
        initial_box: Box, 
        tolerance: float = 1e-1
        ) -> Dict[str, List[Box]]:
    
        
        unknown_boxes = queue.PriorityQueue()
        unknown_boxes.put(initial_box)
        true_boxes = []
        false_boxes = []
        
        tolerance = tolerance ## arbitrary precision - gives the smallest box that you would still want to split.
        counter = 0
        while unknown_boxes.not_empty:
            counter += 1
            l.debug(f"number of unknown boxes: {unknown_boxes.qsize}")
            
            box = unknown_boxes.get() 

            if box.width() < tolerance:
                break
            
            # Check whether box intersects f (false region)
            phi = And(box.to_smt(), Not(self.problem.model.formula))
            res = get_model(phi) 
            if res:
                # box intersects f (false region)

                # Check whether box intersects t (true region)
                phi1 = And(box.to_smt(), self.problem.model.formula)
                res1 = get_model(phi1)
                if res1:
                    # box intersects both t and f, so it must be split
                    b1, b2 = box.split()
                    unknown_boxes.put(b1)
                    unknown_boxes.put(b2) 
                else:
                    # box is a subset of f (intersects f but not t)
                    false_boxes.append(box)  # TODO consider merging lists of boxes
            else: 
                # box does not intersect f, so it is in t (true region)
                true_boxes.append(box) # TODO consider merging lists of boxes

            l.debug(f"false boxes: {false_boxes}")
            l.debug(f"true boxes: {true_boxes}")
            l.debug(f"unknown boxes: {unknown_boxes}")

            if counter%50==0:
                self.plot(true_boxes, false_boxes, unknown_boxes)
        
        return true_boxes, false_boxes, unknown_boxes

    def plotBox(self, interval: Interval = Interval(-20, 20)):
        box = Box(self.problem.parameters)
        for p, _ in box.bounds.items():
            box.bounds[p] = interval
        return box


    def plot(self, true_boxes, false_boxes, unknown_boxes, plot_bounds: Box = None):
        if not plot_bounds:
            plot_bounds = self.plotBox()

        clear_output(wait=True)
        
        if self.num_parameters == 1:
            self.plot1D(true_boxes, false_boxes, unknown_boxes, plot_bounds)
        elif self.num_parameters == 2: 
            self.plot2D(true_boxes, false_boxes, unknown_boxes, plot_bounds)
        else:
            raise Exception(f"Plotting for {self.num_parameters} >= 2 is not supported.")

        custom_lines = [Line2D([0], [0], color='g', lw=4),Line2D([0], [0], color='r', lw=4)]
        plt.legend(custom_lines,['true','false'])
        plt.show(block=False)
    
    def plot1DBox(self, i: Box, p1: Parameter, color='g'):
        x_values = [i.bounds[p1].lb, i.bounds[p1].ub]
        plt.plot(x_values, np.zeros(len(x_values)), color, linestyle="-")

    def plot2DBox(self, i: Box, p1: Parameter, p2: Parameter, color='g'):
        x_limits = i.bounds[p1]
        y_limits = i.bounds[p2]
        x = np.linspace(x_limits.lb, x_limits.ub,1000)
        plt.fill_between(x, y_limits.lb, y_limits.ub, color=color)

    def plot1D(self, true_boxes, false_boxes, unknown_boxes, plot_bounds):
        p1 = self.problem.parameters[0] 
        clear_output(wait=True)
        for i in true_boxes:
            if plot_bounds.contains(i):
                self.plot1DBox(i, p1, color='g')
                
        for i in false_boxes:
            if plot_bounds.contains(i):
                self.plot1DBox(i, p1, color='r')
        plt.xlabel(p1.name)
        plt.title('0 <= x <= 5')
        
            
    def plot2D(self, true_boxes, false_boxes, unknown_boxes, plot_bounds):
        p1 = self.problem.parameters[0]
        p2 = self.problem.parameters[1]
        
        for i in false_boxes:
            if plot_bounds.contains(i):
                self.plot2DBox(i, p1, p2, color='r')
        for i in true_boxes:
            if plot_bounds.contains(i):
                self.plot2DBox(i, p1, p2, color='g')
        plt.xlabel(p1.name)
        plt.ylabel(p2.name)
        plt.title('0 <= x <= 5, 10 <= y <= 12')
        
            