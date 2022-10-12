import ctypes
from curses.ascii import EM
from functools import total_ordering
from multiprocessing import Array, Queue, Value, cpu_count
import os
import time
from typing import Dict, List, Union
from funman.config import Config
from funman.model import Parameter
from pysmt.shortcuts import Real, GE, LT, And, TRUE, Equals
from funman.constants import NEG_INFINITY, POS_INFINITY, BIG_NUMBER


class Interval(object):
    def __init__(self, lb: float, ub: float) -> None:
        self.lb = lb
        self.ub = ub
        self.cached_width = None

    def width(self):
        if self.cached_width is None:
            if self.lb == NEG_INFINITY or self.ub == POS_INFINITY:
                self.cached_width = BIG_NUMBER
            else:
                self.cached_width = self.ub - self.lb
        return self.cached_width

    def __lt__(self, other):
        if isinstance(other, Interval):
            return self.width() < other.width()
        else:
            raise Exception(f"Cannot compare __lt__() Interval to {type(other)}")

    def __eq__(self, other):
        if isinstance(other, Interval):
            return self.lb == other.lb and self.ub == other.ub
        else:
            return False

    def __repr__(self):
        return f"[{self.lb}, {self.ub}]"

    def __str__(self):
        return self.__repr__()

    def finite(self) -> bool:
        return self.lb != NEG_INFINITY and self.ub != POS_INFINITY

    def contains(self, other: "Interval") -> bool:
        lhs = (
            (self.lb == NEG_INFINITY or other.lb != NEG_INFINITY)
            if (self.lb == NEG_INFINITY or other.lb == NEG_INFINITY)
            else self.lb <= other.lb
        )
        rhs = (
            (other.ub != POS_INFINITY or self.ub == POS_INFINITY)
            if (self.ub == POS_INFINITY or other.ub == POS_INFINITY)
            else other.ub <= self.ub
        )
        return lhs and rhs

    def intersects(self, other: "Interval") -> bool:
        lhs = (
            (self.lb == NEG_INFINITY or other.lb != NEG_INFINITY)
            if (self.lb == NEG_INFINITY or other.lb == NEG_INFINITY)
            else self.lb <= other.lb
        )
        rhs = (
            (other.ub != POS_INFINITY or self.ub == POS_INFINITY)
            if (self.ub == POS_INFINITY or other.ub == POS_INFINITY)
            else other.ub <= self.ub
        )
        return lhs or rhs
    
    def intersection(self, other: "Interval") -> "Interval":
        """Given 2 intervals with a = [a0,a1] and b=[b0,b1], check whether they intersect.  If they do, return interval with their intersection."""
        if a[0] <= b[0]:
            minArray = a
            maxArray = b
        else:
            minArray = b
            maxArray = a
        if minArray[1] > maxArray[0]: ## has nonempty intersection. return intersection
            return [maxArray[0], minArray[1]]
        else: ## no intersection.
            return []
        

    def intersection(self, other: "Interval") -> bool:
        # FIXME Drisana
        pass

    def midpoint(self):
        if self.lb == NEG_INFINITY and self.ub == POS_INFINITY:
            return 0
        elif self.lb == NEG_INFINITY:
            return self.ub - BIG_NUMBER
        if self.ub == POS_INFINITY:
            return self.lb + BIG_NUMBER
        else:
            return ((self.ub - self.lb) / 2) + self.lb

    def contains_value(self, value: float) -> bool:
        lhs = (
            self.lb == NEG_INFINITY or self.lb <= value
        )
        rhs = (
            self.ub == POS_INFINITY or value <= self.ub
        )
        return lhs and rhs
        

    def to_smt(self, p: Parameter):
        return And(
            (GE(p.symbol, Real(self.lb)) if self.lb != NEG_INFINITY else TRUE()),
            (LT(p.symbol, Real(self.ub)) if self.ub != POS_INFINITY else TRUE()),
        ).simplify()


class Point(object):
    def __init__(self, parameters) -> None:
        self.values = {p: 0.0 for p in parameters}

    def __str__(self):
        return f"{self.values.values()}"

    def __hash__(self):
        return int(sum([v for _, v in self.values.items()]))

    def __eq__(self, other):
        if isinstance(other, Point):
            return all([self.values[p] == other.values[p] for p in self.values.keys()])
        else:
            return False

    def to_smt(self):
        return And([Equals(p.symbol, Real(value)) for p, value in self.values.items()])

@total_ordering
class Box(object):
    def __init__(self, parameters) -> None:
        self.bounds = {p: Interval(p.lb, p.ub) for p in parameters}
        self.cached_width = None

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
            return False

    def __repr__(self):
        return f"{self.bounds}"

    def __str__(self):
        return f"{self.bounds.values()}"

    def finite(self) -> bool:
        return all([i.finite() for _, i in self.bounds.items()])

    def contains(self, other: "Box") -> bool:
        return all(
            [interval.contains(other.bounds[p]) for p, interval in self.bounds.items()]
        )

    def contains_point(self, point: Point) -> bool:
        return all([interval.contains_value(point.values[p]) for p, interval in self.bounds.items()])

    def intersects(self, other: "Box") -> bool:
        return all(
            [
                interval.intersects(other.bounds[p])
                for p, interval in self.bounds.items()
            ]
        )

    def intersection(self, other: "Box") -> "Box":
        # FIXME Drisana
        pass

    def _get_max_width_parameter(self):
        widths = [bounds.width() for _, bounds in self.bounds.items()]
        max_width = max(widths)
        param = list(self.bounds.keys())[widths.index(max_width)]
        return param, max_width

    def width(self) -> float:
        if self.cached_width is None:
            _, width = self._get_max_width_parameter()
            self.cached_width = width

        return self.cached_width

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

    def intersect_two_boxes(self, other) -> "Box":
        a = self.bounds
        b = other.bounds
        result = []
        d = len(a.values()) ## dimension
        for i in range(d):
            subresult = intersection(a[i],b[i])
            if subresult == []:
                return None
            else:
                result.append(subresult)
        return result

class SearchStatistics(object):
    def __init__(self):
        self.num_true = Value("i", 0)
        self.num_false = Value("i", 0)
        self.num_unknown = Value("i", 0)
        self.residuals = Queue()
        self.current_residual = Value("d", 0.0)
        self.last_time = Array(ctypes.c_wchar, "")
        self.iteration_time = Queue()
        self.iteration_operation = Queue()

    def close(self):
        self.residuals.close()
        self.iteration_time.close()
        self.iteration_operation.close()


class SearchConfig(Config):
    def __init__(self, *args, **kwargs) -> None:
        self.tolerance = kwargs["tolerance"] if "tolerance" in kwargs else 1e-3
        self.queue_timeout = (
            kwargs["queue_timeout"] if "queue_timeout" in kwargs else 120
        )
        self.number_of_processes = (
            kwargs["number_of_processes"]
            if "number_of_processes" in kwargs
            else cpu_count()
        )
