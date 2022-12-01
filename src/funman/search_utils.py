"""
This submodule contains definitions for the behaviors used
during the configuration and execution of a search.
"""
from abc import ABC, abstractmethod
from functools import total_ordering
from multiprocessing.managers import SyncManager
from queue import Queue as SQueue
import traceback
from typing import Dict, List, Union
from funman.config import Config
from funman.model import Parameter
from funman.constants import NEG_INFINITY, POS_INFINITY, BIG_NUMBER

from numpy import average
from pysmt.shortcuts import Real, GE, LT, LE, And, TRUE, Equals
import funman.math_utils as math_utils
import multiprocess as mp
import logging

l = logging.getLogger(__name__)


class Interval(object):
    def __init__(self, lb: float, ub: float) -> None:
        self.lb = lb
        self.ub = ub
        self.cached_width = None

    def make_interval(p_bounds) -> "Interval":
        b0_bounds = p_bounds[0]
        b1_bounds = p_bounds[1]
        b = Interval(lb=b0_bounds, ub=b1_bounds)
        return b

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
            raise Exception(
                f"Cannot compare __lt__() Interval to {type(other)}"
            )

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
            else self.lb < other.lb ## DMI change < to <=
        )
        rhs = (
            (other.ub != POS_INFINITY or self.ub == POS_INFINITY)
            if (self.ub == POS_INFINITY or other.ub == POS_INFINITY)
            else other.ub < self.ub ## DMI change < to <=
        )
        return lhs or rhs

    def intersection(self, other: "Interval") -> bool:
        # FIXME Drisana
        pass

    def midpoint(self, points=None):
        if points:
            # Get value that is the average of points (i.e., midpoint between points)
            return float(average(points))

        if self.lb == NEG_INFINITY and self.ub == POS_INFINITY:
            return 0
        elif self.lb == NEG_INFINITY:
            return self.ub - BIG_NUMBER
        if self.ub == POS_INFINITY:
            return self.lb + BIG_NUMBER
        else:
            return ((self.ub - self.lb) / 2) + self.lb

    def union(self, other: "Interval") -> List["Interval"]:
        if self == other:  ## intervals are equal, so return original interval
            ans = [self]
            total_height = [Interval.width(self)]
            return ans, total_height
        else:  ## intervals are not the same. start by identifying the lower and higher intervals.
            if self.lb == other.lb:
                if math_utils.lt(self.ub, other.lb):
                    minInterval = self
                    maxInterval = other
                else:  ## other.ub > self.ub
                    minInterval = other
                    maxInterval = self
            elif math_utils.lt(self.lb, other.lb):
                minInterval = self
                maxInterval = other
            else:
                minInterval = other
                maxInterval = self
        if math_utils.gte(
            minInterval.ub, maxInterval.lb
        ):  ## intervals intersect.
            ans = Interval.make_interval([minInterval.lb, maxInterval.ub])
            total_height = Interval.width(ans)
            return ans, total_height
        elif math_utils.lt(
            minInterval.ub, maxInterval.lb
        ):  ## intervals are disjoint.
            ans = [minInterval, maxInterval]
            total_height = [
                math_utils.plus(
                    Interval.width(minInterval), Interval.width(maxInterval)
                )
            ]
            return ans, total_height

    def subtract_two_1d_intervals(
        a: "Interval", b: "Interval"
    ) -> "Interval":  ## TODO Drisana - fix
        """Given 2 intervals a = [a0,a1] and b=[b0,b1], return the part of a that does not intersect with b."""
        if math_utils.gte(a.lb, b.lb):
            if math_utils.lte(a.ub, b.ub):  ## a is a subset of b: return None
                return None
            return Interval(b.ub, a.ub)
        elif math_utils.lt(a.lb, b.lb):
            return Interval(a.lb, b.lb)

        return None

    def subtract_two_1d_lists(a, b):
        """Given 2 intervals a = [a0,a1] and b=[b0,b1], return the part of a that does not intersect with b."""

        if math_utils.lt(a[0], b[0]):
            return [a[0], b[0]]

        if math_utils.gt(a[0], b[0]):
            return [b[0], a[0]]

        return None

    def union(self, other: "Interval") -> List["Interval"]:
        if self == other:  ## intervals are equal, so return original interval
            ans = [self]
            total_height = [Interval.width(self)]
            return ans, total_height
        else:  ## intervals are not the same. start by identifying the lower and higher intervals.
            if self.lb == other.lb:
                if math_utils.lt(self.ub, other.lb):
                    minInterval = self
                    maxInterval = other
                else:  ## other.ub > self.ub
                    minInterval = other
                    maxInterval = self
            elif math_utils.lt(self.lb, other.lb):
                minInterval = self
                maxInterval = other
            else:
                minInterval = other
                maxInterval = self
        if math_utils.gte(
            minInterval.ub, maxInterval.lb
        ):  ## intervals intersect.
            ans = Interval.make_interval([minInterval.lb, maxInterval.ub])
            total_height = Interval.width(ans)
            return ans, total_height
        elif math_utils.lt(
            minInterval.ub, maxInterval.lb
        ):  ## intervals are disjoint.
            ans = [minInterval, maxInterval]
            total_height = [
                math_utils.plus(
                    Interval.width(minInterval), Interval.width(maxInterval)
                )
            ]
            return ans, total_height

    def subtract_two_1d_intervals(
        a: "Interval", b: "Interval"
    ) -> "Interval":  ## TODO Drisana - fix
        """Given 2 intervals a = [a0,a1] and b=[b0,b1], return the part of a that does not intersect with b."""
        if math_utils.gte(a.lb, b.lb):
            if math_utils.lte(a.ub, b.ub):  ## a is a subset of b: return None
                return None
            return Interval(b.ub, a.ub)
        elif math_utils.lt(a.lb, b.lb):
            return Interval(a.lb, b.lb)

        return None

    def subtract_two_1d_lists(a, b):
        """Given 2 intervals a = [a0,a1] and b=[b0,b1], return the part of a that does not intersect with b."""

        if math_utils.lt(a[0], b[0]):
            return [a[0], b[0]]

        if math_utils.gt(a[0], b[0]):
            return [b[0], a[0]]

        return None

    def contains_value(self, value: float) -> bool:
        lhs = self.lb == NEG_INFINITY or self.lb <= value
        rhs = self.ub == POS_INFINITY or value <= self.ub
        return lhs and rhs

    def to_smt(self, p: Parameter, closed_upper_bound=False):
        lower = (
            GE(p.symbol(), Real(self.lb)) if self.lb != NEG_INFINITY else TRUE()
        )
        upper_ineq = LE if closed_upper_bound else LT
        upper = (
            upper_ineq(p.symbol(), Real(self.ub))
            if self.ub != POS_INFINITY
            else TRUE()
        )
        return And(
            lower,
            upper,
        ).simplify()

    def to_dict(self):
        return {
            "lb": self.lb,
            "ub": self.ub,
            # "cached_width": self.cached_width
        }

    @staticmethod
    def from_dict(data):
        res = Interval(data["lb"], data["ub"])
        # res.cached_width = data["cached_width"]
        return res


class Point(object):
    def __init__(self, parameters) -> None:
        self.values = {p: 0.0 for p in parameters}

    def __str__(self):
        return f"{self.values.values()}"

    def to_dict(self):
        return {"values": {k.name: v for k, v in self.values.items()}}

    @staticmethod
    def from_dict(data):
        res = Point([])
        res.values = {Parameter(k): v for k, v in data["values"].items()}
        return res

    def __hash__(self):
        return int(sum([v for _, v in self.values.items()]))

    def __eq__(self, other):
        if isinstance(other, Point):
            return all(
                [self.values[p] == other.values[p] for p in self.values.keys()]
            )
        else:
            return False

    def to_smt(self):
        return And(
            [Equals(p.symbol, Real(value)) for p, value in self.values.items()]
        )


@total_ordering
class Box(object):
    def __init__(self, parameters) -> None:
        self.bounds: Dict[Parameter, Interval] = {
            p: Interval(p.lb, p.ub) for p in parameters
        }
        self.cached_width = None

    def to_smt(self, closed_upper_bound=False):
        return And(
            [
                interval.to_smt(p, closed_upper_bound=closed_upper_bound)
                for p, interval in self.bounds.items()
            ]
        )

    def to_dict(self):
        return {
            "bounds": {k.name: v.to_dict() for k, v in self.bounds.items()},
            # "cached_width": self.cached_width
        }

    @staticmethod
    def from_dict(data):
        res = Box([])
        res.bounds = {
            Parameter(k): Interval.from_dict(v)
            for k, v in data["bounds"].items()
        }
        # res.cached_width = data["cached_width"]
        return res

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
            return all(
                [self.bounds[p] == other.bounds[p] for p in self.bounds.keys()]
            )
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
            [
                interval.contains(other.bounds[p])
                for p, interval in self.bounds.items()
            ]
        )

    def contains_point(self, point: Point) -> bool:
        return all(
            [
                interval.contains_value(point.values[p])
                for p, interval in self.bounds.items()
            ]
        )

    def equal(b1, b2, param_list): ## added 11/27/22 DMI
        result = []
        for p1 in param_list:
            for b in b1.bounds:
                if b.name == p1:
                    b1_bounds = [b.lb, b.ub]
            for b in b2.bounds:
                if b.name == p1:
                    b2_bounds = [b.lb, b.ub]
            if b1_bounds == b2_bounds:
                result.append(True)
            else: 
                result.append(False)
        return all(result)

    def intersects(self, other: "Box") -> bool:
#        print([
#                interval.intersects(other.bounds[p])
#                for p, interval in self.bounds.items()
#        ]) ## DMI delete later
        return all( 
            [
                interval.intersects(other.bounds[p])
                for p, interval in self.bounds.items()
            ]
        )

    def make_box_2d(p_bounds) -> "Box":
        b0_bounds = Interval.make_interval(p_bounds[0])
        print(b0_bounds.lb)
        b1_bounds = Interval.make_interval(p_bounds[1])
        b = Box([Parameter("foo", lb=b0_bounds.lb, ub=b0_bounds.ub)])
        return b

    def union(a: "Box", b: "Box") -> List["Box"]:
        ### specify first parameter (the one to check whether there is an intersection on)
        a_params = list(a.bounds.keys())
        b_params = list(b.bounds.keys())

        beta_0_a = a.bounds[a_params[0]]
        beta_1_a = a.bounds[a_params[1]]
        beta_0_b = b.bounds[b_params[0]]
        beta_1_b = b.bounds[b_params[1]]

        beta_0 = Box.intersection(beta_0_a, beta_0_b)
        if beta_0 != []:  ## boxes intersect on the given axis
            bounds = beta_0
            print("todo: also calculate heights for the non-intersected parts.")
        else:  ## boxes do not intersect on the given axis: just return original boxes and their heights
            bounds1 = beta_0_a
            bounds2 = beta_0_b
            height1 = math_utils.minus(beta_1_a.ub, beta_1_a.lb)
            height2 = math_utils.minus(beta_1_b.ub, beta_1_b.lb)
            print(height1, height2)
        beta_1 = Box.intersection(beta_1_a, beta_1_b)
        if beta_1 == []:  ## no intersection
            print(beta_1_a.lb)
            print(beta_1_a.ub)
            print(beta_1_b.lb)
            print(beta_1_b.ub)
            height1 = math_utils.minus(beta_1_a.ub, beta_1_a.lb)
            height2 = math_utils.minus(beta_1_b.ub, beta_1_b.lb)
            total_height = math_utils.plus(height1, height2)
        else:  ## boxes intersect along y-axis
            print("todo")

    def split_bounds(a: "Box", b: "Box"):
        interval_list = []
        height_list = []
        ybounds_list = []
        a_params = list(a.bounds.keys())
        b_params = list(b.bounds.keys())
        beta_0_a = a.bounds[a_params[0]]  # first box, first parameter
        beta_1_a = a.bounds[a_params[1]]  # first box, second parameter
        beta_0_b = b.bounds[b_params[0]]  # second box, first parameter
        beta_1_b = b.bounds[b_params[1]]  # second box, second parameter
        beta_0_intersection = Box.intersection(
            beta_0_a, beta_0_b
        )  # intersection of the first parameter for first box and second box
        beta_0_a_new = Interval.subtract_two_1d_intervals(
            beta_0_a, Interval.make_interval(beta_0_intersection)
        )
        beta_0_b_new = Interval.subtract_two_1d_intervals(
            beta_0_b, Interval.make_interval(beta_0_intersection)
        )
        beta_0_a_new_2 = Interval.subtract_two_1d_intervals(
            Interval.make_interval(beta_0_intersection), beta_0_a
        )
        beta_0_b_new_2 = Interval.subtract_two_1d_intervals(
            Interval.make_interval(beta_0_intersection), beta_0_b
        )
        if beta_0_intersection != None:
            interval = beta_0_intersection
            ybounds = Interval.union(beta_1_a, beta_1_b)[0]
            height = Interval.union(beta_1_a, beta_1_b)[1]
            if interval not in interval_list:
                interval_list.append(interval)
                height_list.append(height)
                ybounds_list.append(ybounds)
        if beta_0_a_new != None:
            interval = beta_0_a_new
            ybounds = beta_1_a
            height = Interval.width(beta_1_a)
            if interval not in interval_list:
                interval_list.append([interval.lb, interval.ub])
                height_list.append(height)
                ybounds_list.append(ybounds)
        if beta_0_a_new_2 != None:
            interval = beta_0_a_new_2
            ybounds = beta_1_a
            height = Interval.width(beta_1_a)
            if interval not in interval_list:
                interval_list.append([interval.lb, interval.ub])
                height_list.append(height)
                ybounds_list.append(ybounds)
        if beta_0_b_new != None:
            interval = beta_0_b_new
            ybounds = beta_1_b
            height = Interval.width(beta_1_b)
            if interval not in interval_list:
                interval_list.append([interval.lb, interval.ub])
                height_list.append(height)
                ybounds_list.append(ybounds)
        if beta_0_b_new_2 != None:
            interval = beta_0_b_new_2
            ybounds = beta_1_b
            height = Interval.width(beta_1_b)
            if interval not in interval_list:
                interval_list.append([interval.lb, interval.ub])
                height_list.append(height)
                ybounds_list.append(ybounds)
        return interval_list, height_list, ybounds_list

    def check_bounds_disjoint_equal_bool(a: "Box", b: "Box"):
        ### specify first parameter (the one to check whether there is an intersection on)
        a_params = list(a.bounds.keys())
        b_params = list(b.bounds.keys())
        beta_0_a = a.bounds[a_params[0]]  # first box, first parameter
        beta_1_a = a.bounds[a_params[1]]  # first box, second parameter
        beta_0_b = b.bounds[b_params[0]]  # second box, first parameter
        beta_1_b = b.bounds[b_params[1]]  # second box, second parameter
        if beta_0_a == beta_0_b:  ## bounds are equal: done
            print("done: beta_0 bounds are equal")
            return True
        else:
            beta_0_intersection = Box.intersection(
                beta_0_a, beta_0_b
            )  # intersection of the first parameter for first box and second box
            if beta_0_intersection == []:  ## disjoint: done
                print("done: beta_0 bounds are disjoint")
                return True
            else:  ## there is both some intersection and some symmetric difference. split accordingly.
                return False

    def check_bounds_disjoint_equal(a: "Box", b: "Box"):
        ### specify first parameter (the one to check whether there is an intersection on)
        a_params = list(a.bounds.keys())
        b_params = list(b.bounds.keys())
        beta_0_a = a.bounds[a_params[0]]  # first box, first parameter
        beta_1_a = a.bounds[a_params[1]]  # first box, second parameter
        beta_0_b = b.bounds[b_params[0]]  # second box, first parameter
        beta_1_b = b.bounds[b_params[1]]  # second box, second parameter
        #        print('beta_0_a:',beta_0_a)
        #        print('beta_0_b:',beta_0_b)
        #        print('beta_1_a:',beta_1_a)
        #        print('beta_1_b:',beta_1_b)
        if beta_0_a == beta_0_b:  ## bounds are equal: done
            print("done: beta_0 bounds are equal")
            ## calculate width of union
            beta_1_union = Interval.union(beta_1_a, beta_1_b)
            beta_1_union_interval = beta_1_union[0]
            beta_1_union_height = beta_1_union[1]
            return True, [beta_0_a], beta_1_union_interval, beta_1_union_height
        else:
            beta_0_intersection = Box.intersection(
                beta_0_a, beta_0_b
            )  # intersection of the first parameter for first box and second box
            if beta_0_intersection == []:  ## disjoint: done
                print("done: beta_0 bounds are disjoint")
                return (
                    True,
                    [beta_0_a, beta_0_b],
                    [beta_1_a, beta_1_b],
                    [Interval.width(beta_1_a), Interval.width(beta_1_b)],
                )
            else:  ## there is both some intersection and some symmetric difference. split accordingly.
                #                print("in progress: splitting")
                return False, Box.split_bounds(
                    a, b
                )  # , Interval.union(beta_1_a, beta_1_b)

    def _get_max_width_point_parameter(self, points):
        # get the average distance from the center point for each parameter
        centers = {
            p: average([pt.values[p] for pt in points]) for p in self.bounds
        }
        point_distances = [
            {p: abs(pt.values[p] - centers[p]) for p in pt.values}
            for pt in points
        ]
        parameter_widths = {
            p: average([pt[p] for pt in point_distances]) for p in self.bounds
        }
        max_width_parameter = max(parameter_widths, key=parameter_widths.get)
        return max_width_parameter

    def _get_max_width_parameter(self):
        widths = [bounds.width() for _, bounds in self.bounds.items()]
        max_width = max(widths)
        param = list(self.bounds.keys())[widths.index(max_width)]
#        print('width info:', param, max_width)## DMI delete
        return param, max_width

    def width(self) -> float:
        if self.cached_width is None:
            _, width = self._get_max_width_parameter()
            self.cached_width = width

        return self.cached_width

    def split(self, points=None):
        """
        Split box along max width dimension. If points are provided, then pick the axis where the points are maximally distant.

        Parameters
        ----------
        points : List[Point], optional
            solution points that the split will separate, by default None

        Returns
        -------
        List[Box]
            Boxes resulting from the split.
        """

        if points:
            p = self._get_max_width_point_parameter(points)
            mid = self.bounds[p].midpoint(
                points=[pt.values[p] for pt in points]
            )
        else:
            p, _ = self._get_max_width_parameter()
            mid = self.bounds[p].midpoint()

        b1 = self._copy()
        b2 = self._copy()

        # b1 is lower half
        assert b1.bounds[p].lb < mid
        b1.bounds[p] = Interval(b1.bounds[p].lb, mid)

        # b2 is upper half
        assert mid <= b2.bounds[p].ub
        b2.bounds[p] = Interval(mid, b2.bounds[p].ub)

        return [b2, b1]

    def intersection(a: Interval, b: Interval) -> Interval:
        """Given 2 intervals with a = [a0,a1] and b=[b0,b1], check whether they intersect.  If they do, return interval with their intersection."""
        if a == b:
            return a
        else:
            if a.lb == b.lb:
                if math_utils.lt(a.ub, b.ub):
                    minArray = a
                    maxArray = b
                else:
                    minArray = b
                    maxArray = a
            elif math_utils.lt(a.lb, b.lb):
                minArray = a
                maxArray = b
            else:
                minArray = b
                maxArray = a
            if math_utils.gt(
                minArray.ub, maxArray.lb
            ):  ## has nonempty intersection. return intersection
                return [float(maxArray.lb), float(minArray.ub)]
            else:  ## no intersection.
                return []

    def intersect_two_boxes_selected_parameters(b1, b2, param_list): ## added 11/21/22 DMM
        result = []
        for p1 in param_list:
            for b in b1.bounds:
                if b.name == p1:
                    b1_bounds = Interval.make_interval([b.lb, b.ub])
            for b in b2.bounds:
                if b.name == p1:
                    b2_bounds = Interval.make_interval([b.lb, b.ub])
            intersection_ans = Box.intersection(b1_bounds, b2_bounds)
            dict_element = Parameter(p1, intersection_ans[0], intersection_ans[1])
            result.append(dict_element)
        return Box({i for i in result})

    def intersect_two_boxes(b1, b2):
        a = list(b1.bounds.values())
        b = list(b2.bounds.values())
        result = []
        d = len(a)  ## dimension
        for i in range(d):
            subresult = Box.intersection(a[i], b[i])
            if subresult == []:
                return None
            else:
                result.append(subresult)
        return result

    def subtract_two_1d_boxes(a, b):
        """Given 2 intervals a = [a0,a1] and b=[b0,b1], return the part of a that does not intersect with b."""
        if intersect_two_1d_boxes(a, b) == None:
            return a
        else:
            if a.lb == b.lb:
                if math_utils.lt(a.ub, b.ub):
                    minArray = a
                    maxArray = b
                else:
                    minArray = b
                    maxArray = a
            elif math_utils.lt(a.lb, b.lb):
                minArray = a
                maxArray = b
            else:
                minArray = b
                maxArray = a
            if math_utils.gt(
                minArray.ub, maxArray.lb
            ):  ## has nonempty intersection. return intersection
                return [float(maxArray.lb), float(minArray.ub)]
            else:  ## no intersection.
                return []

        return [lhs, rhs]

        # if float(a.lb) <= float(b.lb):
        #     minArray = a
        #     maxArray = b
        # else:
        #     minArray = b
        #     maxArray = a
        # if minArray.ub > maxArray.lb: ## has nonempty intersection. return intersection
        #     return [float(maxArray.lb), float(minArray.ub)]
        # else: ## no intersection.
        #     return []

    def intersect_two_boxes(a: "Box", b: "Box"):
        a_params = list(a.bounds.keys())
        b_params = list(b.bounds.keys())

        beta_0_a = a.bounds[a_params[0]]
        beta_1_a = a.bounds[a_params[1]]
        beta_0_b = b.bounds[b_params[0]]
        beta_1_b = b.bounds[b_params[1]]

        beta_0 = Box.intersection(beta_0_a, beta_0_b)
        if len(beta_0) < 1:
            return None
        beta_1 = Box.intersection(beta_1_a, beta_1_b)
        if len(beta_1) < 1:
            return None

        return Box(
            [
                Parameter(a_params[0], lb=beta_0[0], ub=beta_0[1]),
                Parameter(a_params[1], lb=beta_1[0], ub=beta_1[1]),
            ]
        )

        # a = list(b1.bounds.values())
        # b = list(b2.bounds.values())

        # result = []
        # d = len(a) ## dimension
        # for i in range(d):
        #     subresult = Box.intersection(a[i],b[i])
        #     if subresult == []:
        #         return None
        #     else:
        #         result.append(subresult)
        # return result

    def subtract_two_1d_intervals(a: Interval, b: Interval) -> Interval:
        """Given 2 intervals a = [a0,a1] and b=[b0,b1], return the part of a that does not intersect with b."""

        if math_utils.lt(a.lb, b.lb):
            return Interval(a.lb, b.lb)

        if math_utils.gt(a.lb, b.lb):
            return Interval(b.ub, a.ub)

        return None

        # if intersect_two_1d_boxes(a,b) == None:
        #     return a
        # else:
        #     if a[0] < b[0]:
        #         return [a[0],b[0]]
        #     elif a[0] > b[0]:
        #         return [b[1],a[1]]

    ### WIP - just for 2 dimensions at this point.
    @staticmethod
    def symmetric_difference_two_boxes(a: "Box", b: "Box") -> List["Box"]:
        result: List["Box"] = []
        # if the same box then no symmetric difference
        if a == b:
            return result

        ## no intersection so they are disjoint - return both original boxes
        if Box.intersect_two_boxes(a, b) == None:
            return [a, b]

        # There must be some symmetric difference below here
        a_params = list(a.bounds.keys())
        b_params = list(b.bounds.keys())

        beta_0_a = a.bounds[a_params[0]]
        beta_1_a = a.bounds[a_params[1]]
        beta_0_b = b.bounds[b_params[0]]
        beta_1_b = b.bounds[b_params[1]]

        # TODO assumes 2 dimensions and aligned parameter names
        def make_box_2d(p_bounds):
            b0_bounds = p_bounds[0]
            b1_bounds = p_bounds[1]
            b = Box(
                [
                    Parameter(a_params[0], lb=b0_bounds.lb, ub=b0_bounds.ub),
                    Parameter(a_params[1], lb=b1_bounds.lb, ub=b1_bounds.ub),
                ]
            )
            return b

        xbounds = Box.subtract_two_1d_intervals(beta_0_a, beta_0_b)
        if xbounds != None:
            result.append(make_box_2d([xbounds, beta_1_a]))
        xbounds = Box.subtract_two_1d_intervals(beta_0_b, beta_0_a)
        if xbounds != None:
            result.append(make_box_2d([xbounds, beta_1_b]))
        ybounds = Box.subtract_two_1d_intervals(beta_1_a, beta_1_b)
        if ybounds != None:
            result.append(make_box_2d([beta_0_a, ybounds]))
        ybounds = Box.subtract_two_1d_intervals(beta_1_b, beta_1_a)
        if ybounds != None:
            result.append(make_box_2d([beta_0_b, ybounds]))

        return result


class SearchStatistics(object):
    def __init__(self, manager: SyncManager):
        self.multiprocessing = manager is not None
        self.num_true = manager.Value("i", 0) if self.multiprocessing else 0
        self.num_false = manager.Value("i", 0) if self.multiprocessing else 0
        self.num_unknown = manager.Value("i", 0) if self.multiprocessing else 0
        self.residuals = manager.Queue() if self.multiprocessing else SQueue()
        self.current_residual = (
            manager.Value("d", 0.0) if self.multiprocessing else 0.0
        )
        self.last_time = manager.Array("u", "") if self.multiprocessing else []
        self.iteration_time = (
            manager.Queue() if self.multiprocessing else SQueue()
        )
        self.iteration_operation = (
            manager.Queue() if self.multiprocessing else SQueue()
        )

    # def close(self):
    #     self.residuals.close()
    #     self.iteration_time.close()
    #     self.iteration_operation.close()


class WaitAction(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def run(self) -> None:
        pass


class ResultHandler(ABC):
    def __init__(self) -> None:
        pass

    def __enter__(self) -> "ResultHandler":
        self.open()
        return self

    def __exit__(self) -> None:
        self.close()

    @abstractmethod
    def open(self) -> None:
        pass

    @abstractmethod
    def process(self, result: dict) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass


class ResultCombinedHandler(ResultHandler):
    def __init__(self, handlers: List[ResultHandler]) -> None:
        self.handlers = handlers if handlers is not None else []

    def open(self) -> None:
        for h in self.handlers:
            try:
                h.open()
            except Exception as e:
                l.error(traceback.format_exc())

    def process(self, result: dict) -> None:
        for h in self.handlers:
            try:
                h.process(result)
            except Exception as e:
                l.error(traceback.format_exc())

    def close(self) -> None:
        for h in self.handlers:
            try:
                h.close()
            except Exception as e:
                l.error(traceback.format_exc())


class SearchConfig(Config):
    def __init__(
        self,
        *,  # non-positional keywords
        tolerance=1e-2,
        queue_timeout=1,
        number_of_processes=mp.cpu_count(),
        handler: ResultHandler = None,
        wait_timeout=None,
        wait_action=None,
        wait_action_timeout=0.05,
        read_cache=None,
        episode_type=None,
        search=None,
        solver="z3",
    ) -> None:
        self.tolerance = tolerance
        self.queue_timeout = queue_timeout
        self.number_of_processes = number_of_processes
        self.handler: ResultHandler = handler
        self.wait_timeout = wait_timeout
        self.wait_action = wait_action
        self.wait_action_timeout = wait_action_timeout
        self.read_cache = read_cache
        self.episode_type = episode_type
        self.search = search
        self.solver = solver


def _encode_labeled_box(box: Box, label: str):
    return {"label": label, "type": "box", "value": box.to_dict()}


def encode_true_box(box: Box):
    return _encode_labeled_box(box, "true")


def encode_false_box(box: Box):
    return _encode_labeled_box(box, "false")


def encode_unknown_box(box: Box):
    return _encode_labeled_box(box, "unknown")


def decode_labeled_box(box: dict):
    if box["type"] != "box":
        return None
    return (Box.from_dict(box["value"]), box["label"])


def _encode_labeled_point(point: Point, label: str):
    return {"label": label, "type": "point", "value": point.to_dict()}


def encode_true_point(point: Point):
    return _encode_labeled_point(point, "true")


def encode_false_point(point: Point):
    return _encode_labeled_point(point, "false")


def encode_unknown_point(point: Point):
    return _encode_labeled_point(point, "unkown")


def decode_labeled_point(point: dict):
    if point["type"] != "point":
        return None
    return (Point.from_dict(point["value"]), point["label"])


def decode_labeled_object(obj: dict):
    if obj["type"] == "point":
        return (decode_labeled_point(obj), Point)
    if obj["type"] == "box":
        return (decode_labeled_box(obj), Box)
    return None
