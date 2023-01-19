"""
This submodule contains definitions for the classes used
during the configuration and execution of a search.
"""
import logging
from functools import total_ordering
from multiprocessing.managers import SyncManager
from statistics import mean as average
from typing import Dict, List

from pysmt.shortcuts import GE, LE, LT, TRUE, And, Equals, Real

import funman.utils.math_utils as math_utils
from funman.constants import BIG_NUMBER, NEG_INFINITY, POS_INFINITY
from funman.model import Parameter

l = logging.getLogger(__name__)


class Interval(object):
    """
    An interval is a pair [lb, ub) that is open (i.e., an interval specifies all points x where lb <= x and ub < x).
    """

    def __init__(self, lb: float, ub: float) -> None:
        self.lb = lb
        self.ub = ub
        self.cached_width = None

    def width(self):
        """
        The width of an interval is ub - lb.

        Returns
        -------
        float
            ub - lb
        """
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
        return str(self.to_dict())

    def __str__(self):
        return f"Interval([{self.lb}, {self.ub}))"

    def finite(self) -> bool:
        """
        Are the lower and upper bounds finite?

        Returns
        -------
        bool
            bounds are finite
        """
        return self.lb != NEG_INFINITY and self.ub != POS_INFINITY

    def contains(self, other: "Interval") -> bool:
        """
        Does self contain other interval?

        Parameters
        ----------
        other : Interval
            interval to check for containment

        Returns
        -------
        bool
            self contains other
        """
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

    def intersection(self, b: "Interval") -> "Interval":
        """
        Given an interval b with self = [a0,a1] and b=[b0,b1], check whether they intersect.  If they do, return interval with their intersection.

        Parameters
        ----------
        b : Interval
            interval to check for intersection

        Returns
        -------
        Interval
            intersection of self and b
        """
        if self == b:
            return self
        else:
            if self.lb == b.lb:
                if math_utils.lt(self.ub, b.ub):
                    minArray = self
                    maxArray = b
                else:
                    minArray = b
                    maxArray = self
            elif math_utils.lt(self.lb, b.lb):
                minArray = self
                maxArray = b
            else:
                minArray = b
                maxArray = self
            if math_utils.gt(
                minArray.ub, maxArray.lb
            ):  ## has nonempty intersection. return intersection
                return [float(maxArray.lb), float(minArray.ub)]
            else:  ## no intersection.
                return []

    def subtract(self, b: "Interval") -> "Interval":
        """
        Given 2 intervals self = [a0,a1] and b=[b0,b1], return the part of self that does not intersect with b.

        Parameters
        ----------
        b : Interval
            interval to subtract from self

        Returns
        -------
        Interval
            self - b
        """

        if math_utils.lt(self.lb, b.lb):
            return Interval(self.lb, b.lb)

        if math_utils.gt(self.lb, b.lb):
            return Interval(b.ub, self.ub)

        return None

    def midpoint(self, points: List["Point"] = None):
        """
        Compute the midpoint of the interval.

        Parameters
        ----------
        points : List[Point], optional
            if specified, compute midpoint as average of points in the interval, by default None

        Returns
        -------
        float
            midpoint
        """

        # if points:
        #     return float(average(points))

        if self.lb == NEG_INFINITY and self.ub == POS_INFINITY:
            return 0
        elif self.lb == NEG_INFINITY:
            return self.ub - BIG_NUMBER
        if self.ub == POS_INFINITY:
            return self.lb + BIG_NUMBER
        else:
            return ((self.ub - self.lb) / 2) + self.lb

    def union(self, other: "Interval") -> List["Interval"]:
        """
        Union other interval with self.

        Returns
        -------
        List[Interval]
            union of intervals
        """
        if self == other:  ## intervals are equal, so return original interval
            ans = self
            total_height = [self.width()]
            return [ans], total_height
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
            ans = Interval(minInterval.lb, maxInterval.ub)
            total_height = ans.width()
            return [ans], total_height
        elif math_utils.lt(
            minInterval.ub, maxInterval.lb
        ):  ## intervals are disjoint.
            ans = [minInterval, maxInterval]
            total_height = [
                math_utils.plus(minInterval.width(), maxInterval.width())
            ]
            return ans, total_height

    def contains_value(self, value: float) -> bool:
        """
        Does the interval include a value?

        Parameters
        ----------
        value : float
            value to check for containment

        Returns
        -------
        bool
            the value is in the interval
        """
        lhs = self.lb == NEG_INFINITY or self.lb <= value
        rhs = self.ub == POS_INFINITY or value <= self.ub
        return lhs and rhs

    def to_smt(self, p: Parameter, closed_upper_bound=False):
        # FIXME move this into a translate utility
        """
        Convert the interval into contraints on parameter p.

        Parameters
        ----------
        p : Parameter
            parameter to constrain
        closed_upper_bound : bool, optional
            interpret interval as closed (i.e., p <= ub), by default False

        Returns
        -------
        FNode
            formula constraining p to the interval
        """
        lower = (
            GE(p._symbol(), Real(self.lb))
            if self.lb != NEG_INFINITY
            else TRUE()
        )
        upper_ineq = LE if closed_upper_bound else LT
        upper = (
            upper_ineq(p._symbol(), Real(self.ub))
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
        return f"Point({self.values})"

    def to_dict(self):
        return {"values": {k.name: v for k, v in self.values.items()}}

    def __repr__(self) -> str:
        return str(self.to_dict())

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

    @staticmethod
    def _encode_labeled_point(point: "Point", label: str):
        return {"label": label, "type": "point", "value": point.to_dict()}

    @staticmethod
    def encode_true_point(point: "Point"):
        return Point._encode_labeled_point(point, "true")

    @staticmethod
    def encode_false_point(point: "Point"):
        return Point._encode_labeled_point(point, "false")

    @staticmethod
    def encode_unknown_point(point: "Point"):
        return Point._encode_labeled_point(point, "unkown")

    @staticmethod
    def decode_labeled_point(point: dict):
        if point["type"] != "point":
            return None
        return (Point.from_dict(point["value"]), point["label"])


@total_ordering
class Box(object):
    """
    A Box maps n parameters to intervals, representing an n-dimensional connected open subset of R^n.
    """

    def __init__(self, parameters: List[Parameter]) -> None:
        self.bounds: Dict[Parameter, Interval] = {
            p: Interval(p.lb, p.ub) for p in parameters
        }
        self.cached_width = None

    def to_smt(self, closed_upper_bound=False):
        """
        Compile the interval for each parameter into SMT constraints on the corresponding parameter.

        Parameters
        ----------
        closed_upper_bound : bool, optional
            use closed upper bounds for each interval, by default False

        Returns
        -------
        FNode
            formula representing the box as a conjunction of interval constraints.
        """
        return And(
            [
                interval.to_smt(p, closed_upper_bound=closed_upper_bound)
                for p, interval in self.bounds.items()
            ]
        )

    def to_dict(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Covert Box to a dict.

        Returns
        -------
        dict
            dictionary representing the box
        """
        return {
            "bounds": {k.name: v.to_dict() for k, v in self.bounds.items()},
            # "cached_width": self.cached_width
        }

    @staticmethod
    def from_dict(data: Dict[str, Dict[str, Dict[str, float]]]) -> "Box":
        """
        Create a box from a dict

        Parameters
        ----------
        data : dict
            box represented as dict

        Returns
        -------
        Box
            Box object for dict
        """
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
        return str(self.to_dict())

    def __str__(self):
        return f"Box({self.bounds})"

    def finite(self) -> bool:
        """
        Are all parameter intervals finite?

        Returns
        -------
        bool
            all parameter intervals are finite
        """
        return all([i.finite() for _, i in self.bounds.items()])

    def contains(self, other: "Box") -> bool:
        """
        Does the interval for each parameter in self contain the interval for the corresponding parameter in other?

        Parameters
        ----------
        other : Box
            other box

        Returns
        -------
        bool
            self contains other
        """
        return all(
            [
                interval.contains(other.bounds[p])
                for p, interval in self.bounds.items()
            ]
        )

    def contains_point(self, point: Point) -> bool:
        """
        Does the box contain a point?

        Parameters
        ----------
        point : Point
            a point

        Returns
        -------
        bool
            the box contains the point
        """
        return all(
            [
                interval.contains_value(point.values[p])
                for p, interval in self.bounds.items()
            ]
        )

    def equal(
        self, b2: "Box", param_list: List[str] = None
    ) -> bool:  ## added 11/27/22 DMI
        ## FIXME @dmosaphir use Parameter instead of str for param_list
        """
        Are two boxes equal, considering only parameters in param_list?

        Parameters
        ----------
        b1 : Box
            box 1
        b2 : Box
            box 2
        param_list : list
            parameters over which to restrict the comparison

        Returns
        -------
        bool
            boxes are equal
        """

        if param_list:
            result = []
            for p1 in param_list:
                for b in self.bounds:
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
        else:
            return self == b

    def intersects(self, other: "Box") -> bool:
        """
        Does self and other intersect? I.e., do all parameter intervals instersect?

        Parameters
        ----------
        other : Box
            other box

        Returns
        -------
        bool
            self intersects other
        """
        return all(
            [
                interval.intersects(other.bounds[p])
                for p, interval in self.bounds.items()
            ]
        )

    def _get_max_width_point_parameter(self, points: List[Point]):
        """
        Get the parameter that has the maximum average distance from the center point for each parameter and the value for the parameter assigned by each point.

        Parameters
        ----------
        points : List[Point]
            Points in the box

        Returns
        -------
        Parameter
            parameter (dimension of box) where points are most distant from the center of the box.
        """
        #
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
        max_width_parameter = max(
            parameter_widths, key=lambda k: parameter_widths[k]
        )
        return max_width_parameter

    def _get_max_width_parameter(self):
        widths = [bounds.width() for _, bounds in self.bounds.items()]
        max_width = max(widths)
        param = list(self.bounds.keys())[widths.index(max_width)]
        return param, max_width

    def width(self) -> float:
        """
        The width of a box is the maximum width of a parameter interval.

        Returns
        -------
        float
            Max{p: parameter}(p.ub-p.lb)
        """
        if self.cached_width is None:
            _, width = self._get_max_width_parameter()
            self.cached_width = width

        return self.cached_width

    def split(self, points: List[Point] = None):
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
            if mid == self.bounds[p].lb or mid == self.bounds[p].ub:
                # Fall back to box midpoint if point-based mid is degenerate
                p, _ = self._get_max_width_parameter()
                mid = self.bounds[p].midpoint()
        else:
            p, _ = self._get_max_width_parameter()
            mid = self.bounds[p].midpoint()

        b1 = self._copy()
        b2 = self._copy()

        # b1 is lower half
        assert math_utils.lte(b1.bounds[p].lb, mid)
        b1.bounds[p] = Interval(b1.bounds[p].lb, mid)

        # b2 is upper half
        assert math_utils.lte(mid, b2.bounds[p].ub)
        b2.bounds[p] = Interval(mid, b2.bounds[p].ub)

        return [b2, b1]

    @staticmethod
    def _encode_labeled_box(box: "Box", label: str):
        return {"label": label, "type": "box", "value": box.to_dict()}

    @staticmethod
    def _encode_true_box(box: "Box"):
        return Box._encode_labeled_box(box, "true")

    @staticmethod
    def _encode_false_box(box: "Box"):
        return Box._encode_labeled_box(box, "false")

    @staticmethod
    def _encode_unknown_box(box: "Box"):
        return Box._encode_labeled_box(box, "unknown")

    @staticmethod
    def _decode_labeled_box(box: dict):
        if box["type"] != "box":
            return None
        return (Box.from_dict(box["value"]), box["label"])

    def intersect(
        self, b2: "Box", param_list: List[str] = None
    ):  ## added 11/21/22 DMM
        """
        Intersect self with box, optionally over only a subset of the dimensions listed in param_list.

        Parameters
        ----------
        b2 : Box
            box to intersect with
        param_list : List[str], optional
            parameters to intersect, by default None

        Returns
        -------
        Box
            box representing intersection, optionally defined over parameters in param_list (when specified)
        """
        result = []
        common_params = (
            param_list if param_list else [k.name for k in self.bounds]
        )
        for p1 in common_params:
            # FIXME iterating over dict keys is not efficient
            for b in self.bounds:
                if b.name == p1:
                    b1_bounds = Interval(b.lb, b.ub)
            for b in b2.bounds:
                if b.name == p1:
                    b2_bounds = Interval(b.lb, b.ub)
            intersection_ans = b1_bounds.intersection(b2_bounds)
            dict_element = Parameter(
                p1, intersection_ans[0], intersection_ans[1]
            )
            result.append(dict_element)
        return Box({i for i in result})

    def __intersect_two_boxes(b1, b2):
        # FIXME subsumed by Box.intersect(), can be removed.
        a = list(b1.bounds.values())
        b = list(b2.bounds.values())
        result = []
        d = len(a)  ## dimension
        for i in range(d):
            subresult = a[i].intersection(b[i])
            if subresult == []:
                return None
            else:
                result.append(subresult)
        return result

    @staticmethod
    def _subtract_two_1d_boxes(a, b):
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

    @staticmethod
    def __intersect_two_boxes(a: "Box", b: "Box"):
        # FIXME not sure how this is different than Box.intersect()
        a_params = list(a.bounds.keys())
        b_params = list(b.bounds.keys())

        beta_0_a = a.bounds[a_params[0]]
        beta_1_a = a.bounds[a_params[1]]
        beta_0_b = b.bounds[b_params[0]]
        beta_1_b = b.bounds[b_params[1]]

        beta_0 = beta_0_a.intersection(beta_0_b)
        if len(beta_0) < 1:
            return None
        beta_1 = beta_1_a.intersection(beta_1_b)
        if len(beta_1) < 1:
            return None

        return Box(
            [
                Parameter(a_params[0], lb=beta_0[0], ub=beta_0[1]),
                Parameter(a_params[1], lb=beta_1[0], ub=beta_1[1]),
            ]
        )

    ### WIP - just for 2 dimensions at this point.
    def _symmetric_difference_two_boxes(self, b: "Box") -> List["Box"]:
        result: List["Box"] = []
        # if the same box then no symmetric difference
        if self == b:
            return result

        ## no intersection so they are disjoint - return both original boxes
        if Box.__intersect_two_boxes(self, b) == None:
            return [self, b]

        # There must be some symmetric difference below here
        a_params = list(self.bounds.keys())
        b_params = list(b.bounds.keys())

        beta_0_a = self.bounds[a_params[0]]
        beta_1_a = self.bounds[a_params[1]]
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

        xbounds = Box._subtract_two_1d_intervals(beta_0_a, beta_0_b)
        if xbounds != None:
            result.append(make_box_2d([xbounds, beta_1_a]))
        xbounds = Box._subtract_two_1d_intervals(beta_0_b, beta_0_a)
        if xbounds != None:
            result.append(make_box_2d([xbounds, beta_1_b]))
        ybounds = Box._subtract_two_1d_intervals(beta_1_a, beta_1_b)
        if ybounds != None:
            result.append(make_box_2d([beta_0_a, ybounds]))
        ybounds = Box._subtract_two_1d_intervals(beta_1_b, beta_1_a)
        if ybounds != None:
            result.append(make_box_2d([beta_0_b, ybounds]))

        return result


class ParameterSpace(object):
    """
    This class defines the representation of the parameter space that can be
    returned by the parameter synthesis feature of FUNMAN. These parameter spaces
    are represented as a collection of boxes that are either known to be true or
    known to be false.
    """

    def __init__(
        self,
        true_boxes: List[Box],
        false_boxes: List[Box],
        true_points: List[Point],
        false_points: List[Point],
    ) -> None:
        self.true_boxes = true_boxes
        self.false_boxes = false_boxes
        self.true_points = true_points
        self.false_points = false_points

    # STUB project parameter space onto a parameter
    @staticmethod
    def project() -> "ParameterSpace":
        raise NotImplementedError()
        return ParameterSpace()

    @staticmethod
    def _union_boxes(b1s):
        results_list = []
        for i1 in range(len(b1s)):
            for i2 in range(i1 + 1, len(b1s)):
                ans = Box.check_bounds_disjoint_equal(b1s[i1], b1s[i2])
                print(ans)
        return results_list

    @staticmethod
    def _intersect_boxes(b1s, b2s):
        results_list = []
        for box1 in b1s:
            for box2 in b2s:
                subresult = Box.__intersect_two_boxes(box1, box2)
                if subresult != None:
                    results_list.append(subresult)
        return results_list

    # STUB intersect parameters spaces
    @staticmethod
    def intersect(ps1, ps2):
        return ParameterSpace(
            ParameterSpace._intersect_boxes(ps1.true_boxes, ps2.true_boxes),
            ParameterSpace._intersect_boxes(ps1.false_boxes, ps2.false_boxes),
        )

    @staticmethod
    def symmetric_difference(ps1: "ParameterSpace", ps2: "ParameterSpace"):
        return ParameterSpace(
            ParameterSpace._symmetric_difference(
                ps1.true_boxes, ps2.true_boxes
            ),
            ParameterSpace._symmetric_difference(
                ps1.false_boxes, ps2.false_boxes
            ),
        )

    @staticmethod
    def _symmetric_difference(ps1: List[Box], ps2: List[Box]) -> List[Box]:
        results_list = []

        for box2 in ps2:
            box2_results = []
            should_extend = True
            for box1 in ps1:
                subresult = Box._symmetric_difference_two_boxes(box2, box1)
                if subresult != None:
                    box2_results.extend(subresult)
                else:
                    should_extend = False
                    break
            if should_extend:
                results_list.extend(box2_results)

        for box1 in ps1:
            box1_results = []
            should_extend = True
            for box2 in ps2:
                subresult = Box._symmetric_difference_two_boxes(box1, box2)
                if subresult != None:
                    box1_results.extend(subresult)
                else:
                    should_extend = False
                    break
            if should_extend:
                results_list.extend(box1_results)

        return results_list

    # STUB construct space where all parameters are equal
    @staticmethod
    def construct_all_equal(ps) -> "ParameterSpace":
        raise NotImplementedError()
        return ParameterSpace()

    # STUB compare parameter spaces for equality
    @staticmethod
    def compare(ps1, ps2) -> bool:
        raise NotImplementedError()

    def plot(self, color="b", alpha=0.2):
        custom_lines = [
            Line2D([0], [0], color="g", lw=4, alpha=alpha),
            Line2D([0], [0], color="r", lw=4, alpha=alpha),
        ]
        plt.title("Parameter Space")
        plt.xlabel("beta_0")
        plt.ylabel("beta_1")
        plt.legend(custom_lines, ["true", "false"])
        for b1 in self.true_boxes:
            BoxPlotter.plot2DBoxList(b1, color="g")
        for b1 in self.false_boxes:
            BoxPlotter.plot2DBoxList(b1, color="r")
        # plt.show(block=True)

    @staticmethod
    def decode_labeled_object(obj: dict):
        if not isinstance(obj, dict):
            raise Exception("obj is not a dict")
        if "type" not in obj:
            raise Exception("obj does not specify a 'type' field")
        if obj["type"] == "point":
            return (Point.decode_labeled_point(obj), Point)
        if obj["type"] == "box":
            return (Box._decode_labeled_box(obj), Box)
        raise Exception(f"obj of type {obj['type']}")

    def __repr__(self) -> str:
        return str(self.to_dict())

    def to_dict(self) -> Dict[str, Dict]:
        return {
            "true_boxes": list(map(lambda x: x.to_dict(), self.true_boxes)),
            "false_boxes": list(map(lambda x: x.to_dict(), self.false_boxes)),
        }
