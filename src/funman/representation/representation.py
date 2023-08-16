"""
This submodule contains definitions for the classes used
during the configuration and execution of a search.
"""
import copy
import logging
import math
import sys
from decimal import ROUND_CEILING, Decimal
from statistics import mean as average
from typing import Dict, List, Literal, Optional, Union

from pydantic import ConfigDict, BaseModel, Field
from pysmt.fnode import FNode
from pysmt.shortcuts import REAL, Symbol

import funman.utils.math_utils as math_utils
from funman.constants import BIG_NUMBER, NEG_INFINITY, POS_INFINITY
from funman.utils.sympy_utils import to_sympy

from .symbol import ModelSymbol

l = logging.getLogger(__name__)


LABEL_TRUE: Literal["true"] = "true"
LABEL_FALSE: Literal["false"] = "false"
LABEL_UNKNOWN: Literal["unknown"] = "unknown"
LABEL_DROPPED: Literal["dropped"] = "dropped"
Label = Literal["true", "false", "unknown", "dropped"]

LABEL_ANY = "any"
LABEL_ALL = "all"


class ModelParameter(BaseModel):
    name: Union[str, ModelSymbol]
    lb: Union[float, str] = NEG_INFINITY
    ub: Union[float, str] = POS_INFINITY

    def width(self) -> Union[str, float]:
        return math_utils.minus(self.ub, self.lb)
    
    def is_unbound(self) -> bool:
        return self.lb == NEG_INFINITY and self.ub == POS_INFINITY

    def __hash__(self):
        return abs(hash(self.name))


class LabeledParameter(ModelParameter):
    label: Literal["any", "all"] = LABEL_ANY

    def is_synthesized(self) -> bool:
        return self.label == LABEL_ALL and self.width() > 0.0


class StructureParameter(LabeledParameter):
    def is_synthesized(self):
        return True


class ModelParameter(LabeledParameter):
    """
    A parameter is a free variable for a Model.  It has the following attributes:

    * lb: lower bound

    * ub: upper bound

    * symbol: a pysmt FNode corresponding to the parameter variable

    """

    model_config = ConfigDict(extra="forbid")

    _symbol: FNode = None

    def symbol(self):
        """
        Get a pysmt Symbol for the parameter

        Returns
        -------
        pysmt.fnode.FNode
            _description_
        """
        if not self._symbol:
            self._symbol = Symbol(self.name, REAL)
        return self._symbol

    def timed_copy(self, timepoint: int):
        """
        Create a time-stamped copy of a parameter.  E.g., beta becomes beta_t for a timepoint t

        Parameters
        ----------
        timepoint : int
            Integer timepoint

        Returns
        -------
        Parameter
            A time-stamped copy of self.
        """
        timed_parameter = copy.deepcopy(self)
        timed_parameter.name = f"{timed_parameter.name}_{timepoint}"
        return timed_parameter

    def __eq__(self, other):
        if not isinstance(other, ModelParameter):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return self.name == other.name

    def __hash__(self):
        # necessary for instances to behave sanely in dicts and sets.
        return hash(self.name)

    def __repr__(self) -> str:
        return f"{self.name}[{self.lb}, {self.ub})"


class Interval(BaseModel):
    """
    An interval is a pair [lb, ub) that is open (i.e., an interval specifies all points x where lb <= x and ub < x).
    """

    lb: Union[float, str]
    ub: Union[float, str]
    cached_width: Optional[float] = Field(default=None, exclude=True)

    def __hash__(self):
        return int(math_utils.plus(self.lb, self.ub))

    def disjoint(self, other: "Interval") -> bool:
        """
        Is self disjoint (non overlapping) with other?

        Parameters
        ----------
        other : Interval
            other interval

        Returns
        -------
        bool
            are the intervals disjoint?
        """
        return math_utils.lte(self.ub, other.lb) or math_utils.gte(
            self.lb, other.ub
        )

    def width(
        self, normalize: Optional[Union[Decimal, float]] = None
    ) -> Decimal:
        """
        The width of an interval is ub - lb.

        Returns
        -------
        float
            ub - lb
        """
        if self.cached_width is None:
            self.cached_width = Decimal(self.ub) - Decimal(self.lb)
        if normalize is not None:
            return self.cached_width / normalize
        else:
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
        return str(self.dict())

    def __str__(self):
        return f"Interval([{self.lb}, {self.ub}))"

    def meets(self, other: "Interval") -> bool:
        """
        Does self meet other?

        Parameters
        ----------
        other : Interval
            another inteval

        Returns
        -------
        bool
            Does self meet other?
        """
        return self.ub == other.lb or self.lb == other.ub

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
        return (
            self.contains_value(other.lb)
            or other.contains_value(self.lb)
            or (
                self.contains_value(other.ub)
                and math_utils.gt(other.ub, self.lb)
            )
            or (
                other.contains_value(self.ub)
                and math_utils.gt(self.ub, other.lb)
            )
        )

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
            return Interval(lb=self.lb, ub=b.lb)

        if math_utils.gt(self.lb, b.lb):
            return Interval(lb=b.ub, ub=self.ub)

        return None

    def midpoint(self, points: List[List["Point"]] = None):
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

        if points:
            # Find mean of groups and mid point bisects the means
            means = [float(average(grp)) for grp in points]
            mid = float(average(means))
            return mid

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
            # total_height = [self.width()]
            return [ans]
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
            ans = Interval(lb=minInterval.lb, ub=maxInterval.ub)
            # total_height = ans.width()
            return [ans]
        elif math_utils.lt(
            minInterval.ub, maxInterval.lb
        ):  ## intervals are disjoint.
            ans = [minInterval, maxInterval]
            # total_height = [
            #     math_utils.plus(minInterval.width(), maxInterval.width())
            # ]
            return ans

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
        return math_utils.gte(value, self.lb) and math_utils.lt(value, self.ub)


class Point(BaseModel):
    type: Literal["point"] = "point"
    label: Label = LABEL_UNKNOWN
    values: Dict[str, float]

    # def __init__(self, **kw) -> None:
    #     super().__init__(**kw)
    #     self.values = kw['values']

    def __str__(self):
        return f"Point({self.dict()})"

    def __repr__(self) -> str:
        return str(self.dict())

    @staticmethod
    def from_dict(data):
        res = Point(values={k: v for k, v in data["values"].items()})
        return res

    def denormalize(self, model):
        norm = to_sympy(model.normalization(), model._symbols())
        denormalized_values = {
            k: (v * norm if model._is_normalized(k) else v)
            for k, v in self.values.items()
        }
        denormalized_point = Point(
            label=self.label, values=denormalized_values, type=self.type
        )
        return denormalized_point

    def __hash__(self):
        return int(
            sum(
                [
                    v
                    for _, v in self.values.items()
                    if v != sys.float_info.max and not math.isinf(v)
                ]
            )
        )

    def __eq__(self, other):
        if isinstance(other, Point):
            return all(
                [self.values[p] == other.values[p] for p in self.values.keys()]
            )
        else:
            return False


# @total_ordering
class Box(BaseModel):
    """
    A Box maps n parameters to intervals, representing an n-dimensional connected open subset of R^n.
    """

    type: Literal["box"] = "box"
    label: Label = LABEL_UNKNOWN
    bounds: Dict[str, Interval] = {}
    cached_width: Optional[float] = Field(default=None, exclude=True)

    def __hash__(self):
        return int(sum([i.__hash__() for _, i in self.bounds.items()]))

    def advance(self):
        # Advancing a box means that we move the time step forward until it exhausts the possible number of steps
        if (
            "num_steps" in self.bounds
            and self.bounds["num_steps"].lb < self.bounds["num_steps"].ub
        ):
            advanced_box = Box(
                bounds={
                    n: (
                        itv
                        if n != "num_steps"
                        else Interval(lb=itv.lb + 1, ub=itv.ub)
                    )
                    for n, itv in self.bounds.items()
                }
            )
            return advanced_box
        else:
            None

    def current_step(self):
        # Restrict bounds on num_steps to the lower bound (i.e., the current step)
        if "num_steps" in self.bounds:
            current_step_box = Box(
                bounds={
                    n: (
                        itv
                        if n != "num_steps"
                        else Interval(lb=itv.lb, ub=itv.lb)
                    )
                    for n, itv in self.bounds.items()
                }
            )
            return current_step_box
        else:
            None

    def project(self, vars: Union[List[ModelParameter], List[str]]) -> "Box":
        """
        Takes a subset of selected variables (vars_list) of a given box (b) and returns another box that is given by b's values for only the selected variables.

        Parameters
        ----------
        vars : Union[List[Parameter], List[str]]
            variables to project onto

        Returns
        -------
        Box
            projected box

        """
        bp = copy.deepcopy(self)
        if len(vars) > 0:
            if isinstance(vars[0], str):
                bp.bounds = {k: v for k, v in bp.bounds.items() if k in vars}
            elif isinstance(vars[0], ModelParameter):
                vars_str = [v.name for v in vars]
                bp.bounds = {
                    k: v for k, v in bp.bounds.items() if k in vars_str
                }
            else:
                raise Exception(
                    f"Unknown type {type(vars[0])} used as intput to Box.project()"
                )
        else:
            bp.bounds = {}
        return bp

    def _merge(self, other: "Box") -> "Box":
        """
        Merge two boxes.  This function assumes the boxes meet in one dimension and are equal in all others.

        Parameters
        ----------
        other : Box
            other box

        Returns
        -------
        Box
            merge of two boxes that meet in one dimension
        """
        merged = Box(
            bounds={p: Interval(lb=0, ub=0) for p in self.bounds.keys()}
        )
        for p, i in merged.bounds.items():
            if self.bounds[p].meets(other.bounds[p]):
                i.lb = min(self.bounds[p].lb, other.bounds[p].lb)
                i.ub = max(self.bounds[p].ub, other.bounds[p].ub)
            else:
                i.lb = self.bounds[p].lb
                i.ub = self.bounds[p].ub
        return merged

    def _get_merge_candidates(self, boxes: Dict[ModelParameter, List["Box"]]):
        equals_set = set([])
        meets_set = set([])
        disqualified_set = set([])
        for p in boxes:
            sorted = boxes[p]
            # find boxes in sorted that meet or equal self in dimension p
            self_index = sorted.index(self)
            # sorted is sorted by upper bound, and candidate boxes are either
            # before or after self in the list
            # search backward
            for r in [
                reversed(range(self_index)),  # search forward
                range(self_index + 1, len(boxes[p])),  # search backward
            ]:
                for i in r:
                    if (
                        sorted[i].bounds[p].meets(self.bounds[p])
                        and sorted[i] not in disqualified_set
                    ):
                        if sorted[i] in meets_set:
                            # Need exactly one dimension where they meet, so disqualified
                            meets_set.remove(sorted[i])
                            disqualified_set.add(sorted[i])
                        else:
                            meets_set.add(sorted[i])
                    elif (
                        sorted[i].bounds[p] == self.bounds[p]
                        and sorted[i] not in disqualified_set
                    ):
                        equals_set.add(sorted[i])
                    else:
                        if sorted[i] in meets_set:
                            meets_set.remove(sorted[i])
                        if sorted[i] in equals_set:
                            equals_set.remove(sorted[i])
                        disqualified_set.add(sorted[i])
                    if sorted[i].bounds[p].disjoint(
                        self.bounds[p]
                    ) and not sorted[i].bounds[p].meets(self.bounds[p]):
                        break  # Because sorted, no further checking needed
        if len(boxes.keys()) == 1:  # 1D
            candidates = meets_set
        else:  # > 1D
            candidates = meets_set.intersection(equals_set)
        return candidates

    def _copy(self):
        c = Box(
            bounds={
                p: Interval(lb=b.lb, ub=b.ub) for p, b in self.bounds.items()
            }
        )
        return c

    def __lt__(self, other):
        if isinstance(other, Box):
            return self.width() < other.width()
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
        return str(self.dict())

    def __str__(self):
        return f"Box({self.bounds}), width = {self.width()}"

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

    def _get_max_width_point_Parameter(self, points: List[List[Point]]):
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
        group_centers = {
            p: [average([pt.values[p] for pt in grp]) for grp in points]
            for p in self.bounds
        }
        centers = {p: average(grp) for p, grp in group_centers.items()}
        # print(points)
        # print(centers)
        point_distances = [
            {
                p: abs(pt.values[p] - centers[p])
                for p in pt.values
                if p in centers
            }
            for grp in points
            for pt in grp
        ]
        parameter_widths = {
            p: average([pt[p] for pt in point_distances]) for p in self.bounds
        }
        # normalized_parameter_widths = {
        #     p: average([pt[p] for pt in point_distances])
        #     / (self.bounds[p].width())
        #     for p in self.bounds
        #     if self.bounds[p].width() > 0
        # }
        max_width_parameter = max(
            parameter_widths, key=lambda k: parameter_widths[k]
        )
        return max_width_parameter

    def _get_max_width_Parameter(
        self, normalize={}, parameters: List[ModelParameter] = None
    ) -> Union[str, ModelSymbol]:
        if parameters:
            widths = {
                parameter.name: (
                    self.bounds[parameter.name].width(
                        normalize=normalize[parameter.name]
                    )
                    if parameter.name in normalize
                    else self.bounds[parameter.name].width()
                )
                for parameter in parameters
            }
        else:
            widths = {
                p: (
                    self.bounds[p].width(normalize=normalize[p])
                    if p in normalize
                    else self.bounds[p].width()
                )
                for p in self.bounds
            }
        max_width = max(widths, key=widths.get)

        return max_width

    def _get_min_width_Parameter(
        self, normalize={}, parameters: List[ModelParameter] = None
    ) -> Union[str, ModelSymbol]:
        if parameters:
            widths = {
                parameter.name: (
                    self.bounds[parameter.name].width(
                        normalize=normalize[parameter.name]
                    )
                    if parameter.name in normalize
                    else self.bounds[parameter.name].width()
                )
                for parameter in parameters
            }
        else:
            widths = {
                p: (
                    self.bounds[p].width(normalize=normalize[p])
                    if p in normalize
                    else self.bounds[p].width()
                )
                for p in self.bounds
            }
        min_width = min(widths, key=widths.get)

        return min_width

    def volume(
        self,
        normalize=None,
        parameters: List[ModelParameter] = None,
        *,
        ignore_zero_width_dimensions=True,
    ) -> Decimal:
        # construct a list of parameter names to consider
        # if no parameters are requested then use all of the bounds
        if parameters is None:
            pnames = list(self.bounds.keys())
        else:
            pnames = [
                p.name if isinstance(p.name, str) else p.name.name
                for p in parameters
            ]

        # handle the volume of zero dimensions
        if len(pnames) <= 0:
            return Decimal("nan")

        # if no parameters are normalized then default to an empty dict
        if normalize is None:
            normalize = {}

        # get a mapping of parameters to widths
        # use normalize.get(p.name, None) to select between default behavior and normalization
        widths = {
            p: self.bounds[p].width(normalize=normalize.get(p, None))
            for p in pnames
        }
        if ignore_zero_width_dimensions:
            # filter widths of zero from the
            widths = {p: w for p, w in widths.items() if w != 0.0}

        # TODO in there a 'class' of parameters that we can identify
        # that need this same treatment. Specifically looking for
        # strings 'num_steps' and 'step_size' is brittle.
        num_timepoints = 1
        if "num_steps" in widths:
            del widths["num_steps"]
            # TODO this timepoint computation could use more thought
            # for the moment it just takes the ceil(width) + 1.0
            # so num steps 1.0 to 2.5 would result in:
            # ceil(2.5 - 1.0) + 1.0 = 3.0
            num_timepoints = Decimal(
                self.bounds["num_steps"].width()
            ).to_integral_exact(rounding=ROUND_CEILING)
            num_timepoints += 1
        if "step_size" in widths:
            del widths["step_size"]

        if len(widths) <= 0:
            # TODO handle volume of a point
            return Decimal(0.0)

        # compute product
        product = Decimal(1.0)
        for param_width in widths.values():
            if param_width < 0:
                raise Exception("Negative parameter width")
            product *= Decimal(param_width)
        product *= num_timepoints
        return product

    def width(
        self,
        normalize={},
        overwrite_cache=False,
        parameters: List[ModelParameter] = None,
    ) -> float:
        """
        The width of a box is the maximum width of a parameter interval.

        Returns
        -------
        float
            Max{p: parameter}(p.ub-p.lb)
        """
        if self.cached_width is None or overwrite_cache:
            p = self._get_max_width_Parameter(
                normalize=normalize, parameters=parameters
            )
            self.cached_width = self.bounds[p].width(
                normalize=normalize.get(p, None)
            )
        return self.cached_width

    def variance(self, overwrite_cache=False) -> float:
        """
        The variance of a box is the maximum variance of a parameter interval.
        STUB for Milestone 8 sensitivity analysis

        Returns
        -------
        float
            Variance{p: parameter}
        """
        pass

    def split(
        self,
        points: List[List[Point]] = None,
        normalize: Dict[str, float] = {},
        parameters=[],
    ):
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
            p = self._get_max_width_point_Parameter(points)
            mid = self.bounds[p].midpoint(
                points=[[pt.values[p] for pt in grp] for grp in points]
            )
            if mid == self.bounds[p].lb or mid == self.bounds[p].ub:
                # Fall back to box midpoint if point-based mid is degenerate
                p = self._get_max_width_Parameter()
                mid = self.bounds[p].midpoint()
        else:
            p = self._get_max_width_Parameter(
                normalize=normalize, parameters=parameters
            )
            mid = self.bounds[p].midpoint()

        b1 = self._copy()
        b2 = self._copy()

        # b1 is lower half
        assert math_utils.lte(b1.bounds[p].lb, mid)
        b1.bounds[p] = Interval(lb=b1.bounds[p].lb, ub=mid)

        # b2 is upper half
        assert math_utils.lte(mid, b2.bounds[p].ub)
        b2.bounds[p] = Interval(lb=mid, ub=b2.bounds[p].ub)

        return [b2, b1]

    def intersect(self, b2: "Box", param_list: List[str] = None):
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
        params_ans = []
        common_params = (
            param_list if param_list else [k.name for k in self.bounds]
        )
        for p1 in common_params:
            # FIXME iterating over dict keys is not efficient
            for b, i in self.bounds.items():
                if b == p1:
                    b1_bounds = Interval(lb=i.lb, ub=i.ub)
            for b, i in b2.bounds.items():
                if b == p1:
                    b2_bounds = Interval(lb=i.lb, ub=i.ub)
            intersection_ans = b1_bounds.intersection(
                b2_bounds
            )  ## will be a list with 2 elements (lower and upper bound) or empty list
            if (
                len(intersection_ans) < 1
            ):  ## empty list: no intersection in 1 variable means no intersection overall.
                return None
            else:
                new_param = ModelParameter(
                    name=f"{p1}",
                    lb=intersection_ans[0],
                    ub=intersection_ans[1],
                )
                params_ans.append(new_param)
        return Box(
            bounds={p.name: Interval(lb=p.lb, ub=p.ub) for p in params_ans}
        )

    def symm_diff(b1: "Box", b2: "Box"):
        result = []
        ## First check that the two boxes have the same variables
        vars_b1 = set([b for b in b1.bounds])
        vars_b2 = set([b for b in b2.bounds])
        if vars_b1 == vars_b2:
            vars_list = list(vars_b1)
            print("symm diff in progress")
        else:
            print(
                "cannot take the symmetric difference of two boxes that do not have the same variables."
            )
            raise Exception(
                "Cannot take symmetric difference since the two boxes do not have the same variables"
            )
        ### Find intersection
        desired_vars_list = list(vars_b1)
        intersection = b1.intersect(b2, param_list=desired_vars_list)
        ### Calculate symmetric difference based on intersection
        if (
            intersection == None
        ):  ## No intersection, so symmetric difference is just the original boxes
            return [b1, b2]
        else:  ## Calculate symmetric difference
            unknown_boxes = [b1, b2]
            false_boxes = []
            true_boxes = []
            while len(unknown_boxes) > 0:
                b = unknown_boxes.pop()
                if Box.contains(intersection, b) == True:
                    false_boxes.append(b)
                elif Box.contains(b, intersection) == True:
                    new_boxes = Box.split(b)
                    for i in range(len(new_boxes)):
                        unknown_boxes.append(new_boxes[i])
                else:
                    true_boxes.append(b)
            return true_boxes

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
            bounds={
                ModelParameter(
                    name=a_params[0], lb=beta_0[0], ub=beta_0[1]
                ): Interval(lb=beta_0[0], ub=beta_0[1]),
                ModelParameter(
                    name=a_params[1], lb=beta_1[0], ub=beta_1[1]
                ): Interval(lb=beta_1[0], ub=beta_1[1]),
            }
        )

    ### Can remove and replace this with Box.symm_diff, which works for any number of dimensions.  TODO write a corresponding parameter space symmetric difference and use case.
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
                bounds={
                    ModelParameter(
                        name=a_params[0], lb=b0_bounds.lb, ub=b0_bounds.ub
                    ): Interval(lb=b0_bounds.lb, ub=b0_bounds.ub),
                    ModelParameter(
                        name=a_params[1], lb=b1_bounds.lb, ub=b1_bounds.ub
                    ): Interval(lb=b1_bounds.lb, ub=b1_bounds.ub),
                }
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


class ParameterSpace(BaseModel):
    """
    This class defines the representation of the parameter space that can be
    returned by the parameter synthesis feature of FUNMAN. These parameter spaces
    are represented as a collection of boxes that are either known to be true or
    known to be false.
    """

    num_dimensions: int = None
    true_boxes: List[Box] = []
    false_boxes: List[Box] = []
    true_points: List[Point] = []
    false_points: List[Point] = []

    @staticmethod
    def _from_configurations(
        configurations: List[Dict[str, Union[int, "ParameterSpace"]]]
    ) -> "ParameterSpace":
        all_ps = ParameterSpace()
        for result in configurations:
            ps = result["parameter_space"]
            num_steps = result["num_steps"]
            step_size = result["step_size"]
            for box in ps.true_boxes:
                box.bounds["num_steps"] = Interval(
                    lb=num_steps, ub=num_steps + 1
                )
                box.bounds["step_size"] = Interval(
                    lb=step_size, ub=step_size + 1
                )
                all_ps.true_boxes.append(box)
            for box in ps.false_boxes:
                box.bounds["num_steps"] = Interval(
                    lb=num_steps, ub=num_steps + 1
                )
                box.bounds["step_size"] = Interval(
                    lb=step_size, ub=step_size + 1
                )
                all_ps.false_boxes.append(box)
            for point in ps.true_points:
                point.values["num_steps"] = num_steps
                point.values["step_size"] = step_size
                all_ps.true_points.append(point)
            for point in ps.false_points:
                point.values["num_steps"] = num_steps
                point.values["step_size"] = step_size
                all_ps.false_points.append(point)
        return all_ps

    # STUB project parameter space onto a parameter
    @staticmethod
    def project() -> "ParameterSpace":
        raise NotImplementedError()
        return ParameterSpace()

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

        try:
            return Point.parse_obj(obj)
        except:
            pass

        try:
            return Box.parse_obj(obj)
        except:
            pass

        raise Exception(f"obj of type {obj['type']}")

    def __repr__(self) -> str:
        return str(self.dict())

    def append_result(self, result: dict):
        inst = ParameterSpace.decode_labeled_object(result)
        label = inst.label
        if isinstance(inst, Box):
            if label == "true":
                self.true_boxes.append(inst)
            elif label == "false":
                self.false_boxes.append(inst)
            else:
                l.info(f"Skipping Box with label: {label}")
        elif isinstance(inst, Point):
            if label == "true":
                self.true_points.append(inst)
            elif label == "false":
                self.false_points.append(inst)
            else:
                l.info(f"Skipping Point with label: {label}")
        else:
            l.error(f"Skipping invalid object type: {type(inst)}")

    def consistent(self) -> bool:
        """
        Check that the parameter space is consistent:

        * All boxes are disjoint

        * All points are in a respective box

        * No point is both true and false
        """
        boxes = self.true_boxes + self.false_boxes
        for i1, b1 in enumerate(boxes):
            for i2, b2 in enumerate(boxes[i1 + 1 :]):
                if b1.intersects(b2):
                    l.exception(f"Parameter Space Boxes intersect: {b1} {b2}")
                    return False
        for tp in self.true_points:
            if not any([b.contains_point(tp) for b in self.true_boxes]):
                return False
        for fp in self.false_points:
            if not any([b.contains_point(fp) for b in self.false_boxes]):
                return False

        if len(set(self.true_points).intersection(set(self.false_points))) > 0:
            return False
        return True

    def _compact(self):
        """
        Compact the boxes by joining boxes that can create a box
        """
        self.true_boxes = self._box_list_compact(self.true_boxes)
        self.false_boxes = self._box_list_compact(self.false_boxes)

    def labeled_volume(self):
        self._compact()
        labeled_vol = 0
        # TODO should actually be able to compact the true and false boxes together, since they are both labeled.
        # TODO can calculate the percentage of the total parameter space.  Is there an efficient way to get the initial PS so we can find the volume of that box? or to access unknown boxes?
        for box in self.true_boxes:
            true_volume = box.volume()
            labeled_vol += true_volume

        for box in self.false_boxes:
            false_volume = box.volume()
            labeled_vol += false_volume
        return labeled_vol

    def max_true_volume(self):
        self.true_boxes = self._box_list_compact(self.true_boxes)
        max_vol = 0
        max_box = (self.true_boxes)[0]
        for box in self.true_boxes:
            box_vol = box.volume()
            if box_vol > max_vol:
                max_vol = box_vol
                max_box = box

        return max_vol, max_box

    def _box_list_compact(self, group: List[Box]) -> List[Box]:
        """
        Attempt to union adjacent boxes and remove duplicate points.
        """
        # Boxes of dimension N can be merged if they are equal in N-1 dimensions and meet in one dimension.
        # Sort the boxes in each dimension by upper bound.
        # Interate through boxes in order wrt. one of the dimensions. For each box, scan the dimensions, counting the number of dimensions that each box meeting in at least one dimension, meets.
        # Merging a dimension where lb(I) = ub(I'), results in an interval I'' = [lb(I), lb(I')].

        if len(group) <= 0:
            return []

        dimensions = group[0].bounds.keys()
        # keep a sorted list of boxes by dimension based upon the upper bound in the dimension
        sorted_dimensions = {p: [b for b in group] for p in dimensions}
        for p, boxes in sorted_dimensions.items():
            boxes.sort(key=lambda x: x.bounds[p].ub)
        dim = next(iter(sorted_dimensions.keys()))
        merged = True
        while merged:
            merged = False
            for b in sorted_dimensions[dim]:
                # candidates for merge are all boxes that meet or are equal in a dimension
                candidates = b._get_merge_candidates(sorted_dimensions)
                # pick first candidate
                if len(candidates) > 0:
                    c = next(iter(candidates))
                    m = b._merge(c)
                    sorted_dimensions = {
                        p: [
                            box if box != b else m for box in boxes if box != c
                        ]
                        for p, boxes in sorted_dimensions.items()
                    }
                    merged = True
                    break

        return sorted_dimensions[dim]
