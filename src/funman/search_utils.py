import ctypes
from curses.ascii import EM
from functools import total_ordering
from multiprocessing import Array, Queue, Value, cpu_count
import os
import time
from typing import Dict, List, Union
from funman.config import Config
from funman.model import Parameter
from funman.constants import NEG_INFINITY, POS_INFINITY, BIG_NUMBER
from numpy import average
from pysmt.shortcuts import Real, GE, LT, And, TRUE, Equals
import funman.math_utils as math_utils


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
            else self.lb < other.lb ## DMI change < to <=
        )
        rhs = (
            (other.ub != POS_INFINITY or self.ub == POS_INFINITY)
            if (self.ub == POS_INFINITY or other.ub == POS_INFINITY)
            else other.ub < self.ub ## DMI change < to <=
        )
        return lhs or rhs
    
        

    def midpoint(self):
        if self.lb == NEG_INFINITY and self.ub == POS_INFINITY:
            return 0
        elif self.lb == NEG_INFINITY:
            return self.ub - BIG_NUMBER
        if self.ub == POS_INFINITY:
            return self.lb + BIG_NUMBER
        else:
            return ((self.ub - self.lb) / 2) + self.lb
    
    def union(self, other: "Interval") -> List["Interval"]:
        if self == other: ## intervals are equal, so return original interval
            ans = [self]
            total_height = [Interval.width(self)]
            return ans, total_height
        else: ## intervals are not the same. start by identifying the lower and higher intervals.
            if self.lb == other.lb:
                if math_utils.lt(self.ub, other.lb):
                    minInterval = self 
                    maxInterval = other
                else: ## other.ub > self.ub
                    minInterval = other 
                    maxInterval = self
            elif math_utils.lt(self.lb, other.lb):
                minInterval = self
                maxInterval = other
            else:
                minInterval = other
                maxInterval = self
        if math_utils.gte(minInterval.ub,maxInterval.lb): ## intervals intersect. 
            ans = Interval.make_interval([minInterval.lb, maxInterval.ub])
            total_height = Interval.width(ans) 
            return ans, total_height
        elif math_utils.lt(minInterval.ub, maxInterval.lb): ## intervals are disjoint.
            ans = [minInterval, maxInterval]
            total_height = [math_utils.plus(Interval.width(minInterval), Interval.width(maxInterval))]
            return ans, total_height
            
    def subtract_two_1d_intervals(a: "Interval", b: "Interval") -> "Interval": ## TODO Drisana - fix
        """Given 2 intervals a = [a0,a1] and b=[b0,b1], return the part of a that does not intersect with b."""
        if math_utils.gte(a.lb, b.lb):
            if math_utils.lte(a.ub, b.ub): ## a is a subset of b: return None 
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

    def to_smt(self, p: Parameter):
        return And(
            (GE(p.symbol, Real(self.lb)) if self.lb != NEG_INFINITY else TRUE()),
            (LT(p.symbol, Real(self.ub)) if self.ub != POS_INFINITY else TRUE()),
        ).simplify()


@total_ordering
class Box(object):
    def __init__(self, parameters) -> None:
        self.bounds: Dict[Parameter, Interval] = {
            p: Interval(p.lb, p.ub) for p in parameters
        }
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
        return self.__repr__()

    def finite(self) -> bool:
        return all([i.finite() for _, i in self.bounds.items()])

    def contains(self, other: "Box") -> bool:
        return all(
            [interval.contains(other.bounds[p]) for p, interval in self.bounds.items()]
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
        b = Box([
            Parameter('foo', lb=b0_bounds.lb, ub=b0_bounds.ub)
        ])
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
        if beta_0 != []: ## boxes intersect on the given axis
            bounds = beta_0
            print("todo: also calculate heights for the non-intersected parts.")
        else: ## boxes do not intersect on the given axis: just return original boxes and their heights
            bounds1 = beta_0_a
            bounds2 = beta_0_b
            height1 = math_utils.minus(beta_1_a.ub, beta_1_a.lb)
            height2 = math_utils.minus(beta_1_b.ub, beta_1_b.lb)
            print(height1, height2)
        beta_1 = Box.intersection(beta_1_a, beta_1_b)
        if beta_1 == []: ## no intersection
            print(beta_1_a.lb)
            print(beta_1_a.ub)
            print(beta_1_b.lb)
            print(beta_1_b.ub)
            height1 = math_utils.minus(beta_1_a.ub, beta_1_a.lb)
            height2 = math_utils.minus(beta_1_b.ub, beta_1_b.lb)
            total_height = math_utils.plus(height1,height2)
        else: ## boxes intersect along y-axis
            print("todo")
    
    def split_bounds(a: "Box", b: "Box"):
        interval_list = []
        height_list = []
        ybounds_list = []
        a_params = list(a.bounds.keys())
        b_params = list(b.bounds.keys())
        beta_0_a = a.bounds[a_params[0]] # first box, first parameter
        beta_1_a = a.bounds[a_params[1]] # first box, second parameter
        beta_0_b = b.bounds[b_params[0]] # second box, first parameter
        beta_1_b = b.bounds[b_params[1]] # second box, second parameter
        beta_0_intersection = Box.intersection(beta_0_a, beta_0_b) # intersection of the first parameter for first box and second box
        beta_0_a_new = Interval.subtract_two_1d_intervals(beta_0_a, Interval.make_interval(beta_0_intersection))
        beta_0_b_new = Interval.subtract_two_1d_intervals(beta_0_b, Interval.make_interval(beta_0_intersection))
        beta_0_a_new_2 = Interval.subtract_two_1d_intervals(Interval.make_interval(beta_0_intersection), beta_0_a)
        beta_0_b_new_2 = Interval.subtract_two_1d_intervals(Interval.make_interval(beta_0_intersection), beta_0_b)
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
        beta_0_a = a.bounds[a_params[0]] # first box, first parameter
        beta_1_a = a.bounds[a_params[1]] # first box, second parameter
        beta_0_b = b.bounds[b_params[0]] # second box, first parameter
        beta_1_b = b.bounds[b_params[1]] # second box, second parameter
        if beta_0_a == beta_0_b: ## bounds are equal: done
            print("done: beta_0 bounds are equal")
            return True
        else:
            beta_0_intersection = Box.intersection(beta_0_a, beta_0_b) # intersection of the first parameter for first box and second box
            if beta_0_intersection == []: ## disjoint: done
                print("done: beta_0 bounds are disjoint")
                return True
            else: ## there is both some intersection and some symmetric difference. split accordingly.
                return False
                
    def check_bounds_disjoint_equal(a: "Box", b: "Box"):
        ### specify first parameter (the one to check whether there is an intersection on)
        a_params = list(a.bounds.keys())
        b_params = list(b.bounds.keys())
        beta_0_a = a.bounds[a_params[0]] # first box, first parameter
        beta_1_a = a.bounds[a_params[1]] # first box, second parameter
        beta_0_b = b.bounds[b_params[0]] # second box, first parameter
        beta_1_b = b.bounds[b_params[1]] # second box, second parameter
#        print('beta_0_a:',beta_0_a)
#        print('beta_0_b:',beta_0_b)
#        print('beta_1_a:',beta_1_a)
#        print('beta_1_b:',beta_1_b)
        if beta_0_a == beta_0_b: ## bounds are equal: done
            print("done: beta_0 bounds are equal")
            ## calculate width of union
            beta_1_union = Interval.union(beta_1_a, beta_1_b)
            beta_1_union_interval = beta_1_union[0]
            beta_1_union_height = beta_1_union[1]
            return True, [beta_0_a], beta_1_union_interval, beta_1_union_height
        else:
            beta_0_intersection = Box.intersection(beta_0_a, beta_0_b) # intersection of the first parameter for first box and second box
            if beta_0_intersection == []: ## disjoint: done
                print("done: beta_0 bounds are disjoint")
                return True, [beta_0_a, beta_0_b], [beta_1_a, beta_1_b], [Interval.width(beta_1_a), Interval.width(beta_1_b)]
            else: ## there is both some intersection and some symmetric difference. split accordingly.
#                print("in progress: splitting")
                return False, Box.split_bounds(a,b) #, Interval.union(beta_1_a, beta_1_b)
                
                
            
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

    def intersection(a,b):
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
            if math_utils.gt(minArray.ub,maxArray.lb): ## has nonempty intersection. return intersection
                return [float(maxArray.lb), float(minArray.ub)]
            else: ## no intersection.
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

    def intersect_two_boxes(b1,b2):
        a = list(b1.bounds.values())
        b = list(b2.bounds.values())
        result = []
        d = len(a) ## dimension
        for i in range(d):
            subresult = Box.intersection(a[i],b[i])
            if subresult == []:
                return None
            else:
                result.append(subresult)
        return result

    def subtract_two_1d_boxes(a,b):
        """Given 2 intervals a = [a0,a1] and b=[b0,b1], return the part of a that does not intersect with b."""
        if intersect_two_1d_boxes(a,b) == None:
            return a
        else:
            if a[0] < b[0]:
                return [a[0],b[0]]
            elif a[0] > b[0]:
                return [b[1],a[1]]
    
    def symmetric_difference_two_boxes(a,b): ### WIP - just for 2 dimensions at this point.
        result = []
        if a == b:
            result = None 
        elif Box.intersect_two_boxes(a,b) == None: ## no intersection so they are disjoint - return both original boxes
            result = [a,b]
        else:
            xbounds = Box.subtract_two_1d_boxes(a[0],b[0])
            if xbounds != None:
                result.append([xbounds,a[1]])
            xbounds = Box.subtract_two_1d_boxes(b[0],a[0])
            if xbounds != None:
                result.append([xbounds,b[1]])
            ybounds = Box.subtract_two_1d_boxes(a[1],b[1])
            if ybounds != None:
                result.append([a[0],ybounds]) 
            ybounds = Box.subtract_two_1d_boxes(b[1],a[1])
            if ybounds != None:
                result.append([b[0],ybounds])         
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
        self.tolerance = kwargs["tolerance"] if "tolerance" in kwargs else 1e-1
        self.queue_timeout = kwargs["queue_timeout"] if "queue_timeout" in kwargs else 1
        self.number_of_processes = (
            kwargs["number_of_processes"]
            if "number_of_processes" in kwargs
            else cpu_count()
        )
