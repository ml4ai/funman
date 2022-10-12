from typing import List
from funman.plotting import BoxPlotter
from funman.search_episode import SearchEpisode
from funman.search_utils import Box


class ParameterSpace(object):
    def __init__(self, true_boxes: List[Box], false_boxes: List[Box]) -> None:
        self.true_boxes = true_boxes
        self.false_boxes = false_boxes

    # STUB project parameter space onto a parameter
    @staticmethod
    def project() -> "ParameterSpace":
        raise NotImplementedError()
        return ParameterSpace()

#    def intersect(ps1, ps2) -> "ParameterSpace":
#        # FIXME Drisana
#        raise NotImplementedError()
#        return ParameterSpace()

    # STUB intersect parameters spaces
    @staticmethod
    def intersect(ps1, ps2) -> "ParameterSpace":
        results_list = []
        for box1 in ps1.bounds.values():
            for box2 in ps2.bounds.values():
                subresult = intersect_two_boxes(box1, box2)
                if subresult != None:
                    results_list.append(subresult)
        return results_list

    @staticmethod
    def intersect_two_1d_boxes(a,b):
        """Given 2 intervals a = [a0,a1] and b=[b0,b1], check whether they intersect.  If they do, return their intersection."""
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

    @staticmethod
    def intersect_two_boxes(a,b):
        result = []
        d = len(a) ## dimension
        for i in range(d):
            subresult = intersect_two_1d_boxes(a[i],b[i])
            if subresult == []:
                return None
            else:
                result.append(subresult)
        return result

    # STUB symmetric_difference of parameter spaces
    @staticmethod
    def symmetric_difference(ps1, ps2) -> "ParameterSpace":
        results_list = []
        for box2 in ps2:
            box2_results = []
            for box1 in ps1:
                subresult = symmetric_difference_two_boxes(box2, box1)
                if subresult != None:
                    box2_results.append(subresult[0])
                elif subresult == None:
                    box2_results.append(subresult)
    #         print(box1_results)
            if None in box2_results:
                pass
            else:
                results_list.append(box2_results[0])
        for box1 in ps1:
            box1_results = []
            for box2 in ps2:
                subresult = symmetric_difference_two_boxes(box1, box2)
                if subresult != None:
                    box1_results.append(subresult[0])
                elif subresult == None:
                    box1_results.append(subresult)
            if None in box1_results:
                pass
            else:
                results_list.append(box1_results[0])
        return results_list

    @staticmethod
    def symmetric_difference_two_1d_boxes(a,b):
        """Given 2 intervals a = [a0,a1] and b=[b0,b1], return their symmetric difference (points not in their intersection)."""
        if a == b: ## no symmetric difference
            return None
        else:
            if a[0] <= b[0]:
                minArray = a
                maxArray = b
            else:
                minArray = b
                maxArray = a
            if minArray[1] > maxArray[0]: ## has nonempty intersection. form symmetric differences and return them
                return [[minArray[0], maxArray[0]],[minArray[1], maxArray[1]]]
            else: ## no intersection. symmetric difference is the entirety of the 2 arrays
                return [minArray, maxArray]

    @staticmethod
    def subtract_two_1d_boxes(a,b):
        """Given 2 intervals a = [a0,a1] and b=[b0,b1], return the part of a that does not intersect with b."""
        if intersect_two_1d_boxes(a,b) == None:
            return a
        else:
            if a[0] < b[0]:
                return [a[0],b[0]]
            elif a[0] > b[0]:
                return [b[1],a[1]]
    
    @staticmethod
    def symmetric_difference_two_boxes(a,b): ### WIP - just for 2 dimensions at this point.
        result = []
        if a == b:
            result = None 
        elif intersect_two_boxes(a,b) == None: ## no intersection so they are disjoint - return both original boxes
            result = [a,b]
        else:
            xbounds = subtract_two_1d_boxes(a[0],b[0])
            if xbounds != None:
                result.append([xbounds,a[1]])
            xbounds = subtract_two_1d_boxes(b[0],a[0])
            if xbounds != None:
                result.append([xbounds,b[1]])
            ybounds = subtract_two_1d_boxes(a[1],b[1])
            if ybounds != None:
                result.append([a[0],ybounds]) 
            ybounds = subtract_two_1d_boxes(b[1],a[1])
            if ybounds != None:
                result.append([b[0],ybounds])         
        return result
            
    
    # STUB construct space where all parameters are equal
    @staticmethod
    def construct_all_equal(ps) -> "ParameterSpace":
        raise NotImplementedError()
        return ParameterSpace()

    # STUB compare parameter spaces for equality
    @staticmethod
    def compare(ps1, ps2) -> bool:
        raise NotImplementedError()
        raise NotImplementedError()

    def plot():
        # FIXME Drisana
        box_plotter = BoxPlotter([], [])
        box_plotter.plot_list_of_boxes()
        pass
