from typing import List
from funman.plotting import BoxPlotter
from funman.search_episode import SearchEpisode
from funman.search_utils import Box
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

class ParameterSpace(object):
    def __init__(self, true_boxes: List[Box], false_boxes: List[Box]) -> None:
        self.true_boxes = true_boxes
        self.false_boxes = false_boxes

    # STUB project parameter space onto a parameter
    @staticmethod
    def project() -> "ParameterSpace":
        raise NotImplementedError()
        return ParameterSpace()

#    # STUB intersect parameters spaces
    @staticmethod
    def intersect(ps1, ps2):
        results_list = []
        for box1 in ps1:
            for box2 in ps2:
                subresult = Box.intersect_two_boxes(box1, box2)
                if subresult != None:
                    results_list.append(subresult)
        return results_list

    @staticmethod
    def symmetric_difference(ps1, ps2):
        results_list = []
        for box2 in ps2:
            box2_results = []
            for box1 in ps1:
                subresult = Box.symmetric_difference_two_boxes(box2, box1)
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
                subresult = Box.symmetric_difference_two_boxes(box1, box2)
                if subresult != None:
                    box1_results.append(subresult[0])
                elif subresult == None:
                    box1_results.append(subresult)
            if None in box1_results:
                pass
            else:
                results_list.append(box1_results[0])
        return results_list
#
#    @staticmethod
#    def symmetric_difference_two_1d_boxes(a,b):
#        """Given 2 intervals a = [a0,a1] and b=[b0,b1], return their symmetric difference (points not in their intersection)."""
#        if a == b: ## no symmetric difference
#            return None
#        else:
#            if a[0] <= b[0]:
#                minArray = a
#                maxArray = b
#            else:
#                minArray = b
#                maxArray = a
#            if minArray[1] > maxArray[0]: ## has nonempty intersection. form symmetric differences and return them
#                return [[minArray[0], maxArray[0]],[minArray[1], maxArray[1]]]
#            else: ## no intersection. symmetric difference is the entirety of the 2 arrays
#                return [minArray, maxArray]
#
#            
    
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

    @staticmethod
    def plot(ps1, color="b",alpha=0.2):
        custom_lines = [
            Line2D([0], [0], color=color, lw=4,alpha=alpha),
        ]
        
        plt.legend(custom_lines, ["ps1"])
        for b1 in ps1:
            BoxPlotter.plot2DBoxList(b1, color='b')         
        plt.show() 
        pass
