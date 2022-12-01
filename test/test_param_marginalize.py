import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from funman.search import BoxSearch, SearchConfig
from funman.constants import NEG_INFINITY, POS_INFINITY
from pysmt.shortcuts import (
    get_model,
    And,
    Symbol,
    FunctionType,
    Function,
    Equals,
    Int,
    Real,
    substitute,
    TRUE,
    FALSE,
    Iff,
    Plus,
    ForAll,
    LT,
    simplify,
    GT,
    LE,
    GE,
)
from pysmt.typing import INT, REAL, BOOL
import unittest
import os
from funman import Funman
from funman.model import Parameter, EncodedModel
from funman.scenario.parameter_synthesis import ParameterSynthesisScenario
from funman.math_utils import lt
from funman.search_utils import Interval, Box
from funman.parameter_space import ParameterSpace
from funman_demo.box_plotter import BoxPlotter


class TestCompilation(unittest.TestCase):
    #    def test_toy(self):
    #
    #        x = Symbol("x", REAL)
    #        parameters = [Parameter("x", x)]
    #
    #        # 0.0 <= x <= 5
    #        model = Model(And(LE(x, Real(5.0)), GE(x, Real(0.0))))
    #
    #        scenario = ParameterSynthesisScenario(parameters, model, BoxSearch())
    #        funman = Funman()
    #        config = SearchConfig()
    #        result = funman.solve(scenario, config)
    #
    def test_toy_2d(self):

        x = Symbol("x", REAL)
        y = Symbol("y", REAL)
        parameters = [Parameter("x", x), Parameter("y", y)]

        # 0.0 < x < 5.0, 10.0 < y < 12.0
        model = EncodedModel(
            And(
                LE(x, Real(5.0)),
                GE(x, Real(0.0)),
                LE(y, Real(12.0)),
                GE(y, Real(10.0)),
            )
        )

        scenario = ParameterSynthesisScenario(parameters, model, BoxSearch())
        funman = Funman()
        config = SearchConfig(tolerance=1e-1)
        result = funman.solve(scenario, config=config)
        ######## begin interval demos
        #        a1 = Interval.make_interval([0,2])
        #        a2 = Interval.make_interval([1,3])
        #        print(Interval.midpoint(a1))
        #        print(Interval.union(a1,a2))
        ####### begin box demos
        #        for b1 in result.parameter_space.true_boxes:
        #            for b2 in result.parameter_space.false_boxes:
        #                print(Box.intersect_two_boxes(b1,b2))
        #        print(ParameterSpace._intersect_boxes(result.parameter_space.true_boxes, result.parameter_space.false_boxes))
        #        print(ParameterSpace._symmetric_difference(result.parameter_space.true_boxes, result.parameter_space.false_boxes))
        #        for b1 in result.parameter_space.true_boxes:
        #            for b2 in result.parameter_space.false_boxes:
        #                print(Box.union(b1,b2))
        #        print(ParameterSpace._union_boxes(boxes_of_interest))
        boxes_of_interest = result.parameter_space.true_boxes
        for b1 in boxes_of_interest:
            print(Box.union)
        for i1 in range(len(boxes_of_interest)):
            for i2 in range(i1 + 1, len(boxes_of_interest)):
                ans = Box.check_bounds_disjoint_equal(
                    boxes_of_interest[i1], boxes_of_interest[i2]
                )
                if ans[0] == False:
                    print(ans)
                    ## Plot original boxes
                    BoxPlotter.plot2DBoxes_temp(
                        [boxes_of_interest[i1], boxes_of_interest[i2]]
                    )
                    ## Plot heights
                    bounds = ans[1]
                    x_bounds = bounds[0]
                    heights = bounds[1]
                    BoxPlotter.plot2DBoxesByHeight_temp(x_bounds, heights)


#                    for i in range(len(heights)):
#                        BoxPlotter.plot2DBoxByHeight_temp(x_bounds[i],heights[i])
#        assert result

######## begin 3d demos
    def test_toy_3d(self):
        ## Manually make boxes for example.
        dict = {Parameter("x",1,2), Parameter("y",1,2), Parameter("z",5,6)}
        dict2 = {Parameter("x",1.5,2.5), Parameter("y",1.5,2.5), Parameter("z",1,2)}
        box1 = Box(dict)
        box2 = Box(dict2)
#### plotting example (subplots): the graphs show up correctly but can't seem to show all axis labels. 
#        #plt.figure(1)
#        fig, (ax1, ax2, ax3) = plt.subplots(3)
#        ax1 = plt.subplot(3,1,1)
#        ax1.set(xlabel='x',ylabel='y')
#        BoxPlotter.plot2DBoxTemp(box1,'x','y',color='yellow')
#        BoxPlotter.plot2DBoxTemp(box2,'x','y')
#        ax2 = plt.subplot(3,1,2)
#        ax2.set(xlabel='x',ylabel='z')
#        BoxPlotter.plot2DBoxTemp(box1,'x','z',color='yellow')
#        BoxPlotter.plot2DBoxTemp(box2,'x','z')
#        ax3 = plt.subplot(3,1,3)
#        ax3.set(xlabel='y',ylabel='z')
#        BoxPlotter.plot2DBoxTemp(box1,'y','z',color='yellow')
#        BoxPlotter.plot2DBoxTemp(box2,'y','z')
#        plt.suptitle('Original boxes')
#        plt.show()
#### 
        def subset_of_box_variables(b, vars_list):
            """Takes a subset of selected variables (vars_list) of a given box (b) and returns another box that is given by b's values for only the selected variables."""       
            param_list = []
            for i in range(len(list(b.bounds.keys()))):
                variable_name = list(b.bounds.keys())[i].name
                variable_values = list(b.bounds.values())[i]
                if variable_name in vars_list:
                    current_param = Parameter(f"{variable_name}", variable_values.lb, variable_values.ub)
                    param_list.append(current_param)
            param_list = {i for i in param_list}
            box1_result = Box(param_list)
            return box1_result
 
        def marginalize(b1, b2, var):
            ## First check that the two boxes have the same variables
            vars_b1 = set([b.name for b in b1.bounds])
            vars_b2 = set([b.name for b in b2.bounds])
            if vars_b1 == vars_b2:
                vars_list = list(vars_b1) 
                print("marginalization in progress")
            else:
                print("cannot marginalize: variables are not the same.")
                return 0 
            ## Get all combinations of 2 variables and plot them
            for i1 in range(len(vars_list)):
                for i2 in range(i1 + 1, len(vars_list)):
                    subset_of_variables = [vars_list[i1], vars_list[i2]]
                    BoxPlotter.plot2DBoxesTemp([box1, box2],subset_of_variables[0],subset_of_variables[1],colors=['y','b'])
                    custom_lines = [Line2D([0], [0], color='y',alpha=0.2, lw=4),Line2D([0], [0], color='b',alpha=0.2, lw=4)]
                    plt.legend(custom_lines, ['Box 1', 'Box 2'])
                    plt.show()  
            desired_vars_list = []
            for b in b1.bounds:
                if b.name != var:
                    desired_vars_list.append(b.name)
#                    print(b.name, b.lb, b.ub)
            ## Visualize marginalized spaces
            BoxPlotter.plot2DBoxesTemp([b1, b2],desired_vars_list[0],desired_vars_list[1],colors=['y','b'])    
            custom_lines = [Line2D([0], [0], color='y',alpha=0.2, lw=4),Line2D([0], [0], color='b',alpha=0.2, lw=4)]
            plt.legend(custom_lines, ['Box 1', 'Box 2'])
            plt.show() 
            ## Find the intersection (if it exists)
            intersection_marginal = Box.intersect_two_boxes_selected_parameters(b1,b2,desired_vars_list)
            print('intersection test:')
            ## Form versions of boxes minus the part that we're marginalizing: named box1_x_y and box2_x_y
            box1_x_y = subset_of_box_variables(b1,desired_vars_list)
            box2_x_y = subset_of_box_variables(b2,desired_vars_list)
            ## Now form the symmetric difference
            unknown_boxes = [box1_x_y, box2_x_y]
            false_boxes = []
            true_boxes = []
            while len(unknown_boxes) > 0:
                b = unknown_boxes.pop()
                if Box.contains(intersection_marginal, b) == True:
                    false_boxes.append(b)
                elif Box.contains(b, intersection_marginal) == True:
                    new_boxes = Box.split(b)
                    for i in range(len(new_boxes)):
                        unknown_boxes.append(new_boxes[i])
                else:
                    true_boxes.append(b)
            for b in list(true_boxes):
                param_list = []
                for i in range(len(list(b.bounds.keys()))):
                    variable_name = list(b.bounds.keys())[i].name
                    variable_values = list(b.bounds.values())[i]
#                    print('name:', variable_name, 'values:', variable_values)
                    current_param = Parameter(f"{variable_name}", variable_values.lb, variable_values.ub)
                    param_list.append(current_param)
                param_list = {i for i in param_list}
                b = Box(param_list)
                BoxPlotter.plot2DBoxTemp(b,desired_vars_list[0],desired_vars_list[1],color='g') 
            for b in list(false_boxes):
                param_list = []
                for i in range(len(list(b.bounds.keys()))):
                    variable_name = list(b.bounds.keys())[i].name
                    variable_values = list(b.bounds.values())[i]
                    current_param = Parameter(f"{variable_name}", variable_values.lb, variable_values.ub)
                    param_list.append(current_param)
                param_list = {i for i in param_list}
                b = Box(param_list)
                BoxPlotter.plot2DBoxTemp(b,desired_vars_list[0],desired_vars_list[1],color='r')
            custom_lines = [Line2D([0], [0], color='r',alpha=0.2, lw=4),Line2D([0], [0], color='g',alpha=0.2, lw=4)]
            plt.legend(custom_lines, ['Intersection', 'Symmetric Difference'])
            plt.show() 
        marginalize(box1, box2, 'z')  

if __name__ == "__main__":
    unittest.main()
