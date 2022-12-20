import unittest

import matplotlib.pyplot as plt
import numpy as np
from funman_demo.box_plotter import BoxPlotter
from matplotlib.lines import Line2D
from pysmt.shortcuts import (
    FALSE,
    GE,
    GT,
    LE,
    LT,
    TRUE,
    And,
    Equals,
    ForAll,
    Function,
    FunctionType,
    Iff,
    Int,
    Plus,
    Real,
    Symbol,
    get_model,
    simplify,
    substitute,
)
from pysmt.typing import BOOL, INT, REAL

from funman import Funman
from funman.math_utils import lt
from funman.model import EncodedModel, Parameter
from funman.parameter_space import ParameterSpace
from funman.scenario.parameter_synthesis import ParameterSynthesisScenario
from funman.search import BoxSearch, SearchConfig
from funman.search_utils import Box, Interval


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
        dict = {
            Parameter("x", 1, 2),
            Parameter("y", 1, 2),
            Parameter("z", 5, 6),
        }
        dict2 = {
            Parameter("x", 1.5, 2.5),
            Parameter("y", 1.5, 2.5),
            Parameter("z", 1, 2),
        }
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
                    current_param = Parameter(
                        f"{variable_name}",
                        variable_values.lb,
                        variable_values.ub,
                    )
                    param_list.append(current_param)
            param_list = {i for i in param_list}
            box1_result = Box(param_list)
            return box1_result

        def add_box_variable(
            b, vars_list, new_var_name, new_bounds_lb, new_bounds_ub
        ):
            """Takes a subset of selected variables (vars_list) of a given box (b) and returns another box that is given by b's values for only the selected variables."""
            param_list = []
            for i in range(len(list(b.bounds.keys()))):
                variable_name = list(b.bounds.keys())[i].name
                variable_values = list(b.bounds.values())[i]
                if variable_name in vars_list:
                    current_param = Parameter(
                        f"{variable_name}",
                        variable_values.lb,
                        variable_values.ub,
                    )
                    param_list.append(current_param)
            new_param = Parameter(
                f"{new_var_name}", new_bounds_lb, new_bounds_ub
            )
            param_list.append(new_param)
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
                    BoxPlotter.plot2DBoxesTemp(
                        [box1, box2],
                        subset_of_variables[0],
                        subset_of_variables[1],
                        colors=["y", "b"],
                    )
                    custom_lines = [
                        Line2D([0], [0], color="y", alpha=0.2, lw=4),
                        Line2D([0], [0], color="b", alpha=0.2, lw=4),
                    ]
                    plt.legend(custom_lines, ["Box 1", "Box 2"])
                    plt.show()
            desired_vars_list = []
            for b in b1.bounds:
                if b.name != var:
                    desired_vars_list.append(b.name)
            ## Visualize marginalized spaces
            BoxPlotter.plot2DBoxesTemp(
                [b1, b2],
                desired_vars_list[0],
                desired_vars_list[1],
                colors=["y", "b"],
            )
            custom_lines = [
                Line2D([0], [0], color="y", alpha=0.2, lw=4),
                Line2D([0], [0], color="b", alpha=0.2, lw=4),
            ]
            plt.legend(custom_lines, ["Box 1", "Box 2"])
            plt.show()
            ## Find the intersection (if it exists)
            intersection_marginal = (
                Box.intersect_two_boxes_selected_parameters(
                    b1, b2, desired_vars_list
                )
            )
            ## Form versions of boxes minus the part that we're marginalizing: named box1_x_y and box2_x_y
            box1_x_y = subset_of_box_variables(b1, desired_vars_list)
            box2_x_y = subset_of_box_variables(b2, desired_vars_list)
            ## Now form the symmetric difference
            unknown_boxes = [box1_x_y, box2_x_y]
            false_boxes = []
            true_boxes = []
            unknown_boxes_full = [b1, b2]
            false_boxes_full = []
            true_boxes_full = []
            while len(unknown_boxes) > 0:
                b = unknown_boxes.pop()
                b_full = unknown_boxes_full.pop()
                if Box.contains(intersection_marginal, b) == True:
                    false_boxes.append(b)
                    false_boxes_full.append(b_full)
                elif Box.contains(b, intersection_marginal) == True:
                    new_boxes = Box.split(b)
                    for bound in b_full.bounds:
                        if bound.name == var:
                            marg_var = bound.name
                            marg_var_lb = bound.lb
                            marg_var_ub = bound.ub
                    for i in range(len(new_boxes)):
                        unknown_boxes.append(
                            new_boxes[i]
                        )  ## new split boxes: find the marginalization variable (called var) and append it
                        new_box_full = add_box_variable(
                            new_boxes[i],
                            desired_vars_list,
                            var,
                            marg_var_lb,
                            marg_var_ub,
                        )
                        unknown_boxes_full.append(new_box_full)
                else:
                    true_boxes.append(b)
                    true_boxes_full.append(b_full)
            ## Fix true box values and plot
            for b in list(true_boxes):
                b = subset_of_box_variables(b, desired_vars_list)
                BoxPlotter.plot2DBoxTemp(
                    b, desired_vars_list[0], desired_vars_list[1], color="g"
                )
            ## Fix false box values and plot
            for b in list(false_boxes):
                b = subset_of_box_variables(b, desired_vars_list)
                BoxPlotter.plot2DBoxTemp(
                    b, desired_vars_list[0], desired_vars_list[1], color="r"
                )
            custom_lines = [
                Line2D([0], [0], color="r", alpha=0.2, lw=4),
                Line2D([0], [0], color="g", alpha=0.2, lw=4),
            ]
            plt.legend(custom_lines, ["Intersection", "Symmetric Difference"])
            plt.show()
            non_intersecting_boxes = list(true_boxes_full)
            intersecting_boxes = list(false_boxes_full)
            result = []  ## List of boxes to be returned.
            if len(intersecting_boxes) > 1:
                for bound in intersecting_boxes[0].bounds:
                    if bound.name == var:
                        marg_var_0 = bound.name
                        marg_var_lb_0 = bound.lb
                        marg_var_ub_0 = bound.ub
                        bounds_0 = Interval.make_interval(
                            [marg_var_lb_0, marg_var_ub_0]
                        )
                for bound in intersecting_boxes[1].bounds:
                    if bound.name == var:
                        marg_var_1 = bound.name
                        marg_var_lb_1 = bound.lb
                        marg_var_ub_1 = bound.ub
                        bounds_1 = Interval.make_interval(
                            [marg_var_lb_1, marg_var_ub_1]
                        )
                interval_union_result = Interval.union(bounds_0, bounds_1)[0]
                if (
                    len(interval_union_result) == 1
                ):  ## intersection along all variables (including marginal): form the union.
                    ## Make new result with last bound given by the above interval
                    box_subset = subset_of_box_variables(
                        intersecting_boxes[0], desired_vars_list
                    )
                    box_result_union = add_box_variable(
                        box_subset,
                        desired_vars_list,
                        var,
                        interval_union_result[0].lb,
                        interval_union_result[0].ub,
                    )
                    result.append(box_result_union)
                    for box in non_intersecting_boxes:
                        result.append(box)
                else:
                    for box in non_intersecting_boxes:
                        result.append(box)
                    for box in intersecting_boxes:
                        result.append(box)
            return result  ## result is the list of boxes where, for the non-marginalized terms, boxes are either disjoint or equal and for the marginalized terms, the union over the marginalized variable has been taken.

        ## Test: marginalize the boxes box1 and box2 based on the variable z.
        print(marginalize(box1, box2, "z"))


if __name__ == "__main__":
    unittest.main()
