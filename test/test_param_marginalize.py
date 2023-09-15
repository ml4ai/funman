import unittest

from matplotlib.lines import Line2D

from funman.representation import ModelParameter
from funman.search import Box, Interval


def add_box_variable(b, vars_list, new_var_name, new_bounds_lb, new_bounds_ub):
    """Takes a subset of selected variables (vars_list) of a given box (b) and returns another box that is given by b's values for only the selected variables."""
    param_list = []
    for i in range(len(list(b.bounds.keys()))):
        variable_name = list(b.bounds.keys())[i]
        variable_values = list(b.bounds.values())[i]
        if variable_name in vars_list:
            current_param = ModelParameter(
                name=f"{variable_name}",
                lb=variable_values.lb,
                ub=variable_values.ub,
            )
            param_list.append(current_param)
    new_param = ModelParameter(
        name=f"{new_var_name}", lb=new_bounds_lb, ub=new_bounds_ub
    )
    param_list.append(new_param)
    param_list = {i for i in param_list}
    box1_result = Box(
        bounds={p.name: Interval(lb=p.lb, ub=p.ub) for p in param_list}
    )
    return box1_result


def marginalize(b1, b2, var):
    ## First check that the two boxes have the same variables
    vars_b1 = set([b for b in b1.bounds])
    vars_b2 = set([b for b in b2.bounds])
    if vars_b1 == vars_b2 and var in vars_b1:
        vars_list = list(vars_b1)
        print("marginalization in progress")
    else:
        print(
            "cannot marginalize: variables are not the same or marginalized variable is not in list."
        )
        raise Exception("Cannot Marginalize")
    ## Get all combinations of 2 variables and plot them
    for i1 in range(len(vars_list)):
        for i2 in range(i1 + 1, len(vars_list)):
            subset_of_variables = [vars_list[i1], vars_list[i2]]
            # BoxPlotter.plot2DBoxesTemp([box1, box2],subset_of_variables[0],subset_of_variables[1],colors=['y','b'])
            # custom_lines = [Line2D([0], [0], color='y',alpha=0.2, lw=4),Line2D([0], [0], color='b',alpha=0.2, lw=4)]
            # plt.legend(custom_lines, ['PS 1', 'PS 2'])
            # plt.show()
    desired_vars_list = []
    for b in b1.bounds:
        if b != var:
            desired_vars_list.append(b)
    ### Visualize marginalized spaces
    # BoxPlotter.plot2DBoxesTemp([b1, b2],desired_vars_list[0],desired_vars_list[1],colors=['y','b'])
    # custom_lines = [Line2D([0], [0], color='y',alpha=0.2, lw=4),Line2D([0], [0], color='b',alpha=0.2, lw=4)]
    # plt.legend(custom_lines, ['PS 1', 'PS 2'])
    # plt.show()
    ## Find the intersection (if it exists)
    intersection_marginal = b1.intersect(b2, param_list=desired_vars_list)
    ## Form versions of boxes minus the part that we're marginalizing: named box1_x_y and box2_x_y
    box1_x_y = b1.project(desired_vars_list)
    box2_x_y = b2.project(desired_vars_list)
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
                if bound == var:
                    marg_var = bound
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
        b = b.project(desired_vars_list)
        # BoxPlotter.plot2DBoxTemp(b,desired_vars_list[0],desired_vars_list[1],color='g')
    ## Fix false box values and plot
    for b in list(false_boxes):
        b = b.project(desired_vars_list)
        # BoxPlotter.plot2DBoxTemp(b,desired_vars_list[0],desired_vars_list[1],color='r')
    custom_lines = [
        Line2D([0], [0], color="r", alpha=0.2, lw=4),
        Line2D([0], [0], color="g", alpha=0.2, lw=4),
    ]
    # plt.legend(custom_lines, ['Intersection', 'Symmetric Difference'])
    # plt.show()
    non_intersecting_boxes = list(true_boxes_full)
    intersecting_boxes = list(false_boxes_full)
    result = []  ## List of boxes to be returned.
    if len(intersecting_boxes) > 1:
        for bound in intersecting_boxes[0].bounds:
            if bound == var:
                marg_var_0 = bound
                marg_var_lb_0 = bound.lb
                marg_var_ub_0 = bound.ub
                bounds_0 = Interval(marg_var_lb_0, marg_var_ub_0)

        for bound in intersecting_boxes[1].bounds:
            if bound == var:
                marg_var_1 = bound
                marg_var_lb_1 = bound.lb
                marg_var_ub_1 = bound.ub
                bounds_1 = Interval(marg_var_lb_1, marg_var_ub_1)

        interval_union_result = Interval.union(bounds_0, bounds_1)
        print(interval_union_result)
        # interval_union_height = Interval.union(bounds_0, bounds_1)[1]
        # print(interval_union_height)
        if (
            len(interval_union_result) == 1
        ):  ## intersection along all variables (including marginal): form the union.
            ## Make new result with last bound given by the above interval
            box_subset = intersecting_boxes[0].project(desired_vars_list)
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
        for i in range(len(result)):
            print(result[i])
    return result  ## result is the list of boxes where, for the non-marginalized terms, boxes are either disjoint or equal and for the marginalized terms, the union over the marginalized variable has been taken.


@unittest.skip(
    reason="marginalize is broken and uses two boxes instead of one"
)
class TestCompilation(unittest.TestCase):
    ######## begin demos
    def test_marginalize_exception_1(self):
        ## Test: marginalize the boxes box1 and box2 based on the variable w.  This should not work (should raise an exception), since w is not a variable in the set.
        # Manually make boxes for example.
        dict = {
            "x": Interval(lb=1, ub=2),
            "y": Interval(lb=1, ub=2),
            "z": Interval(lb=1, ub=2),
        }
        dict2 = {
            "x": Interval(lb=1.5, ub=2.5),
            "y": Interval(lb=1.5, ub=2.5),
            "z": Interval(lb=1.5, ub=2.5),
        }
        box1 = Box(bounds=dict)
        box2 = Box(bounds=dict2)
        self.assertRaises(Exception, marginalize, box1, box2, "w")

    def test_marginalize_exception_2(self):
        ## Test: marginalize the boxes box1 and box2 based on the variable x.  This should not work (should raise an exception), since the boxes are not made up of the same variables (despite both containing x).
        # Manually make boxes for example.
        dict = {
            "x": Interval(lb=1, ub=2),
            "y": Interval(lb=1, ub=2),
            "z": Interval(lb=1, ub=2),
        }
        dict2 = {
            "x": Interval(lb=1.5, ub=2.5),
            "y": Interval(lb=1.5, ub=2.5),
            "z": Interval(lb=1.5, ub=2.5),
        }
        box1 = Box(bounds=dict)
        box2 = Box(bounds=dict2)
        self.assertRaises(Exception, marginalize, box1, box2, "x")

    def test_marginalize_1(self):
        ## Test: marginalize the boxes box1 and box2 based on the variable z.
        # Manually make boxes for example.
        dict = {
            "x": Interval(lb=1, ub=2),
            "y": Interval(lb=1, ub=2),
            "z": Interval(lb=1, ub=2),
        }
        dict2 = {
            "x": Interval(lb=1.5, ub=2.5),
            "y": Interval(lb=1.5, ub=2.5),
            "z": Interval(lb=5, ub=6),
        }
        box1 = Box(bounds=dict)
        box2 = Box(bounds=dict2)
        result = marginalize(box1, box2, "z")
        assert result


if __name__ == "__main__":
    unittest.main()
