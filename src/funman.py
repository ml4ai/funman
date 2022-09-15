from copy import deepcopy
from pysmt.shortcuts import get_model, And, LT, LE, GE, TRUE, Not, Real
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

POS_INFINITY = "inf"
NEG_INFINITY = "-inf"
BIG_NUMBER = 1e6

class AnalysisScenario(object):
    pass

class ParameterSynthesisScenario(AnalysisScenario):
    def __init__(self, parameters, model) -> None:
        super().__init__()
        self.parameters = parameters
        self.model = model
    
class Box(object):
    def __init__(self, parameters) -> None:
        self.bounds = { p: [NEG_INFINITY, POS_INFINITY] for p in parameters}
    
    def to_smt(self):
        return And(
            [
                And(
                    (GE(p.symbol, Real(b[0])) if b[0] != NEG_INFINITY else TRUE()), 
                    (LT(p.symbol, Real(b[1])) if b[1] != POS_INFINITY else TRUE())
                    ).simplify()
                for p, b in self.bounds.items()
            ]
        )

    def copy(self):
        c = Box(list(self.bounds.keys()))
        for p, b in self.bounds.items():
            c.bounds[p] = b
        return c
    
    def parameter_width(self, bound):
        [lb, ub] = bound
        if lb == NEG_INFINITY or ub == POS_INFINITY:
            return BIG_NUMBER
        else:
            return ub - lb

    def interval_split(self, interval):
        [lb, ub] = interval
        if lb == NEG_INFINITY and ub == POS_INFINITY:
            return 0
        elif lb == NEG_INFINITY:
            return ub - BIG_NUMBER
        if ub == POS_INFINITY:
            return lb + BIG_NUMBER
        else:
            return ((ub - lb)/2) + lb

    def get_max_width_parameter(self):
        widths = [self.parameter_width(bounds) for p, bounds in self.bounds.items()]
        param = list(self.bounds.keys())[widths.index(max(widths))]
        return param

    def split(self):
        p = self.get_max_width_parameter()
        b1 = self.copy()
        b2 = self.copy()

        mid = self.interval_split(self.bounds[p])

        # b1 is lower half
        b1.bounds[p] = [b1.bounds[p][0], mid]

        # b2 is upper half
        b2.bounds[p] = [mid, b2.bounds[p][1]]

        return [b1, b2]
        
        
class AnalysisScenarioResult(object):
    def __init__(self, scenario: AnalysisScenario) -> None:
        self.scenario = scenario

class ParameterSynthesisScenarioResult(AnalysisScenarioResult):
    def __init__(self, scenario: ParameterSynthesisScenario) -> None:
        super().__init__(scenario)

class Model(object):
    def __init__(self, formula) -> None:
        self.formula = formula

class Parameter(object):
    def __init__(self, symbol) -> None:
        self.symbol = symbol

    def __eq__(self, other): 
        if not isinstance(other, Parameter):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return self.symbol.symbol_name() == other.symbol.symbol_name()

    def __hash__(self):
        # necessary for instances to behave sanely in dicts and sets.
        return hash(self.symbol)


class Funman(object):
    def __init__(self) -> None:
        self.scenario_handlers = {
            ParameterSynthesisScenario : self.synthesize_parameters
        }

    def solve(self, problem: AnalysisScenario) -> AnalysisScenarioResult:
        return self.scenario_handlers[type(problem)](problem)

    def synthesize_parameters(self, problem : ParameterSynthesisScenario) -> ParameterSynthesisScenarioResult:
        """ 
        """
        initial_box = Box(problem.parameters)
        num_parameters = len(list(initial_box.bounds.values()))
        unknown_boxes = [initial_box]
        true_boxes = []
        false_boxes = []
        max_uk_box_length = 10e10      ## initialization  
        tol = 10e-3 ## arbitrary precision - gives the smallest box that you would still want to split.
        while len(unknown_boxes) > 0 and max_uk_box_length > tol:
            ##print("number of unknown boxes:",len(unknown_boxes))
            ## Create list of unknown boxes for each parameter
            max_uk_box_length_per_parameter = []
            largest_uk_box_index_per_parameter = []
            for j in range(num_parameters):
                uk_box_bounds = [list(i.bounds.values())[j] for i in unknown_boxes]
                ## Calculate lengths of boxes
                uk_box_lengths = [float(uk_box_bounds[i][1]) - float(uk_box_bounds[i][0]) for i in range(len(uk_box_bounds))]
                max_uk_box_length = max(uk_box_lengths)
                largest_uk_box_index = uk_box_lengths.index(max(uk_box_lengths))
                ## choose to work on the largest unknown box first
                max_uk_box_length_per_parameter.append(max_uk_box_length) 
                largest_uk_box_index_per_parameter.append(largest_uk_box_index)
                ## find parameter with largest width
                parameter_argmax = max_uk_box_length_per_parameter.index(max(max_uk_box_length_per_parameter)) 
                ## find index of largest width element for that parameter
                largest_uk_box_index = largest_uk_box_index_per_parameter[parameter_argmax] 
            box = unknown_boxes.pop(largest_uk_box_index) ## pop(0) is ok too 
            phi = And(box.to_smt(), Not(problem.model.formula).simplify())
            res = get_model(phi) 
            if res:
                # split because values in box are not in model: either f or u
                phi1 = And(box.to_smt(), problem.model.formula).simplify()
                res1 = get_model(phi1)
                if res1:
                    b1, b2 = box.split()
                    unknown_boxes = unknown_boxes + [b2, b1]
                else:
                    false_boxes.append(box)  # TODO consider merging lists of boxes
            else: # done
                true_boxes.append(box) # TODO consider merging lists of boxes

            printable_list_false_boxes = [list((false_boxes[i].bounds.values())) for i in range(len(false_boxes))]
            print('false boxes:', printable_list_false_boxes)
            printable_list_true_boxes = [list((true_boxes[i].bounds.values())) for i in range(len(true_boxes))]
            print('true boxes:', printable_list_true_boxes)
            printable_list_unknown_boxes = [list((unknown_boxes[i].bounds.values())) for i in range(len(unknown_boxes))]           
            print('unknown boxes:', printable_list_unknown_boxes)
            ### 1D Plotting
            if num_parameters == 1: 
                for i in printable_list_unknown_boxes:
                    point1 = float(i[0][0])
                    point2 = float(i[0][1])
                    if point1 > -20 and point2 < 20:
                        x_values = [point1, point2]
                        plt.plot(x_values, np.zeros(len(x_values)),'b', linestyle="-")
                    else:
                        x_values = [-20, 20]
                        plt.plot(x_values, np.zeros(len(x_values)),'b', linestyle="-")
      
                for i in printable_list_true_boxes:
                    point1 = float(i[0][0])
                    point2 = float(i[0][1])
                    if point1 > -20 and point2 < 20:
                        x_values = [point1, point2]
                        plt.plot(x_values, np.zeros(len(x_values)),'g', linestyle="-")
                for i in printable_list_false_boxes:
                    point1 = float(i[0][0])
                    point2 = float(i[0][1])
                    if float(point1) > -20 and float(point2) < 20:
                        x_values = [point1, point2]
                        plt.plot(x_values, np.zeros(len(x_values)),'r', linestyle="-")
                plt.xlabel('x')
                plt.title('0 <= x <= 5')
                custom_lines = [Line2D([0], [0], color='b', lw=4),Line2D([0], [0], color='g', lw=4), Line2D([0], [0], color='r', lw=4)]
                plt.legend(custom_lines,['unknown','true','false'])
                plt.show(block=False)
                plt.pause(.1)
                plt.close()
            ### 2D Plotting
            elif num_parameters == 2: 
                for i in printable_list_unknown_boxes:
                    x_limits = i[0]
                    y_limits = i[1]
                    if float(x_limits[0]) > -20 and float(x_limits[1]) < 20:
                        x = np.linspace(x_limits[0], x_limits[1],1000)
                        plt.fill_between(x, y_limits[0], y_limits[1], color='b')
                    else:
                        x = np.linspace(-20, 20, 1000)
                        plt.fill_between(x, y_limits[0], y_limits[1], color='b')
                for i in printable_list_false_boxes:
                    x_limits = i[0]
                    y_limits = i[1]
                    if float(x_limits[0]) > -20 and float(x_limits[1]) < 20:
                        x = np.linspace(x_limits[0], x_limits[1],1000)
                        plt.fill_between(x, y_limits[0], y_limits[1], color='r')
                for i in printable_list_true_boxes:
                    x_limits = i[0]
                    y_limits = i[1]
                    if float(x_limits[0]) > -20 and float(x_limits[1]) < 20:
                        x = np.linspace(x_limits[0], x_limits[1],1000)
                        plt.fill_between(x, y_limits[0], y_limits[1], color='g')
                plt.xlabel('x')
                plt.ylabel('y')
                plt.title('0 <= x <= 5, 10 <= y <= 12')
                custom_lines = [Line2D([0], [0], color='b', lw=4),Line2D([0], [0], color='g', lw=4), Line2D([0], [0], color='r', lw=4)]
                plt.legend(custom_lines,['unknown','true','false'])
                plt.show(block=False)
                plt.pause(.1)
                plt.close()

            

        # FIXME The code below will create a formula phi that is the model and initial box.
        #       It needs to be extended to find the parameter space, per notes above.
        # You will need to extend the Box class to handle bisecting, and any other manipulations.
        # Also, the call the solver will also need a different phi (e.g., (And(box, Not(model))))
        

        # FIXME the parameter space will be added the to object below, in addition to the problem.
        #       Initially, the parameter space can be a set of boxes
        return ParameterSynthesisScenarioResult(problem)

