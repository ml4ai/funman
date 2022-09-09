
from pysmt.shortcuts import get_model, And, LT, GE, TRUE

POS_INFINITY = "inf"
NEG_INFINITY = "-inf"

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
                    (GE(p.symbol, b[0]) if b[0] != NEG_INFINITY else TRUE()), 
                    (LT(p.symbol, b[1]) if b[1] != POS_INFINITY else TRUE())
                    ).simplify()
                for p, b in self.bounds.items()
            ]
        )
        
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

class Funman(object):
    def __init__(self) -> None:
        self.scenario_handlers = {
            ParameterSynthesisScenario : self.synthesize_parameters
        }

    def solve(self, problem: AnalysisScenario) -> AnalysisScenarioResult:
        return self.scenario_handlers[type(problem)](problem)

    def synthesize_parameters(self, problem : ParameterSynthesisScenario) -> ParameterSynthesisScenarioResult:
        """ Psuedocode:

            p = problem.parameters
            b0 = problem.initial_box(p)  # b0 = {p1: [0, 1], p2: [-inf, inf], ...} b1 = {p1: [-1, 0), ...}
            u = [b0] # unknown boxes
            t = [] # known sat boxes
            f = [] # known unsat boxes

            Case 1:
            while len(u) > 0:
                b = u.pop()
                b' = exists_forall(b, problem.model)   [  b  [ b' ]    ], # b' = {p1: [0.5, 1], p2: [0, 10], ...}
                
                if b':
                    b'' = b \ b' , And(b, Not(b')) = b''
                    u = u + [b'']
                    t = t + [b']
                else: # no b', forall_exists not b'
                    f = f + [b']

                    

            Case 2: 
            (un)sat = exists(Not(And(b, problem.model)))   
            if sat: 
                # b is not in t, and need to split
                b1, b2 = bisect(b)  # b = {p1: [0.5, 1], p2: [0, 10], ...}, b1 = {p1: [0.5, 1], p2: [0, 5], ...}, b2 = {p1: [0.5, 1], p2: [5, 10], ...}
                u = u + [b1, b2]
            elif unsat:
                # b is in t


            - Representation of a box: 
                - Case 1: Implicit, constraints
                - Case 2: Explicit, sets of intervals

            - Search Graph:
                - Case 1: Multi-ary branching, branching driven by ExForAll, children (t-region, u-region, f-region )
                - Case 2: Binary, bisections


            How do we manage search to get the most t/f boxes possible 

            Smoke test: Use Case 2, model = "x < 5, x > 0", param = "x", 

        Args:
            problem (ParameterSynthesisScenario): _description_
        """
        initial_box = Box(problem.parameters)

        # FIXME The code below will create a formula phi that is the model and initial box.
        #       It needs to be extended to find the parameter space, per notes above.
        # You will need to extend the Box class to handle bisecting, and any other manipulations.
        # Also, the call the solver will also need a different phi (e.g., Not(And(box, model)))
        phi = And(initial_box.to_smt(), problem.model.formula).simplify()
        res = get_model(phi) 

        # FIXME the parameter space will be added the to object below, in addition to the problem.
        #       Initially, the parameter space can be a set of boxes
        return ParameterSynthesisScenarioResult(problem)




