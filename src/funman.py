

class Funman(object):
    def __init__(self) -> None:
        self.scenario_handlers = {
            ParameterSynthesisScenario : synthesize_parameters
        }

    def solve(problem: AnalysisScenario):
        return self.scenario_handlers[problem](problem)

    def synthesize_parameters(problem : ParameterSynthesisScenario):
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



class AnalysisScenario(object):
    pass

class ParameterSynthesisScenario(AnalysisScenario):
    def __init__(self) -> None:
        super().__init__()
        self.parameters = [Parameter("foo"), Parameter("bar")]
        self.model = Model()
        
class Model(object):
    pass

class Parameter(object):
    def __init__(self, name) -> None:
        self.name = name

