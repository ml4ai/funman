from pydantic import BaseModel


class Constraint(BaseModel):
    pass


class InitialStateConstraint(Constraint):
    pass


class QueryConstraint(Constraint):
    pass


class TransitionConstraint(Constraint):
    pass
