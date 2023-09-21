from typing import List

from pydantic import BaseModel

from .constraint import Constraint


class Explanation(BaseModel):
    expression: str


class BoxExplanation(Explanation):
    mutex_constraints: List[Constraint] = []


class ParameterSpaceExplanation(Explanation):
    true_explanations: List[BoxExplanation] = []
    false_explanations: List[BoxExplanation] = []
