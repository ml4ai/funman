from pydantic import BaseModel


class Explanation(BaseModel):
    explanation: str
