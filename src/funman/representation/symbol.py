from pydantic import BaseModel


class ModelSymbol(BaseModel):
    name: str
    model: str

    def __str__(self):
        return f"model_{self.model}_{self.name}"
