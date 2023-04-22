from typing import Dict, List, Union

from pydantic import BaseModel

from funman.translate.decapode import DecapodeEncoder

from .model import Model


class DecapodeDynamics(BaseModel):
    json_graph: Dict[str, List[Dict[str, Union[int, str]]]]


class DecapodeModel(Model):
    decapode: DecapodeDynamics

    def default_encoder(self, config: "FUNMANConfig") -> "Encoder":
        """
        Return the default Encoder for the model

        Returns
        -------
        Encoder
            SMT encoder for model
        """
        return DecapodeEncoder(config=config)
