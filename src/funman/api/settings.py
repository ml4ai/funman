from typing import Optional

from pydantic import BaseSettings


class Settings(BaseSettings):
    funman_api_token: Optional[str] = None
    data_path: str = "."
