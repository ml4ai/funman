from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    data_path: str = "."
    funman_admin_token: Optional[str] = None
    funman_api_token: Optional[str] = None
    funman_base_url: Optional[str] = None
