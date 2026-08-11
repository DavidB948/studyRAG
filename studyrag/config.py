"""Central config. Everything tunable lives here, not scattered as literals."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str


settings = Settings()  # type: ignore[call-arg]
