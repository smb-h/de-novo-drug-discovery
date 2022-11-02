from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    # General
    app_name: str = "Roshan TDS"
    # Environment
    ENV: str = Field("dev", env="ENV")
    # Debug
    DEBUG: bool = Field(False, env="DEBUG")
    # Release
    RELEASE: str = Field("0.1.0", env="RELEASE")
    # Reports
    REPORTS_PATH = Field("reports", env="REPORTS_PATH")

    # Config
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
