from pydantic import BaseSettings
from pydantic import Field


class AppConfig(BaseSettings):
    """This class describes the config specific for running this app."""

    data_path: str = Field(
        None,
        description="The directory containing the data",
    )

    project_id: str = Field(None, description="GCP project id to access github.archive")

    github_user: str = Field(
        None, description="Github user to get your starred repos programmatically"
    )

    github_token: str = Field(
        None, description="Github token to get your starred repos programmatically"
    )
