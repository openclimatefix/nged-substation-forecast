from dagster import ConfigurableResource, EnvVar
from pydantic import Field
from obstore.store import S3Store

class S3Resource(ConfigurableResource):
    """
    A Dagster resource for interacting with an S3-compatible store using obstore.
    """

    bucket_url: str = Field(description="The URL of the S3 bucket.")
    access_key: str = Field(
        default=EnvVar("AWS_ACCESS_KEY_ID"), description="The AWS access key."
    )
    secret_key: str = Field(
        default=EnvVar("AWS_SECRET_ACCESS_KEY"), description="The AWS secret key."
    )

    def get_store(self) -> S3Store:
        """
        Returns an initialized obstore.store.S3Store instance.
        """
        return S3Store.from_url(
            url=self.bucket_url,
            config={
                "aws_access_key_id": self.access_key,
                "aws_secret_access_key": self.secret_key,
            },
        )
