from contracts.settings import Settings
from dagster import ConfigurableResource
from obstore.store import S3Store


class S3Resource(ConfigurableResource):
    """
    A Dagster resource for interacting with an S3-compatible store using obstore.
    """

    def get_store(self) -> S3Store:
        """
        Returns an initialized obstore.store.S3Store instance.
        """
        return Settings().get_nged_s3_store()
