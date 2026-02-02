"""Generic CKAN client for interacting with NGED's Connected Data portal."""

import logging
import os
from typing import Any, Final

import httpx
import patito as pt
import polars as pl
from ckanapi import RemoteCKAN
from contracts.data_schemas import SubstationLocations
from dotenv import load_dotenv

from data_nged.schemas import CKANResource, PackageSearchResult

from .utils import change_dataframe_column_names_to_snake_case, find_one_match

log = logging.getLogger(__name__)

# The name of the environment variable that holds the NGED CKAN TOKEN.
NGED_CKAN_TOKEN_ENV_KEY: Final[str] = "NGED_CKAN_TOKEN"


class NGEDCKANClient:
    """Client for NGED's CKAN API."""

    BASE_URL = "https://connecteddata.nationalgrid.co.uk"

    def __init__(self, base_url: str | None = None, api_key: str | None = None) -> None:
        """Initialize the client."""
        self.base_url = base_url or self.BASE_URL
        if api_key:
            self.api_key = api_key
        else:
            load_dotenv()
            try:
                self.api_key = os.environ[NGED_CKAN_TOKEN_ENV_KEY]
            except KeyError:
                raise KeyError(
                    f"You must set {NGED_CKAN_TOKEN_ENV_KEY} in your .env file or in an"
                    " environment variable. See the README for more info."
                )

    def get_primary_substation_locations(self) -> pt.DataFrame[SubstationLocations]:
        with RemoteCKAN(self.base_url, apikey=self.api_key) as nged_ckan:
            ckan_response = nged_ckan.action.resource_search(
                query="name:Primary Substation Location"
            )
        ckan_results = ckan_response["results"]
        ckan_result = find_one_match(lambda result: result["format"].upper() == "CSV", ckan_results)
        url = ckan_result["url"]
        http_response = httpx.get(url, headers=self._auth_headers)
        http_response.raise_for_status()
        locations = pl.read_csv(http_response.content)
        locations = change_dataframe_column_names_to_snake_case(locations)
        locations = locations.filter(
            pl.col("substation_type").str.to_lowercase().str.contains("primary")
        )
        locations = locations.cast(SubstationLocations.dtypes)
        return SubstationLocations.validate(locations, drop_superfluous_columns=True)

    def download_resource(self, resource: CKANResource) -> bytes:
        http_response = httpx.get(str(resource.url), headers=self._auth_headers)
        http_response.raise_for_status()
        return http_response.content

    def get_resources_for_historical_primary_substation_flows(self) -> list[CKANResource]:
        return self.get_resources_for_package('title:"primary transformer flows"')

    def get_resources_for_live_primary_substation_flows(self) -> list[CKANResource]:
        return self.get_resources_for_package('title:"live primary"')

    def get_resources_for_package(self, query: str) -> list[CKANResource]:
        package_search_result = self.package_search(query)
        resources = []
        for result in package_search_result.results:
            resources.extend(result.resources)
        return [CKANResource.model_validate(resource) for resource in resources]

    def package_search(self, query: str) -> PackageSearchResult:
        with RemoteCKAN(self.base_url, apikey=self.api_key) as nged_ckan:
            result: dict[str, Any] = nged_ckan.action.package_search(q=query)
        result_validated = PackageSearchResult.model_validate(result)
        log.debug(
            "%d results found from CKAN 'package_search?q=%s'", len(result_validated.results), query
        )
        return result_validated

    @property
    def _auth_headers(self) -> dict[str, str]:
        return dict(Authorization=self.api_key)
