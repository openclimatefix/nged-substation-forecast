"""Generic CKAN client for interacting with NGED's Connected Data portal."""

import logging
import os
from datetime import datetime, timedelta
from typing import Any, Final

import httpx
import patito as pt
import polars as pl
from ckanapi import RemoteCKAN
from contracts.data_schemas import SubstationLocations
from dotenv import load_dotenv

from nged_data.schemas import CkanResource, PackageSearchResult

from .utils import change_dataframe_column_names_to_snake_case, find_one_match

log = logging.getLogger(__name__)

# The name of the environment variable that holds the NGED CKAN TOKEN.
NGED_CKAN_TOKEN_ENV_KEY: Final[str] = "NGED_CKAN_TOKEN"
BASE_CKAN_URL: Final[str] = "https://connecteddata.nationalgrid.co.uk"


def get_primary_substation_locations() -> pt.DataFrame[SubstationLocations]:
    """Note that 'Park Lane' appears twice (with different substation numbers)."""
    api_key = get_nged_ckan_token_from_env()
    with RemoteCKAN(BASE_CKAN_URL, apikey=api_key) as nged_ckan:
        ckan_response = nged_ckan.action.resource_search(query="name:Primary Substation Location")
    ckan_results = ckan_response["results"]
    ckan_result = find_one_match(lambda result: result["format"].upper() == "CSV", ckan_results)
    url = ckan_result["url"]
    http_response = httpx_get_with_auth(url)
    http_response.raise_for_status()
    locations = pl.read_csv(http_response.content)
    locations = change_dataframe_column_names_to_snake_case(locations)
    locations = locations.filter(
        pl.col("substation_type").str.to_lowercase().str.contains("primary")
    )
    locations = locations.cast(SubstationLocations.dtypes)  # type: ignore[invalid-argument-type]
    return SubstationLocations.validate(locations, drop_superfluous_columns=True)


def download_resource(resource: CkanResource) -> bytes:
    http_response = httpx_get_with_auth(str(resource.url))
    http_response.raise_for_status()
    return http_response.content


def get_csv_resources_for_historical_primary_substation_flows() -> list[CkanResource]:
    return get_csv_resources_for_package(
        'title:"primary transformer flows"', max_age=timedelta(days=2)
    )


def get_csv_resources_for_live_primary_substation_flows() -> list[CkanResource]:
    return get_csv_resources_for_package('title:"live primary"', max_age=timedelta(days=2))


def get_csv_resources_for_package(
    query: str, max_age: timedelta | None = None
) -> list[CkanResource]:
    package_search_result = package_search(query)
    resources = []
    for result in package_search_result.results:
        resources.extend(result.resources)
    resources = [CkanResource.model_validate(resource) for resource in resources]
    resources = [r for r in resources if r.format == "CSV" and r.size > 100]
    resources = remove_duplicate_names(resources)

    if max_age:
        # A handful of "live" resources haven't been updated for months. Let's ignore the old ones.
        min_modification_dt = datetime.now() - max_age
        resources = [r for r in resources if r.last_modified >= min_modification_dt]

    return resources


def package_search(query: str) -> PackageSearchResult:
    api_key = get_nged_ckan_token_from_env()
    with RemoteCKAN(BASE_CKAN_URL, apikey=api_key) as nged_ckan:
        result: dict[str, Any] = nged_ckan.action.package_search(q=query)
    result_validated = PackageSearchResult.model_validate(result)
    log.debug(
        "%d results found from CKAN 'package_search?q=%s'", len(result_validated.results), query
    )
    return result_validated


def httpx_get_with_auth(url: str, **kwargs) -> httpx.Response:
    api_key = get_nged_ckan_token_from_env()
    auth_headers = {"Authorization": api_key}
    return httpx.get(url=url, headers=auth_headers, **kwargs)


def get_nged_ckan_token_from_env() -> str:
    load_dotenv()
    try:
        return os.environ[NGED_CKAN_TOKEN_ENV_KEY]
    except KeyError:
        raise KeyError(
            f"You must set {NGED_CKAN_TOKEN_ENV_KEY} in your .env file or in an"
            " environment variable. See the README for more info."
        )


def remove_duplicate_names(resources: list[CkanResource]) -> list[CkanResource]:
    names = set()
    de_duped_resources = []
    for r in resources:
        if r.name not in names:
            de_duped_resources.append(r)
            names.add(r.name)

    return de_duped_resources
