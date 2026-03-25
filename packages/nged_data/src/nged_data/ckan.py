"""Generic CKAN client for interacting with NGED's Connected Data portal."""

import logging
import time
from datetime import datetime, timedelta
from typing import Any, Final, cast

import httpx
import patito as pt
import polars as pl
from contracts.data_schemas import SubstationLocations

from nged_data.schemas import CkanResource, PackageSearchResult

from .utils import change_dataframe_column_names_to_snake_case, find_one_match

log = logging.getLogger(__name__)

BASE_CKAN_URL: Final[str] = "https://connecteddata.nationalgrid.co.uk"


def get_primary_substation_locations(api_key: str) -> pt.DataFrame[SubstationLocations]:
    """Note that 'Park Lane' appears twice (with different substation numbers)."""
    if api_key:
        log.info(
            "Fetching substation locations with API key (length: %d, prefix: %s...)",
            len(api_key),
            api_key[:4],
        )
    else:
        log.warning("Fetching substation locations WITHOUT API key")

    url = f"{BASE_CKAN_URL}/api/3/action/resource_search"
    headers = {"Authorization": api_key} if api_key else {}
    params = {"query": "name:Primary Substation Location"}

    try:
        resp = httpx.get(url, headers=headers, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        log.error("CKAN resource_search failed: %s", e)
        raise

    if not data.get("success"):
        log.error("CKAN resource_search returned success=False: %s", data.get("error"))
        raise RuntimeError(f"CKAN search failed: {data.get('error')}")

    ckan_results = data["result"]["results"]
    ckan_result = find_one_match(lambda result: result["format"].upper() == "CSV", ckan_results)
    url = str(ckan_result["url"])
    http_response = httpx_get_with_auth(url, api_key=api_key)
    locations = pl.read_csv(http_response.content)
    locations = change_dataframe_column_names_to_snake_case(locations)
    locations = locations.filter(
        pl.col("substation_type").str.to_lowercase().str.contains("primary")
    )
    dtypes = cast(Any, SubstationLocations.dtypes)
    locations = locations.cast(dtypes)
    return SubstationLocations.validate(locations, drop_superfluous_columns=True)


def download_resource(resource: CkanResource, api_key: str) -> bytes:
    http_response = httpx_get_with_auth(str(resource.url), api_key=api_key)
    return http_response.content


def get_csv_resources_for_historical_primary_substation_flows(api_key: str) -> list[CkanResource]:
    return get_csv_resources_for_package(
        'title:"primary transformer flows"', api_key=api_key, max_age=timedelta(days=2)
    )


def get_csv_resources_for_live_primary_substation_flows(api_key: str) -> list[CkanResource]:
    return get_csv_resources_for_package(
        'title:"live primary"', api_key=api_key, max_age=timedelta(days=2)
    )


def get_csv_resources_for_package(
    query: str, api_key: str, max_age: timedelta | None = None
) -> list[CkanResource]:
    package_search_result = package_search(query, api_key=api_key)
    resources = []
    for result in package_search_result.results:
        resources.extend(result.resources)

    if any(str(r.url).lower() == "redacted" for r in resources):
        raise RuntimeError(
            "Found redacted resources. Please check your NGED_CKAN_TOKEN is correct and has the necessary permissions."
        )

    resources = [r for r in resources if r.format == "CSV" and r.size > 100]
    resources = remove_duplicate_names(resources)

    if max_age:
        min_modification_dt = datetime.now() - max_age
        resources = [r for r in resources if r.last_modified >= min_modification_dt]

    return resources


def package_search(query: str, api_key: str) -> PackageSearchResult:
    if api_key:
        log.info(
            "Performing CKAN search with API key (length: %d, prefix: %s...)",
            len(api_key),
            api_key[:4] if api_key else "None",
        )
    else:
        log.warning("Performing CKAN search WITHOUT API key")

    url = f"{BASE_CKAN_URL}/api/3/action/package_search"
    headers = {"Authorization": api_key} if api_key else {}
    params = {"q": query, "rows": 1000}  # Get more results in one go

    try:
        resp = httpx.get(url, headers=headers, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        log.error("CKAN package_search failed: %s", e)
        raise

    if not data.get("success"):
        log.error("CKAN package_search returned success=False: %s", data.get("error"))
        raise RuntimeError(f"CKAN search failed: {data.get('error')}")

    result = data.get("result", {})

    return PackageSearchResult.model_validate(result)


def httpx_get_with_auth(
    url: str, api_key: str, max_retries: int = 3, client: httpx.Client | None = None, **kwargs: Any
) -> httpx.Response:
    auth_headers = {"Authorization": api_key}
    for attempt in range(max_retries):
        try:
            if client:
                response = client.get(url=url, headers=auth_headers, timeout=30, **kwargs)
            else:
                response = httpx.get(url=url, headers=auth_headers, timeout=30, **kwargs)
            response.raise_for_status()
            return response
        except Exception as e:
            log.warning(
                "Attempt %d failed for URL %s (Auth: %s): %s",
                attempt + 1,
                url,
                "Yes" if api_key else "No",
                e,
            )
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** (attempt + 1))
    # This line should never be reached
    raise RuntimeError("Retry loop failed unexpectedly")


def remove_duplicate_names(resources: list[CkanResource]) -> list[CkanResource]:
    names = set()
    de_duped_resources = []
    for r in resources:
        if r.name not in names:
            de_duped_resources.append(r)
            names.add(r.name)

    return de_duped_resources
