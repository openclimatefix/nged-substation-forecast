"""Generic CKAN client for interacting with NGED's Connected Data portal."""

import logging
import time
from datetime import datetime, timedelta
from typing import Any, Final

import httpx
import patito as pt
import polars as pl
from contracts.data_schemas import SubstationLocations

from nged_data.schemas import CkanResource, PackageSearchResult

from .utils import change_dataframe_column_names_to_snake_case, find_one_match

log = logging.getLogger(__name__)

BASE_CKAN_URL: Final[str] = "https://connecteddata.nationalgrid.co.uk"


def _ensure_absolute_url(url: str) -> str:
    if not url:
        return url
    if url.startswith("http://") or url.startswith("https://"):
        return url
    if url.startswith("/"):
        return BASE_CKAN_URL + url
    # Handle cases like "dataset/..."
    return BASE_CKAN_URL + "/" + url


def get_primary_substation_locations(api_key: str) -> pt.DataFrame[SubstationLocations]:
    """Note that 'Park Lane' appears twice (with different substation numbers)."""
    if api_key:
        log.info(
            "Fetching substation locations with API key (length: %d, prefix: %s...)",
            len(api_key),
            api_key[:4] if api_key else "None",
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
    url = _ensure_absolute_url(ckan_result["url"])
    http_response = httpx_get_with_auth(url, api_key=api_key)
    http_response.raise_for_status()
    locations = pl.read_csv(http_response.content)
    locations = change_dataframe_column_names_to_snake_case(locations)
    locations = locations.filter(
        pl.col("substation_type").str.to_lowercase().str.contains("primary")
    )
    locations = locations.cast(SubstationLocations.dtypes)  # type: ignore[invalid-argument-type]
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
    resources = [CkanResource.model_validate(resource) for resource in resources]

    # Log any redacted URLs found
    redacted_count = sum(1 for r in resources if str(r.url).lower() == "redacted")
    if redacted_count > 0:
        log.warning(
            "Found %d resources with 'redacted' URLs in search results (Auth: %s)",
            redacted_count,
            "Yes" if api_key else "No",
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

    # Fix relative URLs and ensure absolute
    for package in result.get("results", []):
        for resource in package.get("resources", []):
            original_url = resource.get("url", "")
            if original_url.lower() == "redacted":
                if api_key:
                    raise RuntimeError(
                        f"CKAN returned 'redacted' URL for resource '{resource.get('name')}' even though an API key was used! This suggests the key is invalid, lacks permissions, or was incorrectly passed (e.g. masked)."
                    )
                else:
                    log.warning("Found redacted URL in unauthenticated CKAN search")

            resource["url"] = _ensure_absolute_url(original_url)

    return PackageSearchResult.model_validate(result)


def httpx_get_with_auth(
    url: str, api_key: str, max_retries: int = 3, client: httpx.Client | None = None, **kwargs: Any
) -> httpx.Response:
    original_url = url
    url = _ensure_absolute_url(url)
    if url != original_url:
        log.debug("Fixed relative URL: %s -> %s", original_url, url)

    if not url.startswith("http"):
        log.error("Invalid URL requested (missing protocol): '%s'", url)

    if not api_key:
        log.warning("No API key provided for CKAN request to %s", url)

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
