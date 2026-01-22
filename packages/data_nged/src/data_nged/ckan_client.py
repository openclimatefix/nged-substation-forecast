"""Generic CKAN client for interacting with NGED's Connected Data portal."""

from typing import Any

import requests


class NGEDCKANClient:
    """Client for NGED's CKAN API."""

    BASE_URL = "https://connecteddata.nationalgrid.co.uk/api/3/action"

    def __init__(self, base_url: str | None = None) -> None:
        """Initialize the client.

        Args:
            base_url: Optional base URL for the CKAN API.
        """
        self.base_url = base_url or self.BASE_URL

    def get_package_show(self, package_id: str) -> dict[str, Any]:
        """Get details about a package.

        Args:
            package_id: The ID of the package.

        Returns:
            dict: The package details.
        """
        response = requests.get(f"{self.base_url}/package_show?id={package_id}", timeout=30)
        response.raise_for_status()
        return response.json()["result"]

    def search_packages(self, query: str) -> list[dict[str, Any]]:
        """Search for packages.

        Args:
            query: The search query.

        Returns:
            list: The search results.
        """
        response = requests.get(f"{self.base_url}/package_search?q={query}", timeout=30)
        response.raise_for_status()
        return response.json()["result"]["results"]
