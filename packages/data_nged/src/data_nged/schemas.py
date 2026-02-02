from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, HttpUrl


class PackageSearchResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    count: int  # The number of results
    facets: dict
    results: list[PackageSearchInnerResult]
    sort: str  # e.g. "score desc, metadata_modified desc"
    search_facets: dict


class PackageSearchInnerResult(BaseModel):
    # The fields below are no where near an exhaustive list of all the fields returned by CKAN!
    model_config = ConfigDict(extra="allow")

    resources: list[CKANResource]


class CKANResource(BaseModel):
    # The fields listed below are just the ones we care about, not al
    # exhaustive list of all the fields returned by CKAN!
    model_config = ConfigDict(extra="allow")

    created: datetime
    description: str | None
    format: Literal["CSV", "PDF"]
    id: str  # e.g. "1be842ce-b1d9-4494-a6ba-bf4bd3cfd336"
    last_modified: datetime
    metadata_modified: datetime
    mimetype: Literal["text/csv", "application/pdf"]
    name: str  # e.g. "Aberaeron 11kV Transformer Flows"
    package_id: str  # e.g. "55d6e4e7-98b7-45c0-969e-379a3652e760"
    restricted_level: Literal["registered", "public"] | None = None
    size: int
    state: Literal["active"]
    url: HttpUrl
