# /// script
# dependencies = [
#     "ckanapi==4.9",
#     "httpx==0.28.1",
#     "marimo",
#     "polars==1.37.1",
#     "python-dotenv==1.2.1",
#     "rapidfuzz==3.14.3",
#     "requests==2.32.5",
# ]
# requires-python = ">=3.14"
# ///

import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    from dotenv import load_dotenv
    import os
    import ckanapi
    from typing import Final
    import logging
    import polars as pl
    from pathlib import PurePosixPath, Path
    import httpx
    from rapidfuzz import process, fuzz

    logging.basicConfig(level=logging.DEBUG)
    log = logging.getLogger(__name__)

    load_dotenv()
    return Final, Path, ckanapi, fuzz, httpx, log, os, pl, process


@app.cell
def _(Final, os):
    API_KEY: Final[str] = os.environ["NGED_CKAN_TOKEN"]
    HEADERS: Final[dict] = {"Authorization": API_KEY}
    return API_KEY, HEADERS


@app.cell
def _(API_KEY: "Final[str]", HEADERS: "Final[dict]", ckanapi, httpx, pl):
    # Get the Primary substation locations. This is also used as the authoritative list of primary substation names.
    def get_primary_substation_locations():
        with ckanapi.RemoteCKAN("https://connecteddata.nationalgrid.co.uk", apikey=API_KEY) as nged_ckan:
            # TODO we should be able to directly get the single dataset:
            # https://connecteddata.nationalgrid.co.uk/dataset/primary-substation-location-easting-northings
            ckan_response = nged_ckan.action.resource_search(query="name:Primary Substation Location")
        assert ckan_response["count"] == 1
        ckan_result = ckan_response["results"][0]
        assert ckan_result["format"].upper() == "CSV"
        url = ckan_result["url"]
        http_response = httpx.get(url, headers=HEADERS)
        http_response.raise_for_status()
        locations = pl.read_csv(http_response.content)
        locations = locations.filter(pl.col("Substation Type").str.to_lowercase().str.contains("primary"))
        return locations.with_columns(
            name_filtered=(
                pl.col("Substation Name")
                .str.replace_all("\d{2,}(?i)kv", "")
                .str.replace_all("/", "", literal=True)
                .str.replace_all("132", "")
                .str.replace_all("66", "")
                .str.replace_all("33", "")
                .str.replace_all("11", "")
                .str.replace_all("6\.6(?i)kv", "")
                .str.replace_all("Primary", "")
                .str.replace_all("S Stn", "")
                .str.replace_all("  ", " ")
                .str.strip_chars()
                .str.strip_chars(".")
            )
        ).sort("name_filtered")


    primary_substation_locations = get_primary_substation_locations()
    primary_substation_locations
    return (primary_substation_locations,)


@app.cell
def _(API_KEY: "Final[str]", ckanapi, log):
    def package_search(query: str) -> list[dict]:
        with ckanapi.RemoteCKAN("https://connecteddata.nationalgrid.co.uk", apikey=API_KEY) as nged_ckan:
            json_response = nged_ckan.action.package_search(q=query)
        results: list[dict] = json_response["results"]
        log.debug("%d results found from CKAN 'package_search?q=%s'", len(results), query)
        resources = []
        for result in results:
            resources.extend(result["resources"])
        return resources


    def extract_substation_names(resources: list[dict]) -> list[str]:
        names: list[str] = []
        for resource in resources:
            name = resource["name"]
            name = name.replace(" Transformer Flows", "")
            name = name.replace(" 11kV", "").replace(" 11Kv", "")  # Historical data
            name = name.replace(" Primary", "")  # Live data
            names.append(name)
        return names
    return extract_substation_names, package_search


@app.cell
def _(extract_substation_names, package_search, pl):
    names_historical = extract_substation_names(package_search('title:"primary transformer flows"'))
    pl.Series(name="names_historical", values=names_historical)
    return (names_historical,)


@app.cell
def _(extract_substation_names, package_search, pl):
    names_live = extract_substation_names(package_search('title:"live primary"'))
    pl.Series(name="names_live", values=names_live)
    return (names_live,)


@app.function
def set_diff(a, b) -> set:
    diff = set(a) - set(b)
    print("Number of items in set A but not in set B:", len(diff))
    return diff


@app.cell
def _(names_historical, names_live):
    set_diff(names_historical, names_live)
    return


@app.cell
def _(names_historical, names_live):
    set_diff(names_live, names_historical)
    return


@app.cell
def _(names_live, primary_substation_locations):
    set_diff(primary_substation_locations["name_filtered"], names_live)
    return


@app.cell
def _(names_live, primary_substation_locations):
    set_diff(names_live, primary_substation_locations["name_filtered"])
    return


@app.cell
def _(names_historical, primary_substation_locations):
    set_diff(primary_substation_locations["name_filtered"], names_historical)
    return


@app.cell
def _(names_historical, primary_substation_locations):
    set_diff(names_historical, primary_substation_locations["name_filtered"])
    return


@app.cell
def _(fuzz, names_live, primary_substation_locations, process):
    from collections import namedtuple

    Match = namedtuple("Match", field_names=["string", "score", "full_name"])


    def match(name: str) -> list[Match]:
        matches: list = process.extract(
            name, choices=primary_substation_locations["name_filtered"].to_list(), scorer=fuzz.WRatio, limit=3
        )
        return [
            Match(string=m[0], score=m[1], full_name=primary_substation_locations["Substation Name"][m[2]]) for m in matches
        ]


    match(names_live[0])
    return (match,)


@app.cell
def _(match, names_historical, pl):
    pl.DataFrame({"name": names_historical}).with_columns(
        match=pl.col("name").map_elements(
            match,
            return_dtype=pl.List(pl.Struct(fields={"string": pl.String, "score": pl.Float32, "full_name": pl.String})),
        )
    ).filter(pl.col("name") != pl.col("match").list.get(0).struct.field("string")).sort("name")
    return


@app.cell(disabled=True)
def _(pl, resources):
    df = pl.DataFrame(resources, infer_schema_length=None)
    df
    return


@app.cell(disabled=True)
def _(API_KEY: "Final[str]", Path, log, requests, resources):
    filtered_resources = list(filter(lambda r: r["format"] == "CSV", resources))

    log.debug("%d resources before filtering. %d after filtering", len(resources), len(filtered_resources))

    output_path = Path("~/data/NGED/CSV_manual_download").expanduser()
    output_path.mkdir(parents=True, exist_ok=True)

    for resource in filtered_resources:
        url = resource["url"]
        base_filename = url.split("/")[-1]
        dst_path = output_path / base_filename
        log.debug("Saving to '%s'", dst_path)
        _response = requests.get(url, headers={"Authorization": API_KEY})
        _response.raise_for_status()
        dst_path.write_bytes(_response.content)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
