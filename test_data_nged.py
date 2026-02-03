# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "altair==6.0.0",
#     "data_nged",
#     "marimo",
# ]
# [tool.uv.sources]
# data_nged = { path = "./packages/data_nged", editable = true }
# ///

import marimo

__generated_with = "0.19.7"
app = marimo.App(width="full")


@app.cell
def _():
    from data_nged.ckan_client import NgedCkanClient
    from data_nged.schemas import CkanResource
    from data_nged.process_flows import process_live_primary_substation_flows
    import polars as pl
    return (
        CkanResource,
        NgedCkanClient,
        pl,
        process_live_primary_substation_flows,
    )


@app.cell
def _(NgedCkanClient):
    nged_ckan = NgedCkanClient()
    return (nged_ckan,)


@app.cell
def _(nged_ckan):
    nged_ckan.get_primary_substation_locations()
    return


@app.cell
def _(nged_ckan):
    resources = nged_ckan.get_csv_resources_for_live_primary_substation_flows()
    return (resources,)


@app.cell
def _():
    return


@app.cell
def _(pl, resources):
    rdf = pl.DataFrame(resources)
    rdf
    return (rdf,)


app._unparsable_cell(
    r"""
    CkanResource.
    """,
    name="_"
)


@app.cell
def _(CkanResource, rdf):
    CkanResource.validate_model(rdf.filter(rdf["name"].is_duplicated())[0].to_dict())
    return


@app.cell
def _(CkanResource, nged_ckan, pl, rdf):
    df = pl.read_csv(nged_ckan.download_resource(CkanResource.validate_model(rdf.filter(rdf["name"].is_duplicated())[0].to_dict())))
    df
    return


@app.cell
def _(nged_ckan, process_live_primary_substation_flows, resources):
    dfs = {}
    for i, resource in enumerate(resources[:10]):
        print(f"{i:03d}", end="\r")
        try:
            dfs[resource.name] = process_live_primary_substation_flows(nged_ckan.download_resource(resource))
        except Exception as e:
            print(e, resource)
            continue
    return (dfs,)


@app.cell
def _(dfs):
    dfs
    return


@app.cell
def _(dfs):
    dfs["Alderton 11Kv Primary Transformer Flows"].plot.line(x="timestamp", y="MW")
    return


@app.cell
def _(pl, resources):
    pl.DataFrame([resource for resource in resources if resource.restricted_level is None])
    return


@app.cell
def _(pl, resources):
    pl.DataFrame([resource for resource in resources if not resource.datastore_active])
    return


@app.cell
def _():
    from pydantic import HttpUrl
    return (HttpUrl,)


@app.cell
def _(HttpUrl):
    from pathlib import PurePosixPath
    PurePosixPath(HttpUrl("http://example.com/foo/bar/baz.csv").path).name
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
