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
    from data_nged.ckan_client import NGEDCKANClient
    from data_nged.process_flows import process_live_primary_substation_flows
    import polars as pl
    return NGEDCKANClient, pl, process_live_primary_substation_flows


@app.cell
def _(NGEDCKANClient):
    nged_ckan = NGEDCKANClient()
    return (nged_ckan,)


@app.cell
def _(nged_ckan):
    nged_ckan.api_key
    return


@app.cell
def _(nged_ckan):
    nged_ckan.get_primary_substation_locations()
    return


@app.cell
def _(nged_ckan):
    resources = nged_ckan.get_resources_for_live_primary_substation_flows()
    return (resources,)


@app.cell
def _(nged_ckan, pl, resources):
    df = pl.read_csv(nged_ckan.download_resource(resources[0]))
    df
    return


@app.cell
def _(nged_ckan, process_live_primary_substation_flows, resources):
    dfs = {}
    for i, resource in enumerate(resources[:10]):
        print(f"{i:03d}", end="\r")
        if resource.format == "CSV" and resource.size > 100:
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
    return


if __name__ == "__main__":
    app.run()
