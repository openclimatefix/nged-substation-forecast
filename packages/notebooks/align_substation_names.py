import marimo

__generated_with = "0.19.7"
app = marimo.App(width="full")


@app.cell
def _():
    from nged_data import ckan
    import polars as pl
    return ckan, pl


@app.cell
def _(ckan):
    locations = ckan.get_primary_substation_locations()
    locations = locations.sort(by="substation_name")
    locations
    return (locations,)


@app.cell
def _(ckan, pl):
    live_flows = ckan.get_csv_resources_for_live_primary_substation_flows()
    live_flows = pl.DataFrame(live_flows)
    live_flows
    return (live_flows,)


@app.cell
def _(live_flows, pl):
    def simplify_substation_name(col_name: str) -> pl.Expr:
        return (
            pl.col(col_name)
            .str.replace_all(" Primary Transformer Flows", "")
            .str.replace_all(r"\d{2,}(?i)kv", "")  # the (?i) means "ignore case"
            .str.replace_all("S/Stn", "", literal=True)  # e.g. "Sheepbridge 11/6 6kv S/Stn"
            .str.replace_all("/", "", literal=True)
            .str.replace_all("132", "")
            .str.replace_all("66", "")
            .str.replace_all("33", "")
            .str.replace_all("11", "")
            .str.replace_all(r"6\.6(?i)kv", "")
            # e.g. "Sheffield Road 33 11 6 6kv S Stn", "Sheepbridge 11/6 6kv S/Stn", "Sandy Lane 33 6 6kv S Stn"
            .str.replace_all("6 6kv", "")
            .str.replace_all("Primary", "")  # Live primary flows tend to end the name with just "Primary".
            .str.replace_all("S Stn", "")  # Assuming "S Stn" is short for substation.
            .str.replace_all(" Kv", "")  # "Infinity Park 33 11 Kv S Stn"
            .str.replace_all("  ", " ")
            .str.strip_chars()
            .str.strip_chars(".")
        )


    live_flows_with_simple_name = live_flows.with_columns(simple_name=simplify_substation_name("name")).sort(
        by="simple_name"
    )
    live_flows_with_simple_name
    return live_flows_with_simple_name, simplify_substation_name


@app.cell
def _(locations, simplify_substation_name):
    locations_with_simple_name = locations.with_columns(simple_name=simplify_substation_name("substation_name")).sort(
        by="simple_name"
    )
    locations_with_simple_name
    return (locations_with_simple_name,)


@app.cell
def _(live_flows_with_simple_name, locations_with_simple_name):
    live_flows_with_simple_name.join(locations_with_simple_name, on="simple_name", how="anti")
    return


@app.cell
def _(live_flows_with_simple_name, locations_with_simple_name):
    locations_with_simple_name.join(live_flows_with_simple_name, on="simple_name", how="anti")
    return


@app.cell
def _(live_flows_with_simple_name, locations_with_simple_name):
    for row in live_flows_with_simple_name.join(locations_with_simple_name, on="simple_name", how="anti")["simple_name"]:
        print(f'["{row}", ""],')
    return


@app.cell
def _(pl):
    pl.DataFrame(
        [
            ["Alliance & Leicester", "Alliance And Leicester"],
            ["Annesley", "Annesley (Kirkby)"],
            ["Boothen", ""],  # Closest match is "Boothville"
            ["Bridgend T E", "Bridgend Trading Estate"],
            ["Broad Street", "Broad St.Barry"],
            ["Caerau Road", "Caerau"],
            ["Cardiff South", ""],  # No substations starting with "Cardiff"
            ["Carlton-On-Trent", "Carlton On Trent"],
            ["Chipping Sodbury", ""],
            ["Church Street", "Church St"],
            ["Commercial Street", "Commercial St Neath"],  # There is also a "Commercial Road"
            ["Court Road", "Court Road Barry"],
            ["Dillotford Avenue", "Dillotford Ave"],
            ["Feckenham", ""],
            ["Feeder Road A", "Feeder Rd"],
            ["Gethin St", "Gethin Street Swansea"],
            ["Hallcroft Road", "Hallcroft Rd"],
            ["Hinksford", ""],
            ["Ketley", ""],
            ["Kidderminster", ""],
            ["Kilton Road", "Kilton Rd"],
            ["Lime St", "Lime Street Gorseinon"],
            ["Ludlow", ""],
            ["Lydney", ""],  # Nearest is "Lydeard St Lawrence"
            ["Mansfield Street", "Mansfield St"],
            ["Meaford", ""],
            ["Moretonhampstead", "Mortonhampstead"],
            ["Mount Bridge", "Mount Bridge Boston"],
            ["New Beacon Road", "New Beacon Road Grantham"],
            ["Newcastle Emlyn Sth", "Newcastle Emlyn South"],
            ["Newhouse", "Newhouse SS"],
            ["Oldbury B", ""],
            ["Pantyffynnon", "Pantyffynon"],
            ["Pengam", ""],
            ["Pont Ar Anell", "Pont Ar Annell"],
            ["Pontllanfraith", "Pontllanffraith"],
            ["Ringland", "Ringland Newport"],
            ["Ryeford", ""],
            ["Ship Hill", "Ship Hill Barry"],
            ["Sleaford Road", "Sleaford Road Boston"],
            ["Spalding", "Spalding Clay Lake"],
            ["Stourport", ""],
            ["Strand", ""],
            ["Swansea Road", "Strand Swansea"],
            ["Swansea Trading Est", "Swansea Trading Estate"],
            ["Talbot Street", "Talbot St"],
            ["Tinwell Road", "Tinwell Road Ketton"],
            ["Tir John", ""],
            ["Tunnel Bank", "Tunnel Bank Bourne"],
            ["Uplands", "Uplands Swansea"],
            ["Upton Warren", ""],
            ["Wellington Street", "Wellington St"],
            ["West Wick", "Westwick"],
            ["Whitfield", ""],
            ["Wolverhampton", ""],
            ["Wolverhampton West", ""],
            ["Wood End", ""],
            ["Wood St", "Wood Street"],
            ["Ynys St", "Ynys Street"],
        ],
        schema=["live_primary_flows", "location_table"],
    ).write_csv("map_substation_names_in_live_primary_flows_to_names_in_location_table.csv")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
