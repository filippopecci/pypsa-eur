# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Power plant list for the ``profile_only`` countries, in the format of
``powerplants_s_{clusters}.csv``.

The pre-matched powerplantmatching list used by ``build_powerplants`` covers Europe only,
so these countries take powerplantmatching's Global Energy Monitor (GEM) trackers
directly. Two powerplantmatching defaults are overridden: ``main_query`` drops every
plant south of 30N, and the gas tracker labels open-cycle gas turbines "Steam Turbine".
The rest follows ``build_powerplants``.
"""

import logging

import country_converter as coco
import geopandas as gpd
import pandas as pd
import powerplantmatching as pm

from scripts._helpers import configure_logging, set_scenario_config
from scripts.build_powerplants import (
    map_to_country_bus,
    replace_natural_gas_fueltype,
    replace_natural_gas_technology,
)

logger = logging.getLogger(__name__)

# GEM gas tracker technologies that powerplantmatching maps to "Steam Turbine".
OPEN_CYCLE = ["gas turbine", "internal combustion"]
GEM_TRACKERS = ["GBPT", "GGPT", "GCPT", "GGTPT", "GNPT", "GSPT", "GWPT", "GHPT"]


def gem_powerplants(countries):
    """GEM units of ``countries`` (ISO2) through powerplantmatching, with open-cycle units as OCGT."""
    config = pm.get_config()
    config["target_countries"] = coco.convert(countries, to="name_short")
    config["main_query"] = "Name != ''"
    # Units under construction mostly lack a start year here, and `powerplants_filter`
    # keeps undated units as existing (e.g. a 10000 MW wind project in Egypt).
    for tracker in GEM_TRACKERS:
        status = config[tracker].get("status", ["operating"])
        config[tracker]["status"] = [s for s in status if s != "construction"]
    ppl = pm.data.GEM(config=config)

    gas = pm.data.GGPT(raw=True, config=config)
    open_cycle = gas.loc[
        gas["Turbine/Engine Technology"].isin(OPEN_CYCLE), "GEM unit ID"
    ]
    ppl.loc[ppl.projectID.isin(open_cycle), "Technology"] = "OCGT"
    return ppl


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake("build_powerplants_profile_only")
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    countries = list(snakemake.params.countries)

    regions = pd.concat(
        [
            gpd.read_file(snakemake.input.regions_onshore),
            gpd.read_file(snakemake.input.regions_offshore),
        ]
    ).dissolve("name")

    ppl = (
        gem_powerplants(countries)
        .powerplant.convert_country_to_alpha2()
        .query("Country in @countries")
        .assign(Technology=replace_natural_gas_technology)
        .assign(Fueltype=replace_natural_gas_fueltype)
        .replace({"Solid Biomass": "Bioenergy", "Biogas": "Bioenergy"})
    )

    ppl_query = snakemake.params.powerplants_filter
    if isinstance(ppl_query, str):
        ppl.query(ppl_query, inplace=True)

    ppl = ppl.dropna(subset=["lat", "lon"])
    ppl = gpd.GeoDataFrame(ppl, geometry=gpd.points_from_xy(ppl.lon, ppl.lat), crs=4326)
    ppl = map_to_country_bus(ppl, regions)

    unassigned = ppl["bus"].isnull()
    if unassigned.any():
        logger.warning(
            "Removing power plants outside the profile-only regions (MW):\n%s",
            ppl.loc[unassigned].groupby(["Country", "Fueltype"]).Capacity.sum(),
        )
        ppl = ppl[~unassigned]

    logger.info(
        "Power plants by country and fuel (MW):\n%s",
        ppl.groupby(["Country", "Fueltype"]).Capacity.sum().unstack(0).round(1),
    )
    ppl.reset_index(drop=True).to_csv(snakemake.output[0])
