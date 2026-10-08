# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Build shapes for the ``profile_only`` countries, which lie outside the modelled network.

Admin-1 boundaries come from OSM and population from the WorldPop raster, as for the
non-NUTS countries in ``build_shapes``. Each country is one region, named by its ISO2
code, so the outputs mirror ``regions_{onshore,offshore}_base_s_{clusters}`` at
administrative level 0. EEZ codes in ``assign`` (e.g. ``{EH: MA}``) are merged into the
country they map to.
"""

import json
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
from shapely.geometry import box

from scripts._helpers import configure_logging, set_scenario_config
from scripts.build_osm_boundaries import build_osm_boundaries
from scripts.build_shapes import eez

logger = logging.getLogger(__name__)

GEO_CRS = "EPSG:4326"

# OSM files Western Sahara (EH) under Morocco's regions; the internationally recognised
# border is the 27°40'N parallel. MA is clipped there unless `assign` merges EH into MA.
RECOGNISED_SOUTHERN_BORDER = {"MA": ("EH", 27 + 40 / 60)}


def iso_codes(adm1_fn, code):
    """
    ISO 3166-2 code by OSM relation id, for the relations that belong to ``code``.

    ``build_osm_boundaries`` renumbers every id once a single relation lacks the tag,
    which defeats its own foreign-relation filter (e.g. Ceuta, ES-CE, among MA).
    """
    codes = {}
    for element in json.load(open(adm1_fn))["elements"]:
        tags = element.get("tags", {})
        iso = tags.get("ISO3166-2") or tags.get("iso3166-2")
        if iso and iso.startswith(f"{code}-"):
            codes[element["id"]] = iso
        else:
            logger.warning(
                f"Dropping OSM relation {element['id']} ({tags.get('name')}, "
                f"ISO3166-2={iso}) from {code}."
            )
    return codes


def make_disjoint(regions):
    """Subtract smaller regions from larger ones, e.g. a new province not yet cut out of its parent."""
    regions = regions.assign(_area=regions.to_crs(6933).area).sort_values("_area")
    covered = None
    geometries = []
    for geom in regions.geometry:
        geometries.append(geom if covered is None else geom.difference(covered))
        covered = geom if covered is None else covered.union(geom)
    return regions.set_geometry(geometries, crs=regions.crs).drop(columns="_area")


def clip_south(regions, lat):
    """Keep the polygonal parts of ``regions`` north of latitude ``lat``."""
    return gpd.clip(regions, box(-180.0, lat, 180.0, 90.0), keep_geom_type=True)


def population_by_region(regions, population_fn):
    """Sum the population raster over each region, in thousands like ``nuts3_shapes``."""
    pop = pd.Series(0.0, index=regions.index)
    with rasterio.open(population_fn) as src:
        for idx, geom in regions.to_crs(src.crs).geometry.items():
            data, _ = mask(src, [geom], crop=True, filled=True, nodata=src.nodata)
            values = data[0]
            valid = np.isfinite(values) & (values > 0)
            if src.nodata is not None:
                valid &= values != src.nodata
            pop[idx] = values[valid].sum() / 1e3
    return pop


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake("build_profile_only_shapes")
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    countries = list(snakemake.params.countries)
    assign = dict(snakemake.params.assign or {})
    owner = {c: assign.get(c, c) for c in countries + list(assign)}

    offshore = eez(snakemake.input.eez, list(owner))
    offshore = (
        offshore.assign(country=offshore.index.map(owner))
        .dissolve(by="country")
        .rename_axis("name")[["geometry"]]
    )

    adm1 = []
    covered = None
    for fn in snakemake.input.adm1:
        code = Path(fn).name.split("_")[0]
        gdf = build_osm_boundaries(code, fn, offshore)
        codes = iso_codes(fn, code)
        gdf = gdf[gdf["osm_id"].isin(codes)]
        if gdf.empty:
            raise ValueError(f"No OSM admin-1 boundaries found for {code} in {fn}.")
        gdf = make_disjoint(gdf.assign(id=gdf["osm_id"].map(codes)))
        if code in RECOGNISED_SOUTHERN_BORDER:
            territory, lat = RECOGNISED_SOUTHERN_BORDER[code]
            if assign.get(territory) != code:
                logger.info(f"Clipping {code} at {lat:.4f}N to exclude {territory}.")
                gdf = clip_south(gdf, lat)
        gdf["country"] = owner[code]
        # Keep countries disjoint where their OSM boundaries overlap.
        if covered is not None:
            gdf["geometry"] = gdf.geometry.difference(covered)
            gdf = gdf[~gdf.geometry.is_empty]
        covered = (
            gdf.geometry.union_all()
            if covered is None
            else covered.union(gdf.geometry.union_all())
        )
        adm1.append(gdf)
    adm1 = gpd.GeoDataFrame(pd.concat(adm1, ignore_index=True), crs=GEO_CRS)

    missing = set(countries) - set(adm1.country)
    if missing:
        raise ValueError(f"No onshore shapes for profile-only countries {missing}.")

    adm1 = adm1.set_index("id")
    adm1.index.name = "index"
    adm1["pop"] = population_by_region(adm1, snakemake.input.population)
    logger.info(
        "Population (million) by country:\n%s",
        (adm1.groupby("country")["pop"].sum() / 1e3).round(2).to_string(),
    )
    country_shapes = (
        adm1.dissolve(by="country").rename_axis("name")[["geometry"]].loc[countries]
    )
    offshore = offshore.loc[offshore.index.intersection(countries)]

    # atlite silently ignores the parts of a region that lie outside the cutout.
    cut = snakemake.params.cutout
    (x0, x1), (y0, y1), hx, hy = cut["x"], cut["y"], cut["dx"] / 2, cut["dy"] / 2
    grid = box(x0 - hx, y0 - hy, x1 + hx, y1 + hy)
    for kind, shapes in [("onshore", country_shapes), ("offshore", offshore)]:
        outside = shapes.geometry.difference(grid).to_crs(6933).area / 1e6
        if (outside > 1.0).any():
            raise ValueError(
                f"{outside[outside > 1.0].round(1).to_dict()} km2 of {kind} regions lie "
                "outside the profile-only cutout; widen profile_only.cutout.x/y."
            )

    adm1[["name", "country", "pop", "geometry"]].reset_index().to_file(
        snakemake.output.nuts3_shapes
    )
    country_shapes.reset_index().to_file(snakemake.output.country_shapes)
    country_shapes.reset_index().to_file(snakemake.output.regions_onshore)
    offshore.reset_index().to_file(snakemake.output.offshore_shapes)
    offshore.reset_index().to_file(snakemake.output.regions_offshore)
