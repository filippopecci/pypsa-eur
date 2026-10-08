# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Land eligibility for the ``profile_only`` countries, which CORINE and Natura 2000 do not
cover.

Follows ``determine_availability_matrix_MD_UA``: Copernicus Global Land Cover codes
emulate the CORINE settings of each technology and WDPA replaces Natura 2000. Depth,
shore-distance and shipping exclusions are applied as in ``determine_availability_matrix``.
"""

import functools
import logging
import os
import time
from tempfile import NamedTemporaryFile

import atlite
import geopandas as gpd
import numpy as np

from scripts._helpers import configure_logging, load_cutout, set_scenario_config
from scripts.determine_availability_matrix_MD_UA import get_wdpa_layer_name

logger = logging.getLogger(__name__)

# Copernicus Global Land Cover classes emulating the CORINE grid codes, as for MD/UA.
COPERNICUS_CODES = {
    "solar": [20, 30, 40, 50, 60, 90, 100],
    "solar-hsat": [20, 30, 40, 50, 60, 90, 100],
    "onwind": [20, 30, 40, 60, 100],
    "offwind": [80, 200],
}
COPERNICUS_DISTANCE_CODES = {"onwind": [50]}


def add_wdpa(excluder, wdpa_fn, regions):
    """Exclude WDPA polygons and buffered points; returns the temporary files to delete."""
    tmp_files = []
    for layer_kind in ("polygons", "points"):
        layer = get_wdpa_layer_name(wdpa_fn, layer_kind)
        wdpa = gpd.read_file(wdpa_fn, bbox=regions.geometry, layer=layer).to_crs(3035)
        if layer_kind == "points":
            wdpa = wdpa[wdpa["REP_AREA"] > 1]
            radius = np.sqrt(wdpa["REP_AREA"] / np.pi) * 1000
            wdpa = wdpa.set_geometry(wdpa.geometry.buffer(radius))
        if wdpa.empty:
            continue
        # temporary file needed for parallelization
        with NamedTemporaryFile(suffix=".geojson", delete=False) as f:
            tmp_fn = f.name
        wdpa[["geometry"]].to_file(tmp_fn)
        excluder.add_geometry(tmp_fn)
        tmp_files.append(tmp_fn)
    return tmp_files


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "determine_availability_matrix_profile_only", technology="solar"
        )
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    nprocesses = int(snakemake.threads)
    noprogress = snakemake.config["run"].get("disable_progressbar", True)
    noprogress = noprogress or not snakemake.config["atlite"]["show_progress"]
    technology = snakemake.wildcards.technology
    params = snakemake.params.renewable[technology]
    kind = "offwind" if technology.startswith("offwind") else technology

    cutout = load_cutout(snakemake.input.cutout)
    regions = gpd.read_file(snakemake.input.regions)
    assert not regions.empty, f"List of regions in {snakemake.input.regions} is empty."
    regions = regions.set_index("name").rename_axis("bus")

    res = params.get("excluder_resolution", 100)
    excluder = atlite.ExclusionContainer(crs=3035, res=res)

    corine = params.get("corine") or {}
    if isinstance(corine, list):
        corine = {"grid_codes": corine}
    if "grid_codes" in corine:
        excluder.add_raster(
            snakemake.input.copernicus,
            codes=COPERNICUS_CODES[kind],
            invert=True,
            crs="EPSG:4326",
        )
    if corine.get("distance", 0.0) > 0.0:
        excluder.add_raster(
            snakemake.input.copernicus,
            codes=COPERNICUS_DISTANCE_CODES[kind],
            buffer=corine["distance"],
            crs="EPSG:4326",
        )

    tmp_files = []
    if params["natura"]:
        wdpa_fn = (
            snakemake.input.wdpa_marine if kind == "offwind" else snakemake.input.wdpa
        )
        tmp_files = add_wdpa(excluder, wdpa_fn, regions)

    if params.get("ship_threshold"):
        shipping_threshold = params["ship_threshold"] * 8760 * 6
        func = functools.partial(np.less, shipping_threshold)
        excluder.add_raster(
            snakemake.input.ship_density, codes=func, crs=4326, allow_no_overlap=True
        )

    if params.get("max_depth"):
        # exclude areas where: -max_depth > grid cell depth
        func = functools.partial(np.greater, -params["max_depth"])
        excluder.add_raster(snakemake.input.gebco, codes=func, crs=4326, nodata=-1000)

    if params.get("min_depth"):
        func = functools.partial(np.greater, -params["min_depth"])
        excluder.add_raster(
            snakemake.input.gebco, codes=func, crs=4326, nodata=-1000, invert=True
        )

    if params.get("min_shore_distance") is not None:
        excluder.add_geometry(
            snakemake.input.country_shapes, buffer=params["min_shore_distance"]
        )

    if params.get("max_shore_distance") is not None:
        excluder.add_geometry(
            snakemake.input.country_shapes,
            buffer=params["max_shore_distance"],
            invert=True,
        )

    logger.info(f"Calculate landuse availability for {technology}...")
    start = time.time()
    kwargs = dict(nprocesses=nprocesses, disable_progressbar=noprogress)
    availability = cutout.availabilitymatrix(regions, excluder, **kwargs)
    duration = time.time() - start
    logger.info(f"Completed landuse availability for {technology} ({duration:2.2f}s)")

    for fn in tmp_files:
        if os.path.exists(fn):
            os.remove(fn)

    availability.to_netcdf(snakemake.output.availability_matrix)
