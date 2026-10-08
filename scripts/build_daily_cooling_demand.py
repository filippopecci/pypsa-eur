# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Build daily cooling demand time series with the cooling degree day (CDD) approximation.

Mirror of ``build_daily_heat_demand`` using ``atlite.convert.cooling_demand``: daily mean
ambient temperature above ``threshold`` (degC), weighted by population within each
onshore region.

.. seealso::
    `Atlite.Cutout.cooling_demand <https://atlite.readthedocs.io/en/master/ref_api.html#module-atlite.convert>`_
"""

import logging

import geopandas as gpd
import numpy as np
import xarray as xr
from dask.distributed import Client, LocalCluster

from scripts._helpers import (
    configure_logging,
    get_snapshots,
    load_cutout,
    set_scenario_config,
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake("build_daily_cooling_demand_profile_only")
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    nprocesses = int(snakemake.threads)
    cluster = LocalCluster(n_workers=nprocesses, threads_per_worker=1)
    client = Client(cluster, asynchronous=True)

    time = get_snapshots(snakemake.params.snapshots, snakemake.params.drop_leap_day)
    daily = get_snapshots(
        snakemake.params.snapshots,
        snakemake.params.drop_leap_day,
        freq="D",
    )

    cutout = load_cutout(snakemake.input.cutout, time=time)

    clustered_regions = (
        gpd.read_file(snakemake.input.regions_onshore).set_index("name").buffer(0)
    )

    I = cutout.indicatormatrix(clustered_regions)  # noqa: E741

    pop_layout = xr.open_dataarray(snakemake.input.pop_layout)

    stacked_pop = pop_layout.stack(spatial=("y", "x"))
    M = I.T.dot(np.diag(I.dot(stacked_pop)))

    cooling_demand = cutout.cooling_demand(
        threshold=snakemake.params.threshold,
        matrix=M.T,
        index=clustered_regions.index,
        dask_kwargs=dict(scheduler=client),
        show_progress=False,
    ).sel(time=daily)

    cooling_demand.to_netcdf(snakemake.output.cooling_demand)
