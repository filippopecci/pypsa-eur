# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Retrieve OSM admin-1 boundaries of one ``profile_only`` country from the Overpass API.

Same query as ``retrieve_osm_boundaries``, but sends the ``overpass_api`` User-Agent
(Overpass rejects requests without one) and fails instead of leaving no output.
"""

import json
import logging
import time

import requests

from scripts._helpers import configure_logging, set_scenario_config
from scripts.retrieve_osm_boundaries import ADM1_SPECIALS

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake("retrieve_osm_boundaries_profile_only", country="MA")
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    country = snakemake.wildcards.country
    api = snakemake.params.overpass_api
    ua = api["user_agent"]
    headers = {
        "User-Agent": f"{ua['project_name']} (Contact: {ua['email']}; Website: {ua['website']})"
    }
    admin_level = ADM1_SPECIALS.get(country, 4)
    query = f"""
        [out:json][timeout:{api["timeout"]}];
        area["ISO3166-1"="{country}"]->.searchArea;
        (
        relation["boundary"="administrative"]["admin_level"={admin_level}]["name"](area.searchArea);
        );
        out body geom;
    """

    for attempt in range(1, api["max_tries"] + 1):
        try:
            logger.info(f"Fetching OSM admin-1 boundaries for {country} (attempt {attempt})")
            response = requests.post(
                api["url"], data=query, headers=headers, timeout=api["timeout"]
            )
            response.raise_for_status()
            data = response.json()
            break
        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            logger.warning(f"Overpass request for {country} failed: {e}")
            if attempt == api["max_tries"]:
                raise
            time.sleep(15 * attempt)

    if not data.get("elements"):
        raise ValueError(
            f"Overpass returned no admin_level={admin_level} relations for {country}."
        )
    logger.info(f"Retrieved {len(data['elements'])} admin-1 relations for {country}.")
    with open(snakemake.output.json, "w") as f:
        json.dump(data, f)
