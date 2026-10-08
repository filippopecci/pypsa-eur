# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

# Weather-driven profiles for `profile_only.countries`, which are outside the modelled
# network (one region per country). Every output lives under resources("profile_only/")
# and feeds no other rule, so the main workflow is identical with this enabled or not.

import re

PROFILE_ONLY = config.get("profile_only", {})
PROFILE_ONLY_CUTOUT_DIR = "data/cutout/profile_only"


def _profile_only_cutout_names(w, setting="default"):
    """Profile-only cutout name(s) mirroring the main-workflow cutout(s) behind `setting`."""
    if setting == "default":
        setting = config_provider("atlite", "default_cutout")(w)
    rename = config_provider("profile_only", "cutout", "rename")(w)

    def _rename(name):
        for old, new in rename.items():
            name = name.replace(old, new)
        return name

    if isinstance(setting, list):
        return [_rename(n) for n in setting]
    return _rename(setting)


def input_profile_only_cutout(w, setting="default"):
    names = _profile_only_cutout_names(w, setting)
    if isinstance(names, list):
        return [f"{PROFILE_ONLY_CUTOUT_DIR}/{n}.nc" for n in names]
    return f"{PROFILE_ONLY_CUTOUT_DIR}/{names}.nc"


def input_profile_only_cutout_grid(w):
    # All profile-only cutouts share one grid, so time-independent rules need only one.
    cutout = input_profile_only_cutout(w)
    return cutout[0] if isinstance(cutout, list) else cutout


def profile_only_cutout_params(w):
    """atlite.Cutout arguments for one profile-only cutout; the year(s) come from its name."""
    cfg = config["profile_only"]["cutout"]
    years = re.findall(r"(?<!\d)\d{4}(?!\d)", w.cutout)
    if not years:
        raise ValueError(
            f"Cannot infer the weather year(s) of profile-only cutout '{w.cutout}' from its name."
        )
    params = {key: cfg[key] for key in ("module", "x", "y", "dx", "dy")}
    params["time"] = [years[0], years[-1]]
    # A name saying "sarah3" must not hide an ERA5-only cutout, or vice versa.
    if ("sarah" in w.cutout) != ("sarah" in params["module"]):
        raise ValueError(
            f"Profile-only cutout '{w.cutout}' does not match profile_only.cutout.module "
            f"{params['module']}; adjust profile_only.cutout.rename."
        )
    if "sarah" in params["module"]:
        if not cfg.get("sarah_dir"):
            raise ValueError(
                "profile_only.cutout.sarah_dir must point to SARAH-3 data when "
                "profile_only.cutout.module includes 'sarah'."
            )
        params["sarah_dir"] = cfg["sarah_dir"]
    params["prepare_kwargs"] = dict(cfg.get("prepare_kwargs") or {})
    return {w.cutout: params}


def _primary_url(name):
    """URL of the latest supported primary source of `name` in data/versions.csv."""
    versions = load_data_versions(workflow.source_path("../data/versions.csv"))
    row = versions.loc[
        (versions["dataset"] == name)
        & (versions["source"] == "primary")
        & versions["supported"]
        & versions["latest"]
    ]
    return row["url"].squeeze()


def _profile_only_crop_bounds(margin=1.0):
    """(x0, x1, y0, y1) of the profile-only cutout plus `margin` degrees."""
    (x0, x1), (y0, y1) = PROFILE_ONLY["cutout"]["x"], PROFILE_ONLY["cutout"]["y"]
    return x0 - margin, x1 + margin, y0 - margin, y1 + margin


def profile_only_renewable(w):
    """`renewable` settings with the `profile_only.renewable` overrides applied, e.g. resource_classes."""
    renewable = copy.deepcopy(config_provider("renewable")(w))
    update_config(renewable, config_provider("profile_only", "renewable")(w) or {})
    return renewable


def input_profile_only_regions(w):
    if w.technology in ("onwind", "solar", "solar-hsat"):
        return resources("profile_only/regions_onshore_country.geojson")
    return resources("profile_only/regions_offshore_country.geojson")


if PROFILE_ONLY.get("enable", False):

    PROFILE_ONLY_TECHNOLOGIES = "|".join(
        re.escape(t) for t in PROFILE_ONLY["technologies"]
    )

    localrules:
        build_profile_only,

    rule build_cutout_profile_only:
        message:
            "Building profile-only cutout {wildcards.cutout}"
        params:
            cutouts=profile_only_cutout_params,
        output:
            cutout=PROFILE_ONLY_CUTOUT_DIR + "/{cutout}.nc",
        log:
            "logs/build_cutout_profile_only/{cutout}.log",
        benchmark:
            "benchmarks/build_cutout_profile_only/{cutout}"
        wildcard_constraints:
            cutout=r"[^/]+",
        threads: config["atlite"].get("nprocesses", 4)
        resources:
            mem_mb=config["atlite"].get("nprocesses", 4) * 1000,
        script:
            scripts("build_cutout.py")

    rule retrieve_osm_boundaries_profile_only:
        message:
            "Retrieving OSM admin boundaries for profile-only country {wildcards.country}"
        params:
            overpass_api=config_provider("overpass_api"),
        output:
            json="data/osm_boundaries/profile_only/{country}_adm1.json",
        log:
            "logs/retrieve_osm_boundaries_profile_only_{country}.log",
        wildcard_constraints:
            country="|".join(PROFILE_ONLY["countries"]),
        threads: 1
        script:
            scripts("retrieve_osm_boundaries_profile_only.py")

    # The main-workflow copies of WorldPop and GEBCO are cropped to Europe even from
    # their primary sources, so the profile-only countries get their own crops.
    rule retrieve_population_count_profile_only:
        message:
            "Retrieving population count data for profile-only countries"
        params:
            bounds=_profile_only_crop_bounds(),
        input:
            tif=storage(_primary_url("population_count")),
        output:
            tif="data/population_count/profile_only/ppp_2019_1km_Aggregated.tif",
        log:
            "logs/retrieve_population_count_profile_only.log",
        retries: 2
        run:
            import rioxarray

            x0, x1, y0, y1 = params.bounds
            da = rioxarray.open_rasterio(input["tif"])
            da.rio.clip_box(minx=x0, miny=y0, maxx=x1, maxy=y1).rio.to_raster(
                output["tif"]
            )

    rule retrieve_gebco_profile_only:
        message:
            "Retrieving GEBCO bathymetry data for profile-only countries"
        params:
            bounds=_profile_only_crop_bounds(),
        input:
            nc=storage(PROFILE_ONLY["gebco_url"]),
        output:
            gebco="data/gebco/profile_only/GEBCO_2014_2D.nc",
        log:
            "logs/retrieve_gebco_profile_only.log",
        retries: 2
        run:
            import xarray as xr

            x0, x1, y0, y1 = params.bounds
            with xr.open_dataset(input["nc"]) as ds:
                ds.sel(lat=slice(y0, y1), lon=slice(x0, x1)).to_netcdf(output["gebco"])

    rule build_profile_only_shapes:
        message:
            "Building country, offshore and admin-1 shapes for profile-only countries"
        params:
            countries=config_provider("profile_only", "countries"),
            assign=config_provider("profile_only", "assign"),
            cutout=config_provider("profile_only", "cutout"),
        input:
            adm1=expand(
                "data/osm_boundaries/profile_only/{country}_adm1.json",
                country=PROFILE_ONLY["countries"],
            ),
            eez=ancient(rules.retrieve_eez.output["gpkg"]),
            population=rules.retrieve_population_count_profile_only.output["tif"],
        output:
            country_shapes=resources("profile_only/shapes_onshore.geojson"),
            offshore_shapes=resources("profile_only/shapes_offshore.geojson"),
            regions_onshore=resources("profile_only/regions_onshore_country.geojson"),
            regions_offshore=resources("profile_only/regions_offshore_country.geojson"),
            nuts3_shapes=resources("profile_only/shapes_adm1.geojson"),
        log:
            logs("build_profile_only_shapes.log"),
        threads: 1
        resources:
            mem_mb=8000,
        script:
            scripts("build_profile_only_shapes.py")

    rule build_powerplants_profile_only:
        message:
            "Building the power plant list for profile-only countries"
        params:
            countries=config_provider("profile_only", "countries"),
            powerplants_filter=config_provider("electricity", "powerplants_filter"),
        input:
            regions_onshore=resources("profile_only/regions_onshore_country.geojson"),
            regions_offshore=resources("profile_only/regions_offshore_country.geojson"),
        output:
            resources("profile_only/powerplants.csv"),
        log:
            logs("build_powerplants_profile_only.log"),
        threads: 1
        resources:
            mem_mb=5000,
        script:
            scripts("build_powerplants_profile_only.py")

    rule build_ship_raster_profile_only:
        message:
            "Building ship density raster for profile-only countries"
        input:
            ship_density=rules.retrieve_ship_raster.output["zip_file"],
            cutout=input_profile_only_cutout_grid,
        output:
            resources("profile_only/shipdensity_raster_cutout.tif"),
        log:
            logs("build_ship_raster_profile_only.log"),
        resources:
            mem_mb=5000,
        script:
            scripts("build_ship_raster.py")

    rule determine_availability_matrix_profile_only:
        message:
            "Determining profile-only availability matrix for {wildcards.technology}"
        params:
            renewable=profile_only_renewable,
        input:
            copernicus=rules.download_copernicus_land_cover.output["tif"],
            wdpa=lambda w: (
                rules.retrieve_wdpa.output["gpkg"]
                if profile_only_renewable(w)[w.technology]["natura"]
                else []
            ),
            wdpa_marine=lambda w: (
                rules.retrieve_wdpa_marine.output["gpkg"]
                if profile_only_renewable(w)[w.technology]["natura"]
                else []
            ),
            gebco=lambda w: (
                rules.retrieve_gebco_profile_only.output["gebco"]
                if (
                    profile_only_renewable(w)[w.technology].get("max_depth")
                    or profile_only_renewable(w)[w.technology].get("min_depth")
                )
                else []
            ),
            ship_density=lambda w: (
                resources("profile_only/shipdensity_raster_cutout.tif")
                if "ship_threshold" in profile_only_renewable(w)[w.technology]
                else []
            ),
            country_shapes=resources("profile_only/shapes_onshore.geojson"),
            offshore_shapes=resources("profile_only/shapes_offshore.geojson"),
            regions=input_profile_only_regions,
            cutout=input_profile_only_cutout_grid,
        output:
            availability_matrix=resources(
                "profile_only/availability_matrix_{technology}.nc"
            ),
        log:
            logs("determine_availability_matrix_profile_only_{technology}.log"),
        benchmark:
            benchmarks("determine_availability_matrix_profile_only_{technology}")
        wildcard_constraints:
            technology=PROFILE_ONLY_TECHNOLOGIES,
        threads: config["atlite"].get("nprocesses", 4)
        resources:
            mem_mb=config["atlite"].get("nprocesses", 4) * 5000,
        script:
            scripts("determine_availability_matrix_profile_only.py")

    rule build_renewable_profiles_profile_only:
        message:
            "Building profile-only renewable profiles for {wildcards.technology}"
        params:
            snapshots=config_provider("snapshots"),
            drop_leap_day=config_provider("enable", "drop_leap_day"),
            renewable=profile_only_renewable,
        input:
            availability_matrix=resources(
                "profile_only/availability_matrix_{technology}.nc"
            ),
            offshore_shapes=resources("profile_only/shapes_offshore.geojson"),
            distance_regions=resources("profile_only/regions_onshore_country.geojson"),
            resource_regions=input_profile_only_regions,
            cutout=lambda w: input_profile_only_cutout(
                w, profile_only_renewable(w)[w.technology]["cutout"]
            ),
        output:
            profile=resources("profile_only/profile_{technology}.nc"),
            class_regions=resources(
                "profile_only/regions_by_class_{technology}.geojson"
            ),
        log:
            logs("build_renewable_profile_profile_only_{technology}.log"),
        benchmark:
            benchmarks("build_renewable_profile_profile_only_{technology}")
        wildcard_constraints:
            technology=PROFILE_ONLY_TECHNOLOGIES,
        threads: config["atlite"].get("nprocesses", 4)
        resources:
            mem_mb=config["atlite"].get("nprocesses", 4) * 5000,
        script:
            scripts("build_renewable_profiles.py")

    rule build_population_layouts_profile_only:
        message:
            "Building profile-only population layouts"
        input:
            nuts3_shapes=resources("profile_only/shapes_adm1.geojson"),
            urban_percent=rules.retrieve_worldbank_urban_population.output["csv"],
            cutout=input_profile_only_cutout_grid,
        output:
            pop_layout_total=resources("profile_only/population_layout_total.nc"),
            pop_layout_urban=resources("profile_only/population_layout_urban.nc"),
            pop_layout_rural=resources("profile_only/population_layout_rural.nc"),
        log:
            logs("build_population_layouts_profile_only.log"),
        resources:
            mem_mb=20000,
        threads: 8
        script:
            scripts("build_population_layouts.py")

    rule build_daily_heat_demand_profile_only:
        message:
            "Building profile-only daily heat demand (heating degree days)"
        params:
            snapshots=config_provider("snapshots"),
            drop_leap_day=config_provider("enable", "drop_leap_day"),
        input:
            pop_layout=resources("profile_only/population_layout_total.nc"),
            regions_onshore=resources("profile_only/regions_onshore_country.geojson"),
            cutout=lambda w: input_profile_only_cutout(
                w, config_provider("sector", "heat_demand_cutout")(w)
            ),
        output:
            heat_demand=resources("profile_only/daily_heat_demand_total.nc"),
        resources:
            mem_mb=20000,
        threads: 8
        log:
            logs("build_daily_heat_demand_profile_only.log"),
        script:
            scripts("build_daily_heat_demand.py")

    rule build_daily_cooling_demand_profile_only:
        message:
            "Building profile-only daily cooling demand (cooling degree days)"
        params:
            snapshots=config_provider("snapshots"),
            drop_leap_day=config_provider("enable", "drop_leap_day"),
            threshold=config_provider("profile_only", "cooling_demand", "threshold"),
        input:
            pop_layout=resources("profile_only/population_layout_total.nc"),
            regions_onshore=resources("profile_only/regions_onshore_country.geojson"),
            cutout=lambda w: input_profile_only_cutout(
                w, config_provider("sector", "heat_demand_cutout")(w)
            ),
        output:
            cooling_demand=resources("profile_only/daily_cooling_demand_total.nc"),
        resources:
            mem_mb=20000,
        threads: 8
        log:
            logs("build_daily_cooling_demand_profile_only.log"),
        script:
            scripts("build_daily_cooling_demand.py")

    rule build_hourly_heat_demand_profile_only:
        message:
            "Building profile-only hourly heat demand from daily demand"
        params:
            snapshots=config_provider("snapshots"),
            drop_leap_day=config_provider("enable", "drop_leap_day"),
            sector=config_provider("sector"),
        input:
            heat_profile="data/heat_load_profile_BDEW.csv",
            heat_demand=resources("profile_only/daily_heat_demand_total.nc"),
        output:
            heat_demand=resources("profile_only/hourly_heat_demand_total.nc"),
            heat_dsm_profile=resources(
                "profile_only/residential_heat_dsm_profile_total.csv"
            ),
        resources:
            mem_mb=2000,
        threads: 8
        log:
            logs("build_hourly_heat_demand_profile_only.log"),
        script:
            scripts("build_hourly_heat_demand.py")

    rule build_temperature_profiles_profile_only:
        message:
            "Building profile-only air and soil temperature profiles"
        params:
            snapshots=config_provider("snapshots"),
            drop_leap_day=config_provider("enable", "drop_leap_day"),
        input:
            pop_layout=resources("profile_only/population_layout_total.nc"),
            regions_onshore=resources("profile_only/regions_onshore_country.geojson"),
            cutout=lambda w: input_profile_only_cutout(
                w, config_provider("sector", "heat_demand_cutout")(w)
            ),
        output:
            temp_soil=resources("profile_only/temp_soil_total.nc"),
            temp_air=resources("profile_only/temp_air_total.nc"),
        resources:
            mem_mb=20000,
        threads: 8
        log:
            logs("build_temperature_profiles_profile_only.log"),
        script:
            scripts("build_temperature_profiles.py")

    rule build_solar_thermal_profiles_profile_only:
        message:
            "Building profile-only solar thermal profiles"
        params:
            snapshots=config_provider("snapshots"),
            drop_leap_day=config_provider("enable", "drop_leap_day"),
            solar_thermal=config_provider("solar_thermal"),
        input:
            pop_layout=resources("profile_only/population_layout_total.nc"),
            regions_onshore=resources("profile_only/regions_onshore_country.geojson"),
            cutout=lambda w: input_profile_only_cutout(
                w, config_provider("solar_thermal", "cutout")(w)
            ),
        output:
            solar_thermal=resources("profile_only/solar_thermal_total.nc"),
        resources:
            mem_mb=20000,
        threads: 16
        log:
            logs("build_solar_thermal_profiles_profile_only.log"),
        script:
            scripts("build_solar_thermal_profiles.py")

    rule build_central_heating_temperature_profiles_profile_only:
        message:
            "Building profile-only central heating temperature profiles for {wildcards.planning_horizons}"
        params:
            max_forward_temperature_central_heating_baseyear=config_provider(
                "sector",
                "district_heating",
                "supply_temperature_approximation",
                "max_forward_temperature_baseyear",
            ),
            min_forward_temperature_central_heating_baseyear=config_provider(
                "sector",
                "district_heating",
                "supply_temperature_approximation",
                "min_forward_temperature_baseyear",
            ),
            return_temperature_central_heating_baseyear=config_provider(
                "sector",
                "district_heating",
                "supply_temperature_approximation",
                "return_temperature_baseyear",
            ),
            snapshots=config_provider("snapshots"),
            drop_leap_day=config_provider("enable", "drop_leap_day"),
            lower_threshold_ambient_temperature=config_provider(
                "sector",
                "district_heating",
                "supply_temperature_approximation",
                "lower_threshold_ambient_temperature",
            ),
            upper_threshold_ambient_temperature=config_provider(
                "sector",
                "district_heating",
                "supply_temperature_approximation",
                "upper_threshold_ambient_temperature",
            ),
            rolling_window_ambient_temperature=config_provider(
                "sector",
                "district_heating",
                "supply_temperature_approximation",
                "rolling_window_ambient_temperature",
            ),
            relative_annual_temperature_reduction=config_provider(
                "sector",
                "district_heating",
                "supply_temperature_approximation",
                "relative_annual_temperature_reduction",
            ),
            energy_totals_year=config_provider("energy", "energy_totals_year"),
        input:
            temp_air_total=resources("profile_only/temp_air_total.nc"),
            regions_onshore=resources("profile_only/regions_onshore_country.geojson"),
        output:
            central_heating_forward_temperature_profiles=resources(
                "profile_only/central_heating_forward_temperature_profiles_{planning_horizons}.nc"
            ),
            central_heating_return_temperature_profiles=resources(
                "profile_only/central_heating_return_temperature_profiles_{planning_horizons}.nc"
            ),
        resources:
            mem_mb=20000,
        log:
            logs(
                "build_central_heating_temperature_profiles_profile_only_{planning_horizons}.log"
            ),
        script:
            scripts("build_central_heating_temperature_profiles/run.py")

    rule build_cop_profiles_profile_only:
        message:
            "Building profile-only COP profiles for {wildcards.planning_horizons}"
        params:
            heat_pump_sink_T_decentral_heating=config_provider(
                "sector", "heat_pump_sink_T_individual_heating"
            ),
            heat_source_cooling_central_heating=config_provider(
                "sector", "district_heating", "heat_source_cooling"
            ),
            heat_pump_cop_approximation_central_heating=config_provider(
                "sector", "district_heating", "heat_pump_cop_approximation"
            ),
            # Only ambient sources: the others need inputs that exist only for modelled countries.
            heat_pump_sources=lambda w: {
                system: [s for s in sources if s in ("air", "ground")]
                for system, sources in config_provider("sector", "heat_pump_sources")(
                    w
                ).items()
                if any(s in ("air", "ground") for s in sources)
            },
            limited_heat_sources=config_provider(
                "sector", "district_heating", "limited_heat_sources"
            ),
            snapshots=config_provider("snapshots"),
        input:
            temp_air=resources("profile_only/temp_air_total.nc"),
            temp_ground=resources("profile_only/temp_soil_total.nc"),
            central_heating_forward_temperature_profiles=resources(
                "profile_only/central_heating_forward_temperature_profiles_{planning_horizons}.nc"
            ),
            central_heating_return_temperature_profiles=resources(
                "profile_only/central_heating_return_temperature_profiles_{planning_horizons}.nc"
            ),
            regions_onshore=resources("profile_only/regions_onshore_country.geojson"),
        output:
            cop_profiles=resources("profile_only/cop_profiles_{planning_horizons}.nc"),
        resources:
            mem_mb=20000,
        log:
            logs("build_cop_profiles_profile_only_{planning_horizons}.log"),
        script:
            scripts("build_cop_profiles/run.py")

    rule build_profile_only:
        message:
            "Collecting profile-only outputs"
        input:
            expand(
                resources("profile_only/profile_{technology}.nc"),
                technology=PROFILE_ONLY["technologies"],
                run=config["run"]["name"],
            ),
            expand(
                resources("profile_only/{fn}"),
                fn=[
                    "powerplants.csv",
                    "population_layout_total.nc",
                    "daily_heat_demand_total.nc",
                    "daily_cooling_demand_total.nc",
                    "hourly_heat_demand_total.nc",
                    "temp_air_total.nc",
                    "solar_thermal_total.nc",
                    "regions_onshore_country.geojson",
                    "regions_offshore_country.geojson",
                ],
                run=config["run"]["name"],
            ),
            expand(
                resources("profile_only/cop_profiles_{planning_horizons}.nc"),
                planning_horizons=config["scenario"]["planning_horizons"],
                run=config["run"]["name"],
            ),
