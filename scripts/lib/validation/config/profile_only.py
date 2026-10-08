# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

"""
Profile-only countries configuration.

Weather-driven profiles for countries outside the modelled network (e.g. North Africa),
built with the same scripts and `renewable` settings as the modelled countries. Nothing in
the main workflow reads these outputs; see `rules/build_profile_only.smk`.
"""

from typing import Any

from pydantic import BaseModel, Field

from scripts.lib.validation.config._base import ConfigModel


class _ProfileOnlyCutoutConfig(ConfigModel):
    """Configuration for `profile_only.cutout` settings."""

    rename: dict[str, str] = Field(
        default_factory=lambda: {"europe": "northafrica"},
        description="String replacements turning each main-workflow cutout name into its profile-only counterpart (same year). For ERA5 only, add `sarah3-era5: era5`.",
    )
    module: list[str] = Field(
        default_factory=lambda: ["sarah", "era5"],
        description="atlite modules of the profile-only cutouts; must agree with the cutout name ('sarah3' or not).",
    )
    sarah_dir: str | None = Field(
        None,
        description="Directory with SARAH-3 SIS and SID 30-min instantaneous files, which atlite cannot download. Required when `module` includes 'sarah'.",
    )
    x: list[float] = Field(
        default_factory=lambda: [-18.0, 38.4],
        description="Longitude range of the cutout, on the 0.3 degree lattice of the pre-built European cutouts.",
    )
    y: list[float] = Field(
        default_factory=lambda: [18.0, 39.6],
        description="Latitude range of the cutout, on the 0.3 degree lattice of the pre-built European cutouts.",
    )
    dx: float = Field(0.3, description="Longitude resolution of the cutout.")
    dy: float = Field(0.3, description="Latitude resolution of the cutout.")
    prepare_kwargs: dict[str, Any] = Field(
        default_factory=lambda: {"monthly_requests": True},
        description="Keyword arguments passed to `atlite.Cutout.prepare`.",
    )


class _ProfileOnlyCoolingDemandConfig(ConfigModel):
    """Configuration for `profile_only.cooling_demand` settings."""

    threshold: float = Field(
        23.0,
        description="Daily mean ambient temperature (degC) above which cooling degree days accrue.",
    )


class ProfileOnlyConfig(BaseModel):
    """Configuration for `profile_only` settings."""

    enable: bool = Field(False, description="Build the profile-only outputs.")
    countries: list[str] = Field(
        default_factory=lambda: ["MA", "DZ", "TN", "LY", "EG"],
        description="ISO2 codes of the profile-only countries; each becomes one region.",
    )
    assign: dict[str, str] = Field(
        default_factory=dict,
        description="Extra EEZ codes merged into a country offshore. {EH: MA} also keeps Western Sahara onshore, which is otherwise clipped from MA at 27°40'N.",
    )
    technologies: list[str] = Field(
        default_factory=lambda: [
            "onwind",
            "solar",
            "solar-hsat",
            "offwind-ac",
            "offwind-dc",
            "offwind-float",
        ],
        description="Renewable technologies to build profiles for.",
    )
    renewable: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Overrides of `renewable` for these countries only, keyed the same way, e.g. {onwind: {resource_classes: 4}}.",
    )
    cutout: _ProfileOnlyCutoutConfig = Field(
        default_factory=_ProfileOnlyCutoutConfig,
        description="Profile-only cutouts, one per main-workflow cutout.",
    )
    gebco_url: str = Field(
        "https://dap.ceda.ac.uk/bodc/gebco/global/gebco_2014/30_second_grid/GEBCO_2014_2D.nc",
        description="Global GEBCO_2014 30 arc-second grid (the European archive copy stops at 32N).",
    )
    cooling_demand: _ProfileOnlyCoolingDemandConfig = Field(
        default_factory=_ProfileOnlyCoolingDemandConfig,
        description="Cooling degree day settings.",
    )
