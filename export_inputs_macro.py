import xarray as xr
import pypsa
import json
import os
import shutil
import glob
import pandas as pd
import numpy as np
import math
import yaml

from collections.abc import Iterable
from typing import Any
def flatten(t: Iterable[Any]) -> str:
    return " ".join(map(str, t))

def get_unique_id(base_id, existing_ids):
    if base_id not in existing_ids:
        return base_id
    i = 1
    while f"{base_id}_{i}" in existing_ids:
        i += 1
    return f"{base_id}_{i}"

#####################################################################################
####### Load case files and prepare Macro input folders #######
number_of_zones = "39"
co2_cap = 0.0
co2_price = None
target_year = 2050
#case_name = "base_s_"+ number_of_zones+ "___2050"
case_name = "base_s_"+ number_of_zones+ "_elec_"

destination = "/Users/fpecci/Code/MacroMed-Data/" + case_name + "_CO2_cap_" + str(co2_cap) + "_price_" + str(co2_price).lower()

if os.path.exists(case_name):
    shutil.rmtree(case_name)

subfolders = ["system", "assets", "settings","extra_input_data"]

for subfolder in subfolders:
    os.makedirs(os.path.join(case_name, subfolder), exist_ok=True)

system_data ={
    "case": [
        {
            "commodities": {
                "path": "system/commodities.json"
            },
            "locations": {
                "path": "system/locations.json"
            },
            "settings": {
                "path": "settings/macro_settings.json"
            },
            "assets": {
                "path": "assets"
            },
            "time_data": {
                "path": "system/time_data.json"
            },
            "nodes": {
                "path": "system/nodes.json"
            }
        }
    ],
    "settings": {
        "path": "settings/case_settings.json"
    }
}

with open(case_name + "/system_data.json", "w") as f:
    json.dump(system_data, f, indent=4)

n = pypsa.Network("resources/networks/" + case_name + ".nc")

config_dict = dict(n.meta)
period_length = n.snapshot_weightings.objective.sum()/8760

with open(os.path.join(case_name, "extra_input_data", "case_config.json"), "w") as f:
    json.dump(config_dict, f, indent=4)

n.export_to_netcdf(os.path.join(case_name, "extra_input_data", case_name + ".nc"));

#####################################################################################
####### Load and save in extra_input_data the CSV file with power plants data #######

powerplants_file = glob.glob("resources/powerplants_s*")[0]
powerplants = pd.read_csv(powerplants_file, index_col=[0, 1]).sort_index()
powerplants.to_csv(os.path.join(case_name, "extra_input_data", "powerplants.csv"))

#####################################################################################
####### Load and process costs #######

costs_file = glob.glob("resources/costs_*")[0]

# Copy marginal_cost and capital_cost for backward compatibility
for key in ("marginal_cost", "capital_cost"):
    if key in config_dict["costs"]:
        config_dict["costs"]["overwrites"][key] = config_dict["costs"][key]


costs = pd.read_csv(costs_file, index_col=[0, 1]).sort_index()

costs.loc[costs.unit.str.contains("/kW"), "value"] *= 1e3
costs.loc[costs.unit.str.contains("/GW"), "value"] /= 1e3

costs.unit = costs.unit.str.replace("/kW", "/MW")
costs.unit = costs.unit.str.replace("/GW", "/MW")

costs = costs.value.unstack(level=1).groupby("technology").sum(min_count=1)
costs = costs.fillna(config_dict["costs"]["fill_values"])

for attr in ("investment", "lifetime", "FOM", "VOM", "efficiency", "fuel"):
        overwrites = config_dict["costs"]["overwrites"].get(attr)
        if overwrites is not None:
            overwrites = pd.Series(overwrites)
            costs.loc[overwrites.index, attr] = overwrites

costs["investment_cost"] = costs["investment"]
costs["fixed_om_cost"] = costs["investment"] * costs["FOM"]/100
costs["variable_om_cost"] = costs["VOM"]

costs.at["OCGT", "fuel"] = costs.at["gas", "fuel"]
costs.at["CCGT", "fuel"] = costs.at["gas", "fuel"]
costs.at["OCGT", "CO2 intensity"] = costs.at["gas", "CO2 intensity"]
costs.at["CCGT", "CO2 intensity"] = costs.at["gas", "CO2 intensity"]

costs.at["solar", "investment_cost"] = costs.at["solar-utility", "investment_cost"]
costs = costs.rename({"solar-utility single-axis tracking": "solar-hsat"})

costs.at["offwind-ac", "variable_om_cost"] = costs.at["offwind", "variable_om_cost"]
costs.at["offwind-dc", "variable_om_cost"] = costs.at["offwind", "variable_om_cost"]

if math.isnan(costs.at["offwind-ac", "discount rate"]):
    costs.at["offwind-ac", "discount rate"] = costs.at["offwind", "discount rate"]
if math.isnan(costs.at["offwind-dc", "discount rate"]):
    costs.at["offwind-dc", "discount rate"] = costs.at["offwind", "discount rate"]
if math.isnan(costs.at["offwind-float", "discount rate"]):
    costs.at["offwind-float", "discount rate"] = costs.at["offwind", "discount rate"]

if math.isnan(costs.at["offwind-ac", "lifetime"]):
    costs.at["offwind-ac", "lifetime"] = costs.at["offwind", "lifetime"]
if math.isnan(costs.at["offwind-dc", "lifetime"]):
    costs.at["offwind-dc", "lifetime"] = costs.at["offwind", "lifetime"]
if math.isnan(costs.at["offwind-float", "lifetime"]):
    costs.at["offwind-float", "lifetime"] = costs.at["offwind", "lifetime"]


overwrites = config_dict["costs"]["overwrites"]["marginal_cost"]
if overwrites is not None:
    overwrites = pd.Series(overwrites)
    idx = overwrites.index.intersection(costs.index)
    costs.loc[idx, "variable_om_cost"] = overwrites.loc[idx]

if "offwind" in overwrites.index:
    costs.at["offwind-ac", "variable_om_cost"] = costs.at["offwind", "variable_om_cost"]
    costs.at["offwind-dc", "variable_om_cost"] = costs.at["offwind", "variable_om_cost"]
    costs.at["offwind-float", "variable_om_cost"] = costs.at["offwind", "variable_om_cost"]
    
if "solar" in overwrites.index:
    costs.at["solar-hsat", "variable_om_cost"] = costs.at["solar", "variable_om_cost"]


costs.to_csv(os.path.join(case_name, "extra_input_data", "techno_economic_assumptions.csv"))

#####################################################################################
####### Load and save fuel prices #######

monthly_fuel_prices = pd.read_csv("resources/monthly_fuel_price.csv", index_col=0, header=0, parse_dates=True)
fuel_prices = monthly_fuel_prices.reindex(n.snapshots).ffill()
fuel_prices.to_csv(os.path.join(case_name, "system", "fuel_prices.csv"))

#####################################################################################
####### Define commodity dictionary and save it #######
commodity_dict ={
    "commodities": [
        "Electricity",
        "NaturalGas",
        "CO2",
        "Biomass",
        "Uranium",
        "Coal",
        "LiquidFuels",
        {"name": "Lignite", "acts_like": "Coal"},
        {"name": "Oil", "acts_like": "LiquidFuels"},
        {"name": "Geothermal", "acts_like": "Uranium"},
    ]
}

with open(case_name + "/system/commodities.json", "w") as f:
    json.dump(commodity_dict, f, indent=4)

timedata ={
    "HoursPerSubperiod": {},
    "HoursPerTimeStep": {},
    "NumberOfSubperiods": 52,
    "TotalHoursModeled": 8760
}
for c in commodity_dict["commodities"]:
    if isinstance(c, str):
        name = c
        timedata["HoursPerSubperiod"][name] = 168
        timedata["HoursPerTimeStep"][name] = 1

with open(case_name + "/system/time_data.json", "w") as f:
    json.dump(timedata, f, indent=4)   

case_settings = {
    "SolutionAlgorithm": "Monolithic"
}
macro_settings = {
  "ConstraintScaling":True,
  "WriteSubcommodities":True,
  "OutputLayout": "wide"
}
with open(case_name + "/settings/case_settings.json", "w") as f:
    json.dump(case_settings, f, indent=4)
with open(case_name + "/settings/macro_settings.json", "w") as f:
    json.dump(macro_settings, f, indent=4)
#####################################################################################
####### Define nodes #######

bus_names = n.buses.index.to_list()

ac_buses = n.buses[n.buses.carrier == "AC"]

h2_buses = n.buses[n.buses.carrier == "H2"]

nodes ={"nodes": []}

nodes["nodes"].append(
    {
        "type": "Electricity",
        "global_data":{
            "time_interval": "Electricity",
            "max_nsd": [
                1
            ],
            "price_nsd": [
                5000.0
            ],
            "constraints": {
                "BalanceConstraint": True,
                "MaxNonServedDemandConstraint": True,
                "MaxNonServedDemandPerSegmentConstraint": True
            }
        },
        "instance_data": []
    },
)

for name in ac_buses.index.to_list():
    node_instance =  {
                            "id": "elec_" + name,
                            "demand": {
                                "timeseries": {
                                    "path": "system/demand.csv",
                                    "header": name,
                                }
                            }
                        }
    nodes["nodes"][0]["instance_data"].append(node_instance)

emission_constraints = {}
rhs_policy = {}
price_unmet_policy = {}
if co2_cap is not None:
    emission_constraints = { "CO2CapConstraint": True }
    rhs_policy = {"CO2CapConstraint": co2_cap}
    if co2_price is not None:
        price_unmet_policy = {"CO2CapConstraint": co2_price}


nodes["nodes"].append(
    {
            "type": "CO2",
            "global_data":{
                "time_interval": "CO2",
            },
            "instance_data": [
                {
                    "id": "co2_sink",
                    "constraints": emission_constraints,
                    "rhs_policy": rhs_policy,
                    "price_unmet_policy": price_unmet_policy,
                }   
            ]
    }
)

nodes["nodes"].append(
    {
            "type": "Coal",
            "global_data":{
                "time_interval": "Coal",
            },
            "instance_data": [
                {
                    "id": "coal_source",
                    "price": {
                        "timeseries": {
                            "path": "system/fuel_prices.csv",
                            "header": "coal"
                        }
                    }
                }   
            ]
    }
)

nodes["nodes"].append(
    {
            "type": "Lignite",
            "global_data":{
                "time_interval": "Lignite",
            },
            "instance_data": [
                {
                    "id": "lignite_source",
                    "price": {
                        "timeseries": {
                            "path": "system/fuel_prices.csv",
                            "header": "lignite"
                        }
                    }
                }   
            ]
    }
)

nodes["nodes"].append(
    {
            "type": "Oil",
            "global_data":{
                "time_interval": "Oil",
            },
            "instance_data": [
                {
                    "id": "oil_source",
                    "price": {
                        "timeseries": {
                            "path": "system/fuel_prices.csv",
                            "header": "oil"
                        }
                    }
                }   
            ]
    }
)

nodes["nodes"].append(
    {
            "type": "Biomass",
            "global_data":{
                "time_interval": "Biomass",
            },
            "instance_data": [
                {
                    "id": "biomass_source",
                    "price": [costs["fuel"]["biomass"]]
                }   
            ]
    }
)

nodes["nodes"].append(
    {
            "type": "Uranium",
            "global_data":{
                "time_interval": "Uranium",
            },
            "instance_data": [
                {
                    "id": "uranium_source",
                    "price": [costs["fuel"]["uranium"]]
                }   
            ]
    }
)

nodes["nodes"].append(
    {
            "type": "NaturalGas",
            "global_data":{
                "time_interval": "NaturalGas",
            },
            "instance_data": [
                {
                    "id": "ng_source",
                    "price": {
                        "timeseries": {
                            "path": "system/fuel_prices.csv",
                            "header": "gas"
                        }
                    }
                }   
            ]
    }
)

nodes["nodes"].append(
    {
            "type": "Geothermal",
            "global_data":{
                "time_interval": "Geothermal",
            },
            "instance_data": [
                {
                    "id": "geothermal_source",
                    "price": [costs["fuel"]["geothermal"]]
                }   
            ]
    }
)

nodes["nodes"].append(
    {
            "type": "Electricity",
            "global_data":{
                "time_interval": "Electricity",
            },
            "instance_data": [
                {
                    "id": "hydro_source",
                }   
            ]
    }
)

with open(case_name + "/system/nodes.json", "w") as f:
    json.dump(nodes, f, indent=4)

#####################################################################################
####### Export demand time series #######
def prepare_electricity_demand_for_sector_coupled_model(n:pypsa.Network):
    ##### NOTE: This function applies only to electricity-only networks, in sector-coupled networks this is already done wihtin PyPSA-Eur
    
    n.loads["carrier"] = "electricity"
    n.loads.rename(lambda x: x.strip(), inplace=True)
    n.loads_t.p_set.rename(lambda x: x.strip(), axis=1, inplace=True)

    heat_demand_shape = (
        xr.open_dataset("resources/hourly_heat_demand_total_base_s_"+number_of_zones+".nc").to_dataframe().unstack(level=1)
    )
    pop_weighted_energy_totals = ( 
    pd.read_csv("resources/pop_weighted_energy_totals_s_"+number_of_zones+".csv", index_col=0)
    )
    pop_weighted_heat_totals = (
        pd.read_csv("resources/pop_weighted_heat_totals_s_"+number_of_zones+".csv", index_col=0)
    )
    pop_weighted_energy_totals.update(pop_weighted_heat_totals)
    industrial_demand = pd.read_csv("resources/industrial_energy_demand_base_s_"+number_of_zones+"_" + target_year + ".csv",index_col=0)*1e6
    electric_heat_supply = {}
    for sector in ["residential", "services"]:
        for use in ["water", "space"]:
            name = f"{sector} {use}"
            print(name)
            electric_heat_supply[name] =  (
                heat_demand_shape[name] / heat_demand_shape[name].sum()
            ).multiply(pop_weighted_energy_totals[f"electricity {sector} {use}"]) * 1e6
    electric_heat_supply = pd.concat(electric_heat_supply, axis=1)

    electric_nodes = n.loads.index[n.loads.carrier == "electricity"]
    
    n.loads_t.p_set[electric_nodes] = (
        n.loads_t.p_set[electric_nodes]
        - electric_heat_supply.T.groupby(level=1).sum().T[electric_nodes]
    )

    for ct in n.buses.country.dropna().unique():
        loads_i = n.loads.index[
                    (n.loads.index.str[:2] == ct) & (n.loads.carrier == "electricity")
                ]
        factor = (
                    1
                    - industrial_demand.loc[loads_i, "current electricity"].sum()
                    / n.loads_t.p_set[loads_i].sum().sum()
                )
        n.loads_t.p_set[loads_i] *= factor

    n.loads_t.p_set.loc[:, n.loads.carrier == "electricity"] *= n.meta["sector"]["transmission_efficiency"]["electricity distribution grid"]["efficiency_static"]

n.loads_t["p_set"].to_csv(case_name + "/system/demand.csv")

#####################################################################################
####### Export generator availability time series #######
n.generators_t["p_max_pu"].to_csv(case_name + "/system/generators_availability.csv")

#####################################################################################
####### Export hydro inflow time series #######
n.storage_units_t["inflow"].to_csv(case_name + "/system/hydro_inflow.csv")

#####################################################################################
####### Define power lines #######
powerlines = {
    "line": [
        {
            "type": "TransmissionLink",
            "global_data": {
                "edges": {
                    "transmission_edge": {
                        "commodity": "Electricity",
                        "has_capacity": True,
                        "constraints": {
                            "CapacityConstraint": True,
                            "MinCapacityConstraint": True,
                            "MaxCapacityConstraint": True
                        }
                    }
                }
            },
            "instance_data": []
        }
    ]
}
existing_ids = set()

for idx, row in n.lines.iterrows():

    temp_id = row.bus0 + "_to_" + row.bus1

    line_id = get_unique_id(temp_id, existing_ids)

    existing_ids.add(line_id)

    powerlines["line"][0]["instance_data"].append(
        {
            "id": line_id,
            "edges": {
                "transmission_edge": {
                    "start_vertex": "elec_" + row.bus0,
                    "end_vertex": "elec_" + row.bus1,
                    "unidirectional": False,
                    "can_expand": row.s_nom_extendable,
                    "can_retire": row.s_nom > row.s_nom_min,
                    "capital_recovery_period": round(costs.at["HVAC overhead", "lifetime"]),
                    "wacc": costs.at["HVAC overhead", "discount rate"],
                    "lifetime": round(costs.at["HVAC overhead", "lifetime"]),
                    "distance": row.length,
                    "existing_capacity": row.s_nom,
                    "capacity_size": row.s_nom_mod if row.s_nom_mod != 0 else 1.0,  # Ensure capacity size is at least 1.0
                    "max_capacity": row.s_nom_max,
                    "min_capacity": row.s_nom_min,
                    "availability": [row.s_max_pu],
                    "loss_fraction": 0.02, #mid point of transmission losses according to: https://www.ceer.eu/publication/3rd-ceer-report-on-power-losses/
                    "fixed_om_cost": row.length * config_dict["lines"]["length_factor"] * costs.at["HVAC overhead", "fixed_om_cost"],
                    "investment_cost": row.length * config_dict["lines"]["length_factor"] * costs.at["HVAC overhead", "investment_cost"]
                }
            }
        }
    )

with open(case_name + "/assets/power_lines.json", "w") as f:
    json.dump(powerlines, f, indent=4)

#####################################################################################
####### Define DC links #######

dclinks = {
    "dclink": [
        {
            "type": "TransmissionLink",
            "global_data": {
                "edges": {
                    "transmission_edge": {
                        "commodity": "Electricity",
                        "has_capacity": True,
                        "constraints": {
                            "CapacityConstraint": True,
                            "MinCapacityConstraint": True,
                            "MaxCapacityConstraint": True
                        }
                    }
                }
            },
            "instance_data": []
        }
    ]
}
existing_ids = set()
for idx, row in n.links[n.links.carrier == "DC"].iterrows():

    temp_id = "DC_link_" + row.bus0 + "_to_" + row.bus1

    link_id = get_unique_id(temp_id, existing_ids)

    existing_ids.add(link_id)

    dclinks["dclink"][0]["instance_data"].append(
        {
            "id": link_id,
            "edges": {
                "transmission_edge": {
                    "start_vertex": "elec_" + row.bus0,
                    "end_vertex": "elec_" + row.bus1,
                    "unidirectional": row.p_min_pu == 0,
                    "can_expand": row.p_nom_extendable,
                    "can_retire": row.p_nom > row.p_nom_min,
                    "capital_recovery_period": round(costs.at["HVDC overhead", "lifetime"]), 
                    "wacc": costs.at["HVDC overhead", "discount rate"], ### This parameter is the same for submarine links
                    "lifetime": round(costs.at["HVDC overhead", "lifetime"]), 
                    "distance": row.length,
                    "existing_capacity": row.p_nom,
                    "capacity_size": row.p_nom_mod if row.p_nom_mod != 0 else 1.0,  # Ensure capacity size is at least 1.0
                    "max_capacity": row.p_nom_max,
                    "min_capacity": row.p_nom_min,
                    "availability": [row.p_max_pu],
                    "loss_fraction": 1.0-row.efficiency,
                    "fixed_om_cost": row.length * config_dict["links"]["length_factor"] * ( (1.0 - row.underwater_fraction)* costs.at["HVDC overhead", "fixed_om_cost"] + row.underwater_fraction * costs.at["HVDC submarine", "fixed_om_cost"] ) + costs.at["HVDC inverter pair", "fixed_om_cost"],
                    "investment_cost": row.length * config_dict["links"]["length_factor"] * ( (1.0 - row.underwater_fraction)* costs.at["HVDC overhead", "investment_cost"] + row.underwater_fraction * costs.at["HVDC submarine", "investment_cost"] ) + costs.at["HVDC inverter pair", "investment_cost"],
                }
            }
        }
    )

with open(case_name + "/assets/dc_links.json", "w") as f:
    json.dump(dclinks, f, indent=4)

#####################################################################################
####### Define thermal generators #######

def build_thermal_power_plant(period_length,selected_generators, generators_with_timeseries, macro_asset_name, macro_commodity_type, macro_fuel_start_node, co2_intensity_tag, costs,uc_data = {}):
        
    thermal_power = {
    macro_asset_name: [
        {"type": "ThermalPower",
            "global_data": {
            "nodes": {},
            "transforms": {
                "timedata": macro_commodity_type,
                "constraints": {
                        "BalanceConstraint": True
                }
            },
            "edges" : {
                "elec_edge":{
                            "commodity": "Electricity",
                            "unidirectional": True,
                            "has_capacity": True,
                            "integer_decisions": False,
                        },
                "fuel_edge": {
                            "commodity": macro_commodity_type,
                            "unidirectional": True,
                            "has_capacity": False,
                            "start_vertex": macro_fuel_start_node,
                        },
                "co2_edge": {
                        "commodity": "CO2",
                        "unidirectional": True,
                        "has_capacity": False,
                        "end_vertex": "co2_sink"
                    }
                }
            },
            "instance_data": []
        }
    ]
}

    for idx, row in selected_generators:

        if row.name in generators_with_timeseries:
            temp_availability = {
            "timeseries": {
                            "path": "system/generators_availability.csv",
                            "header": row.name,
            }
        }
        else:
            temp_availability = [row.p_max_pu]
    
        constr_list = {
        "CapacityConstraint": True,
        }
        
        has_uc = False

        if row.carrier.lower() in uc_data.keys():
            has_uc = True
            gen_uc_data = uc_data[row.carrier.lower()]
            temp_ramp_down_fraction = gen_uc_data["ramp_down_fraction"]["powergenome"]
            temp_ramp_up_fraction = gen_uc_data["ramp_up_fraction"]["powergenome"]
            temp_min_flow_fraction = gen_uc_data["min_flow_fraction"]["eera"]
            temp_min_down_time = gen_uc_data["min_down_time"]["eera"]
            temp_min_up_time = gen_uc_data["min_up_time"]["eera"]
            temp_startup_cost = gen_uc_data["startup_cost"]["eera"]
            temp_startup_fuel_consumption = gen_uc_data["startup_fuel_consumption"]["eera"]
        else:
            temp_min_down_time = 0.0
            temp_min_up_time = 0.0
            temp_startup_cost = 0.0
            temp_startup_fuel_consumption = 0.0
            if (row.ramp_limit_up > 0) & (row.ramp_limit_up < 1):
                temp_ramp_down_fraction = row.ramp_limit_up
                if (row.ramp_limit_down > 0) & (row.ramp_limit_down < 1): 
                    temp_ramp_down_fraction = row.ramp_limit_down
                else:
                    temp_ramp_down_fraction = row.ramp_limit_up
            else:
                temp_ramp_up_fraction = 1.0
                temp_ramp_down_fraction = 1.0
    
        if (row.p_min_pu > 0) & (row.p_min_pu < 1):
            temp_min_flow_fraction = row.p_min_pu
        else:
            temp_min_flow_fraction = 0.0

        if row.p_nom_max < float('inf'):
            constr_list["MaxCapacityConstraint"] = True
            temp_max_cap = row.p_nom_max
        else:
            temp_max_cap = 0.0

        if (row.p_nom_min != row.p_nom) & (row.p_nom_min > 0):
            constr_list["MinCapacityConstraint"] = True

        if (temp_ramp_up_fraction > 0) & (temp_ramp_up_fraction < 1):
            constr_list["RampingLimitConstraint"] = True

        if temp_min_flow_fraction > 0:
            constr_list["MinFlowConstraint"] = True

        if temp_min_down_time > 1:
            constr_list["MinDownTimeConstraint"] = True

        if temp_min_up_time > 1:
            constr_list["MinUpTimeConstraint"] = True

        if row.lifetime == float('inf'):
            gen_lifetime = round(period_length.sum())
        else:
            gen_lifetime = round(row.lifetime)

        if row.p_nom_mod != 0:
            capacity_size = row.p_nom_mod
        elif row.num_units > 0:
            capacity_size = row.p_nom / row.num_units
        else:
            capacity_size = 1.0

        thermal_power[macro_asset_name][0]["instance_data"].append(
        {
            "id": row.carrier + "_" + row.bus + "_" + row.name,
            "transforms":{
                "emission_rate": costs.at[co2_intensity_tag, "CO2 intensity"], # this is given as [tonnes/MWh_fuel], see https://pypsa.readthedocs.io/en/latest/user-guide/components.html#carrier
                "fuel_consumption": 1/row.efficiency, # Note: efficiency is given as [MWh_elec/MWh_fuel], see https://pypsa.readthedocs.io/en/latest/user-guide/components.html#generator
            },
            "edges":{
                "elec_edge":{
                    "end_vertex": "elec_" + row.bus,
                    "can_retire": row.p_nom > row.p_nom_min,
                    "can_expand": row.p_nom_extendable,
                    "lifetime": gen_lifetime, ## TODO: This should be the remaining lifetime of the existing generator (how long until it retires)
                    "capital_recovery_period": round(costs.at[row.carrier, "lifetime"]),
                    "wacc": costs.at[row.carrier, "discount rate"],
                    "min_capacity": row.p_nom_min,
                    "max_capacity": temp_max_cap,
                    "capacity_size": capacity_size,
                    "existing_capacity": row.p_nom,
                    "availability": temp_availability,
                    "uc": has_uc,
                    "min_flow_fraction": temp_min_flow_fraction,
                    "ramp_up_fraction": temp_ramp_up_fraction,
                    "ramp_down_fraction": temp_ramp_down_fraction,
                    "startup_cost": temp_startup_cost,
                    "startup_fuel_consumption": temp_startup_fuel_consumption,
                    "fixed_om_cost": costs.at[row.carrier, "fixed_om_cost"],
                    "investment_cost": costs.at[row.carrier, "investment_cost"],
                    "variable_om_cost": costs.at[row.carrier, "variable_om_cost"],
                    "constraints": constr_list,
                },
            }
        }
    )
        
    return thermal_power

pypsa_uc_data = pd.read_csv("data/unit_commitment.csv", index_col=[0]).sort_index()
pypsa_uc_data = pypsa_uc_data.fillna(0.0)
   
# Check if the unit commitment data matches the generators in the network
for carrier in pypsa_uc_data.keys():
    for attr in pypsa_uc_data.index.tolist():
        bool_check = n.generators.loc[n.generators.carrier==carrier,attr].unique() == pypsa_uc_data.loc[attr,carrier]
        if not bool_check.all():
            print(f"Mismatch for {carrier} and {attr}: {bool_check}")
            print(f"data/unit_commitment.csv: {pypsa_uc_data.loc[attr, carrier]}")
            print(f"n.generators: {n.generators.loc[n.generators.carrier==carrier, attr].unique()}")

for carrier in n.generators.carrier.unique():
    if carrier not in pypsa_uc_data.keys():
        for attr in pypsa_uc_data.index.tolist():
            if not(np.isnan(n.generators.loc[n.generators.carrier==carrier, attr].unique()).all()):
                if (n.generators.loc[n.generators.carrier==carrier, attr].unique() != 1)and(n.generators.loc[n.generators.carrier==carrier, attr].unique() != 0):
                    print(f"Carrier {carrier} has {attr} = {n.generators.loc[n.generators.carrier==carrier, attr].unique()} that is not 0 or 1, but not in data/unit_commitment.csv")

with open("unit_commitment.yaml", "r") as f:
    uc_data = yaml.safe_load(f)
  
ngpower = build_thermal_power_plant(period_length,
                                    n.generators[(n.generators.carrier == "CCGT") | (n.generators.carrier == "OCGT")].iterrows(), 
                                    n.generators_t["p_max_pu"].keys(),
                                    "NaturalGasPower",
                                    "NaturalGas",
                                    "ng_source",
                                    "gas",
                                    costs,
                                    uc_data
)


with open(case_name + "/assets/naturalgas_power.json", "w") as f:
    json.dump(ngpower, f, indent=4)

coalpower = build_thermal_power_plant(period_length,
                                    n.generators[(n.generators.carrier == "coal")].iterrows(), 
                                    n.generators_t["p_max_pu"].keys(),
                                    "CoalPower",
                                    "Coal",
                                    "coal_source",
                                    "coal",
                                    costs,
                                    uc_data,
)

with open(case_name + "/assets/coal_power.json", "w") as f:
    json.dump(coalpower, f, indent=4)

lignitepower = build_thermal_power_plant(period_length,
                                    n.generators[(n.generators.carrier == "lignite")].iterrows(), 
                                    n.generators_t["p_max_pu"].keys(),
                                    "LignitePower",
                                    "Lignite",
                                    "lignite_source",
                                    "lignite",
                                    costs,
                                    uc_data,
)

with open(case_name + "/assets/lignite_power.json", "w") as f:
    json.dump(lignitepower, f, indent=4)

oilpower = build_thermal_power_plant(period_length,
                                    n.generators[(n.generators.carrier == "oil")].iterrows(), 
                                    n.generators_t["p_max_pu"].keys(),
                                    "OilPower",
                                    "Oil",
                                    "oil_source",
                                    "oil",
                                    costs,
                                    uc_data,
)

with open(case_name + "/assets/oil_power.json", "w") as f:
    json.dump(oilpower, f, indent=4)

biomasspower = build_thermal_power_plant(period_length,
                                    n.generators[(n.generators.carrier == "biomass")].iterrows(), 
                                    n.generators_t["p_max_pu"].keys(),
                                    "BiomassPower",
                                    "Biomass",
                                    "biomass_source",
                                    "biomass",
                                    costs,
                                    uc_data,
)

with open(case_name + "/assets/biomass_power.json", "w") as f:
    json.dump(biomasspower, f, indent=4)


nuclearpower = build_thermal_power_plant(period_length,
                                    n.generators[(n.generators.carrier == "nuclear")].iterrows(), 
                                    n.generators_t["p_max_pu"].keys(),
                                    "NuclearPower",
                                    "Uranium",
                                    "uranium_source",
                                    "uranium",
                                    costs,
                                    uc_data,
)

with open(case_name + "/assets/nuclear_power.json", "w") as f:
    json.dump(nuclearpower, f, indent=4)

geothermalpower = build_thermal_power_plant(period_length,
                                    n.generators[(n.generators.carrier == "geothermal")].iterrows(), 
                                    n.generators_t["p_max_pu"].keys(),
                                    "GeothermalPower",
                                    "Geothermal",
                                    "geothermal_source",
                                    "geothermal",
                                    costs,
                                    uc_data,
)

with open(case_name + "/assets/geothermal_power.json", "w") as f:
    json.dump(geothermalpower, f, indent=4)

#####################################################################################
####### Define VRE generators #######
def calculate_connection_cost(carrier,bus_bin,landfall_length,line_length_factor, submarine_cost, underground_cost):
    with xr.open_dataset("resources/profile_" + number_of_zones + "_" + carrier + ".nc") as ds:
        if ds.indexes["bus"].empty:
            pass
        # if-statement for compatibility with old profiles
        if "year" in ds.indexes:
            ds = ds.sel(year=ds.year.min(), drop=True)
        ds = ds.stack(bus_bin=["bus", "bin"])
        distance = ds["average_distance"].to_pandas()
        distance.index = distance.index.map(flatten)
        connection_cost = line_length_factor * (distance * submarine_cost + landfall_length * underground_cost)
    return connection_cost[bus_bin]

def build_vre_power_plant(selected_generators, generators_with_timeseries, macro_asset_name):
    vre = {
            macro_asset_name: [
                {
                    "type": "VRE",
                    "global_data": {
                        "nodes": {},
                        "transforms": {
                            "timedata": "Electricity"
                        },
                        "edges": {
                            "edge": {
                                "commodity": "Electricity",
                                "unidirectional": True,
                                "has_capacity": True,
                            }
                        },
                        "storage": {}
                    },
                    "instance_data": []
                }
            ]
        }
    
    for idx, row in selected_generators:
        if row.name in generators_with_timeseries:
            temp_availability = {
                "timeseries": {
                                "path": "system/generators_availability.csv",
                                "header": row.name,
                }
            }
        else:
            temp_availability = [row.p_max_pu]
        
        constr_list = {
            "CapacityConstraint": True,
        }
        if row.p_nom_max < float('inf'):
            constr_list["MaxCapacityConstraint"] = True
            temp_max_cap = row.p_nom_max
        else:
            temp_max_cap = 0.0
        if (row.p_nom_min != row.p_nom) & (row.p_nom_min > 0):
            constr_list["MinCapacityConstraint"] = True

        if row.p_min_pu > 0:
            constr_list["MinFlowConstraint"] = True

        if row.lifetime == float('inf'):
            gen_lifetime = round(costs.at[row.carrier, "lifetime"])
        else:
            gen_lifetime = min(round(row.lifetime), round(costs.at[row.carrier, "lifetime"]))
        
        
        if row.carrier == "offwind-ac":
            landfall_length = config_dict["renewable"]["offwind-ac"].get("landfall_length", 0.0)
            line_length_factor = config_dict["renewable"]["offwind-ac"].get("line_length_factor", 1.0)
            submarine_cost = costs.at["offwind-ac-connection-submarine", "investment_cost"]
            underground_cost = costs.at["offwind-ac-connection-underground", "investment_cost"]
            connection_cost = calculate_connection_cost("offwind-ac", row.name.rsplit(" ", 1)[0], landfall_length, line_length_factor, submarine_cost, underground_cost)
            investment_cost = costs.at["offwind", "investment_cost"] + costs.at["offwind-ac-station", "investment_cost"] + connection_cost
            fixed_om_cost = costs.at["offwind", "fixed_om_cost"] 
        elif row.carrier == "offwind-dc":
            landfall_length = config_dict["renewable"]["offwind-dc"].get("landfall_length", 0.0)
            line_length_factor = config_dict["renewable"]["offwind-dc"].get("line_length_factor", 1.0)
            submarine_cost = costs.at["offwind-dc-connection-submarine", "investment_cost"]
            underground_cost = costs.at["offwind-dc-connection-underground", "investment_cost"]
            connection_cost = calculate_connection_cost("offwind-dc", row.name.rsplit(" ", 1)[0], landfall_length, line_length_factor, submarine_cost, underground_cost)    
            investment_cost = costs.at["offwind", "investment_cost"] + costs.at["offwind-dc-station", "investment_cost"] + connection_cost
            fixed_om_cost = costs.at["offwind", "fixed_om_cost"] 
        elif row.carrier == "offwind-float":
            landfall_length = config_dict["renewable"]["offwind-float"].get("landfall_length", 0.0)
            line_length_factor = config_dict["renewable"]["offwind-float"].get("line_length_factor", 1.0)
            submarine_cost = costs.at["offwind-float-connection-submarine", "investment_cost"]
            underground_cost = costs.at["offwind-float-connection-underground", "investment_cost"]
            connection_cost = calculate_connection_cost("offwind-float", row.name.rsplit(" ", 1)[0], landfall_length, line_length_factor, submarine_cost, underground_cost)
            investment_cost = costs.at["offwind-float", "investment_cost"] + costs.at["offwind-float-station", "investment_cost"] + connection_cost
            fixed_om_cost = costs.at["offwind-float", "fixed_om_cost"] 
        else: ##onshore wind, solar, solar-hsat, ror
            investment_cost = costs.at[row.carrier, "investment_cost"]
            fixed_om_cost = costs.at[row.carrier, "fixed_om_cost"]

        if row.p_nom_mod != 0:
            capacity_size = row.p_nom_mod
        elif row.num_units > 0:
            capacity_size = row.p_nom / row.num_units
        else:
            capacity_size = 1.0
        
        vre[macro_asset_name][0]["instance_data"].append(
            {
                "id": row.carrier + "_" + row.bus + "_" + row.name if macro_asset_name=="ror" else row.name,
                "edges":{
                    "edge":{
                        "end_vertex": "elec_" + row.bus,
                        "can_retire": (row.p_nom_extendable) and (row.p_nom > row.p_nom_min),
                        "can_expand": row.p_nom_extendable,
                        "lifetime": gen_lifetime, ## TODO: This should be the remaining lifetime for existing generator (how long until it retires)
                        "capital_recovery_period": round(costs.at[row.carrier, "lifetime"]),
                        "wacc": costs.at[row.carrier, "discount rate"],
                        "min_capacity": row.p_nom_min,
                        "max_capacity": temp_max_cap,
                        "capacity_size": capacity_size, 
                        "existing_capacity": row.p_nom,
                        "availability": temp_availability,
                        "min_flow_fraction": row.p_min_pu,
                        "fixed_om_cost": fixed_om_cost,
                        "investment_cost": investment_cost,
                        "variable_om_cost": costs.at[row.carrier, "variable_om_cost"],
                        "constraints": constr_list,
                    }
                }
            }
        )
    
    return vre

solar_pv = build_vre_power_plant(n.generators[(n.generators.carrier == "solar") | (n.generators.carrier == "solar-hsat")].iterrows(), 
                                    n.generators_t["p_max_pu"].keys(),
                                    "solar_pv"
)
with open(case_name + "/assets/solar_pv.json", "w") as f:
    json.dump(solar_pv, f, indent=4)


onwind = build_vre_power_plant(n.generators[(n.generators.carrier == "onwind")].iterrows(), 
                                    n.generators_t["p_max_pu"].keys(),
                                    "onwind"
)
with open(case_name + "/assets/onwind.json", "w") as f:
    json.dump(onwind, f, indent=4)

offwind_ac = build_vre_power_plant(n.generators[(n.generators.carrier == "offwind-ac")].iterrows(), 
                                    n.generators_t["p_max_pu"].keys(),
                                    "offwind-ac"
)
with open(case_name + "/assets/offwind-ac.json", "w") as f:
    json.dump(offwind_ac, f, indent=4)

offwind_dc = build_vre_power_plant(n.generators[(n.generators.carrier == "offwind-dc")].iterrows(), 
                                    n.generators_t["p_max_pu"].keys(),
                                    "offwind-dc"
)
with open(case_name + "/assets/offwind-dc.json", "w") as f:
    json.dump(offwind_dc, f, indent=4)

offwind_float = build_vre_power_plant(n.generators[(n.generators.carrier == "offwind-float")].iterrows(), 
                                    n.generators_t["p_max_pu"].keys(),
                                    "offwind-float"
)
with open(case_name + "/assets/offwind-float.json", "w") as f:
    json.dump(offwind_float, f, indent=4)

ror = build_vre_power_plant(n.generators[(n.generators.carrier == "ror")].iterrows(), 
                                    n.generators_t["p_max_pu"].keys(),
                                    "ror"
)
with open(case_name + "/assets/run_off_river.json", "w") as f:
    json.dump(ror, f, indent=4)

#####################################################################################
####### Export batteries #######
batteries = {
    "battery": [
        {
            "type": "Battery",
            "global_data": {
                "edges": {
                    "discharge_edge": {
                        "commodity": "Electricity",
                        "unidirectional": True,
                        "has_capacity": True,
                        "constraints": {
                            "CapacityConstraint": True,
                            "StorageDischargeLimitConstraint": True
                        }
                    },
                    "charge_edge": {
                        "commodity": "Electricity",
                        "unidirectional": True,
                        "has_capacity": False
                    }
                },
                "storage": {
                    "commodity": "Electricity",
                    "constraints": {
                        "StorageCapacityConstraint": True,
                        "StorageSymmetricCapacityConstraint": True,
                        "StorageMinDurationConstraint": True,
                        "StorageMaxDurationConstraint": True,
                        "BalanceConstraint": True
                    }
                }
            },
            "instance_data": []
        }
    ]
}

for idx, row in n.stores[n.stores.carrier == "battery"].iterrows():
    discharge_link = n.links[n.links.bus0==row.bus].iloc[0]
    charge_link = n.links[n.links.bus1==row.bus].iloc[0]
    batteries["battery"][0]["instance_data"].append(
        {
            "id": row.name,
            "storage":{
                "existing_capacity": row.e_nom,
                "can_retire": row.e_nom > row.e_nom_min,
                "can_expand": row.e_nom_extendable,
                "lifetime": round(min(row.lifetime, costs.at["battery storage", "lifetime"])),
                "capital_recovery_period": round(costs.at["battery storage", "lifetime"]),
                "wacc": costs.at["battery storage", "discount rate"],
                "investment_cost": costs.at["battery storage", "investment_cost"],
                "fixed_om_cost": costs.at["battery storage", "fixed_om_cost"],
                "variable_om_cost": costs.at["battery storage", "variable_om_cost"],
                "loss_fraction": 1-costs.at["battery storage", "efficiency"],
                "min_duration": 0,
                "max_duration": config_dict["electricity"]["max_hours"]["battery"]
            },
            "edges": {
                "discharge_edge": {
                    "end_vertex": "elec_" + discharge_link.bus1,
                    "existing_capacity": discharge_link.p_nom,
                    "can_retire": bool(discharge_link.p_nom > discharge_link.p_nom_min),
                    "can_expand": bool(discharge_link.p_nom_extendable),
                    "lifetime": round(min(discharge_link.lifetime, costs.at["battery inverter", "lifetime"])),
                    "capital_recovery_period": round(costs.at["battery inverter", "lifetime"]),
                    "wacc": costs.at["battery inverter", "discount rate"],
                    "investment_cost": costs.at["battery inverter", "investment_cost"],
                    "fixed_om_cost": costs.at["battery inverter", "fixed_om_cost"],
                    "variable_om_cost": costs.at["battery inverter", "variable_om_cost"],
                    "efficiency": (costs.at["battery inverter", "efficiency"])**0.5,
                },
                "charge_edge":{
                    "start_vertex": "elec_" + charge_link.bus0,
                    "efficiency": (costs.at["battery inverter", "efficiency"])**0.5,
                    "variable_om_cost": costs.at["battery inverter", "variable_om_cost"],
                }
            }
        }
    )

with open(case_name + "/assets/batteries.json", "w") as f:
    json.dump(batteries, f, indent=4)

#####################################################################################
####### Export pumped hydro storage #######

phs = {
    "pumped_hydro": [
        {
            "type": "Battery",
            "global_data": {
                "edges": {
                    "discharge_edge": {
                        "commodity": "Electricity",
                        "unidirectional": True,
                        "has_capacity": True,
                        "constraints": {
                            "CapacityConstraint": True,
                            "StorageDischargeLimitConstraint": True
                        }
                    },
                    "charge_edge": {
                        "commodity": "Electricity",
                        "unidirectional": True,
                        "has_capacity": False
                    }
                },
                "storage": {
                    "commodity": "Electricity",
                    "constraints": {
                        "StorageCapacityConstraint": True,
                        "StorageSymmetricCapacityConstraint": True,
                        "StorageMinDurationConstraint": True,
                        "StorageMaxDurationConstraint": True,
                        "BalanceConstraint": True
                    }
                }
            },
            "instance_data": []
        }
    ]
}

for idx, row in n.storage_units[n.storage_units.carrier == "PHS"].iterrows():
    if row.p_nom_mod != 0:
        discharge_capacity_size = row.p_nom_mod
    elif row.num_units > 0:
        discharge_capacity_size = row.p_nom / row.num_units
    else:
        discharge_capacity_size = 1.0

    phs["pumped_hydro"][0]["instance_data"].append(
        {
            "id": row.carrier + "_" + row.bus + "_" + row.name,
            "storage":{
                "existing_capacity": row.p_nom * row.max_hours,
                "can_retire": (row.p_nom_extendable) and (row.p_nom > row.p_nom_min),
                "can_expand": row.p_nom_extendable,
                "lifetime": round(min(row.lifetime, costs.at["PHS", "lifetime"])),
                "capital_recovery_period": round(costs.at["PHS", "lifetime"]),
                "wacc": costs.at["PHS", "discount rate"],
                "investment_cost": 0.0,
                "fixed_om_cost": 0.0,
                "variable_om_cost": 0.0,
                "loss_fraction": row.standing_loss,
                "min_duration": 0,
                "max_duration": row.max_hours
            },
            "edges": {
                "discharge_edge": {
                    "end_vertex": "elec_" + row.bus,
                    "existing_capacity": row.p_nom,
                    "can_retire": bool(row.p_nom_extendable) and bool(row.p_nom > row.p_nom_min),
                    "can_expand": bool(row.p_nom_extendable),
                    "capacity_size": discharge_capacity_size,
                    "lifetime": round(min(row.lifetime, costs.at["PHS", "lifetime"])),
                    "capital_recovery_period": round(costs.at["PHS", "lifetime"]),
                    "wacc": costs.at["PHS", "discount rate"],
                    "investment_cost": costs.at["PHS", "investment_cost"],
                    "fixed_om_cost": costs.at["PHS", "fixed_om_cost"],
                    "variable_om_cost": costs.at["PHS", "variable_om_cost"],
                    "efficiency": row.efficiency_dispatch,
                },
                "charge_edge":{
                    "start_vertex": "elec_" + row.bus,
                    "efficiency":  row.efficiency_store,
                    "variable_om_cost": costs.at["PHS", "variable_om_cost"],
                }
            }
        }
    )
with open(case_name + "/assets/pumped_hydro_storage.json", "w") as f:
    json.dump(phs, f, indent=4)

#####################################################################################
####### Export hydro reservoirs #######
hydrores = {
    "hydrores": [
        {
            "type": "HydroRes",
            "global_data": {
                "edges": {
                    "discharge_edge": {
                        "commodity": "Electricity",
                        "unidirectional": True,
                        "has_capacity": True
                    },
                    "inflow_edge": {
                        "commodity": "Electricity",
                        "unidirectional": True,
                        "start_vertex": "hydro_source",
                        "has_capacity": True,
                        "can_expand": False,
                        "can_retire": False,
                        "constraints": {
                            "MustRunConstraint": True
                        }
                    },
                    "spill_edge": {
                        "commodity": "Electricity",
                        "unidirectional": True,
                        "end_vertex": "hydro_source",
                        "can_expand": False,
                        "can_retire": False,
                        "has_capacity": False
                    }
                },
                "storage": {
                    "commodity": "Electricity",
                    "has_capacity": True,
                    "long_duration": False,
                }
            },
             "instance_data": []
        }
    ]
}

for idx, row in n.storage_units[n.storage_units.carrier == "hydro"].iterrows():

    if row.p_nom_mod != 0:
        discharge_capacity_size = row.p_nom_mod
    elif row.num_units > 0:
        discharge_capacity_size = row.p_nom / row.num_units
    else:
        discharge_capacity_size = 1.0

    storage_constr_list = {
        "StorageChargeDischargeRatioConstraint": True,
        "BalanceConstraint": True,
        "StorageMinDurationConstraint": True,
        "StorageMaxDurationConstraint": True,
    }

    discharge_constr_list = {
        "CapacityConstraint": True,
    }

    if row.p_nom_max < float('inf'):
        discharge_constr_list["MaxCapacityConstraint"] = True
        temp_max_cap = row.p_nom_max
    else:
        temp_max_cap = 0.0
    if (row.p_nom_min != row.p_nom) & (row.p_nom_min > 0):
        discharge_constr_list["MinCapacityConstraint"] = True

    if row.p_min_pu > 0:
        discharge_constr_list["MinFlowConstraint"] = True

    hydrores["hydrores"][0]["instance_data"].append(
        {
            "id": row.carrier + "_" + row.bus + "_" + row.name,
            "storage":{
                "existing_capacity": row.p_nom * row.max_hours,
                "can_retire": (row.p_nom_extendable) and (row.p_nom > row.p_nom_min),
                "can_expand": row.p_nom_extendable,
                "lifetime": round(min(row.lifetime, costs.at["hydro", "lifetime"])),
                "capital_recovery_period": round(costs.at["hydro", "lifetime"]),
                "wacc": costs.at["hydro", "discount rate"],
                "investment_cost": 0.0,
                "fixed_om_cost": 0.0,
                "variable_om_cost": 0.0,
                "loss_fraction": row.standing_loss,
                "min_duration": 0,
                "max_duration": row.max_hours,
                "charge_discharge_ratio": 1.0,
                "constraints": storage_constr_list,
            },
            "edges": {
                "discharge_edge":{
                    "end_vertex": "elec_" + row.bus,
                    "existing_capacity": row.p_nom,
                    "can_retire": bool(row.p_nom_extendable) and bool(row.p_nom > row.p_nom_min),
                    "can_expand": bool(row.p_nom_extendable),
                    "capacity_size": discharge_capacity_size,
                    "lifetime": round(min(row.lifetime, costs.at["hydro", "lifetime"])),
                    "capital_recovery_period": round(costs.at["hydro", "lifetime"]),
                    "wacc": costs.at["hydro", "discount rate"],
                    "investment_cost": costs.at["hydro", "investment_cost"],
                    "fixed_om_cost": costs.at["hydro", "fixed_om_cost"],
                    "variable_om_cost": costs.at["hydro", "variable_om_cost"],
                    "constraints": discharge_constr_list,
                    "efficiency": row.efficiency_dispatch,
                    "min_flow_fraction": row.p_min_pu,
                },
                "inflow_edge":{
                    "efficiency": 1.0,
                    "availability": {
                                "timeseries": {
                                    "path": "system/hydro_inflow.csv",
                                    "header": row.name
                                }
                            }
                }
            }
       }
    )

with open(case_name + "/assets/hydro_reservoir.json", "w") as f:
    json.dump(hydrores, f, indent=4)


#####################################################################################
# Move the case_name folder to a different location
if os.path.exists(destination):
    shutil.rmtree(destination)
shutil.move(case_name, destination)
