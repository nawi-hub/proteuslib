#################################################################################
# WaterTAP Copyright (c) 2020-2024, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory, Oak Ridge National Laboratory,
# National Renewable Energy Laboratory, and National Energy Technology
# Laboratory (subject to receipt of any required approvals from the U.S. Dept.
# of Energy). All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. These files are also available online at the URL
# "https://github.com/watertap-org/watertap/"
#################################################################################

import pyomo.environ as pyo
from ..util import register_costing_parameter_block, make_capital_cost_var


def build_heat_exchanger_cost_param_block(blk):

    blk.unit_cost = pyo.Var(
        initialize=300,
        doc="Estimated from multiple sources",
        units=pyo.units.USD_2020,
    )

    blk.material_factor_cost = pyo.Var(
        initialize=1,
        doc="Material factor",
        units=pyo.units.dimensionless,
    )

    blk.steam_cost = pyo.Var(
        initialize=0.008,
        units=pyo.units.USD_2018 / (pyo.units.kg),
        doc="steam cost per kg",
    )

    costing = blk.parent_block()
    costing.register_flow_type("steam", blk.steam_cost)


@register_costing_parameter_block(
    build_rule=build_heat_exchanger_cost_param_block,
    parameter_block_name="heat_exchanger",
)
def cost_heat_exchanger(blk, cost_steam_flow=False):
    """
    Heat Exchanger Costing Method

    TODO: describe equations
    """
    make_capital_cost_var(blk)
    blk.costing_package.add_cost_factor(blk, "TIC")
    blk.capital_cost_constraint = pyo.Constraint(
        expr=blk.capital_cost
        == blk.cost_factor
        * pyo.units.convert(
            blk.costing_package.heat_exchanger.unit_cost
            * blk.costing_package.heat_exchanger.material_factor_cost
            * (
                pyo.units.convert(blk.unit_model.area, to_units=(pyo.units.m**2))
                / pyo.units.m**2
            ),
            to_units=blk.costing_package.base_currency,
        )
    )

    if cost_steam_flow:
        blk.costing_package.cost_flow(
            pyo.units.convert(
                (blk.unit_model.hot_side_inlet.flow_mass_phase_comp[0, "Vap", "H2O"]),
                to_units=pyo.units.kg / pyo.units.s,
            ),
            "steam",
        )
