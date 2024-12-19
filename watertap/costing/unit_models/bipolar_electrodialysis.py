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
from watertap.costing.util import (
    register_costing_parameter_block,
    cost_rectifier,
    make_capital_cost_var,
    make_fixed_operating_cost_var,
)


def build_bipolar_electrodialysis_cost_param_block(blk):
    # The following costing itemization and values are referenced to "Desalination 452 (2019) 265–278"
    blk.membrane_capital_cost = pyo.Var(
        initialize=160,
        doc="Membrane and capital costs in [US$/m^2-membrane-area]",
        units=pyo.units.USD_2018 / (pyo.units.meter**2),
    )

    blk.factor_membrane_replacement = pyo.Var(
        initialize=0.2,
        doc="Membrane and equipment (other stack components) housing replacement factor, equal to 1/lifetime.",
        units=pyo.units.year**-1,
    )

    blk.stack_electrode_capital_cost = pyo.Var(
        initialize=2100,
        doc="Electrode cost in [US$/m^2-electrode-area] ",
        units=pyo.units.USD_2018 / (pyo.units.meter**2),
    )

    blk.factor_stack_electrode_replacement = pyo.Var(
        initialize=0.2,
        doc="Stack and electrode replacement factor, equal to 1/lifetime.",
        units=pyo.units.year**-1,
    )


@register_costing_parameter_block(
    build_rule=build_bipolar_electrodialysis_cost_param_block,
    parameter_block_name="bipolar_electrodialysis",
)
def cost_bipolar_electrodialysis(
    blk,
    cost_electricity_flow=True,
    has_rectifier=False,
):
    """
    Function for costing the bipolar electrodialysis unit

    Args:
        cost_electricity_flow (:obj:`bool`, optional): Option for including the
            costing of electricity. Defaults to True.
        has_rectifier (:obj:`bool`, optional): Option for including a rectifier.
            Defaults to False.
    """
    t0 = blk.flowsheet().time.first()

    # Changed this to grab power from performance table which is identified
    # by same key regardless of whether the Electrodialysis unit is 0D or 1D
    if cost_electricity_flow:
        if not has_rectifier:
            blk.costing_package.cost_flow(
                pyo.units.convert(
                    blk.unit_model.get_power_electrical(t0),
                    to_units=pyo.units.kW,
                ),
                "electricity",
            )
        else:
            power = blk.unit_model.get_power_electrical(blk.flowsheet().time.first())
            cost_rectifier(blk, power=power, ac_dc_conversion_efficiency=0.9)

    cost_bipolar_electrodialysis_stack(blk)


def cost_bipolar_electrodialysis_stack(blk):
    """
    Generic function for costing the stack in an electrodialysis unit.
    Assumes the unit_model has a `cell_num`, `cell_width`, and `cell_length`
    set of variables used to size the total membrane area.

    """
    make_capital_cost_var(blk)
    make_fixed_operating_cost_var(blk)
    blk.costing_package.add_cost_factor(blk, "TIC")
    if blk.find_component("capital_cost_rectifier") is not None:
        blk.capital_cost_constraint = pyo.Constraint(
            expr=blk.capital_cost
            == blk.cost_factor
            * (
                pyo.units.convert(
                    blk.costing_package.bipolar_electrodialysis.membrane_capital_cost
                    * (
                        2
                        * blk.unit_model.cell_num
                        * blk.unit_model.cell_width
                        * blk.unit_model.cell_length
                    )
                    + blk.costing_package.bipolar_electrodialysis.stack_electrode_capital_cost
                    * (2 * blk.unit_model.cell_width * blk.unit_model.cell_length),
                    to_units=blk.costing_package.base_currency,
                )
                + blk.capital_cost_rectifier
            )
        )
    else:
        blk.capital_cost_constraint = pyo.Constraint(
            expr=blk.capital_cost
            == blk.cost_factor
            * pyo.units.convert(
                blk.costing_package.bipolar_electrodialysis.membrane_capital_cost
                * (
                    2
                    * blk.unit_model.cell_num
                    * blk.unit_model.cell_width
                    * blk.unit_model.cell_length
                )
                + blk.costing_package.bipolar_electrodialysis.stack_electrode_capital_cost
                * (2 * blk.unit_model.cell_width * blk.unit_model.cell_length),
                to_units=blk.costing_package.base_currency,
            )
        )
    blk.fixed_operating_cost_constraint = pyo.Constraint(
        expr=blk.fixed_operating_cost
        == pyo.units.convert(
            blk.costing_package.bipolar_electrodialysis.factor_membrane_replacement
            * blk.costing_package.bipolar_electrodialysis.membrane_capital_cost
            * (
                2
                * blk.unit_model.cell_num
                * blk.unit_model.cell_width
                * blk.unit_model.cell_length
            )
            + blk.costing_package.bipolar_electrodialysis.factor_stack_electrode_replacement
            * blk.costing_package.bipolar_electrodialysis.stack_electrode_capital_cost
            * (2 * blk.unit_model.cell_width * blk.unit_model.cell_length),
            to_units=blk.costing_package.base_currency
            / blk.costing_package.base_period,
        )
    )
