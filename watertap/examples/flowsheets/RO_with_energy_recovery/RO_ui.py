#################################################################################
# WaterTAP Copyright (c) 2020-2023, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory, Oak Ridge National Laboratory,
# National Renewable Energy Laboratory, and National Energy Technology
# Laboratory (subject to receipt of any required approvals from the U.S. Dept.
# of Energy). All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. These files are also available online at the URL
# "https://github.com/watertap-org/watertap/"
#################################################################################
from idaes.core.solvers import get_solver
from watertap.ui.fsapi import FlowsheetInterface
from watertap.core.util.initialization import assert_degrees_of_freedom
from watertap.examples.flowsheets.RO_with_energy_recovery.RO_with_energy_recovery import (
    build,
    set_operating_conditions,
    initialize_system,
    optimize_set_up,
    solve,
    ERDtype,
)
from pyomo.environ import units as pyunits, assert_optimal_termination
from pyomo.util.check_units import assert_units_consistent


def export_to_ui():
    return FlowsheetInterface(
        name="RO with energy recovery",
        do_export=export_variables,
        do_build=build_flowsheet,
        do_solve=solve_flowsheet,
    )


def export_variables(flowsheet=None, exports=None):
    fs = flowsheet
    # --- Input data ---
    # Feed conditions
    exports.add(
        obj=fs.feed.properties[0].flow_vol_phase["Liq"],
        name="Volumetric flow rate",
        ui_units=pyunits.m**3 / pyunits.hour,
        display_units="m3/hr",
        rounding=2,
        description="Inlet volumetric flow rate",
        is_input=True,
        input_category="Feed",
        is_output=True,
        output_category="Feed",
    )
    exports.add(
        obj=fs.feed.properties[0].mass_frac_phase_comp["Liq", "NaCl"],
        name="NaCl mass fraction",
        ui_units=pyunits.dimensionless,
        display_units="fraction",
        rounding=2,
        description="Inlet NaCl mass fraction",
        is_input=True,
        input_category="Feed",
        is_output=True,
        output_category="Feed",
    )

    # Unit model data, feed pump
    exports.add(
        obj=fs.P1.efficiency_pump[0],
        name="Feed pump efficiency",
        ui_units=pyunits.dimensionless,
        display_units="fraction",
        rounding=2,
        description="Efficiency of feed pump",
        is_input=True,
        input_category="Feed Pump",
        is_output=False,
    )
    exports.add(
        obj=fs.P1.control_volume.properties_out[0].pressure,
        name="Feed pump operating pressure",
        ui_units=pyunits.bar,
        display_units="bar",
        rounding=2,
        description="Operating pressure of feed pump",
        is_input=True,
        input_category="Feed Pump",
        is_output=False,
    )

    # Unit model data, RO
    exports.add(
        obj=fs.RO.A_comp[0, "H2O"],
        name="Water permeability coefficient",
        ui_units=pyunits.m / pyunits.s / pyunits.Pa,
        display_units="m/s-Pa",
        rounding=2,
        description="RO Water permeability coefficient",
        is_input=True,
        input_category="Reverse Osmosis",
        is_output=False,
    )


def build_flowsheet(erd_type=ERDtype.pump_as_turbine):
    # build and solve initial flowsheet
    m = build()

    solver = get_solver()

    # build, set, and initialize
    m = build(erd_type=erd_type)
    set_operating_conditions(m)
    initialize_system(m, solver=solver)

    # optimize and display
    optimize_set_up(m)
    solve(m, solver=solver)
    return m


def solve_flowsheet(flowsheet=None):
    fs = flowsheet
    results = solve(fs)
    return results
