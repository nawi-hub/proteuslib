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
from watertap.ui.fsapi import FlowsheetInterface, FlowsheetCategory
from watertap.examples.flowsheets.generic_desalination_train import generic_train
from pyomo.environ import units as pyunits
from idaes.core.solvers import get_solver


__author__ = "Alexander V. Dudchenko"


def export_to_ui():
    return FlowsheetInterface(
        name="Generic treatment train",
        do_export=export_variables,
        do_build=build_flowsheet,
        do_solve=solve_flowsheet,
        get_diagram=get_diagram,
        requires_idaes_solver=True,
        category=FlowsheetCategory.wastewater,
        # build_options={
        #     # "train_type": {
        #     #     "name": "Treatment train type",
        #     #     "display_name": "Treatment train type",
        #     #     "values_allowed": [
        #     #         "Pretreatment>Desal1>Desal2>Crystalizer>Valorizer",
        #     #         "Pretreatment>Desal1>Desal2>Valorizer",
        #     #     ],
        #     #     "value": "Pretreatment>Desal1>Desal2>Crystalizer>Valorizer",
        #     # },
        #     "water_source_type": {
        #         "name": "Type of source water",
        #         "display_name": "Source water",
        #         "values_allowed": ["generic", "BGW1", "BGW5", "Seawater"],
        #         "value": "generic",
        #     },
        # },
    )


def export_variables(flowsheet=None, exports=None, build_options=None, **kwargs):

    # print("model", model)
    fs = flowsheet  # model.fs
    # --- Input data ---
    # Feed conditions
    exports.add(
        obj=fs.water_recovery,
        name="System water recovery",
        ui_units=pyunits.dimensionless,
        display_units="%",
        rounding=3,
        description="Overall process design",
        is_input=True,
        input_category="Overall process design",
        is_output=True,
        output_category="Overall process design",
    )

    exports.add(
        obj=fs.costing.LCOW,
        name="Total LCOW",
        ui_units=fs.costing.base_currency / pyunits.m**3,
        display_units="$/m^3",
        rounding=3,
        description="Total LCOW",
        is_input=False,
        input_category="Total LCOW",
        is_output=True,
        output_category="Total LCOW",
    )

    for unit in fs.costing.LCOW_unit:
        exports.add(
            obj=fs.costing.LCOW_unit[unit],
            name=f'{unit.replace("_", " ")} LCOW',
            ui_units=fs.costing.base_currency / pyunits.m**3,
            display_units="$/m^3",
            rounding=3,
            description="Total LCOW",
            is_input=False,
            input_category="Total LCOW",
            is_output=True,
            output_category="Total LCOW",
            chart_tye="stacked_bar_with_net",
            chart_group="process_cost",
        )

    exports.add(
        obj=fs.feed.properties[0].flow_vol_phase["Liq"],
        name="Feed flow rate",
        ui_units=pyunits.m**3 / pyunits.day,
        display_units="m^3/day",
        rounding=3,
        description="Feed flow rate",
        is_input=False,
        input_category="Feed",
        is_output=True,
        output_category="Feed",
    )
    exports.add(
        obj=fs.product.properties[0].flow_vol_phase["Liq"],
        name="Product flow rate",
        ui_units=pyunits.m**3 / pyunits.day,
        display_units="m^3/day",
        rounding=3,
        description="Feed flow rate",
        is_input=False,
        input_category="Product",
        is_output=True,
        output_category="Product",
    )
    exports.add(
        obj=fs.disposal.properties[0].flow_vol_phase["Liq"],
        name="Waste flow rate",
        ui_units=pyunits.m**3 / pyunits.day,
        display_units="m^3/day",
        rounding=3,
        description="Feed flow rate",
        is_input=False,
        input_category="Waste",
        is_output=True,
        output_category="Waste",
    )
    for (phase, ion), obj in fs.feed.properties[0].conc_mass_phase_comp.items():
        if ion != "H2O":
            exports.add(
                obj=obj,
                name="Feed {} concentration".format(ion),
                ui_units=pyunits.mg / pyunits.L,
                display_units="mg/L",
                rounding=3,
                description="{} concentration".format(ion),
                is_input=True,
                input_category="Feed",
                is_output=True,
                output_category="Feed",
            )

    for (phase, ion), obj in fs.disposal.properties[0].conc_mass_phase_comp.items():
        if ion != "H2O":
            exports.add(
                obj=obj,
                name="Waste {} concentration".format(ion),
                ui_units=pyunits.g / pyunits.L,
                display_units="g/L",
                rounding=3,
                description="{} concentration".format(ion),
                is_input=False,
                input_category="Waste",
                is_output=True,
                output_category="Waste",
            )
    for (phase, ion), obj in fs.disposal.properties[0].flow_mass_phase_comp.items():
        exports.add(
            obj=obj,
            name="Waste {} mass flow".format(ion),
            ui_units=pyunits.kg / pyunits.day,
            display_units="kg/day",
            rounding=3,
            description="{} mass flow".format(ion),
            is_input=False,
            input_category="Waste",
            is_output=True,
            output_category="Waste",
        )
    exports.add(
        obj=fs.feed.base_cost,
        name="Feed source cost",
        ui_units=fs.costing.base_currency / pyunits.m**3,
        display_units="$/m^3",
        rounding=3,
        description="Overall system costing",
        is_input=True,
        input_category="Overall system costing",
        is_output=False,
        output_category="Overall system costing",
    )
    exports.add(
        obj=fs.disposal.base_cost,
        name="Disposal source cost",
        ui_units=fs.costing.base_currency / pyunits.m**3,
        display_units="$/m^3",
        rounding=3,
        description="Overall system costing",
        is_input=True,
        input_category="Overall system costing",
        is_output=False,
        output_category="Overall system costing",
    )
    exports.add(
        obj=fs.product.base_cost,
        name="Product distribution cost",
        ui_units=fs.costing.base_currency / pyunits.m**3,
        display_units="$/m^3",
        rounding=3,
        description="Overall system costing",
        is_input=True,
        input_category="Overall system costing",
        is_output=False,
        output_category="Overall system costing",
    )

    for proc in fs.process_order:
        block = proc["process_block"]
        process_name = proc["process_name"]
        process_name_nice = proc["process_name"].replace("_", " ")
        generic_costing_options = f"Generic process costing"
        generic_process_operation = "Generic process operation"
        advanced_costing_options = f"Advanced process costing"
        advanced_operation_option = f"Advanced operation options"

        if proc["process_type"] == "desalter":
            exports.add(
                obj=block.desalter.base_cost,
                name=f"{process_name_nice} LCOW",
                ui_units=fs.costing.base_currency / pyunits.m**3,
                display_units="$/m^3",
                rounding=3,
                description=generic_costing_options,
                is_input=True,
                input_category=generic_costing_options,
                is_output=False,
                output_category=generic_costing_options,
            )
            exports.add(
                obj=block.desalter.recovery_cost,
                name=f"{process_name_nice} rate cost LCOW/WR",
                ui_units=fs.costing.base_currency / pyunits.m**3,
                display_units="$/m^3/%",
                rounding=3,
                description=advanced_costing_options,
                is_input=True,
                input_category=advanced_costing_options,
                is_output=False,
                output_category=advanced_costing_options,
            )
            exports.add(
                obj=block.desalter.water_recovery,
                name=f"{process_name_nice} water recovery",
                ui_units=pyunits.dimensionless,
                display_units="%",
                rounding=3,
                description=generic_process_operation,
                is_input=True,
                input_category=generic_process_operation,
                is_output=True,
                output_category=generic_process_operation,
            )
            exports.add(
                obj=block.desalter.LCOW,
                name=f"{process_name_nice} LCOW",
                ui_units=fs.costing.base_currency / pyunits.m**3,
                display_units="$/m^3",
                rounding=3,
                description=generic_costing_options,
                is_input=False,
                input_category=generic_costing_options,
                is_output=True,
                output_category=generic_costing_options,
            )
            if process_name != "Desal_3":
                exports.add(
                    obj=block.desalter.brine_solids_concentration,
                    name=f"{process_name_nice} brine TDS",
                    ui_units=pyunits.g / pyunits.L,
                    display_units="g/L",
                    rounding=3,
                    description=advanced_operation_option,
                    is_input=True,
                    input_category=generic_process_operation,
                    is_output=True,
                    output_category=generic_process_operation,
                )
            else:
                exports.add(
                    obj=block.desalter.brine_solids_concentration,
                    name=f"{process_name_nice} sludge TDS",
                    ui_units=pyunits.g / pyunits.L,
                    display_units="g/L",
                    rounding=3,
                    description=advanced_operation_option,
                    is_input=False,
                    input_category=generic_process_operation,
                    is_output=True,
                    output_category=generic_process_operation,
                )
                exports.add(
                    obj=block.desalter.brine_water_percent,
                    name=f"{process_name_nice} sludge water %",
                    ui_units=pyunits.dimensionless,
                    display_units="%",
                    rounding=2,
                    description=advanced_operation_option,
                    is_input=True,
                    input_category=generic_process_operation,
                    is_output=True,
                    output_category=generic_process_operation,
                )
        if proc["process_type"] == "separator" or proc["process_type"] == "valorizer":
            if proc["process_type"] == "separator":
                exports.add(
                    obj=block.separator.base_cost,
                    name=f"{process_name_nice} LCOW",
                    ui_units=fs.costing.base_currency / pyunits.m**3,
                    display_units="$/m^3",
                    rounding=3,
                    description=generic_costing_options,
                    is_input=True,
                    input_category=generic_costing_options,
                    is_output=False,
                    output_category=generic_costing_options,
                )
            else:
                exports.add(
                    obj=block.separator.mass_base_cost,
                    name=f"{process_name_nice} LCOR",
                    ui_units=fs.costing.base_currency / pyunits.kg,
                    display_units="$/kg",
                    rounding=3,
                    description=generic_costing_options,
                    is_input=True,
                    input_category=generic_costing_options,
                    is_output=False,
                    output_category=generic_costing_options,
                )
            # exports.add(
            #     obj=block.separator.additive_cost,
            #     name="Additive cost",
            #     ui_units=fs.costing.base_currency / pyunits.kg,
            #     display_units="$/kg",
            #     rounding=3,
            #     description=f"{process_name} costing and operation",
            #     is_input=True,
            #     input_category=f"{process_name} costing and operation",
            #     is_output=False,
            #     output_category=f"{process_name} costing and operation",
            # )
            # exports.add(
            #     obj=block.separator.additive_dose,
            #     name="Additive dose",
            #     ui_units=pyunits.mg / pyunits.L,
            #     display_units="PPM",
            #     rounding=3,
            #     description=f"{process_name} costing and operation",
            #     is_input=True,
            #     input_category=f"{process_name} costing and operation",
            #     is_output=False,
            #     output_category=f"{process_name} costing and operation",
            # )
            if proc["process_type"] == "valorizer":
                for ion in block.separator.product_value.keys():
                    if ion != "TDS" and ion != "H2O":
                        exports.add(
                            obj=block.separator.component_removal_percent[ion],
                            name=f"Valorizer {ion} recovery %",
                            ui_units=pyunits.dimensionless,
                            display_units="%",
                            rounding=3,
                            description=generic_costing_options,
                            is_input=True,
                            input_category=generic_costing_options,
                            is_output=False,
                            output_category=generic_costing_options,
                        )
                        exports.add(
                            obj=block.separator.product_value[ion],
                            name=f"Valorizer {ion} value",
                            ui_units=fs.costing.base_currency / pyunits.kg,
                            display_units="$/kg",
                            rounding=3,
                            description=generic_costing_options,
                            is_input=True,
                            input_category=generic_costing_options,
                            is_output=True,
                            output_category=generic_costing_options,
                        )
                exports.add(
                    obj=block.separator.treatment_LCOW,
                    name=f"Valorizer treatment cost (LCOW)",
                    ui_units=fs.costing.base_currency / pyunits.m**3,
                    display_units="$/m^3",
                    rounding=3,
                    description=generic_costing_options,
                    is_input=False,
                    input_category=generic_costing_options,
                    is_output=True,
                    output_category=generic_costing_options,
                )
                exports.add(
                    obj=block.separator.ion_removal_LCOW,
                    name=f"Valorizer levelized revenue (LROW)",
                    ui_units=fs.costing.base_currency / pyunits.m**3,
                    display_units="$/m^3",
                    rounding=3,
                    description=generic_costing_options,
                    is_input=False,
                    input_category=generic_costing_options,
                    is_output=True,
                    output_category=generic_costing_options,
                )
                for (phase, ion), obj in block.separator.product_properties[
                    0
                ].flow_mass_phase_comp.items():
                    if ion != "H2O" and ion != "TDS":
                        exports.add(
                            obj=obj,
                            name="Valorizer {} solids production".format(ion),
                            ui_units=pyunits.kg / pyunits.day,
                            display_units="kg/day",
                            rounding=3,
                            description="{} mass flow".format(ion),
                            is_input=False,
                            input_category="Product",
                            is_output=True,
                            output_category="Product",
                        )


def build_flowsheet(build_options=None, **kwargs):
    # build and solve initial flowsheet
    print("UI FLOWSHEET", build_options)
    # if build_options["Bypass"].value == "true":  # build with bypass

    solver = get_solver()
    m = generic_train.build(
        # train_type=build_options["train_type"].value,
        # water_source=build_options["water_source_type"].value,
    )
    generic_train.initialize(m, solver)
    return m


def get_diagram(build_options):
    # if (
    #     build_options["train_type"].value
    #     == "Pretreatment>Desal1>Desal2>Crystalizer>Valorizer"
    # ):
    #     return "pd1d2cv.png"
    # elif build_options["train_type"].value == "Pretreatment>Desal1>Desal2>Valorizer":
    #     return "pd1d2v.png"
    # else:
    return "fig_with_costs.png"  # "pd1d2cv.png"


def solve_flowsheet(flowsheet):
    solver = get_solver()
    results = generic_train.solve(flowsheet, solver)
    return results
