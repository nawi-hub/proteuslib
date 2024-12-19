import os

from pyomo.environ import (
    ConcreteModel,
    Var,
    Constraint,
    units as pyunits,
    SolverFactory,
    assert_optimal_termination,
)
from pyomo.util.check_units import assert_units_consistent

from idaes.core import FlowsheetBlock
from idaes.core.util.exceptions import ConfigurationError
from idaes.core.util.model_statistics import degrees_of_freedom
import idaes.logger as idaeslog
from idaes.core.surrogate.surrogate_block import SurrogateBlock
from idaes.core import UnitModelCostingBlock

from watertap.property_models.water_prop_pack import WaterParameterBlock
from watertap.property_models.multicomp_aq_sol_prop_pack import (
    MCASParameterBlock,
    MaterialFlowBasis,
)
from watertap.costing import WaterTAPCosting
from watertap.core.solvers import get_solver
from watertap.unit_models.surrogate_crystallizer import SurrogateCrystallizer

from watertap.unit_models.tests.reaktoro_pitzer_crystallizer import add_reaktoro_model


__author__ = "Oluwamayowa Amusat"


def test_reaktoro_crystallizer(cryst_temp, evap_percent, feed_pH):
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.water_properties_vapor = WaterParameterBlock()

    # # These names need to match the model names
    input_ions = ["Cl_-", "Na_+", "SO4_2-", "Mg_2+", "Ca_2+", "K_+", "HCO3_-"]
    solids_list = {
        "Calcite": {"Ca_2+": 1, "HCO3_-": 1},
        "Gypsum": {"Ca_2+": 1, "SO4_2-": 1, "H2O": 2},
        "Brucite": {"Mg_2+": 1, "H2O": 2},
        "Anhydrite": {"Ca_2+": 1, "SO4_2-": 1},
        "Glauberite": {"Ca_2+": 1, "Na_+": 2, "SO4_2-": 2},
        "Halite": {"Na_+": 1, "Cl_-": 1},
    }

    # Use property package from WT softening work
    m.fs.cryst_prop_feed = MCASParameterBlock(
        solute_list=input_ions,
        material_flow_basis=MaterialFlowBasis.mass,
    )

    # Create crystallizer framework
    m.fs.cryst = SurrogateCrystallizer(
        property_package=m.fs.cryst_prop_feed,
        vapor_property_package=m.fs.water_properties_vapor,
        solids_ions_dict=solids_list,
    )
    print(
        "Degrees of freedom of crystallizer model with undefined feed",
        degrees_of_freedom(m),
    )
    m.fs.cryst.inlet.temperature.fix(298.15)
    m.fs.cryst.inlet.pressure.fix(101325)

    feed = {
        "ion_composition_g_kg": {
            "Na_+": 91.49859876,
            "K_+": 3.185931759,
            "Cl_-": 159.0941278,
            "Ca_2+": 0.788476605,
            "Mg_2+": 10.5806906,
            "HCO3_-": 1.614773395,
            "SO4_2-": 22.20893266,
        },
        "water_content_kg": 12.0764972138554,
    }
    g_to_kg = 1e-3
    for i in m.fs.cryst_prop_feed.component_list:
        if i == "H2O":
            m.fs.cryst.inlet.flow_mass_phase_comp[0.0, "Liq", i].fix(
                feed["water_content_kg"]
            )
        else:
            m.fs.cryst.inlet.flow_mass_phase_comp[0.0, "Liq", i].fix(
                feed["water_content_kg"] * feed["ion_composition_g_kg"][i] * g_to_kg
            )

    print(
        "Degrees of freedom of crystallizer model with defined feed",
        degrees_of_freedom(m),
    )
    assert_units_consistent(m)

    # Create input dictionaries for reaktoro backend
    surrogate_inputs = {
        "composition": {"flowsheet_var": m.fs.cryst.inlet},
        "Temperature": {"flowsheet_var": m.fs.cryst.temperature_operating},
        "Evaporation percent": {"flowsheet_var": m.fs.cryst.evaporation_percent},
    }
    # }
    surrogate_outputs = {
        "Calcite": {"Solid": True, "mw": 100},
        "Gypsum": {"Solid": True, "mw": 172},
        "Anhydrite": {"Solid": True, "mw": 136},
        "Glauberite": {"Solid": True, "mw": 278.19},
        "Halite": {"Solid": True, "mw": 58.5},
        "Brucite": {"Solid": True, "mw": 58.32},
    }
    mtw = {
        "Cl_-": 35.5,
        "Na_+": 23,
        "SO4_2-": 96,
        "Mg_2+": 24.3,
        "Ca_2+": 40,
        "K_+": 39,
        "HCO3_-": 61,
        "H2O": 18,
    }

    # Costing
    m.fs.costing = WaterTAPCosting()
    m.fs.cryst.costing = UnitModelCostingBlock(
        flowsheet_costing_block=m.fs.costing,
    )
    m.fs.costing.cost_process()

    # Edit bound on liquid pressures from property package: should this be done on the backend instead?
    m.fs.cryst.properties_out_liq[0].pressure.setlb(1000)

    # 1. Simulate single case
    print("Degrees of freedom before fixing decision variables:", degrees_of_freedom(m))

    m.fs.cryst.temperature_operating.fix(cryst_temp + 273.15)
    m.fs.cryst.evaporation_percent.fix(evap_percent)
    print("Degrees of freedom after fixing decision variables:", degrees_of_freedom(m))

    # Initialize crystallizer without reaktoro block
    assert_units_consistent(m)
    m.fs.cryst.initialize()
    # Now add the reaktoro block and solve
    add_reaktoro_model(
        model=m.fs.cryst,
        input_ions=input_ions,
        input_dict=surrogate_inputs,
        solids_list=solids_list,
        output_dict=surrogate_outputs,
        pH_feed=feed_pH,
        mol_wt_ions=mtw,
    )
    assert degrees_of_freedom(m) == degrees_of_freedom(m.fs.cryst.rkt_block)
    cy_solver = get_solver(solver="cyipopt-watertap")
    cy_solver.options["max_iter"] = 100
    result = cy_solver.solve(m, tee=True)
    assert_optimal_termination(result)
    m.fs.cryst.report()
    return m


if __name__ == "__main__":
    temp_c = 40
    evap = 60
    feed_pH = 6
    m = test_reaktoro_crystallizer(
        cryst_temp=temp_c, evap_percent=evap, feed_pH=feed_pH
    )

    m.fs.cryst.flow_mass_sol_comp_apparent.pprint()
