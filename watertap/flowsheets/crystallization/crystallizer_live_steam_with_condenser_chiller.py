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
from pyomo.environ import (
    ConcreteModel,
    TerminationCondition,
)
from pyomo.environ import (
    ConcreteModel,
    value,
    Constraint,
    Objective,
    Var,
    NonNegativeReals,
    TransformationFactory,
    units as pyunits,
    check_optimal_termination,
)
from pyomo.network import Arc, SequentialDecomposition
from idaes.core import FlowsheetBlock
from watertap.core.solvers import get_solver
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util.initialization import (
    propagate_state,
)
from pyomo.util.check_units import assert_units_consistent

from idaes.core import FlowsheetBlock
from idaes.core.util.model_statistics import degrees_of_freedom

import idaes.core.util.scaling as iscale
import idaes.logger as idaeslog
from watertap.core.solvers import get_solver
from idaes.core import UnitModelCostingBlock

from watertap.property_models.unit_specific import cryst_prop_pack as props
from watertap.unit_models.crystallizer import Crystallization
from watertap.costing import WaterTAPCosting, CrystallizerCostType
from watertap.costing.unit_models.heat_exchanger import (
    cost_heat_exchanger,
)

from io import StringIO
from pyomo.util.infeasible import (
    log_infeasible_constraints,
)
from watertap.unit_models.steam_heater_0D import SteamHeater0D, Mode
from idaes.models.unit_models import Heater, Separator, Mixer, Product, Feed
from idaes.models.unit_models.separator import SplittingType
from watertap.unit_models.pressure_changer import Pump
from watertap.costing.unit_models.pump import (
    cost_pump,
)
from idaes.models.unit_models.mixer import MomentumMixingType
from watertap.core.solvers import get_solver
from watertap.unit_models.mvc.components.lmtd_chen_callback import (
    delta_temperature_chen_callback,
)
from idaes.models.unit_models.heat_exchanger import (
    HeatExchangerFlowPattern,
)
from watertap.costing.unit_models.heater_chiller import (
    cost_heater_chiller,
)
from watertap.property_models.unit_specific import cryst_prop_pack as props
import watertap.property_models.water_prop_pack as props_w
import watertap.property_models.NaCl_T_dep_prop_pack as props_nacl
from pyomo.common.log import LoggingIntercept
from idaes.models.unit_models.translator import Translator
import logging
from watertap.core.util.model_debug_mode import activate
activate()



__author__ = "Elmira Shamlou"


def main():
    solver = get_solver()
    m = build()
    set_operating_conditions(m)
    initialize_system(m, solver=solver)

    optimize_set_up(m)

    #interval_initializer(m)
    solve(m, solver=solver)

    #print("\n***---optimization results---***")
    display_system(m)
    #display_design(m)
    #display_state(m)

    return m


def build():

    # flowsheet set up
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)

    # properties
    m.fs.properties_vapor = props_w.WaterParameterBlock()
    m.fs.properties = props.NaClParameterBlock()
    m.fs.properties_nacl = props_nacl.NaClParameterBlock()

    # Control volume flow blocks
    m.fs.feed = Feed(property_package= m.fs.properties_nacl)
    m.fs.distillate = Product(property_package= m.fs.properties_vapor)
    


    # overall process water recovery (evaporation rate/solid content)


    # unit models: steam heater, mixer, pump, crystalizer

    m.fs.crystallizer = Crystallization(property_package=m.fs.properties)
    m.fs.heater = SteamHeater0D(
        hot_side_name="hot",
        cold_side_name="cold",
        hot={
            "property_package": m.fs.properties_vapor,
        },
        cold={
            "property_package": m.fs.properties_nacl,
        },
        delta_temperature_callback=delta_temperature_chen_callback,
        flow_pattern=HeatExchangerFlowPattern.countercurrent,
        mode=Mode.HEATER,
    )

    m.fs.condenser = SteamHeater0D(
        hot_side_name="hot",
        cold_side_name="cold",
        hot={
            "property_package": m.fs.properties_vapor,
        },
        cold={
            "property_package": m.fs.properties_nacl,
        },
        delta_temperature_callback=delta_temperature_chen_callback,
        flow_pattern=HeatExchangerFlowPattern.countercurrent,
        mode=Mode.CONDENSER,
        estimate_cooling_water=True
    )

    m.fs.chiller = Heater(
        property_package=m.fs.properties_nacl, has_pressure_change=False
    )
  
    m.fs.mixer = Mixer(
        property_package=m.fs.properties_nacl,
        momentum_mixing_type=MomentumMixingType.equality,
        inlet_list=["feed", "recycle"],
    )
    m.fs.mixer.pressure_equality_constraints[0, 2].deactivate()

    m.fs.pump = Pump(property_package=m.fs.properties_nacl)

    m.fs.separator = Separator(
        property_package=m.fs.properties_nacl,
        outlet_list=["recycle", "purge"],
        split_basis=SplittingType.totalFlow,
    )

    m.fs.tb_recycle = Translator(
        inlet_property_package=m.fs.properties,
        outlet_property_package=m.fs.properties_nacl,
    )

    m.fs.tb_vapor = Translator(
        inlet_property_package=m.fs.properties,
        outlet_property_package=m.fs.properties_vapor
    )

    @m.fs.tb_vapor.Constraint()
    def eq_flow_mass_comp(blk):
        return (
            blk.properties_in[0].flow_mass_phase_comp["Vap", "H2O"]
            == blk.properties_out[0].flow_mass_phase_comp["Vap", "H2O"]
        )
    @m.fs.tb_vapor.Constraint()
    def eq_flow_mass_comp_vapor(blk):
        return (
            blk.properties_in[0].flow_mass_phase_comp["Liq", "H2O"]
            == blk.properties_out[0].flow_mass_phase_comp["Liq", "H2O"]
        )

    @m.fs.tb_vapor.Constraint()
    def eq_temperature(blk):
        return blk.properties_in[0].temperature == blk.properties_out[0].temperature

    @m.fs.tb_vapor.Constraint()
    def eq_pressure(blk):
        return blk.properties_in[0].pressure == blk.properties_out[0].pressure

    @m.fs.tb_recycle.Constraint()
    def eq_flow_mass_comp(blk):
        return (
            blk.properties_in[0].flow_mass_phase_comp["Liq", "H2O"]
            == blk.properties_out[0].flow_mass_phase_comp["Liq", "H2O"]
        )
    @m.fs.tb_recycle.Constraint()
    def eq_flow_mass_comp_solute(blk):
        return (
            blk.properties_in[0].flow_mass_phase_comp["Liq", "NaCl"]
            == blk.properties_out[0].flow_mass_phase_comp["Liq", "NaCl"]
        )

    @m.fs.tb_recycle.Constraint()
    def eq_temperature(blk):
        return blk.properties_in[0].temperature == blk.properties_out[0].temperature

    @m.fs.tb_recycle.Constraint()
    def eq_pressure(blk):
        return blk.properties_in[0].pressure == blk.properties_out[0].pressure
    
    m.fs.tb_heater_to_cryst = Translator(
        inlet_property_package=m.fs.properties_nacl,
        outlet_property_package=m.fs.properties,
    )

    @m.fs.tb_heater_to_cryst.Constraint()
    def eq_flow_mass_comp(blk):
        return (
            blk.properties_in[0].flow_mass_phase_comp["Liq", "H2O"]
            == blk.properties_out[0].flow_mass_phase_comp["Liq", "H2O"]
        )
    @m.fs.tb_heater_to_cryst.Constraint()
    def eq_flow_mass_comp_solute(blk):
        return (
            blk.properties_in[0].flow_mass_phase_comp["Liq", "NaCl"]
            == blk.properties_out[0].flow_mass_phase_comp["Liq", "NaCl"]
        )

    @m.fs.tb_heater_to_cryst.Constraint()
    def eq_temperature(blk):
        return blk.properties_in[0].temperature == blk.properties_out[0].temperature

    @m.fs.tb_heater_to_cryst.Constraint()
    def eq_pressure(blk):
        return blk.properties_in[0].pressure == blk.properties_out[0].pressure
    
    
    m.fs.eq_heater_temperature_rise = Constraint(
           expr=m.fs.heater.cold_side_outlet.temperature[0] - m.fs.heater.cold_side_inlet.temperature[0] == 4 * pyunits.K)
    m.fs.eq_chiller = Constraint(
           expr=m.fs.chiller.control_volume.properties_out[0].temperature == m.fs.condenser.cold_side_inlet.temperature[0])

    # performance expressions

    # connections
    m.fs.s01 = Arc(source=m.fs.feed.outlet, destination=m.fs.mixer.feed)
    m.fs.s02 = Arc(
        source=m.fs.separator.recycle, destination=m.fs.mixer.recycle)
    m.fs.s03 = Arc(source=m.fs.mixer.outlet, destination=m.fs.pump.inlet)
    m.fs.s04 = Arc(source=m.fs.pump.outlet, destination=m.fs.heater.cold_side_inlet)
    m.fs.s05 = Arc(source=m.fs.heater.cold_side_outlet, destination=m.fs.tb_heater_to_cryst.inlet)
    m.fs.s06 = Arc(source=m.fs.tb_heater_to_cryst.outlet, destination=m.fs.crystallizer.inlet)

    m.fs.s07 = Arc(source=m.fs.crystallizer.outlet, destination=m.fs.tb_recycle.inlet)
    m.fs.s08 = Arc(source=m.fs.tb_recycle.outlet, destination=m.fs.separator.inlet)
    
    m.fs.s09 = Arc(source=m.fs.crystallizer.vapor, destination=m.fs.tb_vapor.inlet)
    m.fs.s10 = Arc(source=m.fs.tb_vapor.outlet, destination=m.fs.condenser.hot_side_inlet)
    m.fs.s11 = Arc(source=m.fs.condenser.hot_side_outlet, destination=m.fs.distillate.inlet)
    m.fs.s12 = Arc(source=m.fs.condenser.cold_side_outlet, destination=m.fs.chiller.inlet)
   

 
    
  
    TransformationFactory("network.expand_arcs").apply_to(m)
    add_costs(m)



    # set default property values
    m.fs.properties.set_default_scaling(
        "flow_mass_phase_comp", 1, index=("Liq", "H2O")
    )
    m.fs.properties.set_default_scaling(
        "flow_mass_phase_comp", 1e2, index=("Liq", "NaCl")
    )
    m.fs.properties_vapor.set_default_scaling(
        "flow_mass_phase_comp", 1, index=("Liq", "H2O")
    )
    m.fs.properties_vapor.set_default_scaling(
        "flow_mass_phase_comp", 1, index=("Vap", "H2O")
    )

    iscale.set_scaling_factor(m.fs.heater.hot.heat, 1e-3)
    iscale.set_scaling_factor(m.fs.heater.cold.heat, 1e-3)
    iscale.set_scaling_factor(m.fs.heater.overall_heat_transfer_coefficient, 1e-3)
    iscale.set_scaling_factor(m.fs.heater.area, 1e-1)

    iscale.set_scaling_factor(m.fs.condenser.hot.heat, 1e-3)
    iscale.set_scaling_factor(m.fs.condenser.cold.heat, 1e-3)
    iscale.set_scaling_factor(m.fs.condenser.overall_heat_transfer_coefficient, 1e-3)
    iscale.set_scaling_factor(m.fs.condenser.area, 1e-1)
    iscale.set_scaling_factor(m.fs.chiller.control_volume.heat, 1e-3)

    iscale.calculate_scaling_factors(m)

    return m

def add_costs(m):

    m.fs.costing = WaterTAPCosting()
    m.fs.crystallizer.costing = UnitModelCostingBlock(
       flowsheet_costing_block=m.fs.costing,
        costing_method_arguments={"cost_type": CrystallizerCostType.mass_basis},
)
    m.fs.heater.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing, 
                                                costing_method=cost_heat_exchanger,
        costing_method_arguments={"cost_steam_flow": True},)

    m.fs.condenser.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing, 
                                                costing_method=cost_heat_exchanger)

    m.fs.chiller.costing = UnitModelCostingBlock(
        flowsheet_costing_block=m.fs.costing,
        costing_method=cost_heater_chiller,
        costing_method_arguments={"HC_type": "chiller"},
    )
    
    m.fs.mixer.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)
    m.fs.pump.costing = UnitModelCostingBlock(
        flowsheet_costing_block=m.fs.costing,
        costing_method=cost_pump,
        costing_method_arguments={"pump_type": "low_pressure"},
    )

    m.fs.costing.cost_process()
    m.fs.costing.add_annual_water_production(m.fs.distillate.properties[0].flow_vol)
    m.fs.costing.add_specific_energy_consumption(m.fs.distillate.properties[0].flow_vol)
    m.fs.costing.add_LCOW(m.fs.distillate.properties[0].flow_vol)

def set_operating_conditions(m):
    m.fs.feed.flow_mass_phase_comp[0, "Liq", "NaCl"].fix(9.5119 )
    m.fs.feed.flow_mass_phase_comp[0, "Liq", "H2O"].fix(38.9326 )
    m.fs.tb_heater_to_cryst.outlet.flow_mass_phase_comp[0, "Sol", "NaCl"].fix(1e-6)
    m.fs.tb_heater_to_cryst.outlet.flow_mass_phase_comp[0, "Vap", "H2O"].fix(1e-6)
    m.fs.feed.pressure[0].fix(101325)
    m.fs.feed.temperature[0].fix(273.15 + 20)


    m.fs.crystallizer.inlet.temperature[0].set_value(273.15 + 54 )
    #m.fs.crystallizer.solids.flow_mass_phase_comp[0, "Sol", "NaCl"].fix(5)
    m.fs.crystallizer.temperature_operating.set_value(273.15 + 50 )
    m.fs.heater.overall_heat_transfer_coefficient.fix(2e3)
    m.fs.heater.area.set_value(500)

    # Fix
    m.fs.crystallizer.crystal_growth_rate.fix()
    m.fs.crystallizer.souders_brown_constant.fix()
    m.fs.crystallizer.crystal_median_length.fix()

    m.fs.heater.hot_side_inlet.flow_mass_phase_comp[0, "Vap", "H2O"].set_value(30)
    m.fs.heater.hot_side_inlet.flow_mass_phase_comp[0, "Liq", "H2O"].fix(0)
    m.fs.heater.hot_side_inlet.temperature.fix(273.15 + 170)
    m.fs.heater.cold_side_outlet.temperature.fix(273.15 + 54)
    
    m.fs.heater.hot_side_inlet.pressure[0].fix(201325)

    m.fs.crystallizer.inlet.flow_mass_phase_comp[0, "Liq", "NaCl"].set_value(13.5119 *10 )
    m.fs.crystallizer.inlet.flow_mass_phase_comp[0, "Liq", "H2O"].set_value(38.9326 * 10 )
    #m.fs.crystallizer.inlet.temperature[0].set_value(273.15 + 80)
    m.fs.crystallizer.inlet.pressure[0].set_value(101325)

    #m.fs.heater.area.setlb(10)

    m.fs.pump.deltaP.fix(100000)
    m.fs.pump.efficiency_pump.fix(0.8)

    m.fs.crystallizer.solids.flow_mass_phase_comp[0, "Sol", "NaCl"].fix(2)

    m.fs.condenser.cold_side_inlet.pressure[0].fix(101325)
    m.fs.condenser.cold_side_inlet.temperature[0].fix(273.15 + 20)
    m.fs.condenser.cold_side_outlet.temperature[0].fix(273.15 + 35)
    m.fs.condenser.overall_heat_transfer_coefficient.fix(2e3)
    m.fs.condenser.cold_side_inlet.flow_mass_phase_comp[0, "Liq", "NaCl"].fix(
            1e-8
        )
    m.fs.condenser.cold_side_inlet.flow_mass_phase_comp[0, "Liq", "H2O"].set_value(
            12
        )
    m.fs.condenser.area.set_value(10)
    
    print("DOF final:", degrees_of_freedom(m.fs))

def solve(blk, solver=None, tee=True):
    if solver is None:
        solver = get_solver()
    results = solver.solve(blk, tee=tee)
    if not check_optimal_termination(results):
        results = solver.solve(blk, tee=tee)
    return results


def initialize_system(m, solver=None, verbose=True):
    if solver is None:
        solver = get_solver()

    # initialize feed block
    m.fs.feed.initialize()
    m.fs.crystallizer.initialize()


    #separator
    propagate_state(m.fs.s07)
    propagate_state(m.fs.s08)

    m.fs.separator.inlet.flow_mass_phase_comp[0,"Liq", "H2O"] = m.fs.crystallizer.outlet.flow_mass_phase_comp[0,"Liq", "H2O"].value
    m.fs.separator.inlet.flow_mass_phase_comp[0,"Liq", "NaCl"] = m.fs.crystallizer.outlet.flow_mass_phase_comp[0,"Liq", "NaCl"].value
    m.fs.separator.inlet.temperature[0] = m.fs.crystallizer.outlet.temperature[0].value
    m.fs.separator.inlet.pressure[0] = m.fs.crystallizer.outlet.pressure[0].value
    m.fs.separator.split_fraction[0, "purge"].fix(0.1)
   
    m.fs.separator.initialize()
    m.fs.separator.split_fraction[0, "purge"].unfix()

   #mixer
    propagate_state(m.fs.s01)
    propagate_state(m.fs.s02)
    
    m.fs.mixer.initialize()
    m.fs.mixer.pressure_equality_constraints[0, 2].deactivate()

    propagate_state(m.fs.s03)
    m.fs.pump.initialize()
  
    propagate_state(m.fs.s04)
    
    m.fs.heater.initialize()
    m.fs.heater.cold_side_inlet.unfix()
    
    propagate_state(m.fs.s05)
    propagate_state(m.fs.s06)
    m.fs.crystallizer.inlet.flow_mass_phase_comp[0,"Liq", "H2O"] = m.fs.heater.cold_side_outlet.flow_mass_phase_comp[0,"Liq", "H2O"].value
    m.fs.crystallizer.inlet.flow_mass_phase_comp[0,"Liq", "NaCl"] = m.fs.heater.cold_side_outlet.flow_mass_phase_comp[0,"Liq", "NaCl"].value
    m.fs.crystallizer.inlet.temperature[0] = m.fs.heater.cold_side_outlet.temperature[0].value
    m.fs.crystallizer.inlet.pressure[0] = m.fs.heater.cold_side_outlet.pressure[0].value

    m.fs.crystallizer.initialize()

    propagate_state(m.fs.s09)
    propagate_state(m.fs.s10)
    m.fs.condenser.hot_side_inlet.flow_mass_phase_comp[0,"Liq", "H2O"] = m.fs.crystallizer.vapor.flow_mass_phase_comp[0,"Liq", "H2O"].value
    m.fs.condenser.hot_side_inlet.flow_mass_phase_comp[0,"Vap", "H2O"] = m.fs.crystallizer.vapor.flow_mass_phase_comp[0,"Vap", "H2O"].value
    m.fs.condenser.hot_side_inlet.temperature[0] = m.fs.crystallizer.vapor.temperature[0].value
    m.fs.condenser.hot_side_inlet.pressure[0] = m.fs.crystallizer.vapor.pressure[0].value
    m.fs.condenser.initialize()
    m.fs.condenser.hot_side_inlet.unfix()

    propagate_state(m.fs.s11)
    m.fs.distillate.initialize()

    propagate_state(m.fs.s12)
    m.fs.chiller.initialize()


    m.fs.costing.initialize()
    print(f"DOF: {degrees_of_freedom(m)}")

def optimize_set_up(m):
    # add objective
    m.fs.objective = Objective(expr=m.fs.costing.LCOW)


    m.fs.heater.cold_side_outlet.temperature.unfix()
    m.fs.heater.cold_side_outlet.temperature.setlb(273.15 +50)
    m.fs.heater.cold_side_outlet.temperature.setub(273.15 +110)
  
    m.fs.mixer.recycle.flow_mass_phase_comp[0,"Liq", "H2O"].lb = 300
    m.fs.mixer.recycle.flow_mass_phase_comp[0,"Liq", "H2O"].set_value(400)
    

    print("DOF final 2:", degrees_of_freedom(m.fs))


def optimize(m, solver=None):
    # --solve---
    return solve(m, solver=solver)
def display_system(m):
    print("Crystalizer Inlet temperature:", m.fs.crystallizer.inlet.temperature[0].value)
    print("operating temperature:", m.fs.crystallizer.temperature_operating.value)
    print("recycle :", m.fs.mixer.recycle.flow_mass_phase_comp[0,"Liq", "H2O"].value)
    print("Levelized cost of water: %.2f $/m3" % value(m.fs.costing.LCOW))
    print("steam:", m.fs.heater.hot_side_inlet.flow_mass_phase_comp[0,"Vap", "H2O"].value)
    print("heater area:", m.fs.heater.area.value)

   


if __name__ == "__main__":
    m = main()
