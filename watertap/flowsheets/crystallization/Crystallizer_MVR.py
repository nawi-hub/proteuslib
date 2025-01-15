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
from pyomo.network import Arc
from idaes.core import FlowsheetBlock
from watertap.core.solvers import get_solver
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util.initialization import (
    propagate_state,
)
from watertap.unit_models.pressure_changer import Pump
from watertap.costing.unit_models.pump import (
    cost_pump,
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

from io import StringIO
from pyomo.util.infeasible import (
    log_infeasible_constraints,
)
from watertap.unit_models.steam_heater_0D import SteamHeater0D, Mode
from idaes.models.unit_models import Heater, Separator, Mixer, Product, Feed
from idaes.models.unit_models.mixer import MomentumMixingType
from watertap.core.solvers import get_solver
from watertap.unit_models.mvc.components.lmtd_chen_callback import (
    delta_temperature_chen_callback,
)
from idaes.models.unit_models.heat_exchanger import (
    HeatExchangerFlowPattern,
)
from watertap.unit_models.mvc.components import Compressor
from watertap.property_models.unit_specific import cryst_prop_pack as props
import watertap.property_models.water_prop_pack as props_w
import watertap.property_models.NaCl_T_dep_prop_pack as props_nacl
from pyomo.common.log import LoggingIntercept
from idaes.models.unit_models.translator import Translator
from idaes.models.unit_models.separator import SplittingType
import logging
#from watertap.core.util.model_debug_mode import activate
#activate()



__author__ = "Elmira Shamlou"


def main():
    solver = get_solver()
    m = build()
    set_operating_conditions(m)
    initialize_system(m, solver=solver)

    optimize_set_up(m)

    solve(m, solver=solver)

    display_system(m)
  

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


    # unit models: steam heater, mixer, pump, crystalizer, compressor, separator

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
        mode=Mode.CONDENSER,
    )
  
    m.fs.mixer = Mixer(
        property_package=m.fs.properties_nacl,
        momentum_mixing_type=MomentumMixingType.equality,
        inlet_list=["feed", "recycle"],
    )
    m.fs.mixer.pressure_equality_constraints[0, 2].deactivate()
    m.fs.compressor = Compressor(property_package=m.fs.properties_vapor)

    m.fs.pump = Pump(property_package=m.fs.properties_nacl)

    m.fs.separator = Separator(
        property_package=m.fs.properties_nacl,
        outlet_list=["recycle", "purge"],
        split_basis=SplittingType.totalFlow,
    )

    m.fs.tb_vapor = Translator(
        inlet_property_package=m.fs.properties,
        outlet_property_package=m.fs.properties_vapor,
    )

    @m.fs.tb_vapor.Constraint()
    def eq_flow_mass_comp(blk):
        return (
            blk.properties_in[0].flow_mass_phase_comp["Liq", "H2O"]
            == blk.properties_out[0].flow_mass_phase_comp["Liq", "H2O"]
        )
    @m.fs.tb_vapor.Constraint()
    def eq_flow_mass_comp_vapor(blk):
        return (
            blk.properties_in[0].flow_mass_phase_comp["Vap", "H2O"]
            == blk.properties_out[0].flow_mass_phase_comp["Vap", "H2O"]
        )

    @m.fs.tb_vapor.Constraint()
    def eq_temperature(blk):
        return blk.properties_in[0].temperature == blk.properties_out[0].temperature

    @m.fs.tb_vapor.Constraint()
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
    def eq_flow_mass_comp_vapor(blk):
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
    
    m.fs.tb_recycle = Translator(
        inlet_property_package=m.fs.properties,
        outlet_property_package=m.fs.properties_nacl,
    )

    @m.fs.tb_recycle.Constraint()
    def eq_flow_mass_comp(blk):
        return (
            blk.properties_in[0].flow_mass_phase_comp["Liq", "H2O"]
            == blk.properties_out[0].flow_mass_phase_comp["Liq", "H2O"]
        )
    @m.fs.tb_recycle.Constraint()
    def eq_flow_mass_comp_vapor(blk):
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
    
   

    # connections
    m.fs.s01 = Arc(source=m.fs.feed.outlet, destination=m.fs.mixer.feed)
    m.fs.s13 = Arc(
        source=m.fs.separator.recycle, destination=m.fs.mixer.recycle)
    m.fs.s11 = Arc(source=m.fs.mixer.outlet, destination=m.fs.pump.inlet)
    m.fs.s02 = Arc(source=m.fs.pump.outlet, destination=m.fs.heater.cold_side_inlet)
    m.fs.s03 = Arc(source=m.fs.heater.cold_side_outlet, destination=m.fs.tb_heater_to_cryst.inlet)
    m.fs.s04 = Arc(source=m.fs.tb_heater_to_cryst.outlet, destination=m.fs.crystallizer.inlet)
    m.fs.s05 = Arc(source=m.fs.crystallizer.outlet, destination=m.fs.tb_recycle.inlet)
    m.fs.s06 = Arc(source=m.fs.tb_recycle.outlet, destination=m.fs.separator.inlet)
    m.fs.s07 = Arc(source=m.fs.crystallizer.vapor, destination=m.fs.tb_vapor.inlet)
    m.fs.s08 = Arc(source=m.fs.tb_vapor.outlet, destination=m.fs.compressor.inlet)
    m.fs.s09 = Arc(source=m.fs.compressor.outlet, destination=m.fs.heater.hot_side_inlet)
    m.fs.s10 = Arc(source=m.fs.heater.hot_side_outlet, destination=m.fs.distillate.inlet)

    
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
    iscale.set_scaling_factor(m.fs.compressor.control_volume.work, 1e-6)

    iscale.calculate_scaling_factors(m)

    return m

def add_costs(m):

    m.fs.costing = WaterTAPCosting()
    m.fs.crystallizer.costing = UnitModelCostingBlock(
        flowsheet_costing_block=m.fs.costing,
        costing_method_arguments={"cost_type": CrystallizerCostType.mass_basis},
    )
    m.fs.heater.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)
    m.fs.mixer.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)

    m.fs.compressor.costing = UnitModelCostingBlock(
        flowsheet_costing_block=m.fs.costing
   )
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
    m.fs.feed.flow_mass_phase_comp[0, "Liq", "NaCl"].fix(10.5119 )
    m.fs.feed.flow_mass_phase_comp[0, "Liq", "H2O"].fix(38.9326 )
    m.fs.crystallizer.inlet.flow_mass_phase_comp[0, "Sol", "NaCl"].fix(1e-6)
    m.fs.crystallizer.inlet.flow_mass_phase_comp[0, "Vap", "H2O"].fix(1e-6)
    m.fs.feed.pressure[0].fix(101325)
    m.fs.feed.temperature[0].fix(273.15 + 20)


    m.fs.crystallizer.temperature_operating.set_value(273.15 + 50 )
    
    m.fs.heater.overall_heat_transfer_coefficient.fix(2e3)

    # Fix
    m.fs.crystallizer.crystal_growth_rate.fix()
    m.fs.crystallizer.souders_brown_constant.fix()
    m.fs.crystallizer.crystal_median_length.fix()

   

    m.fs.crystallizer.inlet.flow_mass_phase_comp[0, "Liq", "NaCl"].set_value(13.5119 *10 )
    m.fs.crystallizer.inlet.flow_mass_phase_comp[0, "Liq", "H2O"].set_value(38.9326 * 10 )
    m.fs.crystallizer.inlet.temperature[0].fix(273.15 + 55)
    m.fs.heater.cold_side_outlet.temperature[0].set_value(273.15 + 55)
    m.fs.heater.cold_side_inlet.temperature[0].set_value(273.15 + 50)
    m.fs.crystallizer.inlet.pressure[0].set_value(101325)

    m.fs.compressor.pressure_ratio.fix(1.5)
    m.fs.compressor.efficiency.fix(0.8)
    m.fs.compressor.pressure_ratio.setub(3)

    m.fs.pump.deltaP.fix(100000)
    m.fs.pump.efficiency_pump.fix(0.8)

    m.fs.crystallizer.solids.flow_mass_phase_comp[0, "Sol", "NaCl"].fix(2)

  
    
    print("DOF:", degrees_of_freedom(m.fs))

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
    
    propagate_state(m.fs.s07)
    propagate_state(m.fs.s08)
    m.fs.compressor.inlet.flow_mass_phase_comp[0,"Liq", "H2O"] = m.fs.crystallizer.vapor.flow_mass_phase_comp[0,"Liq", "H2O"].value
    m.fs.compressor.inlet.flow_mass_phase_comp[0,"Vap", "H2O"] = m.fs.crystallizer.vapor.flow_mass_phase_comp[0,"Vap", "H2O"].value
    m.fs.compressor.inlet.temperature[0] = m.fs.crystallizer.vapor.temperature[0].value
    m.fs.compressor.inlet.pressure[0] = m.fs.crystallizer.vapor.pressure[0].value
    m.fs.compressor.initialize()

    propagate_state(m.fs.s01)
   
    propagate_state(m.fs.s05)
    propagate_state(m.fs.s06)

    m.fs.separator.inlet.flow_mass_phase_comp[0,"Liq", "H2O"] = m.fs.crystallizer.outlet.flow_mass_phase_comp[0,"Liq", "H2O"].value
    m.fs.separator.inlet.flow_mass_phase_comp[0,"Liq", "NaCl"] = m.fs.crystallizer.outlet.flow_mass_phase_comp[0,"Liq", "NaCl"].value
    m.fs.separator.inlet.temperature[0] = m.fs.crystallizer.outlet.temperature[0].value
    m.fs.separator.inlet.pressure[0] = m.fs.crystallizer.outlet.pressure[0].value
    m.fs.separator.split_fraction[0, "purge"].fix(0.1)
   
    m.fs.separator.initialize()
    m.fs.separator.split_fraction[0, "purge"].unfix()
    
   
    propagate_state(m.fs.s13)
    m.fs.mixer.initialize()
    m.fs.mixer.pressure_equality_constraints[0, 2].deactivate()
    print(f"DOF: {degrees_of_freedom(m)}")

    propagate_state(m.fs.s11)
    m.fs.pump.initialize()

    propagate_state(m.fs.s02)
    propagate_state(m.fs.s09)
    m.fs.heater.initialize()
    m.fs.heater.cold_side_inlet.unfix()
    m.fs.heater.hot_side_inlet.unfix()
    print(f"DOF heater: {degrees_of_freedom(m)}")
   
    propagate_state(m.fs.s03)
    propagate_state(m.fs.s04)
    m.fs.crystallizer.inlet.flow_mass_phase_comp[0,"Liq", "H2O"] = m.fs.heater.cold_side_outlet.flow_mass_phase_comp[0,"Liq", "H2O"].value
    m.fs.crystallizer.inlet.flow_mass_phase_comp[0,"Liq", "NaCl"] = m.fs.heater.cold_side_outlet.flow_mass_phase_comp[0,"Liq", "NaCl"].value
    m.fs.crystallizer.inlet.temperature[0] = m.fs.heater.cold_side_outlet.temperature[0].value
    m.fs.crystallizer.inlet.pressure[0] = m.fs.heater.cold_side_outlet.pressure[0].value
    m.fs.crystallizer.initialize()
    print(f"DOF: {degrees_of_freedom(m)}")
    propagate_state(m.fs.s10)
    m.fs.distillate.initialize()
    print("DOF final:", degrees_of_freedom(m.fs))
    propagate_state(m.fs.s07)
    propagate_state(m.fs.s08)
    m.fs.compressor.inlet.flow_mass_phase_comp[0,"Liq", "H2O"] = m.fs.crystallizer.vapor.flow_mass_phase_comp[0,"Liq", "H2O"].value
    m.fs.compressor.inlet.flow_mass_phase_comp[0,"Vap", "H2O"] = m.fs.crystallizer.vapor.flow_mass_phase_comp[0,"Vap", "H2O"].value
    m.fs.compressor.inlet.temperature[0] = m.fs.crystallizer.vapor.temperature[0].value
    m.fs.compressor.inlet.pressure[0] = m.fs.crystallizer.vapor.pressure[0].value
    m.fs.compressor.initialize()
    m.fs.costing.initialize()
    


def optimize_set_up(m):
    m.fs.objective = Objective(expr=m.fs.costing.LCOW)
    m.fs.compressor.pressure_ratio.unfix()
    m.fs.compressor.pressure_ratio.setlb(1)
    m.fs.compressor.pressure_ratio.setub(3)
    #additional constraint
    m.fs.eq_heater_temperature_rise = Constraint(
         expr=m.fs.heater.cold_side_outlet.temperature[0] - m.fs.heater.cold_side_inlet.temperature[0] <= 4)

    m.fs.crystallizer.inlet.temperature.unfix()
    m.fs.crystallizer.inlet.temperature.setlb(273.15 +50)
    m.fs.crystallizer.inlet.temperature.setub(273.15 +110)
    m.fs.mixer.recycle.flow_mass_phase_comp[0,"Liq", "H2O"].lb = 100
    m.fs.mixer.recycle.flow_mass_phase_comp[0,"Liq", "H2O"].set_value(500)


def optimize(m, solver=None):
    # --solve---
    return solve(m, solver=solver)
   
def display_system(m):
    print("operating temperature:", m.fs.crystallizer.temperature_operating.value)
    print("Levelized cost of water: %.2f $/m3" % value(m.fs.costing.LCOW))
    print("Inlet temperature heater:", m.fs.heater.cold_side_inlet.temperature[0].value)
    print("compressor pressure ratio:", m.fs.compressor.pressure_ratio.value)
    print("separator split recycle:", m.fs.separator.recycle.flow_mass_phase_comp[0,"Liq", "H2O"].value)
    print("separator split purge:", m.fs.separator.purge.flow_mass_phase_comp[0,"Liq", "H2O"].value)
    print("distillate:", m.fs.distillate.flow_mass_phase_comp[0,"Liq", "H2O"].value)
if __name__ == "__main__":
    m = main()

