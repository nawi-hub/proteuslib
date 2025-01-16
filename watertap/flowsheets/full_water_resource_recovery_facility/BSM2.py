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
"""
Flowsheet example full Water Resource Recovery Facility
(WRRF; a.k.a., wastewater treatment plant) with ASM1 and ADM1.

The flowsheet follows the same formulation as benchmark simulation model no.2 (BSM2)
but comprises different specifications for default values than BSM2.

"""
__author__ = "Alejandro Garciadiego, Adam Atia, Marcus Holly, Chenyu Wang, Ben Knueven, Xinhong Liu,"

import pyomo.environ as pyo

from pyomo.network import Arc, SequentialDecomposition
from watertap.unit_models.anaerobic_digester import AD
from watertap.unit_models.thickener import Thickener
from watertap.unit_models.dewatering import DewateringUnit
from watertap.unit_models.cstr import CSTR
from watertap.unit_models.clarifier import Clarifier

from watertap.unit_models.translators.translator_asm1_adm1 import Translator_ASM1_ADM1
from watertap.unit_models.translators.translator_adm1_asm1 import Translator_ADM1_ASM1

import idaes.logger as idaeslog
from watertap.core.solvers import get_solver
import idaes.core.util.scaling as iscale
from watertap.property_models.unit_specific.anaerobic_digestion.adm1_properties import (
    ADM1ParameterBlock,
)
from watertap.property_models.unit_specific.anaerobic_digestion.adm1_reactions import (
    ADM1ReactionParameterBlock,
)
from idaes.models.unit_models.mixer import MomentumMixingType
from idaes.models.unit_models.separator import SplittingType
from watertap.property_models.unit_specific.anaerobic_digestion.adm1_properties_vapor import (
    ADM1_vaporParameterBlock,
)

from idaes.core import FlowsheetBlock, UnitModelCostingBlock, UnitModelBlockData
from idaes.models.unit_models import (
    Feed,
    Mixer,
    Separator,
    PressureChanger,
    Product,
)

from watertap.unit_models.aeration_tank import AerationTank, ElectricityConsumption
from watertap.property_models.unit_specific.activated_sludge.asm1_properties import (
    ASM1ParameterBlock,
)
from watertap.property_models.unit_specific.activated_sludge.asm1_reactions import (
    ASM1ReactionParameterBlock,
)
from watertap.core.util.initialization import assert_degrees_of_freedom
from watertap.costing import WaterTAPCosting
from watertap.costing.unit_models.clarifier import (
    cost_circular_clarifier,
    cost_primary_clarifier,
)
from pyomo.util.check_units import assert_units_consistent


def main(reactor_volume_equalities=False):
    m = build()
    set_operating_conditions(m)

    assert_degrees_of_freedom(m, 0)
    assert_units_consistent(m)
    initialize_system(m)

    assert_degrees_of_freedom(m, 0)

    results = solve(m)

    add_costing(m)
    m.fs.costing.initialize()
    assert_degrees_of_freedom(m, 0)

    results = solve(m)
    pyo.assert_optimal_termination(results)

    print("\n\n=============SIMULATION RESULTS=============\n\n")
    # display_results(m)
    display_costing(m)

    setup_optimization(m, reactor_volume_equalities=reactor_volume_equalities)
    results = solve(m, tee=True)
    pyo.assert_optimal_termination(results)
    print("\n\n=============OPTIMIZATION RESULTS=============\n\n")
    # display_results(m)
    display_costing(m)
    display_performance_metrics(m)

    return m, results


def build():
    m = pyo.ConcreteModel()

    m.fs = FlowsheetBlock(dynamic=False)

    m.fs.props_ASM1 = ASM1ParameterBlock()
    m.fs.props_ADM1 = ADM1ParameterBlock()
    m.fs.props_vap = ADM1_vaporParameterBlock()
    m.fs.ADM1_rxn_props = ADM1ReactionParameterBlock(property_package=m.fs.props_ADM1)
    m.fs.ASM1_rxn_props = ASM1ReactionParameterBlock(property_package=m.fs.props_ASM1)
    # Feed water stream
    m.fs.FeedWater = Feed(property_package=m.fs.props_ASM1)

    # ==========================================================================
    # Activated Sludge Process
    # ==========================================================================
    # Mixer for inlet water and recycled sludge
    m.fs.MX1 = Mixer(
        property_package=m.fs.props_ASM1,
        inlet_list=["feed_water", "recycle"],
        momentum_mixing_type=MomentumMixingType.equality,
    )
    # First reactor (anoxic) - standard CSTR
    m.fs.R1 = CSTR(
        property_package=m.fs.props_ASM1, reaction_package=m.fs.ASM1_rxn_props
    )
    # Second reactor (anoxic) - standard CSTR
    m.fs.R2 = CSTR(
        property_package=m.fs.props_ASM1, reaction_package=m.fs.ASM1_rxn_props
    )
    # Third reactor (aerobic) - CSTR with injection
    m.fs.R3 = AerationTank(
        property_package=m.fs.props_ASM1,
        reaction_package=m.fs.ASM1_rxn_props,
        electricity_consumption=ElectricityConsumption.calculated,
    )
    # Fourth reactor (aerobic) - CSTR with injection
    m.fs.R4 = AerationTank(
        property_package=m.fs.props_ASM1,
        reaction_package=m.fs.ASM1_rxn_props,
        electricity_consumption=ElectricityConsumption.calculated,
    )
    # Fifth reactor (aerobic) - CSTR with injection
    m.fs.R5 = AerationTank(
        property_package=m.fs.props_ASM1,
        reaction_package=m.fs.ASM1_rxn_props,
        electricity_consumption=ElectricityConsumption.calculated,
    )
    m.fs.SP5 = Separator(
        property_package=m.fs.props_ASM1, outlet_list=["underflow", "overflow"]
    )
    # Clarifier
    # TODO: Replace with more detailed model when available
    m.fs.CL1 = Clarifier(
        property_package=m.fs.props_ASM1,
        outlet_list=["underflow", "effluent"],
        split_basis=SplittingType.componentFlow,
    )
    # Sludge purge splitter
    m.fs.SP6 = Separator(
        property_package=m.fs.props_ASM1,
        outlet_list=["recycle", "waste"],
        split_basis=SplittingType.totalFlow,
    )
    # Mixing sludge recycle and R5 underflow
    m.fs.MX6 = Mixer(
        property_package=m.fs.props_ASM1,
        inlet_list=["clarifier", "reactor"],
        momentum_mixing_type=MomentumMixingType.equality,
    )

    # Product Blocks
    m.fs.Treated = Product(property_package=m.fs.props_ASM1)
    m.fs.Sludge = Product(property_package=m.fs.props_ASM1)
    # Recycle pressure changer - use a simple isothermal unit for now
    m.fs.P1 = PressureChanger(property_package=m.fs.props_ASM1)

    # Link units
    m.fs.stream2 = Arc(source=m.fs.MX1.outlet, destination=m.fs.R1.inlet)
    m.fs.stream3 = Arc(source=m.fs.R1.outlet, destination=m.fs.R2.inlet)
    m.fs.stream4 = Arc(source=m.fs.R2.outlet, destination=m.fs.R3.inlet)
    m.fs.stream5 = Arc(source=m.fs.R3.outlet, destination=m.fs.R4.inlet)
    m.fs.stream6 = Arc(source=m.fs.R4.outlet, destination=m.fs.R5.inlet)
    m.fs.stream7 = Arc(source=m.fs.R5.outlet, destination=m.fs.SP5.inlet)
    m.fs.stream8 = Arc(source=m.fs.SP5.overflow, destination=m.fs.CL1.inlet)
    m.fs.stream9 = Arc(source=m.fs.SP5.underflow, destination=m.fs.MX6.reactor)
    m.fs.stream10 = Arc(source=m.fs.CL1.effluent, destination=m.fs.Treated.inlet)
    m.fs.stream11 = Arc(source=m.fs.CL1.underflow, destination=m.fs.SP6.inlet)
    m.fs.stream13 = Arc(source=m.fs.SP6.recycle, destination=m.fs.MX6.clarifier)
    m.fs.stream14 = Arc(source=m.fs.MX6.outlet, destination=m.fs.P1.inlet)
    m.fs.stream15 = Arc(source=m.fs.P1.outlet, destination=m.fs.MX1.recycle)
    pyo.TransformationFactory("network.expand_arcs").apply_to(m)

    # Oxygen concentration in reactors 3 and 4 is governed by mass transfer
    m.fs.R3.KLa = 7.6
    m.fs.R4.KLa = 5.7

    # ======================================================================
    # Anaerobic digester section
    m.fs.asm_adm = Translator_ASM1_ADM1(
        inlet_property_package=m.fs.props_ASM1,
        outlet_property_package=m.fs.props_ADM1,
        reaction_package=m.fs.ADM1_rxn_props,
        has_phase_equilibrium=False,
        outlet_state_defined=True,
    )

    m.fs.RADM = AD(
        liquid_property_package=m.fs.props_ADM1,
        vapor_property_package=m.fs.props_vap,
        reaction_package=m.fs.ADM1_rxn_props,
        has_heat_transfer=True,
        has_pressure_change=False,
    )

    m.fs.adm_asm = Translator_ADM1_ASM1(
        inlet_property_package=m.fs.props_ADM1,
        outlet_property_package=m.fs.props_ASM1,
        reaction_package=m.fs.ADM1_rxn_props,
        has_phase_equilibrium=False,
        outlet_state_defined=True,
    )

    # ====================================================================
    # Primary Clarifier
    m.fs.CL = Clarifier(
        property_package=m.fs.props_ASM1,
        outlet_list=["underflow", "effluent"],
        split_basis=SplittingType.componentFlow,
    )

    # Thickener
    m.fs.TU = Thickener(property_package=m.fs.props_ASM1)
    # Dewaterer
    m.fs.DU = DewateringUnit(property_package=m.fs.props_ASM1)

    m.fs.MX2 = Mixer(
        property_package=m.fs.props_ASM1,
        inlet_list=["feed_water1", "recycle1"],
        momentum_mixing_type=MomentumMixingType.equality,
    )
    m.fs.MX3 = Mixer(
        property_package=m.fs.props_ASM1,
        inlet_list=["feed_water2", "recycle2"],
        momentum_mixing_type=MomentumMixingType.equality,
    )
    m.fs.MX4 = Mixer(
        property_package=m.fs.props_ASM1,
        inlet_list=["thickener", "clarifier"],
        momentum_mixing_type=MomentumMixingType.equality,
    )

    # Make connections related to AD section
    m.fs.stream2adm = Arc(
        source=m.fs.RADM.liquid_outlet, destination=m.fs.adm_asm.inlet
    )
    m.fs.stream6adm = Arc(source=m.fs.SP6.waste, destination=m.fs.TU.inlet)
    m.fs.stream3adm = Arc(source=m.fs.TU.underflow, destination=m.fs.MX4.thickener)
    m.fs.stream7adm = Arc(source=m.fs.TU.overflow, destination=m.fs.MX3.recycle2)
    m.fs.stream9adm = Arc(source=m.fs.CL.underflow, destination=m.fs.MX4.clarifier)
    m.fs.stream4adm = Arc(source=m.fs.adm_asm.outlet, destination=m.fs.DU.inlet)
    m.fs.stream5adm = Arc(source=m.fs.DU.overflow, destination=m.fs.MX2.recycle1)
    m.fs.stream11adm = Arc(source=m.fs.DU.underflow, destination=m.fs.Sludge.inlet)
    m.fs.stream01 = Arc(source=m.fs.FeedWater.outlet, destination=m.fs.MX2.feed_water1)
    m.fs.stream02 = Arc(source=m.fs.MX2.outlet, destination=m.fs.MX3.feed_water2)
    m.fs.stream03 = Arc(source=m.fs.MX3.outlet, destination=m.fs.CL.inlet)
    m.fs.stream04 = Arc(source=m.fs.CL.effluent, destination=m.fs.MX1.feed_water)
    m.fs.stream10adm = Arc(source=m.fs.MX4.outlet, destination=m.fs.asm_adm.inlet)
    m.fs.stream1adm = Arc(source=m.fs.asm_adm.outlet, destination=m.fs.RADM.inlet)

    pyo.TransformationFactory("network.expand_arcs").apply_to(m)

    # keep handy all the mixers
    m.mixers = (m.fs.MX1, m.fs.MX2, m.fs.MX3, m.fs.MX4, m.fs.MX6)

    return m


def set_operating_conditions(m):
    # Feed Water Conditions
    m.fs.FeedWater.flow_vol.fix(20648 * pyo.units.m**3 / pyo.units.day)
    m.fs.FeedWater.temperature.fix(308.15 * pyo.units.K)
    m.fs.FeedWater.pressure.fix(1 * pyo.units.atm)
    m.fs.FeedWater.conc_mass_comp[0, "S_I"].fix(27 * pyo.units.g / pyo.units.m**3)
    m.fs.FeedWater.conc_mass_comp[0, "S_S"].fix(58 * pyo.units.g / pyo.units.m**3)
    m.fs.FeedWater.conc_mass_comp[0, "X_I"].fix(92 * pyo.units.g / pyo.units.m**3)
    m.fs.FeedWater.conc_mass_comp[0, "X_S"].fix(363 * pyo.units.g / pyo.units.m**3)
    m.fs.FeedWater.conc_mass_comp[0, "X_BH"].fix(50 * pyo.units.g / pyo.units.m**3)
    m.fs.FeedWater.conc_mass_comp[0, "X_BA"].fix(0 * pyo.units.g / pyo.units.m**3)
    m.fs.FeedWater.conc_mass_comp[0, "X_P"].fix(0 * pyo.units.g / pyo.units.m**3)
    m.fs.FeedWater.conc_mass_comp[0, "S_O"].fix(0 * pyo.units.g / pyo.units.m**3)
    m.fs.FeedWater.conc_mass_comp[0, "S_NO"].fix(0 * pyo.units.g / pyo.units.m**3)
    m.fs.FeedWater.conc_mass_comp[0, "S_NH"].fix(23 * pyo.units.g / pyo.units.m**3)
    m.fs.FeedWater.conc_mass_comp[0, "S_ND"].fix(5 * pyo.units.g / pyo.units.m**3)
    m.fs.FeedWater.conc_mass_comp[0, "X_ND"].fix(16 * pyo.units.g / pyo.units.m**3)
    m.fs.FeedWater.alkalinity.fix(7 * pyo.units.mol / pyo.units.m**3)

    # Reactor sizing in activated sludge process
    m.fs.R1.volume.fix(1000 * pyo.units.m**3)
    m.fs.R2.volume.fix(1000 * pyo.units.m**3)
    m.fs.R3.volume.fix(1333 * pyo.units.m**3)
    m.fs.R4.volume.fix(1333 * pyo.units.m**3)
    m.fs.R5.volume.fix(1333 * pyo.units.m**3)

    # Injection rates to Reactors 3, 4 and 5 of the activated sludge process
    for j in m.fs.props_ASM1.component_list:
        if j != "S_O":
            # All components except S_O have no injection
            m.fs.R3.injection[:, :, j].fix(0)
            m.fs.R4.injection[:, :, j].fix(0)
            m.fs.R5.injection[:, :, j].fix(0)
    # Then set injections rates for O2
    m.fs.R3.outlet.conc_mass_comp[:, "S_O"].fix(1.72e-3)
    m.fs.R4.outlet.conc_mass_comp[:, "S_O"].fix(2.43e-3)
    m.fs.R5.outlet.conc_mass_comp[:, "S_O"].fix(4.49e-4)

    # Set fraction of outflow from reactor 5 that goes to recycle
    m.fs.SP5.split_fraction[:, "underflow"].fix(0.6)

    # Secondary clarifier
    # TODO: Update once secondary clarifier with more detailed model available
    m.fs.CL1.split_fraction[0, "effluent", "H2O"].fix(0.48956)
    m.fs.CL1.split_fraction[0, "effluent", "S_I"].fix(0.48956)
    m.fs.CL1.split_fraction[0, "effluent", "S_S"].fix(0.48956)
    m.fs.CL1.split_fraction[0, "effluent", "X_I"].fix(0.00187)
    m.fs.CL1.split_fraction[0, "effluent", "X_S"].fix(0.00187)
    m.fs.CL1.split_fraction[0, "effluent", "X_BH"].fix(0.00187)
    m.fs.CL1.split_fraction[0, "effluent", "X_BA"].fix(0.00187)
    m.fs.CL1.split_fraction[0, "effluent", "X_P"].fix(0.00187)
    m.fs.CL1.split_fraction[0, "effluent", "S_O"].fix(0.48956)
    m.fs.CL1.split_fraction[0, "effluent", "S_NO"].fix(0.48956)
    m.fs.CL1.split_fraction[0, "effluent", "S_NH"].fix(0.48956)
    m.fs.CL1.split_fraction[0, "effluent", "S_ND"].fix(0.48956)
    m.fs.CL1.split_fraction[0, "effluent", "X_ND"].fix(0.00187)
    m.fs.CL1.split_fraction[0, "effluent", "S_ALK"].fix(0.48956)

    m.fs.CL1.surface_area.fix(1500 * pyo.units.m**2)

    # Sludge purge separator
    m.fs.SP6.split_fraction[:, "recycle"].fix(0.985)

    # Outlet pressure from recycle pump
    m.fs.P1.outlet.pressure.fix(101325)

    # Primary Clarifier
    # TODO: Update primary clarifier once more detailed model available
    m.fs.CL.split_fraction[0, "effluent", "H2O"].fix(0.993)
    m.fs.CL.split_fraction[0, "effluent", "S_I"].fix(0.993)
    m.fs.CL.split_fraction[0, "effluent", "S_S"].fix(0.993)
    m.fs.CL.split_fraction[0, "effluent", "X_I"].fix(0.5192)
    m.fs.CL.split_fraction[0, "effluent", "X_S"].fix(0.5192)
    m.fs.CL.split_fraction[0, "effluent", "X_BH"].fix(0.5192)
    m.fs.CL.split_fraction[0, "effluent", "X_BA"].fix(0.5192)
    m.fs.CL.split_fraction[0, "effluent", "X_P"].fix(0.5192)
    m.fs.CL.split_fraction[0, "effluent", "S_O"].fix(0.993)
    m.fs.CL.split_fraction[0, "effluent", "S_NO"].fix(0.993)
    m.fs.CL.split_fraction[0, "effluent", "S_NH"].fix(0.993)
    m.fs.CL.split_fraction[0, "effluent", "S_ND"].fix(0.993)
    m.fs.CL.split_fraction[0, "effluent", "X_ND"].fix(0.5192)
    m.fs.CL.split_fraction[0, "effluent", "S_ALK"].fix(0.993)

    # Anaerobic digester
    m.fs.RADM.volume_liquid.fix(3400)
    m.fs.RADM.volume_vapor.fix(300)
    m.fs.RADM.liquid_outlet.temperature.fix(308.15)

    # Dewatering Unit - fix either HRT or volume.
    m.fs.DU.hydraulic_retention_time.fix(1800 * pyo.units.s)

    # Set specific energy consumption averaged for centrifuge
    m.fs.DU.energy_electric_flow_vol_inlet[0] = 0.069 * pyo.units.kWh / pyo.units.m**3

    # Thickener unit
    m.fs.TU.hydraulic_retention_time.fix(86400 * pyo.units.s)
    m.fs.TU.diameter.fix(10 * pyo.units.m)

    # TODO: resolve the danger of redundant constraint related to pressure equality constraints created in mixer, specifically for isobaric conditions. the mixer initializer will turn these constraints back on
    for mx in m.mixers:
        mx.pressure_equality_constraints[0.0, 2].deactivate()

    for var in m.fs.component_data_objects(pyo.Var, descend_into=True):
        if "flow_vol" in var.name:
            iscale.set_scaling_factor(var, 1e1)
        if "temperature" in var.name:
            iscale.set_scaling_factor(var, 1e-1)
        if "pressure" in var.name:
            iscale.set_scaling_factor(var, 1e-6)
        if "conc_mass_comp" in var.name:
            iscale.set_scaling_factor(var, 1e1)

    iscale.calculate_scaling_factors(m)


def initialize_system(m):
    # Initialize flowsheet
    # Apply sequential decomposition - 1 iteration should suffice
    seq = SequentialDecomposition()
    seq.options.tear_method = "Direct"
    seq.options.iterLim = 1
    seq.options.tear_set = [m.fs.stream2, m.fs.stream10adm]

    G = seq.create_graph(m)
    # The code below shows tear set and initialization order
    order = seq.calculation_order(G)
    print("Initialization Order")
    for o in order:
        print(o[0].name)

    # Initial guesses for flow into first reactor
    tear_guesses1 = {
        "flow_vol": {0: 103531 / 24 / 3600},
        "conc_mass_comp": {
            (0, "S_I"): 0.028,
            (0, "S_S"): 0.012,
            (0, "X_I"): 1.532,
            (0, "X_S"): 0.069,
            (0, "X_BH"): 2.233,
            (0, "X_BA"): 0.167,
            (0, "X_P"): 0.964,
            (0, "S_O"): 0.0011,
            (0, "S_NO"): 0.0073,
            (0, "S_NH"): 0.0072,
            (0, "S_ND"): 0.0016,
            (0, "X_ND"): 0.0040,
        },
        "alkalinity": {0: 0.0052},
        "temperature": {0: 308.15},
        "pressure": {0: 101325},
    }

    tear_guesses2 = {
        "flow_vol": {0: 178 / 24 / 3600},
        "conc_mass_comp": {
            (0, "S_I"): 0.028,
            (0, "S_S"): 0.048,
            (0, "X_I"): 10.362,
            (0, "X_S"): 20.375,
            (0, "X_BH"): 10.210,
            (0, "X_BA"): 0.553,
            (0, "X_P"): 3.204,
            (0, "S_O"): 0.00025,
            (0, "S_NO"): 0.00169,
            (0, "S_NH"): 0.0289,
            (0, "S_ND"): 0.00468,
            (0, "X_ND"): 0.906,
        },
        "alkalinity": {0: 0.00715},
        "temperature": {0: 308.15},
        "pressure": {0: 101325},
    }

    # Pass the tear_guess to the SD tool
    seq.set_guesses_for(m.fs.R1.inlet, tear_guesses1)
    seq.set_guesses_for(m.fs.asm_adm.inlet, tear_guesses2)

    def function(unit):
        unit.initialize(outlvl=idaeslog.INFO_HIGH)

    seq.run(m, function)

    # TODO: resolve the danger of redundant constraint related to pressure equality constraints created in mixer, specifically for isobaric conditions. the mixer initializer will turn these constraints back on
    for mx in m.mixers:
        mx.pressure_equality_constraints[0.0, 2].deactivate()


def add_costing(m):
    m.fs.costing = WaterTAPCosting()
    m.fs.costing.base_currency = pyo.units.USD_2020

    # Costing Blocks
    m.fs.R1.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)
    m.fs.R2.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)
    m.fs.R3.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)
    m.fs.R4.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)
    m.fs.R5.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)
    m.fs.CL.costing = UnitModelCostingBlock(
        flowsheet_costing_block=m.fs.costing,
        costing_method=cost_primary_clarifier,
    )

    m.fs.CL1.costing = UnitModelCostingBlock(
        flowsheet_costing_block=m.fs.costing,
        costing_method=cost_circular_clarifier,
    )

    m.fs.RADM.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)
    m.fs.DU.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)
    m.fs.TU.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)

    # TODO: Leaving out mixer costs; consider including later

    # process costing and add system level metrics
    m.fs.costing.cost_process()
    m.fs.costing.add_annual_water_production(m.fs.Treated.properties[0].flow_vol)
    m.fs.costing.add_LCOW(m.fs.FeedWater.properties[0].flow_vol)
    m.fs.costing.add_specific_energy_consumption(m.fs.FeedWater.properties[0].flow_vol)

    m.fs.objective = pyo.Objective(expr=m.fs.costing.LCOW)
    iscale.set_scaling_factor(m.fs.costing.LCOW, 1e3)
    iscale.set_scaling_factor(m.fs.costing.total_capital_cost, 1e-5)

    for block in m.fs.component_objects(pyo.Block, descend_into=True):
        if isinstance(block, UnitModelBlockData) and hasattr(block, "costing"):
            iscale.set_scaling_factor(block.costing.capital_cost, 1e-6)

    iscale.constraint_scaling_transform(m.fs.DU.costing.capital_cost_constraint, 1e-6)
    iscale.constraint_scaling_transform(m.fs.RADM.costing.capital_cost_constraint, 1e-6)


def setup_optimization(m, reactor_volume_equalities=False):

    for i in ["R1", "R2", "R3", "R4", "R5"]:
        reactor = getattr(m.fs, i)
        reactor.volume.unfix()
        reactor.volume.setlb(1)
        # reactor.volume.setub(2000)
    if reactor_volume_equalities:
        add_reactor_volume_equalities(m)
    m.fs.R3.outlet.conc_mass_comp[:, "S_O"].unfix()
    m.fs.R3.outlet.conc_mass_comp[:, "S_O"].setub(8e-3)

    m.fs.R4.outlet.conc_mass_comp[:, "S_O"].unfix()
    m.fs.R4.outlet.conc_mass_comp[:, "S_O"].setub(8e-3)

    m.fs.R5.outlet.conc_mass_comp[:, "S_O"].unfix()
    m.fs.R5.outlet.conc_mass_comp[:, "S_O"].setub(8e-3)

    # Unfix fraction of outflow from reactor 5 that goes to recycle
    m.fs.SP5.split_fraction[:, "underflow"].unfix()
    # m.fs.SP5.split_fraction[:, "underflow"].setlb(0.45)
    m.fs.SP6.split_fraction[:, "recycle"].unfix()

    add_effluent_violations(m)


def add_effluent_violations(m):
    # TODO: update "m" to blk; change ref to m.fs.Treated instead of CL1 effluent
    m.fs.TSS_max = pyo.Var(initialize=0.03, units=pyo.units.kg / pyo.units.m**3)
    m.fs.TSS_max.fix()

    @m.fs.Constraint(m.fs.time)
    def eq_TSS_max(self, t):
        return m.fs.CL1.effluent_state[0].TSS <= m.fs.TSS_max

    m.fs.COD_max = pyo.Var(initialize=0.1, units=pyo.units.kg / pyo.units.m**3)
    m.fs.COD_max.fix()

    @m.fs.Constraint(m.fs.time)
    def eq_COD_max(self, t):
        return m.fs.CL1.effluent_state[0].COD <= m.fs.COD_max

    m.fs.totalN_max = pyo.Var(initialize=0.018, units=pyo.units.kg / pyo.units.m**3)
    m.fs.totalN_max.fix()

    @m.fs.Constraint(m.fs.time)
    def eq_totalN_max(self, t):
        return m.fs.CL1.effluent_state[0].Total_N <= m.fs.totalN_max

    m.fs.BOD5_max = pyo.Var(initialize=0.01, units=pyo.units.kg / pyo.units.m**3)
    m.fs.BOD5_max.fix()

    @m.fs.Constraint(m.fs.time)
    def eq_BOD5_max(self, t):
        return m.fs.CL1.effluent_state[0].BOD5["effluent"] <= m.fs.BOD5_max


def add_reactor_volume_equalities(m):
    # TODO: These constraints were applied for initial optimization of AS reactor volumes; otherwise, volumes drive towards lower bound. Revisit
    @m.fs.Constraint(m.fs.time)
    def Vol_1(self, t):
        return m.fs.R1.volume[0] == m.fs.R2.volume[0]

    @m.fs.Constraint(m.fs.time)
    def Vol_2(self, t):
        return m.fs.R3.volume[0] == m.fs.R4.volume[0]

    @m.fs.Constraint(m.fs.time)
    def Vol_3(self, t):
        return m.fs.R5.volume[0] >= m.fs.R4.volume[0] * 0.5


def solve(blk, solver=None, tee=False):
    if solver is None:
        solver = get_solver()
    results = solver.solve(blk, tee=tee)
    pyo.assert_optimal_termination(results)
    return results


def display_results(m):
    m.display()

    unit_list = [
        "FeedWater",
        "MX1",
        "R1",
        "R2",
        "R3",
        "R4",
        "R5",
        "SP5",
        "CL1",
        "SP6",
        "MX6",
        "Treated",
        "Sludge",
        "P1",
        "asm_adm",
        "RADM",
        "adm_asm",
        "CL",
        "TU",
        "DU",
        "MX2",
        "MX3",
        "MX4",
    ]
    for u in unit_list:
        m.fs.component(u).report()


def display_costing(m):
    print("Levelized cost of water: %.2f $/m3" % pyo.value(m.fs.costing.LCOW))

    print(
        "Total operating cost: %.2f $/yr" % pyo.value(m.fs.costing.total_operating_cost)
    )
    print("Total capital cost: %.2f $" % pyo.value(m.fs.costing.total_capital_cost))

    print(
        "Total annualized cost: %.2f $/yr"
        % pyo.value(m.fs.costing.total_annualized_cost)
    )
    print(
        "capital cost R1",
        pyo.value(m.fs.R1.costing.capital_cost),
        pyo.units.get_units(m.fs.R1.costing.capital_cost),
    )
    print(
        "capital cost R2",
        pyo.value(m.fs.R2.costing.capital_cost),
        pyo.units.get_units(m.fs.R2.costing.capital_cost),
    )
    print(
        "capital cost R3",
        pyo.value(m.fs.R3.costing.capital_cost),
        pyo.units.get_units(m.fs.R3.costing.capital_cost),
    )
    print(
        "capital cost R4",
        pyo.value(m.fs.R4.costing.capital_cost),
        pyo.units.get_units(m.fs.R4.costing.capital_cost),
    )
    print(
        "capital cost R5",
        pyo.value(m.fs.R5.costing.capital_cost),
        pyo.units.get_units(m.fs.R5.costing.capital_cost),
    )
    print(
        "capital cost primary clarifier",
        pyo.value(m.fs.CL.costing.capital_cost),
        pyo.units.get_units(m.fs.CL.costing.capital_cost),
    )
    print(
        "capital cost secondary clarifier",
        pyo.value(m.fs.CL1.costing.capital_cost),
        pyo.units.get_units(m.fs.CL1.costing.capital_cost),
    )
    print(
        "capital cost AD",
        pyo.value(m.fs.RADM.costing.capital_cost),
        pyo.units.get_units(m.fs.RADM.costing.capital_cost),
    )
    print(
        "capital cost dewatering Unit",
        pyo.value(m.fs.DU.costing.capital_cost),
        pyo.units.get_units(m.fs.DU.costing.capital_cost),
    )
    print(
        "capital cost thickener unit",
        pyo.value(m.fs.TU.costing.capital_cost),
        pyo.units.get_units(m.fs.TU.costing.capital_cost),
    )


def display_performance_metrics(m):
    print(
        "Specific energy consumption with respect to influent flowrate: %.1f kWh/m3"
        % pyo.value(m.fs.costing.specific_energy_consumption)
    )

    print(
        "electricity consumption R3",
        pyo.value(m.fs.R3.electricity_consumption[0]),
        pyo.units.get_units(m.fs.R3.electricity_consumption[0]),
    )
    print(
        "electricity consumption R4",
        pyo.value(m.fs.R4.electricity_consumption[0]),
        pyo.units.get_units(m.fs.R4.electricity_consumption[0]),
    )
    print(
        "electricity consumption R5",
        pyo.value(m.fs.R5.electricity_consumption[0]),
        pyo.units.get_units(m.fs.R5.electricity_consumption[0]),
    )
    print(
        "electricity consumption primary clarifier",
        pyo.value(m.fs.CL.electricity_consumption[0]),
        pyo.units.get_units(m.fs.CL.electricity_consumption[0]),
    )
    print(
        "electricity consumption secondary clarifier",
        pyo.value(m.fs.CL1.electricity_consumption[0]),
        pyo.units.get_units(m.fs.CL1.electricity_consumption[0]),
    )
    print(
        "electricity consumption AD",
        pyo.value(m.fs.RADM.electricity_consumption[0]),
        pyo.units.get_units(m.fs.RADM.electricity_consumption[0]),
    )
    print(
        "electricity consumption dewatering Unit",
        pyo.value(m.fs.DU.electricity_consumption[0]),
        pyo.units.get_units(m.fs.DU.electricity_consumption[0]),
    )
    print(
        "electricity consumption thickening Unit",
        pyo.value(m.fs.TU.electricity_consumption[0]),
        pyo.units.get_units(m.fs.TU.electricity_consumption[0]),
    )
    print(
        "Influent flow",
        pyo.value(m.fs.FeedWater.flow_vol[0]),
        pyo.units.get_units(m.fs.FeedWater.flow_vol[0]),
    )
    print(
        "flow into R3",
        pyo.value(m.fs.R3.control_volume.properties_in[0].flow_vol),
        pyo.units.get_units(m.fs.R3.control_volume.properties_in[0].flow_vol),
    )
    print(
        "flow into RADM",
        pyo.value(m.fs.RADM.liquid_phase.properties_in[0].flow_vol),
        pyo.units.get_units(m.fs.RADM.liquid_phase.properties_in[0].flow_vol),
    )


if __name__ == "__main__":
    m, results = main()
