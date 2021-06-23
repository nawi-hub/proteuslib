###############################################################################
# ProteusLib Copyright (c) 2021, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory, Oak Ridge National
# Laboratory, National Renewable Energy Laboratory, and National Energy
# Technology Laboratory (subject to receipt of any required approvals from
# the U.S. Dept. of Energy). All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. These files are also available online at the URL
# "https://github.com/nawi-hub/proteuslib/"
#
###############################################################################
from pyomo.environ import (ConcreteModel,
                           SolverFactory,
                           TerminationCondition,
                           value,
                           Constraint,
                           Expression,
                           Objective,
                           Param,
                           TransformationFactory,
                           units as pyunits)
from pyomo.network import Arc
import pyomo.util.infeasible as infeas
from idaes.core import FlowsheetBlock
from idaes.core.util import get_solver
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util.initialization import (solve_indexed_blocks,
                                            propagate_state,
                                            fix_state_vars,
                                            revert_state_vars)
from idaes.generic_models.unit_models import Mixer, Separator, Product, Feed
from idaes.generic_models.unit_models.mixer import MomentumMixingType
import idaes.core.util.scaling as iscale
import idaes.logger as idaeslog

import proteuslib.property_models.NaCl_prop_pack as props
from proteuslib.unit_models.reverse_osmosis_0D import (ReverseOsmosis0D,
                                                       ConcentrationPolarizationType,
                                                       MassTransferCoefficient,
                                                       PressureChangeType)
from proteuslib.unit_models.pressure_exchanger import PressureExchanger
from proteuslib.unit_models.pump_isothermal import Pump
import proteuslib.flowsheets.RO_with_energy_recovery.financials as financials


def main():
    # set up solver
    solver = get_solver(options={'nlp_scaling_method': 'user-scaling'})

    # build, set, and initialize
    m = build()
    set_operating_conditions(m, solver=solver)
    initialize_system(m, solver=solver)

    # simulate and display
    solve(m, solver=solver)
    print('\n***---Simulation results---***')
    display_system(m)
    display_design(m)
    display_state(m)

    # optimize and display
    optimize(m, solver=solver)
    print('\n***---Optimization results---***')
    display_system(m)
    display_design(m)
    display_state(m)


def build():
    # flowsheet set up
    m = ConcreteModel()
    m.fs = FlowsheetBlock(default={'dynamic': False})
    m.fs.properties = props.NaClParameterBlock()
    financials.add_costing_param_block(m.fs)

    # unit models
    m.fs.feed = Feed(default={'property_package': m.fs.properties})
    m.fs.S1 = Separator(default={
        "property_package": m.fs.properties,
        "outlet_list": ['P1', 'PXR']})
    m.fs.P1 = Pump(default={'property_package': m.fs.properties})
    m.fs.PXR = PressureExchanger(default={'property_package': m.fs.properties})
    m.fs.P2 = Pump(default={'property_package': m.fs.properties})
    m.fs.M1 = Mixer(default={
        "property_package": m.fs.properties,
        "momentum_mixing_type": MomentumMixingType.equality,  # booster pump will match pressure
        "inlet_list": ['P1', 'P2']})
    m.fs.RO = ReverseOsmosis0D(default={
        "property_package": m.fs.properties,
        "has_pressure_change": True,
        "pressure_change_type": PressureChangeType.calculated,
        "mass_transfer_coefficient": MassTransferCoefficient.calculated,
        "concentration_polarization_type": ConcentrationPolarizationType.calculated,
    })
    m.fs.product = Product(default={'property_package': m.fs.properties})
    m.fs.disposal = Product(default={'property_package': m.fs.properties})

    # additional variables or expressions
    feed_flow_vol_total = m.fs.feed.properties[0].flow_vol
    product_flow_vol_total = m.fs.product.properties[0].flow_vol
    m.fs.recovery = Expression(
        expr=product_flow_vol_total/feed_flow_vol_total)
    m.fs.annual_water_production = Expression(
        expr=pyunits.convert(product_flow_vol_total, to_units=pyunits.m ** 3 / pyunits.year)
             * m.fs.costing_param.load_factor)
    pump_power_total = m.fs.P1.work_mechanical[0] + m.fs.P2.work_mechanical[0]
    m.fs.specific_energy_consumption = Expression(
        expr=pyunits.convert(pump_power_total, to_units=pyunits.kW)
             / pyunits.convert(product_flow_vol_total, to_units=pyunits.m**3 / pyunits.hr))

    # costing
    m.fs.P1.get_costing(module=financials, pump_type="High pressure")
    m.fs.P2.get_costing(module=financials, pump_type="High pressure")
    m.fs.RO.get_costing(module=financials)
    m.fs.PXR.get_costing(module=financials)
    financials.get_system_costing(m.fs)

    # connections
    m.fs.s01 = Arc(source=m.fs.feed.outlet, destination=m.fs.S1.inlet)
    m.fs.s02 = Arc(source=m.fs.S1.P1, destination=m.fs.P1.inlet)
    m.fs.s03 = Arc(source=m.fs.P1.outlet, destination=m.fs.M1.P1)
    m.fs.s04 = Arc(source=m.fs.M1.outlet, destination=m.fs.RO.inlet)
    m.fs.s05 = Arc(source=m.fs.RO.permeate, destination=m.fs.product.inlet)
    m.fs.s06 = Arc(source=m.fs.RO.retentate, destination=m.fs.PXR.high_pressure_inlet)
    m.fs.s07 = Arc(source=m.fs.PXR.high_pressure_outlet, destination=m.fs.disposal.inlet)
    m.fs.s08 = Arc(source=m.fs.S1.PXR, destination=m.fs.PXR.low_pressure_inlet)
    m.fs.s09 = Arc(source=m.fs.PXR.low_pressure_outlet, destination=m.fs.P2.inlet)
    m.fs.s10 = Arc(source=m.fs.P2.outlet, destination=m.fs.M1.P2)
    TransformationFactory("network.expand_arcs").apply_to(m)

    # scaling
    m.fs.properties.set_default_scaling('flow_mass_phase_comp', 1, index=('Liq', 'H2O'))
    m.fs.properties.set_default_scaling('flow_mass_phase_comp', 1e2, index=('Liq', 'NaCl'))
    iscale.calculate_scaling_factors(m)

    return m


def set_operating_conditions(m, solver=None):
    if solver is None:
        solver = get_solver(options={'nlp_scaling_method': 'user-scaling'})

    # ---specifications---
    # feed
    feed_flow_mass = 1  # feed mass flow rate [kg/s]
    feed_mass_frac_NaCl = 0.035  # feed NaCl mass fraction [-]
    feed_mass_frac_H2O = 1 - feed_mass_frac_NaCl # feed H20 mass fraction [-]

    m.fs.feed.flow_mass_phase_comp[0, 'Liq', 'NaCl'].fix(feed_flow_mass * feed_mass_frac_NaCl)
    m.fs.feed.flow_mass_phase_comp[0, 'Liq', 'H2O'].fix(feed_flow_mass * feed_mass_frac_H2O)
    m.fs.feed.pressure.fix(101325)  # atmospheric pressure [Pa]
    m.fs.feed.temperature.fix(273.15 + 25)  # room temperature [K]

    # separator, no degrees of freedom (i.e. equal flow rates in PXR determines split fraction)

    # pump 1, high pressure pump, 2 degrees of freedom (efficiency and outlet pressure)
    m.fs.P1.efficiency_pump.fix(0.80)  # pump efficiency [-]
    operating_pressure = calculate_operating_pressure(
        feed_state_block=m.fs.feed.properties[0],
        over_pressure=0.3,
        water_recovery=0.5,
        NaCl_passage=0.01,
        solver=solver)
    m.fs.P1.control_volume.properties_out[0].pressure.fix(operating_pressure)

    # pressure exchanger
    m.fs.PXR.efficiency_pressure_exchanger.fix(0.95)  # pressure exchanger efficiency [-]

    # pump 2, booster pump, 1 degree of freedom (efficiency, pressure must match high pressure pump)
    m.fs.P2.efficiency_pump.fix(0.80)

    # mixer, no degrees of freedom

    # RO unit
    m.fs.RO.A_comp.fix(4.2e-12)  # membrane water permeability coefficient [m/s-Pa]
    m.fs.RO.B_comp.fix(3.5e-8)  # membrane salt permeability coefficient [m/s]
    m.fs.RO.channel_height.fix(1e-3)  # channel height in membrane stage [m]
    m.fs.RO.spacer_porosity.fix(0.97)  # spacer porosity in membrane stage [-]
    m.fs.RO.permeate.pressure[0].fix(101325)  # atmospheric pressure [Pa]
    m.fs.RO.width.fix(5)  # membrane stage width [m]
    # initiate RO feed values to determine area
    m.fs.RO.feed_side.properties_in[0].flow_mass_phase_comp['Liq', 'H2O'] = \
        value(m.fs.feed.properties[0].flow_mass_phase_comp['Liq', 'H2O'])
    m.fs.RO.feed_side.properties_in[0].flow_mass_phase_comp['Liq', 'NaCl'] = \
        value(m.fs.feed.properties[0].flow_mass_phase_comp['Liq', 'NaCl'])
    m.fs.RO.feed_side.properties_in[0].temperature = \
        value(m.fs.feed.properties[0].temperature)
    m.fs.RO.feed_side.properties_in[0].pressure = \
        value(m.fs.P1.control_volume.properties_out[0].pressure)
    RO_area = calculate_RO_area(unit=m.fs.RO, water_recovery=0.5, solver=solver)
    m.fs.RO.area.fix(RO_area)

    # check degrees of freedom
    if degrees_of_freedom(m) != 0:
        raise RuntimeError("The set_operating_conditions function resulted in {} "
                           "degrees of freedom rather than 0. This error suggests "
                           "that too many or not enough variables are fixed for a "
                           "simulation.".format(degrees_of_freedom(m)))


def calculate_operating_pressure(feed_state_block=None, over_pressure=0.15,
                                 water_recovery=0.5, NaCl_passage=0.01, solver=None):
    """
    estimate operating pressure for RO unit model given the following arguments:
        feed_state_block:   the state block of the RO feed that has the non-pressure state
                            variables initialized to their values (default=None)
        over_pressure:  the amount of operating pressure above the brine osmotic pressure
                        represented as a fraction (default=0.15)
        water_recovery: the mass-based fraction of inlet H2O that becomes permeate
                        (default=0.5)
        NaCl_passage:   the mass-based fraction of inlet NaCl that becomes permeate
                        (default=0.01)
        solver:     solver object to be used (default=None)
    """
    t = ConcreteModel()  # create temporary model
    prop = feed_state_block.config.parameters
    t.brine = prop.build_state_block([0], default={})

    # specify state block
    t.brine[0].flow_mass_phase_comp['Liq', 'H2O'].fix(
        value(feed_state_block.flow_mass_phase_comp['Liq', 'H2O']) * (1 - water_recovery))
    t.brine[0].flow_mass_phase_comp['Liq', 'NaCl'].fix(
        value(feed_state_block.flow_mass_phase_comp['Liq', 'NaCl']) * (1 - NaCl_passage))
    t.brine[0].pressure.fix(101325)  # valid when osmotic pressure is independent of hydraulic pressure
    t.brine[0].temperature.fix(value(feed_state_block.temperature))

    # calculate osmotic pressure
    # since properties are created on demand, we must touch the property to create it
    t.brine[0].pressure_osm
    # solve state block
    results = solve_indexed_blocks(solver, [t.brine])
    check_solve(results)

    return value(t.brine[0].pressure_osm) * (1 + over_pressure)


def calculate_RO_area(unit=None, water_recovery=0.5, solver=None):
    """
    determine RO membrane area required to achieve the specified water recovery:
        unit:  the RO unit model, e.g. m.fs.RO, it should have its inlet feed state block
               initiated to the correct values (default=None)
        water_recovery: the mass-based fraction of inlet H2O that becomes permeate
                        (default=0.5)
        solver: solver object to be used (default=None)
    """
    # intialize unit
    unit.initialize(optarg=solver.options if solver else None)
    # fix inlet conditions
    flags = fix_state_vars(unit.feed_side.properties_in)
    # fix unit water recovery
    unit.feed_side.properties_out[0].flow_mass_phase_comp['Liq', 'H2O'].fix(
        unit.feed_side.properties_in[0].flow_mass_phase_comp['Liq', 'H2O'].value * (1 - water_recovery))
    # solve for unit area
    check_dof(unit)
    solve(unit, solver=solver)
    # unfix variables
    revert_state_vars(unit.feed_side.properties_in, flags)
    unit.feed_side.properties_out[0].flow_mass_phase_comp['Liq', 'H2O'].unfix()
    return unit.area.value


def solve(blk, solver=None, tee=False):
    if solver is None:
        solver = get_solver(options={'nlp_scaling_method': 'user-scaling'})
    results = solver.solve(blk, tee=tee)
    check_solve(results)


def check_dof(blk, dof_expected=0):
    if degrees_of_freedom(blk) != dof_expected:
        raise RuntimeError("The degrees of freedom on {blk} were {dof} but {dof_e} "
                           "were expected, check the fixed variables on that block".format(
            blk=blk, dof=degrees_of_freedom(blk), dof_e=dof_expected))


def check_solve(results):
    if results.solver.termination_condition != TerminationCondition.optimal:
        raise RuntimeError("The solver failed to converge to an optimal solution. "
                           "This suggests that the user provided infeasible inputs "
                           "or that the model is poorly scaled.")


def initialize_system(m, solver=None):

    if solver is None:
        solver = get_solver(options={'nlp_scaling_method': 'user-scaling'})
    optarg = solver.options 

    # ---initialize feed block---
    m.fs.feed.initialize(optarg=optarg)

    # ---initialize splitter and pressure exchanger---
    # pressure exchanger high pressure inlet
    propagate_state(m.fs.s06)  # propagate to PXR high pressure inlet from RO retentate
    m.fs.PXR.high_pressure_side.properties_in.initialize(optarg=optarg)

    # splitter inlet
    propagate_state(m.fs.s01)  # propagate to splitter inlet from feed
    m.fs.S1.mixed_state[0].mass_frac_phase_comp  # touch property, so that it is built and can be solved for
    m.fs.S1.mixed_state.initialize(optarg=optarg)

    # splitter outlet to PXR, enforce same flow_vol as PXR high pressure inlet
    m.fs.S1.PXR_state[0].pressure.fix(value(m.fs.S1.mixed_state[0].pressure))
    m.fs.S1.PXR_state[0].temperature.fix(value(m.fs.S1.mixed_state[0].temperature))
    m.fs.S1.PXR_state[0].flow_vol_phase['Liq'].fix(
        value(m.fs.PXR.high_pressure_side.properties_in[0].flow_vol_phase['Liq']))
    m.fs.S1.PXR_state[0].mass_frac_phase_comp['Liq', 'NaCl'].fix(
        value(m.fs.S1.mixed_state[0].mass_frac_phase_comp['Liq', 'NaCl']))

    check_dof(m.fs.S1.PXR_state[0])
    results = solve_indexed_blocks(solver, [m.fs.S1.PXR_state])
    check_solve(results)

    # unfix PXR_state state variables and properties
    m.fs.S1.PXR_state[0].pressure.unfix()
    m.fs.S1.PXR_state[0].temperature.unfix()
    m.fs.S1.PXR_state[0].flow_vol_phase['Liq'].unfix()
    m.fs.S1.PXR_state[0].mass_frac_phase_comp['Liq', 'NaCl'].unfix()
    m.fs.S1.PXR_state[0].flow_mass_phase_comp['Liq', 'NaCl'].fix()

    # splitter initialization
    m.fs.S1.initialize(optarg=optarg)
    m.fs.S1.PXR_state[0].flow_mass_phase_comp['Liq', 'NaCl'].unfix()

    # pressure exchanger low pressure inlet
    propagate_state(m.fs.s08)

    # pressure exchanger initialization
    m.fs.PXR.initialize(optarg=optarg)

    # ---initialize pump 1---
    propagate_state(m.fs.s02)
    m.fs.P1.initialize(optarg=optarg)

    # ---initialize pump 2---
    propagate_state(m.fs.s09)
    m.fs.P2.control_volume.properties_out[0].pressure.fix(
        value(m.fs.P2.control_volume.properties_out[0].pressure))
    m.fs.P2.initialize(optarg=optarg)
    m.fs.P2.control_volume.properties_out[0].pressure.unfix()

    # ---initialize mixer---
    propagate_state(m.fs.s03)
    propagate_state(m.fs.s10)
    m.fs.M1.initialize(optarg=optarg, outlvl=idaeslog.INFO)


def optimize(m, solver=None):
    # objective
    m.fs.objective = Objective(expr=m.fs.costing.LCOW)

    # unfix decision variables and add bounds
    # pump 1 and pump 2
    m.fs.P1.control_volume.properties_out[0].pressure.unfix()
    m.fs.P1.control_volume.properties_out[0].pressure.setlb(10e5)
    m.fs.P1.control_volume.properties_out[0].pressure.setub(80e5)
    m.fs.P1.deltaP.setlb(0)
    m.fs.P2.control_volume.properties_out[0].pressure.setlb(10e5)
    m.fs.P2.control_volume.properties_out[0].pressure.setub(80e5)
    m.fs.P2.deltaP.setlb(0)

    # RO
    m.fs.RO.area.unfix()  # area in membrane stage [m2]
    m.fs.RO.area.setlb(1)
    m.fs.RO.area.setub(150)

    # additional specifications
    m.fs.product_recovery = Param(initialize=0.5, mutable=True)        # product mass flow rate fraction of feed [-]
    m.fs.product_salinity = Param(initialize=500e-6, mutable=True)     # product NaCl mass fraction [-]
    m.fs.minimum_water_flux = Param(initialize=1./3600., mutable=True) # minimum water flux [kg/m2-s]

    # additional constraints
    m.fs.eq_recovery = Constraint(expr=m.fs.product_recovery == m.fs.recovery)
    m.fs.eq_product_quality = Constraint(
        expr=m.fs.product.properties[0].mass_frac_phase_comp['Liq', 'NaCl'] <= m.fs.product_salinity)
    iscale.constraint_scaling_transform(m.fs.eq_product_quality, 1e3)  # scaling constraint
    m.fs.eq_minimum_water_flux = Constraint(
        expr=m.fs.RO.flux_mass_io_phase_comp[0, 'out', 'Liq', 'H2O'] >= m.fs.minimum_water_flux)

    # ---checking model---
    check_dof(m, dof_expected=1)

    # --solve---
    solve(m, solver=solver)


def display_system(m):
    print('---system metrics---')
    feed_flow_mass = sum(m.fs.feed.flow_mass_phase_comp[0, 'Liq', j].value for j in ['H2O', 'NaCl'])
    feed_mass_frac_NaCl = m.fs.feed.flow_mass_phase_comp[0, 'Liq', 'NaCl'].value / feed_flow_mass
    print('Feed: %.2f kg/s, %.0f ppm' % (feed_flow_mass, feed_mass_frac_NaCl * 1e6))

    prod_flow_mass = sum(m.fs.product.flow_mass_phase_comp[0, 'Liq', j].value for j in ['H2O', 'NaCl'])
    prod_mass_frac_NaCl = m.fs.product.flow_mass_phase_comp[0, 'Liq', 'NaCl'].value / prod_flow_mass
    print('Product: %.3f kg/s, %.0f ppm' % (prod_flow_mass, prod_mass_frac_NaCl * 1e6))

    print('Recovery: %.1f%%' % (value(m.fs.recovery) * 100))
    print('Energy Consumption: %.1f kWh/m3' % value(m.fs.specific_energy_consumption))
    print('Levelized cost of water: %.2f $/m3' % value(m.fs.costing.LCOW))


def display_design(m):
    print('---decision variables---')
    print('Operating pressure %.1f bar' % (m.fs.RO.inlet.pressure[0].value/1e5))
    print('Membrane area %.1f m2' % (m.fs.RO.area.value))

    print('---design variables---')
    print('Separator')
    print('Split fraction %.2f' % (m.fs.S1.split_fraction[0, 'PXR'].value*100))
    print('Pump 1\noutlet pressure: %.1f bar\npower %.2f kW'
          % (m.fs.P1.outlet.pressure[0].value / 1e5, m.fs.P1.work_mechanical[0].value / 1e3))
    print('Pump 2\noutlet pressure: %.1f bar\npower %.2f kW'
          % (m.fs.P2.outlet.pressure[0].value / 1e5, m.fs.P2.work_mechanical[0].value / 1e3))


def display_state(m):
    print('---state---')

    def print_state(s, b):
        flow_mass = sum(b.flow_mass_phase_comp[0, 'Liq', j].value for j in ['H2O', 'NaCl'])
        mass_frac_ppm = b.flow_mass_phase_comp[0, 'Liq', 'NaCl'].value / flow_mass * 1e6
        pressure_bar = b.pressure[0].value / 1e5
        print(s + ': %.3f kg/s, %.0f ppm, %.1f bar' % (flow_mass, mass_frac_ppm, pressure_bar))

    print_state('Feed      ', m.fs.feed.outlet)
    print_state('Split 1   ', m.fs.S1.P1)
    print_state('P1 out    ', m.fs.P1.outlet)
    print_state('Split 2   ', m.fs.S1.PXR)
    print_state('PXR LP out', m.fs.PXR.low_pressure_outlet)
    print_state('P2 out    ', m.fs.P2.outlet)
    print_state('Mix out   ', m.fs.M1.outlet)
    print_state('RO perm   ', m.fs.RO.permeate)
    print_state('RO reten  ', m.fs.RO.retentate)
    print_state('PXR HP out', m.fs.PXR.high_pressure_outlet)


if __name__ == "__main__":
    main()
