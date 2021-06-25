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


from enum import Enum
# Import Pyomo libraries
from pyomo.environ import (Var,
                           Set,
                           Param,
                           SolverFactory,
                           Suffix,
                           NonNegativeReals,
                           NegativeReals,
                           Reference,
                           Block,
                           units as pyunits,
                           exp,
                           value)
from pyomo.common.config import ConfigBlock, ConfigValue, In
# Import IDAES cores
from idaes.core import (ControlVolume1DBlock,
                        declare_process_block_class,
                        MaterialBalanceType,
                        EnergyBalanceType,
                        MomentumBalanceType,
                        UnitModelBlockData,
                        useDefault,
                        FlowDirection)
from idaes.core.control_volume1d import DistributedVars
from idaes.core.util.config import is_physical_parameter_block
from idaes.core.util.misc import add_object_reference
from idaes.core.util.tables import create_stream_table_dataframe
from idaes.core.util.constants import Constants as CONST
from idaes.core.util.exceptions import ConfigurationError
from idaes.core.util import get_solver, scaling as iscale

import idaes.logger as idaeslog


__author__ = "Adam Atia"

# Set up logger
_log = idaeslog.getLogger(__name__)


@declare_process_block_class("ReverseOsmosis1D")
class ReverseOsmosis1DData(UnitModelBlockData):
    """Standard 1D Reverse Osmosis Unit Model Class."""

    CONFIG = UnitModelBlockData.CONFIG()

    # Template for config arguments for feed and permeate side
    _SideTemplate = ConfigBlock()

    _SideTemplate.declare("dynamic", ConfigValue(
        default=False,
        domain=In([False]),
        description="Dynamic model flag - must be False",
        doc="""Indicates whether this model will be dynamic or not.
    **default** = False. RO units do not yet support dynamic
    behavior."""))

    _SideTemplate.declare("has_holdup", ConfigValue(
            default=False,
            domain=In([False]),
            description="Holdup construction flag",
            doc="""Indicates whether holdup terms should be constructed or not.
    **default** - False. RO units do not have defined volume, thus
    this must be False."""))

    _SideTemplate.declare("material_balance_type", ConfigValue(
            default=MaterialBalanceType.useDefault,
            domain=In(MaterialBalanceType),
            description="Material balance construction flag",
            doc="""Indicates what type of mass balance should be constructed,
    **default** - MaterialBalanceType.useDefault.
    **Valid values:** {
    **MaterialBalanceType.useDefault - refer to property package for default
    balance type
    **MaterialBalanceType.none** - exclude material balances,
    **MaterialBalanceType.componentPhase** - use phase component balances,
    **MaterialBalanceType.componentTotal** - use total component balances,
    **MaterialBalanceType.elementTotal** - use total element balances,
    **MaterialBalanceType.total** - use total material balance.}"""))

    _SideTemplate.declare("energy_balance_type", ConfigValue(
            default=EnergyBalanceType.useDefault,
            domain=In(EnergyBalanceType),
            description="Energy balance construction flag",
            doc="""Indicates what type of energy balance should be constructed,
    **default** - EnergyBalanceType.useDefault.
    **Valid values:** {
    **EnergyBalanceType.useDefault - refer to property package for default
    balance type
    **EnergyBalanceType.none** - exclude energy balances,
    **EnergyBalanceType.enthalpyTotal** - single enthalpy balance for material,
    **EnergyBalanceType.enthalpyPhase** - enthalpy balances for each phase,
    **EnergyBalanceType.energyTotal** - single energy balance for material,
    **EnergyBalanceType.energyPhase** - energy balances for each phase.}"""))

    _SideTemplate.declare("momentum_balance_type", ConfigValue(
            default=MomentumBalanceType.pressureTotal,
            domain=In(MomentumBalanceType),
            description="Momentum balance construction flag",
            doc="""Indicates what type of momentum balance should be constructed,
    **default** - MomentumBalanceType.pressureTotal.
    **Valid values:** {
    **MomentumBalanceType.none** - exclude momentum balances,
    **MomentumBalanceType.pressureTotal** - single pressure balance for material,
    **MomentumBalanceType.pressurePhase** - pressure balances for each phase,
    **MomentumBalanceType.momentumTotal** - single momentum balance for material,
    **MomentumBalanceType.momentumPhase** - momentum balances for each phase.}"""))

    _SideTemplate.declare("has_pressure_change", ConfigValue(
            default=False,
            domain=In([True, False]),
            description="Pressure change term construction flag",
            doc="""Indicates whether terms for pressure change should be
    constructed,
    **default** - False.
    **Valid values:** {
    **True** - include pressure change terms,
    **False** - exclude pressure change terms.}"""))

    _SideTemplate.declare("area_definition", ConfigValue(
            default=DistributedVars.uniform,
            domain=In(DistributedVars),
            description="Argument for defining form of area variable",
            doc="""Argument defining whether area variable should be spatially
    variant or not. **default** - DistributedVars.uniform.
    **Valid values:** {
    DistributedVars.uniform - area does not vary across spatial domain,
    DistributedVars.variant - area can vary over the domain and is indexed
    by time and space.}"""))

    CONFIG.declare("property_package", ConfigValue(
            default=None,
            domain=is_physical_parameter_block,
            description="Property package to use for control volume",
            doc="""Property parameter object used to define property calculations
    **default** - useDefault.
    **Valid values:** {
    **useDefault** - use default package from parent model or flowsheet,
    **PhysicalParameterObject** - a PhysicalParameterBlock object.}"""))

    CONFIG.declare("property_package_args", ConfigValue(
            default={},
            description="Arguments for constructing property packages",
            doc="""A ConfigBlock with arguments to be passed to a property block(s)
    and used when constructing these.
    **default** - None.
    **Valid values:** {
    see property package for documentation.}"""))

    CONFIG.declare(
        "transformation_method",
        ConfigValue(
            default=useDefault,
            description="Discretization method to use for DAE transformation",
            doc="""Discretization method to use for DAE transformation. See Pyomo
    documentation for supported transformations."""))

    CONFIG.declare("transformation_scheme", ConfigValue(
            default=useDefault,
            description="Discretization scheme to use for DAE transformation",
            doc="""Discretization scheme to use when transforming domain. See
    Pyomo documentation for supported schemes."""))

    # Create individual config blocks for feed and permeate side
    CONFIG.declare("feed_side", _SideTemplate(doc="feed side config arguments"))
    CONFIG.declare("permeate_side", _SideTemplate(doc="permeate side config arguments"))

    # Common config args for both sides
    CONFIG.declare("finite_elements", ConfigValue(
            default=20,
            domain=int,
            description="Number of finite elements in length domain",
            doc="""Number of finite elements to use when discretizing length 
            domain (default=20)"""))

    CONFIG.declare("collocation_points", ConfigValue(
            default=5,
            domain=int,
            description="Number of collocation points per finite element",
            doc="""Number of collocation points to use per finite element when
            discretizing length domain (default=5)"""))

    def _process_config(self):
        #TODO: add config errors here
        for c in self.config.property_package.component_list:
            comp = self.config.property_package.get_component(c)
            try:
                if comp.is_solvent():
                    self.solvent_list.add(c)
                if comp.is_solute():
                    self.solute_list.add(c)
            except TypeError:
                raise ConfigurationError("RO model only supports one solvent and one or more solutes,"
                                         "the provided property package has specified a component '{}' "
                                         "that is not a solvent or solute".format(c))
        if len(self.solvent_list) > 1:
            raise ConfigurationError("RO model only supports one solvent component,"
                                     "the provided property package has specified {} solvent components"
                                     .format(len(self.solvent_list)))

        if self.config.feed_side.transformation_method is useDefault:
            _log.warning(
                "Discretization method was "
                "not specified for the feed side of the "
                "reverse osmosis module. "
                "Defaulting to finite "
                "difference method on the feed side."
            )
            self.config.feed_side.transformation_method = "dae.finite_difference"

        if self.config.permeate_side.transformation_method is useDefault:
            _log.warning(
                "Discretization method was "
                "not specified for the permeate side of the "
                "reverse osmosis module. "
                "Defaulting to finite "
                "difference method on the permeate side."
            )
            self.config.permeate_side.transformation_method = "dae.finite_difference"

        if self.config.feed_side.transformation_scheme is useDefault:
            _log.warning(
                "Discretization scheme was "
                "not specified for the feed side of the "
                "reverse osmosis module."
                "Defaulting to backward finite "
                "difference on the feed side."
            )
            self.config.feed_side.transformation_scheme = "BACKWARD"

        if self.config.permeate_side.transformation_scheme is useDefault:
            _log.warning(
                "Discretization scheme was "
                "not specified for the permeate side of the "
                "reverse osmosis module. "
                "Defaulting to backward finite "
                "difference on the permeate side."
            )
            self.config.permeate_side.transformation_scheme = "BACKWARD"

    def build(self):
        """
        Build 1D RO model (pre-DAE transformation).

        Args:
            None

        Returns:
            None
        """
        # Call UnitModel.build to setup dynamics
        super().build()

        self.scaling_factor = Suffix(direction=Suffix.EXPORT)

        self.solvent_list = Set()
        self.solute_list = Set()
        self._process_config()

        # ==========================================================================
        """ Build 1D Control volume for feed side"""
        self.feed_side = ControlVolume1DBlock(default={
            "dynamic": self.config.feed_side.dynamic,
            "has_holdup": self.config.feed_side.has_holdup,
            "area_definition": self.config.feed_side.area_definition,
            "property_package": self.config.property_package,
            "property_package_args": self.config.property_package_args,
            "transformation_method": self.config.transformation_method,
            "transformation_scheme": self.config.transformation_scheme,
            "finite_elements": self.config.finite_elements,
            "collocation_points": self.config.collocation_points
        })

        # ==========================================================================
        """ Build 1D Control volume for permeate side"""
        self.permeate_side = ControlVolume1DBlock(default={
            "dynamic": self.config.permeate_side.dynamic,
            "has_holdup": self.config.permeate_side.has_holdup,
            "area_definition": self.config.permeate_side.area_definition,
            "property_package": self.config.property_package,
            "property_package_args": self.config.property_package_args,
            "transformation_method": self.config.transformation_method,
            "transformation_scheme": self.config.transformation_scheme,
            "finite_elements": self.config.finite_elements,
            "collocation_points": self.config.collocation_points
        })

        feed_side = self.feed_side
        permeate_side = self.permeate_side

        # ==========================================================================
        """ Add geometry for each control volume"""
        feed_side.add_geometry()
        permeate_side.add_geometry()

        # ==========================================================================
        """ Add state blocks for each control volume"""
        feed_side.add_state_blocks(has_phase_equilibrium=False)
        permeate_side.add_state_blocks(has_phase_equilibrium=False)

        # ==========================================================================
        """ Populate feed side"""
        feed_side.add_material_balances(balance_type=self.config.feed_side.material_balance_type,
                                        has_mass_transfer=True)
        feed_side.add_energy_balances(balance_type=self.config.feed_side.energy_balance_type,
                                           has_enthalpy_transfer=True)
        feed_side.add_momentum_balances(balance_type=self.config.feed_side.momentum_balance_type,
                                        has_pressure_change=self.config.has_pressure_change)

        # ==========================================================================
        """ Only enable mass transfer for permeate side"""
        permeate_side.add_material_balances(balance_type=self.config.permeate_side.material_balance_type.none,
                                            has_mass_transfer=True)
        feed_side.add_energy_balances(balance_type=self.config.permeate_side.energy_balance_type.none,
                                           has_enthalpy_transfer=True)
        feed_side.add_momentum_balances(balance_type=self.config.permeate_side.momentum_balance_type.none,
                                        has_pressure_change=self.config.permeate_side.has_pressure_change)

        # ==========================================================================
        """ Apply transformation to feed and permeate sides"""
        feed_side.apply_transformation()
        permeate_side.apply_transformation()

        # ==========================================================================
        """ Add inlet/outlet ports for feed side and only an outlet port for permeate side"""
        self.add_inlet_port(name="feed_inlet", block=feed_side)
        self.add_outlet_port(name="feed_outlet", block=feed_side)
        permeate_side.permeate_out = self.config.property_package.state_block_class(
            self.flowsheet().config.time,
            doc="Material properties of mixed permeate exiting the module")
        self.add_port(name="permeate_outlet", block=permeate_side.permeate_out)

        # ==========================================================================
        """ Add references to control volume geometry."""
        add_object_reference(self, 'length', feed_side.length)
        add_object_reference(self, 'feed_area_cross', feed_side.area)
        add_object_reference(self, 'length', permeate_side.length)
        add_object_reference(self, 'permeate_area_cross', permeate_side.area)

        # Add reference to pressure drop for feed side only
        if (self.config.feed_side.has_pressure_change is True and
                self.config.feed_side.momentum_balance_type != 'none'):
            add_object_reference(self, 'feed_deltaP', feed_side.deltaP)

        self._make_performance()

    def _make_performance(self):
        """
        Variables and constraints for unit model.

        Args:
            None

        Returns:
            None
        """

        # Units should be the same between feed and perm side
        units_meta = \
            self.config.property_package.get_metadata().get_derived_units


        # ==========================================================================
        """ Unit model variables"""
        self.A_comp = Var(
            self.flowsheet().config.time,
            self.solvent_list,
            initialize=1e-12,
            bounds=(1e-18, 1e-6),
            domain=NonNegativeReals,
            units=units_meta('length') * units_meta('pressure') ** -1 * units_meta('time') ** -1,
            doc="""Solvent permeability coeff.""")
        self.B_comp = Var(
            self.flowsheet().config.time,
            self.solute_list,
            initialize=1e-8,
            bounds=(1e-11, 1e-5),
            domain=NonNegativeReals,
            units=units_meta('length')*units_meta('time')**-1,
            doc='Solute permeability coeff.')
        # TODO: add water density to NaCl prop model and remove here (or use IDAES version)
        self.dens_solvent = Param(
            initialize=1000,
            units=units_meta('mass')*units_meta('length')**-3,
            doc='Pure water density')

        def flux_mass_phase_comp_initialize(b, t, x, p, j):
            if j in self.solvent_list:
                return 5e-4
            elif j in self.solute_list:
                return 1e-6

        def flux_mass_phase_comp_bounds(b, t, x, p, j):
            if j in self.solvent_list:
                ub = 3e-2
                lb = 1e-4
            elif j in self.solute_list:
                ub = 1e-3
                lb = 1e-8
            return lb, ub

        self.flux_mass_phase_comp = Var(
            self.flowsheet().config.time,
            self.config.feed_side.length_domain,
            self.config.property_package.phase_list,
            self.config.property_package.component_list,
            initialize=flux_mass_phase_comp_initialize,
            bounds=flux_mass_phase_comp_bounds,
            units=units_meta('mass')*units_meta('length')**-2*units_meta('time')**-1,
            doc='Mass flux across membrane')

        self.area = Var(
            initialize=10,
            bounds=(1e-1, 1e3),
            domain=NonNegativeReals,
            units=units_meta('length')**2,
            doc='Membrane area')

        # TODO: Add initial value, bounds and real equation later. Just need for now since add_geometry() generates
        #  cross-sectional area var that I am referencing
        self.feed_area_cross = Var(
            initialize=1,
            bounds=(1e-3, 1),
            domain=NonNegativeReals,
            units=units_meta('length')**2,
            doc='Cross-sectional area of feed channel')

        self.permeate_area_cross = Var(
            initialize=1,
            bounds=(1e-3, 1),
            domain=NonNegativeReals,
            units=units_meta('length')**2,
            doc='Cross-sectional area of feed channel')

        self.width = Var(
            initialize=1,
            bounds=(1e-1, 1e3),
            domain=NonNegativeReals,
            units=units_meta('length')**2,
            doc='Membrane area')

        @self.Constraint(doc="Membrane area")
        def eq_area(b):
            return b.area == b.length * b.width

        @self.Constraint(doc="Membrane area")
        def eq_feed_area_cross(b):
            return b.feed_area_cross == 1 * 1 * b.width  # TODO: add channel_height and spacer_porosity

        @self.Constraint(doc="Membrane area")
        def eq_permeate_area_cross(b):
            return b.permeate_area_cross == 1 * 1 * b.width  # TODO: add channel_height and spacer_porosity

        # mass transfer
        # def mass_transfer_phase_comp_initialize(b, t, x, p, j):
        #     return value(self.feed_side.properties[t].get_material_flow_terms('Liq', j)
        #                  * self.recovery_mass_phase_comp[t, 'Liq', j])

        self.mass_transfer_phase_comp = Var(
            self.flowsheet().config.time,
            self.config.feed_side.length_domain,
            self.config.property_package.phase_list,
            self.config.property_package.component_list,
            initialize=0.1, #mass_transfer_phase_comp_initialize,
            bounds=(1e-8, 1e6),
            domain=NonNegativeReals,
            units=units_meta('mass') * units_meta('time')**-1 * units_meta('length')**-1,
            doc='Mass transfer to permeate')

        @self.Constraint(self.flowsheet().config.time,
                         self.config.feed_side.length_domain,
                         self.config.property_package.phase_list,
                         self.config.property_package.component_list,
                         doc="Mass transfer term")
        def eq_mass_transfer_term(self, t, x, p, j):
            return self.mass_transfer_phase_comp[t, x, p, j] == -self.feed_side.mass_transfer_term[t, x, p, j]

        # RO performance equations
        @self.Constraint(self.flowsheet().config.time,
                         self.io_list,
                         self.config.property_package.phase_list,
                         self.config.property_package.component_list,
                         doc="Water and salt mass flux")
        def eq_flux_mass(b, t, x, p, j):
            # if x == 'in':
            #     prop_feed = b.feed_side.properties_in[t]
            #     prop_feed_inter = b.feed_side.properties_interface_in[t]
            # elif x == 'out':
            #     prop_feed = b.feed_side.properties_out[t]
            #     prop_feed_inter = b.feed_side.properties_interface_out[t]
            prop_feed = b.feed_side.properties[t, x]
            prop_perm = b.permeate_side.properties[t, x]
            comp = self.config.property_package.get_component(j)
            if comp.is_solvent():
                return (b.flux_mass_phase_comp[t, x, p, j] == b.A_comp[t, j] * b.dens_solvent
                        * ((prop_feed.pressure - prop_perm.pressure)
                           - (prop_feed.pressure_osm - prop_perm.pressure_osm)))
            elif comp.is_solute():
                return (b.flux_mass_phase_comp[t, x, p, j] == b.B_comp[t, j]
                        * (prop_feed.conc_mass_phase_comp[p, j] - prop_perm.conc_mass_phase_comp[p, j]))

        @self.Expression(self.flowsheet().config.time,
                         self.config.property_package.phase_list,
                         self.config.property_package.component_list,
                         doc="Total component mass flux expression")
        def flux_mass_phase_comp_sum(b, t, p, j):
            return sum(b.flux_mass_phase_comp[t, x, p, j] #* x
                       for x in self.config.feed_side.length_domain)


        @self.Constraint(self.flowsheet().config.time,
                         self.config.feed_side.length_domain,
                         self.config.property_package.phase_list,
                         self.config.property_package.component_list,
                         doc="Permeate mass flow rates exiting unit")
        def eq_permeate_production(b, t, x, p, j):
            return (b.permeate_side.permeate_out[t].get_material_flow_terms(p, j)
                    == b.area * b.flux_mass_phase_comp_sum[t, p, j])

        # Feed and permeate-side connection
        @self.Constraint(self.flowsheet().config.time,
                         self.config.feed_side.length_domain,
                         self.config.property_package.phase_list,
                         self.config.property_package.component_list,
                         doc="Mass transfer from feed to permeate")
        def eq_connect_mass_transfer(b, t, x, p, j):
            return (b.permeate_side.properties[t, x].get_material_flow_terms(p, j)
                    == -b.feed_side.mass_transfer_term[t, x, p, j]) #* x)

        @self.Constraint(self.flowsheet().config.time,
                         self.config.feed_side.length_domain,
                         doc="Enthalpy transfer from feed to permeate")
        def eq_connect_enthalpy_transfer(b, t, x):
            return (b.permeate_side.properties[t, x].get_enthalpy_flow_terms('Liq')
                    == -b.feed_side.enthalpy_transfer[t, x])

        @self.Constraint(self.flowsheet().config.time,
                         self.config.length_domain,
                         doc="Isothermal assumption for permeate")
        def eq_permeate_isothermal(b, t, x):
            return b.feed_side.properties[t, x].temperature == \
                   b.permeate_side.properties_permeate[t, x].temperature

    def initialize(blk,
                   feed_side_args=None,
                   permeate_side_args=None,
                   permeate_block_args=None,
                   outlvl=idaeslog.NOTSET,
                   solver=None,
                   optarg=None):
        """
        Initialization routine for 1D-RO unit.

        Keyword Arguments:
            feed_side_args : a dict of arguments to be passed to the property
             package(s) of the feed_side to provide an initial state for
             initialization (see documentation of the specific
             property package)
            permeate_side_args : a dict of arguments to be passed to the property
             package(s) of the permeate_side to provide an initial state for
             initialization (see documentation of the specific
             property package)
            permeate_block_args : a dict of arguments to be passed to the property
             package(s) of the final permeate StateBlock to provide an initial state for
             initialization (see documentation of the specific
             property package)
            outlvl : sets output level of initialization routine
            solver : str indicating which solver to use during
                     initialization (default = None, use default solver)
            optarg : solver options dictionary object (default=None, use default solver options)

        Returns:
            None
        """
        init_log = idaeslog.getInitLogger(blk.name, outlvl, tag="unit")
        solve_log = idaeslog.getSolveLogger(blk.name, outlvl, tag="unit")

        # Create solver
        opt = get_solver(solver, optarg)

        # ---------------------------------------------------------------------
        # Step 1: Initialize feed_side, permeate_side, and permeate_out blocks
        flags_feed_side = blk.feed_side.initialize(
            outlvl=outlvl,
            optarg=optarg,
            solver=solver,
            state_args=feed_side_args)

        flags_permeate_side = blk.permeate_side.initialize(
            outlvl=outlvl,
            optarg=optarg,
            solver=solver,
            state_args=permeate_side_args)

        flags_permeate_out = blk.permeate_side.permeate_out.initialize(
            outlvl=outlvl,
            optarg=optarg,
            solver=solver,
            state_args=permeate_block_args)

        init_log.info_high("Initialization Step 1 Complete.")

        # ---------------------------------------------------------------------
        # Step 2: Solve unit

    def calculate_scaling_factors(self):
        super().calculate_scaling_factors()
