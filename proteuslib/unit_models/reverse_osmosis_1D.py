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


# Import Pyomo libraries
from pyomo.environ import (Var,
                           Set,
                           Param,
                           Suffix,
                           NonNegativeReals,
                           NegativeReals,
                           units as pyunits,
                           exp,
                           value,
                           Constraint)
from pyomo.common.config import ConfigBlock, ConfigValue, In
# Import IDAES cores
from idaes.core import (ControlVolume1DBlock,
                        declare_process_block_class,
                        MaterialBalanceType,
                        EnergyBalanceType,
                        MomentumBalanceType,
                        UnitModelBlockData,
                        useDefault)
from idaes.core.control_volume1d import DistributedVars
from idaes.core.util.config import is_physical_parameter_block
from idaes.core.util.misc import add_object_reference
from idaes.core.util.tables import create_stream_table_dataframe
from idaes.core.util.exceptions import ConfigurationError
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util import get_solver, scaling as iscale
from idaes.core.util.initialization import solve_indexed_blocks

import idaes.logger as idaeslog


__author__ = "Adam Atia"

# Set up logger
_log = idaeslog.getLogger(__name__)


@declare_process_block_class("ReverseOsmosis1D")
class ReverseOsmosis1DData(UnitModelBlockData):
    """Standard 1D Reverse Osmosis Unit Model Class."""

    CONFIG = ConfigBlock()

    CONFIG.declare("dynamic", ConfigValue(
        default=False,
        domain=In([False]),
        description="Dynamic model flag - must be False",
        doc="""Indicates whether this model will be dynamic or not.
    **default** = False. RO units do not yet support dynamic
    behavior."""))

    CONFIG.declare("has_holdup", ConfigValue(
            default=False,
            domain=In([False]),
            description="Holdup construction flag",
            doc="""Indicates whether holdup terms should be constructed or not.
    **default** - False. RO units do not have defined volume, thus
    this must be False."""))

    CONFIG.declare("has_pressure_change", ConfigValue(
            default=False,
            domain=In([True, False]),
            description="Pressure change term construction flag",
            doc="""Indicates whether terms for pressure change should be
    constructed,
    **default** - False.
    **Valid values:** {
    **True** - include pressure change terms,
    **False** - exclude pressure change terms.}"""))

    CONFIG.declare("area_definition", ConfigValue(
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

    CONFIG.declare("material_balance_type", ConfigValue(
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

    CONFIG.declare("energy_balance_type", ConfigValue(
        default=EnergyBalanceType.useDefault,
        domain=In(EnergyBalanceType),
        description="Energy balance construction flag",
        doc="""Indicates what type of energy balance should be constructed.
    **default** - EnergyBalanceType.useDefault.
    **Valid values:** {
    **EnergyBalanceType.useDefault - refer to property package for default
    balance type
    **EnergyBalanceType.none** - exclude energy balances,
    **EnergyBalanceType.enthalpyTotal** - single enthalpy balance for material,
    **EnergyBalanceType.enthalpyPhase** - enthalpy balances for each phase,
    **EnergyBalanceType.energyTotal** - single energy balance for material,
    **EnergyBalanceType.energyPhase** - energy balances for each phase.}"""))

    CONFIG.declare("momentum_balance_type", ConfigValue(
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
        #TODO: add config errors here:
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

        if self.config.transformation_method is useDefault:
            _log.warning(
                "Discretization method was "
                "not specified for the "
                "reverse osmosis module. "
                "Defaulting to finite "
                "difference method."
            )
            self.config.transformation_method = "dae.finite_difference"

        if self.config.transformation_scheme is useDefault:
            _log.warning(
                "Discretization scheme was "
                "not specified for the "
                "reverse osmosis module."
                "Defaulting to backward finite "
                "difference."
            )
            self.config.transformation_scheme = "BACKWARD"


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

        # Build 1D Control volume for feed side
        self.feed_side = ControlVolume1DBlock(default={
            "dynamic": self.config.dynamic,
            "has_holdup": self.config.has_holdup,
            "area_definition": self.config.area_definition,
            "property_package": self.config.property_package,
            "property_package_args": self.config.property_package_args,
            "transformation_method": self.config.transformation_method,
            "transformation_scheme": self.config.transformation_scheme,
            "finite_elements": self.config.finite_elements,
            "collocation_points": self.config.collocation_points
        })

        feed_side = self.feed_side
        # Add geometry to feed side
        feed_side.add_geometry()
        # Add state blocks to feed side
        feed_side.add_state_blocks(has_phase_equilibrium=False)
        # Populate feed side
        feed_side.add_material_balances(balance_type=self.config.material_balance_type,
                                        has_mass_transfer=True)
        feed_side.add_momentum_balances(balance_type=self.config.momentum_balance_type,
                                        has_pressure_change=self.config.has_pressure_change)
        # Apply transformation to feed side
        feed_side.apply_transformation()
        # Add inlet/outlet ports for feed side
        self.add_inlet_port(name="feed_inlet", block=feed_side)
        self.add_outlet_port(name="feed_outlet", block=feed_side)
        # Make indexed stateblock and separate stateblock for permeate-side and permeate outlet, respectively.
        tmp_dict = dict(**self.config.property_package_args)
        tmp_dict["has_phase_equilibrium"] = False
        tmp_dict["parameters"] = self.config.property_package
        tmp_dict["defined_state"] = False  # these blocks are not inlets
        self.permeate_side = self.config.property_package.state_block_class(
            self.flowsheet().config.time,
            self.feed_side.length_domain,
            doc="Material properties of permeate along permeate channel",
            default=tmp_dict)
        self.permeate_out = self.config.property_package.state_block_class(
            self.flowsheet().config.time,
            doc="Material properties of mixed permeate exiting the module",
            default=tmp_dict)
        # Add port to permeate_out
        self.add_port(name="permeate_outlet", block=self.permeate_out)

        # ==========================================================================
        """ Add references to control volume geometry."""
        add_object_reference(self, 'length', feed_side.length)
        add_object_reference(self, 'feed_area_cross', feed_side.area)

        # Add reference to pressure drop for feed side only
        if (self.config.has_pressure_change is True and
                self.config.momentum_balance_type != MomentumBalanceType.none):
            add_object_reference(self, 'deltaP', feed_side.deltaP)

        self._make_performance()

    def _make_performance(self):
        """
        Variables and constraints for unit model.

        Args:
            None

        Returns:
            None
        """

        # Units
        units_meta = \
            self.config.property_package.get_metadata().get_derived_units

        nfe = len(self.feed_side.length_domain)-1

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
        self.recovery_vol_phase = Var(
            self.flowsheet().config.time,
            self.config.property_package.phase_list,
            initialize=0.4,
            bounds=(1e-2, 1 - 1e-6),
            units=pyunits.dimensionless,
            doc='Volumetric recovery rate')

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
            self.feed_side.length_domain,
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

        self.width = Var(
            initialize=1,
            bounds=(1e-1, 1e3),
            domain=NonNegativeReals,
            units=units_meta('length'),
            doc='Membrane width')

        # mass transfer
        # TODO: replace self.recovery_vol_phase[t, 'Liq'] w/self.recovery_mass_phase_comp[t, 'Liq', j])
        def mass_transfer_phase_comp_initialize(b, t, x, p, j):
            return value(self.feed_side.properties[t, x].get_material_flow_terms('Liq', j)
                         * self.recovery_vol_phase[t, 'Liq'])

        self.mass_transfer_phase_comp = Var(
            self.flowsheet().config.time,
            self.feed_side.length_domain,
            self.config.property_package.phase_list,
            self.config.property_package.component_list,
            initialize=mass_transfer_phase_comp_initialize,
            bounds=(1e-8, 1e6),
            domain=NonNegativeReals,
            units=units_meta('mass') * units_meta('time')**-1 * units_meta('length')**-1,
            doc='Mass transfer to permeate')
        # ==========================================================================
        # Volumetric Recovery rate

        @self.Constraint(self.flowsheet().config.time)
        def eq_recovery_vol_phase(b, t):
            return (b.recovery_vol_phase[t, 'Liq'] ==
                    b.permeate_out[t].flow_vol_phase['Liq'] /
                    b.feed_side.properties[t, self.feed_side.length_domain.first()].flow_vol_phase['Liq'])
        # ==========================================================================
        # Mass transfer term equation

        @self.Constraint(self.flowsheet().config.time,
                         self.feed_side.length_domain,
                         self.config.property_package.phase_list,
                         self.config.property_package.component_list,
                         doc="Mass transfer term")
        def eq_mass_transfer_term(b, t, x, p, j):
            if x == b.feed_side.length_domain.first():
                return Constraint.Skip
            else:
                return b.mass_transfer_phase_comp[t, x, p, j] == -b.feed_side.mass_transfer_term[t, x, p, j]
        # ==========================================================================
        # Membrane area equation

        @self.Constraint(doc="Membrane area")
        def eq_area(b):
            return b.area == b.length * b.width
        # ==========================================================================
        # Mass flux = feed mass transfer equation

        @self.Constraint(self.flowsheet().config.time,
                         self.feed_side.length_domain,
                         self.config.property_package.phase_list,
                         self.config.property_package.component_list,
                         doc="Mass transfer term")
        def eq_mass_flux_equal_mass_transfer(b, t, x, p, j):
            if x == b.feed_side.length_domain.first():
                return Constraint.Skip
            else:
                return b.flux_mass_phase_comp[t, x, p, j] * b.width == -b.feed_side.mass_transfer_term[t, x, p, j]
        # ==========================================================================
        # Mass flux equations (Jw and Js)
        @self.Constraint(self.flowsheet().config.time,
                         self.feed_side.length_domain,
                         self.config.property_package.phase_list,
                         self.config.property_package.component_list,
                         doc="Solvent and solute mass flux")
        def eq_flux_mass(b, t, x, p, j):
            if x == b.feed_side.length_domain.first():
                return Constraint.Skip
            else:
                prop_feed = b.feed_side.properties[t, x]
                prop_perm = b.permeate_side[t, x]
                comp = self.config.property_package.get_component(j)
                if comp.is_solvent():
                    return (b.flux_mass_phase_comp[t, x, p, j] == b.A_comp[t, j] * b.dens_solvent
                            * ((prop_feed.pressure - prop_perm.pressure)
                               - (prop_feed.pressure_osm - prop_perm.pressure_osm)))
                elif comp.is_solute():
                    return (b.flux_mass_phase_comp[t, x, p, j] == b.B_comp[t, j]
                            * (prop_feed.conc_mass_phase_comp[p, j] - prop_perm.conc_mass_phase_comp[p, j]))
        # ==========================================================================
        # Final permeate mass flow rate (of solvent and solute) --> Mp,j, final = sum(Mp,j)

        @self.Constraint(self.flowsheet().config.time,
                         self.config.property_package.phase_list,
                         self.config.property_package.component_list,
                         doc="Permeate mass flow rates exiting unit")
        def eq_permeate_production(b, t, p, j):
            return (b.permeate_out[t].get_material_flow_terms(p, j)
                    == sum(b.permeate_side[t, x].get_material_flow_terms(p, j)
                           for x in b.feed_side.length_domain if x != 0))
        # ==========================================================================
        # Feed and permeate-side mass transfer connection --> Mp,j = Mf,transfer = Jj * W * L/n

        @self.Constraint(self.flowsheet().config.time,
                         self.feed_side.length_domain,
                         self.config.property_package.phase_list,
                         self.config.property_package.component_list,
                         doc="Mass transfer from feed to permeate")
        def eq_connect_mass_transfer(b, t, x, p, j):
            if x == b.feed_side.length_domain.first():
                return Constraint.Skip
            else:
                return (b.permeate_side[t, x].get_material_flow_terms(p, j)
                        == -b.feed_side.mass_transfer_term[t, x, p, j] * b.length / nfe)
        # # ==========================================================================
        # Feed-side isothermal conditions

        @self.Constraint(self.flowsheet().config.time,
                         self.feed_side.length_domain,
                         doc="Isothermal assumption for permeate")
        def eq_feed_isothermal(b, t, x):
            if x == b.feed_side.length_domain.first():
                return Constraint.Skip
            else:
                return b.feed_side.properties[t, b.feed_side.length_domain.first()].temperature == \
                       b.feed_side.properties[t, x].temperature
        # # ==========================================================================
        # Feed and permeate-side isothermal conditions

        @self.Constraint(self.flowsheet().config.time,
                         self.feed_side.length_domain,
                         doc="Isothermal assumption for permeate")
        def eq_permeate_isothermal(b, t, x):
            return b.feed_side.properties[t, x].temperature == \
                   b.permeate_side[t, x].temperature
        # ==========================================================================
        # isothermal conditions at permeate outlet

        @self.Constraint(self.flowsheet().config.time,
                         doc="Isothermal assumption for permeate out")
        def eq_permeate_outlet_isothermal(b, t):
            return b.feed_side.properties[t, b.feed_side.length_domain.first()].temperature == \
                   b.permeate_out[t].temperature
        # ==========================================================================
        # isobaric conditions across permeate channel and at permeate outlet

        @self.Constraint(self.flowsheet().config.time,
                         self.feed_side.length_domain,
                         doc="Isobaric assumption for permeate out")
        def eq_permeate_outlet_isobaric(b, t, x):
            return b.permeate_side[t, x].pressure == \
                   b.permeate_out[t].pressure

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

        init_log.info('Starting initialization')
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

        flags_permeate_out = blk.permeate_out.initialize(
            outlvl=outlvl,
            optarg=optarg,
            solver=solver,
            state_args=permeate_block_args)

        init_log.info_high("Initialization Step 1 Complete.")
        if degrees_of_freedom(blk) != 0:
            raise Exception(f"Initialization was called on {blk} "
                            f"but it had {degrees_of_freedom(blk)} degree(s) of freedom "
                            f"when 0 was expected. Check that the appropriate variables are fixed.")
        # ---------------------------------------------------------------------
        # Step 2: Solve unit
        with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
            results = solve_indexed_blocks(opt, [blk], tee=slc.tee)

        with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
            res = opt.solve(blk, tee=slc.tee)

    def _get_performance_contents(self, time_point=0):
        var_dict = {}
        var_dict["Membrane Area"] = self.area
        var_dict["Membrane Length"] = self.length
        var_dict["Membrane Width"] = self.width
        #TODO: add more vars
        return {"vars": var_dict}

    def _get_stream_table_contents(self, time_point=0):
        return create_stream_table_dataframe(
            {
                "Feed Inlet": self.feed_inlet,
                "Feed Outlet": self.feed_outlet,
                "Permeate Outlet": self.permeate_outlet,
            },
            time_point=time_point,
        )

    def calculate_scaling_factors(self):
        super().calculate_scaling_factors()

        # setting scaling factors for variables
        # these variables should have user input, if not there will be a warning
        if iscale.get_scaling_factor(self.area) is None:
            sf = iscale.get_scaling_factor(self.area, default=1, warning=True)
            iscale.set_scaling_factor(self.area, sf)

        if iscale.get_scaling_factor(self.width) is None:
            sf = iscale.get_scaling_factor(self.width, default=1, warning=True)
            iscale.set_scaling_factor(self.width, sf)

        if iscale.get_scaling_factor(self.length) is None:
            sf = iscale.get_scaling_factor(self.length, default=1, warning=True)
            iscale.set_scaling_factor(self.length, sf)

        # will not override if the user provides the scaling factor
        if iscale.get_scaling_factor(self.A_comp) is None:
            iscale.set_scaling_factor(self.A_comp, 1e12)

        if iscale.get_scaling_factor(self.B_comp) is None:
            iscale.set_scaling_factor(self.B_comp, 1e8)

        if iscale.get_scaling_factor(self.dens_solvent) is None:
            sf = iscale.get_scaling_factor(self.feed_side.properties[0, 0].dens_mass_phase['Liq'])
            iscale.set_scaling_factor(self.dens_solvent, sf)

        for (t, x, p, j), v in self.flux_mass_phase_comp.items():
            if iscale.get_scaling_factor(v) is None:
                comp = self.config.property_package.get_component(j)
                if x == self.feed_side.length_domain.first():
                    if comp.is_solvent():
                        iscale.set_scaling_factor(v, 5e4)  # inverse of initial value from flux_mass_phase_comp_initialize
                    elif comp.is_solute():
                        iscale.set_scaling_factor(v, 1e6)  # inverse of initial value from flux_mass_phase_comp_initialize
                else:
                    if comp.is_solvent():  # scaling based on solvent flux equation
                        sf = (iscale.get_scaling_factor(self.A_comp[t, j])
                              * iscale.get_scaling_factor(self.dens_solvent)
                              * iscale.get_scaling_factor(self.feed_side.properties[t, x].pressure))
                        iscale.set_scaling_factor(v, sf)
                    elif comp.is_solute():  # scaling based on solute flux equation
                        sf = (iscale.get_scaling_factor(self.B_comp[t, j])
                              * iscale.get_scaling_factor(self.feed_side.properties[t, x].conc_mass_phase_comp[p, j]))
                        iscale.set_scaling_factor(v, sf)

        for (t, x, p, j), v in self.eq_mass_flux_equal_mass_transfer.items():
            if iscale.get_scaling_factor(v) is None:
                if x == self.feed_side.length_domain.first():
                    pass
                else:
                    sf = iscale.get_scaling_factor(self.flux_mass_phase_comp[t, x, p, j])\
                         * iscale.get_scaling_factor(self.width)
                    comp = self.config.property_package.get_component(j)
                    if comp.is_solute:
                        sf *= 1e2  # solute typically has mass transfer 2 orders magnitude less than flow
                    iscale.set_scaling_factor(v, sf)

        for (t, x, p, j), v in self.mass_transfer_phase_comp.items():
            if iscale.get_scaling_factor(v) is None:
                sf = iscale.get_scaling_factor(self.feed_side.properties[t, x].get_material_flow_terms(p, j)) \
                     / iscale.get_scaling_factor(self.feed_side.length)
                comp = self.config.property_package.get_component(j)
                if comp.is_solute:
                    sf *= 1e2  # solute typically has mass transfer 2 orders magnitude less than flow
                iscale.set_scaling_factor(v, sf)

        for v in self.feed_side.pressure_dx.values():
            iscale.set_scaling_factor(v, 1e6)

        # Scale constraints
        for ind, c in self.eq_mass_transfer_term.items():
            sf = iscale.get_scaling_factor(self.mass_transfer_phase_comp[ind])
            iscale.constraint_scaling_transform(c, sf)

        for ind, c in self.eq_connect_mass_transfer.items():
            sf = iscale.get_scaling_factor(self.mass_transfer_phase_comp[ind])
            iscale.constraint_scaling_transform(c, sf)

        sf = iscale.get_scaling_factor(self.area)
        iscale.constraint_scaling_transform(self.eq_area, sf)

        for ind, c in self.eq_permeate_production.items():
            # TODO: revise this scaling factor
            iscale.constraint_scaling_transform(c, 1)

        for ind, c in self.eq_flux_mass.items():
            sf = iscale.get_scaling_factor(self.flux_mass_phase_comp[ind])
            iscale.constraint_scaling_transform(c, sf)

        for (t, x), c in self.eq_feed_isothermal.items():
            sf = iscale.get_scaling_factor(self.feed_side.properties[t, x].temperature)
            iscale.constraint_scaling_transform(c, sf)

        for (t, x), c in self.eq_permeate_isothermal.items():
            sf = iscale.get_scaling_factor(self.feed_side.properties[t, x].temperature)
            iscale.constraint_scaling_transform(c, sf)

        for t, c in self.eq_permeate_outlet_isothermal.items():
            sf = iscale.get_scaling_factor(self.feed_side.properties[t, 0].temperature)
            iscale.constraint_scaling_transform(c, sf)

        for (t, x), c in self.eq_permeate_outlet_isobaric.items():
            sf = iscale.get_scaling_factor(self.permeate_side[t, x].pressure)
            iscale.constraint_scaling_transform(c, sf)

        for t, c in self.eq_recovery_vol_phase.items():
            iscale.constraint_scaling_transform(self.eq_recovery_vol_phase[t], 1)
