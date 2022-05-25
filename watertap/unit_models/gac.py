###############################################################################
# WaterTAP Copyright (c) 2021, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory, Oak Ridge National
# Laboratory, National Renewable Energy Laboratory, and National Energy
# Technology Laboratory (subject to receipt of any required approvals from
# the U.S. Dept. of Energy). All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. These files are also available online at the URL
# "https://github.com/watertap-org/watertap/"
#
###############################################################################

from pyomo.environ import (
    Block,
    Set,
    Var,
    Param,
    Suffix,
    NonNegativeReals,
    Reference,
    units as pyunits,
)
from pyomo.common.config import ConfigBlock, ConfigValue, In

from enum import Enum, auto

from idaes.core import (
    ControlVolume0DBlock,
    declare_process_block_class,
    MaterialBalanceType,
    MomentumBalanceType,
    UnitModelBlockData,
    useDefault,
    MaterialFlowBasis,
)
from idaes.core.util import get_solver
from idaes.core.util.tables import create_stream_table_dataframe
from idaes.core.util.config import is_physical_parameter_block
from idaes.core.util.exceptions import ConfigurationError
import idaes.core.util.scaling as iscale
import idaes.logger as idaeslog

__author__ = "Hunter Barber"

_log = idaeslog.getLogger(__name__)

# ---------------------------------------------------------------------
"""
First model build inputs
"""
# Inlet conditions
co = 23.2  # inlet concentration [microg/L]
# TODO: Make sure property pack supports liquid, solute_list
# Freundlich parameters specific for contaminant and GAC particle
k = 37.9  # (microg/gm)/(L/microg)^(1/n)
ninv = 0.8316
# User specified values
kff = 1  # k reduction due to background organics
kf = 3.29e-5  # liquid phase/film transfer rate, [m/s]
Ds = 1.77e-13  # Surface diffusion coefficient, [m2/s]
# Design Conditions
ebct = 5  # empty bed contact time [min]
eps = 0.449  # bed void fraction
ceff = 5  # effluent concentration [microg/L]
rhoa = 722  # apparent gac density [kg/m3]
dp = 0.00106  # gac particle diameter[m]
# constants a0,a1 for min stanton number to achieve constant pattern as a function of ninv and Bi
a0 = 3.68421e0
a1 = 13.1579
# constants b0,b1,b2,b3,b4 for constant pattern solution as a function of ninv and Bi
b0 = 0.784576
b1 = 0.239663
b2 = 0.484422
b3 = 0.003206
b4 = 0.134987
# TODO: Initialize variables and add scaling
# TODO: Convert to steady state assumption
"""
Second model build inputs additions
"""
# TODO: Add Reynolds, Schmidt, and other intermediate equations
# Inlet conditions
qo = 0.89 / 60  # [m3/s]
temp = 10  # [°C]
molvol = 98.1  # molal volume of contaminant [cm3/mol]
# water properties
rhow = 999.7  # [kg/m3]
muw = 1.3097e-3  # [Ns/m2]
# Freundlich parameters
# TODO: Add surface diffusion coefficient correlation option
# TODO: Add Gnielinshi liquid phase film transfer rate correlation option
# User specified values
taup = 1  # tortuosity of the path that the adsorbate must take as compared to the radius, dimensionless
scf = 1.5  # shape correction factor
# Design Conditions
vsup = 5 / 3600  # superficial mass velocity [m/s]
rhof = 450  # bulk gac particle density, [kg/m3]
epsp = 0.641  #
# TODO: Add adaptability for multiple solutes

# ---------------------------------------------------------------------
"""
class SurfaceDiffusionCoeffVal(Enum):
    specified = auto()  # surface diffusion coefficient is a user-specified value
    calculated = auto() # pressure drop across membrane channel is calculated
"""
# ---------------------------------------------------------------------
@declare_process_block_class("GAC")
class GACData(UnitModelBlockData):
    """
    Initial Granular Activated Carbon Model
    """

    # CONFIG are options for the unit model, this simple model only has the mandatory config options
    CONFIG = ConfigBlock()

    CONFIG.declare(
        "dynamic",
        ConfigValue(
            domain=In([False]),
            default=False,
            description="Dynamic model flag - must be False",
            doc="""Indicates whether this model will be dynamic or not,
    **default** = False. The filtration unit does not support dynamic
    behavior, thus this must be False.""",
        ),
    )
    CONFIG.declare(
        "has_holdup",
        ConfigValue(
            default=False,
            domain=In([False]),
            description="Holdup construction flag - must be False",
            doc="""Indicates whether holdup terms should be constructed or not.
    **default** - False. The filtration unit does not have defined volume, thus
    this must be False.""",
        ),
    )
    CONFIG.declare(
        "property_package",
        ConfigValue(
            default=useDefault,
            domain=is_physical_parameter_block,
            description="Property package to use for control volume",
            doc="""Property parameter object used to define property calculations,
    **default** - useDefault.
    **Valid values:** {
    **useDefault** - use default package from parent model or flowsheet,
    **PhysicalParameterObject** - a PhysicalParameterBlock object.}""",
        ),
    )
    CONFIG.declare(
        "property_package_args",
        ConfigBlock(
            implicit=True,
            description="Arguments to use for constructing property packages",
            doc="""A ConfigBlock with arguments to be passed to a property block(s)
    and used when constructing these,
    **default** - None.
    **Valid values:** {
    see property package for documentation.}""",
        ),
    )
    CONFIG.declare(
        "material_balance_type",
        ConfigValue(
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
    **MaterialBalanceType.total** - use total material balance.}""",
        ),
    )
    CONFIG.declare(
        "momentum_balance_type",
        ConfigValue(
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
        **MomentumBalanceType.momentumPhase** - momentum balances for each phase.}""",
        ),
    )
    CONFIG.declare(
        "has_pressure_change",
        ConfigValue(
            default=False,
            domain=In([False]),  # TODO: domain=In([True, False])
            description="Pressure change term construction flag",
            doc="""Indicates whether terms for pressure change should be
        constructed,
        **default** - False.
        **Valid values:** {
        **True** - include pressure change terms,
        **False** - exclude pressure change terms.}""",
        ),
    )

    # ---------------------------------------------------------------------
    def build(self):

        super().build()
        # create blank scaling factors to be populated later
        self.scaling_factor = Suffix(direction=Suffix.EXPORT)
        units_meta = self.config.property_package.get_metadata().get_derived_units

        # Build control volume
        self.treatwater = ControlVolume0DBlock(
            default={
                "dynamic": False,
                "has_holdup": False,
                "property_package": self.config.property_package,
                "property_package_args": self.config.property_package_args,
            }
        )
        self.treatwater.add_state_blocks(has_phase_equilibrium=False)
        self.treatwater.add_material_balances(
            balance_type=self.config.material_balance_type, has_mass_transfer=True
        )
        self.treatwater.add_momentum_balances(
            balance_type=self.config.momentum_balance_type,
            has_pressure_change=False,
        )

        @self.treatwater.Constraint(
            self.flowsheet().config.time, doc="isothermal assumption for water flow"
        )
        def eq_isothermal(b, t):
            return b.properties_in[t].temperature == b.properties_out[t].temperature

        # Add block for spent GAC
        tmp_dict = dict(**self.config.property_package_args)
        tmp_dict["has_phase_equilibrium"] = False
        tmp_dict["parameters"] = self.config.property_package
        tmp_dict["defined_state"] = False  # permeate block is not an inlet
        self.removal = self.config.property_package.state_block_class(
            self.flowsheet().config.time,
            doc="Material properties of spent gac",
            default=tmp_dict,
        )

        @self.Constraint(
            self.flowsheet().config.time, doc="Isothermal assumption for spent GAC"
        )
        def eq_isothermal_adsorbate(b, t):
            return b.treatwater.properties_in[t].temperature == b.removal[t].temperature

        @self.Constraint(
            self.flowsheet().config.time, doc="Isobaric assumption for spent GAC"
        )
        def eq_isobaric_adsorbate(b, t):
            return b.treatwater.properties_in[t].pressure == b.removal[t].pressure

        # Add ports
        self.add_inlet_port(name="inlet", block=self.treatwater)
        self.add_outlet_port(name="outlet", block=self.treatwater)
        self.add_port(name="spent_gac", block=self.removal)

        # ---------------------------------------------------------------------
        # Variable declaration
        # mass transfer
        self.contam_removal = Var(
            self.config.property_package.solute_set,
            initialize=0.9,
            bounds=(0, 1),
            domain=NonNegativeReals,
            units=pyunits.dimensionless,
            doc="Moles of solute absorbed into GAC",
        )

        self.freund_ninv = Var(
            initialize=0.5,
            bounds=(0, 5),
            domain=NonNegativeReals,
            units=pyunits.dimensionless,
            doc="Freundlich 1/n parameter",
        )

        # TODO: Figure out non-fixed exponent
        self.freund_k = Var(
            initialize=10,
            bounds=(0, 1000),
            domain=NonNegativeReals,
            units=((units_meta("length") ** 3) * units_meta("mass") ** -1) ** 0.8316,
            doc="Freundlich k parameter",
        )

        self.ebct = Var(
            initialize=100,
            bounds=(0, None),
            domain=NonNegativeReals,
            units=units_meta("time"),
            doc="Empty bed contact time",
        )

        self.eps_bed = Var(
            initialize=0.5,
            bounds=(0, 1),
            domain=NonNegativeReals,
            units=pyunits.dimensionless,
            doc="GAC bed void fraction",
        )

        self.thru = Var(
            initialize=10,
            bounds=(0, None),
            domain=NonNegativeReals,
            units=pyunits.dimensionless,
            doc="Mass throughput",
        )

        self.tau = Var(
            initialize=100,
            bounds=(0, None),
            domain=NonNegativeReals,
            units=units_meta("time"),
            doc="Packed bed contact time",
        )

        self.replace_time = Var(
            initialize=100,
            bounds=(0, None),
            domain=NonNegativeReals,
            units=units_meta("time"),
            doc="Replace time for saturated GAC",
        )

        # ---------------------------------------------------------------------
        # GAC particle properties
        self.particle_rho_app = Var(
            initialize=800,
            bounds=(0, None),
            domain=NonNegativeReals,
            units=units_meta("mass") * units_meta("length") ** -3,
            doc="GAC particle apparent density",
        )

        self.particle_dp = Var(
            initialize=0.001,
            bounds=(0, None),
            domain=NonNegativeReals,
            units=units_meta("length"),
            doc="GAC particle diameter",
        )

        self.kf = Var(
            initialize=1e-5,
            bounds=(0, None),
            domain=NonNegativeReals,
            units=units_meta("length") * units_meta("time") ** -1,
            doc="Liquid phase film transfer rate",
        )

        self.ds = Var(
            initialize=1e-12,
            bounds=(0, None),
            domain=NonNegativeReals,
            units=units_meta("length") ** 2 * units_meta("time") ** -1,
            doc="Surface diffusion coefficient",
        )

        # ---------------------------------------------------------------------
        # Minimum conditions to achieve a constant pattern solution
        self.min_st = Var(
            initialize=10,
            bounds=(0, None),
            domain=NonNegativeReals,
            units=pyunits.dimensionless,
            doc="Minimum Stanton number",
        )

        self.min_ebct = Var(
            initialize=100,
            bounds=(0, None),
            domain=NonNegativeReals,
            units=units_meta("time"),
            doc="Minimum empty bed contact time",
        )

        self.min_tau = Var(
            initialize=100,
            bounds=(0, None),
            domain=NonNegativeReals,
            units=units_meta("time"),
            doc="Minimum packed bed contact time",
        )

        self.min_time = Var(
            initialize=100,
            bounds=(0, None),
            domain=NonNegativeReals,
            units=units_meta("time"),
            doc="Minimum elapsed time",
        )

        # ---------------------------------------------------------------------
        # Constants in regressed equations
        self.a0 = Var(
            initialize=1,
            bounds=(0, None),
            domain=NonNegativeReals,
            units=pyunits.dimensionless,
            doc="Stanton equation parameter",
        )

        self.a1 = Var(
            initialize=1,
            bounds=(0, None),
            domain=NonNegativeReals,
            units=pyunits.dimensionless,
            doc="Stanton equation parameter",
        )

        self.b0 = Var(
            initialize=0.1,
            bounds=(0, None),
            domain=NonNegativeReals,
            units=pyunits.dimensionless,
            doc="Throughput equation parameter",
        )

        self.b1 = Var(
            initialize=0.1,
            bounds=(0, None),
            domain=NonNegativeReals,
            units=pyunits.dimensionless,
            doc="Throughput equation parameter",
        )

        self.b2 = Var(
            initialize=0.1,
            bounds=(0, None),
            domain=NonNegativeReals,
            units=pyunits.dimensionless,
            doc="Throughput equation parameter",
        )

        self.b3 = Var(
            initialize=0.1,
            bounds=(0, None),
            domain=NonNegativeReals,
            units=pyunits.dimensionless,
            doc="Throughput equation parameter",
        )

        self.b4 = Var(
            initialize=0.1,
            bounds=(0, None),
            domain=NonNegativeReals,
            units=pyunits.dimensionless,
            doc="Throughput equation parameter",
        )

        # ---------------------------------------------------------------------
        # Intermediate variables
        self.dg = Var(
            initialize=1000,
            bounds=(0, None),
            domain=NonNegativeReals,
            units=pyunits.dimensionless,
            doc="Solute distribution parameter",
        )

        self.bi = Var(
            initialize=10,
            bounds=(0, None),
            domain=NonNegativeReals,
            units=pyunits.dimensionless,
            doc="Solute distribution parameter",
        )

        # ---------------------------------------------------------------------
        # TODO: Support mole or mass based property packs
        @self.Constraint(
            self.flowsheet().config.time,
            self.config.property_package.solute_set,
            doc="Mass transfer term for solutes",
        )
        def eq_mass_transfer_solute(b, t, j):
            return (
                b.contam_removal[j]
                * b.treatwater.properties_in[t].get_material_flow_terms("Liq", j)
                == -b.treatwater.mass_transfer_term[t, "Liq", j]
            )

        @self.Constraint(
            self.flowsheet().config.time,
            self.config.property_package.solvent_set,
            doc="No mass transfer of solvents",
        )
        def eq_mass_transfer_solvent(b, t, j):
            return b.treatwater.mass_transfer_term[t, "Liq", j] == 0
            # will set to flow_mol_phase_comp.lb of 1e-8

        @self.Constraint(
            self.flowsheet().config.time,
            self.config.property_package.component_list,
            doc="Contaminant absorbed",
        )
        def eq_mass_transfer_absorbed(b, t, j):
            return (
                b.removal[t].get_material_flow_terms("Liq", j)
                == -b.treatwater.mass_transfer_term[t, "Liq", j]
            )

        @self.Constraint(
            self.flowsheet().config.time,
            self.config.property_package.solute_set,
            doc="Solute distribution parameter",
        )
        def eq_dg(b, t, j):
            return b.dg * b.eps_bed * b.treatwater.properties_in[
                t
            ].conc_mass_phase_comp["Liq", j] == b.particle_rho_app * b.freund_k * (
                b.treatwater.properties_in[t].conc_mass_phase_comp["Liq", j]
                ** b.freund_ninv
            )

        @self.Constraint(doc="Biot number")
        def eq_bi(b):
            return b.bi * b.ds * b.dg * b.eps_bed == b.kf * (b.particle_dp / 2) * (
                1 - b.eps_bed
            )

        @self.Constraint(
            doc="Minimum Stanton number to achieve constant pattern solution"
        )
        def eq_min_st_CPS(b):
            return b.min_st == b.a0 * b.bi + b.a1

        @self.Constraint(
            doc="Minimum empty bed contact time to achieve constant pattern solution"
        )
        def eq_min_ebct_CPS(b):
            return b.min_ebct * (1 - b.eps_bed) * b.kf == b.min_st * (b.particle_dp / 2)

        @self.Constraint(
            doc="Minimum packed bed contact time to achieve constant pattern solution"
        )
        def eq_min_tau(b):
            return b.min_tau == b.eps_bed * b.min_ebct

        @self.Constraint(doc="Minimum elapsed time for constant pattern solution")
        def eq_min_time_CPS(b):
            return b.min_time == b.min_tau * (b.dg + 1) * b.thru

        @self.Constraint(
            self.flowsheet().config.time,
            self.config.property_package.solute_set,
            doc="Throughput based on 5-parameter regression",
        )
        def eq_thru(b, t, j):
            return b.thru == b.b0 + b.b1 * (
                (
                    b.treatwater.properties_in[t].conc_mass_phase_comp["Liq", j]
                    / b.treatwater.properties_out[t].conc_mass_phase_comp["Liq", j]
                )
                ** b.b2
            ) + b.b3 / (
                1.01
                - (
                    (
                        b.treatwater.properties_in[t].conc_mass_phase_comp["Liq", j]
                        / b.treatwater.properties_out[t].conc_mass_phase_comp["Liq", j]
                    )
                    ** b.b4
                )
            )

        @self.Constraint(doc="residence time")
        def eq_tau(b):
            return b.tau == b.eps_bed * b.ebct

        @self.Constraint(doc="Bed replacement time")
        def eq_replacement_time(b):
            return b.replace_time == b.min_time + (b.tau - b.min_tau) * (b.dg + 1)

    # ---------------------------------------------------------------------
    # initialize method
    def initialize_build(
        blk, state_args=None, outlvl=idaeslog.NOTSET, solver=None, optarg=None
    ):
        """
        General wrapper for initialization routines

        Keyword Arguments:
            state_args : a dict of arguments to be passed to the property
                         package(s) to provide an initial state for
                         initialization (see documentation of the specific
                         property package) (default = {}).
            outlvl : sets output level of initialization routine
            optarg : solver options dictionary object (default=None)
            solver : str indicating which solver to use during
                     initialization (default = None)

        Returns: None
        """
        init_log = idaeslog.getInitLogger(blk.name, outlvl, tag="unit")
        solve_log = idaeslog.getSolveLogger(blk.name, outlvl, tag="unit")
        # Set solver options
        opt = get_solver(solver, optarg)

        # ---------------------------------------------------------------------
        # Initialize holdup block
        flags = blk.treatwater.initialize(
            outlvl=outlvl,
            optarg=optarg,
            solver=solver,
            state_args=state_args,
        )

        blk.treatwater.properties_in[0].conc_mass_phase_comp
        blk.treatwater.properties_in[0].flow_vol_phase["Liq"]
        blk.treatwater.properties_out[0].conc_mass_phase_comp
        blk.treatwater.properties_out[0].flow_vol_phase["Liq"]

        init_log.info_high("Initialization Step 1 Complete.")
        # ---------------------------------------------------------------------
        # Initialize permeate
        # Set state_args from inlet state
        if state_args is None:
            state_args = {}
            state_dict = blk.treatwater.properties_in[
                blk.flowsheet().config.time.first()
            ].define_port_members()

            for k in state_dict.keys():
                if state_dict[k].is_indexed():
                    state_args[k] = {}
                    for m in state_dict[k].keys():
                        state_args[k][m] = state_dict[k][m].value
                else:
                    state_args[k] = state_dict[k].value

        blk.removal.initialize(
            outlvl=outlvl,
            optarg=optarg,
            solver=solver,
            state_args=state_args,
        )
        init_log.info_high("Initialization Step 2 Complete.")

        # ---------------------------------------------------------------------
        # Solve unit
        with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
            res = opt.solve(blk, tee=slc.tee)
        init_log.info_high("Initialization Step 3 {}.".format(idaeslog.condition(res)))

        # ---------------------------------------------------------------------
        # Release Inlet state
        blk.treatwater.release_state(flags, outlvl + 1)
        init_log.info("Initialization Complete: {}".format(idaeslog.condition(res)))

    # ---------------------------------------------------------------------
    def calculate_scaling_factors(self):
        super().calculate_scaling_factors()

        # scaling for gac created variables

        # transforming constraints
        for (t, j), c in self.eq_mass_transfer_solute.items():
            sf = iscale.get_scaling_factor(
                self.treatwater.properties_in[t].flow_mol_phase_comp["Liq", j]
            )
            iscale.constraint_scaling_transform(c, sf)

        for (t, j), c in self.eq_mass_transfer_solvent.items():
            sf = 1
            iscale.constraint_scaling_transform(c, sf)

        for (t, j), c in self.eq_mass_transfer_absorbed.items():
            if j in self.config.property_package.solute_set:
                sf = iscale.get_scaling_factor(
                    self.treatwater.properties_in[t].flow_mol_phase_comp["Liq", j]
                )
                iscale.constraint_scaling_transform(c, sf)
            if j in self.config.property_package.solvent_set:
                sf = 1
                iscale.constraint_scaling_transform(c, sf)

        for ind, c in self.treatwater.eq_isothermal.items():
            sf = iscale.get_scaling_factor(self.treatwater.properties_in[0].temperature)
            iscale.constraint_scaling_transform(c, sf)

        for ind, c in self.eq_isothermal_adsorbate.items():
            sf = iscale.get_scaling_factor(self.treatwater.properties_in[0].temperature)
            iscale.constraint_scaling_transform(c, sf)

        for ind, c in self.eq_isobaric_adsorbate.items():
            sf = iscale.get_scaling_factor(self.treatwater.properties_in[0].pressure)
            iscale.constraint_scaling_transform(c, sf)
