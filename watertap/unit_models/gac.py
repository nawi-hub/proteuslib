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
    Var,
    Param,
    Set,
    Suffix,
    NonNegativeReals,
    PositiveIntegers,
    units as pyunits,
    check_optimal_termination,
)
from pyomo.common.config import Bool, ConfigBlock, ConfigValue, In
from enum import Enum, auto

from idaes.core import (
    declare_process_block_class,
    MaterialBalanceType,
    EnergyBalanceType,
    MomentumBalanceType,
    UnitModelBlockData,
    MaterialFlowBasis,
    useDefault,
)
from watertap.core.solvers import get_solver
from idaes.core.util.constants import Constants
from idaes.core.util.config import is_physical_parameter_block
from idaes.core.util.tables import create_stream_table_dataframe
from idaes.core.util.exceptions import ConfigurationError, InitializationError
import idaes.core.util.scaling as iscale
import idaes.logger as idaeslog

from watertap.core import ControlVolume0DBlock, InitializationMixin
from watertap.core.util.initialization import interval_initializer
from watertap.costing.unit_models.gac import cost_gac

__author__ = "Hunter Barber"

_log = idaeslog.getLogger(__name__)


# ---------------------------------------------------------------------
class CPHSDMCalculationMethod(Enum):
    input = auto()  # calculate CPHSDM model by inputting empirical parameters
    surrogate = auto()  # calculate CPHSDM model using pre-trained surrogate


class FilmTransferCoefficientType(Enum):
    fixed = auto()  # liquid phase film transfer coefficient is a user specified value
    calculated = (
        auto()
    )  # calculate liquid phase film transfer coefficient from the Gnielinshi correlation


class SurfaceDiffusionCoefficientType(Enum):
    fixed = auto()  # surface diffusion coefficient is a user specified value
    calculated = auto()  # calculate surface diffusion coefficient


# ---------------------------------------------------------------------
@declare_process_block_class("GAC")
class GACData(InitializationMixin, UnitModelBlockData):
    """
    Empirical Constant-Pattern-Homogeneous-Surface-Diffusion Model (CPHSDM) for Granular Activated Carbon -
    currently only available to model with the multicomp_aq_sol_prop_pack
    """

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
        "is_isothermal",
        ConfigValue(
            default=True,
            domain=Bool,
            description="""Assume isothermal conditions for control volume(s); energy_balance_type must be EnergyBalanceType.none,
    **default** - True.""",
        ),
    )

    CONFIG.declare(
        "energy_balance_type",
        ConfigValue(
            default=EnergyBalanceType.none,
            domain=In(EnergyBalanceType),
            description="Energy balance construction flag",
            doc="""Indicates what type of energy balance should be constructed,
    **default** - EnergyBalanceType.none.
    **Valid values:** {
    **EnergyBalanceType.useDefault - refer to property package for default
    balance type
    **EnergyBalanceType.none** - exclude energy balances,
    **EnergyBalanceType.enthalpyTotal** - single enthalpy balance for material,
    **EnergyBalanceType.enthalpyPhase** - enthalpy balances for each phase,
    **EnergyBalanceType.energyTotal** - single energy balance for material,
    **EnergyBalanceType.energyPhase** - energy balances for each phase.}""",
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
        "cphsdm_calaculation_method",
        ConfigValue(
            default=CPHSDMCalculationMethod.input,
            domain=In(CPHSDMCalculationMethod),
            description="CPHSDM calculation method",
            doc="""Select the method of calculations for the empirical CPHSDM expressions
        **default** - CPHSDMCalculationMethod.input.
        **Valid values:** {
        **CPHSDMCalculationMethod.input** - calculate CPHSDM model by inputting empirical parameterse,
        **CPHSDMCalculationMethod.surrogate** - calculate CPHSDM model using pre-trained surrogate}""",
        ),
    )
    CONFIG.declare(
        "film_transfer_coefficient_type",
        ConfigValue(
            default=FilmTransferCoefficientType.fixed,
            domain=In(FilmTransferCoefficientType),
            description="Surface diffusion coefficient",
            doc="""Indicates whether the liquid phase film transfer rate will be calculated or fixed by the user
        **default** - FilmTransferCoefficientType.fixed.
        **Valid values:** {
        **FilmTransferCoefficientType.fixed** - user specifies film transfer rate,
        **FilmTransferCoefficientType.calculated** - calculates film transfer rate based on the Gnielinshi correlation}""",
        ),
    )
    CONFIG.declare(
        "surface_diffusion_coefficient_type",
        ConfigValue(
            default=SurfaceDiffusionCoefficientType.fixed,
            domain=In(SurfaceDiffusionCoefficientType),
            description="Surface diffusion coefficient",
            doc="""Indicates whether the surface diffusion coefficient will be calculated or fixed by the user
        **default** - SurfaceDiffusionCoefficientType.fixed.
        **Valid values:** {
        **SurfaceDiffusionCoefficientType.fixed** - user specifies surface diffusion coefficient,
        **SurfaceDiffusionCoefficientType.calculated** - calculates surface diffusion coefficient by the Crittenden correlation}""",
        ),
    )
    CONFIG.declare(
        "target_species",
        ConfigValue(
            default=None,
            domain=list,
            description="Species target for adsorption, currently only supports single species",
            doc="""Indicate which component in the property package's component list is the target species
        for adsorption by the GAC system, currently the model supports a single species
        **default** - None.
        **Valid values:** {
        if the property package solute set only contains one item (two component, one solute and one solvent/water),
        the model will accept the single solute as the target species, for multi-solute systems a string of
        the component id must be provided.}""",
        ),
    )
    CONFIG.declare(
        "finite_elements_ss_approximation",
        ConfigValue(
            default=5,
            domain=int,
            description="Number of finite elements in operational time domain for steady state approximation",
            doc="""Number of finite elements to use when discretizing operational time (default=5)""",
        ),
    )

    def _validate_config(self):
        if (
            self.config.is_isothermal
            and self.config.energy_balance_type != EnergyBalanceType.none
        ):
            raise ConfigurationError(
                "If the isothermal assumption is used then the energy balance type must be none"
            )

    # ---------------------------------------------------------------------
    def build(self):

        super().build()

        # create blank scaling factors to be populated later
        self.scaling_factor = Suffix(direction=Suffix.EXPORT)
        # get default units from property package
        units_meta = self.config.property_package.get_metadata().get_derived_units
        # Check configs for errors
        self._validate_config()

        # ---------------------------------------------------------------------
        # separate target_species to be adsorbed and other species considered inert
        # apply target_species automatically if arg left to default and only one viable option exists

        if self.config.target_species is None:
            if len(self.config.property_package.solute_set) == 1:
                self.config.target_species = self.config.property_package.solute_set
                self.target_species = self.config.target_species
            else:
                raise ConfigurationError(
                    "'target species' is not specified for the GAC unit model, either specify 'target species'"
                    " argument or reduce solute set to a single component"
                )
        else:
            self.target_species = Set(dimen=1)
            for str_species in self.config.target_species:
                if not isinstance(str_species, str):
                    raise ConfigurationError(
                        f"item {str_species} within 'target_species' list is not of data type str"
                    )
                if str_species not in self.config.property_package.component_list:
                    raise ConfigurationError(
                        f"item {str_species} within 'target_species' list is not in 'component_list'"
                    )
                self.target_species.add(str_species)

        self.inert_species = Set(
            dimen=1,
            initialize=(
                self.config.property_package.component_list - self.config.target_species
            ),
        )

        # ---------------------------------------------------------------------
        # build control volume
        self.process_flow = ControlVolume0DBlock(
            dynamic=False,
            has_holdup=False,
            property_package=self.config.property_package,
            property_package_args=self.config.property_package_args,
        )
        self.process_flow.add_state_blocks(has_phase_equilibrium=False)
        self.process_flow.add_material_balances(
            balance_type=self.config.material_balance_type, has_mass_transfer=True
        )
        self.process_flow.add_energy_balances(
            balance_type=self.config.energy_balance_type,
            has_enthalpy_transfer=False,
        )
        self.process_flow.add_momentum_balances(
            balance_type=self.config.momentum_balance_type,
            has_pressure_change=False,
        )
        if self.config.is_isothermal:
            self.process_flow.add_isothermal_assumption()

        # inherit flow basis from property package
        self.flow_basis = self.process_flow.properties_in[
            self.flowsheet().config.time.first()
        ].get_material_flow_basis()

        # add port for adsorbed contaminant contained in nearly saturated GAC particles
        tmp_dict = dict(**self.config.property_package_args)
        tmp_dict["has_phase_equilibrium"] = False
        tmp_dict["parameters"] = self.config.property_package
        tmp_dict["defined_state"] = False
        self.gac_removed = self.config.property_package.state_block_class(
            self.flowsheet().config.time,
            doc="state block of the species contained within the gac removed from the bed",
            **tmp_dict,
        )

        # Add ports
        self.add_inlet_port(name="inlet", block=self.process_flow)
        self.add_outlet_port(name="outlet", block=self.process_flow)
        self.add_outlet_port(name="adsorbed", block=self.gac_removed)

        # ---------------------------------------------------------------------
        # Freundlich isotherm parameters and other property variables

        self.freund_k = Var(
            initialize=10,
            bounds=(0, None),
            domain=NonNegativeReals,
            units=pyunits.dimensionless,  # dynamic with freund_ninv, ((length ** 3) * (mass ** -1)) ** freund_ninv,
            doc="Freundlich isotherm k parameter, must be provided in base [L3/M] units",
        )
        self.freund_ninv = Var(
            initialize=0.5,
            bounds=(0, 1),
            domain=NonNegativeReals,
            units=pyunits.dimensionless,
            doc="Freundlich isotherm 1/n parameter",
        )
        self.ds = Var(
            initialize=1e-14,
            bounds=(0, None),
            domain=NonNegativeReals,
            units=units_meta("length") ** 2 * units_meta("time") ** -1,
            doc="surface diffusion coefficient",
        )
        self.kf = Var(
            initialize=1e-5,
            bounds=(0, None),
            domain=NonNegativeReals,
            units=units_meta("length") * units_meta("time") ** -1,
            doc="liquid phase film transfer coefficient",
        )
        self.equil_conc = Var(
            initialize=1e-4,
            bounds=(0, None),
            domain=NonNegativeReals,
            units=pyunits.dimensionless,
            doc="equilibrium concentration of adsorbed phase with liquid phase",
        )
        self.dg = Var(
            initialize=1e5,
            bounds=(0, None),
            domain=NonNegativeReals,
            units=pyunits.dimensionless,
            doc="solute distribution parameter",
        )
        self.N_Bi = Var(
            initialize=10,
            bounds=(0, None),
            domain=NonNegativeReals,
            units=pyunits.dimensionless,
            doc="Biot number",
        )

        # ---------------------------------------------------------------------
        # bed dimensions and related variables

        self.velocity_sup = Var(
            initialize=0.001,
            bounds=(0, None),
            domain=NonNegativeReals,
            units=units_meta("length") * units_meta("time") ** -1,
            doc="superficial velocity",
        )
        self.velocity_int = Var(
            initialize=0.002,
            bounds=(0, None),
            domain=NonNegativeReals,
            units=units_meta("length") * units_meta("time") ** -1,
            doc="interstitial velocity",
        )
        self.bed_voidage = Var(
            initialize=0.5,
            bounds=(0, 1),
            domain=NonNegativeReals,
            units=pyunits.dimensionless,
            doc="bed void fraction",
        )
        self.bed_length = Var(
            initialize=5,
            bounds=(0, None),
            domain=NonNegativeReals,
            units=units_meta("length"),
            doc="bed length",
        )
        self.bed_diameter = Var(
            initialize=1,
            bounds=(0, None),
            domain=NonNegativeReals,
            units=units_meta("length"),
            doc="bed diameter, valid if considering only a single adsorber",
        )
        self.bed_area = Var(
            initialize=1,
            bounds=(0, None),
            domain=NonNegativeReals,
            units=units_meta("length") ** 2,
            doc="bed area, single adsorber area or sum of areas for adsorbers in parallel",
        )
        self.bed_volume = Var(
            initialize=5,
            bounds=(0, None),
            domain=NonNegativeReals,
            units=units_meta("length") ** 3,
            doc="bed volume",
        )
        self.ebct = Var(
            initialize=500,
            bounds=(0, None),
            domain=NonNegativeReals,
            units=units_meta("time"),
            doc="empty bed contact time",
        )
        self.residence_time = Var(
            initialize=100,
            bounds=(0, None),
            domain=NonNegativeReals,
            units=units_meta("time"),
            doc="fluid residence time in the bed",
        )
        self.bed_mass_gac = Var(
            initialize=100,
            bounds=(0, None),
            domain=NonNegativeReals,
            units=units_meta("mass"),
            doc="mass of fresh gac in the bed",
        )

        # ---------------------------------------------------------------------
        # gac particle properties
        # TODO: Potentially create a default option

        self.particle_dens_app = Var(
            initialize=1000,
            bounds=(0, None),
            domain=NonNegativeReals,
            units=units_meta("mass") * units_meta("length") ** -3,
            doc="gac apparent density",
        )
        self.particle_dens_bulk = Var(
            initialize=500,
            bounds=(0, None),
            domain=NonNegativeReals,
            units=units_meta("mass") * units_meta("length") ** -3,
            doc="gac bulk density",
        )
        self.particle_dia = Var(
            initialize=0.001,
            bounds=(0, None),
            domain=NonNegativeReals,
            units=units_meta("length"),
            doc="gac particle diameter",
        )

        # ---------------------------------------------------------------------
        # constants in empirical equations

        self.a0 = Var(
            initialize=1,
            bounds=(0, None),
            domain=NonNegativeReals,
            units=pyunits.dimensionless,
            doc="Stanton equation parameter 0",
        )
        self.a1 = Var(
            initialize=1,
            bounds=(0, None),
            domain=NonNegativeReals,
            units=pyunits.dimensionless,
            doc="Stanton equation parameter 1",
        )
        self.b0 = Var(
            initialize=0.1,
            bounds=(0, None),
            domain=NonNegativeReals,
            units=pyunits.dimensionless,
            doc="throughput equation parameter 0",
        )
        self.b1 = Var(
            initialize=0.1,
            bounds=(0, None),
            domain=NonNegativeReals,
            units=pyunits.dimensionless,
            doc="throughput equation parameter 1",
        )
        self.b2 = Var(
            initialize=0.1,
            bounds=(0, None),
            domain=NonNegativeReals,
            units=pyunits.dimensionless,
            doc="throughput equation parameter 2",
        )
        self.b3 = Var(
            initialize=0.1,
            bounds=(0, None),
            domain=NonNegativeReals,
            units=pyunits.dimensionless,
            doc="throughput equation parameter 3",
        )
        self.b4 = Var(
            initialize=0.1,
            bounds=(0, None),
            domain=NonNegativeReals,
            units=pyunits.dimensionless,
            doc="throughput equation parameter 4",
        )

        # ---------------------------------------------------------------------
        # conditions to achieve a constant pattern solution

        self.min_N_St = Var(
            initialize=10,
            bounds=(0, None),
            domain=NonNegativeReals,
            units=pyunits.dimensionless,
            doc="minimum Stanton number to achieve a constant pattern solution",
        )
        self.min_ebct = Var(
            initialize=500,
            bounds=(0, None),
            domain=NonNegativeReals,
            units=units_meta("time"),
            doc="minimum empty bed contact time to achieve a constant pattern solution",
        )
        self.throughput = Var(
            initialize=1,
            bounds=(0, None),
            domain=NonNegativeReals,
            units=pyunits.dimensionless,
            doc="specific throughput from empirical equation",
        )
        self.min_residence_time = Var(
            initialize=1000,
            bounds=(0, None),
            domain=NonNegativeReals,
            units=units_meta("time"),
            doc="minimum fluid residence time in the bed to achieve a constant pattern solution",
        )
        self.min_operational_time = Var(
            initialize=1e8,
            bounds=(0, None),
            domain=NonNegativeReals,
            units=units_meta("time"),
            doc="minimum operational time of the bed from fresh to achieve a constant pattern solution",
        )

        # ---------------------------------------------------------------------
        # performance variables of CPHSDM

        self.conc_ratio_replace = Var(
            initialize=0.5,
            bounds=(0, 1),
            domain=NonNegativeReals,
            units=pyunits.dimensionless,
            doc="effluent to inlet concentration ratio at operational time",
        )
        self.operational_time = Var(
            initialize=1e5,
            bounds=(0, None),
            domain=NonNegativeReals,
            units=units_meta("time"),
            doc="operational time of the bed from fresh",
        )
        self.bed_volumes_treated = Var(
            initialize=10000,
            bounds=(0, None),
            domain=NonNegativeReals,
            units=pyunits.dimensionless,
            doc="bed volumes treated at operational time",
        )

        # ---------------------------------------------------------------------
        # steady state approximation

        self.elements_ss_approx = Param(
            mutable=True,
            default=5,
            initialize=self.config.finite_elements_ss_approximation,
            domain=PositiveIntegers,
            units=pyunits.dimensionless,
            doc="number of discretized operational time elements used for steady state approximation",
        )
        self.conc_ratio_start_breakthrough = Param(
            mutable=True,
            default=0.01,
            initialize=0.01,
            domain=NonNegativeReals,
            units=pyunits.dimensionless,
            doc="the concentration ratio at which the breakthrough curve is to start, typically between 0.01 amd 0.05",
        )

        # create index for discretized elements with element [0] containing 0s for conc ratio and operational time
        ele_disc = range(self.elements_ss_approx.value + 1)
        ele_index = ele_disc[1:]

        self.ele_throughput = Var(
            ele_index,
            initialize=1,
            bounds=(0, None),
            domain=NonNegativeReals,
            units=pyunits.dimensionless,
            doc="specific throughput from empirical equation by discrete element",
        )
        self.ele_min_operational_time = Var(
            ele_index,
            initialize=1e8,
            bounds=(0, None),
            domain=NonNegativeReals,
            units=units_meta("time"),
            doc="minimum operational time of the bed from fresh to achieve a constant pattern solution by discrete element",
        )
        self.ele_conc_ratio_replace = Var(
            ele_disc,
            initialize=0.05,
            bounds=(0, 1),
            domain=NonNegativeReals,
            units=pyunits.dimensionless,
            doc="effluent to inlet concentration ratio at operational time by discrete element",
        )
        self.ele_operational_time = Var(
            ele_disc,
            initialize=1e5,
            bounds=(0, None),
            domain=NonNegativeReals,
            units=units_meta("time"),
            doc="operational time of the bed from fresh by discrete element",
        )
        self.ele_conc_ratio_term = Var(
            ele_index,
            initialize=0.05,
            bounds=(0, 1),
            domain=NonNegativeReals,
            units=pyunits.dimensionless,
            doc="trapezoid rule of elements for numerical integration of average concentration ratio",
        )
        self.conc_ratio_avg = Var(
            initialize=0.1,
            bounds=(0, 1),
            domain=NonNegativeReals,
            units=pyunits.dimensionless,
            doc="steady state approximation of average effluent to inlet concentration ratio in operational time by trapezoid rule",
        )
        self.mass_adsorbed = Var(
            initialize=10,
            bounds=(0, None),
            domain=NonNegativeReals,
            units=units_meta("mass"),
            doc="total mass of adsorbed species at operational time",
        )
        self.gac_usage_rate = Var(
            initialize=1e-2,
            bounds=(0, None),
            domain=NonNegativeReals,
            units=units_meta("mass") * units_meta("time") ** -1,
            doc="gac usage/replacement/regeneration rate",
        )

        # ---------------------------------------------------------------------
        # property equations and other dimensionless variables

        @self.Constraint(
            self.flowsheet().config.time,
            self.target_species,
            doc="equilibrium concentration",
        )
        def eq_equilibrium_concentration(b, t, j):
            freund_k_units = (
                (units_meta("length") ** 3) * units_meta("mass") ** -1
            ) ** b.freund_ninv
            return b.equil_conc == b.freund_k * freund_k_units * (
                b.process_flow.properties_in[t].conc_mass_phase_comp["Liq", j]
                ** b.freund_ninv
            )

        @self.Constraint(
            self.flowsheet().config.time,
            self.target_species,
            doc="solute distribution parameter",
        )
        def eq_dg(b, t, j):
            return b.dg * b.bed_voidage * b.process_flow.properties_in[
                t
            ].conc_mass_phase_comp["Liq", j] == b.particle_dens_app * b.equil_conc * (
                1 - b.bed_voidage
            )

        @self.Constraint(doc="Biot number")
        def eq_number_bi(b):
            return b.N_Bi * b.ds * b.dg * b.bed_voidage == b.kf * (
                b.particle_dia / 2
            ) * (1 - b.bed_voidage)

        # ---------------------------------------------------------------------
        # bed dimensions, gac particle, and sizing calculations

        @self.Constraint(doc="bed void fraction based on gac particle densities")
        def eq_bed_voidage(b):
            return b.bed_voidage == 1 - (b.particle_dens_bulk / b.particle_dens_app)

        @self.Constraint(doc="relating velocities based on bed voidage")
        def eq_velocity_relation(b):
            return b.velocity_sup == b.velocity_int * b.bed_voidage

        @self.Constraint(doc="bed length based on velocity and ebct")
        def eq_bed_length(b):
            return b.bed_length == b.velocity_sup * b.ebct

        @self.Constraint(doc="bed diameter and area relation")
        def eq_bed_diameter(b):
            return Constants.pi * (b.bed_diameter**2) == 4 * b.bed_area

        @self.Constraint(
            self.flowsheet().config.time,
            doc="bed area based on velocity and volumetric flow",
        )
        def eq_bed_area(b, t):
            return (
                b.bed_area * b.velocity_sup
                == b.process_flow.properties_in[t].flow_vol_phase["Liq"]
            )

        @self.Constraint(doc="bed volume based on cylindrical dimensions")
        def eq_bed_volume(b):
            return b.bed_volume == b.bed_length * b.bed_area

        @self.Constraint(doc="fluid residence time in the bed")
        def eq_residence_time(b):
            return b.residence_time == b.bed_voidage * b.ebct

        @self.Constraint(doc="total mass of gac in the bed")
        def eq_mass_gac_bed(b):
            return b.bed_mass_gac == b.particle_dens_bulk * b.bed_volume

        # ---------------------------------------------------------------------
        # gac CPHSDM intermediate equations

        @self.Constraint(
            doc="minimum Stanton number to achieve constant pattern solution"
        )
        def eq_min_number_st_cps(b):
            return b.min_N_St == b.a0 * b.N_Bi + b.a1

        @self.Constraint(
            doc="minimum empty bed contact time to achieve constant pattern solution"
        )
        def eq_min_ebct_cps(b):
            return b.min_ebct * (1 - b.bed_voidage) * b.kf == b.min_N_St * (
                b.particle_dia / 2
            )

        @self.Constraint(doc="throughput based on empirical 5-parameter regression")
        def eq_throughput(b):
            return b.throughput == b.b0 + b.b1 * (
                b.conc_ratio_replace**b.b2
            ) + b.b3 / (1.01 - (b.conc_ratio_replace**b.b4))

        @self.Constraint(
            doc="minimum fluid residence time in the bed to achieve a constant pattern solution"
        )
        def eq_min_residence_time_cps(b):
            return b.min_residence_time == b.bed_voidage * b.min_ebct

        @self.Constraint(
            doc="minimum operational time of the bed from fresh to achieve a constant pattern solution"
        )
        def eq_minimum_operational_time_cps(b):
            return (
                b.min_operational_time
                == b.min_residence_time * (b.dg + 1) * b.throughput
            )

        # ---------------------------------------------------------------------
        # gac performance equations

        @self.Constraint(
            doc="elapsed operational time between a fresh bed and the theoretical bed replacement"
        )
        def eq_operational_time(b):
            return b.operational_time == b.min_operational_time + (
                b.residence_time - b.min_residence_time
            ) * (b.dg + 1)

        @self.Constraint(doc="bed volumes treated")
        def eq_bed_volumes_treated(b):
            return (
                b.bed_volumes_treated * b.residence_time
                == b.operational_time * b.bed_voidage
            )

        # ---------------------------------------------------------------------
        # steady state approximation

        self.ele_conc_ratio_replace[0].fix(0)
        self.ele_operational_time[0].fix(0)

        @self.Constraint(
            ele_index,
            doc="throughput based on empirical 5-parameter regression by discretized element",
        )
        def eq_ele_throughput(b, ele):
            return b.ele_throughput[ele] == b.b0 + b.b1 * (
                b.ele_conc_ratio_replace[ele] ** b.b2
            ) + b.b3 / (1.01 - b.ele_conc_ratio_replace[ele] ** b.b4)

        @self.Constraint(
            ele_index,
            doc="minimum operational time of the bed from fresh to achieve a constant pattern solution by discretized element",
        )
        def eq_ele_min_operational_time(b, ele):
            return (
                b.ele_min_operational_time[ele]
                == b.min_residence_time * (b.dg + 1) * b.ele_throughput[ele]
            )

        @self.Constraint(
            ele_index,
            doc="creating evenly spaced discretized elements",
        )
        def eq_ele_conc_ratio_replace(b, ele):
            return b.ele_conc_ratio_replace[ele] == b.conc_ratio_start_breakthrough + (
                ele_disc[ele] - 1
            ) * (
                (b.conc_ratio_replace - b.conc_ratio_start_breakthrough)
                / (b.elements_ss_approx - 1)
            )

        @self.Constraint(
            ele_index,
            doc="operational time of the bed by discretized element",
        )
        def eq_ele_operational_time(b, ele):
            return b.ele_operational_time[ele] == b.ele_min_operational_time[ele] + (
                b.residence_time - b.min_residence_time
            ) * (b.dg + 1)

        @self.Constraint(
            ele_index,
            doc="finite element discretization of concentration ratios over time",
        )
        def eq_ele_conc_ratio_term(b, ele):
            return b.ele_conc_ratio_term[ele] == (
                (b.ele_operational_time[ele] - b.ele_operational_time[ele - 1])
                / b.ele_operational_time[self.elements_ss_approx.value]
            ) * (
                (b.ele_conc_ratio_replace[ele] + b.ele_conc_ratio_replace[ele - 1]) / 2
            )

        @self.Constraint(
            doc="summation of finite elements for average concentration during operating time"
        )
        def eq_conc_ratio_avg(b):
            return b.conc_ratio_avg == sum(
                b.ele_conc_ratio_term[ele] for ele in ele_index
            )

        @self.Constraint(
            self.target_species,
            doc="mass adsorbed in the operational time",
        )
        def eq_mass_adsorbed(b, j):
            return (
                b.mass_adsorbed
                == b.gac_removed[0].flow_mass_phase_comp["Liq", j] * b.operational_time
            )

        @self.Constraint(doc="steady state rate of new gac mass required")
        def eq_gac_usage_rate(b):
            return b.gac_usage_rate * b.operational_time == b.bed_mass_gac

        # ---------------------------------------------------------------------
        # balance equations

        # isothermal for port not in control volume
        @self.Constraint(
            self.flowsheet().config.time,
            doc="isothermal assumption for the species contained within the gac removed from the bed",
        )
        def eq_isothermal_gac_removed(b, t):
            return (
                b.process_flow.properties_in[t].temperature
                == b.gac_removed[t].temperature
            )

        # isobaric for port not in control volume
        @self.Constraint(
            self.flowsheet().config.time,
            doc="isobaric assumption for the species contained within the gac removed from the bed",
        )
        def eq_isobaric_gac_removed(b, t):
            return b.process_flow.properties_in[t].pressure == b.gac_removed[t].pressure

        # mass transfer of target_species
        # TODO: check for mass based (not mole) get_material_flow_terms, but ok under mcas_prop_pack
        @self.Constraint(
            self.flowsheet().config.time,
            self.target_species,
            doc="mass transfer for adsorbed solutes in 'target_species' within 'gac_removed' (out of the bed)",
        )
        def eq_mass_transfer_cv(b, t, j):
            return (1 - b.conc_ratio_avg) * b.process_flow.properties_in[
                t
            ].get_material_flow_terms("Liq", j) == (
                -b.process_flow.mass_transfer_term[t, "Liq", j]
            )

        # mass balance for port not in control volume
        @self.Constraint(
            self.flowsheet().config.time,
            self.target_species,
            doc="mass balance for port not in control volume",
        )
        def eq_mass_transfer_port(b, t, j):
            return (
                b.gac_removed[t].get_material_flow_terms("Liq", j)
                == -b.process_flow.mass_transfer_term[t, "Liq", j]
            )

        # no mass transfer of inert_species, fix to 0 to avoid 0 solver tolerances
        for j in self.inert_species:
            self.process_flow.mass_transfer_term[:, "Liq", j].fix(0)
            self.gac_removed[0].get_material_flow_terms("Liq", j).fix(0)

        # ---------------------------------------------------------------------
        if (
            self.config.film_transfer_coefficient_type
            == FilmTransferCoefficientType.calculated
        ):

            self.N_Re = Var(
                initialize=10,
                bounds=(0, None),
                domain=NonNegativeReals,
                units=pyunits.dimensionless,
                doc="Reynolds number, correlations using Reynolds number valid in Re < 2e4",
            )
            self.N_Sc = Var(
                initialize=2000,
                bounds=(0, None),
                domain=NonNegativeReals,
                units=pyunits.dimensionless,
                doc="Schmidt number, correlations using Schmidt number valid in 0.7 < Sc < 1e4",
            )
            self.shape_correction_factor = Var(
                initialize=1,
                bounds=(0, None),
                domain=NonNegativeReals,
                units=pyunits.dimensionless,
                doc="shape correction factor",
            )

            @self.Constraint(
                self.flowsheet().config.time,
                doc="Reynolds number calculation",
            )
            def eq_reynolds_number(b, t):
                return (
                    b.N_Re * b.process_flow.properties_in[t].visc_d_phase["Liq"]
                    == b.process_flow.properties_in[t].dens_mass_phase["Liq"]
                    * b.particle_dia
                    * b.velocity_int
                )

            @self.Constraint(
                self.flowsheet().config.time,
                self.target_species,
                doc="Schmidt number calculation",
            )
            def eq_schmidt_number(b, t, j):
                return (
                    b.N_Sc
                    * b.process_flow.properties_in[t].dens_mass_phase["Liq"]
                    * b.process_flow.properties_in[t].diffus_phase_comp["Liq", j]
                    == b.process_flow.properties_in[t].visc_d_phase["Liq"]
                )

            @self.Constraint(
                self.flowsheet().config.time,
                self.target_species,
                doc="liquid phase film transfer rate from the Gnielinski correlation",
            )
            def eq_film_transfer_rate(b, t, j):
                return 1 == (b.kf * b.particle_dia) / (
                    b.shape_correction_factor
                    * (1 + 1.5 * (1 - b.bed_voidage))
                    * b.process_flow.properties_in[t].diffus_phase_comp["Liq", j]
                    * (2 + 0.644 * (b.N_Re**0.5) * (b.N_Sc ** (1 / 3)))
                )

        # ---------------------------------------------------------------------
        if (
            self.config.surface_diffusion_coefficient_type
            == SurfaceDiffusionCoefficientType.calculated
        ):

            self.particle_porosity = Var(
                initialize=0.65,
                bounds=(0, None),
                domain=NonNegativeReals,
                units=pyunits.dimensionless,
                doc="gac particle porosity",
            )
            self.tort = Var(
                initialize=1,
                bounds=(0, None),
                domain=NonNegativeReals,
                units=pyunits.dimensionless,
                doc="tortuosity of the path that the adsorbate must take as compared to the radius",
            )
            self.spdfr = Var(
                initialize=1,
                bounds=(0, None),
                domain=NonNegativeReals,
                units=pyunits.dimensionless,
                doc="surface-to-pore diffusion flux ratio",
            )

            @self.Constraint(
                self.flowsheet().config.time,
                self.target_species,
                doc="surface diffusion parameter",
            )
            def eq_surface_diffusion_coefficient_calculated(b, t, j):
                return (
                    b.ds * b.tort * b.equil_conc * b.particle_dens_app
                    == b.spdfr
                    * b.process_flow.properties_in[t].diffus_phase_comp["Liq", j]
                    * b.particle_porosity
                    * b.process_flow.properties_in[t].conc_mass_phase_comp["Liq", j]
                )

    # ---------------------------------------------------------------------
    # initialize method
    def initialize_build(
        self,
        state_args=None,
        outlvl=idaeslog.NOTSET,
        solver=None,
        optarg=None,
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
        init_log = idaeslog.getInitLogger(self.name, outlvl, tag="unit")
        solve_log = idaeslog.getSolveLogger(self.name, outlvl, tag="unit")
        # set solver options
        opt = get_solver(solver, optarg)

        # ---------------------------------------------------------------------
        # set state_args from inlet state

        if state_args is None:
            state_args = {}
            state_dict = self.process_flow.properties_in[
                self.flowsheet().config.time.first()
            ].define_port_members()

            for k in state_dict.keys():
                if state_dict[k].is_indexed():
                    state_args[k] = {}
                    for m in state_dict[k].keys():
                        state_args[k][m] = state_dict[k][m].value
                else:
                    state_args[k] = state_dict[k].value

        # initialize control volume
        flags = self.process_flow.initialize(
            outlvl=outlvl,
            optarg=optarg,
            solver=solver,
            state_args=state_args,
        )

        init_log.info_high("Initialization Step 1 Complete.")
        # ---------------------------------------------------------------------
        # initialize gac_removed port

        # all inert species initialized to 0
        for j in self.inert_species:
            if self.flow_basis == MaterialFlowBasis.molar:
                state_args["flow_mol_phase_comp"][("Liq", j)] = 0
            else:
                state_args["flow_mass_phase_comp"][("Liq", j)] = 0

        self.gac_removed.initialize(
            outlvl=outlvl,
            optarg=optarg,
            solver=solver,
            state_args=state_args,
        )

        init_log.info_high("Initialization Step 2 Complete.")
        # --------------------------------------------------------------------
        # solve unit

        # pre-solve using interval arithmetic
        interval_initializer(self)

        with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
            res = opt.solve(self, tee=slc.tee)

        init_log.info_high("Initialization Step 3 {}.".format(idaeslog.condition(res)))
        # ---------------------------------------------------------------------
        # release inlet state

        self.process_flow.release_state(flags, outlvl + 1)
        init_log.info("Initialization Complete: {}".format(idaeslog.condition(res)))

        if not check_optimal_termination(res):
            raise InitializationError(f"Unit model {self.name} failed to initialize")

    # ---------------------------------------------------------------------

    def _get_performance_contents(self, time_point=0):

        var_dict = {}

        # unit model variables
        var_dict["Freundlich isotherm k parameter"] = self.freund_k
        var_dict["Freundlich isotherm 1/n parameter"] = self.freund_ninv
        var_dict["Surface diffusion coefficient"] = self.ds
        var_dict["liquid phase film transfer coefficient"] = self.kf
        var_dict["superficial velocity"] = self.velocity_sup
        var_dict["bed void fraction"] = self.bed_voidage
        var_dict["bed length"] = self.bed_length
        var_dict["bed diameter"] = self.bed_diameter
        var_dict["bed volume"] = self.bed_volume
        var_dict["empty bed contact time"] = self.ebct
        var_dict["fluid residence time in the bed"] = self.residence_time
        var_dict["mass of fresh gac in the bed"] = self.bed_mass_gac
        var_dict["concentration ratio at operational time"] = self.conc_ratio_replace
        var_dict["time for the start of breakthrough"] = self.ele_operational_time[1]
        var_dict["operational time of the bed from fresh"] = self.operational_time
        var_dict["bed volumes treated"] = self.bed_volumes_treated
        var_dict["steady state average concentration ratio"] = self.conc_ratio_avg
        var_dict["gac usage/replacement/regeneration rate"] = self.gac_usage_rate

        # loop through desired state block properties indexed by [phase, comp]
        phase_comp_prop_dict = {
            "flow_mol_phase_comp": "molar flow rate",
            "conc_mass_phase_comp": "mass concentration",
        }
        for prop_name, prop_str in phase_comp_prop_dict.items():
            for j in self.config.property_package.component_list:
                if self.process_flow.properties_in[time_point].is_property_constructed(
                    prop_name
                ):
                    var_dict[f"{prop_str} of {j} @ process inlet"] = getattr(
                        self.process_flow.properties_in[time_point], prop_name
                    )["Liq", j]
                if self.process_flow.properties_out[time_point].is_property_constructed(
                    prop_name
                ):
                    var_dict[f"{prop_str} of {j} @ process outlet"] = getattr(
                        self.process_flow.properties_out[time_point], prop_name
                    )["Liq", j]
                if self.gac_removed[time_point].is_property_constructed(prop_name):
                    var_dict[f"{prop_str} of {j} @ gac removed outlet"] = getattr(
                        self.gac_removed[time_point], prop_name
                    )["Liq", j]

        # loop through desired state block properties indexed by [phase]
        phase_prop_dict = {
            "flow_vol_phase": "volumetric flow rate",
        }
        for prop_name, prop_str in phase_prop_dict.items():
            if self.process_flow.properties_in[time_point].is_property_constructed(
                prop_name
            ):
                var_dict[f"{prop_str} @ process inlet"] = getattr(
                    self.process_flow.properties_in[time_point], prop_name
                )["Liq"]
            if self.process_flow.properties_out[time_point].is_property_constructed(
                prop_name
            ):
                var_dict[f"{prop_str} @ process outlet"] = getattr(
                    self.process_flow.properties_out[time_point], prop_name
                )["Liq"]
            if self.gac_removed[time_point].is_property_constructed(prop_name):
                var_dict[f"{prop_str} @ gac removed outlet"] = getattr(
                    self.gac_removed[time_point], prop_name
                )["Liq"]

        return {"vars": var_dict}

    # ---------------------------------------------------------------------
    def _get_stream_table_contents(self, time_point=0):
        return create_stream_table_dataframe(
            {
                "Process Inlet": self.inlet,
                "Process Outlet": self.outlet,
                "GAC Removed": self.adsorbed,
            },
            time_point=time_point,
        )

    # ---------------------------------------------------------------------
    def calculate_scaling_factors(self):

        super().calculate_scaling_factors()

        # ---------------------------------------------------------------------
        # get scaling factors for target species and water to use for other variables

        for j in self.target_species:
            sf_solute = iscale.get_scaling_factor(
                self.process_flow.properties_in[0].flow_mol_phase_comp["Liq", j],
                default=1e4,  # default based on typical concentration for treatment
                warning=True,
            )

        for j in self.config.property_package.solvent_set:
            sf_solvent = iscale.get_scaling_factor(
                self.process_flow.properties_in[0].flow_mol_phase_comp["Liq", j],
                default=1e-3,  # default based on typical concentration for treatment
                warning=True,
            )

        # sf_volume based on magnitude of 0.018 water mw and 1000 dens
        sf_volume = sf_solvent * (100 / 0.01)
        sf_conc = sf_solute / sf_volume

        # ---------------------------------------------------------------------
        # overwrite default scaling for state block variables

        for j in self.target_species:
            iscale.set_scaling_factor(
                self.process_flow.properties_out[0].flow_mol_phase_comp["Liq", j],
                10 * sf_solute,
            )

        for j in self.inert_species:
            iscale.set_scaling_factor(
                self.gac_removed[0].flow_mol_phase_comp["Liq", j],
                1,
            )  # ensure lower concentration of zero flow components, below zero tol

        # dens stays at 1000 even though water flow is zero, using sf based on 0.1 assumed mw and 1000 dens
        if self.gac_removed[0].is_property_constructed("flow_vol_phase"):
            iscale.set_scaling_factor(
                self.gac_removed[0].flow_vol_phase["Liq"],
                sf_solute * (1000 / 0.1),
            )

        # ---------------------------------------------------------------------
        # scaling for gac created variables that are flow magnitude dependent

        if iscale.get_scaling_factor(self.equil_conc) is None:
            iscale.set_scaling_factor(self.equil_conc, sf_conc * 1e-2)

        if iscale.get_scaling_factor(self.bed_diameter) is None:
            iscale.set_scaling_factor(self.bed_diameter, sf_volume * 1e-2)

        if iscale.get_scaling_factor(self.bed_area) is None:
            iscale.set_scaling_factor(self.bed_area, sf_volume * 1e-2)

        if iscale.get_scaling_factor(self.bed_volume) is None:
            iscale.set_scaling_factor(self.bed_volume, sf_volume * 1e-2)

        if iscale.get_scaling_factor(self.bed_mass_gac) is None:
            iscale.set_scaling_factor(self.bed_mass_gac, sf_volume * 1e-6)

        if iscale.get_scaling_factor(self.mass_adsorbed) is None:
            iscale.set_scaling_factor(self.mass_adsorbed, sf_solute * 1e-6)

        if iscale.get_scaling_factor(self.gac_usage_rate) is None:
            iscale.set_scaling_factor(self.gac_usage_rate, sf_volume)

        # ---------------------------------------------------------------------
        # scaling for gac created variables that are flow magnitude independent

        if iscale.get_scaling_factor(self.freund_k) is None:
            iscale.set_scaling_factor(self.freund_k, 1)

        if iscale.get_scaling_factor(self.freund_ninv) is None:
            iscale.set_scaling_factor(self.freund_ninv, 1)

        if iscale.get_scaling_factor(self.ds) is None:
            iscale.set_scaling_factor(self.ds, 1e14)

        if iscale.get_scaling_factor(self.kf) is None:
            iscale.set_scaling_factor(self.kf, 1e5)

        if iscale.get_scaling_factor(self.dg) is None:
            iscale.set_scaling_factor(self.dg, 1e-4)

        if iscale.get_scaling_factor(self.N_Bi) is None:
            iscale.set_scaling_factor(self.N_Bi, 1)

        if iscale.get_scaling_factor(self.velocity_sup) is None:
            iscale.set_scaling_factor(self.velocity_sup, 1e3)

        if iscale.get_scaling_factor(self.velocity_int) is None:
            iscale.set_scaling_factor(self.velocity_int, 1e3)

        if iscale.get_scaling_factor(self.bed_voidage) is None:
            iscale.set_scaling_factor(self.bed_voidage, 1e1)

        if iscale.get_scaling_factor(self.bed_length) is None:
            iscale.set_scaling_factor(self.bed_length, 1)

        if iscale.get_scaling_factor(self.ebct) is None:
            iscale.set_scaling_factor(self.ebct, 1e-2)

        if iscale.get_scaling_factor(self.residence_time) is None:
            iscale.set_scaling_factor(self.residence_time, 1e-2)

        if iscale.get_scaling_factor(self.particle_dens_app) is None:
            iscale.set_scaling_factor(self.particle_dens_app, 1e-2)

        if iscale.get_scaling_factor(self.particle_dens_bulk) is None:
            iscale.set_scaling_factor(self.particle_dens_bulk, 1e-2)

        if iscale.get_scaling_factor(self.particle_dia) is None:
            iscale.set_scaling_factor(self.particle_dia, 1e3)

        iscale.set_scaling_factor(self.a0, 1)
        iscale.set_scaling_factor(self.a1, 1)
        iscale.set_scaling_factor(self.b0, 1)
        iscale.set_scaling_factor(self.b1, 1)
        iscale.set_scaling_factor(self.b2, 1)
        iscale.set_scaling_factor(self.b3, 10)
        iscale.set_scaling_factor(self.b4, 1)

        if iscale.get_scaling_factor(self.min_N_St) is None:
            iscale.set_scaling_factor(self.min_N_St, 1e-1)

        if iscale.get_scaling_factor(self.min_ebct) is None:
            iscale.set_scaling_factor(self.min_ebct, 1e-2)

        if iscale.get_scaling_factor(self.throughput) is None:
            iscale.set_scaling_factor(self.throughput, 1)

        if iscale.get_scaling_factor(self.min_residence_time) is None:
            iscale.set_scaling_factor(self.min_residence_time, 1e-2)

        if iscale.get_scaling_factor(self.min_operational_time) is None:
            iscale.set_scaling_factor(self.min_operational_time, 1e-6)

        if iscale.get_scaling_factor(self.conc_ratio_replace) is None:
            iscale.set_scaling_factor(self.conc_ratio_replace, 1e1)

        if iscale.get_scaling_factor(self.operational_time) is None:
            iscale.set_scaling_factor(self.operational_time, 1e-6)

        if iscale.get_scaling_factor(self.bed_volumes_treated) is None:
            iscale.set_scaling_factor(self.bed_volumes_treated, 1e-5)

        for ele in range(1, self.elements_ss_approx.value + 1):

            if iscale.get_scaling_factor(self.ele_throughput[ele]) is None:
                iscale.set_scaling_factor(self.ele_throughput[ele], 1)

            if iscale.get_scaling_factor(self.ele_min_operational_time[ele]) is None:
                iscale.set_scaling_factor(self.ele_min_operational_time[ele], 1e-6)

            if iscale.get_scaling_factor(self.ele_conc_ratio_term[ele]) is None:
                iscale.set_scaling_factor(self.ele_conc_ratio_term[ele], 1e2)

        for ele in range(self.elements_ss_approx.value + 1):

            if iscale.get_scaling_factor(self.ele_conc_ratio_replace[ele]) is None:
                iscale.set_scaling_factor(self.ele_conc_ratio_replace[ele], 1e1)

            if iscale.get_scaling_factor(self.ele_operational_time[ele]) is None:
                iscale.set_scaling_factor(self.ele_operational_time[ele], 1e-6)

        if iscale.get_scaling_factor(self.conc_ratio_avg) is None:
            iscale.set_scaling_factor(self.conc_ratio_avg, 1e1)

        if hasattr(self, "N_Re"):
            if iscale.get_scaling_factor(self.N_Re) is None:
                iscale.set_scaling_factor(self.N_Re, 1)

        if hasattr(self, "N_Sc"):
            if iscale.get_scaling_factor(self.N_Sc) is None:
                iscale.set_scaling_factor(self.N_Sc, 1e-3)

        if hasattr(self, "shape_correction_factor"):
            if iscale.get_scaling_factor(self.shape_correction_factor) is None:
                iscale.set_scaling_factor(self.shape_correction_factor, 1)

        if hasattr(self, "particle_porosity"):
            if iscale.get_scaling_factor(self.particle_porosity) is None:
                iscale.set_scaling_factor(self.particle_porosity, 1e1)

        if hasattr(self, "tort"):
            if iscale.get_scaling_factor(self.tort) is None:
                iscale.set_scaling_factor(self.tort, 1)

        if hasattr(self, "spdfr"):
            if iscale.get_scaling_factor(self.spdfr) is None:
                iscale.set_scaling_factor(self.spdfr, 1)

        # (optional) transforming constraints

    @property
    def default_costing_method(self):
        return cost_gac
