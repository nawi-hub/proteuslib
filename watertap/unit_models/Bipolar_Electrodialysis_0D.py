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
import math

# Import Pyomo libraries
from pyomo.environ import (
    Set,
    Var,
    check_optimal_termination,
    Param,
    Suffix,
    NonNegativeReals,
    value,
    Constraint,
    sqrt,
    units as pyunits,
)
from pyomo.common.config import Bool, ConfigBlock, ConfigValue, In

# Import IDAES cores
from idaes.core import (
    declare_process_block_class,
    MaterialBalanceType,
    EnergyBalanceType,
    MomentumBalanceType,
    UnitModelBlockData,
    useDefault,
)
from idaes.core.util.misc import add_object_reference
from watertap.core.solvers import get_solver
from idaes.core.util.tables import create_stream_table_dataframe
from idaes.core.util.config import is_physical_parameter_block
from idaes.core.util.math import smooth_min

from idaes.core.util.exceptions import ConfigurationError, InitializationError

import idaes.core.util.scaling as iscale
import idaes.logger as idaeslog
from idaes.core.util.constants import Constants
from enum import Enum

from watertap.core import ControlVolume0DBlock, InitializationMixin
from watertap.costing.unit_models.bipolar_electrodialysis import (
    cost_bipolar_electrodialysis,
)

__author__ = " Johnson Dhanasekaran, Xiangyu Bi, Austin Ladshaw, Kejia Hu"

_log = idaeslog.getLogger(__name__)


class LimitingCurrentDensitybpemMethod(Enum):
    InitialValue = 0
    Empirical = 1


class LimitingpotentialMethod(Enum):
    InitialValue = 0
    Empirical = 1


class ElectricalOperationMode(Enum):
    Constant_Current = 0
    Constant_Voltage = 1


class PressureDropMethod(Enum):
    none = 0
    experimental = 1
    Darcy_Weisbach = 2


class FrictionFactorMethod(Enum):
    fixed = 0
    Gurreri = 1
    Kuroda = 2


class HydraulicDiameterMethod(Enum):
    fixed = 0
    spacer_specific_area_known = 1
    conventional = 2


# Name of the unit model
@declare_process_block_class("Bipolar_Electrodialysis_0D")
class BipolarElectrodialysis0DData(InitializationMixin, UnitModelBlockData):
    """
    0D Bipolar and Electrodialysis Model
    """

    # CONFIG are options for the unit model
    CONFIG = ConfigBlock()  #

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
        "has_pressure_change",
        ConfigValue(
            default=False,
            domain=In([True, False]),
            description="Pressure change term construction flag",
            doc="""Indicates whether terms for pressure change should be
    constructed,
    **default** - False.
    **Valid values:** {
    **True** - include pressure change terms,
    **False** - exclude pressure change terms.}""",
        ),
    )
    CONFIG.declare(
        "pressure_drop_method",
        ConfigValue(
            default=PressureDropMethod.none,
            domain=In(PressureDropMethod),
            description="Method to calculate the frictional pressure drop in electrodialysis channels",
            doc="""
     **default** - ``PressureDropMethod.none``

       .. csv-table::
           :header: "Configuration Options", "Description"

           "``PressureDropMethod.none``", "The frictional pressure drop is neglected." 
           "``PressureDropMethod.experimental``", "The pressure drop is calculated by an experimental data as pressure drop per unit lenght."
           "``PressureDropMethod.Darcy_Weisbach``", "The pressure drop is calculated by the Darcy-Weisbach equation."
       """,
        ),
    )
    CONFIG.declare(
        "friction_factor_method",
        ConfigValue(
            default=FrictionFactorMethod.fixed,
            domain=In(FrictionFactorMethod),
            description="Method to calculate the Darcy's friction factor",
            doc="""
     **default** - ``FrictionFactorMethod.fixed``

       .. csv-table::
           :header: "Configuration Options", "Description"

           "``FrictionFactorMethod.fixed``", "Friction factor is fixed by users" 
           "``FrictionFactorMethod.Gurreri``", "Friction factor evaluated based on Gurreri's work"
           "``FrictionFactorMethod.Kuroda``", "Friction factor evaluated based on Kuroda's work"
       """,
        ),
    )

    CONFIG.declare(
        "hydraulic_diameter_method",
        ConfigValue(
            default=HydraulicDiameterMethod.conventional,
            domain=In(HydraulicDiameterMethod),
            description="Method to calculate the hydraulic diameter for a rectangular channel in ED",
            doc="""
     **default** - ``HydraulicDiameterMethod.conventional``

       .. csv-table::
           :header: "Configuration Options", "Description"

           "``HydraulicDiameterMethod.fixed``", "Hydraulic diameter is fixed by users" 
           "``HydraulicDiameterMethod.conventional``", "Conventional method for a rectangular channel with spacer porosity considered" 
           "``HydraulicDiameterMethod.spacer_specific_area_known``", "A method for spacer-filled channel requiring the spacer specific area data"
       """,
        ),
    )

    CONFIG.declare(
        "limiting_current_density_method_bpem",
        ConfigValue(
            default=LimitingCurrentDensitybpemMethod.InitialValue,
            domain=In(LimitingCurrentDensitybpemMethod),
            description="Configuration for method to compute the limiting current density across the bipolar membrane",
            doc="""
               **default** - ``LimitingCurrentDensitybpemMethod.InitialValue``

           .. csv-table::
               :header: "Configuration Options", "Description"

               "``LimitingCurrentDensitybpemMethod.InitialValue``", "Limiting current is calculated from a single initial value given by the user."
               "``LimitingCurrentDensitybpemMethod.Empirical``", "Limiting current density is calculated from the empirical relationship"
           """,
        ),
    )

    CONFIG.declare(
        "has_catalyst",
        ConfigValue(
            default=False,
            domain=Bool,
            description="""Catalyst action on water spliting,
            **default** - False.""",
        ),
    )

    CONFIG.declare(
        "limiting_potential_method_bpem",
        ConfigValue(
            default=LimitingpotentialMethod.InitialValue,
            domain=In(LimitingpotentialMethod),
            description="Configuration for method to compute the limiting potential in bipolar membrane",
            doc="""
                   **default** - ``LimitingpotentialMethod.InitialValue``

               .. csv-table::
                   :header: "Configuration Options", "Description"

                   "``LimitingpotentialMethod.InitialValue``", "Limiting current is calculated from a initial value given by the user."
                   "``LimitingpotentialMethod.Empirical``", "Limiting current density is caculated from the empirical equation"
               """,
        ),
    )

    CONFIG.declare(
        "limiting_current_density_bpem_data",
        ConfigValue(
            default=0.5,
            description="Limiting current density data input for bipolar membrane",
        ),
    )

    CONFIG.declare(
        "limiting_potential_data",
        ConfigValue(
            default=0.5,
            description="Limiting potential of the bipolar membrane input",
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

    def _validate_config(self):
        if (
            self.config.is_isothermal
            and self.config.energy_balance_type != EnergyBalanceType.none
        ):
            raise ConfigurationError(
                "If the isothermal assumption is used then the energy balance type must be none"
            )

    def build(self):
        # build always starts by calling super().build()
        # This triggers a lot of boilerplate in the background for you
        super().build()
        # this creates blank scaling factors, which are populated later
        self.scaling_factor = Suffix(direction=Suffix.EXPORT)

        # Check configs for errors
        self._validate_config()

        # Create essential sets.
        self.membrane_set = Set(initialize=["bpem"])

        add_object_reference(self, "ion_set", self.config.property_package.ion_set)

        add_object_reference(
            self, "cation_set", self.config.property_package.cation_set
        )
        add_object_reference(self, "anion_set", self.config.property_package.anion_set)
        # Create unit model parameters and vars
        self.water_density = Param(
            initialize=1000,
            units=pyunits.kg * pyunits.m**-3,
            doc="density of water",
        )

        self.cell_num = Var(
            initialize=1,
            domain=NonNegativeReals,
            bounds=(1, 10000),
            units=pyunits.dimensionless,
            doc="cell pair number in a stack",
        )

        # electrodialysis cell dimensional properties
        self.cell_width = Var(
            initialize=0.1,
            bounds=(1e-3, 1e3),
            units=pyunits.meter,
            doc="The width of the electrodialysis cell, denoted as b in the model description",
        )
        self.cell_length = Var(
            initialize=0.5,
            bounds=(1e-3, 1e2),
            units=pyunits.meter,
            doc="The length of the electrodialysis cell, denoted as l in the model description",
        )
        self.channel_height = Var(
            initialize=0.0001,
            units=pyunits.meter,
            doc="The distance between the consecutive aem and cem",
        )
        self.spacer_porosity = Var(
            initialize=0.7,
            bounds=(0.01, 1),
            units=pyunits.dimensionless,
            doc='The prosity of spacer in the ED channels. This is also referred to elsewhere as "void fraction" or "volume parameters"',
        )

        # Material and Operational properties
        self.membrane_thickness = Var(
            self.membrane_set,
            initialize=0.0001,
            bounds=(1e-6, 1e-1),
            units=pyunits.meter,
            doc="Membrane thickness",
        )
        self.solute_diffusivity_membrane = Var(
            self.membrane_set,
            self.ion_set | self.config.property_package.solute_set,
            initialize=1e-10,
            bounds=(0.0, 1e-6),
            units=pyunits.meter**2 * pyunits.second**-1,
            doc="Solute (ionic and neutral) diffusivity in the membrane phase",
        )
        self.ion_trans_number_membrane = Var(
            self.membrane_set,
            self.ion_set,
            bounds=(0, 1),
            units=pyunits.dimensionless,
            doc="Ion transference number in the membrane phase",
        )
        self.water_trans_number_membrane = Var(
            self.membrane_set,
            initialize=5,
            bounds=(0, 50),
            units=pyunits.dimensionless,
            doc="Transference number of water in membranes",
        )
        self.water_permeability_membrane = Var(
            self.membrane_set,
            initialize=1e-14,
            units=pyunits.meter * pyunits.second**-1 * pyunits.pascal**-1,
            doc="Water permeability coefficient",
        )
        self.membrane_areal_resistance = Var(
            initialize=2e-4,
            bounds=(0, 1),
            units=pyunits.ohm * pyunits.meter**2,
            doc="Surface resistance of membrane",
        )
        self.electrodes_resistance = Var(
            initialize=0,
            bounds=(0, 100),
            domain=NonNegativeReals,
            units=pyunits.ohm * pyunits.meter**2,
            doc="areal resistance of TWO electrode compartments of a stack",
        )
        self.current = Var(
            self.flowsheet().time,
            initialize=1,
            bounds=(0, 1e6),
            units=pyunits.amp,
            doc="Current across a cell-pair or stack",
        )
        self.voltage = Var(
            self.flowsheet().time,
            initialize=100,
            bounds=(0, 1000),
            units=pyunits.volt,
            doc="Voltage across a stack, declared under the 'Constant Voltage' mode only",
        )
        self.current_utilization = Var(
            initialize=1,
            bounds=(0, 1),
            units=pyunits.dimensionless,
            doc="The current utilization including water electro-osmosis and ion diffusion",
        )
        self.shadow_factor = Var(
            initialize=1,
            bounds=(0, 1),
            units=pyunits.dimensionless,
            doc="The reduction in area due to limited cross-section available for flow",
        )

        # Performance metrics
        self.power_electrical = Var(
            self.flowsheet().time,
            initialize=1,
            bounds=(0, 12100),
            domain=NonNegativeReals,
            units=pyunits.watt,
            doc="Electrical power consumption of a stack",
        )
        self.specific_power_electrical = Var(
            self.flowsheet().time,
            initialize=10,
            bounds=(0, 1000),
            domain=NonNegativeReals,
            units=pyunits.kW * pyunits.hour * pyunits.meter**-3,
            doc="acidic-volume-flow-rate-specific electrical power consumption",
        )
        self.acid_produced = Var(
            initialize=55 * 1e3,
            bounds=(0, 1e6),
            units=pyunits.kg * pyunits.second**-1,
            doc="Acid prodcued",
        )
        self.base_produced = Var(
            initialize=55 * 1e3,
            bounds=(0, 1e6),
            units=pyunits.kg * pyunits.second**-1,
            doc="Base prodcued",
        )
        self.velocity_basic = Var(
            self.flowsheet().time,
            initialize=0.01,
            units=pyunits.meter * pyunits.second**-1,
            doc="Linear velocity of flow in the base channel of the bipolar membrane",
        )
        self.velocity_acidic = Var(
            self.flowsheet().time,
            initialize=0.01,
            units=pyunits.meter * pyunits.second**-1,
            doc="Linear velocity of flow in the acid channel of the bipolar membrane",
        )
        # Parameters for bipolar membrane operation
        self.elec_field_non_dim = Var(
            self.flowsheet().time,
            initialize=1,
            # bounds=(0, 1e32),
            units=pyunits.dimensionless,
            doc="Limiting current density across the bipolar membrane as a function of the normalized length",
        )
        self.relative_permittivity = Var(
            self.membrane_set,
            initialize=30,
            bounds=(1, 80),
            domain=NonNegativeReals,
            units=pyunits.dimensionless,
            doc="Relative permittivity",
        )
        self.membrane_fixed_charge = Var(
            self.membrane_set,
            initialize=1.5e3,
            bounds=(1e-1, 1e5),
            units=pyunits.mole * pyunits.meter**-3,
            doc="Membrane fixed charge",
        )
        self.kr = Var(
            self.membrane_set,
            initialize=1.33 * 10**11,
            bounds=(1e-6, 1e16),
            units=pyunits.L * pyunits.mole**-1 * pyunits.second**-1,
            doc="Re-association rate constant",
        )
        self.k2_zero = Var(
            self.membrane_set,
            initialize=2 * 10**-5,
            bounds=(1e-10, 1e2),
            units=pyunits.second**-1,
            doc="Dissociation rate constant at no electric field",
        )
        self.salt_conc_aem = Var(
            self.membrane_set,
            initialize=1e3,
            bounds=(1e-8, 1e6),
            units=pyunits.mole * pyunits.meter**-3,
            doc="Salt concentration on the base channel of the bipolar membrane",
        )
        self.salt_conc_cem = Var(
            self.membrane_set,
            initialize=1e3,
            bounds=(1e-6, 1e4),
            units=pyunits.mole * pyunits.meter**-3,
            doc="Salt concentration on the acid channel of the bipolar membrane",
        )
        self.diffus_mass = Var(
            # self.membrane_set,
            initialize=2e-9,
            bounds=(1e-16, 1e-6),
            units=pyunits.meter**2 * pyunits.second**-1,
            doc="The mass diffusivity of the solute as molecules (not individual ions)",
        )
        self.conc_water = Var(
            self.membrane_set,
            initialize=55 * 1e3,
            bounds=(1e-2, 1e6),
            units=pyunits.mole * pyunits.meter**-3,
            doc="Concentration of water within the channel",
        )

        # Limiting operation quantity
        self.current_dens_lim_bpem = Var(
            self.flowsheet().time,
            initialize=1e2,
            bounds=(0, 1e5),
            units=pyunits.amp * pyunits.meter**-2,
            doc="Limiting current density across the bipolar membrane",
        )

        # Fluxes Vars for constructing mass transfer terms
        self.generation_cem_flux_in = Var(
            self.flowsheet().time,
            self.config.property_package.phase_list,
            self.config.property_package.component_list,
            units=pyunits.mole * pyunits.meter**-2 * pyunits.second**-1,
            doc="Molar flux_in of a component generated by water splitting on the acid channel of the bipolar membrane",
        )
        self.generation_cem_flux_out = Var(
            self.flowsheet().time,
            self.config.property_package.phase_list,
            self.config.property_package.component_list,
            units=pyunits.mole * pyunits.meter**-2 * pyunits.second**-1,
            doc="Molar flux_in of a component generated by water splitting on the acid channel of the bipolar membrane",
        )
        self.generation_aem_flux_in = Var(
            self.flowsheet().time,
            self.config.property_package.phase_list,
            self.config.property_package.component_list,
            units=pyunits.mole * pyunits.meter**-2 * pyunits.second**-1,
            doc="Molar flux_in of a component generated by water splitting on the base channel of the bipolar membrane",
        )
        self.generation_aem_flux_out = Var(
            self.flowsheet().time,
            self.config.property_package.phase_list,
            self.config.property_package.component_list,
            units=pyunits.mole * pyunits.meter**-2 * pyunits.second**-1,
            doc="Molar flux_in of a component generated by water splitting on the base channel of the bipolar membrane",
        )
        self.elec_migration_bpem_flux_in = Var(
            self.flowsheet().time,
            self.config.property_package.phase_list,
            self.config.property_package.component_list,
            units=pyunits.mole * pyunits.meter**-2 * pyunits.second**-1,
            doc="Molar flux_in of a component across the membrane driven by electrical migration across the bipolar membrane",
        )
        self.elec_migration_bpem_flux_out = Var(
            self.flowsheet().time,
            self.config.property_package.phase_list,
            self.config.property_package.component_list,
            units=pyunits.mole * pyunits.meter**-2 * pyunits.second**-1,
            doc="Molar flux_out of a component across the membrane driven by electrical migration across the bipolar membrane",
        )
        self.nonelec_bpem_flux_in = Var(
            self.flowsheet().time,
            self.config.property_package.phase_list,
            self.config.property_package.component_list,
            units=pyunits.mole * pyunits.meter**-2 * pyunits.second**-1,
            doc="Molar flux_in of a component across the membrane driven by non-electrical forces across the bipolar membrane",
        )
        self.nonelec_bpem_flux_out = Var(
            self.flowsheet().time,
            self.config.property_package.phase_list,
            self.config.property_package.component_list,
            units=pyunits.mole * pyunits.meter**-2 * pyunits.second**-1,
            doc="Molar flux_out of a component across the membrane driven by non-electrical forces across the bipolar membrane",
        )

        # Build control volume for the base channel of the bipolar channel
        self.basic = ControlVolume0DBlock(
            dynamic=False,
            has_holdup=False,
            property_package=self.config.property_package,
            property_package_args=self.config.property_package_args,
        )
        self.basic.add_state_blocks(has_phase_equilibrium=False)
        self.basic.add_material_balances(
            balance_type=self.config.material_balance_type, has_mass_transfer=True
        )
        self.basic.add_energy_balances(
            balance_type=self.config.energy_balance_type,
            has_enthalpy_transfer=False,
        )

        if self.config.is_isothermal:
            self.basic.add_isothermal_assumption()
        self.basic.add_momentum_balances(
            balance_type=self.config.momentum_balance_type,
            has_pressure_change=self.config.has_pressure_change,
        )
        # Build control volume for the acid channel of the bipolar membrane channel
        self.acidic = ControlVolume0DBlock(
            dynamic=False,
            has_holdup=False,
            property_package=self.config.property_package,
            property_package_args=self.config.property_package_args,
        )
        self.acidic.add_state_blocks(has_phase_equilibrium=False)
        self.acidic.add_material_balances(
            balance_type=self.config.material_balance_type, has_mass_transfer=True
        )
        self.acidic.add_energy_balances(
            balance_type=self.config.energy_balance_type,
            has_enthalpy_transfer=False,
        )

        if self.config.is_isothermal:
            self.acidic.add_isothermal_assumption()
        self.acidic.add_momentum_balances(
            balance_type=self.config.momentum_balance_type,
            has_pressure_change=self.config.has_pressure_change,
        )

        # den_mass and visc_d in acidic and basic channels are the same
        add_object_reference(
            self, "dens_mass", self.acidic.properties_in[0].dens_mass_phase["Liq"]
        )
        add_object_reference(
            self, "visc_d", self.acidic.properties_in[0].visc_d_phase["Liq"]
        )

        # Add ports (creates inlets and outlets for each channel)
        self.add_inlet_port(name="inlet_basic", block=self.basic)
        self.add_outlet_port(name="outlet_basic", block=self.basic)
        self.add_inlet_port(name="inlet_acidic", block=self.acidic)
        self.add_outlet_port(name="outlet_acidic", block=self.acidic)

        # extension options
        if self.config.has_catalyst == True:
            self._make_catalyst()

        if (
            not self.config.pressure_drop_method == PressureDropMethod.none
        ) and self.config.has_pressure_change:
            self._pressure_drop_calculation()

            @self.Constraint(
                self.flowsheet().time,
                doc="Pressure drop expression as calculated by the pressure drop data, "
                "base channel of the bipolar membrane.",
            )
            def eq_deltaP_basic(self, t):
                return self.basic.deltaP[t] == -self.pressure_drop_total[t]

            @self.Constraint(
                self.flowsheet().time,
                doc="Pressure drop expression as calculated by the pressure drop data,"
                "  acid channel of the bipolar membrane.",
            )
            def eq_deltaP_acidic(self, t):
                return self.acidic.deltaP[t] == -self.pressure_drop_total[t]

        elif self.config.pressure_drop_method == PressureDropMethod.none and (
            not self.config.has_pressure_change
        ):
            pass
        else:
            raise ConfigurationError(
                "A valid (not none) pressure_drop_method and has_pressure_change being True "
                "must be both used or unused at the same time. "
            )

        # Build Constraints

        @self.Constraint(
            self.flowsheet().time,
            doc="Calculate flow velocity in a single base (A.E.M side) channel of the bipolar membrane channel,"
            " based on the average of inlet and outlet",
        )
        def eq_get_velocity_basic(self, t):
            return self.velocity_basic[
                t
            ] * self.cell_width * self.shadow_factor * self.channel_height * self.spacer_porosity * self.cell_num == 0.5 * (
                self.basic.properties_in[0].flow_vol_phase["Liq"]
                + self.basic.properties_out[0].flow_vol_phase["Liq"]
            )

        @self.Constraint(
            self.flowsheet().time,
            doc="Calculate flow velocity in a single acid (C.E.M side) channel of the bipolar membrane channel,"
            " based on the average of inlet and outlet",
        )
        def eq_get_velocity_acidic(self, t):
            return self.velocity_acidic[
                t
            ] * self.cell_width * self.shadow_factor * self.channel_height * self.spacer_porosity * self.cell_num == 0.5 * (
                self.acidic.properties_in[0].flow_vol_phase["Liq"]
                + self.acidic.properties_out[0].flow_vol_phase["Liq"]
            )

        @self.Constraint(
            self.flowsheet().time,
            doc="Calculate limiting current density across the bipolar membrane",
        )
        def eq_current_dens_lim_bpem(self, t):
            if (
                self.config.limiting_current_density_method_bpem
                == LimitingCurrentDensitybpemMethod.InitialValue
            ):
                return self.current_dens_lim_bpem[t] == (
                    self.config.limiting_current_density_bpem_data
                    * pyunits.amp
                    * pyunits.meter**-2
                )
            elif (
                self.config.limiting_current_density_method_bpem
                == LimitingCurrentDensitybpemMethod.Empirical
            ):
                return self.current_dens_lim_bpem[
                    t
                ] == self.diffus_mass * Constants.faraday_constant * (
                    (self.salt_conc_aem["bpem"] + self.salt_conc_cem["bpem"]) * 0.5
                ) ** 2 / (
                    self.membrane_thickness["bpem"] * self.membrane_fixed_charge["bpem"]
                )

        @self.Constraint(
            self.flowsheet().time,
            doc="Calculate the potential drops across the bipolar membrane",
        )
        def eq_potential_barrier_bpem(self, t):
            if self.config.has_catalyst:
                return Constraint.Skip
            else:
                self.potential_barrier_bpem = Var(
                    self.flowsheet().time,
                    initialize=1,
                    bounds=(0, 5000),
                    units=pyunits.volt,
                    doc="Potential barrier across the depletion layer for water splitting to begin",
                )

                if (
                    self.config.limiting_potential_method_bpem
                    == LimitingpotentialMethod.InitialValue
                ):
                    return self.potential_barrier_bpem[t] == (
                        self.config.limiting_potential_data * pyunits.volt
                    )

                elif (
                    self.config.limiting_potential_method_bpem
                    == LimitingpotentialMethod.Empirical
                ):
                    #   [H+][OH-] concentration
                    kw = 10**-8 * pyunits.mol**2 * pyunits.meter**-6

                    # Fraction of threshold of limiting current: currently 0.1 i_lim
                    frac = 1 * 10**-1
                    # Dimensional pre-factor to evaulate non-dimensional electric field
                    const = 0.0936 * pyunits.K**2 * pyunits.volt**-1 * pyunits.meter

                    @self.Constraint(
                        self.flowsheet().time,
                        doc="Calculate the non-dimensional potential drop",
                    )
                    def eq_potential_barrier_bpem_non_dim(self, t):
                        # [y2, qty_une, qty_deux, qty_trois] = dat
                        terms = 40
                        matrx = 0
                        for indx in range(terms):
                            # rev_indx = terms - indx - 1
                            matrx += (
                                2**indx
                                * self.elec_field_non_dim[t] ** indx
                                / (math.factorial(indx) * math.factorial(indx + 1))
                            )

                        matrx *= self.k2_zero["bpem"] * self.conc_water["bpem"]
                        matrx += (
                            -pyunits.convert(
                                self.kr["bpem"],
                                to_units=pyunits.meter**3
                                * pyunits.mole**-1
                                * pyunits.second**-1,
                            )
                            * kw
                        )
                        return (
                            Constants.vacuum_electric_permittivity
                            * self.relative_permittivity["bpem"] ** 2
                            * self.basic.properties_in[t].temperature ** 2
                            * Constants.avogadro_number
                            * Constants.elemental_charge
                        ) / (
                            const
                            * Constants.faraday_constant
                            * self.membrane_fixed_charge["bpem"]
                        ) * matrx * self.elec_field_non_dim[
                            t
                        ] == self.current_dens_lim_bpem[
                            t
                        ] * frac

                    # Dimensional electric field
                    field_generated = (
                        self.elec_field_non_dim[t]
                        * self.relative_permittivity["bpem"]
                        * self.basic.properties_in[t].temperature ** 2
                        / const
                    )

                    # Depletion length at the junction of the bipolar membrane
                    lambda_depletion = (
                        field_generated
                        * Constants.vacuum_electric_permittivity
                        * self.relative_permittivity["bpem"]
                        / (
                            Constants.faraday_constant
                            * self.membrane_fixed_charge["bpem"]
                        )
                    )

                    return (
                        self.potential_barrier_bpem[t]
                        == field_generated * lambda_depletion
                    )

                else:
                    self.potential_barrier_bpem[t].fix(0 * pyunits.volt)
                    return Constraint.Skip

        @self.Constraint(
            self.flowsheet().time,
            self.config.property_package.phase_list,
            doc="Current-Voltage relationship",
        )
        def eq_current_voltage_relation(self, t, p):

            total_areal_resistance = (
                self.membrane_areal_resistance
                + self.channel_height
                * (
                    0.5**-1
                    * (
                        self.basic.properties_in[t].elec_cond_phase["Liq"]
                        + self.basic.properties_out[t].elec_cond_phase["Liq"]
                    )
                    ** -1
                    + 0.5**-1
                    * (
                        self.acidic.properties_in[t].elec_cond_phase["Liq"]
                        + self.acidic.properties_out[t].elec_cond_phase["Liq"]
                    )
                    ** -1
                )
            ) * self.cell_num + self.electrodes_resistance
            # the average conductivity of each channel's inlet and outlet is taken to represent that of the entire channel

            if self.config.has_catalyst:
                voltage_membrane_drop = self.potential_membrane_bpem[t]

                @self.Constraint(
                    self.flowsheet().time,
                    doc="Calculate total current generated via catalyst action",
                )
                def eq_current_relationship(self, t):
                    return self.current[t] == (
                        self.current_dens_lim_bpem[t]
                        + self.flux_splitting[t] * Constants.faraday_constant
                    ) * (self.cell_width * self.shadow_factor * self.cell_length)

            else:
                voltage_membrane_drop = self.potential_barrier_bpem[t]

            return (
                self.current[t]
                * (self.cell_width * self.shadow_factor * self.cell_length) ** -1
                * total_areal_resistance
                + voltage_membrane_drop * self.cell_num
                == self.voltage[t]
            )

        @self.Constraint(
            self.flowsheet().time,
            self.config.property_package.phase_list,
            self.config.property_package.component_list,
            doc="Equation for water splitting acid channel (C.E.M Side) of bipolar membrane flux_in",
        )
        def eq_generation_cem_flux_in(self, t, p, j):
            if j == "H_+":
                if self.config.has_catalyst == True:
                    return (
                        self.generation_cem_flux_in[t, p, j] == self.flux_splitting[t]
                    )

                else:
                    return self.generation_cem_flux_in[t, p, j] == (
                        -smooth_min(
                            -(
                                self.current[t] / pyunits.amp
                                - self.current_dens_lim_bpem[t]
                                * self.cell_width
                                * self.shadow_factor
                                * self.cell_length
                                / pyunits.amp
                            ),
                            0,
                        )
                        * pyunits.amp
                        / (self.cell_width * self.shadow_factor * self.cell_length)
                    ) / (Constants.faraday_constant)

            else:
                if j == "H2O":
                    if self.config.has_catalyst == True:
                        return (
                            self.generation_cem_flux_in[t, p, j]
                            == -0.5 * self.flux_splitting[t]
                        )
                    else:
                        return self.generation_cem_flux_in[t, p, j] == (
                            0.5
                            * smooth_min(
                                -(
                                    self.current[t] / pyunits.amp
                                    - self.current_dens_lim_bpem[t]
                                    * self.cell_width
                                    * self.shadow_factor
                                    * self.cell_length
                                    / pyunits.amp
                                ),
                                0,
                            )
                            * pyunits.amp
                            / (self.cell_width * self.shadow_factor * self.cell_length)
                        ) / (Constants.faraday_constant)

                else:
                    return (
                        self.generation_cem_flux_in[t, p, j]
                        == 0 * pyunits.mol * pyunits.meter**-2 * pyunits.s**-1
                    )

        @self.Constraint(
            self.flowsheet().time,
            self.config.property_package.phase_list,
            self.config.property_package.component_list,
            doc="Equation for water splitting base channel (A.E.M Side) of bipolar membrane flux_in",
        )
        def eq_generation_aem_flux_in(self, t, p, j):
            if j == "OH_-":
                if self.config.has_catalyst == True:
                    return (
                        self.generation_aem_flux_in[t, p, j] == self.flux_splitting[t]
                    )

                else:
                    return self.generation_aem_flux_in[t, p, j] == (
                        -smooth_min(
                            -(
                                self.current[t] / pyunits.amp
                                - self.current_dens_lim_bpem[t]
                                * self.cell_width
                                * self.shadow_factor
                                * self.cell_length
                                / pyunits.amp
                            ),
                            0,
                        )
                        * pyunits.amp
                        / (self.cell_width * self.shadow_factor * self.cell_length)
                    ) / (Constants.faraday_constant)

            else:
                if j == "H2O":
                    if self.config.has_catalyst == True:
                        return (
                            self.generation_aem_flux_in[t, p, j]
                            == -0.5 * self.flux_splitting[t]
                        )

                    else:
                        return self.generation_aem_flux_in[t, p, j] == (
                            0.5
                            * smooth_min(
                                -(
                                    self.current[t] / pyunits.amp
                                    - self.current_dens_lim_bpem[t]
                                    * self.cell_width
                                    * self.shadow_factor
                                    * self.cell_length
                                    / pyunits.amp
                                ),
                                0,
                            )
                            * pyunits.amp
                            / (self.cell_width * self.shadow_factor * self.cell_length)
                        ) / (Constants.faraday_constant)

                else:
                    return (
                        self.generation_aem_flux_in[t, p, j]
                        == 0 * pyunits.mol * pyunits.meter**-2 * pyunits.s**-1
                    )

        @self.Constraint(
            self.flowsheet().time,
            self.config.property_package.phase_list,
            self.config.property_package.component_list,
            doc="Equation for water splitting acid channel (C.E.M Side) of bipolar membrane flux_out",
        )
        def eq_generation_cem_flux_out(self, t, p, j):
            return (
                self.generation_cem_flux_in[t, p, j]
                == self.generation_cem_flux_out[t, p, j]
            )

        @self.Constraint(
            self.flowsheet().time,
            self.config.property_package.phase_list,
            self.config.property_package.component_list,
            doc="Equation for water splitting base channel (A.E.M Side) of bipolar membrane flux_out",
        )
        def eq_generation_aem_flux_out(self, t, p, j):
            return (
                self.generation_aem_flux_in[t, p, j]
                == self.generation_aem_flux_out[t, p, j]
            )

        @self.Constraint(
            self.flowsheet().time,
            self.config.property_package.phase_list,
            self.config.property_package.component_list,
            doc="Equation for electrical migration across the bipolar membrane flux_in",
        )
        def eq_elec_migration_bpem_flux_in(self, t, p, j):
            if j == "H2O":
                return self.elec_migration_bpem_flux_in[t, p, j] == (
                    self.water_trans_number_membrane["bpem"]
                ) * (
                    self.current[t]
                    / (self.cell_width * self.shadow_factor * self.cell_length)
                    / Constants.faraday_constant
                )

            elif j in self.ion_set:
                if not (j == "H_+" or j == "OH_-"):
                    if self.config.has_catalyst == False:

                        return self.elec_migration_bpem_flux_in[t, p, j] == (
                            self.ion_trans_number_membrane["bpem", j]
                        ) * (
                            self.current_utilization
                            * smooth_min(
                                self.current[t] / pyunits.amp,
                                self.current_dens_lim_bpem[t]
                                * self.cell_width
                                * self.shadow_factor
                                * self.cell_length
                                / pyunits.amp,
                            )
                            * pyunits.amp
                            / (self.cell_width * self.shadow_factor * self.cell_length)
                        ) / (
                            self.config.property_package.charge_comp[j]
                            * Constants.faraday_constant
                        )
                    else:
                        return self.elec_migration_bpem_flux_in[t, p, j] == (
                            self.ion_trans_number_membrane["bpem", j]
                        ) * (
                            self.current_utilization * self.current_dens_lim_bpem[t]
                        ) / (
                            self.config.property_package.charge_comp[j]
                            * Constants.faraday_constant
                        )

                else:

                    self.elec_migration_bpem_flux_in[t, p, j].fix(
                        0 * pyunits.mol * pyunits.m**-2 * pyunits.s**-1
                    )
                    return Constraint.Skip
            else:
                self.elec_migration_bpem_flux_in[t, p, j].fix(
                    0 * pyunits.mol * pyunits.m**-2 * pyunits.s**-1
                )
                return Constraint.Skip

        @self.Constraint(
            self.flowsheet().time,
            self.config.property_package.phase_list,
            self.config.property_package.component_list,
            doc="Equation for electrical migration across the bipolar membrane flux_out",
        )
        def eq_elec_migration_bpem_flux_out(self, t, p, j):
            return (
                self.elec_migration_bpem_flux_out[t, p, j]
                == self.elec_migration_bpem_flux_in[t, p, j]
            )

        @self.Constraint(
            self.flowsheet().time,
            self.config.property_package.phase_list,
            self.config.property_package.component_list,
            doc="Equation for non-electrical flux across the bipolar membrane flux_in",
        )
        def eq_nonelec_bpem_flux_in(self, t, p, j):
            if j == "H2O":
                return self.nonelec_bpem_flux_in[
                    t, p, j
                ] == self.water_density / self.config.property_package.mw_comp[j] * (
                    self.water_permeability_membrane["bpem"]
                ) * (
                    self.basic.properties_in[t].pressure_osm_phase[p]
                    - self.acidic.properties_in[t].pressure_osm_phase[p]
                )

            else:
                return (
                    self.nonelec_bpem_flux_in[t, p, j]
                    == 0 * pyunits.mol * pyunits.m**-2 * pyunits.s**-1
                )

        @self.Constraint(
            self.flowsheet().time,
            self.config.property_package.phase_list,
            self.config.property_package.component_list,
            doc="Equation for non-electrical flux across the bipolar membrane flux_out",
        )
        def eq_nonelec_bpem_flux_out(self, t, p, j):
            if j == "H2O":
                return self.nonelec_bpem_flux_out[
                    t, p, j
                ] == self.water_density / self.config.property_package.mw_comp[j] * (
                    self.water_permeability_membrane["bpem"]
                ) * (
                    self.basic.properties_out[t].pressure_osm_phase[p]
                    - self.acidic.properties_out[t].pressure_osm_phase[p]
                )

            else:
                return (
                    self.nonelec_bpem_flux_out[t, p, j]
                    == 0 * pyunits.mol * pyunits.m**-2 * pyunits.s**-1
                )

        # Add constraints for mass transfer terms (base channel/A.E.M Side of the bipolar membrane)
        @self.Constraint(
            self.flowsheet().time,
            self.config.property_package.phase_list,
            self.config.property_package.component_list,
            doc="Mass transfer term for the base channel (A.E.M Side) of the bipolar membrane",
        )
        def eq_mass_transfer_term_basic(self, t, p, j):
            return (
                self.basic.mass_transfer_term[t, p, j]
                == 0.5
                * (
                    self.generation_aem_flux_in[t, p, j]
                    + self.generation_aem_flux_out[t, p, j]
                    + self.elec_migration_bpem_flux_in[t, p, j]
                    + self.elec_migration_bpem_flux_out[t, p, j]
                    + self.nonelec_bpem_flux_in[t, p, j]
                    + self.nonelec_bpem_flux_out[t, p, j]
                )
                * (self.cell_width * self.shadow_factor * self.cell_length)
                * self.cell_num
            )

        # Add constraints for mass transfer terms (acid channel/C.E.M Side of the bipolar membrane)
        @self.Constraint(
            self.flowsheet().time,
            self.config.property_package.phase_list,
            self.config.property_package.component_list,
            doc="Mass transfer term for the acid channel (C.E.M Side) of the bipolar membrane channel",
        )
        def eq_mass_transfer_term_acidic(self, t, p, j):
            return (
                self.acidic.mass_transfer_term[t, p, j]
                == 0.5
                * (
                    self.generation_cem_flux_in[t, p, j]
                    + self.generation_cem_flux_out[t, p, j]
                    + self.elec_migration_bpem_flux_in[t, p, j]
                    + self.elec_migration_bpem_flux_out[t, p, j]
                    + self.nonelec_bpem_flux_in[t, p, j]
                    + self.nonelec_bpem_flux_out[t, p, j]
                )
                * (self.cell_width * self.shadow_factor * self.cell_length)
                * self.cell_num
            )

        # Performance: acid/base produced
        @self.Constraint(
            self.flowsheet().time,
            doc="Evaluate Base produced",
        )
        def eq_product_basic(self, t):
            conc_unit = 1 * pyunits.mole * pyunits.second**-1
            product_net_loc = 0 * pyunits.kg * pyunits.second**-1

            for j in self.config.property_package.cation_set:
                if not j == "H_+":
                    product_in_loc = smooth_min(
                        self.basic.properties_in[t].flow_mol_phase_comp["Liq", j]
                        / conc_unit,
                        self.basic.properties_in[t].flow_mol_phase_comp["Liq", "OH_-"]
                        / conc_unit
                        / self.config.property_package.charge_comp[j],
                    )

                    product_out_loc = smooth_min(
                        self.basic.properties_out[t].flow_mol_phase_comp["Liq", j]
                        / conc_unit,
                        self.basic.properties_out[t].flow_mol_phase_comp["Liq", "OH_-"]
                        / conc_unit
                        / self.config.property_package.charge_comp[j],
                    )

                    product_net_loc += (
                        -1
                        * smooth_min(product_in_loc - product_out_loc, 0)
                        * conc_unit
                        * (
                            self.config.property_package.charge_comp[j]
                            * self.config.property_package.mw_comp["OH_-"]
                            + self.config.property_package.mw_comp[j]
                        )
                    )

            return self.base_produced == product_net_loc

        @self.Constraint(
            self.flowsheet().time,
            doc="Evaluate Acid produced",
        )
        def eq_product_acidic(self, t):
            conc_unit = 1 * pyunits.mole * pyunits.second**-1
            product_net_loc = 0 * pyunits.kg * pyunits.second**-1

            for j in self.config.property_package.anion_set:
                if not j == "OH_-":
                    product_in_loc = smooth_min(
                        self.acidic.properties_in[t].flow_mol_phase_comp["Liq", j]
                        / conc_unit,
                        self.acidic.properties_in[t].flow_mol_phase_comp["Liq", "H_+"]
                        / conc_unit
                        / (-self.config.property_package.charge_comp[j]),
                    )

                    product_out_loc = smooth_min(
                        self.acidic.properties_out[t].flow_mol_phase_comp["Liq", j]
                        / conc_unit,
                        self.acidic.properties_out[t].flow_mol_phase_comp["Liq", "H_+"]
                        / conc_unit
                        / (-self.config.property_package.charge_comp[j]),
                    )

                    product_net_loc += (
                        -1
                        * smooth_min(product_in_loc - product_out_loc, 0)
                        * conc_unit
                        * (
                            (-self.config.property_package.charge_comp[j])
                            * self.config.property_package.mw_comp["H_+"]
                            + self.config.property_package.mw_comp[j]
                        )
                    )

            return self.acid_produced == product_net_loc

        # Performance: Electrical
        @self.Constraint(
            self.flowsheet().time,
            doc="Electrical power consumption of a stack",
        )
        def eq_power_electrical(self, t):
            return self.power_electrical[t] == self.current[t] * self.voltage[t]

        @self.Constraint(
            self.flowsheet().time,
            doc="Diluate_volume_flow_rate_specific electrical power consumption of a stack",
        )
        def eq_specific_power_electrical(self, t):
            return (
                pyunits.convert(
                    self.specific_power_electrical[t],
                    pyunits.watt * pyunits.second * pyunits.meter**-3,
                )
                * self.acidic.properties_out[t].flow_vol_phase["Liq"]
                == self.current[t] * self.voltage[t]
            )

    # Catalyst action:
    def _make_catalyst(self):

        self.potential_membrane_bpem = Var(
            self.flowsheet().time,
            initialize=0.1,
            bounds=(0, 1e8),
            units=pyunits.volt,
            doc="Potential drop across the depletion layer",
        )
        self.flux_splitting = Var(
            self.flowsheet().time,
            initialize=1,
            domain=NonNegativeReals,
            # bounds=(0, 50000),
            units=pyunits.mole * pyunits.meter**-2 * pyunits.second**-1,
            doc="Flux generated",
        )
        self.membrane_fixed_catalyst_aem = Var(
            self.membrane_set,
            initialize=5e3,
            bounds=(1e-1, 1e5),
            units=pyunits.mole * pyunits.meter**-3,
            doc="Membrane fixed charge",
        )
        self.membrane_fixed_catalyst_cem = Var(
            self.membrane_set,
            initialize=5e3,
            bounds=(1e-1, 1e5),
            units=pyunits.mole * pyunits.meter**-3,
            doc="Membrane fixed charge",
        )

        self.k_a = Var(
            initialize=1e-3,
            bounds=(1e-15, 1e15),
            units=pyunits.mole * pyunits.meter**-3,
            doc="Membrane fixed charge",
        )
        self.k_b = Var(
            initialize=3e-2,
            bounds=(1e-15, 1e15),
            units=pyunits.mole * pyunits.meter**-3,
            doc="Membrane fixed charge",
        )

        const = 0.0936 * pyunits.K**2 * pyunits.volt**-1 * pyunits.meter

        @self.Constraint(
            self.flowsheet().time,
            doc="Calculate the non-dimensional potential drop across the depletion region",
        )
        def eq_potential_membrane_bpem_non_dim(self, t):
            return self.elec_field_non_dim[t] == const * self.basic.properties_in[
                t
            ].temperature ** -2 * self.relative_permittivity["bpem"] ** -1 * sqrt(
                (
                    Constants.faraday_constant
                    * self.membrane_fixed_charge["bpem"]
                    * self.potential_membrane_bpem[t]
                )
                / (
                    Constants.vacuum_electric_permittivity
                    * self.relative_permittivity["bpem"]
                )
            )

        @self.Constraint(
            self.flowsheet().time,
            doc="Calculate the potential barrier at limiting current across the bipolar membrane",
        )
        def eq_flux_splitting(self, t):
            terms = 40
            matrx = 0
            for indx in range(terms):
                matrx += (
                    2**indx
                    * self.elec_field_non_dim[t] ** indx
                    / (math.factorial(indx) * math.factorial(indx + 1))
                )

            matrx *= self.k2_zero["bpem"] * self.conc_water["bpem"]
            matrx_a = matrx * self.membrane_fixed_catalyst_cem["bpem"] / self.k_a
            matrx_b = matrx * self.membrane_fixed_catalyst_aem["bpem"] / self.k_b
            return self.flux_splitting[t] == (matrx_a + matrx_b) * sqrt(
                self.potential_membrane_bpem[t]
                * Constants.vacuum_electric_permittivity
                * self.relative_permittivity["bpem"]
                / (Constants.faraday_constant * self.membrane_fixed_charge["bpem"])
            )

    def _get_fluid_dimensionless_quantities(self):
        # self.diffus_mass = Var(
        #     initialize=1e-9,
        #     bounds=(1e-16, 1e-6),
        #     units=pyunits.meter**2 * pyunits.second**-1,
        #     doc="The mass diffusivity of the solute as molecules (not individual ions)",
        # )
        self.hydraulic_diameter = Var(
            initialize=1e-3,
            bounds=(0, None),
            units=pyunits.meter,
            doc="The hydraulic diameter of the channel",
        )
        self.N_Re = Var(
            initialize=50,
            bounds=(0, None),
            units=pyunits.dimensionless,
            doc="Reynolds Number",
        )
        # self.N_Sc = Var(
        #     initialize=2000,
        #     bounds=(0, None),
        #     units=pyunits.dimensionless,
        #     doc="Schmidt Number",
        # )
        # self.N_Sh = Var(
        #     initialize=100,
        #     bounds=(0, None),
        #     units=pyunits.dimensionless,
        #     doc="Sherwood Number",
        # )

        if self.config.hydraulic_diameter_method == HydraulicDiameterMethod.fixed:
            _log.warning("Do not forget to FIX the channel hydraulic diameter in [m]!")
        else:

            @self.Constraint(
                doc="To calculate hydraulic diameter",
            )
            def eq_hydraulic_diameter(self):
                if (
                    self.config.hydraulic_diameter_method
                    == HydraulicDiameterMethod.conventional
                ):
                    return (
                        self.hydraulic_diameter
                        == 2
                        * self.channel_height
                        * self.cell_width
                        * self.shadow_factor
                        * self.spacer_porosity
                        * (self.channel_height + self.cell_width * self.shadow_factor)
                        ** -1
                    )
                else:
                    self.spacer_specific_area = Var(
                        initialize=1e4,
                        bounds=(0, None),
                        units=pyunits.meter**-1,
                        doc="The specific area of the channel",
                    )
                    return (
                        self.hydraulic_diameter
                        == 4
                        * self.spacer_porosity
                        * (
                            2 * self.channel_height**-1
                            + (1 - self.spacer_porosity) * self.spacer_specific_area
                        )
                        ** -1
                    )

        @self.Constraint(
            doc="To calculate Re",
        )
        def eq_Re(self):

            return (
                self.N_Re
                == self.dens_mass
                * self.velocity_acidic[0]
                * self.hydraulic_diameter
                * self.visc_d**-1
            )

        # @self.Constraint(
        #     doc="To calculate Sc",
        # )
        # def eq_Sc(self):
        #
        #     return self.N_Sc == self.visc_d * self.dens_mass**-1 * self.diffus_mass**-1
        #
        # @self.Constraint(
        #     doc="To calculate Sh",
        # )
        # def eq_Sh(self):
        #
        #     return self.N_Sh == 0.29 * self.N_Re**0.5 * self.N_Sc**0.33

    def _pressure_drop_calculation(self):
        self.pressure_drop_total = Var(
            self.flowsheet().time,
            initialize=1e6,
            units=pyunits.pascal,
            doc="pressure drop over an entire ED stack",
        )
        self.pressure_drop = Var(
            self.flowsheet().time,
            initialize=1e3,
            units=pyunits.pascal * pyunits.meter**-1,
            doc="pressure drop per unit of length",
        )

        if self.config.pressure_drop_method == PressureDropMethod.experimental:
            _log.warning(
                "Do not forget to FIX the experimental pressure drop value in [Pa/m]!"
            )
        else:  # PressureDropMethod.Darcy_Weisbach is used
            # if not (self.config.has_Nernst_diffusion_layer == True):
            self._get_fluid_dimensionless_quantities()

            self.friction_factor = Var(
                initialize=10,
                bounds=(0, None),
                units=pyunits.dimensionless,
                doc="friction factor of the channel fluid",
            )

            @self.Constraint(
                self.flowsheet().time,
                doc="To calculate pressure drop over an stack",
            )
            def eq_pressure_drop_total(self, t):
                return (
                    self.pressure_drop_total[t]
                    == self.dens_mass
                    * self.friction_factor
                    * self.velocity_acidic[0] ** 2
                    * 0.5
                    * self.hydraulic_diameter**-1
                    * self.cell_length
                )

            if self.config.friction_factor_method == FrictionFactorMethod.fixed:
                _log.warning("Do not forget to FIX the Darcy's friction factor value!")
            else:

                @self.Constraint(
                    doc="To calculate friction factor",
                )
                def eq_friction_factor(self):
                    if (
                        self.config.friction_factor_method
                        == FrictionFactorMethod.Gurreri
                    ):
                        return (
                            self.friction_factor
                            == 4 * 50.6 * self.spacer_porosity**-7.06 * self.N_Re**-1
                        )
                    elif (
                        self.config.friction_factor_method
                        == FrictionFactorMethod.Kuroda
                    ):
                        return (
                            self.friction_factor
                            == 4 * 9.6 * self.spacer_porosity**-1 * self.N_Re**-0.5
                        )

        @self.Constraint(
            self.flowsheet().time,
            doc="To calculate total pressure drop over a stack",
        )
        def eq_pressure_drop(self, t):
            return (
                self.pressure_drop[t]
                == self.pressure_drop_total[t] * self.cell_length**-1
            )

    # initialize method
    def initialize_build(
        self,
        state_args=None,
        outlvl=idaeslog.NOTSET,
        solver=None,
        optarg=None,
    ):
        """
        General wrapper for pressure changer initialization routines

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
        # Set solver options
        opt = get_solver(solver, optarg)

        # ---------------------------------------------------------------------

        # Set the outlet has the same intial condition of the inlet.
        for k in self.keys():
            for j in self[k].config.property_package.component_list:
                self[k].basic.properties_out[0].flow_mol_phase_comp["Liq", j] = value(
                    self[k].basic.properties_in[0].flow_mol_phase_comp["Liq", j]
                )
                self[k].acidic.properties_out[0].flow_mol_phase_comp["Liq", j] = value(
                    self[k].acidic.properties_in[0].flow_mol_phase_comp["Liq", j]
                )

        # ---------------------------------------------------------------------
        # Initialize concentrate_basic_side block
        flags_basic = self.basic.initialize(
            outlvl=outlvl,
            optarg=optarg,
            solver=solver,
            state_args=state_args,
            hold_state=True,
        )
        init_log.info_high("Initialization Step 2 Complete.")
        # ---------------------------------------------------------------------
        # Initialize concentrate_acidic_side block
        flags_acidic = self.acidic.initialize(
            outlvl=outlvl,
            optarg=optarg,
            solver=solver,
            state_args=state_args,  # inlet var
            hold_state=True,
        )
        init_log.info_high("Initialization Step 3 Complete.")
        # ---------------------------------------------------------------------
        # Solve unit
        with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
            res = opt.solve(self, tee=slc.tee)
        init_log.info_high("Initialization Step 3 {}.".format(idaeslog.condition(res)))
        # ---------------------------------------------------------------------
        # Release state
        self.basic.release_state(flags_basic, outlvl)
        init_log.info("Initialization Complete: {}".format(idaeslog.condition(res)))
        self.acidic.release_state(flags_acidic, outlvl)
        init_log.info("Initialization Complete: {}".format(idaeslog.condition(res)))

        if not check_optimal_termination(res):
            raise InitializationError(f"Unit model {self.name} failed to initialize")

    def calculate_scaling_factors(self):
        super().calculate_scaling_factors()
        # Scaling factors that user may setup
        # The users are highly encouraged to provide scaling factors for assessable vars below.
        # Not providing these vars will give a warning.
        if (
            iscale.get_scaling_factor(self.solute_diffusivity_membrane, warning=True)
            is None
        ):
            iscale.set_scaling_factor(self.solute_diffusivity_membrane, 1e10)
        if iscale.get_scaling_factor(self.membrane_thickness, warning=True) is None:
            iscale.set_scaling_factor(self.membrane_thickness, 1e4)
        if (
            iscale.get_scaling_factor(self.water_permeability_membrane, warning=True)
            is None
        ):
            iscale.set_scaling_factor(self.water_permeability_membrane, 1e14)
        if iscale.get_scaling_factor(self.cell_num, warning=True) is None:
            iscale.set_scaling_factor(self.cell_num, 0.1)
        if iscale.get_scaling_factor(self.cell_length, warning=True) is None:
            iscale.set_scaling_factor(self.cell_length, 1e1)
        if iscale.get_scaling_factor(self.cell_width, warning=True) is None:
            iscale.set_scaling_factor(self.cell_width, 1e2)
        if iscale.get_scaling_factor(self.shadow_factor, warning=True) is None:
            iscale.set_scaling_factor(self.shadow_factor, 1)
        if iscale.get_scaling_factor(self.channel_height, warning=True) is None:
            iscale.set_scaling_factor(self.channel_height, 1e5)
        if iscale.get_scaling_factor(self.spacer_porosity, warning=True) is None:
            iscale.set_scaling_factor(self.spacer_porosity, 1)
        if (
            iscale.get_scaling_factor(self.membrane_areal_resistance, warning=True)
            is None
        ):
            iscale.set_scaling_factor(self.membrane_areal_resistance, 1e5)
        if iscale.get_scaling_factor(self.electrodes_resistance, warning=True) is None:
            iscale.set_scaling_factor(self.electrodes_resistance, 1e1)
        if iscale.get_scaling_factor(self.current, warning=True) is None:
            iscale.set_scaling_factor(self.current, 1)
        if iscale.get_scaling_factor(self.voltage, warning=True) is None:
            iscale.set_scaling_factor(self.voltage, 1)
        if hasattr(self, "conc_water") and (
            iscale.get_scaling_factor(self.conc_water, warning=True) is None
        ):
            iscale.set_scaling_factor(self.conc_water, 1e-4)

        if hasattr(self, "acid_produced") and (
            iscale.get_scaling_factor(self.acid_produced, warning=True) is None
        ):
            sf = 1
            for j in self.config.property_package.anion_set:
                sf = smooth_min(
                    iscale.get_scaling_factor(
                        self.acidic.properties_in[0].flow_mass_phase_comp["Liq", j]
                    ),
                    iscale.get_scaling_factor(
                        self.acidic.properties_in[0].flow_mass_phase_comp["Liq", "H_+"]
                    ),
                )

            iscale.set_scaling_factor(self.acid_produced, sf)

        if hasattr(self, "base_produced") and (
            iscale.get_scaling_factor(self.base_produced, warning=True) is None
        ):
            sf = 1
            for j in self.config.property_package.cation_set:
                sf = smooth_min(
                    iscale.get_scaling_factor(
                        self.basic.properties_in[0].flow_mass_phase_comp["Liq", j]
                    ),
                    iscale.get_scaling_factor(
                        self.basic.properties_in[0].flow_mass_phase_comp["Liq", "OH_-"]
                    ),
                )

            iscale.set_scaling_factor(self.base_produced, sf)

        if self.config.has_catalyst == True:
            if (
                iscale.get_scaling_factor(
                    self.membrane_fixed_catalyst_cem, warning=True
                )
                is None
            ):
                iscale.set_scaling_factor(self.membrane_fixed_catalyst_cem, 1e-3)
            # if self.config.has_catalyst == True:
            if (
                iscale.get_scaling_factor(
                    self.membrane_fixed_catalyst_aem, warning=True
                )
                is None
            ):
                iscale.set_scaling_factor(self.membrane_fixed_catalyst_aem, 1e-3)

            if iscale.get_scaling_factor(self.k_a, warning=True) is None:
                iscale.set_scaling_factor(self.k_a, 1e6)

            if iscale.get_scaling_factor(self.k_b, warning=True) is None:
                iscale.set_scaling_factor(self.k_b, 1e2)

        if (
            self.config.has_catalyst == True
            or self.config.limiting_potential_method_bpem
            == LimitingpotentialMethod.Empirical
        ) and iscale.get_scaling_factor(self.elec_field_non_dim, warning=True) is None:
            iscale.set_scaling_factor(self.elec_field_non_dim, 1e-1)

        if (
            self.config.limiting_potential_method_bpem
            == LimitingpotentialMethod.Empirical
            and iscale.get_scaling_factor(self.potential_barrier_bpem, warning=True)
            is None
        ):
            iscale.set_scaling_factor(self.potential_barrier_bpem, 1)

        if hasattr(self, "membrane_fixed_charge") and (
            iscale.get_scaling_factor(self.membrane_fixed_charge, warning=True) is None
        ):
            iscale.set_scaling_factor(self.membrane_fixed_charge, 1e-3)
        if hasattr(self, "diffus_mass") and (
            iscale.get_scaling_factor(self.diffus_mass, warning=True) is None
        ):
            iscale.set_scaling_factor(self.diffus_mass, 1e9)
        if hasattr(self, "salt_conc_aem") and (
            iscale.get_scaling_factor(self.salt_conc_aem, warning=True) is None
        ):
            iscale.set_scaling_factor(self.salt_conc_aem, 1e-2)
        if hasattr(self, "salt_conc_cem") and (
            iscale.get_scaling_factor(self.salt_conc_cem, warning=True) is None
        ):
            iscale.set_scaling_factor(self.salt_conc_cem, 1e-2)

        if (
            self.config.has_catalyst == True
            or self.config.limiting_potential_method_bpem
            == LimitingpotentialMethod.Empirical
        ):

            if hasattr(self, "relative_permittivity") and (
                iscale.get_scaling_factor(self.relative_permittivity, warning=True)
                is None
            ):
                iscale.set_scaling_factor(self.relative_permittivity, 1e-1)

        if iscale.get_scaling_factor(self.kr, warning=True) is None:
            iscale.set_scaling_factor(self.kr, 1e-11)
        if (
            hasattr(self, "k2_zero")
            and iscale.get_scaling_factor(self.k2_zero, warning=True) is None
        ):
            iscale.set_scaling_factor(self.k2_zero, 1e5)

        # The folloing Vars are built for constructing constraints and their sf are computed from other Vars.

        if (
            self.config.has_catalyst == True
            and iscale.get_scaling_factor(self.potential_membrane_bpem, warning=True)
            is None
        ):
            sf = (
                (
                    iscale.get_scaling_factor(self.elec_field_non_dim)
                    * iscale.get_scaling_factor(self.relative_permittivity)
                    * 293**-2
                    / 0.09636**-1
                )
                ** 2
                * value(Constants.vacuum_electric_permittivity) ** -1
                * iscale.get_scaling_factor(self.relative_permittivity)
                * value(Constants.faraday_constant) ** -1
                * iscale.get_scaling_factor(self.membrane_fixed_charge)
            )

            iscale.set_scaling_factor(self.potential_membrane_bpem, 1e1)

        if self.config.has_catalyst == True:
            if iscale.get_scaling_factor(self.flux_splitting, warning=True) is None:

                terms = 40
                sf = 0
                for indx in range(terms):
                    sf += (
                        2**indx
                        * iscale.get_scaling_factor(self.elec_field_non_dim) ** -indx
                        / (math.factorial(indx) * math.factorial(indx + 1))
                    )

                sf **= -1
                sf *= iscale.get_scaling_factor(
                    self.k2_zero
                ) * iscale.get_scaling_factor(self.conc_water)
                sf_a = (
                    sf
                    * iscale.get_scaling_factor(self.membrane_fixed_catalyst_cem)
                    / iscale.get_scaling_factor(self.k_a)
                )
                sf_b = (
                    sf
                    * iscale.get_scaling_factor(self.membrane_fixed_catalyst_aem)
                    / iscale.get_scaling_factor(self.k_b)
                )

                sf = (sf_a**-1 + sf_b**-1) ** -1 * sqrt(
                    iscale.get_scaling_factor(self.potential_membrane_bpem)
                    * value(Constants.vacuum_electric_permittivity) ** -1
                    * iscale.get_scaling_factor(self.relative_permittivity)
                    / (
                        value(Constants.faraday_constant) ** -1
                        * iscale.get_scaling_factor(self.membrane_fixed_charge)
                    )
                )

                iscale.set_scaling_factor(self.flux_splitting, sf)

        iscale.set_scaling_factor(
            self.elec_migration_bpem_flux_in,
            iscale.get_scaling_factor(self.current)
            * iscale.get_scaling_factor(self.cell_length) ** -1
            * iscale.get_scaling_factor(self.cell_width) ** -1
            * iscale.get_scaling_factor(self.shadow_factor) ** -1
            * 1e5,
        )
        iscale.set_scaling_factor(
            self.elec_migration_bpem_flux_out,
            iscale.get_scaling_factor(self.current)
            * iscale.get_scaling_factor(self.cell_length) ** -1
            * iscale.get_scaling_factor(self.cell_width) ** -1
            * iscale.get_scaling_factor(self.shadow_factor) ** -1
            * 1e5,
        )

        for ind in self.generation_cem_flux_in:
            if ind[2] == "H_+" or "H2O":
                if self.config.has_catalyst == True:
                    sf = 0.5 * iscale.get_scaling_factor(self.flux_splitting)
                else:
                    sf = iscale.get_scaling_factor(self.elec_migration_bpem_flux_in)
            else:
                sf = 1

            iscale.set_scaling_factor(self.generation_cem_flux_in[ind], sf)

        for ind in self.generation_cem_flux_out:
            if ind[2] == "H_+" or "H2O":
                if self.config.has_catalyst == True:
                    sf = 0.5 * iscale.get_scaling_factor(self.flux_splitting)
                else:
                    sf = iscale.get_scaling_factor(self.elec_migration_bpem_flux_out)
            else:
                sf = 1

            iscale.set_scaling_factor(self.generation_cem_flux_out[ind], sf)

        for ind in self.generation_aem_flux_in:
            if ind[2] == "OH_-" or "H2O":
                if self.config.has_catalyst == True:
                    sf = iscale.get_scaling_factor(self.flux_splitting)
                else:
                    sf = iscale.get_scaling_factor(self.elec_migration_bpem_flux_in)
            else:
                sf = 1

            iscale.set_scaling_factor(self.generation_aem_flux_in[ind], sf)

        for ind in self.generation_aem_flux_out:
            if ind[2] == "OH_-" or "H2O":
                if self.config.has_catalyst == True:
                    sf = iscale.get_scaling_factor(self.flux_splitting)
                else:
                    sf = iscale.get_scaling_factor(self.elec_migration_bpem_flux_out)
            else:
                sf = 1

            iscale.set_scaling_factor(self.generation_aem_flux_out[ind], sf)

        for ind in self.nonelec_bpem_flux_in:
            if ind[2] == "H2O":
                sf = (
                    1e-3
                    * 0.018
                    * iscale.get_scaling_factor(self.water_permeability_membrane)
                    * iscale.get_scaling_factor(
                        self.acidic.properties_in[ind[0]].pressure_osm_phase[ind[1]]
                    )
                )
            else:
                sf = 1
            iscale.set_scaling_factor(self.nonelec_bpem_flux_in[ind], sf)
        for ind in self.nonelec_bpem_flux_out:
            if ind[2] == "H2O":
                sf = (
                    1e-3
                    * 0.018
                    * iscale.get_scaling_factor(self.water_permeability_membrane)
                    * iscale.get_scaling_factor(
                        self.acidic.properties_out[ind[0]].pressure_osm_phase[ind[1]]
                    )
                )
            else:
                sf = 1

            iscale.set_scaling_factor(self.nonelec_bpem_flux_out[ind], sf)

        for ind in self.acidic.mass_transfer_term:
            if ind[2] == "H_+":
                sf = iscale.get_scaling_factor(self.generation_cem_flux_in[ind])
            else:
                if ind[2] == "H2O":
                    sf = iscale.get_scaling_factor(self.nonelec_bpem_flux_in[ind])
                else:
                    sf = iscale.get_scaling_factor(
                        self.elec_migration_bpem_flux_in[ind]
                    )

            sf *= (
                iscale.get_scaling_factor(self.cell_width)
                * iscale.get_scaling_factor(self.shadow_factor)
                * iscale.get_scaling_factor(self.cell_length)
                * iscale.get_scaling_factor(self.cell_num)
            )
            iscale.set_scaling_factor(self.acidic.mass_transfer_term[ind], sf)
        #
        for ind in self.basic.mass_transfer_term:
            if ind[2] == "OH_-":
                sf = iscale.get_scaling_factor(self.generation_aem_flux_in[ind])
            else:
                if ind[2] == "H2O":
                    sf = iscale.get_scaling_factor(self.nonelec_bpem_flux_in[ind])
                else:
                    sf = iscale.get_scaling_factor(
                        self.elec_migration_bpem_flux_in[ind]
                    )

            sf *= (
                iscale.get_scaling_factor(self.cell_width)
                * iscale.get_scaling_factor(self.shadow_factor)
                * iscale.get_scaling_factor(self.cell_length)
                * iscale.get_scaling_factor(self.cell_num)
            )
            iscale.set_scaling_factor(self.basic.mass_transfer_term[ind], sf)

        for ind, c in self.specific_power_electrical.items():
            iscale.set_scaling_factor(
                self.specific_power_electrical[ind],
                3.6e6
                * iscale.get_scaling_factor(self.current[ind])
                * iscale.get_scaling_factor(self.voltage[ind])
                * iscale.get_scaling_factor(
                    self.acidic.properties_out[ind].flow_vol_phase["Liq"]
                )
                ** -1,
            )

        for ind in self.velocity_basic:
            if (
                iscale.get_scaling_factor(self.velocity_basic[ind], warning=False)
                is None
            ):
                sf = (
                    iscale.get_scaling_factor(
                        self.basic.properties_in[ind].flow_vol_phase["Liq"]
                    )
                    * iscale.get_scaling_factor(self.cell_width) ** -1
                    * iscale.get_scaling_factor(self.shadow_factor) ** -1
                    * iscale.get_scaling_factor(self.channel_height) ** -1
                    * iscale.get_scaling_factor(self.spacer_porosity) ** -1
                    * iscale.get_scaling_factor(self.cell_num) ** -1
                )

                iscale.set_scaling_factor(self.velocity_basic[ind], sf)

        for ind in self.velocity_acidic:
            if (
                iscale.get_scaling_factor(self.velocity_acidic[ind], warning=False)
                is None
            ):
                sf = (
                    iscale.get_scaling_factor(
                        self.acidic.properties_in[ind].flow_vol_phase["Liq"]
                    )
                    * iscale.get_scaling_factor(self.cell_width) ** -1
                    * iscale.get_scaling_factor(self.shadow_factor) ** -1
                    * iscale.get_scaling_factor(self.channel_height) ** -1
                    * iscale.get_scaling_factor(self.spacer_porosity) ** -1
                    * iscale.get_scaling_factor(self.cell_num) ** -1
                )

                iscale.set_scaling_factor(self.velocity_acidic[ind], sf)

        if hasattr(self, "spacer_specific_area") and (
            iscale.get_scaling_factor(self.spacer_specific_area, warning=True) is None
        ):
            iscale.set_scaling_factor(self.spacer_specific_area, 1e-4)
        if hasattr(self, "hydraulic_diameter") and (
            iscale.get_scaling_factor(self.hydraulic_diameter, warning=True) is None
        ):
            iscale.set_scaling_factor(self.hydraulic_diameter, 1e4)
        if hasattr(self, "N_Re") and (
            iscale.get_scaling_factor(self.N_Re, warning=True) is None
        ):
            sf = (
                iscale.get_scaling_factor(self.dens_mass)
                * iscale.get_scaling_factor(self.velocity_acidic[0])
                * iscale.get_scaling_factor(self.hydraulic_diameter)
                * iscale.get_scaling_factor(self.visc_d) ** -1
            )
            iscale.set_scaling_factor(self.N_Re, sf)
        if hasattr(self, "N_Sc") and (
            iscale.get_scaling_factor(self.N_Sc, warning=True) is None
        ):
            sf = (
                iscale.get_scaling_factor(self.visc_d)
                * iscale.get_scaling_factor(self.dens_mass) ** -1
                * iscale.get_scaling_factor(self.diffus_mass) ** -1
            )
            iscale.set_scaling_factor(self.N_Sc, sf)
        if hasattr(self, "N_Sh") and (
            iscale.get_scaling_factor(self.N_Sh, warning=True) is None
        ):
            sf = (
                10
                * iscale.get_scaling_factor(self.N_Re) ** 0.5
                * iscale.get_scaling_factor(self.N_Sc) ** 0.33
            )
            iscale.set_scaling_factor(self.N_Sh, sf)
        if hasattr(self, "friction_factor") and (
            iscale.get_scaling_factor(self.friction_factor, warning=True) is None
        ):
            if self.config.friction_factor_method == FrictionFactorMethod.fixed:
                sf = 0.1
            elif self.config.friction_factor_method == FrictionFactorMethod.Gurreri:
                sf = (
                    (4 * 50.6) ** -1
                    * (iscale.get_scaling_factor(self.spacer_porosity)) ** -7.06
                    * iscale.get_scaling_factor(self.N_Re) ** -1
                )
            elif self.config.friction_factor_method == FrictionFactorMethod.Kuroda:
                sf = (4 * 9.6) ** -1 * iscale.get_scaling_factor(self.N_Re) ** -0.5
            iscale.set_scaling_factor(self.friction_factor, sf)

        if hasattr(self, "pressure_drop") and (
            iscale.get_scaling_factor(self.pressure_drop, warning=True) is None
        ):
            if self.config.pressure_drop_method == PressureDropMethod.experimental:
                sf = 1e-5
            else:
                sf = (
                    iscale.get_scaling_factor(self.dens_mass)
                    * iscale.get_scaling_factor(self.friction_factor)
                    * iscale.get_scaling_factor(self.velocity_acidic[0]) ** 2
                    * 2
                    * iscale.get_scaling_factor(self.hydraulic_diameter) ** -1
                )
            iscale.set_scaling_factor(self.pressure_drop, sf)

        if hasattr(self, "pressure_drop_total") and (
            iscale.get_scaling_factor(self.pressure_drop_total, warning=True) is None
        ):
            if self.config.pressure_drop_method == PressureDropMethod.experimental:
                sf = 1e-5 * iscale.get_scaling_factor(self.cell_length)
            else:
                sf = (
                    iscale.get_scaling_factor(self.dens_mass)
                    * iscale.get_scaling_factor(self.friction_factor)
                    * iscale.get_scaling_factor(self.velocity_acidic[0]) ** 2
                    * 2
                    * iscale.get_scaling_factor(self.hydraulic_diameter) ** -1
                    * iscale.get_scaling_factor(self.cell_length)
                )
            iscale.set_scaling_factor(self.pressure_drop_total, sf)

        if hasattr(self, "current_dens_lim_bpem"):
            if iscale.get_scaling_factor(self.current_dens_lim_bpem) is None:
                if (
                    self.config.limiting_current_density_method_bpem
                    == LimitingCurrentDensitybpemMethod.InitialValue
                ):
                    sf = self.config.limiting_current_density_bpem_data**-1
                    iscale.set_scaling_factor(self.current_dens_lim_bpem, sf)
                elif (
                    self.config.limiting_current_density_method_bpem
                    == LimitingCurrentDensitybpemMethod.Empirical
                ):
                    sf = (
                        iscale.get_scaling_factor(self.diffus_mass)
                        * value(Constants.faraday_constant) ** -1
                        * (1 / 2 * iscale.get_scaling_factor(self.salt_conc_aem)) ** 2
                        / (
                            iscale.get_scaling_factor(self.membrane_thickness)
                            * iscale.get_scaling_factor(self.membrane_fixed_charge)
                        )
                    )

                    iscale.set_scaling_factor(self.current_dens_lim_bpem, sf)

        iscale.set_scaling_factor(
            self.power_electrical,
            iscale.get_scaling_factor(self.current)
            * iscale.get_scaling_factor(self.voltage),
        )

        # Constraint scaling
        for ind, c in self.eq_current_voltage_relation.items():
            iscale.constraint_scaling_transform(
                c, iscale.get_scaling_factor(self.membrane_areal_resistance)
            )
        for ind, c in self.eq_power_electrical.items():
            iscale.constraint_scaling_transform(
                c, iscale.get_scaling_factor(self.power_electrical)
            )
        for ind, c in self.eq_specific_power_electrical.items():
            iscale.constraint_scaling_transform(
                c, iscale.get_scaling_factor(self.specific_power_electrical[ind])
            )
        for ind, c in self.eq_elec_migration_bpem_flux_in.items():
            iscale.constraint_scaling_transform(
                c, iscale.get_scaling_factor(self.elec_migration_bpem_flux_in)
            )
        for ind, c in self.eq_elec_migration_bpem_flux_out.items():
            iscale.constraint_scaling_transform(
                c, iscale.get_scaling_factor(self.elec_migration_bpem_flux_in)
            )

        for ind, c in self.eq_nonelec_bpem_flux_in.items():
            iscale.constraint_scaling_transform(
                c, iscale.get_scaling_factor(self.nonelec_bpem_flux_in[ind])
            )

        for ind, c in self.eq_nonelec_bpem_flux_out.items():
            iscale.constraint_scaling_transform(
                c, iscale.get_scaling_factor(self.nonelec_bpem_flux_out[ind])
            )

        for ind, c in self.eq_mass_transfer_term_basic.items():
            iscale.constraint_scaling_transform(
                c,
                min(
                    iscale.get_scaling_factor(self.generation_aem_flux_in[ind]),
                    iscale.get_scaling_factor(self.elec_migration_bpem_flux_in[ind]),
                    iscale.get_scaling_factor(
                        self.nonelec_bpem_flux_in[ind],
                        self.elec_migration_bpem_flux_out[ind],
                    ),
                    iscale.get_scaling_factor(self.nonelec_bpem_flux_out[ind]),
                )
                * iscale.get_scaling_factor(self.cell_width)
                * iscale.get_scaling_factor(self.shadow_factor)
                * iscale.get_scaling_factor(self.cell_length)
                * iscale.get_scaling_factor(self.cell_num),
            )
        for ind, c in self.eq_mass_transfer_term_acidic.items():
            iscale.constraint_scaling_transform(
                c,
                min(
                    iscale.get_scaling_factor(self.generation_cem_flux_in[ind]),
                    iscale.get_scaling_factor(self.elec_migration_bpem_flux_out[ind]),
                    iscale.get_scaling_factor(
                        self.nonelec_bpem_flux_in[ind],
                        self.elec_migration_bpem_flux_out[ind],
                    ),
                    iscale.get_scaling_factor(self.nonelec_bpem_flux_out[ind]),
                )
                * iscale.get_scaling_factor(self.cell_width)
                * iscale.get_scaling_factor(self.shadow_factor)
                * iscale.get_scaling_factor(self.cell_length)
                * iscale.get_scaling_factor(self.cell_num),
            )

        for ind, c in self.eq_power_electrical.items():
            iscale.constraint_scaling_transform(
                c,
                iscale.get_scaling_factor(self.power_electrical[ind]),
            )

        for ind, c in self.eq_specific_power_electrical.items():
            iscale.constraint_scaling_transform(
                c,
                iscale.get_scaling_factor(self.specific_power_electrical[ind])
                * iscale.get_scaling_factor(
                    self.acidic.properties_out[ind].flow_vol_phase["Liq"]
                ),
            )

    def _get_stream_table_contents(self, time_point=0):
        return create_stream_table_dataframe(
            {
                "base channel of the bipolar membrane Channel Inlet": self.inlet_basic,
                "acid channel of the bipolar membrane Channel Inlet": self.inlet_acidic,
                "base channel of the bipolar membrane Channel Outlet": self.outlet_basic,
                "acid channel of the bipolar membrane Channel Outlet": self.outlet_acidic,
            },
            time_point=time_point,
        )

    def _get_performance_contents(self, time_point=0):
        return {
            "vars": {
                "Total electrical power consumption(Watt)": self.power_electrical[
                    time_point
                ],
                "Specific electrical power consumption (kW*h/m**3)": self.specific_power_electrical[
                    time_point
                ],
            },
            "exprs": {},
            "params": {},
        }

    def get_power_electrical(self, time_point=0):
        return self.power_electrical[time_point]

    @property
    def default_costing_method(self):
        return cost_bipolar_electrodialysis
