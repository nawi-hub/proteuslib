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
    Set,
    Var,
    Param,
    check_optimal_termination,
    Suffix,
    Constraint,
    NonNegativeReals,
    value,
    units as pyunits,
    log,
)
from pyomo.dae import (
    DerivativeVar,
)
from pyomo.common.config import ConfigBlock, ConfigValue, In, Bool

# Import Watertap cores
from watertap.core.util.initialization import check_solve, check_dof

# Import IDAES cores
from idaes.core import (
    declare_process_block_class,
    EnergyBalanceType,
    MaterialBalanceType,
    MomentumBalanceType,
    UnitModelBlockData,
    useDefault,
)
from idaes.core.util.constants import Constants
from watertap.core.solvers import get_solver
from idaes.core.util.tables import create_stream_table_dataframe
from idaes.core.util.config import is_physical_parameter_block
from idaes.core.util.exceptions import ConfigurationError, InitializationError
from idaes.core.util.misc import add_object_reference
import idaes.core.util.scaling as iscale
import idaes.logger as idaeslog
from enum import Enum

from watertap.core import ControlVolume1DBlock, InitializationMixin
from watertap.costing.unit_models.electrodialysis import cost_electrodialysis

__author__ = "Xiangyu Bi, Austin Ladshaw"

_log = idaeslog.getLogger(__name__)


class LimitingCurrentDensityMethod(Enum):
    InitialValue = 0
    Empirical = 1
    Theoretical = 2


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
@declare_process_block_class("Electrodialysis1D")
class Electrodialysis1DData(InitializationMixin, UnitModelBlockData):
    """
    1D Electrodialysis Model
    """

    # CONFIG are options for the unit model
    CONFIG = ConfigBlock()

    # These config args are common to any control volume
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
    **default** - False.""",
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
        "operation_mode",
        ConfigValue(
            default=ElectricalOperationMode.Constant_Voltage,
            domain=In(ElectricalOperationMode),
            description="The electrical operation mode. To be selected between Constant Current and Constant Voltage",
        ),
    )
    CONFIG.declare(
        "limiting_current_density_method",
        ConfigValue(
            default=LimitingCurrentDensityMethod.InitialValue,
            domain=In(LimitingCurrentDensityMethod),
            description="Configuration for method to compute the limiting current density",
            doc="""
           **default** - ``LimitingCurrentDensityMethod.InitialValue``

       .. csv-table::
           :header: "Configuration Options", "Description"

           "``LimitingCurrentDensityMethod.InitialValue``", "Limiting current is calculated from a single initial value of the feed solution tested by the user." 
           "``LimitingCurrentDensityMethod.Empirical``", "Limiting current density is calculated from the empirical equation."
           "``LimitingCurrentDensityMethod.Theoretical``", "Limiting current density is calculated from a theoretical equation."
       """,
        ),
    )

    CONFIG.declare(
        "limiting_current_density_data",
        ConfigValue(
            default=500,
            description="Limiting current density data input",
        ),
    )

    CONFIG.declare(
        "has_nonohmic_potential_membrane",
        ConfigValue(
            default=False,
            domain=Bool,
            description="Configuration for whether to model the nonohmic potential across ion exchange membranes",
        ),
    )

    CONFIG.declare(
        "has_Nernst_diffusion_layer",
        ConfigValue(
            default=False,
            domain=Bool,
            description="Configuration for whether to simulate the concentration-polarized diffusion layers",
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

    CONFIG.declare(
        "transformation_method",
        ConfigValue(
            default="dae.finite_difference",
            description="Discretization method to use for DAE transformation",
            doc="""Discretization method to use for DAE transformation. See Pyomo
    documentation for supported transformations.""",
        ),
    )

    CONFIG.declare(
        "transformation_scheme",
        ConfigValue(
            default="BACKWARD",
            description="Discretization scheme to use for DAE transformation",
            doc="""Discretization scheme to use when transforming domain. See
    Pyomo documentation for supported schemes.""",
        ),
    )

    CONFIG.declare(
        "finite_elements",
        ConfigValue(
            default=10,
            domain=int,
            description="Number of finite elements in length domain",
            doc="""Number of finite elements to use when discretizing length
            domain (default=10)""",
        ),
    )

    CONFIG.declare(
        "collocation_points",
        ConfigValue(
            default=2,
            domain=int,
            description="Number of collocation points per finite element",
            doc="""Number of collocation points to use per finite element when
            discretizing length domain (default=2)""",
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
        self._validate_config()
        # this creates blank scaling factors, which are populated later
        self.scaling_factor = Suffix(direction=Suffix.EXPORT)
        # Get the base units of measurement from the property definition
        units_meta = self.config.property_package.get_metadata().get_derived_units
        # Create essential sets
        add_object_reference(
            self, "component_set", self.config.property_package.component_list
        )
        add_object_reference(
            self, "solute_set", self.config.property_package.solute_set
        )
        add_object_reference(self, "ion_set", self.config.property_package.ion_set)

        add_object_reference(
            self, "cation_set", self.config.property_package.cation_set
        )
        add_object_reference(self, "anion_set", self.config.property_package.anion_set)

        self.membrane_set = Set(
            initialize=["cem", "aem"]
        )  #   cem = Cation-Exchange Membrane aem = Anion-Exchange Membrane
        self.electrode_side_set = Set(initialize=["cathode_left", "anode_right"])
        self.flow_channel_set = Set(initialize=["concentrate", "diluate"])
        # Length var for building 1D control volume
        self.cell_length = Var(
            initialize=0.5,
            bounds=(1e-3, 1e2),
            units=pyunits.meter,
            doc="Length of the electrodialysis cell/stack, in parallel to the convective flow",
        )

        # Control Volume for the Diluate channel:
        self.diluate = ControlVolume1DBlock(
            dynamic=self.config.dynamic,
            has_holdup=self.config.has_holdup,
            property_package=self.config.property_package,
            property_package_args=self.config.property_package_args,
            transformation_method=self.config.transformation_method,
            transformation_scheme=self.config.transformation_scheme,
            finite_elements=self.config.finite_elements,
            collocation_points=self.config.collocation_points,
        )
        self.diluate.add_geometry(length_var=self.cell_length)
        self.diluate.add_state_blocks(has_phase_equilibrium=False)
        self.diluate.add_material_balances(
            balance_type=self.config.material_balance_type, has_mass_transfer=True
        )

        self.diluate.add_energy_balances(
            balance_type=self.config.energy_balance_type,
            has_enthalpy_transfer=False,
        )

        if self.config.is_isothermal:
            self.diluate.add_isothermal_assumption()

        self.diluate.add_momentum_balances(
            balance_type=self.config.momentum_balance_type,
            has_pressure_change=self.config.has_pressure_change,
        )

        # Below is declared the electrical power var and its derivative var,
        # which is a performance metric of the entire electrodialysis stack.
        # This var takes the "diluate" as the parent to utilize the discretization (as in Pyomo DAE) of this block for solving.
        self.diluate.power_electrical_x = Var(
            self.flowsheet().time,
            self.diluate.length_domain,
            initialize=0,
            bounds=(0, 12100),
            domain=NonNegativeReals,
            units=pyunits.watt,
            doc="Electrical power consumption of a stack",
        )
        self.diluate.Dpower_electrical_Dx = DerivativeVar(
            self.diluate.power_electrical_x,
            wrt=self.diluate.length_domain,
            units=pyunits.watt,
        )
        add_object_reference(
            self, "dens_mass", self.diluate.properties[0, 0].dens_mass_phase["Liq"]
        )
        add_object_reference(
            self, "visc_d", self.diluate.properties[0, 0].visc_d_phase["Liq"]
        )

        # Apply the discretization transformation (Pyomo DAE) to the diluate block
        self.diluate.apply_transformation()

        # taken from membrane_channel1d because model is currently using ControlVolume1D
        def _skip_element(x):
            if self.config.transformation_scheme != "FORWARD":
                return x == self.diluate.length_domain.first()
            else:
                return x == self.diluate.length_domain.last()

        self.difference_elements = Set(
            ordered=True,
            initialize=(x for x in self.diluate.length_domain if not _skip_element(x)),
        )

        # Control volume for the Concentrate channel
        self.concentrate = ControlVolume1DBlock(
            dynamic=self.config.dynamic,
            has_holdup=self.config.has_holdup,
            property_package=self.config.property_package,
            property_package_args=self.config.property_package_args,
            transformation_method=self.config.transformation_method,
            transformation_scheme=self.config.transformation_scheme,
            finite_elements=self.config.finite_elements,
            collocation_points=self.config.collocation_points,
        )
        self.concentrate.add_geometry(length_var=self.cell_length)
        self.concentrate.add_state_blocks(has_phase_equilibrium=False)
        self.concentrate.add_material_balances(
            balance_type=self.config.material_balance_type, has_mass_transfer=True
        )

        self.concentrate.add_energy_balances(
            balance_type=self.config.energy_balance_type,
            has_enthalpy_transfer=False,
        )

        if self.config.is_isothermal:
            self.concentrate.add_isothermal_assumption()

        self.concentrate.add_momentum_balances(
            balance_type=self.config.momentum_balance_type,
            has_pressure_change=self.config.has_pressure_change,
        )
        self.concentrate.apply_transformation()

        # Add inlet and outlet as ports for Diluate and Concentrate
        self.add_inlet_port(name="inlet_diluate", block=self.diluate)
        self.add_outlet_port(name="outlet_diluate", block=self.diluate)
        self.add_inlet_port(name="inlet_concentrate", block=self.concentrate)
        self.add_outlet_port(name="outlet_concentrate", block=self.concentrate)

        # Create unit model parameters and vars
        self.cell_pair_num = Var(
            initialize=1,
            domain=NonNegativeReals,
            bounds=(1, 10000),
            units=pyunits.dimensionless,
            doc="cell pair number in a stack",
        )
        # Note: domain is set as Reals for optimizaiton convenience,
        # although this var is technically integer.

        # electrodialysis cell dimensional properties
        self.cell_width = Var(
            initialize=0.1,
            bounds=(1e-3, 1e2),
            units=pyunits.meter,
            doc="The width of the electrodialysis cell, parallel to the current/ion flux direction",
        )
        self.channel_height = Var(
            initialize=0.0001,
            units=pyunits.meter,
            doc="The distance between consecutive aem and cem",
        )
        self.spacer_porosity = Var(
            initialize=0.7,
            bounds=(0.01, 1),
            units=pyunits.dimensionless,
            doc='The porosity of spacer in the ED channels. This is also referred to elsewhere as "void fraction" or "volume parameters"',
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
            self.config.property_package.ion_set
            | self.config.property_package.solute_set,
            initialize=1e-10,
            bounds=(1e-16, 1e-6),
            units=pyunits.meter**2 * pyunits.second**-1,
            doc="Solute (ionic and neutral) diffusivity in the membrane phase",
        )
        self.ion_trans_number_membrane = Var(
            self.membrane_set,
            self.config.property_package.ion_set,
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
            self.membrane_set,
            initialize=2e-4,
            bounds=(1e-6, 1),
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
        self.total_areal_resistance_x = Var(
            self.flowsheet().time,
            self.diluate.length_domain,
            initialize=1e-2,
            bounds=(0, 100),
            domain=NonNegativeReals,
            units=pyunits.ohm * pyunits.meter**2,
            doc="Total areal resistance of a stack ",
        )
        if self.config.operation_mode == ElectricalOperationMode.Constant_Current:
            self.current_applied = Var(
                self.flowsheet().time,
                initialize=1,
                bounds=(0, 1000),
                units=pyunits.amp,
                doc="Current across a cell-pair or stack, declared under the 'Constant Current' mode only",
            )

        self.current_density_x = Var(
            self.flowsheet().time,
            self.diluate.length_domain,
            initialize=1,
            bounds=(0, 1000),
            units=pyunits.amp * pyunits.meter**-2,
            doc="Current density accross the membrane as a function of the normalized length",
        )

        if self.config.operation_mode == ElectricalOperationMode.Constant_Voltage:
            self.voltage_applied = Var(
                self.flowsheet().time,
                initialize=100,
                bounds=(0, 1000),
                units=pyunits.volt,
                doc="Voltage across a stack, declared under the 'Constant Voltage' mode only",
            )

        self.voltage_x = Var(
            self.flowsheet().time,
            self.diluate.length_domain,
            initialize=100,
            bounds=(0, 1000),
            units=pyunits.volt,
            doc="Voltage across a stack",
        )
        self.current_utilization = Var(
            initialize=1,
            bounds=(0, 1),
            units=pyunits.dimensionless,
            doc="The current utilization including water electro-osmosis and ion diffusion",
        )
        # Performance metrics
        # Note that the power_electrical_x was declared under the diluate block
        self.specific_power_electrical = Var(
            self.flowsheet().time,
            initialize=10,
            bounds=(0, 1000),
            domain=NonNegativeReals,
            units=pyunits.kW * pyunits.hour * pyunits.meter**-3,
            doc="Diluate-volume-flow-rate-specific electrical power consumption",
        )
        self.current_efficiency_x = Var(
            self.flowsheet().time,
            self.diluate.length_domain,
            initialize=0.9,
            bounds=(0, 1 + 1e-10),
            units=pyunits.dimensionless,
            doc="The overall current efficiency for deionization",
        )
        self.recovery_mass_H2O = Var(
            self.flowsheet().time,
            initialize=0.5,
            bounds=(0, 1),
            domain=NonNegativeReals,
            units=pyunits.dimensionless,
            doc="water recovery ratio calculated by mass",
        )
        self.velocity = Var(
            self.flow_channel_set,
            self.flowsheet().time,
            self.diluate.length_domain,
            initialize=0.01,
            bounds=(1e-6, 1e2),
            units=pyunits.meter * pyunits.second**-1,
            doc="Linear velocity of flow",
        )
        # TODO: consider adding more performance as needed.
        self._make_performance()

    def _make_performance(self):
        if (
            self.config.has_nonohmic_potential_membrane
            or self.config.has_Nernst_diffusion_layer
        ):
            self.conc_mem_surf_mol_x = Var(
                self.membrane_set,
                self.electrode_side_set,
                self.flowsheet().time,
                self.diluate.length_domain,
                self.config.property_package.ion_set,
                initialize=500,
                bounds=(0, 1e5),
                units=pyunits.mol * pyunits.meter**-3,
                doc="Membane surface concentration of components",
            )
        if self.config.has_nonohmic_potential_membrane:
            self._make_performance_nonohm_mem()
        if self.config.has_Nernst_diffusion_layer:
            self._make_performance_dl_polarization()
        if (
            not self.config.pressure_drop_method == PressureDropMethod.none
        ) and self.config.has_pressure_change:
            self._pressure_drop_calculation()

            @self.Constraint(
                self.flow_channel_set,
                self.flowsheet().time,
                self.difference_elements,
                doc="Express deltaP_term by the calculated pressure drop data.",
            )
            def eq_deltaP(self, c, t, x):
                if x not in self.difference_elements:
                    self.concentrate.deltaP[t, x].fix(0)
                    self.diluate.deltaP[t, x].fix(0)
                else:
                    if c == "concentrate":
                        return (
                            self.concentrate.deltaP[t, x] == -self.pressure_dx[c, t, x]
                        )
                    elif c == "diluate":
                        return self.diluate.deltaP[t, x] == -self.pressure_dx[c, t, x]

        elif self.config.pressure_drop_method == PressureDropMethod.none and (
            not self.config.has_pressure_change
        ):
            pass
        else:
            raise ConfigurationError(
                "A valid (not none) pressure_drop_method and has_pressure_change being True "
                "must be both used or unused at the same time. "
            )
        # To require H2O must be in the component
        if "H2O" not in self.component_set:
            raise ConfigurationError(
                "Property Package MUST contain 'H2O' as a component"
            )

        # -------- Add constraints ---------

        @self.Constraint(
            self.flow_channel_set,
            self.flowsheet().time,
            self.diluate.length_domain,
            doc="Calculate flow velocity in each channel",
        )
        def eq_get_velocity(self, c, t, x):
            if c == "concentrate":
                flow_vol_ref = self.concentrate.properties[t, x].flow_vol_phase["Liq"]
            elif c == "diluate":
                flow_vol_ref = self.diluate.properties[t, x].flow_vol_phase["Liq"]
            return (
                self.velocity[c, t, x]
                * self.cell_width
                * self.channel_height
                * self.spacer_porosity
                * self.cell_pair_num
                == flow_vol_ref
            )

        @self.Constraint(
            self.flowsheet().time,
            self.diluate.length_domain,
            doc="Calculate the total areal resistance of a stack",
        )
        def eq_get_total_areal_resistance_x(self, t, x):
            if self.config.has_Nernst_diffusion_layer:
                return self.total_areal_resistance_x[t, x] == (
                    (
                        self.membrane_areal_resistance["aem"]
                        + self.membrane_areal_resistance["cem"]
                        + (
                            self.channel_height
                            - self.dl_thickness_x["cem", "cathode_left", t, x]
                            - self.dl_thickness_x["aem", "anode_right", t, x]
                        )
                        * self.concentrate.properties[t, x].elec_cond_phase["Liq"] ** -1
                        + (
                            self.channel_height
                            - self.dl_thickness_x["cem", "anode_right", t, x]
                            - self.dl_thickness_x["aem", "cathode_left", t, x]
                        )
                        * self.diluate.properties[t, x].elec_cond_phase["Liq"] ** -1
                    )
                    * self.cell_pair_num
                    + self.electrodes_resistance
                )
            else:
                return self.total_areal_resistance_x[t, x] == (
                    (
                        self.membrane_areal_resistance["aem"]
                        + self.membrane_areal_resistance["cem"]
                        + self.channel_height
                        * (
                            self.concentrate.properties[t, x].elec_cond_phase["Liq"]
                            ** -1
                            + self.diluate.properties[t, x].elec_cond_phase["Liq"] ** -1
                        )
                    )
                    * self.cell_pair_num
                    + self.electrodes_resistance
                )

        @self.Constraint(
            self.flowsheet().time,
            self.diluate.length_domain,
            doc="calcualte current density from the electrical input",
        )
        def eq_get_current_density(self, t, x):
            if self.config.operation_mode == ElectricalOperationMode.Constant_Current:
                return (
                    self.current_density_x[t, x] * self.cell_width * self.diluate.length
                    == self.current_applied[t]
                )
            else:
                if self.config.has_nonohmic_potential_membrane:
                    if self.config.has_Nernst_diffusion_layer:
                        return (
                            self.current_density_x[t, x]
                            * self.total_areal_resistance_x[t, x]
                            + (
                                self.potential_ohm_dl_x["cem", t, x]
                                + self.potential_ohm_dl_x["aem", t, x]
                                + self.potential_nonohm_dl_x["cem", t, x]
                                + self.potential_nonohm_dl_x["aem", t, x]
                                + self.potential_nonohm_membrane_x["cem", t, x]
                                + self.potential_nonohm_membrane_x["aem", t, x]
                            )
                            * self.cell_pair_num
                            == self.voltage_applied[t]
                        )
                    else:
                        return (
                            self.current_density_x[t, x]
                            * self.total_areal_resistance_x[t, x]
                            + (
                                self.potential_nonohm_membrane_x["cem", t, x]
                                + self.potential_nonohm_membrane_x["aem", t, x]
                            )
                            * self.cell_pair_num
                            == self.voltage_applied[t]
                        )
                else:
                    if self.config.has_Nernst_diffusion_layer:
                        return (
                            self.current_density_x[t, x]
                            * self.total_areal_resistance_x[t, x]
                            + (
                                self.potential_ohm_dl_x["cem", t, x]
                                + self.potential_ohm_dl_x["aem", t, x]
                                + self.potential_nonohm_dl_x["cem", t, x]
                                + self.potential_nonohm_dl_x["aem", t, x]
                            )
                            * self.cell_pair_num
                            == self.voltage_applied[t]
                        )
                    else:
                        return (
                            self.current_density_x[t, x]
                            * self.total_areal_resistance_x[t, x]
                            == self.voltage_applied[t]
                        )

        @self.Constraint(
            self.flowsheet().time,
            self.diluate.length_domain,
            doc="calcualte length_indexed voltage",
        )
        def eq_get_voltage_x(self, t, x):
            if self.config.has_nonohmic_potential_membrane:
                if self.config.has_Nernst_diffusion_layer:
                    return (
                        self.current_density_x[t, x]
                        * self.total_areal_resistance_x[t, x]
                        + (
                            self.potential_ohm_dl_x["cem", t, x]
                            + self.potential_ohm_dl_x["aem", t, x]
                            + self.potential_nonohm_dl_x["cem", t, x]
                            + self.potential_nonohm_dl_x["aem", t, x]
                            + self.potential_nonohm_membrane_x["cem", t, x]
                            + self.potential_nonohm_membrane_x["aem", t, x]
                        )
                        * self.cell_pair_num
                        == self.voltage_x[t, x]
                    )
                else:
                    return (
                        self.current_density_x[t, x]
                        * self.total_areal_resistance_x[t, x]
                        + (
                            self.potential_nonohm_membrane_x["cem", t, x]
                            + self.potential_nonohm_membrane_x["aem", t, x]
                        )
                        * self.cell_pair_num
                        == self.voltage_x[t, x]
                    )
            else:
                if self.config.has_Nernst_diffusion_layer:
                    return (
                        self.current_density_x[t, x]
                        * self.total_areal_resistance_x[t, x]
                        + (
                            self.potential_ohm_dl_x["cem", t, x]
                            + self.potential_ohm_dl_x["aem", t, x]
                            + self.potential_nonohm_dl_x["cem", t, x]
                            + self.potential_nonohm_dl_x["aem", t, x]
                        )
                        * self.cell_pair_num
                        == self.voltage_x[t, x]
                    )
                else:
                    return (
                        self.current_density_x[t, x]
                        * self.total_areal_resistance_x[t, x]
                        == self.voltage_x[t, x]
                    )

        # Mass Transfer for the Diluate
        @self.Constraint(
            self.flowsheet().time,
            self.diluate.length_domain,
            self.config.property_package.phase_list,
            self.config.property_package.component_list,
            doc="Mass transfer term diluate channel",
        )
        def eq_mass_transfer_term_diluate(self, t, x, p, j):

            if j == "H2O":
                if self.config.has_Nernst_diffusion_layer:
                    return (
                        self.diluate.mass_transfer_term[t, x, p, j]
                        == (
                            -self.cell_width
                            * (
                                self.water_trans_number_membrane["cem"]
                                + self.water_trans_number_membrane["aem"]
                            )
                            * (
                                self.current_density_x[t, x]
                                / Constants.faraday_constant
                            )
                            - self.cell_width
                            * self.diluate.properties[
                                0, self.diluate.length_domain.first()
                            ].dens_mass_solvent
                            / self.config.property_package.mw_comp[j]
                            * (
                                self.water_permeability_membrane["cem"]
                                + self.water_permeability_membrane["aem"]
                            )
                            * (
                                self.concentrate.properties[t, x].pressure_osm_phase[p]
                                * (
                                    1
                                    + self.current_density_x[t, x]
                                    / self.current_dens_lim_x[t, x]
                                )
                                - self.diluate.properties[t, x].pressure_osm_phase[p]
                                * (
                                    1
                                    - self.current_density_x[t, x]
                                    / self.current_dens_lim_x[t, x]
                                )
                            )
                        )
                        * self.cell_pair_num
                    )
                else:
                    return (
                        self.diluate.mass_transfer_term[t, x, p, j]
                        == (
                            -self.cell_width
                            * (
                                self.water_trans_number_membrane["cem"]
                                + self.water_trans_number_membrane["aem"]
                            )
                            * (
                                self.current_density_x[t, x]
                                / Constants.faraday_constant
                            )
                            - self.cell_width
                            * self.diluate.properties[
                                0, self.diluate.length_domain.first()
                            ].dens_mass_solvent
                            / self.config.property_package.mw_comp[j]
                            * (
                                self.water_permeability_membrane["cem"]
                                + self.water_permeability_membrane["aem"]
                            )
                            * (
                                self.concentrate.properties[t, x].pressure_osm_phase[p]
                                - self.diluate.properties[t, x].pressure_osm_phase[p]
                            )
                        )
                        * self.cell_pair_num
                    )
            elif j in self.config.property_package.ion_set:
                if self.config.has_Nernst_diffusion_layer:
                    return (
                        self.diluate.mass_transfer_term[t, x, p, j]
                        == (
                            -self.cell_width
                            * (
                                self.ion_trans_number_membrane["cem", j]
                                - self.ion_trans_number_membrane["aem", j]
                            )
                            * (self.current_utilization * self.current_density_x[t, x])
                            / (
                                self.config.property_package.charge_comp[j]
                                * Constants.faraday_constant
                            )
                            + self.cell_width
                            * (
                                self.solute_diffusivity_membrane["cem", j]
                                / self.membrane_thickness["cem"]
                                * (
                                    self.conc_mem_surf_mol_x[
                                        "cem", "cathode_left", t, x, j
                                    ]
                                    - self.conc_mem_surf_mol_x[
                                        "cem", "anode_right", t, x, j
                                    ]
                                )
                                + self.solute_diffusivity_membrane["aem", j]
                                / self.membrane_thickness["aem"]
                                * (
                                    self.conc_mem_surf_mol_x[
                                        "aem", "anode_right", t, x, j
                                    ]
                                    - self.conc_mem_surf_mol_x[
                                        "aem", "cathode_left", t, x, j
                                    ]
                                )
                            )
                        )
                        * self.cell_pair_num
                    )
                else:
                    return (
                        self.diluate.mass_transfer_term[t, x, p, j]
                        == (
                            -self.cell_width
                            * (
                                self.ion_trans_number_membrane["cem", j]
                                - self.ion_trans_number_membrane["aem", j]
                            )
                            * (self.current_utilization * self.current_density_x[t, x])
                            / (
                                self.config.property_package.charge_comp[j]
                                * Constants.faraday_constant
                            )
                            + self.cell_width
                            * (
                                self.solute_diffusivity_membrane["cem", j]
                                / self.membrane_thickness["cem"]
                                + self.solute_diffusivity_membrane["aem", j]
                                / self.membrane_thickness["aem"]
                            )
                            * (
                                self.concentrate.properties[t, x].conc_mol_phase_comp[
                                    p, j
                                ]
                                - self.diluate.properties[t, x].conc_mol_phase_comp[
                                    p, j
                                ]
                            )
                        )
                        * self.cell_pair_num
                    )
            else:
                return (
                    self.diluate.mass_transfer_term[t, x, p, j]
                    == (
                        self.cell_width
                        * (
                            self.solute_diffusivity_membrane["cem", j]
                            / self.membrane_thickness["cem"]
                            + self.solute_diffusivity_membrane["aem", j]
                            / self.membrane_thickness["aem"]
                        )
                        * (
                            self.concentrate.properties[t, x].conc_mol_phase_comp[p, j]
                            - self.diluate.properties[t, x].conc_mol_phase_comp[p, j]
                        )
                    )
                    * self.cell_pair_num
                )

        # Mass Transfer for the Concentrate
        @self.Constraint(
            self.flowsheet().config.time,
            self.diluate.length_domain,
            self.config.property_package.phase_list,
            self.config.property_package.component_list,
            doc="Mass transfer term concentrate channel",
        )
        def eq_mass_transfer_term_concentrate(self, t, x, p, j):

            return (
                self.concentrate.mass_transfer_term[t, x, p, j]
                == -self.diluate.mass_transfer_term[t, x, p, j]
            )

        # Performance Metrics calculation
        @self.Constraint(
            self.flowsheet().config.time,
            self.diluate.length_domain,
            doc="Electrical power consumption of a stack",
        )
        def eq_power_electrical(self, t, x):
            if x == self.diluate.length_domain.first():
                self.diluate.power_electrical_x[t, x].fix(0)
                return Constraint.Skip
            else:
                return (
                    self.diluate.Dpower_electrical_Dx[t, x]
                    == self.voltage_x[t, x]
                    * self.current_density_x[t, x]
                    * self.cell_width
                    * self.diluate.length
                )

        @self.Constraint(
            self.flowsheet().config.time,
            doc="Diluate_volume_flow_rate_specific electrical power consumption of a stack",
        )
        def eq_specific_power_electrical(self, t):
            return (
                pyunits.convert(
                    self.specific_power_electrical[t],
                    pyunits.watt * pyunits.second * pyunits.meter**-3,
                )
                * self.diluate.properties[
                    t, self.diluate.length_domain.last()
                ].flow_vol_phase["Liq"]
                == self.diluate.power_electrical_x[t, self.diluate.length_domain.last()]
            )

        @self.Constraint(
            self.flowsheet().config.time,
            self.diluate.length_domain,
            self.config.property_package.phase_list,
            doc="Overall current efficiency evaluation",
        )
        def eq_current_efficiency_x(self, t, x, p):

            return (
                self.current_efficiency_x[t, x]
                * self.current_density_x[t, x]
                * self.cell_width
                * self.cell_pair_num
                == -sum(
                    self.diluate.mass_transfer_term[t, x, p, j]
                    * self.config.property_package.charge_comp[j]
                    for j in self.cation_set
                )
                * Constants.faraday_constant
            )

        @self.Constraint(
            self.flowsheet().config.time,
            doc="Water recovery by mass",
        )
        def eq_recovery_mass_H2O(self, t):
            return (
                self.recovery_mass_H2O[t]
                * (
                    self.diluate.properties[
                        t, self.diluate.length_domain.first()
                    ].flow_mass_phase_comp["Liq", "H2O"]
                    + self.concentrate.properties[
                        t, self.diluate.length_domain.first()
                    ].flow_mass_phase_comp["Liq", "H2O"]
                )
                == self.diluate.properties[
                    t, self.diluate.length_domain.last()
                ].flow_mass_phase_comp["Liq", "H2O"]
            )

    def _make_performance_nonohm_mem(self):

        self.potential_nonohm_membrane_x = Var(
            self.membrane_set,
            self.flowsheet().time,
            self.diluate.length_domain,
            initialize=0.01,  # to inspect
            bounds=(-50, 50),
            units=pyunits.volt,
            doc="Nonohmic potential across a membane",
        )

        @self.Constraint(
            self.membrane_set,
            self.electrode_side_set,
            self.flowsheet().time,
            self.diluate.length_domain,
            self.ion_set,
            doc="calcualte current density from the electrical input",
        )
        def eq_set_surface_conc(self, mem, side, t, x, j):
            if not self.config.has_Nernst_diffusion_layer:
                if (mem == "cem" and side == "cathode_left") or (
                    mem == "aem" and side == "anode_right"
                ):
                    return (
                        self.conc_mem_surf_mol_x[mem, side, t, x, j]
                        == self.concentrate.properties[t, x].conc_mol_phase_comp[
                            "Liq", j
                        ]
                    )
                else:
                    return (
                        self.conc_mem_surf_mol_x[mem, side, t, x, j]
                        == self.diluate.properties[t, x].conc_mol_phase_comp["Liq", j]
                    )
            else:
                return Constraint.Skip

        @self.Constraint(
            self.membrane_set,
            self.flowsheet().time,
            self.diluate.length_domain,
            doc="Calculate the total non-ohmic potential across an iem; this takes account of diffusion and Donnan Potentials",
        )
        def eq_potential_nonohm_membrane_x(self, mem, t, x):
            if (
                not self.config.has_Nernst_diffusion_layer
            ) and x == self.diluate.length_domain.first():
                self.potential_nonohm_membrane_x[mem, t, x].fix(0)
                return Constraint.Skip

            return self.potential_nonohm_membrane_x[mem, t, x] == (
                Constants.gas_constant
                * self.diluate.properties[t, x].temperature
                / Constants.faraday_constant
                * (
                    sum(
                        self.ion_trans_number_membrane[mem, j]
                        / self.config.property_package.charge_comp[j]
                        for j in self.cation_set
                    )
                    + sum(
                        self.ion_trans_number_membrane[mem, j]
                        / self.config.property_package.charge_comp[j]
                        for j in self.anion_set
                    )
                )
                * log(
                    sum(
                        self.conc_mem_surf_mol_x[mem, "cathode_left", t, x, j]
                        for j in self.ion_set
                    )
                    / sum(
                        self.conc_mem_surf_mol_x[mem, "anode_right", t, x, j]
                        for j in self.ion_set
                    )
                )
            )  # Note: the log argument is taken as *total ion concentration* while in the original equation as *electrolyte concentration*
            # This is mathematically valid as the ratio eliminate the difference and electro-neutrality stands.

    def _make_performance_dl_polarization(self):

        self.current_dens_lim_x = Var(
            self.flowsheet().time,
            self.diluate.length_domain,
            initialize=500,
            bounds=(0, 10000),
            units=pyunits.amp * pyunits.meter**-2,
            doc="Limiting Current Density accross the membrane as a function of the normalized length",
        )

        self.potential_nonohm_dl_x = Var(
            self.membrane_set,
            self.flowsheet().time,
            self.diluate.length_domain,
            initialize=0.01,  # to inspect
            bounds=(-50, 50),
            units=pyunits.volt,
            doc="Nonohmic potential in two diffusion layers on the two sides of a membrane",
        )

        self.potential_ohm_dl_x = Var(
            self.membrane_set,
            self.flowsheet().time,
            self.diluate.length_domain,
            initialize=0.01,  # to inspect
            bounds=(0, 50),
            units=pyunits.volt,
            doc="Ohmic potential in two diffusion layers on the two sides of a membrane",
        )

        self.dl_thickness_x = Var(
            self.membrane_set,
            self.electrode_side_set,
            self.flowsheet().time,
            self.diluate.length_domain,
            initialize=0.0005,
            bounds=(0, 1e-2),
            units=pyunits.m,
            doc="Thickness of the diffusion layer",
        )

        if (
            self.config.limiting_current_density_method
            == LimitingCurrentDensityMethod.InitialValue
        ):

            @self.Constraint(
                self.flowsheet().time,
                self.diluate.length_domain,
                doc="Calculate length-indexed limiting current density",
            )
            def eq_current_dens_lim_x(self, t, x):
                return self.current_dens_lim_x[t, x] == (
                    self.config.limiting_current_density_data
                    * pyunits.amp
                    * pyunits.meter**-2
                    / sum(
                        self.diluate.properties[t, 0].conc_mol_phase_comp["Liq", j]
                        for j in self.cation_set
                    )
                    * sum(
                        self.diluate.properties[t, x].conc_mol_phase_comp["Liq", j]
                        for j in self.cation_set
                    )
                )

        elif (
            self.config.limiting_current_density_method
            == LimitingCurrentDensityMethod.Empirical
        ):
            self.param_b = Param(
                initialize=0.5,
                units=pyunits.dimensionless,
                doc="empirical parameter b to calculate limiting current density",
            )
            self.param_a = Param(
                initialize=25,
                units=pyunits.coulomb
                * pyunits.mol**-1
                * pyunits.meter ** (1 - self.param_b)
                * pyunits.second ** (self.param_b - 1),
                doc="empirical parameter a to calculate limiting current density",
            )

            @self.Constraint(
                self.flowsheet().time,
                self.diluate.length_domain,
                doc="Calculate length-indexed limiting current density",
            )
            def eq_current_dens_lim_x(self, t, x):

                return self.current_dens_lim_x[t, x] == self.param_a * self.velocity[
                    "diluate", t, x
                ] ** self.param_b * sum(
                    self.config.property_package.charge_comp[j]
                    * self.diluate.properties[t, x].conc_mol_phase_comp["Liq", j]
                    for j in self.cation_set
                )

        elif (
            self.config.limiting_current_density_method
            == LimitingCurrentDensityMethod.Theoretical
        ):
            self._get_fluid_dimensionless_quantities()

            @self.Constraint(
                self.flowsheet().time,
                self.diluate.length_domain,
                doc="Calculate length-indexed limiting current density",
            )
            def eq_current_dens_lim_x(self, t, x):
                return self.current_dens_lim_x[t, x] == self.N_Sh[
                    "diluate", t, x
                ] * self.diffus_mass * self.hydraulic_diameter**-1 * Constants.faraday_constant * (
                    sum(
                        self.ion_trans_number_membrane["cem", j]
                        / self.config.property_package.charge_comp[j]
                        for j in self.cation_set
                    )
                    - sum(
                        self.diluate.properties[t, x].trans_num_phase_comp["Liq", j]
                        / self.config.property_package.charge_comp[j]
                        for j in self.cation_set
                    )
                ) ** -1 * sum(
                    self.config.property_package.charge_comp[j]
                    * self.diluate.properties[t, x].conc_mol_phase_comp["Liq", j]
                    for j in self.cation_set
                )

        @self.Constraint(
            self.membrane_set,
            self.electrode_side_set,
            self.flowsheet().time,
            self.diluate.length_domain,
            self.config.property_package.ion_set,
            doc="Establish relationship between interfacial concentration polarization ratio and current density",
        )
        def eq_conc_polarization_ratio(self, mem, side, t, x, j):
            if (mem == "cem" and side == "cathode_left") or (
                mem == "aem" and side == "anode_right"
            ):
                return self.conc_mem_surf_mol_x[
                    mem, side, t, x, j
                ] / self.concentrate.properties[t, x].conc_mol_phase_comp["Liq", j] == (
                    1 + self.current_density_x[t, x] / self.current_dens_lim_x[t, x]
                )
            else:
                return self.conc_mem_surf_mol_x[
                    mem, side, t, x, j
                ] / self.diluate.properties[t, x].conc_mol_phase_comp["Liq", j] == (
                    1 - self.current_density_x[t, x] / self.current_dens_lim_x[t, x]
                )

        @self.Constraint(
            self.membrane_set,
            self.flowsheet().time,
            self.diluate.length_domain,
            doc="Calculate the total non-ohmic potential across the two diffusion layers of an iem.",
        )
        def eq_potential_nonohm_dl(self, mem, t, x):
            if mem == "cem":
                return self.potential_nonohm_dl_x[mem, t, x] == (
                    Constants.gas_constant
                    * self.diluate.properties[t, x].temperature
                    / Constants.faraday_constant
                    * (
                        sum(
                            self.diluate.properties[t, x].trans_num_phase_comp["Liq", j]
                            / self.config.property_package.charge_comp[j]
                            for j in self.cation_set
                        )
                        + sum(
                            self.diluate.properties[t, x].trans_num_phase_comp["Liq", j]
                            / self.config.property_package.charge_comp[j]
                            for j in self.anion_set
                        )
                    )
                    * log(
                        sum(
                            self.conc_mem_surf_mol_x[mem, "anode_right", t, x, j]
                            for j in self.ion_set
                        )
                        * sum(
                            self.concentrate.properties[t, x].conc_mol_phase_comp[
                                "Liq", j
                            ]
                            for j in self.ion_set
                        )
                        * sum(
                            self.conc_mem_surf_mol_x[mem, "cathode_left", t, x, j]
                            for j in self.ion_set
                        )
                        ** -1
                        * sum(
                            self.diluate.properties[t, x].conc_mol_phase_comp["Liq", j]
                            for j in self.ion_set
                        )
                        ** -1
                    )
                )
            else:
                return self.potential_nonohm_dl_x[mem, t, x] == (
                    Constants.gas_constant
                    * self.diluate.properties[t, x].temperature
                    / Constants.faraday_constant
                    * (
                        sum(
                            self.diluate.properties[t, x].trans_num_phase_comp["Liq", j]
                            / self.config.property_package.charge_comp[j]
                            for j in self.cation_set
                        )
                        + sum(
                            self.diluate.properties[t, x].trans_num_phase_comp["Liq", j]
                            / self.config.property_package.charge_comp[j]
                            for j in self.anion_set
                        )
                    )
                    * log(
                        sum(
                            self.conc_mem_surf_mol_x[mem, "anode_right", t, x, j]
                            for j in self.ion_set
                        )
                        * sum(
                            self.diluate.properties[t, x].conc_mol_phase_comp["Liq", j]
                            for j in self.ion_set
                        )
                        * sum(
                            self.conc_mem_surf_mol_x[mem, "cathode_left", t, x, j]
                            for j in self.ion_set
                        )
                        ** -1
                        * sum(
                            self.concentrate.properties[t, x].conc_mol_phase_comp[
                                "Liq", j
                            ]
                            for j in self.ion_set
                        )
                        ** -1
                    )
                )

        @self.Constraint(
            self.membrane_set,
            self.flowsheet().time,
            self.diluate.length_domain,
            doc="Calculate the total ohmic potential across the two diffusion layers of an iem.",
        )
        def eq_potential_ohm_dl_x(self, mem, t, x):
            if mem == "cem":
                return self.potential_ohm_dl_x[mem, t, x] == (
                    Constants.faraday_constant
                    * (
                        sum(
                            self.diluate.properties[t, x].diffus_phase_comp["Liq", j]
                            ** -1
                            for j in self.ion_set
                        )
                        ** -1
                        * len(self.ion_set)
                    )
                    * (
                        sum(
                            self.ion_trans_number_membrane[mem, j]
                            / self.config.property_package.charge_comp[j]
                            for j in self.cation_set
                        )
                        - sum(
                            self.diluate.properties[t, x].trans_num_phase_comp["Liq", j]
                            / self.config.property_package.charge_comp[j]
                            for j in self.cation_set
                        )
                    )
                    ** -1
                    * self.diluate.properties[t, x].equiv_conductivity_phase["Liq"]
                    ** -1
                    * log(
                        sum(
                            self.conc_mem_surf_mol_x[mem, "anode_right", t, x, j]
                            for j in self.ion_set
                        )
                        ** -1
                        * sum(
                            self.concentrate.properties[t, x].conc_mol_phase_comp[
                                "Liq", j
                            ]
                            for j in self.ion_set
                        )
                        ** -1
                        * sum(
                            self.conc_mem_surf_mol_x[mem, "cathode_left", t, x, j]
                            for j in self.ion_set
                        )
                        * sum(
                            self.diluate.properties[t, x].conc_mol_phase_comp["Liq", j]
                            for j in self.ion_set
                        )
                    )
                )
            else:
                return self.potential_ohm_dl_x[mem, t, x] == (
                    -Constants.faraday_constant
                    * (
                        sum(
                            self.diluate.properties[t, x].diffus_phase_comp["Liq", j]
                            ** -1
                            for j in self.ion_set
                        )
                        ** -1
                        * len(self.ion_set)
                    )
                    * (
                        sum(
                            self.ion_trans_number_membrane[mem, j]
                            / self.config.property_package.charge_comp[j]
                            for j in self.cation_set
                        )
                        - sum(
                            self.diluate.properties[t, x].trans_num_phase_comp["Liq", j]
                            / self.config.property_package.charge_comp[j]
                            for j in self.cation_set
                        )
                    )
                    ** -1
                    * self.diluate.properties[t, x].equiv_conductivity_phase["Liq"]
                    ** -1
                    * log(
                        sum(
                            self.conc_mem_surf_mol_x[mem, "anode_right", t, x, j]
                            for j in self.ion_set
                        )
                        * sum(
                            self.diluate.properties[t, x].conc_mol_phase_comp["Liq", j]
                            for j in self.ion_set
                        )
                        * sum(
                            self.conc_mem_surf_mol_x[mem, "cathode_left", t, x, j]
                            for j in self.ion_set
                        )
                        ** -1
                        * sum(
                            self.concentrate.properties[t, x].conc_mol_phase_comp[
                                "Liq", j
                            ]
                            for j in self.ion_set
                        )
                        ** -1
                    )
                )

        @self.Constraint(
            self.membrane_set,
            self.electrode_side_set,
            self.flowsheet().time,
            self.diluate.length_domain,
            doc="Calculate the total non-ohmic potential across the two diffusion layers of an iem.",
        )
        def eq_dl_thickness(self, mem, side, t, x):
            if mem == "cem" and side == "cathode_left":
                return self.dl_thickness_x[mem, side, t, x] == (
                    Constants.faraday_constant
                    * (
                        sum(
                            self.diluate.properties[t, x].diffus_phase_comp["Liq", j]
                            ** -1
                            for j in self.ion_set
                        )
                        ** -1
                        * len(self.ion_set)
                    )
                    * (
                        sum(
                            self.ion_trans_number_membrane[mem, j]
                            / self.config.property_package.charge_comp[j]
                            for j in self.cation_set
                        )
                        - sum(
                            self.diluate.properties[t, x].trans_num_phase_comp["Liq", j]
                            / self.config.property_package.charge_comp[j]
                            for j in self.cation_set
                        )
                    )
                    ** -1
                    * sum(
                        self.concentrate.properties[t, x].conc_mol_phase_comp["Liq", j]
                        for j in self.cation_set
                    )
                    * self.current_dens_lim_x[t, x] ** -1
                )
            elif mem == "cem" and side == "anode_right":
                return self.dl_thickness_x[mem, side, t, x] == (
                    Constants.faraday_constant
                    * (
                        sum(
                            self.diluate.properties[t, x].diffus_phase_comp["Liq", j]
                            ** -1
                            for j in self.ion_set
                        )
                        ** -1
                        * len(self.ion_set)
                    )
                    * (
                        sum(
                            self.ion_trans_number_membrane[mem, j]
                            / self.config.property_package.charge_comp[j]
                            for j in self.cation_set
                        )
                        - sum(
                            self.diluate.properties[t, x].trans_num_phase_comp["Liq", j]
                            / self.config.property_package.charge_comp[j]
                            for j in self.cation_set
                        )
                    )
                    ** -1
                    * sum(
                        self.diluate.properties[t, x].conc_mol_phase_comp["Liq", j]
                        for j in self.cation_set
                    )
                    * self.current_dens_lim_x[t, x] ** -1
                )
            elif mem == "aem" and side == "cathode_left":
                return self.dl_thickness_x[mem, side, t, x] == (
                    -Constants.faraday_constant
                    * (
                        sum(
                            self.diluate.properties[t, x].diffus_phase_comp["Liq", j]
                            ** -1
                            for j in self.ion_set
                        )
                        ** -1
                        * len(self.ion_set)
                    )
                    * (
                        sum(
                            self.ion_trans_number_membrane[mem, j]
                            / self.config.property_package.charge_comp[j]
                            for j in self.cation_set
                        )
                        - sum(
                            self.diluate.properties[t, x].trans_num_phase_comp["Liq", j]
                            / self.config.property_package.charge_comp[j]
                            for j in self.cation_set
                        )
                    )
                    ** -1
                    * sum(
                        self.diluate.properties[t, x].conc_mol_phase_comp["Liq", j]
                        for j in self.cation_set
                    )
                    * self.current_dens_lim_x[t, x] ** -1
                )
            else:
                return self.dl_thickness_x[mem, side, t, x] == (
                    -Constants.faraday_constant
                    * (
                        sum(
                            self.diluate.properties[t, x].diffus_phase_comp["Liq", j]
                            ** -1
                            for j in self.ion_set
                        )
                        ** -1
                        * len(self.ion_set)
                    )
                    * (
                        sum(
                            self.ion_trans_number_membrane[mem, j]
                            / self.config.property_package.charge_comp[j]
                            for j in self.cation_set
                        )
                        - sum(
                            self.diluate.properties[t, x].trans_num_phase_comp["Liq", j]
                            / self.config.property_package.charge_comp[j]
                            for j in self.cation_set
                        )
                    )
                    ** -1
                    * sum(
                        self.concentrate.properties[t, x].conc_mol_phase_comp["Liq", j]
                        for j in self.cation_set
                    )
                    * self.current_dens_lim_x[t, x] ** -1
                )

    def _get_fluid_dimensionless_quantities(self):
        self.diffus_mass = Var(
            initialize=1e-9,
            bounds=(1e-16, 1e-6),
            units=pyunits.meter**2 * pyunits.second**-1,
            doc="The mass diffusivity of the solute as molecules (not individual ions)",
        )
        self.hydraulic_diameter = Var(
            initialize=1e-3,
            bounds=(0, None),
            units=pyunits.meter,
            doc="The hydraulic diameter of the channel",
        )
        self.N_Re = Var(
            self.flow_channel_set,
            self.flowsheet().time,
            self.diluate.length_domain,
            initialize=50,
            bounds=(0, None),
            units=pyunits.dimensionless,
            doc="Reynolds Number",
        )
        self.N_Sc = Var(
            self.flow_channel_set,
            self.flowsheet().time,
            self.diluate.length_domain,
            initialize=2000,
            bounds=(0, None),
            units=pyunits.dimensionless,
            doc="Schmidt Number",
        )
        self.N_Sh = Var(
            self.flow_channel_set,
            self.flowsheet().time,
            self.diluate.length_domain,
            initialize=100,
            bounds=(0, None),
            units=pyunits.dimensionless,
            doc="Sherwood Number",
        )

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
                        * self.spacer_porosity
                        * (self.channel_height + self.cell_width) ** -1
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
            self.flow_channel_set,
            self.flowsheet().time,
            self.diluate.length_domain,
            doc="To calculate Re",
        )
        def eq_Re(self, c, t, x):
            if c == "concentrate":
                viscosity_ref = self.concentrate.properties[t, x].visc_d_phase["Liq"]
            elif c == "diluate":
                viscosity_ref = self.diluate.properties[t, x].visc_d_phase["Liq"]
            return (
                self.N_Re[c, t, x]
                == self.dens_mass
                * self.velocity[c, t, x]
                * self.hydraulic_diameter
                * viscosity_ref**-1
            )

        @self.Constraint(
            self.flow_channel_set,
            self.flowsheet().time,
            self.diluate.length_domain,
            doc="To calculate Sc",
        )
        def eq_Sc(self, c, t, x):
            if c == "concentrate":
                visc_ref = self.concentrate.properties[t, x].visc_d_phase["Liq"]
                dens_ref = self.concentrate.properties[t, x].dens_mass_phase["Liq"]
            elif c == "diluate":
                visc_ref = self.diluate.properties[t, x].visc_d_phase["Liq"]
                dens_ref = self.diluate.properties[t, x].dens_mass_phase["Liq"]
            return self.N_Sc[c, t, x] == visc_ref * dens_ref**-1 * self.diffus_mass**-1

        @self.Constraint(
            self.flow_channel_set,
            self.flowsheet().time,
            self.diluate.length_domain,
            doc="To calculate Sh",
        )
        def eq_Sh(self, c, t, x):

            return (
                self.N_Sh[c, t, x]
                == 0.29 * self.N_Re[c, t, x] ** 0.5 * self.N_Sc[c, t, x] ** 0.33
            )

    def _pressure_drop_calculation(self):
        self.pressure_dx = Var(
            self.flow_channel_set,
            self.flowsheet().time,
            self.difference_elements,
            initialize=1e4,
            bounds=(0, 1e8),
            units=pyunits.pascal * pyunits.meter**-1,
            doc="pressure drop per unit of length",
        )
        self.pressure_drop_total = Var(
            self.flow_channel_set,
            self.flowsheet().time,
            initialize=1e6,
            bounds=(0, 1e8),
            units=pyunits.pascal,
            doc="pressure drop over an entire ED stack",
        )

        if self.config.pressure_drop_method == PressureDropMethod.experimental:
            _log.warning(
                "Do not forget to FIX the experimental pressure drop value in [Pa/m]!"
            )
        else:  # PressureDropMethod.Darcy_Weisbach is used
            if not (
                self.config.has_Nernst_diffusion_layer
                and self.config.limiting_current_density_method
                == LimitingCurrentDensityMethod.Theoretical
            ):
                self._get_fluid_dimensionless_quantities()

            self.friction_factor = Var(
                self.flow_channel_set,
                self.flowsheet().time,
                self.diluate.length_domain,
                initialize=10,
                bounds=(0, None),
                units=pyunits.dimensionless,
                doc="friction factor of the channel fluid",
            )

            @self.Constraint(
                self.flow_channel_set,
                self.flowsheet().time,
                self.difference_elements,
                doc="To calculate pressure drop per unit length",
            )
            def eq_pressure_dx(self, c, t, x):
                return (
                    self.pressure_dx[c, t, x]
                    == self.dens_mass
                    * self.friction_factor[c, t, x]
                    * self.velocity[c, t, x] ** 2
                    * 0.5
                    * self.hydraulic_diameter**-1
                )

            if self.config.friction_factor_method == FrictionFactorMethod.fixed:
                _log.warning("Do not forget to FIX the Darcy's friction factor value!")
            else:

                @self.Constraint(
                    self.flow_channel_set,
                    self.flowsheet().time,
                    self.diluate.length_domain,
                    doc="To calculate friction factor",
                )
                def eq_friction_factor(self, c, t, x):
                    if (
                        self.config.friction_factor_method
                        == FrictionFactorMethod.Gurreri
                    ):
                        return (
                            self.friction_factor[c, t, x]
                            == 4
                            * 50.6
                            * self.spacer_porosity**-7.06
                            * self.N_Re[c, t, x] ** -1
                        )
                    elif (
                        self.config.friction_factor_method
                        == FrictionFactorMethod.Kuroda
                    ):
                        return (
                            self.friction_factor[c, t, x]
                            == 4
                            * 9.6
                            * self.spacer_porosity**-1
                            * self.N_Re[c, t, x] ** -0.5
                        )

        @self.Constraint(
            self.flow_channel_set,
            self.flowsheet().time,
            doc="To calculate total pressure drop over a stack",
        )
        def eq_pressure_drop_total(self, c, t):
            return self.pressure_drop_total[c, t] == sum(
                self.pressure_dx[c, t, x]
                * self.cell_length
                / len(self.difference_elements)
                for x in self.difference_elements
            )

    # Intialization routines
    def initialize_build(
        self,
        state_args=None,
        outlvl=idaeslog.NOTSET,
        solver=None,
        optarg=None,
        fail_on_warning=False,
        ignore_dof=False,
    ):
        """
        General wrapper for electrodialysis_1D initialization routines

        Keyword Arguments:
            state_args : a dict of arguments to be passed to the property
                         package(s) to provide an initial state for
                         initialization (see documentation of the specific
                         property package) (default = {}).
            outlvl : sets output level of initialization routine
            optarg : solver options dictionary object (default=None)
            solver : str indicating which solver to use during
                     initialization (default = None)
            fail_on_warning : boolean argument to fail or only produce  warning upon unsuccessful solve (default=False)
            ignore_dof : boolean argument to ignore when DOF != 0 (default=False)

        Returns: None
        """
        init_log = idaeslog.getInitLogger(self.name, outlvl, tag="unit")
        solve_log = idaeslog.getSolveLogger(self.name, outlvl, tag="unit")
        # Set solver options
        opt = get_solver(solver, optarg)
        # Set the intial conditions over the 1D length from the state vars
        for k in self.keys():
            for set in self[k].diluate.properties:
                if ("flow_mol_phase_comp" or "flow_mass_phase_comp") not in self[
                    k
                ].diluate.properties[set].define_state_vars():
                    raise ConfigurationError(
                        "Electrodialysis1D unit model requires "
                        "either a 'flow_mol_phase_comp' or 'flow_mass_phase_comp' "
                        "state variable basis to apply the 'propogate_initial_state' method"
                    )
                if "temperature" in self[k].diluate.properties[set].define_state_vars():
                    self[k].diluate.properties[set].temperature = value(
                        self[k].diluate.properties[(0.0, 0.0)].temperature
                    )
                if "pressure" in self[k].diluate.properties[set].define_state_vars():
                    self[k].diluate.properties[set].pressure = value(
                        self[k].diluate.properties[(0.0, 0.0)].pressure
                    )
                if (
                    "flow_mol_phase_comp"
                    in self[k].diluate.properties[set].define_state_vars()
                ):
                    for ind in self[k].diluate.properties[set].flow_mol_phase_comp:
                        self[k].diluate.properties[set].flow_mol_phase_comp[ind] = (
                            value(
                                self[k]
                                .diluate.properties[(0.0, 0.0)]
                                .flow_mol_phase_comp[ind]
                            )
                        )
                if (
                    "flow_mass_phase_comp"
                    in self[k].diluate.properties[set].define_state_vars()
                ):
                    for ind in self[k].diluate.properties[set].flow_mass_phase_comp:
                        self[k].diluate.properties[set].flow_mass_phase_comp[ind] = (
                            value(
                                self[k]
                                .diluate.properties[(0.0, 0.0)]
                                .flow_mass_phase_comp[ind]
                            )
                        )
                if hasattr(self[k], "conc_mem_surf_mol_x"):
                    for mem in self[k].membrane_set:
                        for side in self[k].electrode_side_set:
                            for j in self[k].ion_set:
                                self[k].conc_mem_surf_mol_x[
                                    mem, side, set, j
                                ].set_value(
                                    self[k]
                                    .diluate.properties[set]
                                    .conc_mol_phase_comp["Liq", j]
                                )

            for set in self[k].concentrate.properties:
                if ("flow_mol_phase_comp" or "flow_mass_phase_comp") not in self[
                    k
                ].concentrate.properties[set].define_state_vars():
                    raise ConfigurationError(
                        "Electrodialysis1D unit model requires "
                        "either a 'flow_mol_phase_comp' or 'flow_mass_phase_comp' "
                        "state variable basis to apply the 'propogate_initial_state' method"
                    )
                if (
                    "temperature"
                    in self[k].concentrate.properties[set].define_state_vars()
                ):
                    self[k].concentrate.properties[set].temperature = value(
                        self[k].concentrate.properties[(0.0, 0.0)].temperature
                    )
                if (
                    "pressure"
                    in self[k].concentrate.properties[set].define_state_vars()
                ):
                    self[k].concentrate.properties[set].pressure = value(
                        self[k].concentrate.properties[(0.0, 0.0)].pressure
                    )
                if (
                    "flow_mol_phase_comp"
                    in self[k].concentrate.properties[set].define_state_vars()
                ):
                    for ind in self[k].concentrate.properties[set].flow_mol_phase_comp:
                        self[k].concentrate.properties[set].flow_mol_phase_comp[ind] = (
                            value(
                                self[k]
                                .concentrate.properties[(0.0, 0.0)]
                                .flow_mol_phase_comp[ind]
                            )
                        )
                if (
                    "flow_mass_phase_comp"
                    in self[k].concentrate.properties[set].define_state_vars()
                ):
                    for ind in self[k].concentrate.properties[set].flow_mass_phase_comp:
                        self[k].concentrate.properties[set].flow_mass_phase_comp[
                            ind
                        ] = value(
                            self[k]
                            .concentrate.properties[(0.0, 0.0)]
                            .flow_mass_phase_comp[ind]
                        )
                if hasattr(self[k], "conc_mem_surf_mol_x"):
                    for mem in self[k].membrane_set:
                        for side in self[k].electrode_side_set:
                            for j in self[k].ion_set:
                                self[k].conc_mem_surf_mol_x[
                                    mem, side, set, j
                                ].set_value(
                                    self[k]
                                    .concentrate.properties[set]
                                    .conc_mol_phase_comp["Liq", j]
                                )
                self[k].total_areal_resistance_x[set].set_value(
                    (
                        self[k].membrane_areal_resistance["aem"]
                        + self[k].membrane_areal_resistance["cem"]
                        + self[k].channel_height
                        * (
                            self[k].concentrate.properties[set].elec_cond_phase["Liq"]
                            ** -1
                            + self[k].diluate.properties[set].elec_cond_phase["Liq"]
                            ** -1
                        )
                    )
                    * self[k].cell_pair_num
                    + self[k].electrodes_resistance
                )

        # ---------------------------------------------------------------------
        # Initialize diluate block
        flags_diluate = self.diluate.initialize(
            outlvl=outlvl,
            optarg=optarg,
            solver=solver,
            state_args=state_args,
            hold_state=True,
        )
        init_log.info_high("Initialization Step 1 Complete.")
        # ---------------------------------------------------------------------
        # Initialize concentrate block
        flags_concentrate = self.concentrate.initialize(
            outlvl=outlvl,
            optarg=optarg,
            solver=solver,
            state_args=state_args,
            hold_state=True,
        )
        init_log.info_high("Initialization Step 2 Complete.")
        # ---------------------------------------------------------------------
        # Solve unit
        if not ignore_dof:
            check_dof(self, fail_flag=fail_on_warning, logger=init_log)

        with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
            res = opt.solve(self, tee=slc.tee)
        init_log.info_high("Initialization Step 3 {}.".format(idaeslog.condition(res)))
        check_solve(
            res,
            logger=init_log,
            fail_flag=fail_on_warning,
            checkpoint="Initialization Step 3",
        )
        # ---------------------------------------------------------------------
        # Release state
        self.diluate.release_state(flags_diluate, outlvl)
        self.concentrate.release_state(flags_concentrate, outlvl)
        init_log.info("Initialization Complete: {}".format(idaeslog.condition(res)))

        if not check_optimal_termination(res):
            raise InitializationError(f"Unit model {self.name} failed to initialize")

    def calculate_scaling_factors(self):
        super().calculate_scaling_factors()
        units_meta = self.config.property_package.get_metadata().get_derived_units

        # Scaling factors that user may setup
        # The users are highly encouraged to provide scaling factors for assessable vars below.
        # Not providing these vars will give a warning.
        if iscale.get_scaling_factor(self.cell_width, warning=True) is None:
            iscale.set_scaling_factor(self.cell_width, 1)
        if iscale.get_scaling_factor(self.cell_length, warning=True) is None:
            iscale.set_scaling_factor(self.cell_length, 1)
        if iscale.get_scaling_factor(self.cell_pair_num, warning=True) is None:
            iscale.set_scaling_factor(self.cell_pair_num, 0.1)
        if iscale.get_scaling_factor(self.membrane_thickness, warning=True) is None:
            iscale.set_scaling_factor(self.membrane_thickness, 1e4)

        if (
            iscale.get_scaling_factor(self.membrane_areal_resistance, warning=True)
            is None
        ):
            iscale.set_scaling_factor(self.membrane_areal_resistance, 1e4)

        if (
            iscale.get_scaling_factor(self.solute_diffusivity_membrane, warning=True)
            is None
        ):
            iscale.set_scaling_factor(self.solute_diffusivity_membrane, 1e10)

        if hasattr(self, "diffus_mass") and (
            iscale.get_scaling_factor(self.diffus_mass, warning=True) is None
        ):
            iscale.set_scaling_factor(self.diffus_mass, 1e9)

        if (
            iscale.get_scaling_factor(self.water_permeability_membrane, warning=True)
            is None
        ):
            iscale.set_scaling_factor(self.water_permeability_membrane, 1e4)

        if iscale.get_scaling_factor(self.channel_height, warning=True) is None:
            iscale.set_scaling_factor(self.channel_height, 1e4)
        if iscale.get_scaling_factor(self.spacer_porosity, warning=True) is None:
            iscale.set_scaling_factor(self.spacer_porosity, 1)
        if iscale.get_scaling_factor(self.electrodes_resistance, warning=True) is None:
            iscale.set_scaling_factor(self.electrodes_resistance, 1e4)
        for ind in self.velocity:
            if iscale.get_scaling_factor(self.velocity[ind], warning=False) is None:
                sf = (
                    iscale.get_scaling_factor(
                        self.diluate.properties[0, 0].flow_vol_phase["Liq"]
                    )
                    * iscale.get_scaling_factor(self.cell_width) ** -1
                    * iscale.get_scaling_factor(self.channel_height) ** -1
                    * iscale.get_scaling_factor(self.spacer_porosity) ** -1
                    * iscale.get_scaling_factor(self.cell_pair_num) ** -1
                )
                iscale.set_scaling_factor(self.velocity[ind], sf)
        if hasattr(self, "voltage_applied") and (
            iscale.get_scaling_factor(self.voltage_applied, warning=True) is None
        ):
            iscale.set_scaling_factor(self.voltage_applied, 1)
        if hasattr(self, "current_applied") and (
            iscale.get_scaling_factor(self.current_applied, warning=True) is None
        ):
            iscale.set_scaling_factor(self.current_applied, 1)
        if hasattr(self, "spacer_specific_area") and (
            iscale.get_scaling_factor(self.spacer_specific_area, warning=True) is None
        ):
            iscale.set_scaling_factor(self.spacer_specific_area, 1e-4)
        if hasattr(self, "hydraulic_diameter") and (
            iscale.get_scaling_factor(self.hydraulic_diameter, warning=True) is None
        ):
            iscale.set_scaling_factor(self.hydraulic_diameter, 1e4)
        if hasattr(self, "N_Re"):
            for ind in self.N_Re:
                if iscale.get_scaling_factor(self.N_Re[ind], warning=True) is None:
                    sf = (
                        iscale.get_scaling_factor(self.dens_mass)
                        * iscale.get_scaling_factor(self.velocity[ind])
                        * iscale.get_scaling_factor(self.hydraulic_diameter)
                        * iscale.get_scaling_factor(self.visc_d) ** -1
                    )
                    iscale.set_scaling_factor(self.N_Re[ind], sf)
        if hasattr(self, "N_Sc"):
            for ind in self.N_Sc:
                if iscale.get_scaling_factor(self.N_Sc[ind], warning=True) is None:
                    sf = (
                        iscale.get_scaling_factor(self.visc_d)
                        * iscale.get_scaling_factor(self.dens_mass) ** -1
                        * iscale.get_scaling_factor(self.diffus_mass) ** -1
                    )
                    iscale.set_scaling_factor(self.N_Sc[ind], sf)
        if hasattr(self, "N_Sh"):
            for ind in self.N_Sh:
                if iscale.get_scaling_factor(self.N_Sh[ind], warning=True) is None:
                    sf = (
                        10
                        * iscale.get_scaling_factor(self.N_Re[ind]) ** 0.5
                        * iscale.get_scaling_factor(self.N_Sc[ind]) ** 0.33
                    )
                    iscale.set_scaling_factor(self.N_Sh[ind], sf)
        if hasattr(self, "friction_factor"):
            for ind in self.friction_factor:
                if (
                    iscale.get_scaling_factor(self.friction_factor[ind], warning=True)
                    is None
                ):
                    if self.config.friction_factor_method == FrictionFactorMethod.fixed:
                        sf = 0.1
                    elif (
                        self.config.friction_factor_method
                        == FrictionFactorMethod.Gurreri
                    ):
                        sf = (
                            (4 * 50.6) ** -1
                            * (iscale.get_scaling_factor(self.spacer_porosity)) ** -7.06
                            * iscale.get_scaling_factor(self.N_Re[ind]) ** -1
                        )
                    elif (
                        self.config.friction_factor_method
                        == FrictionFactorMethod.Kuroda
                    ):
                        sf = (4 * 9.6) ** -1 * iscale.get_scaling_factor(
                            self.N_Re[ind]
                        ) ** -0.5
                    iscale.set_scaling_factor(self.friction_factor[ind], sf)
        if hasattr(self, "pressure_dx"):
            for ind in self.pressure_dx:
                if (
                    iscale.get_scaling_factor(self.pressure_dx[ind], warning=True)
                    is None
                ):
                    if (
                        self.config.pressure_drop_method
                        == PressureDropMethod.experimental
                    ):
                        sf = 1e-5
                    else:
                        sf = (
                            iscale.get_scaling_factor(self.dens_mass)
                            * iscale.get_scaling_factor(self.friction_factor[ind])
                            * iscale.get_scaling_factor(self.velocity[ind]) ** 2
                            * 2
                            * iscale.get_scaling_factor(self.hydraulic_diameter) ** -1
                        )
                    iscale.set_scaling_factor(self.pressure_dx[ind], sf)
        if hasattr(self, "pressure_drop_total"):
            for ind in self.pressure_drop_total:
                if (
                    iscale.get_scaling_factor(
                        self.pressure_drop_total[ind], warning=True
                    )
                    is None
                ):
                    sf = iscale.get_scaling_factor(
                        self.pressure_dx[
                            ind[0], ind[1], self.difference_elements.first()
                        ]
                    ) * iscale.get_scaling_factor(self.cell_length)
                    iscale.set_scaling_factor(self.pressure_drop_total[ind], sf)

        # For vars below, the users can choose but not required to provide scaling factors.
        # No warnings if no providing.

        for ind in self.total_areal_resistance_x:
            if (
                iscale.get_scaling_factor(
                    self.total_areal_resistance_x[ind], warning=False
                )
                is None
            ):
                sf = (
                    iscale.get_scaling_factor(self.membrane_areal_resistance) ** 2
                    + iscale.get_scaling_factor(self.channel_height) ** 2
                    * (
                        iscale.get_scaling_factor(
                            self.diluate.properties[ind].elec_cond_phase["Liq"]
                        )
                        ** -2
                        + iscale.get_scaling_factor(
                            self.concentrate.properties[ind].elec_cond_phase["Liq"]
                        )
                        ** -2
                    )
                ) ** 0.5 * iscale.get_scaling_factor(self.cell_pair_num)
                iscale.set_scaling_factor(self.total_areal_resistance_x[ind], sf)

        for ind in self.current_density_x:
            if (
                iscale.get_scaling_factor(self.current_density_x[ind], warning=False)
                is None
            ):
                if (
                    self.config.operation_mode
                    == ElectricalOperationMode.Constant_Current
                ):
                    sf = (
                        iscale.get_scaling_factor(self.current_applied)
                        / iscale.get_scaling_factor(self.cell_width)
                        / iscale.get_scaling_factor(self.cell_length)
                    )
                    iscale.set_scaling_factor(self.current_density_x[ind], sf)
                else:
                    sf = iscale.get_scaling_factor(
                        self.voltage_applied
                    ) / iscale.get_scaling_factor(self.total_areal_resistance_x[ind])
                    iscale.set_scaling_factor(self.current_density_x[ind], sf)

        for ind in self.voltage_x:
            if iscale.get_scaling_factor(self.voltage_x[ind], warning=False) is None:
                sf = iscale.get_scaling_factor(
                    self.current_density_x[ind]
                ) * iscale.get_scaling_factor(self.total_areal_resistance_x[ind])
                iscale.set_scaling_factor(self.voltage_x[ind], sf)
        if iscale.get_scaling_factor(self.spacer_porosity, warning=False) is None:
            iscale.set_scaling_factor(self.spacer_porosity, 1)
        if hasattr(self, "conc_mem_surf_mol_x"):
            for ind in self.conc_mem_surf_mol_x:
                if iscale.get_scaling_factor(self.conc_mem_surf_mol_x[ind]) is None:
                    if (ind[0] == "cem" and ind[1] == "cathode_left") or (
                        ind[0] == "aem" and ind[1] == "anode_right"
                    ):
                        iscale.set_scaling_factor(
                            self.conc_mem_surf_mol_x[ind],
                            iscale.get_scaling_factor(
                                self.concentrate.properties[
                                    ind[2], ind[3]
                                ].conc_mol_phase_comp["Liq", ind[4]]
                            ),
                        )
                    else:
                        iscale.set_scaling_factor(
                            self.conc_mem_surf_mol_x[ind],
                            iscale.get_scaling_factor(
                                self.diluate.properties[
                                    ind[2], ind[3]
                                ].conc_mol_phase_comp["Liq", ind[4]]
                            ),
                        )

        if hasattr(self, "current_dens_lim_x"):
            if iscale.get_scaling_factor(self.current_dens_lim_x) is None:
                if (
                    self.config.limiting_current_density_method
                    == LimitingCurrentDensityMethod.InitialValue
                ):
                    for ind in self.current_dens_lim_x:
                        sf = (
                            self.config.limiting_current_density_data**-1
                            * sum(
                                iscale.get_scaling_factor(
                                    self.diluate.properties[
                                        ind[0], 0
                                    ].conc_mol_phase_comp["Liq", j]
                                )
                                ** 2
                                for j in self.cation_set
                            )
                            ** -0.5
                            * sum(
                                iscale.get_scaling_factor(
                                    self.diluate.properties[ind].conc_mol_phase_comp[
                                        "Liq", j
                                    ]
                                )
                                ** 2
                                for j in self.cation_set
                            )
                            ** 0.5
                        )
                        iscale.set_scaling_factor(self.current_dens_lim_x[ind], sf)
                elif (
                    self.config.limiting_current_density_method
                    == LimitingCurrentDensityMethod.Empirical
                ):
                    for ind in self.current_dens_lim_x:
                        sf = 25**-1 * sum(
                            iscale.get_scaling_factor(
                                self.diluate.properties[ind].conc_mol_phase_comp[
                                    "Liq", j
                                ]
                            )
                            ** 2
                            for j in self.cation_set
                        ) ** (0.5 * 0.5)
                        iscale.set_scaling_factor(self.current_dens_lim_x[ind], sf)
                elif (
                    self.config.limiting_current_density_method
                    == LimitingCurrentDensityMethod.Theoretical
                ):
                    for ind in self.current_dens_lim_x:
                        sf = (
                            iscale.get_scaling_factor(
                                self.N_Sh["diluate", ind[0], ind[1]]
                            )
                            * iscale.get_scaling_factor(self.diffus_mass)
                            * iscale.get_scaling_factor(self.hydraulic_diameter) ** -1
                            * 96485**-1
                            * sum(
                                iscale.get_scaling_factor(
                                    self.diluate.properties[ind].conc_mol_phase_comp[
                                        "Liq", j
                                    ]
                                )
                                ** 2
                                for j in self.cation_set
                            )
                            ** 0.5
                        )
                        iscale.set_scaling_factor(self.current_dens_lim_x[ind], sf)

        if hasattr(self, "potential_nonohm_membrane_x"):
            if iscale.get_scaling_factor(self.potential_nonohm_membrane_x) is None:
                for ind in self.potential_nonohm_membrane_x:
                    sf = (
                        value(Constants.faraday_constant)
                        * value(Constants.gas_constant) ** -1
                        * 298.15**-1
                    )
                    iscale.set_scaling_factor(self.potential_nonohm_membrane_x[ind], sf)
        if hasattr(self, "potential_nonohm_dl_x"):
            if iscale.get_scaling_factor(self.potential_nonohm_dl_x) is None:
                for ind in self.potential_nonohm_dl_x:
                    sf = (
                        value(Constants.faraday_constant)
                        * value(Constants.gas_constant) ** -1
                        * 298.15**-1
                    )
                    iscale.set_scaling_factor(self.potential_nonohm_dl_x[ind], sf)
        if hasattr(self, "potential_ohm_dl_x"):
            if iscale.get_scaling_factor(self.potential_ohm_dl_x) is None:
                for ind in self.potential_nonohm_dl_x:
                    sf = (
                        96485**-1
                        * sum(
                            iscale.get_scaling_factor(
                                self.diluate.properties[0, 0].diffus_phase_comp[
                                    "Liq", j
                                ]
                            )
                            ** -2
                            for j in self.ion_set
                        )
                        ** -0.5
                        * float(len(self.ion_set)) ** -1
                    )
                    iscale.set_scaling_factor(self.potential_ohm_dl_x[ind], sf)
        if hasattr(self, "dl_thickness_x"):
            if iscale.get_scaling_factor(self.dl_thickness_x) is None:
                for ind in self.dl_thickness_x:
                    sf = (
                        96485**-1
                        * sum(
                            iscale.get_scaling_factor(
                                self.diluate.properties[0, 0].diffus_phase_comp[
                                    "Liq", j
                                ]
                            )
                            ** -2
                            for j in self.ion_set
                        )
                        ** -0.5
                        * len(self.ion_set) ** -1
                        * sum(
                            iscale.get_scaling_factor(self.conc_mem_surf_mol_x[ind, j])
                            ** 2
                            for j in self.cation_set
                        )
                        ** 0.5
                        * iscale.get_scaling_factor(
                            self.current_density_x[ind[2], ind[3]]
                        )
                        ** -1
                    )
                    iscale.set_scaling_factor(self.dl_thickness_x[ind], sf)

        for ind in self.diluate.power_electrical_x:
            if (
                iscale.get_scaling_factor(
                    self.diluate.power_electrical_x[ind], warning=False
                )
                is None
            ):
                iscale.set_scaling_factor(
                    self.diluate.power_electrical_x[ind],
                    iscale.get_scaling_factor(self.voltage_x[ind])
                    * iscale.get_scaling_factor(self.current_density_x[ind])
                    * iscale.get_scaling_factor(self.cell_width)
                    * iscale.get_scaling_factor(self.cell_length),
                )
        for ind in self.diluate.Dpower_electrical_Dx:
            if (
                iscale.get_scaling_factor(
                    self.diluate.Dpower_electrical_Dx[ind], warning=False
                )
                is None
            ):
                iscale.set_scaling_factor(
                    self.diluate.Dpower_electrical_Dx[ind],
                    iscale.get_scaling_factor(self.diluate.power_electrical_x[ind]),
                )
        if (
            iscale.get_scaling_factor(self.specific_power_electrical, warning=False)
            is None
        ):
            iscale.set_scaling_factor(
                self.specific_power_electrical,
                3.6e6
                * iscale.get_scaling_factor(
                    self.diluate.power_electrical_x[
                        0, self.diluate.length_domain.last()
                    ]
                )
                * (
                    iscale.get_scaling_factor(
                        self.diluate.properties[
                            0, self.diluate.length_domain.last()
                        ].flow_vol_phase["Liq"]
                    )
                    * iscale.get_scaling_factor(self.cell_pair_num)
                )
                ** -1,
            )

        # Set up constraint scaling
        for ind, c in self.eq_get_total_areal_resistance_x.items():
            iscale.constraint_scaling_transform(
                c, iscale.get_scaling_factor(self.total_areal_resistance_x[ind])
            )

        for ind, c in self.eq_get_current_density.items():
            if self.config.operation_mode == ElectricalOperationMode.Constant_Current:
                iscale.constraint_scaling_transform(
                    c, iscale.get_scaling_factor(self.current_applied[ind[0]])
                )
            else:
                iscale.constraint_scaling_transform(
                    c, iscale.get_scaling_factor(self.voltage_applied[ind[0]])
                )
        for ind, c in self.eq_get_voltage_x.items():
            iscale.constraint_scaling_transform(
                c, iscale.get_scaling_factor(self.voltage_x[ind])
            )
        for ind, c in self.eq_get_velocity.items():
            iscale.constraint_scaling_transform(
                c,
                iscale.get_scaling_factor(
                    self.diluate.properties[0, 0].flow_vol_phase["Liq"]
                ),
            )

        if hasattr(self, "eq_Re"):
            for ind, c in self.eq_Re.items():
                iscale.constraint_scaling_transform(
                    c, iscale.get_scaling_factor(self.N_Re[ind])
                )
        if hasattr(self, "eq_Sc"):
            for ind, c in self.eq_Sc.items():
                iscale.constraint_scaling_transform(
                    c, iscale.get_scaling_factor(self.N_Sc[ind])
                )
        if hasattr(self, "eq_Sh"):
            for ind, c in self.eq_Sh.items():
                iscale.constraint_scaling_transform(
                    c, iscale.get_scaling_factor(self.N_Sh[ind])
                )
        if hasattr(self, "eq_pressure_dx"):
            for ind, c in self.eq_pressure_dx.items():
                iscale.constraint_scaling_transform(
                    c, iscale.get_scaling_factor(self.pressure_dx[ind])
                )
        if hasattr(self, "eq_deltaP"):
            for ind, c in self.eq_deltaP.items():
                iscale.constraint_scaling_transform(
                    c,
                    iscale.get_scaling_factor(
                        self.pressure_dx[
                            ind[0], ind[1], self.difference_elements.first()
                        ]
                    ),
                )
        if hasattr(self, "eq_hydraulic_diameter"):
            for ind, c in self.eq_hydraulic_diameter.items():
                iscale.constraint_scaling_transform(
                    c, iscale.get_scaling_factor(self.hydraulic_diameter)
                )
        if hasattr(self, "eq_pressure_drop_total"):
            for ind, c in self.eq_pressure_drop_total.items():
                iscale.constraint_scaling_transform(
                    c, iscale.get_scaling_factor(self.pressure_drop_total[ind])
                )
        if hasattr(self, "eq_friction_factor"):
            for ind, c in self.eq_friction_factor.items():
                iscale.constraint_scaling_transform(
                    c, iscale.get_scaling_factor(self.friction_factor[ind])
                )
        if hasattr(self, "eq_potential_nonohm_membrane_x"):
            for ind, c in self.eq_potential_nonohm_membrane_x.items():
                iscale.constraint_scaling_transform(
                    c, iscale.get_scaling_factor(self.potential_nonohm_membrane_x[ind])
                )
        if hasattr(self, "eq_current_dens_lim_x"):
            for ind, c in self.eq_current_dens_lim_x.items():
                iscale.constraint_scaling_transform(
                    c, iscale.get_scaling_factor(self.current_dens_lim_x[ind])
                )
        if hasattr(self, "eq_conc_polarization_ratio"):
            for ind, c in self.eq_conc_polarization_ratio.items():
                iscale.constraint_scaling_transform(c, 1)
        if hasattr(self, "eq_potential_nonohm_dl_x"):
            for ind, c in self.eq_potential_nonohm_dl_x.items():
                iscale.constraint_scaling_transform(
                    c, iscale.get_scaling_factor(self.potential_nonohm_dl_x[ind])
                )
        if hasattr(self, "eq_potential_ohm_dl_x"):
            for ind, c in self.eq_potential_ohm_dl_x.items():
                iscale.constraint_scaling_transform(
                    c, iscale.get_scaling_factor(self.potential_ohm_dl_x[ind])
                )
        if hasattr(self, "eq_dl_thickness_x"):
            for ind, c in self.eq_dl_thickness_x.items():
                iscale.constraint_scaling_transform(
                    c, iscale.get_scaling_factor(self.dl_thickness_x[ind])
                )

        for ind, c in self.eq_mass_transfer_term_diluate.items():
            if ind[3] == "H2O":
                sf_osm = (
                    1e-3
                    * 0.018
                    * iscale.get_scaling_factor(self.water_permeability_membrane)
                    * (
                        (
                            iscale.get_scaling_factor(
                                self.diluate.properties[
                                    (ind[0], ind[1])
                                ].pressure_osm_phase[ind[2]]
                            )
                        )
                        ** 2
                        + (
                            iscale.get_scaling_factor(
                                self.diluate.properties[
                                    (ind[0], ind[1])
                                ].pressure_osm_phase[ind[2]]
                            )
                        )
                        ** 2
                    )
                    ** 0.5
                )
                sf_eleosm = value(Constants.faraday_constant)
                iscale.constraint_scaling_transform(
                    c,
                    (sf_osm**2 + sf_eleosm**2) ** 0.5
                    * iscale.get_scaling_factor(self.cell_pair_num),
                )
            elif ind[3] in self.config.property_package.ion_set:
                sf_diff = (
                    iscale.get_scaling_factor(self.solute_diffusivity_membrane)
                    / iscale.get_scaling_factor(self.membrane_thickness)
                    * (
                        (
                            (
                                iscale.get_scaling_factor(
                                    self.concentrate.properties[
                                        (ind[0], ind[1])
                                    ].conc_mol_phase_comp[ind[2], ind[3]]
                                )
                            )
                            ** 2
                            + (
                                iscale.get_scaling_factor(
                                    self.diluate.properties[
                                        (ind[0], ind[1])
                                    ].conc_mol_phase_comp[ind[2], ind[3]]
                                )
                            )
                            ** 2
                        )
                        ** 0.5
                    )
                )
                sf_elemig = value(Constants.faraday_constant)
                iscale.constraint_scaling_transform(
                    c,
                    (sf_diff**2 + sf_elemig**2) ** 0.5
                    * iscale.get_scaling_factor(self.cell_pair_num),
                )
            else:
                iscale.constraint_scaling_transform(
                    c,
                    iscale.get_scaling_factor(self.solute_diffusivity_membrane)
                    / iscale.get_scaling_factor(self.membrane_thickness)
                    * (
                        (
                            (
                                iscale.get_scaling_factor(
                                    self.concentrate.properties[
                                        (ind[0], ind[1])
                                    ].conc_mol_phase_comp[ind[2], ind[3]]
                                )
                            )
                            ** 2
                            + (
                                iscale.get_scaling_factor(
                                    self.diluate.properties[
                                        (ind[0], ind[1])
                                    ].conc_mol_phase_comp[ind[2], ind[3]]
                                )
                            )
                            ** 2
                        )
                        ** 0.5
                    )
                    * iscale.get_scaling_factor(self.cell_pair_num),
                )
        for ind, c in self.eq_mass_transfer_term_concentrate.items():
            if ind[3] == "H2O":
                sf_osm = (
                    1e-3
                    * 0.018
                    * iscale.get_scaling_factor(self.water_permeability_membrane)
                    * (
                        (
                            iscale.get_scaling_factor(
                                self.diluate.properties[
                                    (ind[0], ind[1])
                                ].pressure_osm_phase[ind[2]]
                            )
                        )
                        ** 2
                        + (
                            iscale.get_scaling_factor(
                                self.diluate.properties[
                                    (ind[0], ind[1])
                                ].pressure_osm_phase[ind[2]]
                            )
                        )
                        ** 2
                    )
                    ** 0.5
                )
                sf_eleosm = value(Constants.faraday_constant)
                iscale.constraint_scaling_transform(
                    c,
                    (sf_osm**2 + sf_eleosm**2) ** 0.5
                    * iscale.get_scaling_factor(self.cell_pair_num),
                )
            elif ind[3] in self.config.property_package.ion_set:
                sf_diff = (
                    iscale.get_scaling_factor(self.solute_diffusivity_membrane)
                    / iscale.get_scaling_factor(self.membrane_thickness)
                    * (
                        (
                            (
                                iscale.get_scaling_factor(
                                    self.concentrate.properties[
                                        (ind[0], ind[1])
                                    ].conc_mol_phase_comp[ind[2], ind[3]]
                                )
                            )
                            ** 2
                            + (
                                iscale.get_scaling_factor(
                                    self.diluate.properties[
                                        (ind[0], ind[1])
                                    ].conc_mol_phase_comp[ind[2], ind[3]]
                                )
                            )
                            ** 2
                        )
                        ** 0.5
                    )
                )
                sf_elemig = value(Constants.faraday_constant)
                iscale.constraint_scaling_transform(
                    c,
                    (sf_diff**2 + sf_elemig**2) ** 0.5
                    * iscale.get_scaling_factor(self.cell_pair_num),
                )
            else:
                iscale.constraint_scaling_transform(
                    c,
                    iscale.get_scaling_factor(self.solute_diffusivity_membrane)
                    / iscale.get_scaling_factor(self.membrane_thickness)
                    * (
                        (
                            (
                                iscale.get_scaling_factor(
                                    self.concentrate.properties[
                                        (ind[0], ind[1])
                                    ].conc_mol_phase_comp[ind[2], ind[3]]
                                )
                            )
                            ** 2
                            + (
                                iscale.get_scaling_factor(
                                    self.diluate.properties[
                                        (ind[0], ind[1])
                                    ].conc_mol_phase_comp[ind[2], ind[3]]
                                )
                            )
                            ** 2
                        )
                        ** 0.5
                    )
                    * iscale.get_scaling_factor(self.cell_pair_num),
                )
        for ind, c in self.eq_power_electrical.items():
            iscale.constraint_scaling_transform(
                c, iscale.get_scaling_factor(self.diluate.power_electrical_x[ind])
            )

        for ind, c in self.eq_specific_power_electrical.items():
            iscale.constraint_scaling_transform(
                c, iscale.get_scaling_factor(self.specific_power_electrical)
            )

        for ind, c in self.eq_current_efficiency_x.items():
            iscale.constraint_scaling_transform(
                c,
                iscale.get_scaling_factor(self.current_density_x[ind[0], ind[1]])
                * iscale.get_scaling_factor(self.cell_width),
            )

    def _get_stream_table_contents(self, time_point=0):
        return create_stream_table_dataframe(
            {
                "Diluate Channel Inlet": self.inlet_diluate,
                "Concentrate Channel Inlet": self.inlet_concentrate,
                "Diluate Channel Outlet": self.outlet_diluate,
                "Concentrate Channel Outlet": self.outlet_concentrate,
            },
            time_point=time_point,
        )

    def _get_performance_contents(self, time_point=0):

        return {
            "vars": {
                "Total electrical power consumption": self.diluate.power_electrical_x[
                    time_point, self.diluate.length_domain.last()
                ],
                "Specific electrical power consumption, ED stack": self.specific_power_electrical[
                    time_point
                ],
                "Water recovery by mass": self.recovery_mass_H2O[time_point],
                "Channel inlet velocity, diluate": self.velocity[
                    "diluate", time_point, self.diluate.length_domain.first()
                ],
                "Channel inlet velocity, concentrate": self.velocity[
                    "concentrate", time_point, self.diluate.length_domain.first()
                ],
            },
            "exprs": {},
            "params": {},
        }

    def get_power_electrical(self, time_point=0):
        return self.diluate.power_electrical_x[
            time_point, self.diluate.length_domain.last()
        ]

    @property
    def default_costing_method(self):
        return cost_electrodialysis
