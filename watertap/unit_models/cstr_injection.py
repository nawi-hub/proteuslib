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
Modified CSTR model which includes terms for injection of species/reactants.

This is copied from the standard IDAES CSTR with the addition of mass transfer terms.
NOTE: This is likely a temporary model until a more detailed model is available.
"""

# Import Pyomo libraries
from pyomo.common.config import ConfigBlock, ConfigValue, In, Bool
from pyomo.environ import (
    Reference,
    Var,
    Param,
    units as pyunits,
    NonNegativeReals,
)

# Import IDAES cores
from idaes.core import (
    ControlVolume0DBlock,
    declare_process_block_class,
    MaterialBalanceType,
    EnergyBalanceType,
    MomentumBalanceType,
    UnitModelBlockData,
    useDefault,
)
from idaes.core.scaling import CustomScalerBase, ConstraintScalingScheme
from idaes.core.util.config import (
    is_physical_parameter_block,
    is_reaction_parameter_block,
)
from idaes.core.util.exceptions import ConfigurationError
from enum import Enum, auto

from watertap.core import InitializationMixin

from watertap.costing.unit_models.cstr_injection import cost_cstr_injection

__author__ = "Andrew Lee, Adam Atia, Vibhav Dabadghao"


class ElectricityConsumption(Enum):
    """
    none: no electricity consumption
    fixed: assume electricity intensity
    calculated: calculate based on aeration energy equation from BSM2 documentation
    """

    none = auto()
    fixed = auto()
    calculated = auto()


class CSTR_InjectionScaler(CustomScalerBase):
    """
    Default modular scaler for CSTR with injection.

    This Scaler relies on the associated property and reaction packages,
    either through user provided options (submodel_scalers argument) or by default
    Scalers assigned to the packages.
    """

    DEFAULT_SCALING_FACTORS = {
        "volume": 1e-3,
    }

    def variable_scaling_routine(
        self, model, overwrite: bool = False, submodel_scalers: dict = None
    ):
        """
        Routine to apply scaling factors to variables in model.

        Args:
            model: model to be scaled
            overwrite: whether to overwrite existing scaling factors
            submodel_scalers: dict of Scalers to use for sub-models, keyed by submodel local name

        Returns:
            None
        """
        # Call scaling methods for sub-models
        self.call_submodel_scaler_method(
            submodel=model.control_volume.properties_in,
            method="variable_scaling_routine",
            submodel_scalers=submodel_scalers,
            overwrite=overwrite,
        )
        self.propagate_state_scaling(
            target_state=model.control_volume.properties_out,
            source_state=model.control_volume.properties_in,
            overwrite=overwrite,
        )

        self.call_submodel_scaler_method(
            submodel=model.control_volume.properties_out,
            method="variable_scaling_routine",
            submodel_scalers=submodel_scalers,
            overwrite=overwrite,
        )
        self.call_submodel_scaler_method(
            submodel=model.control_volume.reactions,
            method="variable_scaling_routine",
            submodel_scalers=submodel_scalers,
            overwrite=overwrite,
        )

        # Scaling control volume variables
        self.scale_variable_by_default(
            model.control_volume.volume[0], overwrite=overwrite
        )

    def constraint_scaling_routine(
        self, model, overwrite: bool = False, submodel_scalers: dict = None
    ):
        """
        Routine to apply scaling factors to constraints in model.

        Submodel Scalers are called for the property and reaction blocks. All other constraints
        are scaled using the inverse maximum scheme.

        Args:
            model: model to be scaled
            overwrite: whether to overwrite existing scaling factors
            submodel_scalers: dict of Scalers to use for sub-models, keyed by submodel local name

        Returns:
            None
        """
        # Call scaling methods for sub-models
        self.call_submodel_scaler_method(
            submodel=model.control_volume.properties_in,
            method="constraint_scaling_routine",
            submodel_scalers=submodel_scalers,
            overwrite=overwrite,
        )
        self.call_submodel_scaler_method(
            submodel=model.control_volume.properties_out,
            method="constraint_scaling_routine",
            submodel_scalers=submodel_scalers,
            overwrite=overwrite,
        )
        self.call_submodel_scaler_method(
            submodel=model.control_volume.reactions,
            method="constraint_scaling_routine",
            submodel_scalers=submodel_scalers,
            overwrite=overwrite,
        )

        # Scale unit level constraints
        if hasattr(model, "eq_hydraulic_retention_time"):
            self.scale_constraint_by_nominal_value(
                model.eq_hydraulic_retention_time[0],
                scheme=ConstraintScalingScheme.inverseMaximum,
                overwrite=overwrite,
            )

        if hasattr(model, "eq_mass_transfer"):
            self.scale_constraint_by_nominal_value(
                model.eq_mass_transfer[0],
                scheme=ConstraintScalingScheme.inverseMaximum,
                overwrite=overwrite,
            )

        if hasattr(model, "eq_electricity_consumption"):
            self.scale_constraint_by_nominal_value(
                model.eq_electricity_consumption[0],
                scheme=ConstraintScalingScheme.inverseMaximum,
                overwrite=overwrite,
            )


@declare_process_block_class("CSTR_Injection")
class CSTR_InjectionData(InitializationMixin, UnitModelBlockData):
    """
    CSTR Unit Model with Injection Class
    """

    default_scaler = CSTR_InjectionScaler

    CONFIG = UnitModelBlockData.CONFIG()

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
        "energy_balance_type",
        ConfigValue(
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
        "has_heat_transfer",
        ConfigValue(
            default=False,
            domain=Bool,
            description="Heat transfer term construction flag",
            doc="""Indicates whether terms for heat transfer should be constructed,
**default** - False.
**Valid values:** {
**True** - include heat transfer terms,
**False** - exclude heat transfer terms.}""",
        ),
    )
    CONFIG.declare(
        "has_pressure_change",
        ConfigValue(
            default=False,
            domain=Bool,
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
        "has_equilibrium_reactions",
        ConfigValue(
            default=False,
            domain=Bool,
            description="Equilibrium reaction construction flag",
            doc="""Indicates whether terms for equilibrium controlled reactions
should be constructed,
**default** - True.
**Valid values:** {
**True** - include equilibrium reaction terms,
**False** - exclude equilibrium reaction terms.}""",
        ),
    )
    CONFIG.declare(
        "has_phase_equilibrium",
        ConfigValue(
            default=False,
            domain=Bool,
            description="Phase equilibrium construction flag",
            doc="""Indicates whether terms for phase equilibrium should be
constructed,
**default** = False.
**Valid values:** {
**True** - include phase equilibrium terms
**False** - exclude phase equilibrium terms.}""",
        ),
    )
    CONFIG.declare(
        "has_heat_of_reaction",
        ConfigValue(
            default=False,
            domain=Bool,
            description="Heat of reaction term construction flag",
            doc="""Indicates whether terms for heat of reaction terms should be
constructed,
**default** - False.
**Valid values:** {
**True** - include heat of reaction terms,
**False** - exclude heat of reaction terms.}""",
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
        "reaction_package",
        ConfigValue(
            default=None,
            domain=is_reaction_parameter_block,
            description="Reaction package to use for control volume",
            doc="""Reaction parameter object used to define reaction calculations,
**default** - None.
**Valid values:** {
**None** - no reaction package,
**ReactionParameterBlock** - a ReactionParameterBlock object.}""",
        ),
    )
    CONFIG.declare(
        "reaction_package_args",
        ConfigBlock(
            implicit=True,
            description="Arguments to use for constructing reaction packages",
            doc="""A ConfigBlock with arguments to be passed to a reaction block(s)
and used when constructing these,
**default** - None.
**Valid values:** {
see reaction package for documentation.}""",
        ),
    )

    CONFIG.declare(
        "electricity_consumption",
        ConfigValue(
            default=ElectricityConsumption.fixed,
            domain=In(ElectricityConsumption),
            description="Electricity consumption calculation",
            doc="""Indicates whether electricity consumption is fixed by the user or excluded
        **default** - ElectricityConsumption.none.
        **Valid values:** {
        **ElectricityConsumption.none** - no electricity consumption within the unit,
        **ElectricityConsumption.fixed** - calculate electricity consumption based on assumed electricity intensity in kWh/m3,
        **ElectricityConsumption.aeration_calculation** - calculate electricity consumption based on aeration energy}""",
        ),
    )
    CONFIG.declare(
        "has_aeration",
        ConfigValue(
            default=False,
            domain=Bool,
            description="Aeration flag",
            doc="""Indicates whether terms for aeration terms should be
    expected,
    **default** - False.
    **Valid values:** {
    **True** - include aeration terms,
    **False** - exclude aeration terms.}""",
        ),
    )

    def build(self):
        """
        Begin building model (pre-DAE transformation).
        Args:
            None
        Returns:
            None
        """
        # Call UnitModel.build to setup dynamics
        super(CSTR_InjectionData, self).build()

        if self.config.has_aeration:
            # Assuming that the supplied property package doesn't have both S_O and SO2 to represent oxygen.
            if "S_O" in self.config.property_package.component_list:
                oxygen_str = "S_O"
            elif "S_O2" in self.config.property_package.component_list:
                oxygen_str = "S_O2"
            else:
                raise ConfigurationError(
                    "has_aeration was set to True, but the property package has neither 'S_O' nor 'S_O2' in its list of components."
                )

        # Build Control Volume
        self.control_volume = ControlVolume0DBlock(
            dynamic=self.config.dynamic,
            has_holdup=self.config.has_holdup,
            property_package=self.config.property_package,
            property_package_args=self.config.property_package_args,
            reaction_package=self.config.reaction_package,
            reaction_package_args=self.config.reaction_package_args,
        )

        self.control_volume.add_geometry()

        self.control_volume.add_state_blocks(
            has_phase_equilibrium=self.config.has_phase_equilibrium
        )

        self.control_volume.add_reaction_blocks(
            has_equilibrium=self.config.has_equilibrium_reactions
        )

        self.control_volume.add_material_balances(
            balance_type=self.config.material_balance_type,
            has_rate_reactions=True,
            has_equilibrium_reactions=self.config.has_equilibrium_reactions,
            has_phase_equilibrium=self.config.has_phase_equilibrium,
            has_mass_transfer=True,
        )

        self.control_volume.add_energy_balances(
            balance_type=self.config.energy_balance_type,
            has_heat_of_reaction=self.config.has_heat_of_reaction,
            has_heat_transfer=self.config.has_heat_transfer,
        )

        self.control_volume.add_momentum_balances(
            balance_type=self.config.momentum_balance_type,
            has_pressure_change=self.config.has_pressure_change,
        )

        # Add Ports
        self.add_inlet_port()
        self.add_outlet_port()

        # Add object references
        self.volume = Reference(self.control_volume.volume[:])
        self.injection = Reference(self.control_volume.mass_transfer_term[...])

        # Add CSTR performance equation
        @self.Constraint(
            self.flowsheet().time,
            self.config.reaction_package.rate_reaction_idx,
            doc="CSTR performance equation",
        )
        def cstr_performance_eqn(b, t, r):
            return b.control_volume.rate_reaction_extent[t, r] == (
                b.volume[t] * b.control_volume.reactions[t].reaction_rate[r]
            )

        # Set references to balance terms at unit level
        if (
            self.config.has_heat_transfer is True
            and self.config.energy_balance_type != EnergyBalanceType.none
        ):
            self.heat_duty = Reference(self.control_volume.heat[:])

        if (
            self.config.has_pressure_change is True
            and self.config.momentum_balance_type != MomentumBalanceType.none
        ):
            self.deltaP = Reference(self.control_volume.deltaP[:])

        self.hydraulic_retention_time = Var(
            self.flowsheet().time,
            initialize=1460.407,
            domain=NonNegativeReals,
            units=pyunits.s,
            doc="Hydraulic retention time",
        )

        @self.Constraint(
            self.flowsheet().time,
            doc="Hydraulic retention time",
        )
        def eq_hydraulic_retention_time(self, t):
            return (
                self.hydraulic_retention_time[t]
                == self.volume[t] / self.control_volume.properties_in[t].flow_vol
            )

        if self.config.has_aeration:

            # TODO: add temp dependence for KLa and consider moving to prop model
            self.KLa = Var(
                initialize=5,
                units=pyunits.hour**-1,
                doc="Lumped mass transfer coefficient for oxygen",
            )
            # TODO: add temp dependence for S_O_eq and move to prop model
            self.S_O_eq = Param(
                default=8e-3,
                units=pyunits.kg / pyunits.m**3,
                mutable=True,
                doc="Dissolved oxygen concentration at equilibrium",
            )

            @self.Constraint(
                self.flowsheet().config.time,
                doc="Oxygen mass transfer rate",
            )
            def eq_mass_transfer(self, t):
                return pyunits.convert(
                    self.injection[t, "Liq", oxygen_str],
                    to_units=pyunits.kg / pyunits.hour,
                ) == (
                    self.KLa
                    * self.volume[t]
                    * (self.S_O_eq - self.outlet.conc_mass_comp[t, oxygen_str])
                )

        if self.config.electricity_consumption != ElectricityConsumption.none:
            self.electricity_consumption = Var(
                self.flowsheet().time,
                units=pyunits.kW,
                bounds=(0, None),
                doc="Electricity consumption of unit",
            )
            if self.config.electricity_consumption == ElectricityConsumption.fixed:

                # The value is taken from Maravelias' data
                self.energy_electric_flow_vol_inlet = Param(
                    initialize=0.0108,
                    units=pyunits.kWh / pyunits.m**3,
                    mutable=True,
                    doc="Electricity intensity with respect to inlet flow",
                )

                # Electricity constraint
                @self.Constraint(
                    self.flowsheet().time,
                    doc="Unit level electricity consumption",
                )
                def eq_electricity_consumption(self, t):
                    return self.electricity_consumption[t] == (
                        self.energy_electric_flow_vol_inlet
                        * pyunits.convert(
                            self.control_volume.properties_in[t].flow_vol,
                            to_units=pyunits.m**3 / pyunits.hr,
                        )
                    )

            elif (
                self.config.electricity_consumption == ElectricityConsumption.calculated
                and self.config.has_aeration
            ):
                # Electricity constraint
                @self.Constraint(
                    self.flowsheet().time,
                    doc="Unit level electricity consumption",
                )
                def eq_electricity_consumption(self, t):
                    return self.electricity_consumption[t] == (
                        # TODO: revisit origin of 1.8 factor and more general aeration energy equation
                        # 1.8 may be the aeration efficiency which is oxygen mass transfer divided by power input
                        self.S_O_eq
                        / 1.8
                        / pyunits.kg
                        * pyunits.kWh
                        * pyunits.convert(
                            self.control_volume.volume[t] * self.KLa,
                            to_units=pyunits.m**3 / pyunits.hr,
                        )
                    )

            elif (
                not self.config.has_aeration
                and self.config.electricity_consumption
                == ElectricityConsumption.calculated
            ):
                raise ConfigurationError(
                    f"electricity_consumption=ElectricityConsumption.calculated is currently only supported for aeration. Set has_aeration=True to compute electricity consumption or set electricity_consumption=ElectricityConsumption.fixed to compute electricity consumption based on an assumed specific energy consumption."
                )
            else:
                raise ConfigurationError(
                    f"{self.config.electricity_consumption} is not a valid option for determining electricity consumption."
                )

    def _get_performance_contents(self, time_point=0):
        var_dict = {"Volume": self.volume[time_point]}
        for k, v in self.injection.items():
            if k[0] == time_point:
                var_dict[f"Injection [{k[1:]}]"] = v
        if hasattr(self, "heat_duty"):
            var_dict["Heat Duty"] = self.heat_duty[time_point]
        if hasattr(self, "deltaP"):
            var_dict["Pressure Change"] = self.deltaP[time_point]
        if hasattr(self, "electricity_consumption"):
            var_dict["Electricity Consumption"] = self.electricity_consumption[
                time_point
            ]

        return {"vars": var_dict}

    @property
    def default_costing_method(self):
        return cost_cstr_injection
