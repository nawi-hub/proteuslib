import pytest
import idaes.logger as idaeslog
from pyomo.environ import (
    ConcreteModel,
    value,
    Var,
    Constraint,
    assert_optimal_termination,
)
from pyomo.environ import (
    NonNegativeReals,
    Param,
    Var,
    check_optimal_termination,
    exp,
    units as pyunits,
)
from idaes.core.solvers import get_solver
from .model_diagnostics import DiagnosticsToolbox
from pyomo.environ import *
from pyomo.util.infeasible import log_infeasible_constraints
from pyomo.network import Port
from idaes.core.solvers import get_solver
from idaes.core.util.model_statistics import (
    degrees_of_freedom,
    number_variables,
    number_total_constraints,
    number_unused_variables,
)
from idaes.core.util.testing import initialization_tester
from idaes.core.util.scaling import (
    calculate_scaling_factors,
    unscaled_variables_generator,
    badly_scaled_var_generator,
)
from idaes.core import (
    FlowsheetBlock,
    MaterialBalanceType,
    EnergyBalanceType,
    MomentumBalanceType,
    FlowDirection,
)
from pyomo.common.config import ConfigBlock
import pyomo.environ as pyo
import idaes.logger as idaeslog
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.environ import ConcreteModel
from idaes.core import FlowsheetBlock
import watertap.property_models.seawater_prop_pack as props_sw
import watertap.property_models.water_prop_pack as props_w
from watertap.unit_models.MD.membrane_distillation_0D import MembraneDistillation0D
from watertap.unit_models.MD.MD_channel_0D import MDChannel0DBlock
from watertap.unit_models.MD.MD_channel_base import (
    ConcentrationPolarizationType,
    TemperaturePolarizationType,
    MassTransferCoefficient,
    PressureChangeType,
    FrictionFactor,
)


from pyomo.core.base.units_container import units as pyunits
from pyomo.environ import Constraint, Var

solver = get_solver()


@pytest.mark.unit
def test_config():

    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.properties_hot_ch = props_sw.SeawaterParameterBlock()
    m.fs.properties_cold_ch = props_sw.SeawaterParameterBlock()
    m.fs.properties_vapor = props_w.WaterParameterBlock()

    m.fs.unit = MembraneDistillation0D(
        hot_ch={
            "property_package": m.fs.properties_hot_ch,
            "property_package_vapor": m.fs.properties_vapor,
        },
        cold_ch={
            "property_package": m.fs.properties_cold_ch,
            "property_package_vapor": m.fs.properties_vapor,
        },
    )
    assert len(m.fs.unit.config) == 4
    assert not m.fs.unit.config.dynamic
    assert not m.fs.unit.config.has_holdup

    assert isinstance(m.fs.unit.config.hot_ch, ConfigBlock)
    assert isinstance(m.fs.unit.config.cold_ch, ConfigBlock)

    assert len(m.fs.unit.config.hot_ch) == 16
    assert len(m.fs.unit.config.cold_ch) == 16

    assert not m.fs.unit.config.hot_ch.dynamic
    assert not m.fs.unit.config.hot_ch.has_holdup
    assert (
        m.fs.unit.config.hot_ch.material_balance_type == MaterialBalanceType.useDefault
    )
    assert m.fs.unit.config.hot_ch.energy_balance_type == EnergyBalanceType.useDefault
    assert (
        m.fs.unit.config.hot_ch.momentum_balance_type
        == MomentumBalanceType.pressureTotal
    )
    assert not m.fs.unit.config.hot_ch.has_pressure_change
    assert m.fs.unit.config.hot_ch.property_package is m.fs.properties_hot_ch
    assert m.fs.unit.config.hot_ch.flow_direction == FlowDirection.forward
    assert (
        m.fs.unit.config.hot_ch.temperature_polarization_type
        == TemperaturePolarizationType.calculated
    )
    assert (
        m.fs.unit.config.hot_ch.concentration_polarization_type
        == ConcentrationPolarizationType.calculated
    )
    assert (
        m.fs.unit.config.hot_ch.mass_transfer_coefficient
        == MassTransferCoefficient.calculated
    )
    assert (
        m.fs.unit.config.hot_ch.pressure_change_type
        == PressureChangeType.fixed_per_stage
    )

    assert not m.fs.unit.config.cold_ch.dynamic
    assert not m.fs.unit.config.cold_ch.has_holdup
    assert (
        m.fs.unit.config.cold_ch.material_balance_type == MaterialBalanceType.useDefault
    )
    assert m.fs.unit.config.cold_ch.energy_balance_type == EnergyBalanceType.useDefault
    assert (
        m.fs.unit.config.cold_ch.momentum_balance_type
        == MomentumBalanceType.pressureTotal
    )
    assert not m.fs.unit.config.cold_ch.has_pressure_change
    assert m.fs.unit.config.cold_ch.property_package is m.fs.properties_cold_ch
    assert m.fs.unit.config.cold_ch.flow_direction == FlowDirection.forward
    assert (
        m.fs.unit.config.cold_ch.temperature_polarization_type
        == TemperaturePolarizationType.calculated
    )
    assert (
        m.fs.unit.config.cold_ch.concentration_polarization_type
        == ConcentrationPolarizationType.calculated
    )
    assert (
        m.fs.unit.config.cold_ch.mass_transfer_coefficient
        == MassTransferCoefficient.calculated
    )
    assert (
        m.fs.unit.config.cold_ch.pressure_change_type
        == PressureChangeType.fixed_per_stage
    )


@pytest.mark.unit
def test_option_temperature_polarization_type_fixed():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.properties_hot_ch = props_sw.SeawaterParameterBlock()
    m.fs.properties_cold_ch = props_sw.SeawaterParameterBlock()
    m.fs.properties_vapor = props_w.WaterParameterBlock()
    m.fs.unit = MembraneDistillation0D(
        hot_ch={
            "property_package": m.fs.properties_hot_ch,
            "property_package_vapor": m.fs.properties_vapor,
            "has_pressure_change": True,
            "concentration_polarization_type": ConcentrationPolarizationType.none,
            "mass_transfer_coefficient": MassTransferCoefficient.none,
            "temperature_polarization_type": TemperaturePolarizationType.fixed,
        },
        cold_ch={
            "property_package": m.fs.properties_cold_ch,
            "property_package_vapor": m.fs.properties_vapor,
            "has_pressure_change": True,
            "temperature_polarization_type": TemperaturePolarizationType.fixed,
        },
    )

    assert (
        m.fs.unit.config.hot_ch.concentration_polarization_type
        == ConcentrationPolarizationType.none
    )

    assert (
        m.fs.unit.config.hot_ch.temperature_polarization_type
        == TemperaturePolarizationType.fixed
    )

    assert (
        m.fs.unit.config.cold_ch.temperature_polarization_type
        == TemperaturePolarizationType.fixed
    )

    assert isinstance(m.fs.unit.hot_ch.h_conv, Var)
    assert isinstance(m.fs.unit.cold_ch.h_conv, Var)


@pytest.mark.unit
def test_option_temperature_polarization_type_calculated():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.properties_hot_ch = props_sw.SeawaterParameterBlock()
    m.fs.properties_cold_ch = props_sw.SeawaterParameterBlock()
    m.fs.properties_vapor = props_w.WaterParameterBlock()
    m.fs.unit = MembraneDistillation0D(
        hot_ch={
            "property_package": m.fs.properties_hot_ch,
            "property_package_vapor": m.fs.properties_vapor,
            "has_pressure_change": True,
            "concentration_polarization_type": ConcentrationPolarizationType.none,
            "mass_transfer_coefficient": MassTransferCoefficient.none,
            "temperature_polarization_type": TemperaturePolarizationType.calculated,
        },
        cold_ch={
            "property_package": m.fs.properties_cold_ch,
            "property_package_vapor": m.fs.properties_vapor,
            "has_pressure_change": True,
            "temperature_polarization_type": TemperaturePolarizationType.calculated,
        },
    )

    assert (
        m.fs.unit.config.hot_ch.concentration_polarization_type
        == ConcentrationPolarizationType.none
    )
    assert (
        m.fs.unit.config.hot_ch.temperature_polarization_type
        == TemperaturePolarizationType.calculated
    )
    assert (
        m.fs.unit.config.cold_ch.temperature_polarization_type
        == TemperaturePolarizationType.calculated
    )

    assert isinstance(m.fs.unit.hot_ch.h_conv, Var)
    assert isinstance(m.fs.unit.cold_ch.h_conv, Var)
    assert isinstance(m.fs.unit.hot_ch.N_Pr, Var)
    assert isinstance(m.fs.unit.hot_ch.N_Nu, Var)
    assert isinstance(m.fs.unit.hot_ch.area, Var)
    assert isinstance(m.fs.unit.hot_ch.channel_height, Var)
    assert isinstance(m.fs.unit.hot_ch.dh, Var)
    assert isinstance(m.fs.unit.hot_ch.spacer_porosity, Var)
    assert isinstance(m.fs.unit.hot_ch.N_Re, Var)

    assert isinstance(m.fs.unit.hot_ch.eq_h_conv, Constraint)
    assert isinstance(m.fs.unit.hot_ch.eq_N_Nu, Constraint)
    assert isinstance(m.fs.unit.hot_ch.eq_N_Pr, Constraint)
    assert isinstance(m.fs.unit.hot_ch.eq_dh, Constraint)
    assert isinstance(m.fs.unit.hot_ch.eq_area, Constraint)
    assert isinstance(m.fs.unit.hot_ch.eq_N_Re, Constraint)

    assert isinstance(m.fs.unit.cold_ch.N_Pr, Var)
    assert isinstance(m.fs.unit.cold_ch.N_Nu, Var)
    assert isinstance(m.fs.unit.cold_ch.area, Var)
    assert isinstance(m.fs.unit.cold_ch.channel_height, Var)
    assert isinstance(m.fs.unit.cold_ch.dh, Var)
    assert isinstance(m.fs.unit.cold_ch.spacer_porosity, Var)
    assert isinstance(m.fs.unit.cold_ch.N_Re, Var)

    assert isinstance(m.fs.unit.cold_ch.eq_h_conv, Constraint)
    assert isinstance(m.fs.unit.cold_ch.eq_N_Nu, Constraint)
    assert isinstance(m.fs.unit.cold_ch.eq_N_Pr, Constraint)
    assert isinstance(m.fs.unit.cold_ch.eq_dh, Constraint)
    assert isinstance(m.fs.unit.cold_ch.eq_area, Constraint)
    assert isinstance(m.fs.unit.cold_ch.eq_N_Re, Constraint)


@pytest.mark.unit
def test_option_has_pressure_change():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.properties_hot_ch = props_sw.SeawaterParameterBlock()
    m.fs.properties_cold_ch = props_sw.SeawaterParameterBlock()
    m.fs.properties_vapor = props_w.WaterParameterBlock()

    m.fs.unit = MembraneDistillation0D(
        hot_ch={
            "property_package": m.fs.properties_hot_ch,
            "property_package_vapor": m.fs.properties_vapor,
            "has_pressure_change": True,
        },
        cold_ch={
            "property_package": m.fs.properties_cold_ch,
            "property_package_vapor": m.fs.properties_vapor,
            "has_pressure_change": True,
        },
    )

    assert isinstance(m.fs.unit.hot_ch.deltaP, Var)
    assert isinstance(m.fs.unit.cold_ch.deltaP, Var)


# for hot side only
@pytest.mark.unit
def test_option_concentration_polarization_type_fixed():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.properties_hot_ch = props_sw.SeawaterParameterBlock()
    m.fs.properties_cold_ch = props_sw.SeawaterParameterBlock()
    m.fs.properties_vapor = props_w.WaterParameterBlock()
    m.fs.unit = MembraneDistillation0D(
        hot_ch={
            "property_package": m.fs.properties_hot_ch,
            "property_package_vapor": m.fs.properties_vapor,
            "has_pressure_change": True,
            "concentration_polarization_type": ConcentrationPolarizationType.fixed,
            "mass_transfer_coefficient": MassTransferCoefficient.none,
        },
        cold_ch={
            "property_package": m.fs.properties_cold_ch,
            "property_package_vapor": m.fs.properties_vapor,
            "has_pressure_change": True,
        },
    )

    assert (
        m.fs.unit.config.hot_ch.concentration_polarization_type
        == ConcentrationPolarizationType.fixed
    )

    assert isinstance(m.fs.unit.hot_ch.cp_modulus, Var)


# hot side only
@pytest.mark.unit
def test_option_concentration_polarization_type_calculated_kf_fixed():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.properties_hot_ch = props_sw.SeawaterParameterBlock()
    m.fs.properties_cold_ch = props_sw.SeawaterParameterBlock()
    m.fs.properties_vapor = props_w.WaterParameterBlock()
    m.fs.unit = MembraneDistillation0D(
        hot_ch={
            "property_package": m.fs.properties_hot_ch,
            "property_package_vapor": m.fs.properties_vapor,
            "has_pressure_change": True,
            "concentration_polarization_type": ConcentrationPolarizationType.calculated,
            "mass_transfer_coefficient": MassTransferCoefficient.fixed,
        },
        cold_ch={
            "property_package": m.fs.properties_cold_ch,
            "property_package_vapor": m.fs.properties_vapor,
            "has_pressure_change": True,
        },
    )

    assert (
        m.fs.unit.config.hot_ch.concentration_polarization_type
        == ConcentrationPolarizationType.calculated
    )
    assert (
        m.fs.unit.config.hot_ch.mass_transfer_coefficient
        == MassTransferCoefficient.fixed
    )
    assert isinstance(m.fs.unit.hot_ch.K, Var)


@pytest.mark.unit
def test_option_concentration_polarization_type_calculated_kf_calculated():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.properties_hot_ch = props_sw.SeawaterParameterBlock()
    m.fs.properties_cold_ch = props_sw.SeawaterParameterBlock()
    m.fs.properties_vapor = props_w.WaterParameterBlock()
    m.fs.unit = MembraneDistillation0D(
        hot_ch={
            "property_package": m.fs.properties_hot_ch,
            "property_package_vapor": m.fs.properties_vapor,
            "has_pressure_change": True,
            "concentration_polarization_type": ConcentrationPolarizationType.calculated,
            "mass_transfer_coefficient": MassTransferCoefficient.calculated,
        },
        cold_ch={
            "property_package": m.fs.properties_cold_ch,
            "property_package_vapor": m.fs.properties_vapor,
            "has_pressure_change": True,
        },
    )

    assert (
        m.fs.unit.config.hot_ch.concentration_polarization_type
        == ConcentrationPolarizationType.calculated
    )
    assert (
        m.fs.unit.config.hot_ch.mass_transfer_coefficient
        == MassTransferCoefficient.calculated
    )
    assert isinstance(m.fs.unit.hot_ch.K, Var)
    assert isinstance(m.fs.unit.hot_ch.channel_height, Var)
    assert isinstance(m.fs.unit.width, Var)
    assert isinstance(m.fs.unit.length, Var)
    assert isinstance(m.fs.unit.hot_ch.dh, Var)
    assert isinstance(m.fs.unit.hot_ch.spacer_porosity, Var)
    assert isinstance(m.fs.unit.hot_ch.N_Sc_comp, Var)
    assert isinstance(m.fs.unit.hot_ch.N_Sh_comp, Var)
    assert isinstance(m.fs.unit.hot_ch.N_Re, Var)


@pytest.mark.unit
def test_option_pressure_change_calculated():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.properties_hot_ch = props_sw.SeawaterParameterBlock()
    m.fs.properties_cold_ch = props_sw.SeawaterParameterBlock()
    m.fs.properties_vapor = props_w.WaterParameterBlock()
    m.fs.unit = MembraneDistillation0D(
        hot_ch={
            "property_package": m.fs.properties_hot_ch,
            "property_package_vapor": m.fs.properties_vapor,
            "has_pressure_change": True,
            "concentration_polarization_type": ConcentrationPolarizationType.none,
            "mass_transfer_coefficient": MassTransferCoefficient.none,
            "pressure_change_type": PressureChangeType.calculated,
        },
        cold_ch={
            "property_package": m.fs.properties_cold_ch,
            "property_package_vapor": m.fs.properties_vapor,
            "has_pressure_change": True,
            "pressure_change_type": PressureChangeType.calculated,
        },
    )

    assert (
        m.fs.unit.config.hot_ch.concentration_polarization_type
        == ConcentrationPolarizationType.none
    )
    assert (
        m.fs.unit.config.hot_ch.mass_transfer_coefficient
        == MassTransferCoefficient.none
    )
    assert m.fs.unit.config.hot_ch.pressure_change_type == PressureChangeType.calculated
    assert (
        m.fs.unit.config.cold_ch.pressure_change_type == PressureChangeType.calculated
    )
    assert isinstance(m.fs.unit.hot_ch.deltaP, Var)
    assert isinstance(m.fs.unit.hot_ch.channel_height, Var)
    assert isinstance(m.fs.unit.width, Var)
    assert isinstance(m.fs.unit.length, Var)
    assert isinstance(m.fs.unit.hot_ch.dh, Var)
    assert isinstance(m.fs.unit.hot_ch.spacer_porosity, Var)
    assert isinstance(m.fs.unit.hot_ch.N_Re, Var)

    assert isinstance(m.fs.unit.cold_ch.deltaP, Var)
    assert isinstance(m.fs.unit.cold_ch.channel_height, Var)
    assert isinstance(m.fs.unit.cold_ch.dh, Var)
    assert isinstance(m.fs.unit.cold_ch.spacer_porosity, Var)
    assert isinstance(m.fs.unit.cold_ch.N_Re, Var)
    assert isinstance(m.fs.unit.eq_area, Constraint)


@pytest.mark.unit
def test_option_friction_factor_spiral_wound():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.properties_hot_ch = props_sw.SeawaterParameterBlock()
    m.fs.properties_cold_ch = props_sw.SeawaterParameterBlock()
    m.fs.properties_vapor = props_w.WaterParameterBlock()
    m.fs.unit = MembraneDistillation0D(
        hot_ch={
            "property_package": m.fs.properties_hot_ch,
            "property_package_vapor": m.fs.properties_vapor,
            "has_pressure_change": True,
            "concentration_polarization_type": ConcentrationPolarizationType.none,
            "mass_transfer_coefficient": MassTransferCoefficient.none,
            "pressure_change_type": PressureChangeType.calculated,
            "friction_factor": FrictionFactor.spiral_wound,
        },
        cold_ch={
            "property_package": m.fs.properties_cold_ch,
            "property_package_vapor": m.fs.properties_vapor,
            "has_pressure_change": True,
            "pressure_change_type": PressureChangeType.calculated,
            "friction_factor": FrictionFactor.spiral_wound,
        },
    )
    # it's probably better to unify friction factor for both side in DCMD (i.e. friction factor be config option not config.channel option. However keeping them separate might become useful for other config types)
    assert m.fs.unit.config.hot_ch.friction_factor == FrictionFactor.spiral_wound
    assert m.fs.unit.config.cold_ch.friction_factor == FrictionFactor.spiral_wound

    assert isinstance(m.fs.unit.hot_ch.velocity, Var)
    assert isinstance(m.fs.unit.hot_ch.eq_friction_factor, Constraint)
    assert isinstance(m.fs.unit.cold_ch.velocity, Var)
    assert isinstance(m.fs.unit.cold_ch.eq_friction_factor, Constraint)


class TestMembraneDistillation:
    @pytest.fixture(scope="class")
    def MD_frame(self):
        m = ConcreteModel()
        m.fs = FlowsheetBlock(dynamic=False)
        m.fs.properties_hot_ch = props_sw.SeawaterParameterBlock()
        m.fs.properties_cold_ch = props_sw.SeawaterParameterBlock()
        m.fs.properties_vapor = props_w.WaterParameterBlock()
        m.fs.unit = MembraneDistillation0D(
            hot_ch={
                "property_package": m.fs.properties_hot_ch,
                "property_package_vapor": m.fs.properties_vapor,
                "has_pressure_change": True,
                "temperature_polarization_type": TemperaturePolarizationType.fixed,
                "concentration_polarization_type": ConcentrationPolarizationType.fixed,
                "mass_transfer_coefficient": MassTransferCoefficient.none,
                "pressure_change_type": PressureChangeType.fixed_per_stage,
                "flow_direction": FlowDirection.forward,
            },
            cold_ch={
                "property_package": m.fs.properties_cold_ch,
                "property_package_vapor": m.fs.properties_vapor,
                "has_pressure_change": True,
                "temperature_polarization_type": TemperaturePolarizationType.fixed,
                "mass_transfer_coefficient": MassTransferCoefficient.none,
                "concentration_polarization_type": ConcentrationPolarizationType.none,
                "pressure_change_type": PressureChangeType.fixed_per_stage,
                "flow_direction": FlowDirection.backward,
            },
        )

        # fully specify system
        hot_ch_flow_mass = 1
        hot_ch_mass_frac_TDS = 0.035
        hot_ch_pressure = 7e5
        membrane_pressure_drop = -0.5e5
        membrane_area = 12
        hot_ch_mass_frac_H2O = 1 - hot_ch_mass_frac_TDS
        m.fs.unit.hot_ch_inlet.flow_mass_phase_comp[0, "Liq", "TDS"].fix(
            hot_ch_flow_mass * hot_ch_mass_frac_TDS
        )
        m.fs.unit.hot_ch_inlet.flow_mass_phase_comp[0, "Liq", "H2O"].fix(
            hot_ch_flow_mass * hot_ch_mass_frac_H2O
        )

        m.fs.unit.hot_ch_inlet.pressure[0].fix(hot_ch_pressure)
        m.fs.unit.hot_ch_inlet.temperature[0].fix(273.15 + 90)
        m.fs.unit.area.fix(membrane_area)
        m.fs.unit.permeability_coef.fix(1e-10)
        m.fs.unit.membrane_thickness.fix(1e-4)
        m.fs.unit.membrane_tc.fix(0.0002)
        m.fs.unit.hot_ch.cp_modulus.fix(1)

        m.fs.unit.cold_ch_inlet.flow_mass_phase_comp[0, "Liq", "H2O"].fix(
            hot_ch_flow_mass
        )
        m.fs.unit.cold_ch_inlet.flow_mass_phase_comp[0, "Liq", "TDS"].fix(0)
        m.fs.unit.cold_ch_inlet.pressure[0].fix(101325)
        m.fs.unit.cold_ch_inlet.temperature[0].fix(273.15 + 25)

        m.fs.unit.hot_ch.deltaP.fix(0)
        m.fs.unit.cold_ch.deltaP.fix(0)

        m.fs.unit.hot_ch.h_conv.fix(2400)
        m.fs.unit.cold_ch.h_conv.fix(2400)

        return m

    @pytest.mark.unit
    def test_build(self, MD_frame):
        m = MD_frame

        # test ports
        port_lst = ["hot_ch_inlet", "hot_ch_outlet", "cold_ch_inlet", "cold_ch_outlet"]
        for port_str in port_lst:
            port = getattr(m.fs.unit, port_str)
            assert isinstance(port, Port)
            # number of state variables for seawater property package
            assert len(port.vars) == 3

        assert isinstance(m.fs.unit.hot_ch, MDChannel0DBlock)
        assert isinstance(m.fs.unit.cold_ch, MDChannel0DBlock)

        # test statistics
        assert number_variables(m) == 384
        assert number_total_constraints(m) == 132
        assert number_unused_variables(m) == 138

    @pytest.mark.unit
    def test_dof(self, MD_frame):
        m = MD_frame
        assert degrees_of_freedom(m) == 0  # your original assertion

    @pytest.mark.unit
    def test_calculate_scaling(self, MD_frame):
        m = MD_frame

        m.fs.properties_hot_ch.set_default_scaling(
            "flow_mass_phase_comp", 1, index=("Liq", "H2O")
        )
        m.fs.properties_hot_ch.set_default_scaling(
            "flow_mass_phase_comp", 1e2, index=("Liq", "TDS")
        )

        m.fs.properties_cold_ch.set_default_scaling(
            "flow_mass_phase_comp", 1, index=("Liq", "H2O")
        )
        m.fs.properties_cold_ch.set_default_scaling(
            "flow_mass_phase_comp", 1e2, index=("Liq", "TDS")
        )

        m.fs.properties_cold_ch.set_default_scaling(
            "flow_mass_phase_comp", 1e2, index=("Liq", "TDS")
        )

        m.fs.properties_vapor.set_default_scaling(
            "flow_mass_phase_comp", 1, index=("Vap", "H2O")
        )
        m.fs.properties_vapor.set_default_scaling(
            "flow_mass_phase_comp", 1, index=("Liq", "H2O")
        )
        calculate_scaling_factors(m)

        # check that all variables have scaling factors
        unscaled_var_list = list(unscaled_variables_generator(m))
        print("Unscaled Variables:")
        for var in unscaled_var_list:
            print(var.name)
        assert len(unscaled_var_list) == 0

        for i in badly_scaled_var_generator(m):
            print(i[0].name, i[1])

    @pytest.mark.component
    def test_initialize(self, MD_frame):
        initialization_tester(MD_frame, outlvl=idaeslog.DEBUG)
        m = ConcreteModel()
        m.fs = FlowsheetBlock(dynamic=False)
        m.fs.properties_hot_ch = props_sw.SeawaterParameterBlock()
        m.fs.properties_cold_ch = props_sw.SeawaterParameterBlock()
        m.fs.properties_vapor = props_w.WaterParameterBlock()
        m.fs.unit = MembraneDistillation0D(
            hot_ch={
                "property_package": m.fs.properties_hot_ch,
                "property_package_vapor": m.fs.properties_vapor,
                "has_pressure_change": True,
                "temperature_polarization_type": TemperaturePolarizationType.fixed,
                "concentration_polarization_type": ConcentrationPolarizationType.fixed,
                "mass_transfer_coefficient": MassTransferCoefficient.none,
                "pressure_change_type": PressureChangeType.fixed_per_stage,
                "flow_direction": FlowDirection.forward,
            },
            cold_ch={
                "property_package": m.fs.properties_cold_ch,
                "property_package_vapor": m.fs.properties_vapor,
                "has_pressure_change": True,
                "temperature_polarization_type": TemperaturePolarizationType.fixed,
                "mass_transfer_coefficient": MassTransferCoefficient.none,
                "concentration_polarization_type": ConcentrationPolarizationType.none,
                "pressure_change_type": PressureChangeType.fixed_per_stage,
                "flow_direction": FlowDirection.backward,
            },
        )

        # fully specify system
        hot_ch_flow_mass = 1
        hot_ch_mass_frac_TDS = 0.035
        hot_ch_pressure = 7e5
        membrane_pressure_drop = -0.5e5
        membrane_area = 12
        hot_ch_mass_frac_H2O = 1 - hot_ch_mass_frac_TDS
        m.fs.unit.hot_ch_inlet.flow_mass_phase_comp[0, "Liq", "TDS"].fix(
            hot_ch_flow_mass * hot_ch_mass_frac_TDS
        )
        m.fs.unit.hot_ch_inlet.flow_mass_phase_comp[0, "Liq", "H2O"].fix(
            hot_ch_flow_mass * hot_ch_mass_frac_H2O
        )

        m.fs.unit.hot_ch_inlet.pressure[0].fix(hot_ch_pressure)
        m.fs.unit.hot_ch_inlet.temperature[0].fix(273.15 + 90)
        m.fs.unit.area.fix(membrane_area)
        m.fs.unit.permeability_coef.fix(1e-10)
        m.fs.unit.membrane_thickness.fix(1e-4)
        m.fs.unit.membrane_tc.fix(0.0002)
        m.fs.unit.hot_ch.cp_modulus.fix(1)

        m.fs.unit.cold_ch_inlet.flow_mass_phase_comp[0, "Liq", "H2O"].fix(
            hot_ch_flow_mass
        )
        m.fs.unit.cold_ch_inlet.flow_mass_phase_comp[0, "Liq", "TDS"].fix(0)
        m.fs.unit.cold_ch_inlet.pressure[0].fix(101325)
        m.fs.unit.cold_ch_inlet.temperature[0].fix(273.15 + 25)

        m.fs.unit.hot_ch.deltaP.fix(0)
        # m.fs.unit.cold_ch.deltaP.fix(0)

        m.fs.unit.hot_ch.h_conv.fix(2400)
        m.fs.unit.cold_ch.h_conv.fix(2400)
        
    @pytest.mark.component
    def test_initialize(self, MD_frame):
        initialization_tester(MD_frame, outlvl=idaeslog.DEBUG)

    @pytest.mark.component
    def test_var_scaling(self, MD_frame):
        m = MD_frame
        badly_scaled_var_lst = list(badly_scaled_var_generator(m))
        [print(i[0], i[1]) for i in badly_scaled_var_lst]
        assert badly_scaled_var_lst == []

    @pytest.mark.component
    def test_solve(self, MD_frame):
        m = MD_frame
        results = solver.solve(m)

        # Check for optimal solution
        assert_optimal_termination(results) 

    
    @pytest.mark.component
    def test_solution(self, MD_frame):
        m = MD_frame

        # Adjusted the flux value to 12 LMH
        assert pytest.approx(0.0033, rel=1e-3) == value(
            m.fs.unit.flux_mass[0, "Liq", "H2O"]
        )

        # Replaced NaCl with TDS
        assert pytest.approx(0.5873e-8, rel=1e-3) == value(
            m.fs.unit.flux_mass[0, "Liq", "TDS"]
        )

        # Replaced feed_side with hot_ch
        assert pytest.approx(0.9545, rel=1e-3) == value(
            m.fs.unit.hot_ch_outlet.flow_mass_phase_comp[0, "Liq", "H2O"]
        )

        # Replaced NaCl with TDS and feed_side with hot_ch
        assert pytest.approx(0.03423, rel=1e-3) == value(
            m.fs.unit.hot_ch_outlet.flow_mass_phase_comp[0, "Liq", "TDS"]
        )

        # Added assertion for outlet temperature of hot_ch
        assert pytest.approx(300, rel=1e-3) == value(
            m.fs.unit.hot_ch_outlet.temperature[273.15+77]
        )

        # Added assertion for outlet temperature of cold_ch
        assert pytest.approx(280, rel=1e-3) == value(
            m.fs.unit.cold_ch_outlet.temperature[273.15+15.6]
        )

