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
Tests for dewatering unit example.
"""
from io import StringIO
import pytest
from pyomo.environ import (
    ConcreteModel,
    value,
    assert_optimal_termination,
    Suffix,
    TransformationFactory,
    units as pyunits,
)

from idaes.core import (
    FlowsheetBlock,
    MaterialBalanceType,
    MomentumBalanceType,
)

from idaes.core.util.scaling import (
    get_jacobian,
    jacobian_cond,
)
from idaes.core.scaling.scaling_base import ScalerBase
from idaes.core.scaling.scaler_profiling import ScalingProfiler

from idaes.models.unit_models.separator import SplittingType

from pyomo.environ import (
    units,
)

from watertap.core.solvers import get_solver
from idaes.core.util.model_statistics import (
    degrees_of_freedom,
    number_variables,
    number_total_constraints,
    number_unused_variables,
)
import idaes.core.util.scaling as iscale
from idaes.core.util.testing import (
    initialization_tester,
)

from idaes.core.util.exceptions import (
    ConfigurationError,
)

from watertap.unit_models.dewatering import (
    DewateringUnit,
    ActivatedSludgeModelType,
    DewatererScaler,
)
from watertap.property_models.unit_specific.activated_sludge.asm1_properties import (
    ASM1ParameterBlock,
    ASM1PropertiesScaler,
)

from watertap.property_models.unit_specific.activated_sludge.asm2d_properties import (
    ASM2dParameterBlock,
)
from watertap.property_models.unit_specific.activated_sludge.modified_asm2d_properties import (
    ModifiedASM2dParameterBlock,
)
from pyomo.util.check_units import assert_units_consistent
from watertap.costing import WaterTAPCosting
from watertap.costing.unit_models.dewatering import (
    cost_dewatering,
    cost_centrifuge,
    DewateringType,
)
from idaes.core import UnitModelCostingBlock


__author__ = "Alejandro Garciadiego, Adam Atia"

# -----------------------------------------------------------------------------
# Get default solver for testing
solver = get_solver()


# -----------------------------------------------------------------------------
@pytest.mark.unit
def test_config():
    m = ConcreteModel()

    m.fs = FlowsheetBlock(dynamic=False)

    m.fs.props = ASM1ParameterBlock()

    m.fs.unit = DewateringUnit(property_package=m.fs.props)

    assert len(m.fs.unit.config) == 16

    assert not m.fs.unit.config.dynamic
    assert not m.fs.unit.config.has_holdup
    assert m.fs.unit.config.material_balance_type == MaterialBalanceType.useDefault
    assert m.fs.unit.config.momentum_balance_type == MomentumBalanceType.pressureTotal
    assert m.fs.unit.config.activated_sludge_model == ActivatedSludgeModelType.ASM1

    assert "underflow" in m.fs.unit.config.outlet_list
    assert "overflow" in m.fs.unit.config.outlet_list
    assert SplittingType.componentFlow is m.fs.unit.config.split_basis


@pytest.mark.unit
def test_list_error():
    m = ConcreteModel()

    m.fs = FlowsheetBlock(dynamic=False)

    m.fs.props = ASM1ParameterBlock()

    with pytest.raises(
        ConfigurationError,
        match="fs.unit encountered unrecognised "
        "outlet_list. This should not "
        "occur - please use overflow "
        "and underflow as outlets.",
    ):
        m.fs.unit = DewateringUnit(
            property_package=m.fs.props,
            outlet_list=["outlet1", "outlet2"],
        )


# -----------------------------------------------------------------------------
class TestDu(object):
    @pytest.fixture(scope="class")
    def du(self):
        m = ConcreteModel()
        m.fs = FlowsheetBlock(dynamic=False)

        m.fs.props = ASM1ParameterBlock()

        m.fs.unit = DewateringUnit(property_package=m.fs.props)

        m.fs.unit.inlet.flow_vol.fix(178.4674 * units.m**3 / units.day)
        m.fs.unit.inlet.temperature.fix(308.15 * units.K)
        m.fs.unit.inlet.pressure.fix(1 * units.atm)

        m.fs.unit.inlet.conc_mass_comp[0, "S_I"].fix(130.867 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_S"].fix(258.5789 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_I"].fix(
            17216.2434 * units.mg / units.liter
        )
        m.fs.unit.inlet.conc_mass_comp[0, "X_S"].fix(2611.4843 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_BH"].fix(1e-6 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_BA"].fix(1e-6 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_P"].fix(626.0652 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_O"].fix(1e-6 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_NO"].fix(1e-6 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_NH"].fix(
            1442.7882 * units.mg / units.liter
        )
        m.fs.unit.inlet.conc_mass_comp[0, "S_ND"].fix(0.54323 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_ND"].fix(100.8668 * units.mg / units.liter)
        m.fs.unit.inlet.alkalinity.fix(97.8459 * units.mol / units.m**3)

        m.fs.unit.hydraulic_retention_time.fix()

        return m

    @pytest.mark.build
    @pytest.mark.unit
    def test_build(self, du):
        assert hasattr(du.fs.unit, "hydraulic_retention_time")
        assert hasattr(du.fs.unit, "volume")
        assert hasattr(du.fs.unit, "inlet")
        assert len(du.fs.unit.inlet.vars) == 5
        assert hasattr(du.fs.unit.inlet, "flow_vol")
        assert hasattr(du.fs.unit.inlet, "conc_mass_comp")
        assert hasattr(du.fs.unit.inlet, "temperature")
        assert hasattr(du.fs.unit.inlet, "pressure")
        assert hasattr(du.fs.unit.inlet, "alkalinity")

        assert hasattr(du.fs.unit, "underflow")
        assert len(du.fs.unit.underflow.vars) == 5
        assert hasattr(du.fs.unit.underflow, "flow_vol")
        assert hasattr(du.fs.unit.underflow, "conc_mass_comp")
        assert hasattr(du.fs.unit.underflow, "temperature")
        assert hasattr(du.fs.unit.underflow, "pressure")
        assert hasattr(du.fs.unit.underflow, "alkalinity")

        assert hasattr(du.fs.unit, "overflow")
        assert len(du.fs.unit.overflow.vars) == 5
        assert hasattr(du.fs.unit.overflow, "flow_vol")
        assert hasattr(du.fs.unit.overflow, "conc_mass_comp")
        assert hasattr(du.fs.unit.overflow, "temperature")
        assert hasattr(du.fs.unit.overflow, "pressure")
        assert hasattr(du.fs.unit.overflow, "alkalinity")

        assert number_variables(du) == 83
        assert number_total_constraints(du) == 62
        assert number_unused_variables(du) == 4

    @pytest.mark.unit
    def test_dof(self, du):
        assert degrees_of_freedom(du) == 0

    @pytest.mark.unit
    def test_units(self, du):
        assert_units_consistent(du)

    @pytest.mark.solver
    @pytest.mark.skipif(solver is None, reason="Solver not available")
    @pytest.mark.component
    def test_initialize(self, du):
        iscale.calculate_scaling_factors(du)
        initialization_tester(du)

    @pytest.mark.solver
    @pytest.mark.skipif(solver is None, reason="Solver not available")
    @pytest.mark.component
    def test_solve(self, du):
        solver = get_solver()
        results = solver.solve(du)
        assert_optimal_termination(results)

    @pytest.mark.solver
    @pytest.mark.skipif(solver is None, reason="Solver not available")
    @pytest.mark.component
    def test_solution(self, du):
        assert pytest.approx(101325.0, rel=1e-3) == value(
            du.fs.unit.overflow.pressure[0]
        )
        assert pytest.approx(308.15, rel=1e-3) == value(
            du.fs.unit.overflow.temperature[0]
        )
        assert pytest.approx(0.001954, rel=1e-3) == value(
            du.fs.unit.overflow.flow_vol[0]
        )
        assert pytest.approx(0.1308, rel=1e-3) == value(
            du.fs.unit.overflow.conc_mass_comp[0, "S_I"]
        )
        assert pytest.approx(0.2585, rel=1e-3) == value(
            du.fs.unit.overflow.conc_mass_comp[0, "S_S"]
        )
        assert pytest.approx(0.3638, rel=1e-3) == value(
            du.fs.unit.overflow.conc_mass_comp[0, "X_I"]
        )
        assert pytest.approx(0.0552, rel=1e-3) == value(
            du.fs.unit.overflow.conc_mass_comp[0, "X_S"]
        )
        assert value(du.fs.unit.overflow.conc_mass_comp[0, "X_BH"]) <= 1e-6
        assert value(du.fs.unit.overflow.conc_mass_comp[0, "X_BA"]) <= 1e-6
        assert pytest.approx(0.01323, rel=1e-3) == value(
            du.fs.unit.overflow.conc_mass_comp[0, "X_P"]
        )
        assert value(du.fs.unit.overflow.conc_mass_comp[0, "S_O"]) <= 1e-6
        assert value(du.fs.unit.overflow.conc_mass_comp[0, "S_NO"]) <= 1e-6
        assert pytest.approx(1.4427, rel=1e-3) == value(
            du.fs.unit.overflow.conc_mass_comp[0, "S_NH"]
        )
        assert pytest.approx(0.000543, rel=1e-3) == value(
            du.fs.unit.overflow.conc_mass_comp[0, "S_ND"]
        )
        assert pytest.approx(0.00213, rel=1e-3) == value(
            du.fs.unit.overflow.conc_mass_comp[0, "X_ND"]
        )
        assert pytest.approx(0.09784, rel=1e-3) == value(
            du.fs.unit.overflow.alkalinity[0]
        )
        assert pytest.approx(3.718, rel=1e-3) == value(du.fs.unit.volume[0])

    @pytest.mark.solver
    @pytest.mark.skipif(solver is None, reason="Solver not available")
    @pytest.mark.component
    def test_conservation(self, du):
        assert (
            abs(
                value(
                    du.fs.unit.inlet.flow_vol[0] * du.fs.props.dens_mass
                    - du.fs.unit.overflow.flow_vol[0] * du.fs.props.dens_mass
                    - du.fs.unit.underflow.flow_vol[0] * du.fs.props.dens_mass
                )
            )
            <= 1e-6
        )
        for i in du.fs.props.solute_set:
            assert (
                abs(
                    value(
                        du.fs.unit.inlet.flow_vol[0]
                        * du.fs.unit.inlet.conc_mass_comp[0, i]
                        - du.fs.unit.overflow.flow_vol[0]
                        * du.fs.unit.overflow.conc_mass_comp[0, i]
                        - du.fs.unit.underflow.flow_vol[0]
                        * du.fs.unit.underflow.conc_mass_comp[0, i]
                    )
                )
                <= 1e-6
            )

    @pytest.mark.unit
    def test_report(self, du):
        du.fs.unit.report()


class TestDUASM2d(object):
    @pytest.fixture(scope="class")
    def du_asm2d(self):
        m = ConcreteModel()
        m.fs = FlowsheetBlock(dynamic=False)

        m.fs.props = ASM2dParameterBlock()

        m.fs.unit = DewateringUnit(
            property_package=m.fs.props,
            activated_sludge_model=ActivatedSludgeModelType.ASM2D,
        )

        # NOTE: Concentrations of exactly 0 result in singularities, use EPS instead
        EPS = 1e-8

        m.fs.unit.inlet.flow_vol.fix(300 * units.m**3 / units.day)
        m.fs.unit.inlet.temperature.fix(308.15 * units.K)
        m.fs.unit.inlet.pressure.fix(1 * units.atm)
        m.fs.unit.inlet.conc_mass_comp[0, "S_O2"].fix(7.9707 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_N2"].fix(29.0603 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_NH4"].fix(8.0209 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_NO3"].fix(6.6395 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_PO4"].fix(7.8953 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_F"].fix(0.4748 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_A"].fix(0.0336 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_I"].fix(30 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_I"].fix(1695.7695 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_S"].fix(68.2975 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_H"].fix(1855.5067 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_PAO"].fix(
            214.5319 * units.mg / units.liter
        )
        m.fs.unit.inlet.conc_mass_comp[0, "X_PP"].fix(63.5316 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_PHA"].fix(2.7381 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_AUT"].fix(
            118.3582 * units.mg / units.liter
        )
        m.fs.unit.inlet.conc_mass_comp[0, "X_MeOH"].fix(EPS * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_MeP"].fix(EPS * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_TSS"].fix(
            3525.429 * units.mg / units.liter
        )
        m.fs.unit.inlet.alkalinity[0].fix(4.6663 * units.mmol / units.liter)

        m.fs.unit.hydraulic_retention_time.fix()

        return m

    @pytest.mark.unit
    def test_dof(self, du_asm2d):
        assert degrees_of_freedom(du_asm2d) == 0

    @pytest.mark.unit
    def test_units(self, du_asm2d):
        assert_units_consistent(du_asm2d)

    @pytest.mark.solver
    @pytest.mark.skipif(solver is None, reason="Solver not available")
    @pytest.mark.component
    def test_initialize(self, du_asm2d):

        iscale.calculate_scaling_factors(du_asm2d)
        initialization_tester(du_asm2d)

    @pytest.mark.solver
    @pytest.mark.skipif(solver is None, reason="Solver not available")
    @pytest.mark.component
    def test_solve(self, du_asm2d):
        solver = get_solver()
        results = solver.solve(du_asm2d)
        assert_optimal_termination(results)


class TestDUModifiedASM2d(object):
    @pytest.fixture(scope="class")
    def du_mod_asm2d(self):
        m = ConcreteModel()
        m.fs = FlowsheetBlock(dynamic=False)

        m.fs.props = ModifiedASM2dParameterBlock()

        m.fs.unit = DewateringUnit(
            property_package=m.fs.props,
            activated_sludge_model=ActivatedSludgeModelType.modified_ASM2D,
        )

        # NOTE: Concentrations of exactly 0 result in singularities, use EPS instead
        EPS = 1e-8

        m.fs.unit.inlet.flow_vol.fix(300 * units.m**3 / units.day)
        m.fs.unit.inlet.temperature.fix(308.15 * units.K)
        m.fs.unit.inlet.pressure.fix(1 * units.atm)
        m.fs.unit.inlet.conc_mass_comp[0, "S_O2"].fix(7.9707 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_N2"].fix(29.0603 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_NH4"].fix(8.0209 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_NO3"].fix(6.6395 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_PO4"].fix(7.8953 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_F"].fix(0.4748 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_A"].fix(0.0336 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_I"].fix(30 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_K"].fix(7 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_Mg"].fix(6 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_IC"].fix(10 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_I"].fix(1695.7695 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_S"].fix(68.2975 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_H"].fix(1855.5067 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_PAO"].fix(
            214.5319 * units.mg / units.liter
        )
        m.fs.unit.inlet.conc_mass_comp[0, "X_PP"].fix(63.5316 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_PHA"].fix(2.7381 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_AUT"].fix(
            118.3582 * units.mg / units.liter
        )
        m.fs.unit.hydraulic_retention_time.fix()

        return m

    @pytest.mark.unit
    def test_dof(self, du_mod_asm2d):
        assert degrees_of_freedom(du_mod_asm2d) == 0

    @pytest.mark.unit
    def test_units(self, du_mod_asm2d):
        assert_units_consistent(du_mod_asm2d)

    @pytest.mark.solver
    @pytest.mark.skipif(solver is None, reason="Solver not available")
    @pytest.mark.component
    def test_initialize(self, du_mod_asm2d):

        iscale.calculate_scaling_factors(du_mod_asm2d)
        initialization_tester(du_mod_asm2d)

    @pytest.mark.solver
    @pytest.mark.skipif(solver is None, reason="Solver not available")
    @pytest.mark.component
    def test_solve(self, du_mod_asm2d):
        solver = get_solver()
        results = solver.solve(du_mod_asm2d)
        assert_optimal_termination(results)


@pytest.mark.solver
@pytest.mark.skipif(solver is None, reason="Solver not available")
@pytest.mark.component
def test_du_default_costing():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)

    m.fs.props = ASM1ParameterBlock()

    m.fs.unit = DewateringUnit(property_package=m.fs.props)

    m.fs.unit.inlet.flow_vol.fix(178.4674 * units.m**3 / units.day)
    m.fs.unit.inlet.temperature.fix(308.15 * units.K)
    m.fs.unit.inlet.pressure.fix(1 * units.atm)

    m.fs.unit.inlet.conc_mass_comp[0, "S_I"].fix(130.867 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "S_S"].fix(258.5789 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "X_I"].fix(17216.2434 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "X_S"].fix(2611.4843 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "X_BH"].fix(1e-6 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "X_BA"].fix(1e-6 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "X_P"].fix(626.0652 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "S_O"].fix(1e-6 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "S_NO"].fix(1e-6 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "S_NH"].fix(1442.7882 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "S_ND"].fix(0.54323 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "X_ND"].fix(100.8668 * units.mg / units.liter)
    m.fs.unit.inlet.alkalinity.fix(97.8459 * units.mol / units.m**3)

    m.fs.unit.hydraulic_retention_time.fix()

    m.fs.costing = WaterTAPCosting()

    m.fs.unit.costing = UnitModelCostingBlock(
        flowsheet_costing_block=m.fs.costing,
    )

    m.fs.costing.cost_process()

    assert degrees_of_freedom(m) == 0

    results = solver.solve(m)

    assert_optimal_termination(results)

    assert hasattr(m.fs.costing, "centrifuge")
    assert m.fs.unit.default_costing_method is cost_dewatering
    assert value(m.fs.costing.centrifuge.capital_a_parameter) == 328.03
    assert value(m.fs.costing.centrifuge.capital_b_parameter) == 751295

    # Check solutions
    assert pytest.approx(1964.42, rel=1e-5) == value(
        pyunits.convert(m.fs.unit.inlet.flow_vol[0], to_units=pyunits.gal / pyunits.hr)
    )
    assert pytest.approx(1602087.9 * 2, rel=1e-5) == value(
        m.fs.unit.costing.capital_cost
    )
    assert pytest.approx(1602087.9, rel=1e-5) == value(
        pyunits.convert(
            (328.03 * 1964.42 + 751295) * pyunits.USD_2007,
            to_units=m.fs.costing.base_currency,
        )
    )


@pytest.mark.solver
@pytest.mark.skipif(solver is None, reason="Solver not available")
@pytest.mark.component
def test_du_centrifuge_costing():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)

    m.fs.props = ASM1ParameterBlock()

    m.fs.unit = DewateringUnit(property_package=m.fs.props)

    m.fs.unit.inlet.flow_vol.fix(178.4674 * units.m**3 / units.day)
    m.fs.unit.inlet.temperature.fix(308.15 * units.K)
    m.fs.unit.inlet.pressure.fix(1 * units.atm)

    m.fs.unit.inlet.conc_mass_comp[0, "S_I"].fix(130.867 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "S_S"].fix(258.5789 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "X_I"].fix(17216.2434 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "X_S"].fix(2611.4843 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "X_BH"].fix(1e-6 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "X_BA"].fix(1e-6 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "X_P"].fix(626.0652 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "S_O"].fix(1e-6 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "S_NO"].fix(1e-6 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "S_NH"].fix(1442.7882 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "S_ND"].fix(0.54323 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "X_ND"].fix(100.8668 * units.mg / units.liter)
    m.fs.unit.inlet.alkalinity.fix(97.8459 * units.mol / units.m**3)

    m.fs.unit.hydraulic_retention_time.fix()

    m.fs.costing = WaterTAPCosting()

    m.fs.unit.costing = UnitModelCostingBlock(
        flowsheet_costing_block=m.fs.costing,
        costing_method=cost_centrifuge,
    )
    # Using average specific energy consumption of 0.069 for centrifuge as a function of capacity
    m.fs.unit.energy_electric_flow_vol_inlet[0] = 0.069 * pyunits.kWh / pyunits.m**3
    m.fs.costing.cost_process()

    assert degrees_of_freedom(m) == 0

    results = solver.solve(m)

    assert_optimal_termination(results)

    assert hasattr(m.fs.costing, "centrifuge")
    assert value(m.fs.costing.centrifuge.capital_a_parameter) == 328.03
    assert value(m.fs.costing.centrifuge.capital_b_parameter) == 751295

    # Check solutions
    assert pytest.approx(1602087.9 * 2, rel=1e-5) == value(
        m.fs.unit.costing.capital_cost
    )
    assert pytest.approx(1602087.9, rel=1e-5) == value(
        pyunits.convert(
            (328.03 * 1964.42 + 751295) * pyunits.USD_2007,
            to_units=m.fs.costing.base_currency,
        )
    )
    assert pytest.approx(7.4361 * 0.069, rel=1e-5) == value(
        m.fs.unit.electricity_consumption[0]
    )


@pytest.mark.solver
@pytest.mark.skipif(solver is None, reason="Solver not available")
@pytest.mark.component
def test_du_centrifuge_costing2():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)

    m.fs.props = ASM1ParameterBlock()

    m.fs.unit = DewateringUnit(property_package=m.fs.props)

    m.fs.unit.inlet.flow_vol.fix(178.4674 * units.m**3 / units.day)
    m.fs.unit.inlet.temperature.fix(308.15 * units.K)
    m.fs.unit.inlet.pressure.fix(1 * units.atm)

    m.fs.unit.inlet.conc_mass_comp[0, "S_I"].fix(130.867 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "S_S"].fix(258.5789 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "X_I"].fix(17216.2434 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "X_S"].fix(2611.4843 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "X_BH"].fix(1e-6 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "X_BA"].fix(1e-6 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "X_P"].fix(626.0652 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "S_O"].fix(1e-6 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "S_NO"].fix(1e-6 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "S_NH"].fix(1442.7882 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "S_ND"].fix(0.54323 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "X_ND"].fix(100.8668 * units.mg / units.liter)
    m.fs.unit.inlet.alkalinity.fix(97.8459 * units.mol / units.m**3)

    m.fs.unit.hydraulic_retention_time.fix()

    m.fs.costing = WaterTAPCosting()

    m.fs.unit.costing = UnitModelCostingBlock(
        flowsheet_costing_block=m.fs.costing,
        costing_method=cost_dewatering,
        costing_method_arguments={
            "dewatering_type": DewateringType.centrifuge,
            "cost_electricity_flow": False,
        },
    )

    m.fs.costing.cost_process()

    assert degrees_of_freedom(m) == 0

    results = solver.solve(m)

    assert_optimal_termination(results)

    assert hasattr(m.fs.costing, "centrifuge")
    assert value(m.fs.costing.centrifuge.capital_a_parameter) == 328.03
    assert value(m.fs.costing.centrifuge.capital_b_parameter) == 751295
    assert "electricity" not in m.fs.costing.aggregate_flow_costs.keys()

    # Check solutions
    assert pytest.approx(1602087.9 * 2, rel=1e-5) == value(
        m.fs.unit.costing.capital_cost
    )
    assert pytest.approx(1602087.9, rel=1e-5) == value(
        pyunits.convert(
            (328.03 * 1964.42 + 751295) * pyunits.USD_2007,
            to_units=m.fs.costing.base_currency,
        )
    )


@pytest.mark.solver
@pytest.mark.skipif(solver is None, reason="Solver not available")
@pytest.mark.component
def test_du_filter_plate_press_costing():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)

    m.fs.props = ASM1ParameterBlock()

    m.fs.unit = DewateringUnit(property_package=m.fs.props)

    m.fs.unit.inlet.flow_vol.fix(178.4674 * units.m**3 / units.day)
    m.fs.unit.inlet.temperature.fix(308.15 * units.K)
    m.fs.unit.inlet.pressure.fix(1 * units.atm)

    m.fs.unit.inlet.conc_mass_comp[0, "S_I"].fix(130.867 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "S_S"].fix(258.5789 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "X_I"].fix(17216.2434 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "X_S"].fix(2611.4843 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "X_BH"].fix(1e-6 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "X_BA"].fix(1e-6 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "X_P"].fix(626.0652 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "S_O"].fix(1e-6 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "S_NO"].fix(1e-6 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "S_NH"].fix(1442.7882 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "S_ND"].fix(0.54323 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "X_ND"].fix(100.8668 * units.mg / units.liter)
    m.fs.unit.inlet.alkalinity.fix(97.8459 * units.mol / units.m**3)

    m.fs.unit.hydraulic_retention_time.fix()

    m.fs.costing = WaterTAPCosting()

    m.fs.unit.costing = UnitModelCostingBlock(
        flowsheet_costing_block=m.fs.costing,
        costing_method=cost_dewatering,
        costing_method_arguments={
            "dewatering_type": DewateringType.filter_plate_press,
            "cost_electricity_flow": True,
        },
    )
    # Using average specific energy consumption of 0.0039 for screw press as a function of capacity
    m.fs.unit.energy_electric_flow_vol_inlet[0] = 0.0039 * pyunits.kWh / pyunits.m**3
    m.fs.costing.cost_process()

    assert degrees_of_freedom(m) == 0

    results = solver.solve(m)

    assert_optimal_termination(results)
    assert_units_consistent(m)
    assert hasattr(m.fs.costing, "filter_plate_press")
    assert value(m.fs.costing.filter_plate_press.capital_a_parameter) == 102794
    assert value(m.fs.costing.filter_plate_press.capital_b_parameter) == 0.4216

    # Check solutions
    assert pytest.approx(2885989.2 * 2, rel=1e-5) == value(
        m.fs.unit.costing.capital_cost
    )
    assert pytest.approx(2885989.2, rel=1e-5) == value(
        pyunits.convert(
            (102794 * 1964.42**0.4216) * pyunits.USD_2007,
            to_units=m.fs.costing.base_currency,
        )
    )
    assert pytest.approx(7.4361 * 0.0039, rel=1e-5) == value(
        m.fs.unit.electricity_consumption[0]
    )


@pytest.mark.solver
@pytest.mark.skipif(solver is None, reason="Solver not available")
@pytest.mark.component
def test_du_filter_belt_press_costing():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)

    m.fs.props = ASM1ParameterBlock()

    m.fs.unit = DewateringUnit(property_package=m.fs.props)

    m.fs.unit.inlet.flow_vol.fix(178.4674 * units.m**3 / units.day)
    m.fs.unit.inlet.temperature.fix(308.15 * units.K)
    m.fs.unit.inlet.pressure.fix(1 * units.atm)

    m.fs.unit.inlet.conc_mass_comp[0, "S_I"].fix(130.867 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "S_S"].fix(258.5789 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "X_I"].fix(17216.2434 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "X_S"].fix(2611.4843 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "X_BH"].fix(1e-6 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "X_BA"].fix(1e-6 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "X_P"].fix(626.0652 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "S_O"].fix(1e-6 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "S_NO"].fix(1e-6 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "S_NH"].fix(1442.7882 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "S_ND"].fix(0.54323 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "X_ND"].fix(100.8668 * units.mg / units.liter)
    m.fs.unit.inlet.alkalinity.fix(97.8459 * units.mol / units.m**3)

    m.fs.unit.hydraulic_retention_time.fix()

    m.fs.costing = WaterTAPCosting()

    m.fs.unit.costing = UnitModelCostingBlock(
        flowsheet_costing_block=m.fs.costing,
        costing_method=cost_dewatering,
        costing_method_arguments={
            "dewatering_type": DewateringType.filter_belt_press,
            "cost_electricity_flow": True,
        },
    )
    # Using average specific energy consumption of 0.006 for screw press as a function of capacity
    m.fs.unit.energy_electric_flow_vol_inlet[0] = 0.006 * pyunits.kWh / pyunits.m**3
    m.fs.costing.cost_process()

    assert degrees_of_freedom(m) == 0

    results = solver.solve(m)

    assert_optimal_termination(results)
    assert_units_consistent(m)
    assert hasattr(m.fs.costing, "filter_belt_press")
    assert value(m.fs.costing.filter_belt_press.capital_a_parameter) == 146.29
    assert value(m.fs.costing.filter_belt_press.capital_b_parameter) == 433972

    # Check solutions
    assert pytest.approx(828025.2 * 2, rel=1e-5) == value(
        m.fs.unit.costing.capital_cost
    )
    assert pytest.approx(828025.2, rel=1e-5) == value(
        pyunits.convert(
            (146.29 * 1964.42 + 433972) * pyunits.USD_2007,
            to_units=m.fs.costing.base_currency,
        )
    )
    assert pytest.approx(7.4361 * 0.006, rel=1e-5) == value(
        m.fs.unit.electricity_consumption[0]
    )


@pytest.mark.solver
@pytest.mark.skipif(solver is None, reason="Solver not available")
@pytest.mark.component
def test_du_costing_config_err():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)

    m.fs.props = ASM1ParameterBlock()

    m.fs.unit = DewateringUnit(property_package=m.fs.props)

    m.fs.unit.inlet.flow_vol.fix(178.4674 * units.m**3 / units.day)
    m.fs.unit.inlet.temperature.fix(308.15 * units.K)
    m.fs.unit.inlet.pressure.fix(1 * units.atm)

    m.fs.unit.inlet.conc_mass_comp[0, "S_I"].fix(130.867 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "S_S"].fix(258.5789 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "X_I"].fix(17216.2434 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "X_S"].fix(2611.4843 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "X_BH"].fix(1e-6 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "X_BA"].fix(1e-6 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "X_P"].fix(626.0652 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "S_O"].fix(1e-6 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "S_NO"].fix(1e-6 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "S_NH"].fix(1442.7882 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "S_ND"].fix(0.54323 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "X_ND"].fix(100.8668 * units.mg / units.liter)
    m.fs.unit.inlet.alkalinity.fix(97.8459 * units.mol / units.m**3)

    m.fs.unit.hydraulic_retention_time.fix()

    m.fs.costing = WaterTAPCosting()

    with pytest.raises(
        ConfigurationError,
        match="fs.unit received invalid argument for dewatering_type: foo. Argument must be a member of the DewateringType Enum class.",
    ):
        m.fs.unit.costing = UnitModelCostingBlock(
            flowsheet_costing_block=m.fs.costing,
            costing_method=cost_dewatering,
            costing_method_arguments={
                "dewatering_type": "foo",
            },
        )


class TestThickenerScaler:
    @pytest.fixture
    def model(self):
        m = ConcreteModel()
        m.fs = FlowsheetBlock(dynamic=False)

        m.fs.props = ASM1ParameterBlock()

        m.fs.unit = DewateringUnit(property_package=m.fs.props)

        m.fs.unit.inlet.flow_vol.fix(178.4674 * units.m**3 / units.day)
        m.fs.unit.inlet.temperature.fix(308.15 * units.K)
        m.fs.unit.inlet.pressure.fix(1 * units.atm)

        m.fs.unit.inlet.conc_mass_comp[0, "S_I"].fix(130.867 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_S"].fix(258.5789 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_I"].fix(
            17216.2434 * units.mg / units.liter
        )
        m.fs.unit.inlet.conc_mass_comp[0, "X_S"].fix(2611.4843 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_BH"].fix(1e-6 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_BA"].fix(1e-6 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_P"].fix(626.0652 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_O"].fix(1e-6 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_NO"].fix(1e-6 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_NH"].fix(
            1442.7882 * units.mg / units.liter
        )
        m.fs.unit.inlet.conc_mass_comp[0, "S_ND"].fix(0.54323 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_ND"].fix(100.8668 * units.mg / units.liter)
        m.fs.unit.inlet.alkalinity.fix(97.8459 * units.mol / units.m**3)

        m.fs.unit.hydraulic_retention_time.fix()

        return m

    @pytest.mark.component
    def test_variable_scaling_routine(self, model):
        scaler = model.fs.unit.default_scaler()

        assert isinstance(scaler, DewatererScaler)

        scaler.variable_scaling_routine(model.fs.unit)

        # Inlet state
        sfx_in = model.fs.unit.mixed_state[0].scaling_factor
        assert isinstance(sfx_in, Suffix)
        assert len(sfx_in) == 3
        assert sfx_in[model.fs.unit.mixed_state[0].flow_vol] == pytest.approx(
            1e1, rel=1e-8
        )
        assert sfx_in[model.fs.unit.mixed_state[0].pressure] == pytest.approx(
            1e-6, rel=1e-8
        )
        assert sfx_in[model.fs.unit.mixed_state[0].temperature] == pytest.approx(
            1e-1, rel=1e-8
        )

        # Outlet state - should be the same as the inlet
        sfx_underflow = model.fs.unit.underflow_state[0].scaling_factor
        assert isinstance(sfx_underflow, Suffix)
        assert len(sfx_underflow) == 3
        assert sfx_underflow[
            model.fs.unit.underflow_state[0].flow_vol
        ] == pytest.approx(1e1, rel=1e-8)
        assert sfx_underflow[
            model.fs.unit.underflow_state[0].pressure
        ] == pytest.approx(1e-6, rel=1e-8)
        assert sfx_underflow[
            model.fs.unit.underflow_state[0].temperature
        ] == pytest.approx(1e-1, rel=1e-8)

        sfx_overflow = model.fs.unit.overflow_state[0].scaling_factor
        assert isinstance(sfx_overflow, Suffix)
        assert len(sfx_overflow) == 3
        assert sfx_overflow[model.fs.unit.overflow_state[0].flow_vol] == pytest.approx(
            1e1, rel=1e-8
        )
        assert sfx_overflow[model.fs.unit.overflow_state[0].pressure] == pytest.approx(
            1e-6, rel=1e-8
        )
        assert sfx_overflow[
            model.fs.unit.overflow_state[0].temperature
        ] == pytest.approx(1e-1, rel=1e-8)

        # Check that unit model has scaling factors
        sfx_unit = model.fs.unit.scaling_factor
        assert isinstance(sfx_unit, Suffix)
        assert len(sfx_unit) == 1
        assert sfx_unit[model.fs.unit.volume[0]] == pytest.approx(1e-3, rel=1e-3)

    @pytest.mark.component
    def test_constraint_scaling_routine(self, model):
        scaler = model.fs.unit.default_scaler()

        assert isinstance(scaler, DewatererScaler)

        scaler.constraint_scaling_routine(model.fs.unit)

        sfx_unit = model.fs.unit.scaling_factor
        assert isinstance(sfx_unit, Suffix)
        assert len(sfx_unit) == 62
        assert sfx_unit[model.fs.unit.eq_electricity_consumption[0]] == pytest.approx(
            1, rel=1e-8
        )
        assert sfx_unit[model.fs.unit.eq_hydraulic_retention[0]] == pytest.approx(
            1.14755273e-6, rel=1e-8
        )
        assert sfx_unit[
            model.fs.unit.overflow_particulate_fraction[0, "X_I"]
        ] == pytest.approx(1, rel=1e-8)
        assert sfx_unit[
            model.fs.unit.overflow_particulate_fraction[0, "X_S"]
        ] == pytest.approx(1, rel=1e-8)
        assert sfx_unit[
            model.fs.unit.overflow_particulate_fraction[0, "X_P"]
        ] == pytest.approx(1, rel=1e-8)
        assert sfx_unit[
            model.fs.unit.overflow_particulate_fraction[0, "X_BH"]
        ] == pytest.approx(1, rel=1e-8)
        assert sfx_unit[
            model.fs.unit.overflow_particulate_fraction[0, "X_BA"]
        ] == pytest.approx(1, rel=1e-8)
        assert sfx_unit[
            model.fs.unit.overflow_particulate_fraction[0, "X_ND"]
        ] == pytest.approx(1, rel=1e-8)
        assert sfx_unit[
            model.fs.unit.non_particulate_components[0, "H2O"]
        ] == pytest.approx(1, rel=1e-8)
        assert sfx_unit[
            model.fs.unit.non_particulate_components[0, "S_I"]
        ] == pytest.approx(1, rel=1e-8)
        assert sfx_unit[
            model.fs.unit.non_particulate_components[0, "S_S"]
        ] == pytest.approx(1, rel=1e-8)
        assert sfx_unit[
            model.fs.unit.non_particulate_components[0, "S_O"]
        ] == pytest.approx(1, rel=1e-8)
        assert sfx_unit[
            model.fs.unit.non_particulate_components[0, "S_NO"]
        ] == pytest.approx(1, rel=1e-8)
        assert sfx_unit[
            model.fs.unit.non_particulate_components[0, "S_NH"]
        ] == pytest.approx(1, rel=1e-8)
        assert sfx_unit[
            model.fs.unit.non_particulate_components[0, "S_ND"]
        ] == pytest.approx(1, rel=1e-8)
        assert sfx_unit[
            model.fs.unit.non_particulate_components[0, "S_ALK"]
        ] == pytest.approx(1, rel=1e-8)

    @pytest.mark.component
    def test_scale_model(self, model):
        scaler = model.fs.unit.default_scaler()

        assert isinstance(scaler, DewatererScaler)

        scaler.scale_model(model.fs.unit)

        # Inlet state
        sfx_in = model.fs.unit.mixed_state[0].scaling_factor
        assert isinstance(sfx_in, Suffix)
        assert len(sfx_in) == 3
        assert sfx_in[model.fs.unit.mixed_state[0].flow_vol] == pytest.approx(
            1e1, rel=1e-8
        )
        assert sfx_in[model.fs.unit.mixed_state[0].pressure] == pytest.approx(
            1e-6, rel=1e-8
        )
        assert sfx_in[model.fs.unit.mixed_state[0].temperature] == pytest.approx(
            1e-1, rel=1e-8
        )

        # Outlet state - should be the same as the inlet
        sfx_underflow = model.fs.unit.underflow_state[0].scaling_factor
        assert isinstance(sfx_underflow, Suffix)
        assert len(sfx_underflow) == 3
        assert sfx_underflow[
            model.fs.unit.underflow_state[0].flow_vol
        ] == pytest.approx(1e1, rel=1e-8)
        assert sfx_underflow[
            model.fs.unit.underflow_state[0].pressure
        ] == pytest.approx(1e-6, rel=1e-8)
        assert sfx_underflow[
            model.fs.unit.underflow_state[0].temperature
        ] == pytest.approx(1e-1, rel=1e-8)

        sfx_overflow = model.fs.unit.underflow_state[0].scaling_factor
        assert isinstance(sfx_overflow, Suffix)
        assert len(sfx_overflow) == 3
        assert sfx_overflow[model.fs.unit.underflow_state[0].flow_vol] == pytest.approx(
            1e1, rel=1e-8
        )
        assert sfx_overflow[model.fs.unit.underflow_state[0].pressure] == pytest.approx(
            1e-6, rel=1e-8
        )
        assert sfx_overflow[
            model.fs.unit.underflow_state[0].temperature
        ] == pytest.approx(1e-1, rel=1e-8)

        # Check that unit model has scaling factors
        sfx_unit = model.fs.unit.scaling_factor
        assert isinstance(sfx_unit, Suffix)
        assert len(sfx_unit) == 63
        assert sfx_unit[model.fs.unit.volume[0]] == pytest.approx(1e-3, rel=1e-3)
        assert sfx_unit[model.fs.unit.eq_electricity_consumption[0]] == pytest.approx(
            1, rel=1e-8
        )
        assert sfx_unit[model.fs.unit.eq_hydraulic_retention[0]] == pytest.approx(
            2.06559491e-6, rel=1e-8
        )
        assert sfx_unit[
            model.fs.unit.overflow_particulate_fraction[0, "X_I"]
        ] == pytest.approx(1, rel=1e-8)
        assert sfx_unit[
            model.fs.unit.overflow_particulate_fraction[0, "X_S"]
        ] == pytest.approx(1, rel=1e-8)
        assert sfx_unit[
            model.fs.unit.overflow_particulate_fraction[0, "X_P"]
        ] == pytest.approx(1, rel=1e-8)
        assert sfx_unit[
            model.fs.unit.overflow_particulate_fraction[0, "X_BH"]
        ] == pytest.approx(1, rel=1e-8)
        assert sfx_unit[
            model.fs.unit.overflow_particulate_fraction[0, "X_BA"]
        ] == pytest.approx(1, rel=1e-8)
        assert sfx_unit[
            model.fs.unit.overflow_particulate_fraction[0, "X_ND"]
        ] == pytest.approx(1, rel=1e-8)
        assert sfx_unit[
            model.fs.unit.non_particulate_components[0, "H2O"]
        ] == pytest.approx(1, rel=1e-8)
        assert sfx_unit[
            model.fs.unit.non_particulate_components[0, "S_I"]
        ] == pytest.approx(1, rel=1e-8)
        assert sfx_unit[
            model.fs.unit.non_particulate_components[0, "S_S"]
        ] == pytest.approx(1, rel=1e-8)
        assert sfx_unit[
            model.fs.unit.non_particulate_components[0, "S_O"]
        ] == pytest.approx(1, rel=1e-8)
        assert sfx_unit[
            model.fs.unit.non_particulate_components[0, "S_NO"]
        ] == pytest.approx(1, rel=1e-8)
        assert sfx_unit[
            model.fs.unit.non_particulate_components[0, "S_NH"]
        ] == pytest.approx(1, rel=1e-8)
        assert sfx_unit[
            model.fs.unit.non_particulate_components[0, "S_ND"]
        ] == pytest.approx(1, rel=1e-8)
        assert sfx_unit[
            model.fs.unit.non_particulate_components[0, "S_ALK"]
        ] == pytest.approx(1, rel=1e-8)

    # TODO: Remove test once iscale is deprecated
    @pytest.mark.integration
    def test_example_case_iscale(self):
        m = ConcreteModel()
        m.fs = FlowsheetBlock(dynamic=False)

        m.fs.props = ASM1ParameterBlock()

        m.fs.unit = DewateringUnit(property_package=m.fs.props)

        m.fs.unit.inlet.flow_vol.fix(178.4674 * units.m**3 / units.day)
        m.fs.unit.inlet.temperature.fix(308.15 * units.K)
        m.fs.unit.inlet.pressure.fix(1 * units.atm)

        m.fs.unit.inlet.conc_mass_comp[0, "S_I"].fix(130.867 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_S"].fix(258.5789 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_I"].fix(
            17216.2434 * units.mg / units.liter
        )
        m.fs.unit.inlet.conc_mass_comp[0, "X_S"].fix(2611.4843 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_BH"].fix(1e-6 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_BA"].fix(1e-6 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_P"].fix(626.0652 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_O"].fix(1e-6 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_NO"].fix(1e-6 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_NH"].fix(
            1442.7882 * units.mg / units.liter
        )
        m.fs.unit.inlet.conc_mass_comp[0, "S_ND"].fix(0.54323 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_ND"].fix(100.8668 * units.mg / units.liter)
        m.fs.unit.inlet.alkalinity.fix(97.8459 * units.mol / units.m**3)

        m.fs.unit.hydraulic_retention_time.fix()

        iscale.calculate_scaling_factors(m.fs.unit)

        # Check condition number to confirm scaling
        sm = TransformationFactory("core.scale_model").create_using(m, rename=False)
        jac, _ = get_jacobian(sm, scaled=False)
        assert (jacobian_cond(jac=jac, scaled=False)) == pytest.approx(
            3.3758830009e5, rel=1e-3
        )

    @pytest.mark.integration
    def test_example_case_scaler(self):
        m = ConcreteModel()
        m.fs = FlowsheetBlock(dynamic=False)

        m.fs.props = ASM1ParameterBlock()

        m.fs.unit = DewateringUnit(property_package=m.fs.props)

        m.fs.unit.inlet.flow_vol.fix(178.4674 * units.m**3 / units.day)
        m.fs.unit.inlet.temperature.fix(308.15 * units.K)
        m.fs.unit.inlet.pressure.fix(1 * units.atm)

        m.fs.unit.inlet.conc_mass_comp[0, "S_I"].fix(130.867 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_S"].fix(258.5789 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_I"].fix(
            17216.2434 * units.mg / units.liter
        )
        m.fs.unit.inlet.conc_mass_comp[0, "X_S"].fix(2611.4843 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_BH"].fix(1e-6 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_BA"].fix(1e-6 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_P"].fix(626.0652 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_O"].fix(1e-6 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_NO"].fix(1e-6 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_NH"].fix(
            1442.7882 * units.mg / units.liter
        )
        m.fs.unit.inlet.conc_mass_comp[0, "S_ND"].fix(0.54323 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_ND"].fix(100.8668 * units.mg / units.liter)
        m.fs.unit.inlet.alkalinity.fix(97.8459 * units.mol / units.m**3)

        m.fs.unit.hydraulic_retention_time.fix()

        sb = ScalerBase()

        scaler = DewatererScaler()
        scaler.scale_model(
            m.fs.unit,
            submodel_scalers={
                m.fs.unit.mixed_state: ASM1PropertiesScaler,
                m.fs.unit.underflow_state: ASM1PropertiesScaler,
                m.fs.unit.overflow_state: ASM1PropertiesScaler,
            },
        )

        # Check condition number to confirm scaling
        sm = TransformationFactory("core.scale_model").create_using(m, rename=False)
        jac, _ = get_jacobian(sm, scaled=False)
        assert (jacobian_cond(jac=jac, scaled=False)) == pytest.approx(
            2.100583344e4, rel=1e-3
        )


def build_model():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)

    m.fs.props = ASM1ParameterBlock()

    m.fs.unit = DewateringUnit(property_package=m.fs.props)

    m.fs.unit.inlet.flow_vol.fix(178.4674 * units.m**3 / units.day)
    m.fs.unit.inlet.temperature.fix(308.15 * units.K)
    m.fs.unit.inlet.pressure.fix(1 * units.atm)

    m.fs.unit.inlet.conc_mass_comp[0, "S_I"].fix(130.867 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "S_S"].fix(258.5789 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "X_I"].fix(17216.2434 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "X_S"].fix(2611.4843 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "X_BH"].fix(1e-6 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "X_BA"].fix(1e-6 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "X_P"].fix(626.0652 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "S_O"].fix(1e-6 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "S_NO"].fix(1e-6 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "S_NH"].fix(1442.7882 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "S_ND"].fix(0.54323 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "X_ND"].fix(100.8668 * units.mg / units.liter)
    m.fs.unit.inlet.alkalinity.fix(97.8459 * units.mol / units.m**3)

    m.fs.unit.hydraulic_retention_time.fix()

    solver = get_solver()
    solver.solve(m)

    return m


def scale_vars_with_scalers(m):
    scaler = DewatererScaler()
    scaler.scale_model(
        m.fs.unit,
        submodel_scalers={
            m.fs.unit.mixed_state: ASM1PropertiesScaler,
            m.fs.unit.underflow_state: ASM1PropertiesScaler,
            m.fs.unit.overflow_state: ASM1PropertiesScaler,
        },
    )


def scale_vars_with_iscale(m):
    # Set scaling factors for badly scaled variables
    iscale.calculate_scaling_factors(m.fs.unit)


def perturb_solution(m):
    m.fs.unit.inlet.flow_vol.fix(178.4674 * 0.8 * units.m**3 / units.day)
    m.fs.unit.inlet.conc_mass_comp[0, "S_I"].fix(
        130.867 * 0.55 * units.mg / units.liter
    )


@pytest.mark.requires_idaes_solver
@pytest.mark.unit
def test_scaling_profiler_with_scalers():
    sp = ScalingProfiler(
        build_model=build_model,
        user_scaling=scale_vars_with_scalers,
        perturb_state=perturb_solution,
    )

    stream = StringIO()

    sp.report_scaling_profiles(stream=stream)

    expected = """
============================================================================
Scaling Profile Report
----------------------------------------------------------------------------
Scaling Method           || User Scaling           || Perfect Scaling
Unscaled                 || 4.958E+07 | Solved 1   ||
Vars Only                || 1.379E+13 | Solved 3   || 1.270E+18 | Solved 5  
Harmonic                 || 1.379E+13 | Solved 3   || 1.314E+05 | Solved 1  
Inverse Sum              || 1.379E+13 | Solved 3   || 4.501E+03 | Solved 1  
Inverse Root Sum Squares || 1.379E+13 | Solved 3   || 6.085E+03 | Solved 1  
Inverse Maximum          || 1.379E+13 | Solved 3   || 8.354E+03 | Solved 1  
Inverse Minimum          || 1.379E+13 | Solved 3   || 1.236E+05 | Solved 1  
Nominal L1 Norm          || 1.379E+13 | Solved 3   || 4.933E+03 | Solved 1  
Nominal L2 Norm          || 1.379E+13 | Solved 3   || 5.883E+03 | Solved 1  
Actual L1 Norm           || 1.379E+13 | Solved 3   || 5.468E+13 | Solved 1  
Actual L2 Norm           || 1.379E+13 | Solved 3   || 6.356E+13 | Solved 1  
============================================================================
"""

    assert stream.getvalue() == expected


@pytest.mark.requires_idaes_solver
@pytest.mark.unit
def test_scaling_profiler_with_iscale():
    sp = ScalingProfiler(
        build_model=build_model,
        user_scaling=scale_vars_with_iscale,
        perturb_state=perturb_solution,
    )

    stream = StringIO()

    sp.report_scaling_profiles(stream=stream)

    expected = """
============================================================================
Scaling Profile Report
----------------------------------------------------------------------------
Scaling Method           || User Scaling           || Perfect Scaling
Unscaled                 || 4.958E+07 | Solved 1   ||
Vars Only                || 9.503E+10 | Solved 3   || 1.270E+18 | Solved 5  
Harmonic                 || 5.041E+15 | Solved 3   || 1.314E+05 | Solved 1  
Inverse Sum              || 9.487E+14 | Solved 3   || 4.501E+03 | Solved 1  
Inverse Root Sum Squares || 7.199E+14 | Solved 3   || 6.085E+03 | Solved 1  
Inverse Maximum          || 5.224E+14 | Solved 3   || 8.354E+03 | Solved 1  
Inverse Minimum          || 9.593E+15 | Solved 3   || 1.236E+05 | Solved 1  
Nominal L1 Norm          || 1.493E+09 | Solved 1   || 4.933E+03 | Solved 1  
Nominal L2 Norm          || 1.558E+09 | Solved 1   || 5.883E+03 | Solved 1  
Actual L1 Norm           || 2.955E+07 | Solved 2   || 5.468E+13 | Solved 1  
Actual L2 Norm           || 3.193E+07 | Solved 2   || 6.356E+13 | Solved 1  
============================================================================
"""

    assert stream.getvalue() == expected
