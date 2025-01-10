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
Tests for thickener unit example.
"""
from io import StringIO
import pytest
from pyomo.environ import (
    ConcreteModel,
    units,
    Suffix,
    TransformationFactory,
    Var,
)

from idaes.core import FlowsheetBlock

from idaes.models.unit_models.separator import SplittingType

from watertap.core.solvers import get_solver

import idaes.core.util.scaling as iscale
from idaes.core.util.scaling import (
    get_jacobian,
    jacobian_cond,
)
from idaes.core.scaling.scaler_profiling import ScalingProfiler
from idaes.core.scaling.scaling_base import ScalerBase

from idaes.core.util.exceptions import (
    ConfigurationError,
)
from watertap.unit_models.tests.unit_test_harness import UnitTestHarness
from watertap.unit_models.thickener import (
    Thickener,
    ActivatedSludgeModelType,
    ThickenerScaler,
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
from watertap.costing import WaterTAPCosting
from idaes.core import UnitModelCostingBlock

__author__ = "Alejandro Garciadiego, Adam Atia"

# -----------------------------------------------------------------------------
# Get default solver for testing
solver = get_solver()


# -----------------------------------------------------------------------------
def build_ASM1():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)

    m.fs.props = ASM1ParameterBlock()

    m.fs.unit = Thickener(
        property_package=m.fs.props,
        outlet_list=["underflow", "overflow"],
        split_basis=SplittingType.componentFlow,
    )

    m.fs.unit.inlet.flow_vol.fix(300 * units.m**3 / units.day)

    m.fs.unit.inlet.temperature.fix(308.15 * units.K)
    m.fs.unit.inlet.pressure.fix(1 * units.atm)

    m.fs.unit.inlet.conc_mass_comp[0, "S_I"].fix(28.0643 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "S_S"].fix(0.67336 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "X_I"].fix(3036.2175 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "X_S"].fix(63.2392 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "X_BH"].fix(4442.8377 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "X_BA"].fix(332.5958 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "X_P"].fix(1922.8108 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "S_O"].fix(1.3748 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "S_NO"].fix(9.1948 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "S_NH"].fix(0.15845 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "S_ND"].fix(0.55943 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "X_ND"].fix(4.7411 * units.mg / units.liter)
    m.fs.unit.inlet.alkalinity.fix(4.5646 * units.mol / units.m**3)

    m.fs.unit.hydraulic_retention_time.fix()
    m.fs.unit.diameter.fix()

    # Set scaling factors for badly scaled variables
    iscale.set_scaling_factor(m.fs.unit.underflow_state[0.0].flow_vol, 1e4)
    iscale.set_scaling_factor(m.fs.unit.underflow_state[0.0].pressure, 1e-6)
    iscale.set_scaling_factor(m.fs.unit.underflow_state[0.0].conc_mass_comp["S_S"], 1e4)
    iscale.set_scaling_factor(
        m.fs.unit.underflow_state[0.0].conc_mass_comp["S_NH"], 1e4
    )
    iscale.set_scaling_factor(
        m.fs.unit.underflow_state[0.0].conc_mass_comp["S_ND"], 1e4
    )
    iscale.set_scaling_factor(m.fs.unit.overflow_state[0.0].pressure, 1e-6)
    iscale.set_scaling_factor(m.fs.unit.overflow_state[0.0].conc_mass_comp["S_S"], 1e4)
    iscale.set_scaling_factor(m.fs.unit.overflow_state[0.0].conc_mass_comp["S_NH"], 1e4)
    iscale.set_scaling_factor(m.fs.unit.overflow_state[0.0].conc_mass_comp["S_ND"], 1e4)
    iscale.set_scaling_factor(m.fs.unit.overflow_state[0.0].conc_mass_comp["X_ND"], 1e4)

    return m


def build_ASM2d():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)

    m.fs.props = ASM2dParameterBlock()

    m.fs.unit = Thickener(
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
    m.fs.unit.inlet.conc_mass_comp[0, "X_PAO"].fix(214.5319 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "X_PP"].fix(63.5316 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "X_PHA"].fix(2.7381 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "X_AUT"].fix(118.3582 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "X_MeOH"].fix(EPS * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "X_MeP"].fix(EPS * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "X_TSS"].fix(3525.429 * units.mg / units.liter)
    m.fs.unit.inlet.alkalinity[0].fix(4.6663 * units.mmol / units.liter)

    m.fs.unit.hydraulic_retention_time.fix()
    m.fs.unit.diameter.fix()

    # Set scaling factors for badly scaled variables
    iscale.set_scaling_factor(m.fs.unit.underflow_state[0.0].flow_vol, 1e5)
    iscale.set_scaling_factor(m.fs.unit.underflow_state[0.0].pressure, 1e-6)
    iscale.set_scaling_factor(m.fs.unit.underflow_state[0.0].conc_mass_comp["S_A"], 1e5)
    iscale.set_scaling_factor(m.fs.unit.underflow_state[0.0].conc_mass_comp["S_F"], 1e4)
    iscale.set_scaling_factor(
        m.fs.unit.underflow_state[0.0].conc_mass_comp["X_MeOH"], 1e7
    )
    iscale.set_scaling_factor(
        m.fs.unit.underflow_state[0.0].conc_mass_comp["X_MeP"], 1e7
    )
    iscale.set_scaling_factor(m.fs.unit.overflow_state[0.0].pressure, 1e-6)
    iscale.set_scaling_factor(m.fs.unit.overflow_state[0.0].conc_mass_comp["S_A"], 1e5)
    iscale.set_scaling_factor(m.fs.unit.overflow_state[0.0].conc_mass_comp["S_F"], 1e4)
    iscale.set_scaling_factor(
        m.fs.unit.overflow_state[0.0].conc_mass_comp["X_MeOH"], 1e10
    )
    iscale.set_scaling_factor(
        m.fs.unit.overflow_state[0.0].conc_mass_comp["X_MeP"], 1e10
    )
    iscale.set_scaling_factor(
        m.fs.unit.overflow_state[0.0].conc_mass_comp["X_PHA"], 1e5
    )

    iscale.calculate_scaling_factors(m.fs.unit)

    return m


def build_ASM2d_modified():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)

    m.fs.props = ModifiedASM2dParameterBlock()

    m.fs.unit = Thickener(
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
    m.fs.unit.inlet.conc_mass_comp[0, "X_PAO"].fix(214.5319 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "X_PP"].fix(63.5316 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "X_PHA"].fix(2.7381 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "X_AUT"].fix(118.3582 * units.mg / units.liter)

    iscale.calculate_scaling_factors(m.fs.unit)

    m.fs.unit.hydraulic_retention_time.fix()
    m.fs.unit.diameter.fix()

    # Set scaling factors for badly scaled variables
    iscale.set_scaling_factor(m.fs.unit.underflow_state[0.0].flow_vol, 1e5)
    iscale.set_scaling_factor(m.fs.unit.underflow_state[0.0].pressure, 1e-6)
    iscale.set_scaling_factor(m.fs.unit.underflow_state[0.0].conc_mass_comp["S_A"], 1e5)
    iscale.set_scaling_factor(m.fs.unit.underflow_state[0.0].conc_mass_comp["S_F"], 1e4)
    iscale.set_scaling_factor(m.fs.unit.overflow_state[0.0].pressure, 1e-6)
    iscale.set_scaling_factor(m.fs.unit.overflow_state[0.0].conc_mass_comp["S_A"], 1e5)
    iscale.set_scaling_factor(m.fs.unit.overflow_state[0.0].conc_mass_comp["S_F"], 1e4)
    iscale.set_scaling_factor(
        m.fs.unit.overflow_state[0.0].conc_mass_comp["X_PHA"], 1e5
    )

    return m


class TestThickener_ASM1(UnitTestHarness):
    def configure(self):
        m = build_ASM1()

        self.unit_solutions[m.fs.unit.overflow.pressure[0]] = 101325.0
        self.unit_solutions[m.fs.unit.overflow.temperature[0]] = 308.15
        self.unit_solutions[m.fs.unit.overflow.flow_vol[0]] = 0.003115
        self.unit_solutions[m.fs.unit.overflow.conc_mass_comp[0, "S_I"]] = 0.02806
        self.unit_solutions[m.fs.unit.overflow.conc_mass_comp[0, "S_S"]] = 0.000673
        self.unit_solutions[m.fs.unit.overflow.conc_mass_comp[0, "X_I"]] = 0.06768
        self.unit_solutions[m.fs.unit.overflow.conc_mass_comp[0, "X_S"]] = 0.001409
        self.unit_solutions[m.fs.unit.overflow.conc_mass_comp[0, "X_P"]] = 0.04286
        self.unit_solutions[m.fs.unit.overflow.conc_mass_comp[0, "X_BH"]] = 0.099046
        self.unit_solutions[m.fs.unit.overflow.conc_mass_comp[0, "X_BA"]] = 0.007414
        self.unit_solutions[m.fs.unit.overflow.conc_mass_comp[0, "S_O"]] = 0.001374
        self.unit_solutions[m.fs.unit.overflow.conc_mass_comp[0, "S_NO"]] = 0.009194
        self.unit_solutions[m.fs.unit.overflow.conc_mass_comp[0, "S_NH"]] = 0.0001584
        self.unit_solutions[m.fs.unit.overflow.conc_mass_comp[0, "S_ND"]] = 0.0005594
        self.unit_solutions[m.fs.unit.overflow.conc_mass_comp[0, "X_ND"]] = 0.0001056
        self.unit_solutions[m.fs.unit.overflow.alkalinity[0]] = 0.004564

        self.conservation_equality = {
            "Check 1": {
                "in": m.fs.unit.inlet.flow_vol[0],
                "out": (
                    m.fs.unit.overflow.flow_vol[0] * m.fs.props.dens_mass
                    + m.fs.unit.underflow.flow_vol[0] * m.fs.props.dens_mass
                )
                / m.fs.props.dens_mass,
            },
            "Check 2": {
                "in": m.fs.unit.inlet.flow_vol[0]
                * sum(
                    m.fs.unit.inlet.conc_mass_comp[0, j] for j in m.fs.props.solute_set
                ),
                "out": sum(
                    m.fs.unit.overflow.flow_vol[0]
                    * m.fs.unit.overflow.conc_mass_comp[0, j]
                    + m.fs.unit.underflow.flow_vol[0]
                    * m.fs.unit.underflow.conc_mass_comp[0, j]
                    for j in m.fs.props.solute_set
                ),
            },
        }

        return m


class TestThickener_ASM2d(UnitTestHarness):
    def configure(self):
        m = build_ASM2d()

        self.unit_solutions[m.fs.unit.overflow.conc_mass_comp[0, "S_A"]] = 3.360667e-05
        self.unit_solutions[m.fs.unit.overflow.conc_mass_comp[0, "S_F"]] = 0.0004748
        self.unit_solutions[m.fs.unit.overflow.conc_mass_comp[0, "S_I"]] = 0.03
        self.unit_solutions[m.fs.unit.overflow.conc_mass_comp[0, "S_N2"]] = 0.029060
        self.unit_solutions[m.fs.unit.overflow.conc_mass_comp[0, "S_NH4"]] = 0.008020
        self.unit_solutions[m.fs.unit.overflow.conc_mass_comp[0, "S_NO3"]] = 0.006639
        self.unit_solutions[m.fs.unit.overflow.conc_mass_comp[0, "S_O2"]] = 0.007970
        self.unit_solutions[m.fs.unit.overflow.conc_mass_comp[0, "S_PO4"]] = 0.007895
        self.unit_solutions[m.fs.unit.overflow.conc_mass_comp[0, "X_AUT"]] = 0.002490
        self.unit_solutions[m.fs.unit.overflow.conc_mass_comp[0, "X_H"]] = 0.039036
        self.unit_solutions[m.fs.unit.overflow.conc_mass_comp[0, "X_I"]] = 0.035676
        self.unit_solutions[m.fs.unit.overflow.conc_mass_comp[0, "X_MeOH"]] = 0.0
        self.unit_solutions[m.fs.unit.overflow.conc_mass_comp[0, "X_MeP"]] = 0.0
        self.unit_solutions[m.fs.unit.overflow.conc_mass_comp[0, "X_PAO"]] = 0.004513
        self.unit_solutions[m.fs.unit.overflow.conc_mass_comp[0, "X_PHA"]] = (
            5.761182e-05
        )
        self.unit_solutions[m.fs.unit.overflow.conc_mass_comp[0, "X_PP"]] = 0.001336
        self.unit_solutions[m.fs.unit.overflow.conc_mass_comp[0, "X_S"]] = 0.001436
        self.unit_solutions[m.fs.unit.overflow.conc_mass_comp[0, "X_TSS"]] = 0.074169
        self.unit_solutions[m.fs.unit.overflow.alkalinity[0]] = 0.004666

        self.conservation_equality = {
            "Check 1": {
                "in": m.fs.unit.inlet.flow_vol[0],
                "out": (
                    m.fs.unit.overflow.flow_vol[0] * m.fs.props.dens_mass
                    + m.fs.unit.underflow.flow_vol[0] * m.fs.props.dens_mass
                )
                / m.fs.props.dens_mass,
            },
            "Check 2": {
                "in": m.fs.unit.inlet.flow_vol[0]
                * sum(
                    m.fs.unit.inlet.conc_mass_comp[0, j] for j in m.fs.props.solute_set
                ),
                "out": sum(
                    m.fs.unit.overflow.flow_vol[0]
                    * m.fs.unit.overflow.conc_mass_comp[0, j]
                    + m.fs.unit.underflow.flow_vol[0]
                    * m.fs.unit.underflow.conc_mass_comp[0, j]
                    for j in m.fs.props.solute_set
                ),
            },
        }

        return m


class TestThickener_ASM2d_modified(UnitTestHarness):
    def configure(self):
        m = build_ASM2d_modified()

        self.unit_solutions[m.fs.unit.overflow.conc_mass_comp[0, "S_A"]] = 3.360667e-05
        self.unit_solutions[m.fs.unit.overflow.conc_mass_comp[0, "S_F"]] = 0.0004748
        self.unit_solutions[m.fs.unit.overflow.conc_mass_comp[0, "S_I"]] = 0.03
        self.unit_solutions[m.fs.unit.overflow.conc_mass_comp[0, "S_N2"]] = 0.029060
        self.unit_solutions[m.fs.unit.overflow.conc_mass_comp[0, "S_NH4"]] = 0.008020
        self.unit_solutions[m.fs.unit.overflow.conc_mass_comp[0, "S_NO3"]] = 0.006639
        self.unit_solutions[m.fs.unit.overflow.conc_mass_comp[0, "S_O2"]] = 0.007970
        self.unit_solutions[m.fs.unit.overflow.conc_mass_comp[0, "S_PO4"]] = 0.007895
        self.unit_solutions[m.fs.unit.overflow.conc_mass_comp[0, "X_AUT"]] = 0.002490
        self.unit_solutions[m.fs.unit.overflow.conc_mass_comp[0, "X_H"]] = 0.038882
        self.unit_solutions[m.fs.unit.overflow.conc_mass_comp[0, "X_I"]] = 0.035535
        self.unit_solutions[m.fs.unit.overflow.conc_mass_comp[0, "X_PAO"]] = 0.0044955
        self.unit_solutions[m.fs.unit.overflow.conc_mass_comp[0, "X_PHA"]] = (
            5.761182e-05
        )
        self.unit_solutions[m.fs.unit.overflow.conc_mass_comp[0, "X_PP"]] = 0.001336
        self.unit_solutions[m.fs.unit.overflow.conc_mass_comp[0, "X_S"]] = 0.001436

        self.conservation_equality = {
            "Check 1": {
                "in": m.fs.unit.inlet.flow_vol[0],
                "out": (
                    m.fs.unit.overflow.flow_vol[0] * m.fs.props.dens_mass
                    + m.fs.unit.underflow.flow_vol[0] * m.fs.props.dens_mass
                )
                / m.fs.props.dens_mass,
            },
            "Check 2": {
                "in": m.fs.unit.inlet.flow_vol[0]
                * sum(
                    m.fs.unit.inlet.conc_mass_comp[0, j] for j in m.fs.props.solute_set
                ),
                "out": sum(
                    m.fs.unit.overflow.flow_vol[0]
                    * m.fs.unit.overflow.conc_mass_comp[0, j]
                    + m.fs.unit.underflow.flow_vol[0]
                    * m.fs.unit.underflow.conc_mass_comp[0, j]
                    for j in m.fs.props.solute_set
                ),
            },
        }

        return m


class TestCosting(UnitTestHarness):
    def configure(self):
        m = build_ASM1()

        # Add unit model costing
        m.fs.costing = WaterTAPCosting()

        m.fs.unit.costing = UnitModelCostingBlock(
            flowsheet_costing_block=m.fs.costing,
        )

        iscale.set_scaling_factor(m.fs.unit.costing.capital_cost, 1e-6)

        m.fs.costing.cost_process()

        self.unit_solutions[m.fs.unit.costing.capital_cost] = 220675.79 * 2
        self.unit_solutions[m.fs.unit.electricity_consumption[0]] = 12.5 * 0.01255

        self.conservation_equality = {
            "Check 1": {
                "in": m.fs.unit.inlet.flow_vol[0],
                "out": m.fs.unit.underflow.flow_vol[0] + m.fs.unit.overflow.flow_vol[0],
            },
        }

        return m


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
        m.fs.unit = Thickener(
            property_package=m.fs.props, outlet_list=["outlet1", "outlet2"]
        )


class TestThickenerScaler:
    @pytest.fixture
    def model(self):
        m = ConcreteModel()
        m.fs = FlowsheetBlock(dynamic=False)

        m.fs.props = ASM1ParameterBlock()

        m.fs.unit = Thickener(
            property_package=m.fs.props,
            outlet_list=["underflow", "overflow"],
            split_basis=SplittingType.componentFlow,
        )

        m.fs.unit.inlet.flow_vol.fix(300 * units.m**3 / units.day)

        m.fs.unit.inlet.temperature.fix(308.15 * units.K)
        m.fs.unit.inlet.pressure.fix(1 * units.atm)

        m.fs.unit.inlet.conc_mass_comp[0, "S_I"].fix(28.0643 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_S"].fix(0.67336 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_I"].fix(3036.2175 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_S"].fix(63.2392 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_BH"].fix(
            4442.8377 * units.mg / units.liter
        )
        m.fs.unit.inlet.conc_mass_comp[0, "X_BA"].fix(332.5958 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_P"].fix(1922.8108 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_O"].fix(1.3748 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_NO"].fix(9.1948 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_NH"].fix(0.15845 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_ND"].fix(0.55943 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_ND"].fix(4.7411 * units.mg / units.liter)
        m.fs.unit.inlet.alkalinity.fix(4.5646 * units.mol / units.m**3)

        m.fs.unit.hydraulic_retention_time.fix()
        m.fs.unit.diameter.fix()

        return m

    @pytest.mark.component
    def test_variable_scaling_routine(self, model):
        scaler = model.fs.unit.default_scaler()

        assert isinstance(scaler, ThickenerScaler)

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

        assert isinstance(scaler, ThickenerScaler)

        scaler.constraint_scaling_routine(model.fs.unit)

        sfx_unit = model.fs.unit.scaling_factor
        assert isinstance(sfx_unit, Suffix)
        assert len(sfx_unit) == 63
        assert sfx_unit[model.fs.unit.eq_electricity_consumption[0]] == pytest.approx(
            1, rel=1e-8
        )
        assert sfx_unit[model.fs.unit.eq_hydraulic_retention[0]] == pytest.approx(
            5.5555555556e-4, rel=1e-8
        )
        assert sfx_unit[model.fs.unit.eq_volume[0]] == pytest.approx(
            5.5555555556e-4, rel=1e-8
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

        assert isinstance(scaler, ThickenerScaler)

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
        assert len(sfx_unit) == 64
        assert sfx_unit[model.fs.unit.volume[0]] == pytest.approx(1e-3, rel=1e-3)
        assert sfx_unit[model.fs.unit.eq_electricity_consumption[0]] == pytest.approx(
            1, rel=1e-8
        )
        assert sfx_unit[model.fs.unit.eq_hydraulic_retention[0]] == pytest.approx(
            0.001, rel=1e-8
        )
        assert sfx_unit[model.fs.unit.eq_volume[0]] == pytest.approx(0.001, rel=1e-8)
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

        m.fs.unit = Thickener(
            property_package=m.fs.props,
            outlet_list=["underflow", "overflow"],
            split_basis=SplittingType.componentFlow,
        )

        m.fs.unit.inlet.flow_vol.fix(300 * units.m**3 / units.day)

        m.fs.unit.inlet.temperature.fix(308.15 * units.K)
        m.fs.unit.inlet.pressure.fix(1 * units.atm)

        m.fs.unit.inlet.conc_mass_comp[0, "S_I"].fix(28.0643 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_S"].fix(0.67336 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_I"].fix(3036.2175 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_S"].fix(63.2392 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_BH"].fix(
            4442.8377 * units.mg / units.liter
        )
        m.fs.unit.inlet.conc_mass_comp[0, "X_BA"].fix(332.5958 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_P"].fix(1922.8108 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_O"].fix(1.3748 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_NO"].fix(9.1948 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_NH"].fix(0.15845 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_ND"].fix(0.55943 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_ND"].fix(4.7411 * units.mg / units.liter)
        m.fs.unit.inlet.alkalinity.fix(4.5646 * units.mol / units.m**3)

        m.fs.unit.hydraulic_retention_time.fix()
        m.fs.unit.diameter.fix()

        # Set scaling factors for badly scaled variables
        iscale.set_scaling_factor(m.fs.unit.underflow_state[0.0].flow_vol, 1e4)
        iscale.set_scaling_factor(m.fs.unit.underflow_state[0.0].pressure, 1e-6)
        iscale.set_scaling_factor(
            m.fs.unit.underflow_state[0.0].conc_mass_comp["S_S"], 1e4
        )
        iscale.set_scaling_factor(
            m.fs.unit.underflow_state[0.0].conc_mass_comp["S_NH"], 1e4
        )
        iscale.set_scaling_factor(
            m.fs.unit.underflow_state[0.0].conc_mass_comp["S_ND"], 1e4
        )
        iscale.set_scaling_factor(m.fs.unit.overflow_state[0.0].pressure, 1e-6)
        iscale.set_scaling_factor(
            m.fs.unit.overflow_state[0.0].conc_mass_comp["S_S"], 1e4
        )
        iscale.set_scaling_factor(
            m.fs.unit.overflow_state[0.0].conc_mass_comp["S_NH"], 1e4
        )
        iscale.set_scaling_factor(
            m.fs.unit.overflow_state[0.0].conc_mass_comp["S_ND"], 1e4
        )
        iscale.set_scaling_factor(
            m.fs.unit.overflow_state[0.0].conc_mass_comp["X_ND"], 1e4
        )

        iscale.calculate_scaling_factors(m.fs.unit)

        # Check condition number to confirm scaling
        sm = TransformationFactory("core.scale_model").create_using(m, rename=False)
        jac, _ = get_jacobian(sm, scaled=False)
        assert (jacobian_cond(jac=jac, scaled=False)) == pytest.approx(
            1.63071968e8, rel=1e-3
        )

    @pytest.mark.integration
    def test_example_case_scaler(self):
        m = ConcreteModel()
        m.fs = FlowsheetBlock(dynamic=False)

        m.fs.props = ASM1ParameterBlock()

        m.fs.unit = Thickener(
            property_package=m.fs.props,
            outlet_list=["underflow", "overflow"],
            split_basis=SplittingType.componentFlow,
        )

        m.fs.unit.inlet.flow_vol.fix(300 * units.m**3 / units.day)

        m.fs.unit.inlet.temperature.fix(308.15 * units.K)
        m.fs.unit.inlet.pressure.fix(1 * units.atm)

        m.fs.unit.inlet.conc_mass_comp[0, "S_I"].fix(28.0643 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_S"].fix(0.67336 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_I"].fix(3036.2175 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_S"].fix(63.2392 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_BH"].fix(
            4442.8377 * units.mg / units.liter
        )
        m.fs.unit.inlet.conc_mass_comp[0, "X_BA"].fix(332.5958 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_P"].fix(1922.8108 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_O"].fix(1.3748 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_NO"].fix(9.1948 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_NH"].fix(0.15845 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_ND"].fix(0.55943 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_ND"].fix(4.7411 * units.mg / units.liter)
        m.fs.unit.inlet.alkalinity.fix(4.5646 * units.mol / units.m**3)

        m.fs.unit.hydraulic_retention_time.fix()
        m.fs.unit.diameter.fix()

        sb = ScalerBase()

        # Apply scaling to poorly scaled variables
        for var in m.fs.component_data_objects(Var, descend_into=True):
            if "flow_vol" in var.name:
                sb.set_variable_scaling_factor(var, 1)
            if "conc_mass_comp" in var.name:
                sb.set_variable_scaling_factor(var, 1e1)

        scaler = ThickenerScaler()
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
            4.8463788512e2, rel=1e-3
        )


def build_model():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)

    m.fs.props = ASM1ParameterBlock()

    m.fs.unit = Thickener(
        property_package=m.fs.props,
        outlet_list=["underflow", "overflow"],
        split_basis=SplittingType.componentFlow,
    )

    m.fs.unit.inlet.flow_vol.fix(300 * units.m**3 / units.day)

    m.fs.unit.inlet.temperature.fix(308.15 * units.K)
    m.fs.unit.inlet.pressure.fix(1 * units.atm)

    m.fs.unit.inlet.conc_mass_comp[0, "S_I"].fix(28.0643 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "S_S"].fix(0.67336 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "X_I"].fix(3036.2175 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "X_S"].fix(63.2392 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "X_BH"].fix(4442.8377 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "X_BA"].fix(332.5958 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "X_P"].fix(1922.8108 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "S_O"].fix(1.3748 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "S_NO"].fix(9.1948 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "S_NH"].fix(0.15845 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "S_ND"].fix(0.55943 * units.mg / units.liter)
    m.fs.unit.inlet.conc_mass_comp[0, "X_ND"].fix(4.7411 * units.mg / units.liter)
    m.fs.unit.inlet.alkalinity.fix(4.5646 * units.mol / units.m**3)

    m.fs.unit.hydraulic_retention_time.fix()
    m.fs.unit.diameter.fix()

    solver = get_solver()
    solver.solve(m)

    return m


def scale_vars_with_scalers(m):
    scaler = ThickenerScaler()
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
    iscale.set_scaling_factor(m.fs.unit.underflow_state[0.0].flow_vol, 1e4)
    iscale.set_scaling_factor(m.fs.unit.underflow_state[0.0].pressure, 1e-6)
    iscale.set_scaling_factor(m.fs.unit.underflow_state[0.0].conc_mass_comp["S_S"], 1e4)
    iscale.set_scaling_factor(
        m.fs.unit.underflow_state[0.0].conc_mass_comp["S_NH"], 1e4
    )
    iscale.set_scaling_factor(
        m.fs.unit.underflow_state[0.0].conc_mass_comp["S_ND"], 1e4
    )
    iscale.set_scaling_factor(m.fs.unit.overflow_state[0.0].pressure, 1e-6)
    iscale.set_scaling_factor(m.fs.unit.overflow_state[0.0].conc_mass_comp["S_S"], 1e4)
    iscale.set_scaling_factor(m.fs.unit.overflow_state[0.0].conc_mass_comp["S_NH"], 1e4)
    iscale.set_scaling_factor(m.fs.unit.overflow_state[0.0].conc_mass_comp["S_ND"], 1e4)
    iscale.set_scaling_factor(m.fs.unit.overflow_state[0.0].conc_mass_comp["X_ND"], 1e4)

    iscale.calculate_scaling_factors(m.fs.unit)


def perturb_solution(m):
    m.fs.unit.inlet.flow_vol.fix(300 * 0.8 * units.m**3 / units.day)
    m.fs.unit.inlet.conc_mass_comp[0, "S_I"].fix(
        28.0643 * 0.55 * units.mg / units.liter
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
Unscaled                 || 1.384E+07 | Solved 1   ||
Vars Only                || 7.865E+06 | Solved 1   || 2.767E+12 | Solved 1  
Harmonic                 || 7.865E+06 | Solved 1   || 1.083E+03 | Solved 1  
Inverse Sum              || 7.865E+06 | Solved 1   || 1.968E+03 | Solved 1  
Inverse Root Sum Squares || 7.865E+06 | Solved 1   || 1.950E+03 | Solved 1  
Inverse Maximum          || 7.865E+06 | Solved 1   || 1.968E+03 | Solved 1  
Inverse Minimum          || 7.865E+06 | Solved 1   || 1.443E+03 | Solved 1  
Nominal L1 Norm          || 7.865E+06 | Solved 1   || 4.416E+02 | Solved 1  
Nominal L2 Norm          || 7.865E+06 | Solved 1   || 4.802E+02 | Solved 1  
Actual L1 Norm           || 7.865E+06 | Solved 1   || 4.769E+02 | Solved 1  
Actual L2 Norm           || 7.865E+06 | Solved 1   || 5.404E+02 | Solved 1  
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
Unscaled                 || 1.384E+07 | Solved 1   ||
Vars Only                || 7.339E+09 | Solved 1   || 2.767E+12 | Solved 1  
Harmonic                 || 1.566E+06 | Solved 1   || 1.083E+03 | Solved 1  
Inverse Sum              || 1.991E+06 | Solved 1   || 1.968E+03 | Solved 1  
Inverse Root Sum Squares || 1.804E+06 | Solved 1   || 1.950E+03 | Solved 1  
Inverse Maximum          || 1.677E+06 | Solved 1   || 1.968E+03 | Solved 1  
Inverse Minimum          || 2.279E+06 | Solved 1   || 1.443E+03 | Solved 1  
Nominal L1 Norm          || 3.505E+09 | Solved 1   || 4.416E+02 | Solved 1  
Nominal L2 Norm          || 3.327E+09 | Solved 1   || 4.802E+02 | Solved 1  
Actual L1 Norm           || 3.015E+04 | Solved 1   || 4.769E+02 | Solved 1  
Actual L2 Norm           || 3.344E+04 | Solved 1   || 5.404E+02 | Solved 1  
============================================================================
"""

    assert stream.getvalue() == expected
