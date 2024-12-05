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
Tests for clarifier.
"""
__author__ = "Chenyu Wang"
from pyomo.environ import (
    ConcreteModel,
    units,
)

from idaes.core import (
    FlowsheetBlock,
)

from watertap.core.solvers import get_solver

from watertap.unit_models.tests.unit_test_harness import UnitTestHarness
import idaes.core.util.scaling as iscale

from watertap.unit_models.clarifier import Clarifier
from idaes.models.unit_models.separator import SplittingType

from watertap.property_models.unit_specific.activated_sludge.asm1_properties import (
    ASM1ParameterBlock,
)

from idaes.core import UnitModelCostingBlock
from watertap.costing import WaterTAPCosting
from watertap.costing.unit_models.clarifier import (
    cost_circular_clarifier,
    cost_rectangular_clarifier,
    cost_primary_clarifier,
)

# -----------------------------------------------------------------------------
# Get default solver for testing
solver = get_solver()


# -----------------------------------------------------------------------------
def build():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)

    m.fs.properties = ASM1ParameterBlock()

    m.fs.unit = Clarifier(
        property_package=m.fs.properties,
        outlet_list=["underflow", "effluent"],
        split_basis=SplittingType.componentFlow,
    )

    m.fs.unit.inlet.temperature.fix(298.15 * units.K)
    m.fs.unit.inlet.pressure.fix(1 * units.atm)
    m.fs.unit.inlet.flow_vol.fix(18446 * units.m**3 / units.day)

    m.fs.unit.inlet.conc_mass_comp[0, "S_I"].fix(27 * units.g / units.m**3)
    m.fs.unit.inlet.conc_mass_comp[0, "S_S"].fix(58 * units.g / units.m**3)
    m.fs.unit.inlet.conc_mass_comp[0, "X_I"].fix(92 * units.g / units.m**3)
    m.fs.unit.inlet.conc_mass_comp[0, "X_S"].fix(363 * units.g / units.m**3)
    m.fs.unit.inlet.conc_mass_comp[0, "X_BH"].fix(50 * units.g / units.m**3)
    m.fs.unit.inlet.conc_mass_comp[0, "X_BA"].fix(1e-3 * units.g / units.m**3)
    m.fs.unit.inlet.conc_mass_comp[0, "X_P"].fix(1e-3 * units.g / units.m**3)
    m.fs.unit.inlet.conc_mass_comp[0, "S_O"].fix(1e-3 * units.g / units.m**3)
    m.fs.unit.inlet.conc_mass_comp[0, "S_NO"].fix(1e-3 * units.g / units.m**3)
    m.fs.unit.inlet.conc_mass_comp[0, "S_NH"].fix(23 * units.g / units.m**3)
    m.fs.unit.inlet.conc_mass_comp[0, "S_ND"].fix(5 * units.g / units.m**3)
    m.fs.unit.inlet.conc_mass_comp[0, "X_ND"].fix(16 * units.g / units.m**3)

    # Alkalinity was given in mg/L based on C
    m.fs.unit.inlet.alkalinity[0].fix(7 * units.mol / units.m**3)

    # Unit option
    m.fs.unit.split_fraction[0, "effluent", "H2O"].fix(0.993)
    m.fs.unit.split_fraction[0, "effluent", "S_I"].fix(0.993)
    m.fs.unit.split_fraction[0, "effluent", "S_S"].fix(0.993)
    m.fs.unit.split_fraction[0, "effluent", "X_I"].fix(0.5192)
    m.fs.unit.split_fraction[0, "effluent", "X_S"].fix(0.5192)
    m.fs.unit.split_fraction[0, "effluent", "X_BH"].fix(0.5192)
    m.fs.unit.split_fraction[0, "effluent", "X_BA"].fix(0.5192)
    m.fs.unit.split_fraction[0, "effluent", "X_P"].fix(0.5192)
    m.fs.unit.split_fraction[0, "effluent", "S_O"].fix(0.993)
    m.fs.unit.split_fraction[0, "effluent", "S_NO"].fix(0.993)
    m.fs.unit.split_fraction[0, "effluent", "S_NH"].fix(0.993)
    m.fs.unit.split_fraction[0, "effluent", "S_ND"].fix(0.993)
    m.fs.unit.split_fraction[0, "effluent", "X_ND"].fix(0.5192)
    m.fs.unit.split_fraction[0, "effluent", "S_ALK"].fix(0.993)

    # Set scaling factors for badly scaled variables
    iscale.set_scaling_factor(
        m.fs.unit.underflow_state[0.0].conc_mass_comp["X_BA"], 1e3
    )
    iscale.set_scaling_factor(m.fs.unit.underflow_state[0.0].conc_mass_comp["X_P"], 1e3)
    iscale.set_scaling_factor(m.fs.unit.underflow_state[0.0].conc_mass_comp["S_O"], 1e3)
    iscale.set_scaling_factor(
        m.fs.unit.underflow_state[0.0].conc_mass_comp["S_NO"], 1e3
    )
    iscale.set_scaling_factor(m.fs.unit.effluent_state[0.0].conc_mass_comp["X_BA"], 1e7)
    iscale.set_scaling_factor(m.fs.unit.effluent_state[0.0].conc_mass_comp["X_P"], 1e7)
    iscale.set_scaling_factor(m.fs.unit.effluent_state[0.0].conc_mass_comp["S_O"], 1e7)
    iscale.set_scaling_factor(m.fs.unit.effluent_state[0.0].conc_mass_comp["S_NO"], 1e7)

    iscale.calculate_scaling_factors(m.fs.unit)

    return m


class TestClarifier(UnitTestHarness):
    def configure(self):
        self.default_small = 1e-08
        m = build()

        self.unit_solutions[m.fs.unit.underflow.conc_mass_comp[0, "S_I"]] = 0.027
        self.unit_solutions[m.fs.unit.underflow.conc_mass_comp[0, "S_S"]] = 0.058
        self.unit_solutions[m.fs.unit.underflow.conc_mass_comp[0, "X_I"]] = 6.3190857
        self.unit_solutions[m.fs.unit.underflow.conc_mass_comp[0, "X_S"]] = 24.9329142
        self.unit_solutions[m.fs.unit.underflow.conc_mass_comp[0, "X_BH"]] = 3.4342857
        self.unit_solutions[m.fs.unit.underflow.conc_mass_comp[0, "X_BA"]] = 6.868571e-5
        self.unit_solutions[m.fs.unit.underflow.conc_mass_comp[0, "X_P"]] = 6.868571e-5
        self.unit_solutions[m.fs.unit.underflow.conc_mass_comp[0, "S_O"]] = 1e-6
        self.unit_solutions[m.fs.unit.underflow.conc_mass_comp[0, "S_NO"]] = 1e-6
        self.unit_solutions[m.fs.unit.underflow.conc_mass_comp[0, "S_NH"]] = 0.023
        self.unit_solutions[m.fs.unit.underflow.conc_mass_comp[0, "S_ND"]] = 0.005
        self.unit_solutions[m.fs.unit.underflow.conc_mass_comp[0, "X_ND"]] = 1.098971

        self.unit_solutions[m.fs.unit.effluent.conc_mass_comp[0, "S_I"]] = 0.027
        self.unit_solutions[m.fs.unit.effluent.conc_mass_comp[0, "S_S"]] = 0.058
        self.unit_solutions[m.fs.unit.effluent.conc_mass_comp[0, "X_I"]] = 0.0481031
        self.unit_solutions[m.fs.unit.effluent.conc_mass_comp[0, "X_S"]] = 0.189798
        self.unit_solutions[m.fs.unit.effluent.conc_mass_comp[0, "X_BH"]] = 0.0261430
        self.unit_solutions[m.fs.unit.effluent.conc_mass_comp[0, "X_BA"]] = 5.228600e-7
        self.unit_solutions[m.fs.unit.effluent.conc_mass_comp[0, "X_P"]] = 5.228600e-7
        self.unit_solutions[m.fs.unit.effluent.conc_mass_comp[0, "S_O"]] = 1e-6
        self.unit_solutions[m.fs.unit.effluent.conc_mass_comp[0, "S_NO"]] = 1e-6
        self.unit_solutions[m.fs.unit.effluent.conc_mass_comp[0, "S_NH"]] = 0.023
        self.unit_solutions[m.fs.unit.effluent.conc_mass_comp[0, "S_ND"]] = 0.005
        self.unit_solutions[m.fs.unit.effluent.conc_mass_comp[0, "X_ND"]] = 0.00836576

        self.conservation_equality = {
            "Check 1": {
                "in": m.fs.unit.inlet.flow_vol[0]
                * (
                    m.fs.unit.inlet.conc_mass_comp[0, "S_I"]
                    + m.fs.unit.inlet.conc_mass_comp[0, "S_S"]
                    + m.fs.unit.inlet.conc_mass_comp[0, "X_I"]
                    + m.fs.unit.inlet.conc_mass_comp[0, "X_S"]
                    + m.fs.unit.inlet.conc_mass_comp[0, "X_BH"]
                    + m.fs.unit.inlet.conc_mass_comp[0, "X_BA"]
                    + m.fs.unit.inlet.conc_mass_comp[0, "X_P"]
                    + m.fs.unit.inlet.conc_mass_comp[0, "S_O"]
                    + m.fs.unit.inlet.conc_mass_comp[0, "S_NO"]
                    + m.fs.unit.inlet.conc_mass_comp[0, "S_NH"]
                    + m.fs.unit.inlet.conc_mass_comp[0, "S_ND"]
                    + m.fs.unit.inlet.conc_mass_comp[0, "X_ND"]
                ),
                "out": m.fs.unit.underflow.flow_vol[0]
                * (
                    m.fs.unit.underflow.conc_mass_comp[0, "S_I"]
                    + m.fs.unit.underflow.conc_mass_comp[0, "S_S"]
                    + m.fs.unit.underflow.conc_mass_comp[0, "X_I"]
                    + m.fs.unit.underflow.conc_mass_comp[0, "X_S"]
                    + m.fs.unit.underflow.conc_mass_comp[0, "X_BH"]
                    + m.fs.unit.underflow.conc_mass_comp[0, "X_BA"]
                    + m.fs.unit.underflow.conc_mass_comp[0, "X_P"]
                    + m.fs.unit.underflow.conc_mass_comp[0, "S_O"]
                    + m.fs.unit.underflow.conc_mass_comp[0, "S_NO"]
                    + m.fs.unit.underflow.conc_mass_comp[0, "S_NH"]
                    + m.fs.unit.underflow.conc_mass_comp[0, "S_ND"]
                    + m.fs.unit.underflow.conc_mass_comp[0, "X_ND"]
                )
                + m.fs.unit.effluent.flow_vol[0]
                * (
                    m.fs.unit.effluent.conc_mass_comp[0, "S_I"]
                    + m.fs.unit.effluent.conc_mass_comp[0, "S_S"]
                    + m.fs.unit.effluent.conc_mass_comp[0, "X_I"]
                    + m.fs.unit.effluent.conc_mass_comp[0, "X_S"]
                    + m.fs.unit.effluent.conc_mass_comp[0, "X_BH"]
                    + m.fs.unit.effluent.conc_mass_comp[0, "X_BA"]
                    + m.fs.unit.effluent.conc_mass_comp[0, "X_P"]
                    + m.fs.unit.effluent.conc_mass_comp[0, "S_O"]
                    + m.fs.unit.effluent.conc_mass_comp[0, "S_NO"]
                    + m.fs.unit.effluent.conc_mass_comp[0, "S_NH"]
                    + m.fs.unit.effluent.conc_mass_comp[0, "S_ND"]
                    + m.fs.unit.effluent.conc_mass_comp[0, "X_ND"]
                ),
            },
        }

        return m


class TestCircularCosting(UnitTestHarness):
    def configure(self):
        self.default_small = 1e-08
        m = build()

        # Add unit model costing
        m.fs.costing = WaterTAPCosting()

        m.fs.unit.costing = UnitModelCostingBlock(
            flowsheet_costing_block=m.fs.costing, costing_method=cost_circular_clarifier
        )
        m.fs.unit.surface_area.fix(1500 * units.m**2)

        iscale.set_scaling_factor(m.fs.unit.costing.capital_cost, 1e-6)

        m.fs.costing.cost_process()

        self.unit_solutions[m.fs.unit.costing.capital_cost] = 1681573 * 2

        self.conservation_equality = {
            "Check 1": {
                "in": m.fs.unit.inlet.flow_vol[0],
                "out": m.fs.unit.underflow.flow_vol[0] + m.fs.unit.effluent.flow_vol[0],
            },
        }

        return m


class TestRectangularCosting(UnitTestHarness):
    def configure(self):
        self.default_small = 1e-08
        m = build()

        # Add unit model costing
        m.fs.costing = WaterTAPCosting()

        m.fs.unit.costing = UnitModelCostingBlock(
            flowsheet_costing_block=m.fs.costing,
            costing_method=cost_rectangular_clarifier,
        )
        m.fs.unit.surface_area.fix(1500 * units.m**2)

        iscale.set_scaling_factor(m.fs.unit.costing.capital_cost, 1e-6)

        m.fs.costing.cost_process()

        self.unit_solutions[m.fs.unit.costing.capital_cost] = 2131584 * 2

        self.conservation_equality = {
            "Check 1": {
                "in": m.fs.unit.inlet.flow_vol[0],
                "out": m.fs.unit.underflow.flow_vol[0] + m.fs.unit.effluent.flow_vol[0],
            },
        }

        return m


class TestPrimaryClarifierCosting(UnitTestHarness):
    def configure(self):
        self.default_small = 1e-08
        m = build()

        # Add unit model costing
        m.fs.costing = WaterTAPCosting()

        m.fs.unit.costing = UnitModelCostingBlock(
            flowsheet_costing_block=m.fs.costing, costing_method=cost_primary_clarifier
        )
        m.fs.unit.surface_area.fix(1500 * units.m**2)

        iscale.set_scaling_factor(m.fs.unit.costing.capital_cost, 1e-6)

        m.fs.costing.cost_process()

        self.unit_solutions[m.fs.unit.costing.capital_cost] = 1390570 * 2

        self.conservation_equality = {
            "Check 1": {
                "in": m.fs.unit.inlet.flow_vol[0],
                "out": m.fs.unit.underflow.flow_vol[0] + m.fs.unit.effluent.flow_vol[0],
            },
        }

        return m
