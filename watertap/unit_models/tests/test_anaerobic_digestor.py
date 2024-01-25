#################################################################################
# WaterTAP Copyright (c) 2020-2023, The Regents of the University of California,
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
Tests for anaerobic digestor example.

Verified against results from:

Rosen, C. and Jeppsson, U., 2006.
Aspects on ADM1 Implementation within the BSM2 Framework.
Department of Industrial Electrical Engineering and Automation, Lund University, Lund, Sweden, pp.1-35.

"""

import pytest
from pyomo.environ import (
    ConcreteModel,
)

from idaes.core import (
    FlowsheetBlock,
)

from idaes.core.solvers import get_solver

from watertap.unit_models.anaerobic_digestor import AD
from watertap.property_models.anaerobic_digestion.adm1_properties import (
    ADM1ParameterBlock,
)
from watertap.property_models.anaerobic_digestion.adm1_properties_vapor import (
    ADM1_vaporParameterBlock,
)
from watertap.property_models.anaerobic_digestion.adm1_reactions import (
    ADM1ReactionParameterBlock,
)

from watertap.unit_models.tests.unit_test_harness import UnitTestHarness
import idaes.core.util.scaling as iscale

from idaes.core import UnitModelCostingBlock
from watertap.costing import WaterTAPCosting

# -----------------------------------------------------------------------------
# Get default solver for testing
solver = get_solver()


# -----------------------------------------------------------------------------
class TestUnitDefault(UnitTestHarness):
    @pytest.mark.unit
    def configure(self):
        m = ConcreteModel()
        m.fs = FlowsheetBlock(dynamic=False)

        m.fs.props = ADM1ParameterBlock()
        m.fs.props_vap = ADM1_vaporParameterBlock()
        m.fs.rxn_props = ADM1ReactionParameterBlock(property_package=m.fs.props)

        m.fs.unit = AD(
            liquid_property_package=m.fs.props,
            vapor_property_package=m.fs.props_vap,
            reaction_package=m.fs.rxn_props,
            has_heat_transfer=True,
            has_pressure_change=False,
        )

        # Set the operating conditions
        m.fs.unit.inlet.flow_vol.fix(170 / 24 / 3600)
        m.fs.unit.inlet.temperature.fix(308.15)
        m.fs.unit.inlet.pressure.fix(101325)

        m.fs.unit.inlet.conc_mass_comp[0, "S_su"].fix(0.01)
        m.fs.unit.inlet.conc_mass_comp[0, "S_aa"].fix(0.001)
        m.fs.unit.inlet.conc_mass_comp[0, "S_fa"].fix(0.001)
        m.fs.unit.inlet.conc_mass_comp[0, "S_va"].fix(0.001)
        m.fs.unit.inlet.conc_mass_comp[0, "S_bu"].fix(0.001)
        m.fs.unit.inlet.conc_mass_comp[0, "S_pro"].fix(0.001)
        m.fs.unit.inlet.conc_mass_comp[0, "S_ac"].fix(0.001)
        m.fs.unit.inlet.conc_mass_comp[0, "S_h2"].fix(1e-8)
        m.fs.unit.inlet.conc_mass_comp[0, "S_ch4"].fix(1e-5)
        m.fs.unit.inlet.conc_mass_comp[0, "S_IC"].fix(0.48)
        m.fs.unit.inlet.conc_mass_comp[0, "S_IN"].fix(0.14)
        m.fs.unit.inlet.conc_mass_comp[0, "S_I"].fix(0.02)

        m.fs.unit.inlet.conc_mass_comp[0, "X_c"].fix(2)
        m.fs.unit.inlet.conc_mass_comp[0, "X_ch"].fix(5)
        m.fs.unit.inlet.conc_mass_comp[0, "X_pr"].fix(20)
        m.fs.unit.inlet.conc_mass_comp[0, "X_li"].fix(5)
        m.fs.unit.inlet.conc_mass_comp[0, "X_su"].fix(0.0)
        m.fs.unit.inlet.conc_mass_comp[0, "X_aa"].fix(0.010)
        m.fs.unit.inlet.conc_mass_comp[0, "X_fa"].fix(0.010)
        m.fs.unit.inlet.conc_mass_comp[0, "X_c4"].fix(0.010)
        m.fs.unit.inlet.conc_mass_comp[0, "X_pro"].fix(0.010)
        m.fs.unit.inlet.conc_mass_comp[0, "X_ac"].fix(0.010)
        m.fs.unit.inlet.conc_mass_comp[0, "X_h2"].fix(0.010)
        m.fs.unit.inlet.conc_mass_comp[0, "X_I"].fix(25)

        m.fs.unit.inlet.cations[0].fix(0.04)
        m.fs.unit.inlet.anions[0].fix(0.02)

        m.fs.unit.volume_liquid.fix(3400)
        m.fs.unit.volume_vapor.fix(300)
        m.fs.unit.liquid_outlet.temperature.fix(308.15)

        # Add unit model costing
        m.fs.costing = WaterTAPCosting()

        m.fs.unit.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)
        m.fs.costing.cost_process()
        # m.fs.costing.add_LCOW(m.fs.unit.liquid_phase.properties_out[0].flow_vol)

        # Set scaling factors for badly scaled variables
        iscale.set_scaling_factor(
            m.fs.unit.liquid_phase.mass_transfer_term[0, "Liq", "S_h2"], 1e7
        )
        iscale.set_scaling_factor(m.fs.unit.costing.capital_cost, 1e-6)

        self.unit_model_block = m.fs.unit
        self.unit_statistics = {
            "number_variables": 203,
            "number_total_constraints": 171,
            "number_unused_variables": 0,
        }
        self.unit_solutions = {
            "liquid_outlet.pressure[0]": 101325,
            "liquid_outlet.temperature[0]": 308.15,
            "liquid_outlet.conc_mass_comp[0, S_I]": 0.328772,
            "liquid_outlet.conc_mass_comp[0, S_aa]": 0.00531408,
            "liquid_outlet.conc_mass_comp[0, S_ac]": 0.197783,
            "liquid_outlet.conc_mass_comp[0, S_bu]": 0.0132484,
            "liquid_outlet.conc_mass_comp[0, S_ch4]": 0.0549707,
            "liquid_outlet.conc_mass_comp[0, S_fa]": 0.0986058,
            "liquid_outlet.conc_mass_comp[0, S_h2]": 2.35916e-07,
            "liquid_outlet.conc_mass_comp[0, S_pro]": 0.01578123,
            "liquid_outlet.conc_mass_comp[0, S_su]": 0.0119533,
            "liquid_outlet.conc_mass_comp[0, S_va]": 0.0116230,
            "liquid_outlet.conc_mass_comp[0, X_I]": 25.6217,
            "liquid_outlet.conc_mass_comp[0, X_aa]": 1.1793,
            "liquid_outlet.conc_mass_comp[0, X_ac]": 0.760653,
            "liquid_outlet.conc_mass_comp[0, X_c]": 0.308718,
            "liquid_outlet.conc_mass_comp[0, X_c4]": 0.431974,
            "liquid_outlet.conc_mass_comp[0, X_ch]": 0.0279475,
            "liquid_outlet.conc_mass_comp[0, X_fa]": 0.243068,
            "liquid_outlet.conc_mass_comp[0, X_h2]": 0.3170629,
            "liquid_outlet.conc_mass_comp[0, X_li]": 0.0294834,
            "liquid_outlet.conc_mass_comp[0, X_pr]": 0.102574,
            "liquid_outlet.conc_mass_comp[0, X_pro]": 0.137323,
            "liquid_outlet.conc_mass_comp[0, X_su]": 0.420219,
            "liquid_outlet.conc_mass_comp[0, S_IC]": 1.8321,
            "liquid_outlet.conc_mass_comp[0, S_IN]": 1.8232,
            "liquid_outlet.anions[0]": 0.02,
            "liquid_outlet.cations[0]": 0.04,
            "vapor_outlet.pressure[0]": 106659,
            "vapor_outlet.temperature[0]": 308.15,
            "vapor_outlet.flow_vol[0]": 0.032496,
            "vapor_outlet.conc_mass_comp[0, S_ch4]": 1.6216,
            "vapor_outlet.conc_mass_comp[0, S_co2]": 0.169417,
            "KH_co2[0]": 0.0271467,
            "KH_ch4[0]": 0.0011619,
            "KH_h2[0]": 7.38e-4,
            "electricity_consumption[0]": 23.7291,
            "hydraulic_retention_time[0]": 1880470.588,
            "costing.capital_cost": 2166581.415,
        }
