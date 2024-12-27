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
Tests for ASM1-ADM1 Translator unit model.

Verified against approximated results results from:

Nopens, I., Batstone, D, Copp, J, Jeppsson, U. Rosen, C., Volcke, E., Alex, J.,
and Vanrolleghem, P.
water Research Vol. 43 pp.1913–1923.
"""

import pytest
from pyomo.environ import (
    ConcreteModel,
    value,
    assert_optimal_termination,
    Param,
    Suffix,
    TransformationFactory,
)

from idaes.core import FlowsheetBlock
import idaes.core.util.scaling as iscale
from pyomo.environ import units

from watertap.core.solvers import get_solver
from idaes.core.util.model_statistics import (
    degrees_of_freedom,
    number_variables,
    number_total_constraints,
    number_unused_variables,
)

import idaes.logger as idaeslog
from idaes.core.util.testing import initialization_tester
from idaes.core.util.scaling import (
    get_jacobian,
    jacobian_cond,
)
from idaes.core.scaling.scaling_base import ScalerBase

from watertap.unit_models.translators.translator_asm1_adm1 import (
    Translator_ASM1_ADM1,
    ASM1ADM1Scaler,
)
from watertap.property_models.unit_specific.anaerobic_digestion.adm1_properties import (
    ADM1ParameterBlock,
    ADM1PropertiesScaler,
)

from watertap.property_models.unit_specific.activated_sludge.asm1_properties import (
    ASM1ParameterBlock,
    ASM1PropertiesScaler,
)

from watertap.property_models.unit_specific.anaerobic_digestion.adm1_reactions import (
    ADM1ReactionParameterBlock,
)


from pyomo.util.check_units import assert_units_consistent

# -----------------------------------------------------------------------------
# Get default solver for testing
solver = get_solver()


# -----------------------------------------------------------------------------
@pytest.mark.unit
def test_config():
    m = ConcreteModel()

    m.fs = FlowsheetBlock(dynamic=False)

    m.fs.props_ASM1 = ASM1ParameterBlock()
    m.fs.props_ADM1 = ADM1ParameterBlock()
    m.fs.ADM1_rxn_props = ADM1ReactionParameterBlock(property_package=m.fs.props_ADM1)

    m.fs.unit = Translator_ASM1_ADM1(
        inlet_property_package=m.fs.props_ASM1,
        outlet_property_package=m.fs.props_ADM1,
        reaction_package=m.fs.ADM1_rxn_props,
        has_phase_equilibrium=False,
        outlet_state_defined=True,
    )

    assert len(m.fs.unit.config) == 10

    assert m.fs.unit.config.outlet_state_defined == True
    assert not m.fs.unit.config.dynamic
    assert not m.fs.unit.config.has_holdup
    assert not m.fs.unit.config.has_phase_equilibrium
    assert m.fs.unit.config.inlet_property_package is m.fs.props_ASM1
    assert m.fs.unit.config.outlet_property_package is m.fs.props_ADM1
    assert m.fs.unit.config.reaction_package is m.fs.ADM1_rxn_props


# -----------------------------------------------------------------------------
class TestAsm1Adm1(object):
    @pytest.fixture(scope="class")
    def asmadm(self):
        m = ConcreteModel()

        m.fs = FlowsheetBlock(dynamic=False)

        m.fs.props_ASM1 = ASM1ParameterBlock()
        m.fs.props_ADM1 = ADM1ParameterBlock()
        m.fs.ADM1_rxn_props = ADM1ReactionParameterBlock(
            property_package=m.fs.props_ADM1
        )

        m.fs.unit = Translator_ASM1_ADM1(
            inlet_property_package=m.fs.props_ASM1,
            outlet_property_package=m.fs.props_ADM1,
            reaction_package=m.fs.ADM1_rxn_props,
            has_phase_equilibrium=False,
            outlet_state_defined=True,
        )

        m.fs.unit.inlet.flow_vol.fix(178.4674 * units.m**3 / units.day)
        m.fs.unit.inlet.temperature.fix(308.15 * units.K)
        m.fs.unit.inlet.pressure.fix(1 * units.atm)

        m.fs.unit.inlet.conc_mass_comp[0, "S_I"].fix(28.0665 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_S"].fix(48.9526 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_I"].fix(
            10361.7101 * units.mg / units.liter
        )
        m.fs.unit.inlet.conc_mass_comp[0, "X_S"].fix(
            20375.0176 * units.mg / units.liter
        )
        m.fs.unit.inlet.conc_mass_comp[0, "X_BH"].fix(
            10210.0698 * units.mg / units.liter
        )
        m.fs.unit.inlet.conc_mass_comp[0, "X_BA"].fix(553.2808 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_P"].fix(3204.6601 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_O"].fix(0.25225 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_NO"].fix(1.6871 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_NH"].fix(28.9098 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_ND"].fix(4.6834 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_ND"].fix(906.0933 * units.mg / units.liter)
        m.fs.unit.inlet.alkalinity.fix(7.1549 * units.mol / units.m**3)

        iscale.calculate_scaling_factors(m)

        return m

    @pytest.mark.build
    @pytest.mark.unit
    def test_build(self, asmadm):
        assert isinstance(asmadm.fs.unit.i_xe, Param)
        assert value(asmadm.fs.unit.i_xe) == 0.06
        assert isinstance(asmadm.fs.unit.i_xb, Param)
        assert value(asmadm.fs.unit.i_xb) == 0.08
        assert isinstance(asmadm.fs.unit.f_xI, Param)
        assert value(asmadm.fs.unit.f_xI) == 0.05

        assert hasattr(asmadm.fs.unit, "inlet")
        assert len(asmadm.fs.unit.inlet.vars) == 5
        assert hasattr(asmadm.fs.unit.inlet, "flow_vol")
        assert hasattr(asmadm.fs.unit.inlet, "conc_mass_comp")
        assert hasattr(asmadm.fs.unit.inlet, "temperature")
        assert hasattr(asmadm.fs.unit.inlet, "pressure")
        assert hasattr(asmadm.fs.unit.inlet, "alkalinity")

        assert hasattr(asmadm.fs.unit, "outlet")
        assert len(asmadm.fs.unit.outlet.vars) == 6
        assert hasattr(asmadm.fs.unit.outlet, "flow_vol")
        assert hasattr(asmadm.fs.unit.outlet, "conc_mass_comp")
        assert hasattr(asmadm.fs.unit.outlet, "temperature")
        assert hasattr(asmadm.fs.unit.outlet, "pressure")
        assert hasattr(asmadm.fs.unit.outlet, "anions")
        assert hasattr(asmadm.fs.unit.outlet, "cations")

        assert number_variables(asmadm) == 141
        assert number_total_constraints(asmadm) == 34

        assert number_unused_variables(asmadm.fs.unit) == 0

    @pytest.mark.component
    def test_units(self, asmadm):
        assert_units_consistent(asmadm)

    @pytest.mark.unit
    def test_dof(self, asmadm):
        assert degrees_of_freedom(asmadm) == 0

    @pytest.mark.solver
    @pytest.mark.skipif(solver is None, reason="Solver not available")
    @pytest.mark.component
    def test_initialize(self, asmadm):
        initialization_tester(asmadm)

    @pytest.mark.solver
    @pytest.mark.skipif(solver is None, reason="Solver not available")
    @pytest.mark.component
    def test_solve(self, asmadm):
        asmadm.fs.unit.initialize(outlvl=idaeslog.INFO_HIGH)
        solver = get_solver()
        results = solver.solve(asmadm, tee=True)
        assert_optimal_termination(results)

    @pytest.mark.solver
    @pytest.mark.skipif(solver is None, reason="Solver not available")
    @pytest.mark.component
    def test_solution(self, asmadm):
        assert pytest.approx(101325.0, rel=1e-3) == value(
            asmadm.fs.unit.outlet.pressure[0]
        )
        assert pytest.approx(308.15, rel=1e-3) == value(
            asmadm.fs.unit.outlet.temperature[0]
        )
        assert pytest.approx(1e-10, abs=1e-6) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "S_su"]
        )
        assert pytest.approx(0.04388, rel=1e-3) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "S_aa"]
        )
        assert pytest.approx(1e-10, abs=1e-6) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "S_fa"]
        )
        assert pytest.approx(1e-10, abs=1e-6) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "S_va"]
        )
        assert pytest.approx(1e-10, abs=1e-6) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "S_bu"]
        )
        assert pytest.approx(1e-10, abs=1e-6) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "S_pro"]
        )
        assert pytest.approx(1e-10, abs=1e-6) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "S_ac"]
        )
        assert pytest.approx(1e-10, abs=1e-6) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "S_h2"]
        )
        assert pytest.approx(1e-10, abs=1e-6) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "S_ch4"]
        )
        assert pytest.approx(0.0858, rel=1e-3) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "S_IC"]
        )
        assert pytest.approx(0.91266, rel=1e-3) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "S_IN"]
        )
        assert pytest.approx(0.02806, rel=1e-3) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "S_I"]
        )

        assert pytest.approx(0.6782, rel=1e-3) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "X_I"]
        )
        assert pytest.approx(44.0264, rel=1e-3) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "X_c"]
        )
        assert pytest.approx(1e-10, abs=1e-6) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "X_ch"]
        )
        assert pytest.approx(1e-10, abs=1e-6) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "X_pr"]
        )
        assert pytest.approx(1e-10, abs=1e-6) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "X_li"]
        )
        assert pytest.approx(1e-10, abs=1e-6) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "X_su"]
        )
        assert pytest.approx(1e-10, abs=1e-6) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "X_aa"]
        )
        assert pytest.approx(1e-10, abs=1e-6) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "X_fa"]
        )
        assert pytest.approx(1e-10, abs=1e-6) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "X_c4"]
        )
        assert pytest.approx(1e-10, abs=1e-6) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "X_pro"]
        )
        assert pytest.approx(1e-10, abs=1e-6) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "X_ac"]
        )
        assert pytest.approx(1e-10, abs=1e-6) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "X_h2"]
        )
        assert pytest.approx(0.00715, rel=1e-3) == value(
            asmadm.fs.unit.outlet.cations[0]
        )
        assert pytest.approx(0.06517, rel=1e-3) == value(
            asmadm.fs.unit.outlet.anions[0]
        )

    @pytest.mark.solver
    @pytest.mark.skipif(solver is None, reason="Solver not available")
    @pytest.mark.component
    def test_conservation(self, asmadm):
        assert (
            abs(
                value(
                    asmadm.fs.unit.inlet.flow_vol[0] * asmadm.fs.props_ADM1.dens_mass
                    - asmadm.fs.unit.outlet.flow_vol[0] * asmadm.fs.props_ASM1.dens_mass
                )
            )
            <= 1e-6
        )

        assert (
            abs(
                value(
                    (
                        asmadm.fs.unit.inlet.flow_vol[0]
                        * asmadm.fs.props_ADM1.dens_mass
                        * asmadm.fs.props_ADM1.cp_mass
                        * (
                            asmadm.fs.unit.inlet.temperature[0]
                            - asmadm.fs.props_ADM1.temperature_ref
                        )
                    )
                    - (
                        asmadm.fs.unit.outlet.flow_vol[0]
                        * asmadm.fs.props_ASM1.dens_mass
                        * asmadm.fs.props_ASM1.cp_mass
                        * (
                            asmadm.fs.unit.outlet.temperature[0]
                            - asmadm.fs.props_ASM1.temperature_ref
                        )
                    )
                )
            )
            <= 1e-6
        )

    @pytest.mark.solver
    @pytest.mark.skipif(solver is None, reason="Solver not available")
    @pytest.mark.component
    def test_cod_tkn_conservation(self, asmadm):
        assert (
            abs(
                value(
                    asmadm.fs.unit.CODs[0]
                    - (
                        asmadm.fs.unit.outlet.conc_mass_comp[0, "S_I"]
                        + asmadm.fs.unit.outlet.conc_mass_comp[0, "S_su"]
                        + asmadm.fs.unit.outlet.conc_mass_comp[0, "S_aa"]
                        + asmadm.fs.unit.outlet.conc_mass_comp[0, "S_fa"]
                        + asmadm.fs.unit.outlet.conc_mass_comp[0, "S_va"]
                        + asmadm.fs.unit.outlet.conc_mass_comp[0, "S_bu"]
                        + asmadm.fs.unit.outlet.conc_mass_comp[0, "S_pro"]
                        + asmadm.fs.unit.outlet.conc_mass_comp[0, "S_ac"]
                        + asmadm.fs.unit.outlet.conc_mass_comp[0, "S_h2"]
                        + asmadm.fs.unit.outlet.conc_mass_comp[0, "S_ch4"]
                    )
                )
            )
            <= 1e-5
        )

        assert (
            abs(
                value(
                    asmadm.fs.unit.CODp[0]
                    - (
                        asmadm.fs.unit.outlet.conc_mass_comp[0, "X_I"]
                        + asmadm.fs.unit.outlet.conc_mass_comp[0, "X_su"]
                        + asmadm.fs.unit.outlet.conc_mass_comp[0, "X_aa"]
                        + asmadm.fs.unit.outlet.conc_mass_comp[0, "X_fa"]
                        + asmadm.fs.unit.outlet.conc_mass_comp[0, "X_c4"]
                        + asmadm.fs.unit.outlet.conc_mass_comp[0, "X_pro"]
                        + asmadm.fs.unit.outlet.conc_mass_comp[0, "X_ac"]
                        + asmadm.fs.unit.outlet.conc_mass_comp[0, "X_h2"]
                        + asmadm.fs.unit.outlet.conc_mass_comp[0, "X_c"]
                        + asmadm.fs.unit.outlet.conc_mass_comp[0, "X_ch"]
                        + asmadm.fs.unit.outlet.conc_mass_comp[0, "X_pr"]
                        + asmadm.fs.unit.outlet.conc_mass_comp[0, "X_li"]
                    )
                )
            )
            <= 1e-5
        )

        assert (
            abs(
                value(
                    asmadm.fs.unit.TKN_in[0]
                    - (
                        asmadm.fs.unit.outlet.conc_mass_comp[0, "S_IN"]
                        + asmadm.fs.unit.config.reaction_package.N_xc
                        * 14
                        * units.kg
                        / units.kmol
                        * asmadm.fs.unit.outlet.conc_mass_comp[0, "X_c"]
                        + asmadm.fs.unit.config.reaction_package.N_I
                        * 14
                        * units.kg
                        / units.kmol
                        * (
                            asmadm.fs.unit.outlet.conc_mass_comp[0, "S_I"]
                            + asmadm.fs.unit.outlet.conc_mass_comp[0, "X_I"]
                        )
                        + asmadm.fs.unit.config.reaction_package.N_aa
                        * 14
                        * units.kg
                        / units.kmol
                        * (
                            asmadm.fs.unit.outlet.conc_mass_comp[0, "X_pr"]
                            + asmadm.fs.unit.outlet.conc_mass_comp[0, "S_aa"]
                        )
                        + asmadm.fs.unit.config.reaction_package.N_bac
                        * 14
                        * units.kg
                        / units.kmol
                        * (
                            asmadm.fs.unit.outlet.conc_mass_comp[0, "X_su"]
                            + asmadm.fs.unit.outlet.conc_mass_comp[0, "X_aa"]
                            + asmadm.fs.unit.outlet.conc_mass_comp[0, "X_fa"]
                            + asmadm.fs.unit.outlet.conc_mass_comp[0, "X_c4"]
                            + asmadm.fs.unit.outlet.conc_mass_comp[0, "X_pro"]
                            + asmadm.fs.unit.outlet.conc_mass_comp[0, "X_ac"]
                            + asmadm.fs.unit.outlet.conc_mass_comp[0, "X_h2"]
                        )
                    )
                )
            )
            <= 1e-6
        )

    @pytest.mark.solver
    @pytest.mark.skipif(solver is None, reason="Solver not available")
    @pytest.mark.component
    def test_intermidiates(self, asmadm):
        assert pytest.approx(0.00508, rel=1e-3) == value(asmadm.fs.unit.CODd[0])
        assert value(asmadm.fs.unit.CODd2[0]) <= 1e-6
        assert value(asmadm.fs.unit.CODd3[0]) <= 1e-6
        assert value(asmadm.fs.unit.CODd4[0]) <= 1e-6
        assert value(asmadm.fs.unit.CODd5[0]) <= 1e-6
        assert pytest.approx(44.7310, rel=1e-3) == value(asmadm.fs.unit.COD_remain_a[0])
        assert pytest.approx(44.703, rel=1e-3) == value(asmadm.fs.unit.COD_remain_b[0])
        assert pytest.approx(44.026, rel=1e-3) == value(asmadm.fs.unit.COD_remain_c[0])
        assert pytest.approx(2.5813, rel=1e-3) == value(asmadm.fs.unit.ORGN_remain_a[0])
        assert pytest.approx(2.5796, rel=1e-3) == value(asmadm.fs.unit.ORGN_remain_b[0])
        assert pytest.approx(2.5389, rel=1e-3) == value(asmadm.fs.unit.ORGN_remain_c[0])
        assert pytest.approx(0.001684, rel=1e-3) == value(asmadm.fs.unit.ReqOrgNS[0])
        assert pytest.approx(0.040699, rel=1e-3) == value(asmadm.fs.unit.ReqOrgNx[0])
        assert pytest.approx(67.5251, rel=1e-3) == value(asmadm.fs.unit.ReqCODXc[0])
        assert pytest.approx(0.04779, rel=1e-3) == value(asmadm.fs.unit.ReqCODs[0])


class TestASM1ADM1Scaler:
    @pytest.fixture
    def model(self):
        m = ConcreteModel()

        m.fs = FlowsheetBlock(dynamic=False)

        m = ConcreteModel()

        m.fs = FlowsheetBlock(dynamic=False)

        m.fs.props_ASM1 = ASM1ParameterBlock()
        m.fs.props_ADM1 = ADM1ParameterBlock()
        m.fs.ADM1_rxn_props = ADM1ReactionParameterBlock(
            property_package=m.fs.props_ADM1
        )

        m.fs.unit = Translator_ASM1_ADM1(
            inlet_property_package=m.fs.props_ASM1,
            outlet_property_package=m.fs.props_ADM1,
            reaction_package=m.fs.ADM1_rxn_props,
            has_phase_equilibrium=False,
            outlet_state_defined=True,
        )

        m.fs.unit.inlet.flow_vol.fix(178.4674 * units.m**3 / units.day)
        m.fs.unit.inlet.temperature.fix(308.15 * units.K)
        m.fs.unit.inlet.pressure.fix(1 * units.atm)

        m.fs.unit.inlet.conc_mass_comp[0, "S_I"].fix(28.0665 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_S"].fix(48.9526 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_I"].fix(
            10361.7101 * units.mg / units.liter
        )
        m.fs.unit.inlet.conc_mass_comp[0, "X_S"].fix(
            20375.0176 * units.mg / units.liter
        )
        m.fs.unit.inlet.conc_mass_comp[0, "X_BH"].fix(
            10210.0698 * units.mg / units.liter
        )
        m.fs.unit.inlet.conc_mass_comp[0, "X_BA"].fix(553.2808 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_P"].fix(3204.6601 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_O"].fix(0.25225 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_NO"].fix(1.6871 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_NH"].fix(28.9098 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_ND"].fix(4.6834 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_ND"].fix(906.0933 * units.mg / units.liter)
        m.fs.unit.inlet.alkalinity.fix(7.1549 * units.mol / units.m**3)

        return m

    @pytest.mark.component
    def test_variable_scaling_routine(self, model):
        scaler = model.fs.unit.default_scaler()

        assert isinstance(scaler, ASM1ADM1Scaler)

        scaler.variable_scaling_routine(model.fs.unit)

        # Inlet state
        sfx_in = model.fs.unit.properties_in[0].scaling_factor
        assert isinstance(sfx_in, Suffix)
        assert len(sfx_in) == 3
        assert sfx_in[model.fs.unit.properties_in[0].flow_vol] == pytest.approx(
            1e1, rel=1e-8
        )
        assert sfx_in[model.fs.unit.properties_in[0].pressure] == pytest.approx(
            1e-6, rel=1e-8
        )
        assert sfx_in[model.fs.unit.properties_in[0].temperature] == pytest.approx(
            1e-1, rel=1e-8
        )

        # Outlet state - should be the same as the inlet
        sfx_underflow = model.fs.unit.properties_out[0].scaling_factor
        assert isinstance(sfx_underflow, Suffix)
        assert len(sfx_underflow) == 3
        assert sfx_underflow[model.fs.unit.properties_out[0].flow_vol] == pytest.approx(
            1e5, rel=1e-8
        )
        assert sfx_underflow[model.fs.unit.properties_out[0].pressure] == pytest.approx(
            1e-6, rel=1e-8
        )
        assert sfx_underflow[
            model.fs.unit.properties_out[0].temperature
        ] == pytest.approx(1e-1, rel=1e-8)

    @pytest.mark.component
    def test_constraint_scaling_routine(self, model):
        scaler = model.fs.unit.default_scaler()

        assert isinstance(scaler, ASM1ADM1Scaler)

        scaler.constraint_scaling_routine(model.fs.unit)

        sfx_unit = model.fs.unit.scaling_factor
        assert isinstance(sfx_unit, Suffix)
        assert len(sfx_unit) == 34

    @pytest.mark.component
    def test_scale_model(self, model):
        scaler = model.fs.unit.default_scaler()

        assert isinstance(scaler, ASM1ADM1Scaler)

        scaler.scale_model(model.fs.unit)

        # Inlet state
        sfx_in = model.fs.unit.properties_in[0].scaling_factor
        assert isinstance(sfx_in, Suffix)
        assert len(sfx_in) == 3
        assert sfx_in[model.fs.unit.properties_in[0].flow_vol] == pytest.approx(
            1e1, rel=1e-8
        )
        assert sfx_in[model.fs.unit.properties_in[0].pressure] == pytest.approx(
            1e-6, rel=1e-8
        )
        assert sfx_in[model.fs.unit.properties_in[0].temperature] == pytest.approx(
            1e-1, rel=1e-8
        )

        # Outlet state - should be the same as the inlet
        sfx_underflow = model.fs.unit.properties_out[0].scaling_factor
        assert isinstance(sfx_underflow, Suffix)
        assert len(sfx_underflow) == 3
        assert sfx_underflow[model.fs.unit.properties_out[0].flow_vol] == pytest.approx(
            1e5, rel=1e-8
        )
        assert sfx_underflow[model.fs.unit.properties_out[0].pressure] == pytest.approx(
            1e-6, rel=1e-8
        )
        assert sfx_underflow[
            model.fs.unit.properties_out[0].temperature
        ] == pytest.approx(1e-1, rel=1e-8)

        sfx_overflow = model.fs.unit.properties_out[0].scaling_factor
        assert isinstance(sfx_overflow, Suffix)
        assert len(sfx_overflow) == 3
        assert sfx_overflow[model.fs.unit.properties_out[0].flow_vol] == pytest.approx(
            1e5, rel=1e-8
        )
        assert sfx_overflow[model.fs.unit.properties_out[0].pressure] == pytest.approx(
            1e-6, rel=1e-8
        )
        assert sfx_overflow[
            model.fs.unit.properties_out[0].temperature
        ] == pytest.approx(1e-1, rel=1e-8)

        # Check that unit model has scaling factors
        sfx_unit = model.fs.unit.scaling_factor
        assert isinstance(sfx_unit, Suffix)
        assert len(sfx_unit) == 34

    # TODO: Remove test once iscale is deprecated
    @pytest.mark.integration
    def test_example_case_iscale(self):
        m = ConcreteModel()

        m.fs = FlowsheetBlock(dynamic=False)
        m = ConcreteModel()

        m.fs = FlowsheetBlock(dynamic=False)

        m.fs.props_ASM1 = ASM1ParameterBlock()
        m.fs.props_ADM1 = ADM1ParameterBlock()
        m.fs.ADM1_rxn_props = ADM1ReactionParameterBlock(
            property_package=m.fs.props_ADM1
        )

        m.fs.unit = Translator_ASM1_ADM1(
            inlet_property_package=m.fs.props_ASM1,
            outlet_property_package=m.fs.props_ADM1,
            reaction_package=m.fs.ADM1_rxn_props,
            has_phase_equilibrium=False,
            outlet_state_defined=True,
        )

        m.fs.unit.inlet.flow_vol.fix(178.4674 * units.m**3 / units.day)
        m.fs.unit.inlet.temperature.fix(308.15 * units.K)
        m.fs.unit.inlet.pressure.fix(1 * units.atm)

        m.fs.unit.inlet.conc_mass_comp[0, "S_I"].fix(28.0665 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_S"].fix(48.9526 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_I"].fix(
            10361.7101 * units.mg / units.liter
        )
        m.fs.unit.inlet.conc_mass_comp[0, "X_S"].fix(
            20375.0176 * units.mg / units.liter
        )
        m.fs.unit.inlet.conc_mass_comp[0, "X_BH"].fix(
            10210.0698 * units.mg / units.liter
        )
        m.fs.unit.inlet.conc_mass_comp[0, "X_BA"].fix(553.2808 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_P"].fix(3204.6601 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_O"].fix(0.25225 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_NO"].fix(1.6871 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_NH"].fix(28.9098 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_ND"].fix(4.6834 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_ND"].fix(906.0933 * units.mg / units.liter)
        m.fs.unit.inlet.alkalinity.fix(7.1549 * units.mol / units.m**3)

        # Check condition number to confirm scaling
        iscale.calculate_scaling_factors(m)

        iscale.set_scaling_factor(m.fs.unit.properties_out[0].flow_vol, 1)
        iscale.set_scaling_factor(m.fs.unit.properties_out[0].conc_mass_comp["S_IC"], 1)
        iscale.set_scaling_factor(m.fs.unit.properties_out[0].conc_mass_comp["S_IN"], 1)
        iscale.set_scaling_factor(m.fs.unit.properties_out[0].conc_mass_comp["X_aa"], 1)
        iscale.set_scaling_factor(
            m.fs.unit.properties_out[0].conc_mass_comp["X_I"], 1e-1
        )

        sm = TransformationFactory("core.scale_model").create_using(m, rename=False)
        jac, _ = get_jacobian(sm, scaled=False)
        assert (jacobian_cond(jac=jac, scaled=False)) == pytest.approx(
            1.00056509896e11, rel=1e-3
        )

    @pytest.mark.integration
    def test_example_case_scaler(self):
        m = ConcreteModel()

        m.fs = FlowsheetBlock(dynamic=False)

        m.fs.props_ASM1 = ASM1ParameterBlock()
        m.fs.props_ADM1 = ADM1ParameterBlock()
        m.fs.ADM1_rxn_props = ADM1ReactionParameterBlock(
            property_package=m.fs.props_ADM1
        )

        m.fs.unit = Translator_ASM1_ADM1(
            inlet_property_package=m.fs.props_ASM1,
            outlet_property_package=m.fs.props_ADM1,
            reaction_package=m.fs.ADM1_rxn_props,
            has_phase_equilibrium=False,
            outlet_state_defined=True,
        )

        m.fs.unit.inlet.flow_vol.fix(178.4674 * units.m**3 / units.day)
        m.fs.unit.inlet.temperature.fix(308.15 * units.K)
        m.fs.unit.inlet.pressure.fix(1 * units.atm)

        m.fs.unit.inlet.conc_mass_comp[0, "S_I"].fix(28.0665 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_S"].fix(48.9526 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_I"].fix(
            10361.7101 * units.mg / units.liter
        )
        m.fs.unit.inlet.conc_mass_comp[0, "X_S"].fix(
            20375.0176 * units.mg / units.liter
        )
        m.fs.unit.inlet.conc_mass_comp[0, "X_BH"].fix(
            10210.0698 * units.mg / units.liter
        )
        m.fs.unit.inlet.conc_mass_comp[0, "X_BA"].fix(553.2808 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_P"].fix(3204.6601 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_O"].fix(0.25225 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_NO"].fix(1.6871 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_NH"].fix(28.9098 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_ND"].fix(4.6834 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_ND"].fix(906.0933 * units.mg / units.liter)
        m.fs.unit.inlet.alkalinity.fix(7.1549 * units.mol / units.m**3)

        scaler = ASM1ADM1Scaler()
        scaler.scale_model(
            m.fs.unit,
            submodel_scalers={
                m.fs.unit.properties_in: ASM1PropertiesScaler,
                m.fs.unit.properties_out: ADM1PropertiesScaler,
            },
        )

        sb = ScalerBase()
        sb.set_variable_scaling_factor(m.fs.unit.properties_out[0].flow_vol, 1)
        sb.set_variable_scaling_factor(
            m.fs.unit.properties_out[0].conc_mass_comp["S_aa"], 1e3
        )
        sb.set_variable_scaling_factor(
            m.fs.unit.properties_out[0].conc_mass_comp["S_h2"], 1e7
        )

        # Check condition number to confirm scaling
        sm = TransformationFactory("core.scale_model").create_using(m, rename=False)
        jac, _ = get_jacobian(sm, scaled=False)
        assert (jacobian_cond(jac=jac, scaled=False)) == pytest.approx(
            3.2518e4, rel=1e-3
        )
