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
Tests for Translator ADM1-ASM1 unit model.

Verified against approximated results from:

Nopens, I., Batstone, D, Copp, J, Jeppsson, U. Rosen, C., Volcke, E., Alex, J., 
and Vanrolleghem, P.
water Research Vol. 43 pp.1913–1923.
"""

import pytest
from pyomo.environ import (
    ConcreteModel,
    value,
    assert_optimal_termination,
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

from idaes.core.util.testing import initialization_tester
from idaes.core.util.scaling import (
    get_jacobian,
    jacobian_cond,
)

from watertap.unit_models.translators.translator_adm1_asm1 import (
    Translator_ADM1_ASM1,
    ADM1ASM1Scaler,
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

    m.fs.unit = Translator_ADM1_ASM1(
        inlet_property_package=m.fs.props_ADM1,
        outlet_property_package=m.fs.props_ASM1,
        reaction_package=m.fs.ADM1_rxn_props,
        has_phase_equilibrium=False,
        outlet_state_defined=True,
    )

    assert len(m.fs.unit.config) == 10

    assert m.fs.unit.config.outlet_state_defined == True
    assert not m.fs.unit.config.dynamic
    assert not m.fs.unit.config.has_holdup
    assert not m.fs.unit.config.has_phase_equilibrium
    assert m.fs.unit.config.inlet_property_package is m.fs.props_ADM1
    assert m.fs.unit.config.outlet_property_package is m.fs.props_ASM1
    assert m.fs.unit.config.reaction_package is m.fs.ADM1_rxn_props


# -----------------------------------------------------------------------------
class TestAdm1Asm1(object):
    @pytest.fixture(scope="class")
    def admasm(self):
        m = ConcreteModel()

        m.fs = FlowsheetBlock(dynamic=False)

        m.fs.props_ASM1 = ASM1ParameterBlock()
        m.fs.props_ADM1 = ADM1ParameterBlock()
        m.fs.ADM1_rxn_props = ADM1ReactionParameterBlock(
            property_package=m.fs.props_ADM1
        )

        m.fs.unit = Translator_ADM1_ASM1(
            inlet_property_package=m.fs.props_ADM1,
            outlet_property_package=m.fs.props_ASM1,
            reaction_package=m.fs.ADM1_rxn_props,
            has_phase_equilibrium=False,
            outlet_state_defined=True,
        )

        m.fs.unit.inlet.flow_vol.fix(170 * units.m**3 / units.day)
        m.fs.unit.inlet.temperature.fix(308.15 * units.K)
        m.fs.unit.inlet.pressure.fix(1 * units.atm)

        m.fs.unit.inlet.conc_mass_comp[0, "S_su"].fix(12.394 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_aa"].fix(5.54 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_fa"].fix(107.41 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_va"].fix(12.33 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_bu"].fix(14.00 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_pro"].fix(17.584 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_ac"].fix(89.315 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_h2"].fix(2.55e-4 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_ch4"].fix(55.49 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_IC"].fix(
            95.149 * units.mmol / units.liter * 12 * units.mg / units.mmol
        )
        m.fs.unit.inlet.conc_mass_comp[0, "S_IN"].fix(
            94.468 * units.mmol / units.liter * 14 * units.mg / units.mmol
        )
        m.fs.unit.inlet.conc_mass_comp[0, "S_I"].fix(130.87 * units.mg / units.liter)

        m.fs.unit.inlet.conc_mass_comp[0, "X_c"].fix(107.92 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_ch"].fix(20.517 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_pr"].fix(84.22 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_li"].fix(43.629 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_su"].fix(312.22 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_aa"].fix(931.67 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_fa"].fix(338.39 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_c4"].fix(335.77 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_pro"].fix(101.12 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_ac"].fix(677.24 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_h2"].fix(284.84 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_I"].fix(17216 * units.mg / units.liter)

        m.fs.unit.inlet.cations[0].fix(1e-8 * units.mmol / units.liter)
        m.fs.unit.inlet.anions[0].fix(5.21 * units.mmol / units.liter)

        return m

    @pytest.mark.build
    @pytest.mark.unit
    def test_build(self, admasm):

        assert hasattr(admasm.fs.unit, "inlet")
        assert len(admasm.fs.unit.inlet.vars) == 6
        assert hasattr(admasm.fs.unit.inlet, "flow_vol")
        assert hasattr(admasm.fs.unit.inlet, "conc_mass_comp")
        assert hasattr(admasm.fs.unit.inlet, "temperature")
        assert hasattr(admasm.fs.unit.inlet, "pressure")
        assert hasattr(admasm.fs.unit.inlet, "anions")
        assert hasattr(admasm.fs.unit.inlet, "cations")

        assert hasattr(admasm.fs.unit, "outlet")
        assert len(admasm.fs.unit.outlet.vars) == 5
        assert hasattr(admasm.fs.unit.outlet, "flow_vol")
        assert hasattr(admasm.fs.unit.outlet, "conc_mass_comp")
        assert hasattr(admasm.fs.unit.outlet, "temperature")
        assert hasattr(admasm.fs.unit.outlet, "pressure")
        assert hasattr(admasm.fs.unit.outlet, "alkalinity")

        assert number_variables(admasm) == 136
        assert number_total_constraints(admasm) == 16
        assert number_unused_variables(admasm.fs.unit) == 4

    @pytest.mark.component
    def test_units(self, admasm):
        assert_units_consistent(admasm)

    @pytest.mark.unit
    def test_dof(self, admasm):
        assert degrees_of_freedom(admasm) == 0

    @pytest.mark.solver
    @pytest.mark.skipif(solver is None, reason="Solver not available")
    @pytest.mark.component
    def test_initialize(self, admasm):
        initialization_tester(admasm)

    @pytest.mark.solver
    @pytest.mark.skipif(solver is None, reason="Solver not available")
    @pytest.mark.component
    def test_solve(self, admasm):
        solver = get_solver()
        results = solver.solve(admasm)
        assert_optimal_termination(results)

    @pytest.mark.solver
    @pytest.mark.skipif(solver is None, reason="Solver not available")
    @pytest.mark.component
    def test_solution(self, admasm):
        assert pytest.approx(101325.0, rel=1e-3) == value(
            admasm.fs.unit.outlet.pressure[0]
        )
        assert pytest.approx(308.15, rel=1e-3) == value(
            admasm.fs.unit.outlet.temperature[0]
        )
        assert pytest.approx(0.1308, rel=1e-3) == value(
            admasm.fs.unit.outlet.conc_mass_comp[0, "S_I"]
        )
        assert pytest.approx(0.2585, rel=1e-3) == value(
            admasm.fs.unit.outlet.conc_mass_comp[0, "S_S"]
        )
        assert pytest.approx(17.216, rel=1e-3) == value(
            admasm.fs.unit.outlet.conc_mass_comp[0, "X_I"]
        )
        assert pytest.approx(3.2375, rel=1e-3) == value(
            admasm.fs.unit.outlet.conc_mass_comp[0, "X_S"]
        )
        assert pytest.approx(1e-10, abs=1e-6) == value(
            admasm.fs.unit.outlet.conc_mass_comp[0, "X_BH"]
        )
        assert pytest.approx(1e-10, abs=1e-6) == value(
            admasm.fs.unit.outlet.conc_mass_comp[0, "X_BA"]
        )
        assert pytest.approx(1e-10, abs=1e-6) == value(
            admasm.fs.unit.outlet.conc_mass_comp[0, "X_P"]
        )
        assert pytest.approx(1e-10, abs=1e-6) == value(
            admasm.fs.unit.outlet.conc_mass_comp[0, "S_O"]
        )
        assert pytest.approx(1e-10, abs=1e-6) == value(
            admasm.fs.unit.outlet.conc_mass_comp[0, "S_NO"]
        )
        assert pytest.approx(1.322, rel=1e-3) == value(
            admasm.fs.unit.outlet.conc_mass_comp[0, "S_NH"]
        )
        assert pytest.approx(0.00839, rel=1e-3) == value(
            admasm.fs.unit.outlet.conc_mass_comp[0, "S_ND"]
        )
        assert pytest.approx(0.251, rel=1e-3) == value(
            admasm.fs.unit.outlet.conc_mass_comp[0, "X_ND"]
        )
        assert pytest.approx(0.095061, rel=1e-3) == value(
            admasm.fs.unit.outlet.alkalinity[0]
        )

    @pytest.mark.solver
    @pytest.mark.skipif(solver is None, reason="Solver not available")
    @pytest.mark.component
    def test_conservation(self, admasm):
        assert (
            abs(
                value(
                    admasm.fs.unit.inlet.flow_vol[0] * admasm.fs.props_ADM1.dens_mass
                    - admasm.fs.unit.outlet.flow_vol[0] * admasm.fs.props_ASM1.dens_mass
                )
            )
            <= 1e-6
        )

        assert (
            abs(
                value(
                    (
                        admasm.fs.unit.inlet.flow_vol[0]
                        * admasm.fs.props_ADM1.dens_mass
                        * admasm.fs.props_ADM1.cp_mass
                        * (
                            admasm.fs.unit.inlet.temperature[0]
                            - admasm.fs.props_ADM1.temperature_ref
                        )
                    )
                    - (
                        admasm.fs.unit.outlet.flow_vol[0]
                        * admasm.fs.props_ASM1.dens_mass
                        * admasm.fs.props_ASM1.cp_mass
                        * (
                            admasm.fs.unit.outlet.temperature[0]
                            - admasm.fs.props_ASM1.temperature_ref
                        )
                    )
                )
            )
            <= 1e-6
        )


class TestADM1ASM1Scaler:
    @pytest.fixture
    def model(self):
        m = ConcreteModel()

        m.fs = FlowsheetBlock(dynamic=False)

        m.fs.props_ASM1 = ASM1ParameterBlock()
        m.fs.props_ADM1 = ADM1ParameterBlock()
        m.fs.ADM1_rxn_props = ADM1ReactionParameterBlock(
            property_package=m.fs.props_ADM1
        )

        m.fs.unit = Translator_ADM1_ASM1(
            inlet_property_package=m.fs.props_ADM1,
            outlet_property_package=m.fs.props_ASM1,
            reaction_package=m.fs.ADM1_rxn_props,
            has_phase_equilibrium=False,
            outlet_state_defined=True,
        )

        m.fs.unit.inlet.flow_vol.fix(170 * units.m**3 / units.day)
        m.fs.unit.inlet.temperature.fix(308.15 * units.K)
        m.fs.unit.inlet.pressure.fix(1 * units.atm)

        m.fs.unit.inlet.conc_mass_comp[0, "S_su"].fix(12.394 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_aa"].fix(5.54 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_fa"].fix(107.41 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_va"].fix(12.33 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_bu"].fix(14.00 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_pro"].fix(17.584 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_ac"].fix(89.315 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_h2"].fix(2.55e-4 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_ch4"].fix(55.49 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_IC"].fix(
            95.149 * units.mmol / units.liter * 12 * units.mg / units.mmol
        )
        m.fs.unit.inlet.conc_mass_comp[0, "S_IN"].fix(
            94.468 * units.mmol / units.liter * 14 * units.mg / units.mmol
        )
        m.fs.unit.inlet.conc_mass_comp[0, "S_I"].fix(130.87 * units.mg / units.liter)

        m.fs.unit.inlet.conc_mass_comp[0, "X_c"].fix(107.92 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_ch"].fix(20.517 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_pr"].fix(84.22 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_li"].fix(43.629 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_su"].fix(312.22 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_aa"].fix(931.67 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_fa"].fix(338.39 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_c4"].fix(335.77 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_pro"].fix(101.12 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_ac"].fix(677.24 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_h2"].fix(284.84 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_I"].fix(17216 * units.mg / units.liter)

        m.fs.unit.inlet.cations[0].fix(1e-8 * units.mmol / units.liter)
        m.fs.unit.inlet.anions[0].fix(5.21 * units.mmol / units.liter)

        return m

    @pytest.mark.component
    def test_variable_scaling_routine(self, model):
        scaler = model.fs.unit.default_scaler()

        assert isinstance(scaler, ADM1ASM1Scaler)

        scaler.variable_scaling_routine(model.fs.unit)

        # Inlet state
        sfx_in = model.fs.unit.properties_in[0].scaling_factor
        assert isinstance(sfx_in, Suffix)
        assert len(sfx_in) == 3
        assert sfx_in[model.fs.unit.properties_in[0].flow_vol] == pytest.approx(
            1e5, rel=1e-8
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

        assert isinstance(scaler, ADM1ASM1Scaler)

        scaler.constraint_scaling_routine(model.fs.unit)

        sfx_unit = model.fs.unit.scaling_factor
        assert isinstance(sfx_unit, Suffix)
        assert len(sfx_unit) == 16

    @pytest.mark.component
    def test_scale_model(self, model):
        scaler = model.fs.unit.default_scaler()

        assert isinstance(scaler, ADM1ASM1Scaler)

        scaler.scale_model(model.fs.unit)

        # Inlet state
        sfx_in = model.fs.unit.properties_in[0].scaling_factor
        assert isinstance(sfx_in, Suffix)
        assert len(sfx_in) == 3
        assert sfx_in[model.fs.unit.properties_in[0].flow_vol] == pytest.approx(
            1e5, rel=1e-8
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
        assert len(sfx_unit) == 16

    # TODO: Remove test once iscale is deprecated
    @pytest.mark.integration
    def test_example_case_iscale(self):
        m = ConcreteModel()

        m.fs = FlowsheetBlock(dynamic=False)

        m.fs.props_ASM1 = ASM1ParameterBlock()
        m.fs.props_ADM1 = ADM1ParameterBlock()
        m.fs.ADM1_rxn_props = ADM1ReactionParameterBlock(
            property_package=m.fs.props_ADM1
        )

        m.fs.unit = Translator_ADM1_ASM1(
            inlet_property_package=m.fs.props_ADM1,
            outlet_property_package=m.fs.props_ASM1,
            reaction_package=m.fs.ADM1_rxn_props,
            has_phase_equilibrium=False,
            outlet_state_defined=True,
        )

        m.fs.unit.inlet.flow_vol.fix(170 * units.m**3 / units.day)
        m.fs.unit.inlet.temperature.fix(308.15 * units.K)
        m.fs.unit.inlet.pressure.fix(1 * units.atm)

        m.fs.unit.inlet.conc_mass_comp[0, "S_su"].fix(12.394 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_aa"].fix(5.54 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_fa"].fix(107.41 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_va"].fix(12.33 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_bu"].fix(14.00 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_pro"].fix(17.584 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_ac"].fix(89.315 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_h2"].fix(2.55e-4 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_ch4"].fix(55.49 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_IC"].fix(
            95.149 * units.mmol / units.liter * 12 * units.mg / units.mmol
        )
        m.fs.unit.inlet.conc_mass_comp[0, "S_IN"].fix(
            94.468 * units.mmol / units.liter * 14 * units.mg / units.mmol
        )
        m.fs.unit.inlet.conc_mass_comp[0, "S_I"].fix(130.87 * units.mg / units.liter)

        m.fs.unit.inlet.conc_mass_comp[0, "X_c"].fix(107.92 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_ch"].fix(20.517 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_pr"].fix(84.22 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_li"].fix(43.629 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_su"].fix(312.22 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_aa"].fix(931.67 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_fa"].fix(338.39 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_c4"].fix(335.77 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_pro"].fix(101.12 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_ac"].fix(677.24 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_h2"].fix(284.84 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_I"].fix(17216 * units.mg / units.liter)

        m.fs.unit.inlet.cations[0].fix(1e-8 * units.mmol / units.liter)
        m.fs.unit.inlet.anions[0].fix(5.21 * units.mmol / units.liter)

        # Check condition number to confirm scaling
        iscale.calculate_scaling_factors(m)

        sm = TransformationFactory("core.scale_model").create_using(m, rename=False)
        jac, _ = get_jacobian(sm, scaled=False)
        assert (jacobian_cond(jac=jac, scaled=False)) == pytest.approx(
            3.872983349e5, rel=1e-3
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

        m.fs.unit = Translator_ADM1_ASM1(
            inlet_property_package=m.fs.props_ADM1,
            outlet_property_package=m.fs.props_ASM1,
            reaction_package=m.fs.ADM1_rxn_props,
            has_phase_equilibrium=False,
            outlet_state_defined=True,
        )

        m.fs.unit.inlet.flow_vol.fix(170 * units.m**3 / units.day)
        m.fs.unit.inlet.temperature.fix(308.15 * units.K)
        m.fs.unit.inlet.pressure.fix(1 * units.atm)

        m.fs.unit.inlet.conc_mass_comp[0, "S_su"].fix(12.394 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_aa"].fix(5.54 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_fa"].fix(107.41 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_va"].fix(12.33 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_bu"].fix(14.00 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_pro"].fix(17.584 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_ac"].fix(89.315 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_h2"].fix(2.55e-4 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_ch4"].fix(55.49 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "S_IC"].fix(
            95.149 * units.mmol / units.liter * 12 * units.mg / units.mmol
        )
        m.fs.unit.inlet.conc_mass_comp[0, "S_IN"].fix(
            94.468 * units.mmol / units.liter * 14 * units.mg / units.mmol
        )
        m.fs.unit.inlet.conc_mass_comp[0, "S_I"].fix(130.87 * units.mg / units.liter)

        m.fs.unit.inlet.conc_mass_comp[0, "X_c"].fix(107.92 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_ch"].fix(20.517 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_pr"].fix(84.22 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_li"].fix(43.629 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_su"].fix(312.22 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_aa"].fix(931.67 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_fa"].fix(338.39 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_c4"].fix(335.77 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_pro"].fix(101.12 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_ac"].fix(677.24 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_h2"].fix(284.84 * units.mg / units.liter)
        m.fs.unit.inlet.conc_mass_comp[0, "X_I"].fix(17216 * units.mg / units.liter)

        m.fs.unit.inlet.cations[0].fix(1e-8 * units.mmol / units.liter)
        m.fs.unit.inlet.anions[0].fix(5.21 * units.mmol / units.liter)

        scaler = ADM1ASM1Scaler()
        scaler.scale_model(
            m.fs.unit,
            submodel_scalers={
                m.fs.unit.properties_in: ADM1PropertiesScaler,
                m.fs.unit.properties_out: ASM1PropertiesScaler,
            },
        )

        # Check condition number to confirm scaling
        sm = TransformationFactory("core.scale_model").create_using(m, rename=False)
        jac, _ = get_jacobian(sm, scaled=False)
        assert (jacobian_cond(jac=jac, scaled=False)) == pytest.approx(
            5.4739117e3, rel=1e-3
        )
