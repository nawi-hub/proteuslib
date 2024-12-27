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
Tests for Translator ADM1-ASM2D unit model.
Verified against approximated results from:
Flores-Alsina, X., Solon, K., Mbamba, C.K., Tait, S., Gernaey, K.V., Jeppsson, U. and Batstone, D.J., 2016.
Modelling phosphorus (P), sulfur (S) and iron (Fe) interactions for dynamic simulations of anaerobic digestion processes.
Water Research, 95, pp.370-382.
"""

import pytest
from pyomo.environ import (
    ConcreteModel,
    value,
    assert_optimal_termination,
    Suffix,
    TransformationFactory,
)

from idaes.core import (
    FlowsheetBlock,
)

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
from idaes.core.util.scaling import (
    get_jacobian,
    jacobian_cond,
)
import idaes.core.util.scaling as iscale

from idaes.core.util.testing import initialization_tester

from watertap.unit_models.translators.translator_adm1_asm2d import (
    Translator_ADM1_ASM2D,
    ADM1ASM2dScaler,
)
from watertap.property_models.unit_specific.anaerobic_digestion.modified_adm1_properties import (
    ModifiedADM1ParameterBlock,
    ModifiedADM1PropertiesScaler,
)

from watertap.property_models.unit_specific.activated_sludge.modified_asm2d_properties import (
    ModifiedASM2dParameterBlock,
    ModifiedASM2dPropertiesScaler,
)

from watertap.property_models.unit_specific.anaerobic_digestion.modified_adm1_reactions import (
    ModifiedADM1ReactionParameterBlock,
)

from watertap.property_models.unit_specific.activated_sludge.modified_asm2d_reactions import (
    ModifiedASM2dReactionParameterBlock,
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

    m.fs.props_ASM2D = ModifiedASM2dParameterBlock()
    m.fs.ASM2d_rxn_props = ModifiedASM2dReactionParameterBlock(
        property_package=m.fs.props_ASM2D
    )
    m.fs.props_ADM1 = ModifiedADM1ParameterBlock()
    m.fs.ADM1_rxn_props = ModifiedADM1ReactionParameterBlock(
        property_package=m.fs.props_ADM1
    )

    m.fs.unit = Translator_ADM1_ASM2D(
        inlet_property_package=m.fs.props_ADM1,
        outlet_property_package=m.fs.props_ASM2D,
        inlet_reaction_package=m.fs.ADM1_rxn_props,
        outlet_reaction_package=m.fs.ASM2d_rxn_props,
        has_phase_equilibrium=False,
        outlet_state_defined=True,
    )

    assert len(m.fs.unit.config) == 12

    assert m.fs.unit.config.outlet_state_defined == True
    assert not m.fs.unit.config.dynamic
    assert not m.fs.unit.config.has_holdup
    assert not m.fs.unit.config.has_phase_equilibrium
    assert m.fs.unit.config.inlet_property_package is m.fs.props_ADM1
    assert m.fs.unit.config.outlet_property_package is m.fs.props_ASM2D
    assert m.fs.unit.config.inlet_reaction_package is m.fs.ADM1_rxn_props
    assert m.fs.unit.config.outlet_reaction_package is m.fs.ASM2d_rxn_props


# -----------------------------------------------------------------------------
class TestAdm1Asm2d(object):
    @pytest.fixture(scope="class")
    def asmadm(self):
        m = ConcreteModel()

        m.fs = FlowsheetBlock(dynamic=False)

        m.fs.props_ASM2D = ModifiedASM2dParameterBlock()
        m.fs.ASM2d_rxn_props = ModifiedASM2dReactionParameterBlock(
            property_package=m.fs.props_ASM2D
        )
        m.fs.props_ADM1 = ModifiedADM1ParameterBlock()
        m.fs.ADM1_rxn_props = ModifiedADM1ReactionParameterBlock(
            property_package=m.fs.props_ADM1
        )

        m.fs.unit = Translator_ADM1_ASM2D(
            inlet_property_package=m.fs.props_ADM1,
            outlet_property_package=m.fs.props_ASM2D,
            inlet_reaction_package=m.fs.ADM1_rxn_props,
            outlet_reaction_package=m.fs.ASM2d_rxn_props,
            has_phase_equilibrium=False,
            outlet_state_defined=True,
        )

        m.fs.unit.inlet.flow_vol.fix(170 * units.m**3 / units.day)
        m.fs.unit.inlet.temperature.fix(308.15 * units.K)
        m.fs.unit.inlet.pressure.fix(1 * units.atm)

        m.fs.unit.inlet.conc_mass_comp[0, "S_su"].fix(0.034597)
        m.fs.unit.inlet.conc_mass_comp[0, "S_aa"].fix(0.015037)
        m.fs.unit.inlet.conc_mass_comp[0, "S_fa"].fix(0)
        m.fs.unit.inlet.conc_mass_comp[0, "S_va"].fix(0)
        m.fs.unit.inlet.conc_mass_comp[0, "S_bu"].fix(0)
        m.fs.unit.inlet.conc_mass_comp[0, "S_pro"].fix(0)
        m.fs.unit.inlet.conc_mass_comp[0, "S_ac"].fix(0.025072)
        m.fs.unit.inlet.conc_mass_comp[0, "S_h2"].fix(0)
        m.fs.unit.inlet.conc_mass_comp[0, "S_ch4"].fix(0)
        m.fs.unit.inlet.conc_mass_comp[0, "S_IC"].fix(0.34628)
        m.fs.unit.inlet.conc_mass_comp[0, "S_IN"].fix(0.60014)
        m.fs.unit.inlet.conc_mass_comp[0, "S_IP"].fix(0.22677)
        m.fs.unit.inlet.conc_mass_comp[0, "S_I"].fix(0.026599)

        m.fs.unit.inlet.conc_mass_comp[0, "X_ch"].fix(7.3687)
        m.fs.unit.inlet.conc_mass_comp[0, "X_pr"].fix(7.7308)
        m.fs.unit.inlet.conc_mass_comp[0, "X_li"].fix(10.3288)
        m.fs.unit.inlet.conc_mass_comp[0, "X_su"].fix(0)
        m.fs.unit.inlet.conc_mass_comp[0, "X_aa"].fix(0)
        m.fs.unit.inlet.conc_mass_comp[0, "X_fa"].fix(0)
        m.fs.unit.inlet.conc_mass_comp[0, "X_c4"].fix(0)
        m.fs.unit.inlet.conc_mass_comp[0, "X_pro"].fix(0)
        m.fs.unit.inlet.conc_mass_comp[0, "X_ac"].fix(0)
        m.fs.unit.inlet.conc_mass_comp[0, "X_h2"].fix(0)
        m.fs.unit.inlet.conc_mass_comp[0, "X_I"].fix(12.7727)
        m.fs.unit.inlet.conc_mass_comp[0, "X_PHA"].fix(0.0022493)
        m.fs.unit.inlet.conc_mass_comp[0, "X_PP"].fix(1.04110)
        m.fs.unit.inlet.conc_mass_comp[0, "X_PAO"].fix(3.4655)
        m.fs.unit.inlet.conc_mass_comp[0, "S_K"].fix(0.02268)
        m.fs.unit.inlet.conc_mass_comp[0, "S_Mg"].fix(0.02893)

        m.fs.unit.inlet.cations[0].fix(0.04)
        m.fs.unit.inlet.anions[0].fix(0.02)

        return m

    @pytest.mark.build
    @pytest.mark.unit
    def test_build(self, asmadm):

        assert hasattr(asmadm.fs.unit, "inlet")
        assert len(asmadm.fs.unit.inlet.vars) == 6
        assert hasattr(asmadm.fs.unit.inlet, "flow_vol")
        assert hasattr(asmadm.fs.unit.inlet, "conc_mass_comp")
        assert hasattr(asmadm.fs.unit.inlet, "temperature")
        assert hasattr(asmadm.fs.unit.inlet, "pressure")
        assert hasattr(asmadm.fs.unit.inlet, "anions")
        assert hasattr(asmadm.fs.unit.inlet, "cations")

        assert hasattr(asmadm.fs.unit, "outlet")
        assert len(asmadm.fs.unit.outlet.vars) == 4
        assert hasattr(asmadm.fs.unit.outlet, "flow_vol")
        assert hasattr(asmadm.fs.unit.outlet, "conc_mass_comp")
        assert hasattr(asmadm.fs.unit.outlet, "temperature")
        assert hasattr(asmadm.fs.unit.outlet, "pressure")

        assert number_variables(asmadm) == 278
        assert number_total_constraints(asmadm) == 21

        assert number_unused_variables(asmadm.fs.unit) == 4

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
        solver = get_solver()
        results = solver.solve(asmadm)
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
        assert pytest.approx(0.0273213, rel=1e-3) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "S_A"]
        )
        assert pytest.approx(0.049635, rel=1e-3) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "S_F"]
        )
        assert pytest.approx(0.026599, rel=1e-3) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "S_I"]
        )
        assert pytest.approx(0.776867, rel=1e-3) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "S_NH4"]
        )
        assert pytest.approx(1e-10, abs=1e-6) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "S_N2"]
        )
        assert pytest.approx(1e-10, abs=1e-6) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "S_NO3"]
        )
        assert pytest.approx(1e-10, abs=1e-6) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "S_O2"]
        )
        assert pytest.approx(0.35699, rel=1e-3) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "S_PO4"]
        )
        assert pytest.approx(1e-10, abs=1e-6) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "X_AUT"]
        )
        assert pytest.approx(1e-10, abs=1e-6) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "X_H"]
        )
        assert pytest.approx(13.11925, rel=1e-3) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "X_I"]
        )
        assert pytest.approx(1e-10, abs=1e-6) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "X_PAO"]
        )
        assert pytest.approx(1e-8, rel=1e-3) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "X_PHA"]
        )
        assert pytest.approx(1e-8, rel=1e-3) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "X_PP"]
        )
        assert pytest.approx(28.54725, rel=1e-3) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "X_S"]
        )
        assert pytest.approx(0.02268, rel=1e-3) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "S_K"]
        )
        assert pytest.approx(0.02893, rel=1e-3) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "S_Mg"]
        )
        assert pytest.approx(0.734657, rel=1e-3) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "S_IC"]
        )

    @pytest.mark.solver
    @pytest.mark.skipif(solver is None, reason="Solver not available")
    @pytest.mark.component
    def test_conservation(self, asmadm):
        assert (
            abs(
                value(
                    asmadm.fs.unit.inlet.flow_vol[0] * asmadm.fs.props_ADM1.dens_mass
                    - asmadm.fs.unit.outlet.flow_vol[0]
                    * asmadm.fs.props_ASM2D.dens_mass
                )
            )
            <= 1e-5
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
                        * asmadm.fs.props_ASM2D.dens_mass
                        * asmadm.fs.props_ASM2D.cp_mass
                        * (
                            asmadm.fs.unit.outlet.temperature[0]
                            - asmadm.fs.props_ASM2D.temperature_ref
                        )
                    )
                )
            )
            <= 1e-6
        )


class TestADM1ASM2dScaler:
    @pytest.fixture
    def model(self):
        m = ConcreteModel()

        m.fs = FlowsheetBlock(dynamic=False)

        m.fs.props_ASM2D = ModifiedASM2dParameterBlock()
        m.fs.ASM2d_rxn_props = ModifiedASM2dReactionParameterBlock(
            property_package=m.fs.props_ASM2D
        )
        m.fs.props_ADM1 = ModifiedADM1ParameterBlock()
        m.fs.ADM1_rxn_props = ModifiedADM1ReactionParameterBlock(
            property_package=m.fs.props_ADM1
        )

        m.fs.unit = Translator_ADM1_ASM2D(
            inlet_property_package=m.fs.props_ADM1,
            outlet_property_package=m.fs.props_ASM2D,
            inlet_reaction_package=m.fs.ADM1_rxn_props,
            outlet_reaction_package=m.fs.ASM2d_rxn_props,
            has_phase_equilibrium=False,
            outlet_state_defined=True,
        )

        m.fs.unit.inlet.flow_vol.fix(170 * units.m**3 / units.day)
        m.fs.unit.inlet.temperature.fix(308.15 * units.K)
        m.fs.unit.inlet.pressure.fix(1 * units.atm)

        m.fs.unit.inlet.conc_mass_comp[0, "S_su"].fix(0.034597)
        m.fs.unit.inlet.conc_mass_comp[0, "S_aa"].fix(0.015037)
        m.fs.unit.inlet.conc_mass_comp[0, "S_fa"].fix(0)
        m.fs.unit.inlet.conc_mass_comp[0, "S_va"].fix(0)
        m.fs.unit.inlet.conc_mass_comp[0, "S_bu"].fix(0)
        m.fs.unit.inlet.conc_mass_comp[0, "S_pro"].fix(0)
        m.fs.unit.inlet.conc_mass_comp[0, "S_ac"].fix(0.025072)
        m.fs.unit.inlet.conc_mass_comp[0, "S_h2"].fix(0)
        m.fs.unit.inlet.conc_mass_comp[0, "S_ch4"].fix(0)
        m.fs.unit.inlet.conc_mass_comp[0, "S_IC"].fix(0.34628)
        m.fs.unit.inlet.conc_mass_comp[0, "S_IN"].fix(0.60014)
        m.fs.unit.inlet.conc_mass_comp[0, "S_IP"].fix(0.22677)
        m.fs.unit.inlet.conc_mass_comp[0, "S_I"].fix(0.026599)

        m.fs.unit.inlet.conc_mass_comp[0, "X_ch"].fix(7.3687)
        m.fs.unit.inlet.conc_mass_comp[0, "X_pr"].fix(7.7308)
        m.fs.unit.inlet.conc_mass_comp[0, "X_li"].fix(10.3288)
        m.fs.unit.inlet.conc_mass_comp[0, "X_su"].fix(0)
        m.fs.unit.inlet.conc_mass_comp[0, "X_aa"].fix(0)
        m.fs.unit.inlet.conc_mass_comp[0, "X_fa"].fix(0)
        m.fs.unit.inlet.conc_mass_comp[0, "X_c4"].fix(0)
        m.fs.unit.inlet.conc_mass_comp[0, "X_pro"].fix(0)
        m.fs.unit.inlet.conc_mass_comp[0, "X_ac"].fix(0)
        m.fs.unit.inlet.conc_mass_comp[0, "X_h2"].fix(0)
        m.fs.unit.inlet.conc_mass_comp[0, "X_I"].fix(12.7727)
        m.fs.unit.inlet.conc_mass_comp[0, "X_PHA"].fix(0.0022493)
        m.fs.unit.inlet.conc_mass_comp[0, "X_PP"].fix(1.04110)
        m.fs.unit.inlet.conc_mass_comp[0, "X_PAO"].fix(3.4655)
        m.fs.unit.inlet.conc_mass_comp[0, "S_K"].fix(0.02268)
        m.fs.unit.inlet.conc_mass_comp[0, "S_Mg"].fix(0.02893)

        m.fs.unit.inlet.cations[0].fix(0.04)
        m.fs.unit.inlet.anions[0].fix(0.02)

        return m

    @pytest.mark.component
    def test_variable_scaling_routine(self, model):
        scaler = model.fs.unit.default_scaler()

        assert isinstance(scaler, ADM1ASM2dScaler)

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
        sfx_out = model.fs.unit.properties_out[0].scaling_factor
        assert isinstance(sfx_out, Suffix)
        assert len(sfx_out) == 3
        assert sfx_out[model.fs.unit.properties_out[0].flow_vol] == pytest.approx(
            1e5, rel=1e-8
        )
        assert sfx_out[model.fs.unit.properties_out[0].pressure] == pytest.approx(
            1e-5, rel=1e-8
        )
        assert sfx_out[model.fs.unit.properties_out[0].temperature] == pytest.approx(
            1e-2, rel=1e-8
        )

    @pytest.mark.component
    def test_constraint_scaling_routine(self, model):
        scaler = model.fs.unit.default_scaler()

        assert isinstance(scaler, ADM1ASM2dScaler)

        scaler.constraint_scaling_routine(model.fs.unit)

        sfx_out = model.fs.unit.properties_out[0].scaling_factor
        assert isinstance(sfx_out, Suffix)
        assert len(sfx_out) == 1

    @pytest.mark.component
    def test_scale_model(self, model):
        scaler = model.fs.unit.default_scaler()

        assert isinstance(scaler, ADM1ASM2dScaler)

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
        sfx_out = model.fs.unit.properties_out[0].scaling_factor
        assert isinstance(sfx_out, Suffix)
        assert len(sfx_out) == 3
        assert sfx_out[model.fs.unit.properties_out[0].flow_vol] == pytest.approx(
            1e5, rel=1e-8
        )
        assert sfx_out[model.fs.unit.properties_out[0].pressure] == pytest.approx(
            1e-5, rel=1e-8
        )
        assert sfx_out[model.fs.unit.properties_out[0].temperature] == pytest.approx(
            1e-2, rel=1e-8
        )

    # TODO: Remove test once iscale is deprecated
    @pytest.mark.integration
    def test_example_case_iscale(self):
        m = ConcreteModel()

        m.fs = FlowsheetBlock(dynamic=False)

        m.fs.props_ASM2D = ModifiedASM2dParameterBlock()
        m.fs.ASM2d_rxn_props = ModifiedASM2dReactionParameterBlock(
            property_package=m.fs.props_ASM2D
        )
        m.fs.props_ADM1 = ModifiedADM1ParameterBlock()
        m.fs.ADM1_rxn_props = ModifiedADM1ReactionParameterBlock(
            property_package=m.fs.props_ADM1
        )

        m.fs.unit = Translator_ADM1_ASM2D(
            inlet_property_package=m.fs.props_ADM1,
            outlet_property_package=m.fs.props_ASM2D,
            inlet_reaction_package=m.fs.ADM1_rxn_props,
            outlet_reaction_package=m.fs.ASM2d_rxn_props,
            has_phase_equilibrium=False,
            outlet_state_defined=True,
        )

        m.fs.unit.inlet.flow_vol.fix(170 * units.m**3 / units.day)
        m.fs.unit.inlet.temperature.fix(308.15 * units.K)
        m.fs.unit.inlet.pressure.fix(1 * units.atm)

        m.fs.unit.inlet.conc_mass_comp[0, "S_su"].fix(0.034597)
        m.fs.unit.inlet.conc_mass_comp[0, "S_aa"].fix(0.015037)
        m.fs.unit.inlet.conc_mass_comp[0, "S_fa"].fix(0)
        m.fs.unit.inlet.conc_mass_comp[0, "S_va"].fix(0)
        m.fs.unit.inlet.conc_mass_comp[0, "S_bu"].fix(0)
        m.fs.unit.inlet.conc_mass_comp[0, "S_pro"].fix(0)
        m.fs.unit.inlet.conc_mass_comp[0, "S_ac"].fix(0.025072)
        m.fs.unit.inlet.conc_mass_comp[0, "S_h2"].fix(0)
        m.fs.unit.inlet.conc_mass_comp[0, "S_ch4"].fix(0)
        m.fs.unit.inlet.conc_mass_comp[0, "S_IC"].fix(0.34628)
        m.fs.unit.inlet.conc_mass_comp[0, "S_IN"].fix(0.60014)
        m.fs.unit.inlet.conc_mass_comp[0, "S_IP"].fix(0.22677)
        m.fs.unit.inlet.conc_mass_comp[0, "S_I"].fix(0.026599)

        m.fs.unit.inlet.conc_mass_comp[0, "X_ch"].fix(7.3687)
        m.fs.unit.inlet.conc_mass_comp[0, "X_pr"].fix(7.7308)
        m.fs.unit.inlet.conc_mass_comp[0, "X_li"].fix(10.3288)
        m.fs.unit.inlet.conc_mass_comp[0, "X_su"].fix(0)
        m.fs.unit.inlet.conc_mass_comp[0, "X_aa"].fix(0)
        m.fs.unit.inlet.conc_mass_comp[0, "X_fa"].fix(0)
        m.fs.unit.inlet.conc_mass_comp[0, "X_c4"].fix(0)
        m.fs.unit.inlet.conc_mass_comp[0, "X_pro"].fix(0)
        m.fs.unit.inlet.conc_mass_comp[0, "X_ac"].fix(0)
        m.fs.unit.inlet.conc_mass_comp[0, "X_h2"].fix(0)
        m.fs.unit.inlet.conc_mass_comp[0, "X_I"].fix(12.7727)
        m.fs.unit.inlet.conc_mass_comp[0, "X_PHA"].fix(0.0022493)
        m.fs.unit.inlet.conc_mass_comp[0, "X_PP"].fix(1.04110)
        m.fs.unit.inlet.conc_mass_comp[0, "X_PAO"].fix(3.4655)
        m.fs.unit.inlet.conc_mass_comp[0, "S_K"].fix(0.02268)
        m.fs.unit.inlet.conc_mass_comp[0, "S_Mg"].fix(0.02893)

        m.fs.unit.inlet.cations[0].fix(0.04)
        m.fs.unit.inlet.anions[0].fix(0.02)

        iscale.calculate_scaling_factors(m.fs.unit)

        # Check condition number to confirm scaling
        sm = TransformationFactory("core.scale_model").create_using(m, rename=False)
        jac, _ = get_jacobian(sm, scaled=False)
        assert (jacobian_cond(jac=jac, scaled=False)) == pytest.approx(
            4.47213596e5, rel=1e-3
        )

    @pytest.mark.integration
    def test_example_case_scaler(self):
        m = ConcreteModel()

        m.fs = FlowsheetBlock(dynamic=False)

        m.fs.props_ASM2D = ModifiedASM2dParameterBlock()
        m.fs.ASM2d_rxn_props = ModifiedASM2dReactionParameterBlock(
            property_package=m.fs.props_ASM2D
        )
        m.fs.props_ADM1 = ModifiedADM1ParameterBlock()
        m.fs.ADM1_rxn_props = ModifiedADM1ReactionParameterBlock(
            property_package=m.fs.props_ADM1
        )

        m.fs.unit = Translator_ADM1_ASM2D(
            inlet_property_package=m.fs.props_ADM1,
            outlet_property_package=m.fs.props_ASM2D,
            inlet_reaction_package=m.fs.ADM1_rxn_props,
            outlet_reaction_package=m.fs.ASM2d_rxn_props,
            has_phase_equilibrium=False,
            outlet_state_defined=True,
        )

        m.fs.unit.inlet.flow_vol.fix(170 * units.m**3 / units.day)
        m.fs.unit.inlet.temperature.fix(308.15 * units.K)
        m.fs.unit.inlet.pressure.fix(1 * units.atm)

        m.fs.unit.inlet.conc_mass_comp[0, "S_su"].fix(0.034597)
        m.fs.unit.inlet.conc_mass_comp[0, "S_aa"].fix(0.015037)
        m.fs.unit.inlet.conc_mass_comp[0, "S_fa"].fix(0)
        m.fs.unit.inlet.conc_mass_comp[0, "S_va"].fix(0)
        m.fs.unit.inlet.conc_mass_comp[0, "S_bu"].fix(0)
        m.fs.unit.inlet.conc_mass_comp[0, "S_pro"].fix(0)
        m.fs.unit.inlet.conc_mass_comp[0, "S_ac"].fix(0.025072)
        m.fs.unit.inlet.conc_mass_comp[0, "S_h2"].fix(0)
        m.fs.unit.inlet.conc_mass_comp[0, "S_ch4"].fix(0)
        m.fs.unit.inlet.conc_mass_comp[0, "S_IC"].fix(0.34628)
        m.fs.unit.inlet.conc_mass_comp[0, "S_IN"].fix(0.60014)
        m.fs.unit.inlet.conc_mass_comp[0, "S_IP"].fix(0.22677)
        m.fs.unit.inlet.conc_mass_comp[0, "S_I"].fix(0.026599)

        m.fs.unit.inlet.conc_mass_comp[0, "X_ch"].fix(7.3687)
        m.fs.unit.inlet.conc_mass_comp[0, "X_pr"].fix(7.7308)
        m.fs.unit.inlet.conc_mass_comp[0, "X_li"].fix(10.3288)
        m.fs.unit.inlet.conc_mass_comp[0, "X_su"].fix(0)
        m.fs.unit.inlet.conc_mass_comp[0, "X_aa"].fix(0)
        m.fs.unit.inlet.conc_mass_comp[0, "X_fa"].fix(0)
        m.fs.unit.inlet.conc_mass_comp[0, "X_c4"].fix(0)
        m.fs.unit.inlet.conc_mass_comp[0, "X_pro"].fix(0)
        m.fs.unit.inlet.conc_mass_comp[0, "X_ac"].fix(0)
        m.fs.unit.inlet.conc_mass_comp[0, "X_h2"].fix(0)
        m.fs.unit.inlet.conc_mass_comp[0, "X_I"].fix(12.7727)
        m.fs.unit.inlet.conc_mass_comp[0, "X_PHA"].fix(0.0022493)
        m.fs.unit.inlet.conc_mass_comp[0, "X_PP"].fix(1.04110)
        m.fs.unit.inlet.conc_mass_comp[0, "X_PAO"].fix(3.4655)
        m.fs.unit.inlet.conc_mass_comp[0, "S_K"].fix(0.02268)
        m.fs.unit.inlet.conc_mass_comp[0, "S_Mg"].fix(0.02893)

        m.fs.unit.inlet.cations[0].fix(0.04)
        m.fs.unit.inlet.anions[0].fix(0.02)

        scaler = ADM1ASM2dScaler()
        scaler.scale_model(
            m.fs.unit,
            submodel_scalers={
                m.fs.unit.properties_in: ModifiedADM1PropertiesScaler,
                m.fs.unit.properties_out: ModifiedASM2dPropertiesScaler,
            },
        )

        # Check condition number to confirm scaling
        sm = TransformationFactory("core.scale_model").create_using(m, rename=False)
        jac, _ = get_jacobian(sm, scaled=False)
        assert (jacobian_cond(jac=jac, scaled=False)) == pytest.approx(
            7.180968e3, rel=1e-3
        )
