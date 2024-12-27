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
Tests for ASM2d-ADM1 Translator unit model.

Verified against approximated results results from:

Flores-Alsina, X., Solon, K., Mbamba, C.K., Tait, S., Gernaey, K.V., Jeppsson, U. and Batstone, D.J., 2016.
Modelling phosphorus (P), sulfur (S) and iron (Fe) interactions for dynamic simulations of anaerobic digestion processes.
Water Research, 95, pp.370-382.
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

from pyomo.environ import units

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
from idaes.core.scaling.scaling_base import ScalerBase

import idaes.logger as idaeslog
from idaes.core.util.testing import initialization_tester

from watertap.unit_models.translators.translator_asm2d_adm1 import (
    Translator_ASM2d_ADM1,
    ASM2dADM1Scaler,
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

    m.fs.props_ASM2d = ModifiedASM2dParameterBlock()
    m.fs.props_ADM1 = ModifiedADM1ParameterBlock()
    m.fs.ADM1_rxn_props = ModifiedADM1ReactionParameterBlock(
        property_package=m.fs.props_ADM1
    )
    m.fs.ASM2d_rxn_props = ModifiedASM2dReactionParameterBlock(
        property_package=m.fs.props_ASM2d
    )

    m.fs.unit = Translator_ASM2d_ADM1(
        inlet_property_package=m.fs.props_ASM2d,
        outlet_property_package=m.fs.props_ADM1,
        inlet_reaction_package=m.fs.ASM2d_rxn_props,
        outlet_reaction_package=m.fs.ADM1_rxn_props,
        has_phase_equilibrium=False,
        outlet_state_defined=True,
    )

    assert len(m.fs.unit.config) == 13

    assert m.fs.unit.config.outlet_state_defined == True
    assert not m.fs.unit.config.dynamic
    assert not m.fs.unit.config.has_holdup
    assert not m.fs.unit.config.has_phase_equilibrium
    assert m.fs.unit.config.inlet_property_package is m.fs.props_ASM2d
    assert m.fs.unit.config.outlet_property_package is m.fs.props_ADM1
    assert m.fs.unit.config.inlet_reaction_package is m.fs.ASM2d_rxn_props
    assert m.fs.unit.config.outlet_reaction_package is m.fs.ADM1_rxn_props


# -----------------------------------------------------------------------------
class TestAsm2dAdm1_bioP_true(object):
    @pytest.fixture(scope="class")
    def asmadm(self):
        m = ConcreteModel()

        m.fs = FlowsheetBlock(dynamic=False)

        m.fs.props_ASM2d = ModifiedASM2dParameterBlock()
        m.fs.props_ADM1 = ModifiedADM1ParameterBlock()
        m.fs.ADM1_rxn_props = ModifiedADM1ReactionParameterBlock(
            property_package=m.fs.props_ADM1
        )
        m.fs.ASM2d_rxn_props = ModifiedASM2dReactionParameterBlock(
            property_package=m.fs.props_ASM2d,
        )

        m.fs.unit = Translator_ASM2d_ADM1(
            inlet_property_package=m.fs.props_ASM2d,
            outlet_property_package=m.fs.props_ADM1,
            inlet_reaction_package=m.fs.ASM2d_rxn_props,
            outlet_reaction_package=m.fs.ADM1_rxn_props,
            has_phase_equilibrium=False,
            outlet_state_defined=True,
            bio_P=True,
        )

        m.fs.unit.inlet.flow_vol.fix(18446 * units.m**3 / units.day)
        m.fs.unit.inlet.temperature.fix(308.15 * units.K)
        m.fs.unit.inlet.pressure.fix(1 * units.atm)
        eps = 1e-9 * units.kg / units.m**3

        m.fs.unit.inlet.conc_mass_comp[0, "S_O2"].fix(eps)
        m.fs.unit.inlet.conc_mass_comp[0, "S_F"].fix(0.02644 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "S_A"].fix(0.01766 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "S_I"].fix(0.02723 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "S_NH4"].fix(0.01858 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "S_N2"].fix(0.00507 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "S_NO3"].fix(0.00002 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "S_PO4"].fix(0.00469 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "S_IC"].fix(0.07899 * units.kg / units.m**3)

        m.fs.unit.inlet.conc_mass_comp[0, "X_I"].fix(10.96441 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "X_S"].fix(19.08476 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "X_H"].fix(9.47939 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "X_PAO"].fix(3.8622 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "X_PP"].fix(0.45087 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "X_PHA"].fix(0.02464 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "X_AUT"].fix(0.33379 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "S_K"].fix(0.01979 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "S_Mg"].fix(0.18987 * units.kg / units.m**3)

        return m

    @pytest.mark.build
    @pytest.mark.unit
    def test_build(self, asmadm):

        assert isinstance(asmadm.fs.unit.f_XPHA_Sva, Param)
        assert value(asmadm.fs.unit.f_XPHA_Sva) == 0.1
        assert isinstance(asmadm.fs.unit.f_XPHA_Sbu, Param)
        assert value(asmadm.fs.unit.f_XPHA_Sbu) == 0.1
        assert isinstance(asmadm.fs.unit.f_XPHA_Spro, Param)
        assert value(asmadm.fs.unit.f_XPHA_Spro) == 0.4
        assert isinstance(asmadm.fs.unit.f_XPHA_Sac, Param)
        assert value(asmadm.fs.unit.f_XPHA_Sac) == 0.4

        assert isinstance(asmadm.fs.unit.C_PHA, Param)
        assert value(asmadm.fs.unit.C_PHA) == 0.3 / 12
        assert isinstance(asmadm.fs.unit.i_PSI, Param)
        assert value(asmadm.fs.unit.i_PSI) == 0.00649

        assert hasattr(asmadm.fs.unit, "inlet")
        assert len(asmadm.fs.unit.inlet.vars) == 4
        assert hasattr(asmadm.fs.unit.inlet, "flow_vol")
        assert hasattr(asmadm.fs.unit.inlet, "conc_mass_comp")
        assert hasattr(asmadm.fs.unit.inlet, "temperature")
        assert hasattr(asmadm.fs.unit.inlet, "pressure")

        assert hasattr(asmadm.fs.unit, "outlet")
        assert len(asmadm.fs.unit.outlet.vars) == 6
        assert hasattr(asmadm.fs.unit.outlet, "flow_vol")
        assert hasattr(asmadm.fs.unit.outlet, "conc_mass_comp")
        assert hasattr(asmadm.fs.unit.outlet, "temperature")
        assert hasattr(asmadm.fs.unit.outlet, "pressure")
        assert hasattr(asmadm.fs.unit.outlet, "anions")
        assert hasattr(asmadm.fs.unit.outlet, "cations")

        assert number_variables(asmadm) == 278
        assert number_total_constraints(asmadm) == 34

        # TODO: Result of SN2_AS2 being unused. Remove? It's also unused in the c-code
        assert number_unused_variables(asmadm.fs.unit) == 1

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
        assert pytest.approx(0.01843, rel=1e-3) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "S_su"]
        )
        assert pytest.approx(0.00801, rel=1e-3) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "S_aa"]
        )
        assert pytest.approx(1e-9, abs=1e-6) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "S_fa"]
        )
        assert pytest.approx(1e-10, abs=1e-6) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "S_va"]
        )
        assert pytest.approx(1e-10, abs=1e-6) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "S_bu"]
        )
        assert pytest.approx(4e-10, abs=1e-6) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "S_pro"]
        )
        assert pytest.approx(0.0175076, rel=1e-3) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "S_ac"]
        )
        assert pytest.approx(1e-9, abs=1e-6) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "S_h2"]
        )
        assert pytest.approx(1e-9, abs=1e-6) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "S_ch4"]
        )
        assert pytest.approx(0.382172, rel=1e-3) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "S_IC"]
        )
        assert pytest.approx(0.6985, rel=1e-3) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "S_IN"]
        )
        assert pytest.approx(0.71189, rel=1e-3) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "S_IP"]
        )
        assert pytest.approx(0.02723, rel=1e-3) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "S_I"]
        )

        assert pytest.approx(9.082, rel=1e-3) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "X_ch"]
        )
        assert pytest.approx(9.5424, rel=1e-3) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "X_pr"]
        )
        assert pytest.approx(12.7683, rel=1e-3) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "X_li"]
        )
        assert pytest.approx(1e-9, abs=1e-6) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "X_su"]
        )
        assert pytest.approx(1e-9, abs=1e-6) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "X_aa"]
        )
        assert pytest.approx(1e-9, abs=1e-6) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "X_fa"]
        )
        assert pytest.approx(1e-9, abs=1e-6) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "X_c4"]
        )
        assert pytest.approx(1e-9, abs=1e-6) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "X_pro"]
        )
        assert pytest.approx(1e-9, abs=1e-6) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "X_ac"]
        )
        assert pytest.approx(1e-9, abs=1e-6) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "X_h2"]
        )
        assert pytest.approx(12.3320, rel=1e-3) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "X_I"]
        )
        assert pytest.approx(1e-9, abs=1e-6) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "X_PHA"]
        )
        assert pytest.approx(1e-9, abs=1e-6) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "X_PP"]
        )
        assert pytest.approx(1e-9, abs=1e-6) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "X_PAO"]
        )
        assert pytest.approx(0.1701, rel=1e-3) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "S_K"]
        )
        assert pytest.approx(0.34016, rel=1e-3) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "S_Mg"]
        )
        assert pytest.approx(0.031848, rel=1e-3) == value(
            asmadm.fs.unit.outlet.cations[0]
        )
        assert pytest.approx(0.04989, rel=1e-3) == value(
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
                    - asmadm.fs.unit.outlet.flow_vol[0]
                    * asmadm.fs.props_ASM2d.dens_mass
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
                        * asmadm.fs.props_ASM2d.dens_mass
                        * asmadm.fs.props_ASM2d.cp_mass
                        * (
                            asmadm.fs.unit.outlet.temperature[0]
                            - asmadm.fs.props_ASM2d.temperature_ref
                        )
                    )
                )
            )
            <= 1e-6
        )


# -----------------------------------------------------------------------------
class TestAsm2dAdm1_bioP_false(object):
    @pytest.fixture(scope="class")
    def asmadm(self):
        m = ConcreteModel()

        m.fs = FlowsheetBlock(dynamic=False)

        m.fs.props_ASM2d = ModifiedASM2dParameterBlock()
        m.fs.props_ADM1 = ModifiedADM1ParameterBlock()
        m.fs.ADM1_rxn_props = ModifiedADM1ReactionParameterBlock(
            property_package=m.fs.props_ADM1
        )
        m.fs.ASM2d_rxn_props = ModifiedASM2dReactionParameterBlock(
            property_package=m.fs.props_ASM2d,
        )

        m.fs.unit = Translator_ASM2d_ADM1(
            inlet_property_package=m.fs.props_ASM2d,
            outlet_property_package=m.fs.props_ADM1,
            inlet_reaction_package=m.fs.ASM2d_rxn_props,
            outlet_reaction_package=m.fs.ADM1_rxn_props,
            has_phase_equilibrium=False,
            outlet_state_defined=True,
            bio_P=False,
        )

        m.fs.unit.inlet.flow_vol.fix(18446 * units.m**3 / units.day)
        m.fs.unit.inlet.temperature.fix(308.15 * units.K)
        m.fs.unit.inlet.pressure.fix(1 * units.atm)
        eps = 1e-9 * units.kg / units.m**3

        m.fs.unit.inlet.conc_mass_comp[0, "S_O2"].fix(eps)
        m.fs.unit.inlet.conc_mass_comp[0, "S_F"].fix(0.02644 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "S_A"].fix(0.01766 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "S_I"].fix(0.02723 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "S_NH4"].fix(0.01858 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "S_N2"].fix(0.00507 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "S_NO3"].fix(0.00002 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "S_PO4"].fix(0.00469 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "S_IC"].fix(0.07899 * units.kg / units.m**3)

        m.fs.unit.inlet.conc_mass_comp[0, "X_I"].fix(10.96441 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "X_S"].fix(19.08476 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "X_H"].fix(9.47939 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "X_PAO"].fix(3.8622 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "X_PP"].fix(0.45087 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "X_PHA"].fix(0.02464 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "X_AUT"].fix(0.33379 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "S_K"].fix(0.01979 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "S_Mg"].fix(0.18987 * units.kg / units.m**3)

        return m

    @pytest.mark.build
    @pytest.mark.unit
    def test_build(self, asmadm):

        assert isinstance(asmadm.fs.unit.f_XPHA_Sva, Param)
        assert value(asmadm.fs.unit.f_XPHA_Sva) == 0.1
        assert isinstance(asmadm.fs.unit.f_XPHA_Sbu, Param)
        assert value(asmadm.fs.unit.f_XPHA_Sbu) == 0.1
        assert isinstance(asmadm.fs.unit.f_XPHA_Spro, Param)
        assert value(asmadm.fs.unit.f_XPHA_Spro) == 0.4
        assert isinstance(asmadm.fs.unit.f_XPHA_Sac, Param)
        assert value(asmadm.fs.unit.f_XPHA_Sac) == 0.4

        assert isinstance(asmadm.fs.unit.C_PHA, Param)
        assert value(asmadm.fs.unit.C_PHA) == 0.3 / 12
        assert isinstance(asmadm.fs.unit.i_PSI, Param)
        assert value(asmadm.fs.unit.i_PSI) == 0.00649

        assert hasattr(asmadm.fs.unit, "inlet")
        assert len(asmadm.fs.unit.inlet.vars) == 4
        assert hasattr(asmadm.fs.unit.inlet, "flow_vol")
        assert hasattr(asmadm.fs.unit.inlet, "conc_mass_comp")
        assert hasattr(asmadm.fs.unit.inlet, "temperature")
        assert hasattr(asmadm.fs.unit.inlet, "pressure")

        assert hasattr(asmadm.fs.unit, "outlet")
        assert len(asmadm.fs.unit.outlet.vars) == 6
        assert hasattr(asmadm.fs.unit.outlet, "flow_vol")
        assert hasattr(asmadm.fs.unit.outlet, "conc_mass_comp")
        assert hasattr(asmadm.fs.unit.outlet, "temperature")
        assert hasattr(asmadm.fs.unit.outlet, "pressure")
        assert hasattr(asmadm.fs.unit.outlet, "anions")
        assert hasattr(asmadm.fs.unit.outlet, "cations")

        assert number_variables(asmadm) == 278
        assert number_total_constraints(asmadm) == 34

        # TODO: Result of SN2_AS2 being unused. Remove? It's also unused in the c-code
        assert number_unused_variables(asmadm.fs.unit) == 1

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
        assert pytest.approx(0.01843, rel=1e-3) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "S_su"]
        )
        assert pytest.approx(0.00801, rel=1e-3) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "S_aa"]
        )
        assert pytest.approx(1e-9, abs=1e-6) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "S_fa"]
        )
        assert pytest.approx(9.9999e-10, abs=1e-6) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "S_va"]
        )
        assert pytest.approx(9.9999e-10, abs=1e-6) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "S_bu"]
        )
        assert pytest.approx(9.9999e-10, abs=1e-6) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "S_pro"]
        )
        assert pytest.approx(0.0175076, rel=1e-3) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "S_ac"]
        )
        assert pytest.approx(1e-9, abs=1e-6) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "S_h2"]
        )
        assert pytest.approx(1e-9, abs=1e-6) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "S_ch4"]
        )
        assert pytest.approx(0.253013, rel=1e-3) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "S_IC"]
        )
        assert pytest.approx(0.50648, rel=1e-3) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "S_IN"]
        )
        assert pytest.approx(0.19475, rel=1e-3) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "S_IP"]
        )
        assert pytest.approx(0.02723, rel=1e-3) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "S_I"]
        )

        assert pytest.approx(8.0199, rel=1e-3) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "X_ch"]
        )
        assert pytest.approx(8.48026, rel=1e-3) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "X_pr"]
        )
        assert pytest.approx(11.4165, rel=1e-3) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "X_li"]
        )
        assert pytest.approx(1e-9, abs=1e-6) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "X_su"]
        )
        assert pytest.approx(1e-9, abs=1e-6) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "X_aa"]
        )
        assert pytest.approx(1e-9, abs=1e-6) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "X_fa"]
        )
        assert pytest.approx(1e-9, abs=1e-6) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "X_c4"]
        )
        assert pytest.approx(1e-9, abs=1e-6) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "X_pro"]
        )
        assert pytest.approx(1e-9, abs=1e-6) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "X_ac"]
        )
        assert pytest.approx(1e-9, abs=1e-6) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "X_h2"]
        )
        assert pytest.approx(11.94574, rel=1e-3) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "X_I"]
        )
        assert pytest.approx(0.02464, rel=1e-3) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "X_PHA"]
        )
        assert pytest.approx(0.45087, rel=1e-3) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "X_PP"]
        )
        assert pytest.approx(3.8622, rel=1e-3) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "X_PAO"]
        )
        assert pytest.approx(0.01979, rel=1e-3) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "S_K"]
        )
        assert pytest.approx(0.18987, rel=1e-3) == value(
            asmadm.fs.unit.outlet.conc_mass_comp[0, "S_Mg"]
        )
        assert pytest.approx(0.0210843, rel=1e-3) == value(
            asmadm.fs.unit.outlet.cations[0]
        )
        assert pytest.approx(0.03618, rel=1e-3) == value(
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
                    - asmadm.fs.unit.outlet.flow_vol[0]
                    * asmadm.fs.props_ASM2d.dens_mass
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
                        * asmadm.fs.props_ASM2d.dens_mass
                        * asmadm.fs.props_ASM2d.cp_mass
                        * (
                            asmadm.fs.unit.outlet.temperature[0]
                            - asmadm.fs.props_ASM2d.temperature_ref
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

        m.fs.props_ASM2d = ModifiedASM2dParameterBlock()
        m.fs.props_ADM1 = ModifiedADM1ParameterBlock()
        m.fs.ADM1_rxn_props = ModifiedADM1ReactionParameterBlock(
            property_package=m.fs.props_ADM1
        )
        m.fs.ASM2d_rxn_props = ModifiedASM2dReactionParameterBlock(
            property_package=m.fs.props_ASM2d,
        )

        m.fs.unit = Translator_ASM2d_ADM1(
            inlet_property_package=m.fs.props_ASM2d,
            outlet_property_package=m.fs.props_ADM1,
            inlet_reaction_package=m.fs.ASM2d_rxn_props,
            outlet_reaction_package=m.fs.ADM1_rxn_props,
            has_phase_equilibrium=False,
            outlet_state_defined=True,
            bio_P=True,
        )

        m.fs.unit.inlet.flow_vol.fix(18446 * units.m**3 / units.day)
        m.fs.unit.inlet.temperature.fix(308.15 * units.K)
        m.fs.unit.inlet.pressure.fix(1 * units.atm)
        eps = 1e-9 * units.kg / units.m**3

        m.fs.unit.inlet.conc_mass_comp[0, "S_O2"].fix(eps)
        m.fs.unit.inlet.conc_mass_comp[0, "S_F"].fix(0.02644 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "S_A"].fix(0.01766 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "S_I"].fix(0.02723 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "S_NH4"].fix(0.01858 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "S_N2"].fix(0.00507 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "S_NO3"].fix(0.00002 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "S_PO4"].fix(0.00469 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "S_IC"].fix(0.07899 * units.kg / units.m**3)

        m.fs.unit.inlet.conc_mass_comp[0, "X_I"].fix(10.96441 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "X_S"].fix(19.08476 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "X_H"].fix(9.47939 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "X_PAO"].fix(3.8622 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "X_PP"].fix(0.45087 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "X_PHA"].fix(0.02464 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "X_AUT"].fix(0.33379 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "S_K"].fix(0.01979 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "S_Mg"].fix(0.18987 * units.kg / units.m**3)

        return m

    @pytest.mark.component
    def test_variable_scaling_routine(self, model):
        scaler = model.fs.unit.default_scaler()

        assert isinstance(scaler, ASM2dADM1Scaler)

        scaler.variable_scaling_routine(model.fs.unit)

        # Inlet state
        sfx_in = model.fs.unit.properties_in[0].scaling_factor
        assert isinstance(sfx_in, Suffix)
        assert len(sfx_in) == 3
        assert sfx_in[model.fs.unit.properties_in[0].flow_vol] == pytest.approx(
            1e1, rel=1e-8
        )
        assert sfx_in[model.fs.unit.properties_in[0].pressure] == pytest.approx(
            1e-5, rel=1e-8
        )
        assert sfx_in[model.fs.unit.properties_in[0].temperature] == pytest.approx(
            1e-2, rel=1e-8
        )

        # Outlet state - should be the same as the inlet
        sfx_out = model.fs.unit.properties_out[0].scaling_factor
        assert isinstance(sfx_out, Suffix)
        assert len(sfx_out) == 3
        assert sfx_out[model.fs.unit.properties_out[0].flow_vol] == pytest.approx(
            1e5, rel=1e-8
        )
        assert sfx_out[model.fs.unit.properties_out[0].pressure] == pytest.approx(
            1e-6, rel=1e-8
        )
        assert sfx_out[model.fs.unit.properties_out[0].temperature] == pytest.approx(
            1e-1, rel=1e-8
        )

    @pytest.mark.component
    def test_constraint_scaling_routine(self, model):
        scaler = model.fs.unit.default_scaler()

        assert isinstance(scaler, ASM2dADM1Scaler)

        scaler.constraint_scaling_routine(model.fs.unit)

        sfx_out = model.fs.unit.properties_out[0].scaling_factor
        assert isinstance(sfx_out, Suffix)
        assert len(sfx_out) == 0

    @pytest.mark.component
    def test_scale_model(self, model):
        scaler = model.fs.unit.default_scaler()

        assert isinstance(scaler, ASM2dADM1Scaler)

        scaler.scale_model(model.fs.unit)

        # Inlet state
        sfx_in = model.fs.unit.properties_in[0].scaling_factor
        assert isinstance(sfx_in, Suffix)
        assert len(sfx_in) == 3
        assert sfx_in[model.fs.unit.properties_in[0].flow_vol] == pytest.approx(
            1e1, rel=1e-8
        )
        assert sfx_in[model.fs.unit.properties_in[0].pressure] == pytest.approx(
            1e-5, rel=1e-8
        )
        assert sfx_in[model.fs.unit.properties_in[0].temperature] == pytest.approx(
            1e-2, rel=1e-8
        )

        # Outlet state - should be the same as the inlet
        sfx_out = model.fs.unit.properties_out[0].scaling_factor
        assert isinstance(sfx_out, Suffix)
        assert len(sfx_out) == 3
        assert sfx_out[model.fs.unit.properties_out[0].flow_vol] == pytest.approx(
            1e5, rel=1e-8
        )
        assert sfx_out[model.fs.unit.properties_out[0].pressure] == pytest.approx(
            1e-6, rel=1e-8
        )
        assert sfx_out[model.fs.unit.properties_out[0].temperature] == pytest.approx(
            1e-1, rel=1e-8
        )

    # TODO: Remove test once iscale is deprecated
    @pytest.mark.integration
    def test_example_case_iscale(self):
        m = ConcreteModel()

        m.fs = FlowsheetBlock(dynamic=False)

        m.fs.props_ASM2d = ModifiedASM2dParameterBlock()
        m.fs.props_ADM1 = ModifiedADM1ParameterBlock()
        m.fs.ADM1_rxn_props = ModifiedADM1ReactionParameterBlock(
            property_package=m.fs.props_ADM1
        )
        m.fs.ASM2d_rxn_props = ModifiedASM2dReactionParameterBlock(
            property_package=m.fs.props_ASM2d,
        )

        m.fs.unit = Translator_ASM2d_ADM1(
            inlet_property_package=m.fs.props_ASM2d,
            outlet_property_package=m.fs.props_ADM1,
            inlet_reaction_package=m.fs.ASM2d_rxn_props,
            outlet_reaction_package=m.fs.ADM1_rxn_props,
            has_phase_equilibrium=False,
            outlet_state_defined=True,
            bio_P=True,
        )

        m.fs.unit.inlet.flow_vol.fix(18446 * units.m**3 / units.day)
        m.fs.unit.inlet.temperature.fix(308.15 * units.K)
        m.fs.unit.inlet.pressure.fix(1 * units.atm)
        eps = 1e-9 * units.kg / units.m**3

        m.fs.unit.inlet.conc_mass_comp[0, "S_O2"].fix(eps)
        m.fs.unit.inlet.conc_mass_comp[0, "S_F"].fix(0.02644 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "S_A"].fix(0.01766 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "S_I"].fix(0.02723 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "S_NH4"].fix(0.01858 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "S_N2"].fix(0.00507 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "S_NO3"].fix(0.00002 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "S_PO4"].fix(0.00469 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "S_IC"].fix(0.07899 * units.kg / units.m**3)

        m.fs.unit.inlet.conc_mass_comp[0, "X_I"].fix(10.96441 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "X_S"].fix(19.08476 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "X_H"].fix(9.47939 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "X_PAO"].fix(3.8622 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "X_PP"].fix(0.45087 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "X_PHA"].fix(0.02464 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "X_AUT"].fix(0.33379 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "S_K"].fix(0.01979 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "S_Mg"].fix(0.18987 * units.kg / units.m**3)

        iscale.set_scaling_factor(
            m.fs.unit.properties_out[0].conc_mass_comp["S_aa"], 1e3
        )

        iscale.calculate_scaling_factors(m.fs.unit)

        # Check condition number to confirm scaling
        sm = TransformationFactory("core.scale_model").create_using(m, rename=False)
        jac, _ = get_jacobian(sm, scaled=False)
        assert (jacobian_cond(jac=jac, scaled=False)) == pytest.approx(
            5.745705986257e3, rel=1e-3
        )

    @pytest.mark.integration
    def test_example_case_scaler(self):
        m = ConcreteModel()

        m.fs = FlowsheetBlock(dynamic=False)

        m.fs.props_ASM2d = ModifiedASM2dParameterBlock()
        m.fs.props_ADM1 = ModifiedADM1ParameterBlock()
        m.fs.ADM1_rxn_props = ModifiedADM1ReactionParameterBlock(
            property_package=m.fs.props_ADM1
        )
        m.fs.ASM2d_rxn_props = ModifiedASM2dReactionParameterBlock(
            property_package=m.fs.props_ASM2d,
        )

        m.fs.unit = Translator_ASM2d_ADM1(
            inlet_property_package=m.fs.props_ASM2d,
            outlet_property_package=m.fs.props_ADM1,
            inlet_reaction_package=m.fs.ASM2d_rxn_props,
            outlet_reaction_package=m.fs.ADM1_rxn_props,
            has_phase_equilibrium=False,
            outlet_state_defined=True,
            bio_P=True,
        )

        m.fs.unit.inlet.flow_vol.fix(18446 * units.m**3 / units.day)
        m.fs.unit.inlet.temperature.fix(308.15 * units.K)
        m.fs.unit.inlet.pressure.fix(1 * units.atm)
        eps = 1e-9 * units.kg / units.m**3

        m.fs.unit.inlet.conc_mass_comp[0, "S_O2"].fix(eps)
        m.fs.unit.inlet.conc_mass_comp[0, "S_F"].fix(0.02644 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "S_A"].fix(0.01766 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "S_I"].fix(0.02723 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "S_NH4"].fix(0.01858 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "S_N2"].fix(0.00507 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "S_NO3"].fix(0.00002 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "S_PO4"].fix(0.00469 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "S_IC"].fix(0.07899 * units.kg / units.m**3)

        m.fs.unit.inlet.conc_mass_comp[0, "X_I"].fix(10.96441 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "X_S"].fix(19.08476 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "X_H"].fix(9.47939 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "X_PAO"].fix(3.8622 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "X_PP"].fix(0.45087 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "X_PHA"].fix(0.02464 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "X_AUT"].fix(0.33379 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "S_K"].fix(0.01979 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "S_Mg"].fix(0.18987 * units.kg / units.m**3)

        scaler = ASM2dADM1Scaler()
        scaler.scale_model(
            m.fs.unit,
            submodel_scalers={
                m.fs.unit.properties_in: ModifiedASM2dPropertiesScaler,
                m.fs.unit.properties_out: ModifiedADM1PropertiesScaler,
            },
        )

        sb = ScalerBase()
        sb.set_variable_scaling_factor(
            m.fs.unit.properties_out[0].flow_vol, 1, overwrite=True
        )
        sb.set_variable_scaling_factor(
            m.fs.unit.properties_out[0].conc_mass_comp["S_aa"], 1e3
        )
        sb.set_variable_scaling_factor(
            m.fs.unit.properties_out[0].conc_mass_comp["S_ch4"], 1e9
        )
        sb.set_variable_scaling_factor(
            m.fs.unit.properties_out[0].conc_mass_comp["X_su"], 1e9
        )
        sb.set_variable_scaling_factor(
            m.fs.unit.properties_out[0].conc_mass_comp["X_fa"], 1e9
        )
        sb.set_variable_scaling_factor(
            m.fs.unit.properties_out[0].conc_mass_comp["X_c4"], 1e9
        )
        sb.set_variable_scaling_factor(
            m.fs.unit.properties_out[0].conc_mass_comp["X_pro"], 1e9
        )
        sb.set_variable_scaling_factor(
            m.fs.unit.properties_out[0].conc_mass_comp["X_ac"], 1e9
        )

        # Check condition number to confirm scaling
        sm = TransformationFactory("core.scale_model").create_using(m, rename=False)
        jac, _ = get_jacobian(sm, scaled=False)
        assert (jacobian_cond(jac=jac, scaled=False)) == pytest.approx(
            5.640234319533e3, rel=1e-3
        )
