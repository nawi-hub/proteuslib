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
Tests for CSTR unit model.
Authors: Marcus Holly
"""
from io import StringIO
import pytest

from pyomo.environ import (
    ConcreteModel,
    units,
    value,
    Objective,
    Suffix,
    TransformationFactory,
)

from watertap.unit_models.tests.unit_test_harness import UnitTestHarness
import idaes.core.util.scaling as iscale
from idaes.core import (
    FlowsheetBlock,
    UnitModelCostingBlock,
)
from idaes.core.util.scaling import (
    get_jacobian,
    jacobian_cond,
)
from idaes.core.scaling.scaler_profiling import ScalingProfiler
from watertap.unit_models.cstr import CSTR, CSTRScaler
from watertap.costing import WaterTAPCosting

from watertap.property_models.unit_specific.activated_sludge.asm1_properties import (
    ASM1ParameterBlock,
    ASM1PropertiesScaler,
)
from watertap.property_models.unit_specific.activated_sludge.asm1_reactions import (
    ASM1ReactionParameterBlock,
    ASM1ReactionScaler,
)

from idaes.models.properties.examples.saponification_thermo import (
    SaponificationParameterBlock,
)
from idaes.models.properties.examples.saponification_reactions import (
    SaponificationReactionParameterBlock,
)
from watertap.core.solvers import get_solver
from idaes.core.initialization import (
    BlockTriangularizationInitializer,
    SingleControlVolumeUnitInitializer,
    InitializationStatus,
)
import idaes.logger as idaeslog

# -----------------------------------------------------------------------------
# Get default solver for testing
solver = get_solver()


# -----------------------------------------------------------------------------
def build():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)

    m.fs.properties = SaponificationParameterBlock()
    m.fs.reactions = SaponificationReactionParameterBlock(
        property_package=m.fs.properties
    )

    m.fs.unit = CSTR(
        property_package=m.fs.properties,
        reaction_package=m.fs.reactions,
        has_equilibrium_reactions=False,
        has_heat_transfer=True,
        has_heat_of_reaction=True,
        has_pressure_change=True,
    )

    m.fs.unit.inlet.flow_vol.fix(1.0e-03)
    m.fs.unit.inlet.conc_mol_comp[0, "H2O"].fix(55388.0)
    m.fs.unit.inlet.conc_mol_comp[0, "NaOH"].fix(100.0)
    m.fs.unit.inlet.conc_mol_comp[0, "EthylAcetate"].fix(100.0)
    m.fs.unit.inlet.conc_mol_comp[0, "SodiumAcetate"].fix(0.0)
    m.fs.unit.inlet.conc_mol_comp[0, "Ethanol"].fix(0.0)

    m.fs.unit.inlet.temperature.fix(303.15)
    m.fs.unit.inlet.pressure.fix(101325.0)

    m.fs.unit.volume.fix(1.5e-03)
    m.fs.unit.heat_duty.fix(0)
    m.fs.unit.deltaP.fix(0)

    iscale.set_scaling_factor(
        m.fs.unit.control_volume.properties_out[0.0].conc_mol_comp["H2O"], 1e-5
    )
    iscale.set_scaling_factor(
        m.fs.unit.control_volume.properties_out[0.0].pressure, 1e-6
    )

    iscale.calculate_scaling_factors(m.fs.unit)

    return m


def build_ASM1():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)

    m.fs.props_ASM1 = ASM1ParameterBlock()
    m.fs.ASM1_rxn_props = ASM1ReactionParameterBlock(property_package=m.fs.props_ASM1)

    m.fs.unit = CSTR(
        property_package=m.fs.props_ASM1, reaction_package=m.fs.ASM1_rxn_props
    )

    m.fs.unit.inlet.flow_vol[0].fix(1.2199 * units.m**3 / units.s)
    m.fs.unit.inlet.alkalinity[0].fix(4.5102 * units.mole / units.m**3)
    m.fs.unit.inlet.conc_mass_comp[0, "S_I"].fix(0.061909 * units.kg / units.m**3)
    m.fs.unit.inlet.conc_mass_comp[0, "S_S"].fix(0.012366 * units.kg / units.m**3)
    m.fs.unit.inlet.conc_mass_comp[0, "X_I"].fix(1.4258 * units.kg / units.m**3)
    m.fs.unit.inlet.conc_mass_comp[0, "X_S"].fix(0.090508 * units.kg / units.m**3)
    m.fs.unit.inlet.conc_mass_comp[0, "X_BH"].fix(2.8404 * units.kg / units.m**3)
    m.fs.unit.inlet.conc_mass_comp[0, "X_BA"].fix(0.20512 * units.kg / units.m**3)
    m.fs.unit.inlet.conc_mass_comp[0, "X_P"].fix(0.58681 * units.kg / units.m**3)
    m.fs.unit.inlet.conc_mass_comp[0, "S_O"].fix(0.00036092 * units.kg / units.m**3)
    m.fs.unit.inlet.conc_mass_comp[0, "S_NO"].fix(0.012424 * units.kg / units.m**3)
    m.fs.unit.inlet.conc_mass_comp[0, "S_NH"].fix(0.0076936 * units.kg / units.m**3)
    m.fs.unit.inlet.conc_mass_comp[0, "S_ND"].fix(0.0019068 * units.kg / units.m**3)
    m.fs.unit.inlet.conc_mass_comp[0, "X_ND"].fix(0.0053166 * units.kg / units.m**3)

    m.fs.unit.inlet.temperature[0].fix(308.15 * units.K)
    m.fs.unit.inlet.pressure[0].fix(84790.0 * units.Pa)

    m.fs.unit.volume[0].fix(1000 * units.m**3)

    # Set scaling factors for badly scaled variables
    iscale.set_scaling_factor(m.fs.unit.control_volume.properties_out[0].pressure, 1e-5)
    iscale.set_scaling_factor(
        m.fs.unit.control_volume.properties_out[0.0].conc_mass_comp["S_O"], 1e5
    )
    iscale.set_scaling_factor(
        m.fs.unit.control_volume.rate_reaction_generation[0.0, "Liq", "X_BA"], 1e5
    )
    iscale.set_scaling_factor(
        m.fs.unit.control_volume.rate_reaction_generation[0.0, "Liq", "X_P"], 1e5
    )
    iscale.set_scaling_factor(
        m.fs.unit.control_volume.rate_reaction_generation[0.0, "Liq", "S_O"], 1e5
    )
    iscale.set_scaling_factor(
        m.fs.unit.control_volume.rate_reaction_generation[0.0, "Liq", "S_NH"], 1e5
    )
    iscale.set_scaling_factor(
        m.fs.unit.control_volume.rate_reaction_generation[0.0, "Liq", "S_ND"], 1e5
    )
    iscale.set_scaling_factor(
        m.fs.unit.control_volume.rate_reaction_generation[0.0, "Liq", "X_ND"], 1e5
    )

    iscale.set_scaling_factor(
        m.fs.unit.control_volume.rate_reaction_extent[0.0, "R1"], 1e5
    )
    iscale.set_scaling_factor(
        m.fs.unit.control_volume.rate_reaction_extent[0.0, "R3"], 1e5
    )
    iscale.set_scaling_factor(
        m.fs.unit.control_volume.rate_reaction_extent[0.0, "R5"], 1e5
    )

    iscale.set_scaling_factor(
        m.fs.unit.control_volume.reactions[0.0].reaction_rate["R1"], 1e7
    )
    iscale.set_scaling_factor(
        m.fs.unit.control_volume.reactions[0.0].reaction_rate["R2"], 1e7
    )
    iscale.set_scaling_factor(
        m.fs.unit.control_volume.reactions[0.0].reaction_rate["R3"], 1e7
    )
    iscale.set_scaling_factor(
        m.fs.unit.control_volume.reactions[0.0].reaction_rate["R4"], 1e7
    )
    iscale.set_scaling_factor(
        m.fs.unit.control_volume.reactions[0.0].reaction_rate["R5"], 1e7
    )
    iscale.set_scaling_factor(
        m.fs.unit.control_volume.reactions[0.0].reaction_rate["R6"], 1e7
    )
    iscale.set_scaling_factor(
        m.fs.unit.control_volume.reactions[0.0].reaction_rate["R7"], 1e7
    )
    iscale.set_scaling_factor(
        m.fs.unit.control_volume.reactions[0.0].reaction_rate["R8"], 1e7
    )

    iscale.calculate_scaling_factors(m.fs.unit)

    return m


class TestCSTR(UnitTestHarness):
    def configure(self):
        m = build()

        self.unit_solutions[m.fs.unit.outlet.pressure[0]] = 101325.0
        self.unit_solutions[m.fs.unit.outlet.temperature[0]] = 304.09
        self.unit_solutions[m.fs.unit.outlet.conc_mol_comp[0, "EthylAcetate"]] = 20.32

        self.conservation_equality = {
            "Check 1": {
                "in": m.fs.unit.inlet.flow_vol[0]
                * sum(
                    m.fs.unit.inlet.conc_mol_comp[0, j]
                    for j in m.fs.properties.component_list
                ),
                "out": m.fs.unit.outlet.flow_vol[0]
                * sum(
                    m.fs.unit.inlet.conc_mol_comp[0, j]
                    for j in m.fs.properties.component_list
                ),
            },
            "Check 2": {
                "in": (
                    m.fs.unit.inlet.flow_vol[0]
                    * m.fs.properties.dens_mol
                    * m.fs.properties.cp_mol
                    * (m.fs.unit.inlet.temperature[0] - m.fs.properties.temperature_ref)
                ),
                "out": m.fs.unit.outlet.flow_vol[0]
                * m.fs.properties.dens_mol
                * m.fs.properties.cp_mol
                * (m.fs.unit.outlet.temperature[0] - m.fs.properties.temperature_ref)
                - m.fs.unit.control_volume.heat_of_reaction[0],
            },
        }

        return m


class TestInitializers:
    @pytest.fixture
    def model(self):
        m = build()
        m = ConcreteModel()
        m.fs = FlowsheetBlock(dynamic=False)

        m.fs.properties = SaponificationParameterBlock()
        m.fs.reactions = SaponificationReactionParameterBlock(
            property_package=m.fs.properties
        )

        m.fs.unit = CSTR(
            property_package=m.fs.properties,
            reaction_package=m.fs.reactions,
            has_equilibrium_reactions=False,
            has_heat_transfer=True,
            has_heat_of_reaction=True,
            has_pressure_change=True,
        )

        m.fs.unit.inlet.flow_vol[0].set_value(1.0e-03)
        m.fs.unit.inlet.conc_mol_comp[0, "H2O"].set_value(55388.0)
        m.fs.unit.inlet.conc_mol_comp[0, "NaOH"].set_value(100.0)
        m.fs.unit.inlet.conc_mol_comp[0, "EthylAcetate"].set_value(100.0)
        m.fs.unit.inlet.conc_mol_comp[0, "SodiumAcetate"].set_value(0.0)
        m.fs.unit.inlet.conc_mol_comp[0, "Ethanol"].set_value(0.0)

        m.fs.unit.inlet.temperature[0].set_value(303.15)
        m.fs.unit.inlet.pressure[0].set_value(101325.0)

        m.fs.unit.volume[0].fix(1.5e-03)
        m.fs.unit.heat_duty[0].fix(0)
        m.fs.unit.deltaP[0].fix(0)

        iscale.calculate_scaling_factors(m)
        return m

    @pytest.mark.requires_idaes_solver
    @pytest.mark.component
    def test_general_hierarchical(self, model):
        initializer = SingleControlVolumeUnitInitializer(
            writer_config={"linear_presolve": False}
        )

        initializer.initialize(model.fs.unit, output_level=idaeslog.DEBUG)

        assert initializer.summary[model.fs.unit]["status"] == InitializationStatus.Ok

        assert value(model.fs.unit.outlet.flow_vol[0]) == pytest.approx(1e-3, rel=1e-5)
        assert value(model.fs.unit.outlet.conc_mol_comp[0, "H2O"]) == pytest.approx(
            55388, rel=1e-5
        )
        assert value(model.fs.unit.outlet.conc_mol_comp[0, "NaOH"]) == pytest.approx(
            20.31609, rel=1e-5
        )
        assert value(
            model.fs.unit.outlet.conc_mol_comp[0, "EthylAcetate"]
        ) == pytest.approx(20.31609, rel=1e-5)
        assert value(
            model.fs.unit.outlet.conc_mol_comp[0, "SodiumAcetate"]
        ) == pytest.approx(79.683910, rel=1e-5)
        assert value(model.fs.unit.outlet.conc_mol_comp[0, "Ethanol"]) == pytest.approx(
            79.683910, rel=1e-5
        )
        assert value(model.fs.unit.outlet.temperature[0]) == pytest.approx(
            304.0856, rel=1e-5
        )
        assert value(model.fs.unit.outlet.pressure[0]) == pytest.approx(
            101325, rel=1e-5
        )

        assert not model.fs.unit.inlet.flow_vol[0].fixed
        assert not model.fs.unit.inlet.conc_mol_comp[0, "H2O"].fixed
        assert not model.fs.unit.inlet.conc_mol_comp[0, "NaOH"].fixed
        assert not model.fs.unit.inlet.conc_mol_comp[0, "EthylAcetate"].fixed
        assert not model.fs.unit.inlet.conc_mol_comp[0, "SodiumAcetate"].fixed
        assert not model.fs.unit.inlet.conc_mol_comp[0, "Ethanol"].fixed

        assert not model.fs.unit.inlet.temperature[0].fixed
        assert not model.fs.unit.inlet.pressure[0].fixed

    @pytest.mark.component
    def test_block_triangularization(self, model):
        initializer = BlockTriangularizationInitializer(constraint_tolerance=2e-5)
        initializer.initialize(model.fs.unit)

        assert initializer.summary[model.fs.unit]["status"] == InitializationStatus.Ok

        assert value(model.fs.unit.outlet.flow_vol[0]) == pytest.approx(1e-3, rel=1e-5)
        assert value(model.fs.unit.outlet.conc_mol_comp[0, "H2O"]) == pytest.approx(
            55388, rel=1e-5
        )
        assert value(model.fs.unit.outlet.conc_mol_comp[0, "NaOH"]) == pytest.approx(
            20.31609, rel=1e-5
        )
        assert value(
            model.fs.unit.outlet.conc_mol_comp[0, "EthylAcetate"]
        ) == pytest.approx(20.31609, rel=1e-5)
        assert value(
            model.fs.unit.outlet.conc_mol_comp[0, "SodiumAcetate"]
        ) == pytest.approx(79.683910, rel=1e-5)
        assert value(model.fs.unit.outlet.conc_mol_comp[0, "Ethanol"]) == pytest.approx(
            79.683910, rel=1e-5
        )
        assert value(model.fs.unit.outlet.temperature[0]) == pytest.approx(
            304.0856, rel=1e-5
        )
        assert value(model.fs.unit.outlet.pressure[0]) == pytest.approx(
            101325, rel=1e-5
        )

        assert not model.fs.unit.inlet.flow_vol[0].fixed
        assert not model.fs.unit.inlet.conc_mol_comp[0, "H2O"].fixed
        assert not model.fs.unit.inlet.conc_mol_comp[0, "NaOH"].fixed
        assert not model.fs.unit.inlet.conc_mol_comp[0, "EthylAcetate"].fixed
        assert not model.fs.unit.inlet.conc_mol_comp[0, "SodiumAcetate"].fixed
        assert not model.fs.unit.inlet.conc_mol_comp[0, "Ethanol"].fixed

        assert not model.fs.unit.inlet.temperature[0].fixed
        assert not model.fs.unit.inlet.pressure[0].fixed


class TestCosting(UnitTestHarness):
    def configure(self):
        m = build_ASM1()
        m.fs.unit.initialize()

        solver.solve(m)

        # Add unit model costing
        m.fs.costing = WaterTAPCosting()

        m.fs.unit.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)
        m.fs.costing.cost_process()
        m.fs.costing.add_LCOW(m.fs.unit.control_volume.properties_out[0].flow_vol)
        m.objective = Objective(expr=m.fs.costing.LCOW)

        iscale.set_scaling_factor(
            m.fs.unit.control_volume.properties_out[0.0].conc_mass_comp["S_O"], 1e6
        )

        iscale.set_scaling_factor(m.fs.unit.costing.capital_cost, 1e-7)

        iscale.calculate_scaling_factors(m.fs.unit)

        self.unit_solutions[m.fs.unit.costing.capital_cost] = 566989.10
        self.unit_solutions[m.fs.costing.LCOW] = 0.002127

        solver.solve(m, tee=True)

        assert pytest.approx(0.002127, rel=1e-3) == value(m.fs.costing.LCOW)

        self.conservation_equality = {
            "Check 1": {
                "in": m.fs.unit.inlet.flow_vol[0],
                "out": m.fs.unit.outlet.flow_vol[0],
            },
        }

        return m


class TestCSTRScaler:
    @pytest.fixture
    def model(self):
        m = ConcreteModel()
        m.fs = FlowsheetBlock(dynamic=False)

        m.fs.props_ASM1 = ASM1ParameterBlock()
        m.fs.ASM1_rxn_props = ASM1ReactionParameterBlock(
            property_package=m.fs.props_ASM1
        )

        m.fs.unit = CSTR(
            property_package=m.fs.props_ASM1, reaction_package=m.fs.ASM1_rxn_props
        )

        m.fs.unit.inlet.flow_vol[0].fix(1.2199 * units.m**3 / units.s)
        m.fs.unit.inlet.alkalinity[0].fix(4.5102 * units.mole / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "S_I"].fix(0.061909 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "S_S"].fix(0.012366 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "X_I"].fix(1.4258 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "X_S"].fix(0.090508 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "X_BH"].fix(2.8404 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "X_BA"].fix(0.20512 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "X_P"].fix(0.58681 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "S_O"].fix(0.00036092 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "S_NO"].fix(0.012424 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "S_NH"].fix(0.0076936 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "S_ND"].fix(0.0019068 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "X_ND"].fix(0.0053166 * units.kg / units.m**3)

        m.fs.unit.inlet.temperature[0].fix(308.15 * units.K)
        m.fs.unit.inlet.pressure[0].fix(84790.0 * units.Pa)

        m.fs.unit.volume[0].fix(1000 * units.m**3)

        return m

    @pytest.mark.component
    def test_variable_scaling_routine(self, model):
        scaler = model.fs.unit.default_scaler()

        assert isinstance(scaler, CSTRScaler)

        scaler.variable_scaling_routine(model.fs.unit)

        # Inlet state
        sfx_in = model.fs.unit.control_volume.properties_in[0].scaling_factor
        assert isinstance(sfx_in, Suffix)
        assert len(sfx_in) == 3
        assert sfx_in[
            model.fs.unit.control_volume.properties_in[0].flow_vol
        ] == pytest.approx(1e1, rel=1e-8)
        assert sfx_in[
            model.fs.unit.control_volume.properties_in[0].pressure
        ] == pytest.approx(1e-6, rel=1e-8)
        assert sfx_in[
            model.fs.unit.control_volume.properties_in[0].temperature
        ] == pytest.approx(1e-1, rel=1e-8)

        # Outlet state - should be the same as the inlet
        sfx_out = model.fs.unit.control_volume.properties_out[0].scaling_factor
        assert isinstance(sfx_out, Suffix)
        assert len(sfx_out) == 3
        assert sfx_out[
            model.fs.unit.control_volume.properties_out[0].flow_vol
        ] == pytest.approx(1e1, rel=1e-8)
        assert sfx_out[
            model.fs.unit.control_volume.properties_out[0].pressure
        ] == pytest.approx(1e-6, rel=1e-8)
        assert sfx_out[
            model.fs.unit.control_volume.properties_out[0].temperature
        ] == pytest.approx(1e-1, rel=1e-8)

        # Reaction block
        sfx_rxn = model.fs.unit.control_volume.reactions[0].scaling_factor
        assert isinstance(sfx_rxn, Suffix)
        assert len(sfx_rxn) == 8
        assert sfx_rxn[
            model.fs.unit.control_volume.reactions[0].reaction_rate["R1"]
        ] == pytest.approx(1e2, rel=1e-8)
        assert sfx_rxn[
            model.fs.unit.control_volume.reactions[0].reaction_rate["R2"]
        ] == pytest.approx(1e2, rel=1e-8)
        assert sfx_rxn[
            model.fs.unit.control_volume.reactions[0].reaction_rate["R3"]
        ] == pytest.approx(1e2, rel=1e-8)
        assert sfx_rxn[
            model.fs.unit.control_volume.reactions[0].reaction_rate["R4"]
        ] == pytest.approx(1e2, rel=1e-8)
        assert sfx_rxn[
            model.fs.unit.control_volume.reactions[0].reaction_rate["R5"]
        ] == pytest.approx(1e2, rel=1e-8)
        assert sfx_rxn[
            model.fs.unit.control_volume.reactions[0].reaction_rate["R6"]
        ] == pytest.approx(1e2, rel=1e-8)
        assert sfx_rxn[
            model.fs.unit.control_volume.reactions[0].reaction_rate["R7"]
        ] == pytest.approx(1e2, rel=1e-8)
        assert sfx_rxn[
            model.fs.unit.control_volume.reactions[0].reaction_rate["R8"]
        ] == pytest.approx(1e2, rel=1e-8)

        # Check that unit model has scaling factors
        sfx_cv = model.fs.unit.control_volume.scaling_factor
        assert isinstance(sfx_cv, Suffix)
        assert len(sfx_cv) == 1
        assert sfx_cv[model.fs.unit.control_volume.volume[0]] == pytest.approx(
            1e-3, rel=1e-3
        )

    @pytest.mark.component
    def test_constraint_scaling_routine(self, model):
        scaler = model.fs.unit.default_scaler()

        assert isinstance(scaler, CSTRScaler)

        scaler.constraint_scaling_routine(model.fs.unit)

        sfx_out = model.fs.unit.control_volume.properties_out[0].scaling_factor
        assert isinstance(sfx_out, Suffix)
        assert len(sfx_out) == 0

        sfx_rxn = model.fs.unit.control_volume.reactions[0].scaling_factor
        assert isinstance(sfx_rxn, Suffix)
        assert len(sfx_rxn) == 8
        assert sfx_rxn[
            model.fs.unit.control_volume.reactions[0.0].rate_expression["R1"]
        ] == pytest.approx(2.380752e5, rel=1e-8)
        assert sfx_rxn[
            model.fs.unit.control_volume.reactions[0.0].rate_expression["R2"]
        ] == pytest.approx(1.49540985e8, rel=1e-8)
        assert sfx_rxn[
            model.fs.unit.control_volume.reactions[0.0].rate_expression["R3"]
        ] == pytest.approx(1.75226112e6, rel=1e-8)
        assert sfx_rxn[
            model.fs.unit.control_volume.reactions[0.0].rate_expression["R4"]
        ] == pytest.approx(2.88e6, rel=1e-8)
        assert sfx_rxn[
            model.fs.unit.control_volume.reactions[0.0].rate_expression["R5"]
        ] == pytest.approx(1.728e7, rel=1e-8)
        assert sfx_rxn[
            model.fs.unit.control_volume.reactions[0.0].rate_expression["R6"]
        ] == pytest.approx(1.728e5, rel=1e-8)
        assert sfx_rxn[
            model.fs.unit.control_volume.reactions[0.0].rate_expression["R7"]
        ] == pytest.approx(3.174336e5, rel=1e-8)
        assert sfx_rxn[
            model.fs.unit.control_volume.reactions[0.0].rate_expression["R8"]
        ] == pytest.approx(1, rel=1e-8)

        sfx_unit = model.fs.unit.scaling_factor
        assert isinstance(sfx_unit, Suffix)
        assert len(sfx_unit) == 1
        assert sfx_unit[model.fs.unit.CSTR_retention_time[0]] == pytest.approx(
            1.2199e-3, rel=1e-8
        )

    @pytest.mark.component
    def test_scale_model(self, model):
        scaler = model.fs.unit.default_scaler()

        assert isinstance(scaler, CSTRScaler)

        scaler.scale_model(model.fs.unit)

        # Inlet state
        sfx_in = model.fs.unit.control_volume.properties_in[0].scaling_factor
        assert isinstance(sfx_in, Suffix)
        assert len(sfx_in) == 3
        assert sfx_in[
            model.fs.unit.control_volume.properties_in[0].flow_vol
        ] == pytest.approx(1e1, rel=1e-8)
        assert sfx_in[
            model.fs.unit.control_volume.properties_in[0].pressure
        ] == pytest.approx(1e-6, rel=1e-8)
        assert sfx_in[
            model.fs.unit.control_volume.properties_in[0].temperature
        ] == pytest.approx(1e-1, rel=1e-8)

        # Outlet state - should be the same as the inlet
        sfx_out = model.fs.unit.control_volume.properties_out[0].scaling_factor
        assert isinstance(sfx_out, Suffix)
        assert len(sfx_out) == 3
        assert sfx_out[
            model.fs.unit.control_volume.properties_out[0].flow_vol
        ] == pytest.approx(1e1, rel=1e-8)
        assert sfx_out[
            model.fs.unit.control_volume.properties_out[0].pressure
        ] == pytest.approx(1e-6, rel=1e-8)
        assert sfx_out[
            model.fs.unit.control_volume.properties_out[0].temperature
        ] == pytest.approx(1e-1, rel=1e-8)

        # Reaction block
        sfx_rxn = model.fs.unit.control_volume.reactions[0].scaling_factor
        assert isinstance(sfx_rxn, Suffix)
        assert len(sfx_rxn) == 16
        assert sfx_rxn[
            model.fs.unit.control_volume.reactions[0].reaction_rate["R1"]
        ] == pytest.approx(1e2, rel=1e-8)
        assert sfx_rxn[
            model.fs.unit.control_volume.reactions[0].reaction_rate["R2"]
        ] == pytest.approx(1e2, rel=1e-8)
        assert sfx_rxn[
            model.fs.unit.control_volume.reactions[0].reaction_rate["R3"]
        ] == pytest.approx(1e2, rel=1e-8)
        assert sfx_rxn[
            model.fs.unit.control_volume.reactions[0].reaction_rate["R4"]
        ] == pytest.approx(1e2, rel=1e-8)
        assert sfx_rxn[
            model.fs.unit.control_volume.reactions[0].reaction_rate["R5"]
        ] == pytest.approx(1e2, rel=1e-8)
        assert sfx_rxn[
            model.fs.unit.control_volume.reactions[0].reaction_rate["R6"]
        ] == pytest.approx(1e2, rel=1e-8)
        assert sfx_rxn[
            model.fs.unit.control_volume.reactions[0].reaction_rate["R7"]
        ] == pytest.approx(1e2, rel=1e-8)
        assert sfx_rxn[
            model.fs.unit.control_volume.reactions[0].reaction_rate["R8"]
        ] == pytest.approx(1e2, rel=1e-8)
        assert sfx_rxn[
            model.fs.unit.control_volume.reactions[0.0].rate_expression["R1"]
        ] == pytest.approx(1e2, rel=1e-8)
        assert sfx_rxn[
            model.fs.unit.control_volume.reactions[0.0].rate_expression["R2"]
        ] == pytest.approx(1e2, rel=1e-8)
        assert sfx_rxn[
            model.fs.unit.control_volume.reactions[0.0].rate_expression["R3"]
        ] == pytest.approx(1e2, rel=1e-8)
        assert sfx_rxn[
            model.fs.unit.control_volume.reactions[0.0].rate_expression["R4"]
        ] == pytest.approx(1e2, rel=1e-8)
        assert sfx_rxn[
            model.fs.unit.control_volume.reactions[0.0].rate_expression["R5"]
        ] == pytest.approx(1e2, rel=1e-8)
        assert sfx_rxn[
            model.fs.unit.control_volume.reactions[0.0].rate_expression["R6"]
        ] == pytest.approx(1e2, rel=1e-8)
        assert sfx_rxn[
            model.fs.unit.control_volume.reactions[0.0].rate_expression["R7"]
        ] == pytest.approx(1e2, rel=1e-8)
        assert sfx_rxn[
            model.fs.unit.control_volume.reactions[0.0].rate_expression["R8"]
        ] == pytest.approx(1e2, rel=1e-8)

        # Check that unit model has scaling factors
        sfx_cv = model.fs.unit.control_volume.scaling_factor
        assert isinstance(sfx_cv, Suffix)
        assert len(sfx_cv) == 1
        assert sfx_cv[model.fs.unit.control_volume.volume[0]] == pytest.approx(
            1e-3, rel=1e-3
        )

        sfx_unit = model.fs.unit.scaling_factor
        assert isinstance(sfx_unit, Suffix)
        assert len(sfx_unit) == 1
        assert sfx_unit[model.fs.unit.CSTR_retention_time[0]] == pytest.approx(
            0.0012199, rel=1e-8
        )

    # TODO: Remove test once iscale is deprecated
    @pytest.mark.integration
    def test_example_case_iscale(self):
        m = ConcreteModel()
        m.fs = FlowsheetBlock(dynamic=False)

        m.fs.props_ASM1 = ASM1ParameterBlock()
        m.fs.ASM1_rxn_props = ASM1ReactionParameterBlock(
            property_package=m.fs.props_ASM1
        )

        m.fs.unit = CSTR(
            property_package=m.fs.props_ASM1, reaction_package=m.fs.ASM1_rxn_props
        )

        m.fs.unit.inlet.flow_vol[0].fix(1.2199 * units.m**3 / units.s)
        m.fs.unit.inlet.alkalinity[0].fix(4.5102 * units.mole / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "S_I"].fix(0.061909 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "S_S"].fix(0.012366 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "X_I"].fix(1.4258 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "X_S"].fix(0.090508 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "X_BH"].fix(2.8404 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "X_BA"].fix(0.20512 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "X_P"].fix(0.58681 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "S_O"].fix(0.00036092 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "S_NO"].fix(0.012424 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "S_NH"].fix(0.0076936 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "S_ND"].fix(0.0019068 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "X_ND"].fix(0.0053166 * units.kg / units.m**3)

        m.fs.unit.inlet.temperature[0].fix(308.15 * units.K)
        m.fs.unit.inlet.pressure[0].fix(84790.0 * units.Pa)

        m.fs.unit.volume[0].fix(1000 * units.m**3)

        # Set scaling factors for badly scaled variables
        iscale.set_scaling_factor(
            m.fs.unit.control_volume.properties_out[0].pressure, 1e-5
        )
        iscale.set_scaling_factor(
            m.fs.unit.control_volume.properties_out[0.0].conc_mass_comp["S_O"], 1e5
        )
        iscale.set_scaling_factor(
            m.fs.unit.control_volume.rate_reaction_generation[0.0, "Liq", "X_BA"], 1e5
        )
        iscale.set_scaling_factor(
            m.fs.unit.control_volume.rate_reaction_generation[0.0, "Liq", "X_P"], 1e5
        )
        iscale.set_scaling_factor(
            m.fs.unit.control_volume.rate_reaction_generation[0.0, "Liq", "S_O"], 1e5
        )
        iscale.set_scaling_factor(
            m.fs.unit.control_volume.rate_reaction_generation[0.0, "Liq", "S_NH"], 1e5
        )
        iscale.set_scaling_factor(
            m.fs.unit.control_volume.rate_reaction_generation[0.0, "Liq", "S_ND"], 1e5
        )
        iscale.set_scaling_factor(
            m.fs.unit.control_volume.rate_reaction_generation[0.0, "Liq", "X_ND"], 1e5
        )

        iscale.set_scaling_factor(
            m.fs.unit.control_volume.rate_reaction_extent[0.0, "R1"], 1e5
        )
        iscale.set_scaling_factor(
            m.fs.unit.control_volume.rate_reaction_extent[0.0, "R3"], 1e5
        )
        iscale.set_scaling_factor(
            m.fs.unit.control_volume.rate_reaction_extent[0.0, "R5"], 1e5
        )

        iscale.set_scaling_factor(
            m.fs.unit.control_volume.reactions[0.0].reaction_rate["R1"], 1e7
        )
        iscale.set_scaling_factor(
            m.fs.unit.control_volume.reactions[0.0].reaction_rate["R2"], 1e7
        )
        iscale.set_scaling_factor(
            m.fs.unit.control_volume.reactions[0.0].reaction_rate["R3"], 1e7
        )
        iscale.set_scaling_factor(
            m.fs.unit.control_volume.reactions[0.0].reaction_rate["R4"], 1e7
        )
        iscale.set_scaling_factor(
            m.fs.unit.control_volume.reactions[0.0].reaction_rate["R5"], 1e7
        )
        iscale.set_scaling_factor(
            m.fs.unit.control_volume.reactions[0.0].reaction_rate["R6"], 1e7
        )
        iscale.set_scaling_factor(
            m.fs.unit.control_volume.reactions[0.0].reaction_rate["R7"], 1e7
        )
        iscale.set_scaling_factor(
            m.fs.unit.control_volume.reactions[0.0].reaction_rate["R8"], 1e7
        )

        iscale.calculate_scaling_factors(m.fs.unit)

        # Check condition number to confirm scaling
        sm = TransformationFactory("core.scale_model").create_using(m, rename=False)
        jac, _ = get_jacobian(sm, scaled=False)
        assert (jacobian_cond(jac=jac, scaled=False)) == pytest.approx(
            6.1277317e13, rel=1e-3
        )

    @pytest.mark.integration
    def test_example_case_scaler(self):
        m = ConcreteModel()
        m.fs = FlowsheetBlock(dynamic=False)

        m.fs.props_ASM1 = ASM1ParameterBlock()
        m.fs.ASM1_rxn_props = ASM1ReactionParameterBlock(
            property_package=m.fs.props_ASM1
        )

        m.fs.unit = CSTR(
            property_package=m.fs.props_ASM1, reaction_package=m.fs.ASM1_rxn_props
        )

        m.fs.unit.inlet.flow_vol[0].fix(1.2199 * units.m**3 / units.s)
        m.fs.unit.inlet.alkalinity[0].fix(4.5102 * units.mole / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "S_I"].fix(0.061909 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "S_S"].fix(0.012366 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "X_I"].fix(1.4258 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "X_S"].fix(0.090508 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "X_BH"].fix(2.8404 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "X_BA"].fix(0.20512 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "X_P"].fix(0.58681 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "S_O"].fix(0.00036092 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "S_NO"].fix(0.012424 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "S_NH"].fix(0.0076936 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "S_ND"].fix(0.0019068 * units.kg / units.m**3)
        m.fs.unit.inlet.conc_mass_comp[0, "X_ND"].fix(0.0053166 * units.kg / units.m**3)

        m.fs.unit.inlet.temperature[0].fix(308.15 * units.K)
        m.fs.unit.inlet.pressure[0].fix(84790.0 * units.Pa)

        m.fs.unit.volume[0].fix(1000 * units.m**3)

        scaler = CSTRScaler()
        scaler.scale_model(
            m.fs.unit,
            submodel_scalers={
                m.fs.unit.control_volume.properties_in: ASM1PropertiesScaler,
                m.fs.unit.control_volume.properties_out: ASM1PropertiesScaler,
                m.fs.unit.control_volume.reactions: ASM1ReactionScaler,
            },
        )

        # Check condition number to confirm scaling
        sm = TransformationFactory("core.scale_model").create_using(m, rename=False)
        jac, _ = get_jacobian(sm, scaled=False)
        assert (jacobian_cond(jac=jac, scaled=False)) == pytest.approx(
            3.626085e10, rel=1e-3
        )


def build_model():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)

    m.fs.props_ASM1 = ASM1ParameterBlock()
    m.fs.ASM1_rxn_props = ASM1ReactionParameterBlock(property_package=m.fs.props_ASM1)

    m.fs.unit = CSTR(
        property_package=m.fs.props_ASM1, reaction_package=m.fs.ASM1_rxn_props
    )

    m.fs.unit.inlet.flow_vol[0].fix(1.2199 * units.m**3 / units.s)
    m.fs.unit.inlet.alkalinity[0].fix(4.5102 * units.mole / units.m**3)
    m.fs.unit.inlet.conc_mass_comp[0, "S_I"].fix(0.061909 * units.kg / units.m**3)
    m.fs.unit.inlet.conc_mass_comp[0, "S_S"].fix(0.012366 * units.kg / units.m**3)
    m.fs.unit.inlet.conc_mass_comp[0, "X_I"].fix(1.4258 * units.kg / units.m**3)
    m.fs.unit.inlet.conc_mass_comp[0, "X_S"].fix(0.090508 * units.kg / units.m**3)
    m.fs.unit.inlet.conc_mass_comp[0, "X_BH"].fix(2.8404 * units.kg / units.m**3)
    m.fs.unit.inlet.conc_mass_comp[0, "X_BA"].fix(0.20512 * units.kg / units.m**3)
    m.fs.unit.inlet.conc_mass_comp[0, "X_P"].fix(0.58681 * units.kg / units.m**3)
    m.fs.unit.inlet.conc_mass_comp[0, "S_O"].fix(0.00036092 * units.kg / units.m**3)
    m.fs.unit.inlet.conc_mass_comp[0, "S_NO"].fix(0.012424 * units.kg / units.m**3)
    m.fs.unit.inlet.conc_mass_comp[0, "S_NH"].fix(0.0076936 * units.kg / units.m**3)
    m.fs.unit.inlet.conc_mass_comp[0, "S_ND"].fix(0.0019068 * units.kg / units.m**3)
    m.fs.unit.inlet.conc_mass_comp[0, "X_ND"].fix(0.0053166 * units.kg / units.m**3)

    m.fs.unit.inlet.temperature[0].fix(308.15 * units.K)
    m.fs.unit.inlet.pressure[0].fix(84790.0 * units.Pa)

    m.fs.unit.volume[0].fix(1000 * units.m**3)

    solver = get_solver()
    solver.solve(m)

    return m


def scale_vars_with_scalers(m):
    scaler = CSTRScaler()
    scaler.scale_model(
        m.fs.unit,
        submodel_scalers={
            m.fs.unit.control_volume.properties_in: ASM1PropertiesScaler,
            m.fs.unit.control_volume.properties_out: ASM1PropertiesScaler,
            m.fs.unit.control_volume.reactions: ASM1ReactionScaler,
        },
    )


def scale_vars_with_iscale(m):
    # Set scaling factors for badly scaled variables
    iscale.set_scaling_factor(m.fs.unit.control_volume.properties_out[0].pressure, 1e-5)
    iscale.set_scaling_factor(
        m.fs.unit.control_volume.properties_out[0.0].conc_mass_comp["S_O"], 1e5
    )
    iscale.set_scaling_factor(
        m.fs.unit.control_volume.rate_reaction_generation[0.0, "Liq", "X_BA"], 1e5
    )
    iscale.set_scaling_factor(
        m.fs.unit.control_volume.rate_reaction_generation[0.0, "Liq", "X_P"], 1e5
    )
    iscale.set_scaling_factor(
        m.fs.unit.control_volume.rate_reaction_generation[0.0, "Liq", "S_O"], 1e5
    )
    iscale.set_scaling_factor(
        m.fs.unit.control_volume.rate_reaction_generation[0.0, "Liq", "S_NH"], 1e5
    )
    iscale.set_scaling_factor(
        m.fs.unit.control_volume.rate_reaction_generation[0.0, "Liq", "S_ND"], 1e5
    )
    iscale.set_scaling_factor(
        m.fs.unit.control_volume.rate_reaction_generation[0.0, "Liq", "X_ND"], 1e5
    )

    iscale.set_scaling_factor(
        m.fs.unit.control_volume.rate_reaction_extent[0.0, "R1"], 1e5
    )
    iscale.set_scaling_factor(
        m.fs.unit.control_volume.rate_reaction_extent[0.0, "R3"], 1e5
    )
    iscale.set_scaling_factor(
        m.fs.unit.control_volume.rate_reaction_extent[0.0, "R5"], 1e5
    )

    iscale.set_scaling_factor(
        m.fs.unit.control_volume.reactions[0.0].reaction_rate["R1"], 1e7
    )
    iscale.set_scaling_factor(
        m.fs.unit.control_volume.reactions[0.0].reaction_rate["R2"], 1e7
    )
    iscale.set_scaling_factor(
        m.fs.unit.control_volume.reactions[0.0].reaction_rate["R3"], 1e7
    )
    iscale.set_scaling_factor(
        m.fs.unit.control_volume.reactions[0.0].reaction_rate["R4"], 1e7
    )
    iscale.set_scaling_factor(
        m.fs.unit.control_volume.reactions[0.0].reaction_rate["R5"], 1e7
    )
    iscale.set_scaling_factor(
        m.fs.unit.control_volume.reactions[0.0].reaction_rate["R6"], 1e7
    )
    iscale.set_scaling_factor(
        m.fs.unit.control_volume.reactions[0.0].reaction_rate["R7"], 1e7
    )
    iscale.set_scaling_factor(
        m.fs.unit.control_volume.reactions[0.0].reaction_rate["R8"], 1e7
    )

    iscale.calculate_scaling_factors(m.fs.unit)


def perturb_solution(m):
    m.fs.unit.inlet.flow_vol[0].fix(1.2199 * 0.5 * units.m**3 / units.s)
    m.fs.unit.volume[0].fix(1000 * 0.75 * units.m**3)


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
Unscaled                 || 1.196E+12 | Solved 4   ||
Vars Only                || 4.434E+10 | Solved 5   || 1.674E+17 | Solved 1  
Harmonic                 || 1.124E+10 | Solved 5   || 1.161E+05 | Solved 3  
Inverse Sum              || 2.031E+07 | Solved 5   || 2.054E+02 | Solved 3  
Inverse Root Sum Squares || 3.266E+07 | Solved 5   || 2.131E+02 | Solved 3  
Inverse Maximum          || 4.057E+07 | Solved 5   || 2.427E+02 | Solved 3  
Inverse Minimum          || 1.089E+10 | Solved 5   || 1.317E+05 | Solved 3  
Nominal L1 Norm          || 1.799E+04 | Solved 5   || 2.183E+02 | Solved 3  
Nominal L2 Norm          || 1.789E+04 | Solved 5   || 2.051E+02 | Solved 3  
Actual L1 Norm           || 1.548E+04 | Solved 5   || 2.146E+02 | Solved 3  
Actual L2 Norm           || 1.569E+04 | Solved 5   || 2.114E+02 | Solved 3  
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
Unscaled                 || 1.196E+12 | Solved 4   ||
Vars Only                || 3.003E+13 | Solved 3   || 1.674E+17 | Solved 1  
Harmonic                 || 9.790E+09 | Solved 5   || 1.161E+05 | Solved 3  
Inverse Sum              || 2.885E+07 | Solved 5   || 2.054E+02 | Solved 3  
Inverse Root Sum Squares || 3.046E+07 | Solved 5   || 2.131E+02 | Solved 3  
Inverse Maximum          || 3.164E+07 | Solved 5   || 2.427E+02 | Solved 3  
Inverse Minimum          || 1.879E+10 | Solved 5   || 1.317E+05 | Solved 3  
Nominal L1 Norm          || 1.261E+09 | Solved 5   || 2.183E+02 | Solved 3  
Nominal L2 Norm          || 9.543E+08 | Solved 5   || 2.051E+02 | Solved 3  
Actual L1 Norm           || 1.665E+06 | Solved 4   || 2.146E+02 | Solved 3  
Actual L2 Norm           || 1.736E+06 | Solved 4   || 2.114E+02 | Solved 3  
============================================================================
"""

    assert stream.getvalue() == expected
