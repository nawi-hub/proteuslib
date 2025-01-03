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

import re
import pytest

from pyomo.environ import (
    assert_optimal_termination,
    ConcreteModel,
    Constraint,
    Set,
    Suffix,
    TransformationFactory,
    value,
    Var,
)
from pyomo.network import Port

from idaes.core import EnergyBalanceType, FlowsheetBlock, MomentumBalanceType
from idaes.core.initialization import InitializationStatus
from idaes.core.scaling import set_scaling_factor
from idaes.core.solvers import get_solver
from idaes.core.util.exceptions import ConfigurationError
from idaes.core.util.model_diagnostics import DiagnosticsToolbox
from idaes.core.util.testing import PhysicalParameterTestBlock

from watertap.property_models.multicomp_aq_sol_prop_pack import (
    MCASParameterBlock,
)
from watertap.unit_models.nanofiltration_0D import (
    Nanofiltration0D,
    Nanofiltration0DInitializer,
    Nanofiltration0DScaler,
)


class TestBuild:
    @pytest.mark.unit
    def test_config(self):
        m = ConcreteModel()
        m.fs = FlowsheetBlock(dynamic=False)
        m.fs.properties = MCASParameterBlock(
            solute_list=["Na_+", "Cl_-"],
            charge={"Na_+": 1, "Cl_-": -1},
        )
        m.fs.unit = Nanofiltration0D(property_package=m.fs.properties)

        assert len(m.fs.unit.config) == 11
        assert not m.fs.unit.config.dynamic
        assert not m.fs.unit.config.has_holdup
        assert m.fs.unit.config.property_package is m.fs.properties
        assert m.fs.unit.config.property_package_args == {}
        assert (
            m.fs.unit.config.momentum_balance_type == MomentumBalanceType.pressureTotal
        )
        assert not m.fs.unit.config.has_pressure_change
        assert m.fs.unit.config.energy_balance_type == EnergyBalanceType.isothermal
        assert m.fs.unit.config.electroneutrality_ion == "Cl_-"
        assert m.fs.unit.config.passing_species_list is None
        assert m.fs.unit.config.default_passing_rejection == pytest.approx(
            0.1, rel=1e-12
        )
        assert m.fs.unit.config.default_excluded_rejection == pytest.approx(
            1 - 1e-10, rel=1e-12
        )

    @pytest.mark.unit
    def test_default_initializer_and_scaler(self):
        m = ConcreteModel()
        m.fs = FlowsheetBlock(dynamic=False)
        m.fs.properties = MCASParameterBlock(
            solute_list=["Na_+", "Cl_-"],
            charge={"Na_+": 1, "Cl_-": -1},
        )
        m.fs.unit = Nanofiltration0D(property_package=m.fs.properties)

        assert m.fs.unit.default_initializer is Nanofiltration0DInitializer
        assert m.fs.unit.default_scaler is Nanofiltration0DScaler

    @pytest.mark.unit
    def test_multiphase(self):
        m = ConcreteModel()
        m.fs = FlowsheetBlock(dynamic=False)
        m.fs.properties = PhysicalParameterTestBlock()

        with pytest.raises(
            ConfigurationError,
            match="Nanofiltration0D model only supports single phase "
            "property packages.",
        ):
            m.fs.unit = Nanofiltration0D(property_package=m.fs.properties)

    @pytest.mark.unit
    def test_electroneutrality_invalid_component(self):
        m = ConcreteModel()
        m.fs = FlowsheetBlock(dynamic=False)
        m.fs.properties = MCASParameterBlock(
            solute_list=["Na_+", "Cl_-"],
            charge={"Na_+": 1, "Cl_-": -1},
        )

        with pytest.raises(
            ConfigurationError,
            match=re.escape(
                "electroneutrality_ion (foo) "
                "is not a valid component in property package."
            ),
        ):
            m.fs.unit = Nanofiltration0D(
                property_package=m.fs.properties,
                electroneutrality_ion="foo",
            )

    @pytest.mark.unit
    def test_electroneutrality_not_in_passing_list(self):
        m = ConcreteModel()
        m.fs = FlowsheetBlock(dynamic=False)
        m.fs.properties = MCASParameterBlock(
            solute_list=["Na_+", "Cl_-"],
            charge={"Na_+": 1, "Cl_-": -1},
        )

        with pytest.raises(
            ConfigurationError,
            match=re.escape(
                "electroneutrality_ion (Na_+) "
                "must be a member of the passing species list."
            ),
        ):
            m.fs.unit = Nanofiltration0D(
                property_package=m.fs.properties,
                electroneutrality_ion="Na_+",
                passing_species_list=["Cl_-"],
            )

    @pytest.mark.unit
    def test_electroneutrality_divalent(self):
        m = ConcreteModel()
        m.fs = FlowsheetBlock(dynamic=False)
        m.fs.properties = MCASParameterBlock(
            solute_list=["Na_+", "Cl_-"],
            charge={"Na_+": 1, "Cl_-": -2},
        )

        with pytest.raises(
            ConfigurationError,
            match=re.escape("electroneutrality_ion (Cl_-) must be a monovalent ion."),
        ):
            m.fs.unit = Nanofiltration0D(
                property_package=m.fs.properties,
            )

    unsupported_balance_types = [
        MomentumBalanceType.pressurePhase,
        MomentumBalanceType.momentumTotal,
        MomentumBalanceType.momentumPhase,
    ]

    @pytest.mark.unit
    @pytest.mark.parametrize("balance_type", unsupported_balance_types)
    def test_unsupported_momentum_balance(self, balance_type):
        m = ConcreteModel()
        m.fs = FlowsheetBlock(dynamic=False)
        m.fs.properties = MCASParameterBlock(
            solute_list=["Na_+", "Cl_-"],
            charge={"Na_+": 1, "Cl_-": -1},
        )

        with pytest.raises(
            ConfigurationError,
            match=re.escape(
                f"Nanofiltration0D model only supports total pressure balance or no pressure balance "
                f"(assigned {balance_type})"
            ),
        ):
            m.fs.unit = Nanofiltration0D(
                property_package=m.fs.properties,
                momentum_balance_type=balance_type,
            )

    @pytest.mark.unit
    def test_invalid_pressure_change(self):
        m = ConcreteModel()
        m.fs = FlowsheetBlock(dynamic=False)
        m.fs.properties = MCASParameterBlock(
            solute_list=["Na_+", "Cl_-"],
            charge={"Na_+": 1, "Cl_-": -1},
        )

        with pytest.raises(
            ConfigurationError,
            match="Inconsistent configuration arguments. has_pressure_change=True "
            "requires that momentum_balance_type not equal MomentumBalanceType.none.",
        ):
            m.fs.unit = Nanofiltration0D(
                property_package=m.fs.properties,
                momentum_balance_type=MomentumBalanceType.none,
                has_pressure_change=True,
            )

    unsupported_ebalance_types = [
        e
        for e in EnergyBalanceType
        if e not in (EnergyBalanceType.none, EnergyBalanceType.isothermal)
    ]

    @pytest.mark.unit
    @pytest.mark.parametrize("balance_type", unsupported_ebalance_types)
    def test_unsupported_energy_balance(self, balance_type):
        m = ConcreteModel()
        m.fs = FlowsheetBlock(dynamic=False)
        m.fs.properties = MCASParameterBlock(
            solute_list=["Na_+", "Cl_-"],
            charge={"Na_+": 1, "Cl_-": -1},
        )

        with pytest.raises(
            ConfigurationError,
            match=re.escape(
                f"Nanofiltration0D model only supports isothermal operation or no energy balance "
                f"(assigned {balance_type})"
            ),
        ):
            m.fs.unit = Nanofiltration0D(
                property_package=m.fs.properties,
                energy_balance_type=balance_type,
            )

    @pytest.mark.unit
    def test_default_build(self):
        m = ConcreteModel()
        m.fs = FlowsheetBlock(dynamic=False)
        m.fs.properties = MCASParameterBlock(
            solute_list=["Ca_2+", "SO4_2-", "Mg_2+", "Na_+", "Cl_-"],
            charge={"Ca_2+": 2, "Mg_2+": 2, "SO4_2-": -2, "Na_+": 1, "Cl_-": -1},
        )
        m.fs.unit = Nanofiltration0D(property_package=m.fs.properties)

        assert hasattr(m.fs.unit, "properties_in")
        assert hasattr(m.fs.unit, "properties_retentate")
        assert hasattr(m.fs.unit, "properties_permeate")

        assert isinstance(m.fs.unit.inlet, Port)
        assert isinstance(m.fs.unit.retentate, Port)
        assert isinstance(m.fs.unit.permeate, Port)

        assert isinstance(m.fs.unit._solute_set, Set)
        assert len(m.fs.unit._solute_set) == 4
        for j in m.fs.unit._solute_set:
            assert j in ["Na_+", "Ca_2+", "SO4_2-", "Mg_2+"]

        assert isinstance(m.fs.unit.solvent_recovery, Var)
        assert value(m.fs.unit.solvent_recovery[0]) == pytest.approx(0.8, rel=1e-8)

        assert isinstance(m.fs.unit.rejection_comp, Var)
        for (_, j), v in m.fs.unit.rejection_comp.items():
            if j in ["Ca_2+", "SO4_2-", "Mg_2+"]:
                assert value(v) == pytest.approx(
                    m.fs.unit.config.default_excluded_rejection, abs=1e-12
                )
            else:
                assert value(v) == pytest.approx(
                    m.fs.unit.config.default_passing_rejection, abs=1e-12
                )

        assert not hasattr(m.fs.unit, "deltaP")

        assert isinstance(m.fs.unit.material_balances, Constraint)
        assert len(m.fs.unit.material_balances) == 6  # 1 time point, 6 components
        for (t, j), cd in m.fs.unit.material_balances.items():
            assert str(cd.expr) == str(
                m.fs.unit.properties_in[t].flow_mol_phase_comp["Liq", j]
                == m.fs.unit.properties_retentate[t].flow_mol_phase_comp["Liq", j]
                + m.fs.unit.properties_permeate[t].flow_mol_phase_comp["Liq", j]
            )

        assert isinstance(m.fs.unit.separation_constraint, Constraint)
        assert (
            len(m.fs.unit.separation_constraint) == 5
        )  # 1 time point, 5 components (Cl- is skipped)
        for (t, p, j), cd in m.fs.unit.separation_constraint.items():
            assert j != "Cl_-"

            if j == "H2O":
                assert str(cd.expr) == str(
                    m.fs.unit.solvent_recovery[t]
                    * m.fs.unit.properties_in[t].flow_mol_phase_comp[p, j]
                    == m.fs.unit.properties_permeate[t].flow_mol_phase_comp[p, j]
                )
            else:
                assert str(cd.expr) == str(
                    (
                        (1 - m.fs.unit.rejection_comp[t, j])
                        * m.fs.unit.properties_in[t].conc_mol_phase_comp[p, j]
                    )
                    == m.fs.unit.properties_permeate[t].conc_mol_phase_comp[p, j]
                )

        assert isinstance(m.fs.unit.permeate_electronegativity, Constraint)
        assert len(m.fs.unit.permeate_electronegativity) == 1
        assert str(m.fs.unit.permeate_electronegativity[0].expr) == str(
            0
            == sum(
                m.fs.properties.get_component(j).config.charge
                * m.fs.unit.properties_permeate[0].flow_mol_phase_comp["Liq", j]
                for j in m.fs.properties.ion_set
            )
        )

        assert isinstance(m.fs.unit.retentate_pressure_balance, Constraint)
        assert len(m.fs.unit.retentate_pressure_balance) == 1
        assert str(m.fs.unit.retentate_pressure_balance[0].expr) == str(
            m.fs.unit.properties_in[0].pressure
            == m.fs.unit.properties_retentate[0].pressure
        )

        assert isinstance(m.fs.unit.retentate_temperature_equality, Constraint)
        assert len(m.fs.unit.retentate_temperature_equality) == 1
        assert str(m.fs.unit.retentate_temperature_equality[0].expr) == str(
            m.fs.unit.properties_in[0].temperature
            == m.fs.unit.properties_retentate[0].temperature
        )

        assert isinstance(m.fs.unit.permeate_temperature_equality, Constraint)
        assert len(m.fs.unit.permeate_temperature_equality) == 1
        assert str(m.fs.unit.permeate_temperature_equality[0].expr) == str(
            m.fs.unit.properties_in[0].temperature
            == m.fs.unit.properties_permeate[0].temperature
        )

    @pytest.mark.unit
    def test_w_passing_listd(self):
        m = ConcreteModel()
        m.fs = FlowsheetBlock(dynamic=False)
        m.fs.properties = MCASParameterBlock(
            solute_list=["Ca_2+", "SO4_2-", "Mg_2+", "Na_+", "Cl_-"],
            charge={"Ca_2+": 2, "Mg_2+": 2, "SO4_2-": -2, "Na_+": 1, "Cl_-": -1},
        )
        m.fs.unit = Nanofiltration0D(
            property_package=m.fs.properties,
            passing_species_list=["Mg_2+", "Na_+", "Cl_-"],
        )

        assert hasattr(m.fs.unit, "properties_in")
        assert hasattr(m.fs.unit, "properties_retentate")
        assert hasattr(m.fs.unit, "properties_permeate")

        assert isinstance(m.fs.unit.inlet, Port)
        assert isinstance(m.fs.unit.retentate, Port)
        assert isinstance(m.fs.unit.permeate, Port)

        assert isinstance(m.fs.unit._solute_set, Set)
        assert len(m.fs.unit._solute_set) == 4
        for j in m.fs.unit._solute_set:
            assert j in ["Na_+", "Ca_2+", "SO4_2-", "Mg_2+"]

        assert isinstance(m.fs.unit.solvent_recovery, Var)
        assert value(m.fs.unit.solvent_recovery[0]) == pytest.approx(0.8, rel=1e-8)

        assert isinstance(m.fs.unit.rejection_comp, Var)
        for (_, j), v in m.fs.unit.rejection_comp.items():
            if j in ["Ca_2+", "SO4_2-"]:
                assert value(v) == pytest.approx(
                    m.fs.unit.config.default_excluded_rejection, abs=1e-12
                )
            else:
                assert value(v) == pytest.approx(
                    m.fs.unit.config.default_passing_rejection, abs=1e-12
                )

        assert not hasattr(m.fs.unit, "deltaP")

        assert isinstance(m.fs.unit.material_balances, Constraint)
        assert len(m.fs.unit.material_balances) == 6  # 1 time point, 6 components
        for (t, j), cd in m.fs.unit.material_balances.items():
            assert str(cd.expr) == str(
                m.fs.unit.properties_in[t].flow_mol_phase_comp["Liq", j]
                == m.fs.unit.properties_retentate[t].flow_mol_phase_comp["Liq", j]
                + m.fs.unit.properties_permeate[t].flow_mol_phase_comp["Liq", j]
            )

        assert isinstance(m.fs.unit.separation_constraint, Constraint)
        assert (
            len(m.fs.unit.separation_constraint) == 5
        )  # 1 time point, 5 components (Cl- is skipped)
        for (t, p, j), cd in m.fs.unit.separation_constraint.items():
            assert j != "Cl_-"

            if j == "H2O":
                assert str(cd.expr) == str(
                    m.fs.unit.solvent_recovery[t]
                    * m.fs.unit.properties_in[t].flow_mol_phase_comp[p, j]
                    == m.fs.unit.properties_permeate[t].flow_mol_phase_comp[p, j]
                )
            else:
                assert str(cd.expr) == str(
                    (
                        (1 - m.fs.unit.rejection_comp[t, j])
                        * m.fs.unit.properties_in[t].conc_mol_phase_comp[p, j]
                    )
                    == m.fs.unit.properties_permeate[t].conc_mol_phase_comp[p, j]
                )

        assert isinstance(m.fs.unit.permeate_electronegativity, Constraint)
        assert len(m.fs.unit.permeate_electronegativity) == 1
        assert str(m.fs.unit.permeate_electronegativity[0].expr) == str(
            0
            == sum(
                m.fs.properties.get_component(j).config.charge
                * m.fs.unit.properties_permeate[0].flow_mol_phase_comp["Liq", j]
                for j in m.fs.properties.ion_set
            )
        )

        assert isinstance(m.fs.unit.retentate_pressure_balance, Constraint)
        assert len(m.fs.unit.retentate_pressure_balance) == 1
        assert str(m.fs.unit.retentate_pressure_balance[0].expr) == str(
            m.fs.unit.properties_in[0].pressure
            == m.fs.unit.properties_retentate[0].pressure
        )

        assert isinstance(m.fs.unit.retentate_temperature_equality, Constraint)
        assert len(m.fs.unit.retentate_temperature_equality) == 1
        assert str(m.fs.unit.retentate_temperature_equality[0].expr) == str(
            m.fs.unit.properties_in[0].temperature
            == m.fs.unit.properties_retentate[0].temperature
        )

        assert isinstance(m.fs.unit.permeate_temperature_equality, Constraint)
        assert len(m.fs.unit.permeate_temperature_equality) == 1
        assert str(m.fs.unit.permeate_temperature_equality[0].expr) == str(
            m.fs.unit.properties_in[0].temperature
            == m.fs.unit.properties_permeate[0].temperature
        )

    @pytest.mark.unit
    def test_build_w_pressure_drop(self):
        m = ConcreteModel()
        m.fs = FlowsheetBlock(dynamic=False)
        m.fs.properties = MCASParameterBlock(
            solute_list=["Ca_2+", "SO4_2-", "Mg_2+", "Na_+", "Cl_-"],
            charge={"Ca_2+": 2, "Mg_2+": 2, "SO4_2-": -2, "Na_+": 1, "Cl_-": -1},
        )
        m.fs.unit = Nanofiltration0D(
            property_package=m.fs.properties,
            has_pressure_change=True,
        )

        assert hasattr(m.fs.unit, "properties_in")
        assert hasattr(m.fs.unit, "properties_retentate")
        assert hasattr(m.fs.unit, "properties_permeate")

        assert isinstance(m.fs.unit.inlet, Port)
        assert isinstance(m.fs.unit.retentate, Port)
        assert isinstance(m.fs.unit.permeate, Port)

        assert isinstance(m.fs.unit._solute_set, Set)
        assert len(m.fs.unit._solute_set) == 4
        for j in m.fs.unit._solute_set:
            assert j in ["Na_+", "Ca_2+", "SO4_2-", "Mg_2+"]

        assert isinstance(m.fs.unit.solvent_recovery, Var)
        assert value(m.fs.unit.solvent_recovery[0]) == pytest.approx(0.8, rel=1e-8)

        assert isinstance(m.fs.unit.rejection_comp, Var)
        for (_, j), v in m.fs.unit.rejection_comp.items():
            if j in ["Ca_2+", "SO4_2-", "Mg_2+"]:
                assert value(v) == pytest.approx(
                    m.fs.unit.config.default_excluded_rejection, abs=1e-12
                )
            else:
                assert value(v) == pytest.approx(
                    m.fs.unit.config.default_passing_rejection, abs=1e-12
                )

        assert isinstance(m.fs.unit.deltaP, Var)

        assert isinstance(m.fs.unit.material_balances, Constraint)
        assert len(m.fs.unit.material_balances) == 6  # 1 time point, 6 components
        for (t, j), cd in m.fs.unit.material_balances.items():
            assert str(cd.expr) == str(
                m.fs.unit.properties_in[t].flow_mol_phase_comp["Liq", j]
                == m.fs.unit.properties_retentate[t].flow_mol_phase_comp["Liq", j]
                + m.fs.unit.properties_permeate[t].flow_mol_phase_comp["Liq", j]
            )

        assert isinstance(m.fs.unit.separation_constraint, Constraint)
        assert (
            len(m.fs.unit.separation_constraint) == 5
        )  # 1 time point, 5 components (Cl- is skipped)
        for (t, p, j), cd in m.fs.unit.separation_constraint.items():
            assert j != "Cl_-"

            if j == "H2O":
                assert str(cd.expr) == str(
                    m.fs.unit.solvent_recovery[t]
                    * m.fs.unit.properties_in[t].flow_mol_phase_comp[p, j]
                    == m.fs.unit.properties_permeate[t].flow_mol_phase_comp[p, j]
                )
            else:
                assert str(cd.expr) == str(
                    (
                        (1 - m.fs.unit.rejection_comp[t, j])
                        * m.fs.unit.properties_in[t].conc_mol_phase_comp[p, j]
                    )
                    == m.fs.unit.properties_permeate[t].conc_mol_phase_comp[p, j]
                )

        assert isinstance(m.fs.unit.permeate_electronegativity, Constraint)
        assert len(m.fs.unit.permeate_electronegativity) == 1
        assert str(m.fs.unit.permeate_electronegativity[0].expr) == str(
            0
            == sum(
                m.fs.properties.get_component(j).config.charge
                * m.fs.unit.properties_permeate[0].flow_mol_phase_comp["Liq", j]
                for j in m.fs.properties.ion_set
            )
        )

        assert isinstance(m.fs.unit.retentate_pressure_balance, Constraint)
        assert len(m.fs.unit.retentate_pressure_balance) == 1
        assert str(m.fs.unit.retentate_pressure_balance[0].expr) == str(
            m.fs.unit.properties_in[0].pressure
            == m.fs.unit.properties_retentate[0].pressure - m.fs.unit.deltaP[0]
        )

        assert isinstance(m.fs.unit.retentate_temperature_equality, Constraint)
        assert len(m.fs.unit.retentate_temperature_equality) == 1
        assert str(m.fs.unit.retentate_temperature_equality[0].expr) == str(
            m.fs.unit.properties_in[0].temperature
            == m.fs.unit.properties_retentate[0].temperature
        )

        assert isinstance(m.fs.unit.permeate_temperature_equality, Constraint)
        assert len(m.fs.unit.permeate_temperature_equality) == 1
        assert str(m.fs.unit.permeate_temperature_equality[0].expr) == str(
            m.fs.unit.properties_in[0].temperature
            == m.fs.unit.properties_permeate[0].temperature
        )

    @pytest.mark.unit
    def test_build_no_pt_electroneutrality(self):
        m = ConcreteModel()
        m.fs = FlowsheetBlock(dynamic=False)
        m.fs.properties = MCASParameterBlock(
            solute_list=["Ca_2+", "SO4_2-", "Mg_2+", "Na_+", "Cl_-"],
            charge={"Ca_2+": 2, "Mg_2+": 2, "SO4_2-": -2, "Na_+": 1, "Cl_-": -1},
        )
        m.fs.unit = Nanofiltration0D(
            property_package=m.fs.properties,
            electroneutrality_ion=None,
            momentum_balance_type=MomentumBalanceType.none,
            energy_balance_type=EnergyBalanceType.none,
        )

        assert hasattr(m.fs.unit, "properties_in")
        assert hasattr(m.fs.unit, "properties_retentate")
        assert hasattr(m.fs.unit, "properties_permeate")

        assert isinstance(m.fs.unit.inlet, Port)
        assert isinstance(m.fs.unit.retentate, Port)
        assert isinstance(m.fs.unit.permeate, Port)

        assert isinstance(m.fs.unit._solute_set, Set)
        assert len(m.fs.unit._solute_set) == 5
        for j in m.fs.unit._solute_set:
            assert j in ["Na_+", "Ca_2+", "SO4_2-", "Mg_2+", "Cl_-"]

        assert isinstance(m.fs.unit.solvent_recovery, Var)
        assert value(m.fs.unit.solvent_recovery[0]) == pytest.approx(0.8, rel=1e-8)

        assert isinstance(m.fs.unit.rejection_comp, Var)
        for (_, j), v in m.fs.unit.rejection_comp.items():
            if j in ["Ca_2+", "SO4_2-", "Mg_2+"]:
                assert value(v) == pytest.approx(
                    m.fs.unit.config.default_excluded_rejection, abs=1e-12
                )
            else:
                assert value(v) == pytest.approx(
                    m.fs.unit.config.default_passing_rejection, abs=1e-12
                )

        assert not hasattr(m.fs.unit, "deltaP")

        assert isinstance(m.fs.unit.material_balances, Constraint)
        assert len(m.fs.unit.material_balances) == 6  # 1 time point, 6 components
        for (t, j), cd in m.fs.unit.material_balances.items():
            assert str(cd.expr) == str(
                m.fs.unit.properties_in[t].flow_mol_phase_comp["Liq", j]
                == m.fs.unit.properties_retentate[t].flow_mol_phase_comp["Liq", j]
                + m.fs.unit.properties_permeate[t].flow_mol_phase_comp["Liq", j]
            )

        assert isinstance(m.fs.unit.separation_constraint, Constraint)
        assert len(m.fs.unit.separation_constraint) == 6  # 1 time point, 6 components
        for (t, p, j), cd in m.fs.unit.separation_constraint.items():
            if j == "H2O":
                assert str(cd.expr) == str(
                    m.fs.unit.solvent_recovery[t]
                    * m.fs.unit.properties_in[t].flow_mol_phase_comp[p, j]
                    == m.fs.unit.properties_permeate[t].flow_mol_phase_comp[p, j]
                )
            else:
                assert str(cd.expr) == str(
                    (
                        (1 - m.fs.unit.rejection_comp[t, j])
                        * m.fs.unit.properties_in[t].conc_mol_phase_comp[p, j]
                    )
                    == m.fs.unit.properties_permeate[t].conc_mol_phase_comp[p, j]
                )

        assert not hasattr(m.fs.unit, "permeate_electronegativity")
        assert not hasattr(m.fs.unit, "retentate_pressure_balance")
        assert not hasattr(m.fs.unit, "retentate_temperature_equality")
        assert not hasattr(m.fs.unit, "permeate_temperature_equality")


@pytest.fixture
def mcas_model():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.properties = MCASParameterBlock(
        solute_list=["Ca_2+", "SO4_2-", "Mg_2+", "Na_+", "Cl_-"],
        diffusivity_data={
            ("Liq", "Ca_2+"): 9.2e-10,
            ("Liq", "SO4_2-"): 1.06e-09,
            ("Liq", "Mg_2+"): 7.06e-10,
            ("Liq", "Na_+"): 1.33e-09,
            ("Liq", "Cl_-"): 2.03e-09,
        },
        mw_data={
            "H2O": 0.018,
            "Ca_2+": 0.04,
            "Mg_2+": 0.024,
            "SO4_2-": 0.096,
            "Na_+": 0.023,
            "Cl_-": 0.035,
        },
        stokes_radius_data={
            "Ca_2+": 3.09e-10,
            "Mg_2+": 3.47e-10,
            "SO4_2-": 2.3e-10,
            "Cl_-": 1.21e-10,
            "Na_+": 1.84e-10,
        },
        charge={"Ca_2+": 2, "Mg_2+": 2, "SO4_2-": -2, "Na_+": 1, "Cl_-": -1},
    )

    m.fs.unit = Nanofiltration0D(
        property_package=m.fs.properties,
        has_pressure_change=True,
    )

    return m


class TestNanofiltration0DInitializer:
    @pytest.mark.component
    def test_initialize(self, mcas_model):
        # Fix DoF
        mcas_model.fs.unit.inlet.flow_mol_phase_comp[0, "Liq", "H2O"].fix(53.6036)
        mcas_model.fs.unit.inlet.flow_mol_phase_comp[0, "Liq", "Ca_2+"].fix(0.00955)
        mcas_model.fs.unit.inlet.flow_mol_phase_comp[0, "Liq", "Mg_2+"].fix(0.5808)
        mcas_model.fs.unit.inlet.flow_mol_phase_comp[0, "Liq", "SO4_2-"].fix(0.02225)
        mcas_model.fs.unit.inlet.flow_mol_phase_comp[0, "Liq", "Cl_-"].fix(1.61977)
        mcas_model.fs.unit.inlet.flow_mol_phase_comp[0, "Liq", "Na_+"].fix(0.48357)
        mcas_model.fs.unit.inlet.temperature[0].fix(298.15)
        mcas_model.fs.unit.inlet.pressure[0].fix(4e5)

        mcas_model.fs.unit.solvent_recovery.fix()
        mcas_model.fs.unit.rejection_comp.fix()
        mcas_model.fs.unit.permeate.pressure[0].fix(1e5)
        mcas_model.fs.unit.deltaP.fix(-1e4)

        # Initialize
        initializer = Nanofiltration0DInitializer()

        initializer.initialize(mcas_model.fs.unit)

        assert (
            initializer.summary[mcas_model.fs.unit]["status"] == InitializationStatus.Ok
        )

        for c in mcas_model.component_data_objects(Constraint, descend_into=True):
            assert (
                abs(value(c.body - c.lower)) <= 1e-5
                and abs(value(c.upper - c.body)) <= 1e-5
            )

    @pytest.mark.unit
    def test_init_conc(self, mcas_model):
        # Use MCAS model to dummy calls to initializer sub methods
        initializer = Nanofiltration0DInitializer()

        mcas_model.in_var = Var(mcas_model.fs.properties.component_list, initialize=10)
        mcas_model.ret_var = Var(mcas_model.fs.properties.component_list, initialize=0)
        mcas_model.perm_var = Var(mcas_model.fs.properties.component_list, initialize=0)

        initializer._init_conc(
            mcas_model.fs.unit,
            0,  # time index
            mcas_model.fs.unit.properties_in[0],
            mcas_model.in_var,
            mcas_model.ret_var,
            mcas_model.perm_var,
        )

        assert value(mcas_model.ret_var["H2O"]) == pytest.approx(
            10, rel=1e-10
        )  # solvent concentration unchanged
        assert value(mcas_model.ret_var["Cl_-"]) == pytest.approx(
            2, rel=1e-10
        )  # should use solvent recovery
        assert value(mcas_model.ret_var["Na_+"]) == pytest.approx(1, rel=1e-10)
        assert value(mcas_model.ret_var["Ca_2+"]) == pytest.approx(
            10 * (1 - 1e-10), rel=1e-10
        )
        assert value(mcas_model.ret_var["Mg_2+"]) == pytest.approx(
            10 * (1 - 1e-10), rel=1e-10
        )
        assert value(mcas_model.ret_var["SO4_2-"]) == pytest.approx(
            10 * (1 - 1e-10), rel=1e-10
        )

        assert value(mcas_model.perm_var["H2O"]) == pytest.approx(
            10, rel=1e-10
        )  # solvent concentration unchanged
        assert value(mcas_model.perm_var["Cl_-"]) == pytest.approx(
            8, rel=1e-10
        )  # should use solvent recovery
        assert value(mcas_model.perm_var["Na_+"]) == pytest.approx(9, rel=1e-10)
        assert value(mcas_model.perm_var["Ca_2+"]) == pytest.approx(
            10 * 1e-10, rel=1e-10
        )
        assert value(mcas_model.perm_var["Mg_2+"]) == pytest.approx(
            10 * 1e-10, rel=1e-10
        )
        assert value(mcas_model.perm_var["SO4_2-"]) == pytest.approx(
            10 * 1e-10, rel=1e-10
        )

    @pytest.mark.unit
    def test_init_flow_comp(self, mcas_model):
        # Use MCA model to dummy calls to initializer sub methods
        initializer = Nanofiltration0DInitializer()

        mcas_model.in_var_comp = Var(
            mcas_model.fs.properties.component_list, initialize=10
        )
        mcas_model.ret_var = Var(mcas_model.fs.properties.component_list, initialize=0)
        mcas_model.perm_var = Var(mcas_model.fs.properties.component_list, initialize=0)

        initializer._init_flow(
            mcas_model.fs.unit,
            0,  # time index
            mcas_model.fs.unit.properties_in[0],
            mcas_model.in_var_comp,
            mcas_model.ret_var,
            mcas_model.perm_var,
        )

        assert value(mcas_model.ret_var["H2O"]) == pytest.approx(2, rel=1e-10)
        assert value(mcas_model.ret_var["Cl_-"]) == pytest.approx(
            2, rel=1e-10
        )  # should use solvent recovery
        assert value(mcas_model.ret_var["Na_+"]) == pytest.approx(1, rel=1e-10)
        assert value(mcas_model.ret_var["Ca_2+"]) == pytest.approx(
            10 * (1 - 1e-10), rel=1e-10
        )
        assert value(mcas_model.ret_var["Mg_2+"]) == pytest.approx(
            10 * (1 - 1e-10), rel=1e-10
        )
        assert value(mcas_model.ret_var["SO4_2-"]) == pytest.approx(
            10 * (1 - 1e-10), rel=1e-10
        )

        assert value(mcas_model.perm_var["H2O"]) == pytest.approx(8, rel=1e-10)
        assert value(mcas_model.perm_var["Cl_-"]) == pytest.approx(
            8, rel=1e-10
        )  # should use solvent recovery
        assert value(mcas_model.perm_var["Na_+"]) == pytest.approx(9, rel=1e-10)
        assert value(mcas_model.perm_var["Ca_2+"]) == pytest.approx(
            10 * 1e-10, rel=1e-10
        )
        assert value(mcas_model.perm_var["Mg_2+"]) == pytest.approx(
            10 * 1e-10, rel=1e-10
        )
        assert value(mcas_model.perm_var["SO4_2-"]) == pytest.approx(
            10 * 1e-10, rel=1e-10
        )

    @pytest.mark.unit
    def test_init_flow_vol(self, mcas_model):
        # Use MCA model to dummy calls to initializer sub methods
        initializer = Nanofiltration0DInitializer()

        mcas_model.in_var_vol = Var(initialize=10)
        mcas_model.ret_var = Var(initialize=0)
        mcas_model.perm_var = Var(initialize=0)

        initializer._init_flow(
            mcas_model.fs.unit,
            0,  # time index
            mcas_model.fs.unit.properties_in[0],
            mcas_model.in_var_vol,
            mcas_model.ret_var,
            mcas_model.perm_var,
        )

        assert value(mcas_model.ret_var) == pytest.approx(2, rel=1e-10)
        assert value(mcas_model.perm_var) == pytest.approx(8, rel=1e-10)

    @pytest.mark.unit
    def test_init_frac(self, mcas_model):
        # Use MCA model to dummy calls to initializer sub methods
        initializer = Nanofiltration0DInitializer()

        mcas_model.in_var = Var(mcas_model.fs.properties.component_list, initialize=10)
        mcas_model.ret_var = Var(mcas_model.fs.properties.component_list, initialize=0)
        mcas_model.perm_var = Var(mcas_model.fs.properties.component_list, initialize=0)

        initializer._init_frac(
            mcas_model.fs.unit,
            0,  # time index
            mcas_model.fs.unit.properties_in[0],
            mcas_model.in_var,
            mcas_model.ret_var,
            mcas_model.perm_var,
        )

        assert value(mcas_model.ret_var["H2O"]) == pytest.approx(2 / 35, rel=1e-6)
        assert value(mcas_model.ret_var["Cl_-"]) == pytest.approx(2 / 35, rel=1e-6)
        assert value(mcas_model.ret_var["Na_+"]) == pytest.approx(1 / 35, rel=1e-6)
        assert value(mcas_model.ret_var["Ca_2+"]) == pytest.approx(10 / 35, rel=1e-6)
        assert value(mcas_model.ret_var["Mg_2+"]) == pytest.approx(10 / 35, rel=1e-6)
        assert value(mcas_model.ret_var["SO4_2-"]) == pytest.approx(10 / 35, rel=1e-6)

        assert value(mcas_model.perm_var["H2O"]) == pytest.approx(8 / 25, rel=1e-6)
        assert value(mcas_model.perm_var["Cl_-"]) == pytest.approx(8 / 25, rel=1e-6)
        assert value(mcas_model.perm_var["Na_+"]) == pytest.approx(9 / 25, rel=1e-6)
        assert value(mcas_model.perm_var["Ca_2+"]) == pytest.approx(0, abs=1e-6)
        assert value(mcas_model.perm_var["Mg_2+"]) == pytest.approx(0, abs=1e-6)
        assert value(mcas_model.perm_var["SO4_2-"]) == pytest.approx(0, abs=1e-6)


class TestNanofiltration0DScaler:
    @pytest.mark.unit
    def test_variable_scaling_routine(self, mcas_model):
        scaler = Nanofiltration0DScaler()

        scaler.variable_scaling_routine(mcas_model.fs.unit)

        # Check that submodels have scaling factors
        assert isinstance(mcas_model.fs.unit.properties_in[0].scaling_factor, Suffix)
        assert len(mcas_model.fs.unit.properties_in[0].scaling_factor) > 1

        assert isinstance(
            mcas_model.fs.unit.properties_retentate[0].scaling_factor, Suffix
        )
        assert len(mcas_model.fs.unit.properties_retentate[0].scaling_factor) > 1

        assert isinstance(
            mcas_model.fs.unit.properties_permeate[0].scaling_factor, Suffix
        )
        assert len(mcas_model.fs.unit.properties_permeate[0].scaling_factor) > 1

        # Check scaling factors for unit variables
        assert isinstance(mcas_model.fs.unit.scaling_factor, Suffix)
        assert len(mcas_model.fs.unit.scaling_factor) == 6

        assert mcas_model.fs.unit.scaling_factor[
            mcas_model.fs.unit.deltaP[0]
        ] == pytest.approx(1e-4, rel=1e-8)

        for j in ["Ca_2+", "Mg_2+", "SO4_2-", "Na_+"]:
            assert (
                mcas_model.fs.unit.scaling_factor[
                    mcas_model.fs.unit.rejection_comp[0, j]
                ]
                == 1e2
            )
        assert (
            mcas_model.fs.unit.scaling_factor[mcas_model.fs.unit.solvent_recovery[0]]
            == 10
        )

    @pytest.mark.unit
    def test_constraint_scaling_routine(self, mcas_model):
        scaler = Nanofiltration0DScaler()

        scaler.constraint_scaling_routine(mcas_model.fs.unit)

        # Check that submodels have scaling factors
        # No constraint scaling factors applied ot sub-models, so no check for length of Suffix
        assert isinstance(mcas_model.fs.unit.properties_in[0].scaling_factor, Suffix)
        assert isinstance(
            mcas_model.fs.unit.properties_retentate[0].scaling_factor, Suffix
        )
        assert isinstance(
            mcas_model.fs.unit.properties_permeate[0].scaling_factor, Suffix
        )

        # Check scaling factors for unit variables
        assert isinstance(mcas_model.fs.unit.scaling_factor, Suffix)
        assert len(mcas_model.fs.unit.scaling_factor) == 15

        assert (
            mcas_model.fs.unit.scaling_factor[
                mcas_model.fs.unit.material_balances[0.0, "Ca_2+"]
            ]
            == 10.0
        )
        assert (
            mcas_model.fs.unit.scaling_factor[
                mcas_model.fs.unit.material_balances[0.0, "Cl_-"]
            ]
            == 10.0
        )
        assert (
            mcas_model.fs.unit.scaling_factor[
                mcas_model.fs.unit.material_balances[0.0, "H2O"]
            ]
            == 10.0
        )
        assert (
            mcas_model.fs.unit.scaling_factor[
                mcas_model.fs.unit.material_balances[0.0, "Mg_2+"]
            ]
            == 10.0
        )
        assert (
            mcas_model.fs.unit.scaling_factor[
                mcas_model.fs.unit.material_balances[0.0, "Na_+"]
            ]
            == 10.0
        )
        assert (
            mcas_model.fs.unit.scaling_factor[
                mcas_model.fs.unit.material_balances[0.0, "SO4_2-"]
            ]
            == 10.0
        )
        assert (
            mcas_model.fs.unit.scaling_factor[
                mcas_model.fs.unit.permeate_electronegativity[0.0]
            ]
            == 5.0
        )
        assert mcas_model.fs.unit.scaling_factor[
            mcas_model.fs.unit.permeate_temperature_equality[0.0]
        ] == pytest.approx(3.354016e-3, rel=1e-5)

        assert (
            mcas_model.fs.unit.scaling_factor[
                mcas_model.fs.unit.separation_constraint[0.0, "Liq", "Ca_2+"]
            ]
            == 0.002
        )
        assert (
            mcas_model.fs.unit.scaling_factor[
                mcas_model.fs.unit.separation_constraint[0.0, "Liq", "H2O"]
            ]
            == 10.0
        )
        assert (
            mcas_model.fs.unit.scaling_factor[
                mcas_model.fs.unit.separation_constraint[0.0, "Liq", "Mg_2+"]
            ]
            == 0.002
        )
        assert (
            mcas_model.fs.unit.scaling_factor[
                mcas_model.fs.unit.separation_constraint[0.0, "Liq", "Na_+"]
            ]
            == 0.002
        )
        assert (
            mcas_model.fs.unit.scaling_factor[
                mcas_model.fs.unit.separation_constraint[0.0, "Liq", "SO4_2-"]
            ]
            == 0.002
        )
        assert mcas_model.fs.unit.scaling_factor[
            mcas_model.fs.unit.retentate_pressure_balance[0.0]
        ] == pytest.approx(9.869233e-06, rel=1e-5)
        assert mcas_model.fs.unit.scaling_factor[
            mcas_model.fs.unit.retentate_temperature_equality[0.0]
        ] == pytest.approx(3.354016e-3, rel=1e-5)


class TestMCAS:
    @pytest.fixture(scope="class")
    def mcas_case(self):
        m = ConcreteModel()
        m.fs = FlowsheetBlock(dynamic=False)
        m.fs.properties = MCASParameterBlock(
            solute_list=["Ca_2+", "SO4_2-", "Mg_2+", "Na_+", "Cl_-"],
            diffusivity_data={
                ("Liq", "Ca_2+"): 9.2e-10,
                ("Liq", "SO4_2-"): 1.06e-09,
                ("Liq", "Mg_2+"): 7.06e-10,
                ("Liq", "Na_+"): 1.33e-09,
                ("Liq", "Cl_-"): 2.03e-09,
            },
            mw_data={
                "H2O": 0.018,
                "Ca_2+": 0.04,
                "Mg_2+": 0.024,
                "SO4_2-": 0.096,
                "Na_+": 0.023,
                "Cl_-": 0.035,
            },
            stokes_radius_data={
                "Ca_2+": 3.09e-10,
                "Mg_2+": 3.47e-10,
                "SO4_2-": 2.3e-10,
                "Cl_-": 1.21e-10,
                "Na_+": 1.84e-10,
            },
            charge={"Ca_2+": 2, "Mg_2+": 2, "SO4_2-": -2, "Na_+": 1, "Cl_-": -1},
        )

        m.fs.unit = Nanofiltration0D(
            property_package=m.fs.properties,
            has_pressure_change=True,
        )

        # Fix other inlet state variables
        m.fs.unit.inlet.flow_mol_phase_comp[0, "Liq", "H2O"].fix(53.6036)
        m.fs.unit.inlet.flow_mol_phase_comp[0, "Liq", "Ca_2+"].fix(0.00955)
        m.fs.unit.inlet.flow_mol_phase_comp[0, "Liq", "Mg_2+"].fix(0.5808)
        m.fs.unit.inlet.flow_mol_phase_comp[0, "Liq", "SO4_2-"].fix(0.02225)
        m.fs.unit.inlet.flow_mol_phase_comp[0, "Liq", "Cl_-"].fix(1.61977)
        m.fs.unit.inlet.flow_mol_phase_comp[0, "Liq", "Na_+"].fix(0.48357)
        m.fs.unit.inlet.temperature[0].fix(298.15)
        m.fs.unit.inlet.pressure[0].fix(4e5)

        m.fs.unit.solvent_recovery.fix()
        m.fs.unit.rejection_comp.fix()
        m.fs.unit.deltaP.fix(-1e4)
        m.fs.unit.permeate.pressure[0].fix(101325)

        return m

    @pytest.mark.component
    def test_structural_issues(self, mcas_case):
        dt = DiagnosticsToolbox(mcas_case)

        dt.assert_no_structural_warnings(ignore_evaluation_errors=True)

    @pytest.mark.component
    def test_scaling(self, mcas_case):
        # Scale model
        set_scaling_factor(
            mcas_case.fs.unit.inlet.flow_mol_phase_comp[0, "Liq", "H2O"], 1e-1
        )
        set_scaling_factor(
            mcas_case.fs.unit.inlet.flow_mol_phase_comp[0, "Liq", "Ca_2+"], 1e3
        )
        set_scaling_factor(
            mcas_case.fs.unit.inlet.flow_mol_phase_comp[0, "Liq", "Mg_2+"], 10
        )
        set_scaling_factor(
            mcas_case.fs.unit.inlet.flow_mol_phase_comp[0, "Liq", "SO4_2-"], 1e2
        )
        set_scaling_factor(
            mcas_case.fs.unit.inlet.flow_mol_phase_comp[0, "Liq", "Cl_-"], 1
        )
        set_scaling_factor(
            mcas_case.fs.unit.inlet.flow_mol_phase_comp[0, "Liq", "Na_+"], 10
        )

        scaler = Nanofiltration0DScaler()
        scaler.scale_model(mcas_case.fs.unit)

        assert (
            mcas_case.fs.unit.scaling_factor[mcas_case.fs.unit.solvent_recovery[0]]
            == 10
        )
        for j in ["Ca_2+", "Mg_2+", "SO4_2-", "Na_+"]:
            assert (
                mcas_case.fs.unit.scaling_factor[mcas_case.fs.unit.rejection_comp[0, j]]
                == 100
            )

        assert mcas_case.fs.unit.scaling_factor[
            mcas_case.fs.unit.deltaP[0.0]
        ] == pytest.approx(0.0001, rel=1e-5)

        assert mcas_case.fs.unit.scaling_factor[
            mcas_case.fs.unit.material_balances[0.0, "H2O"]
        ] == pytest.approx(0.01866, rel=1e-3)
        assert mcas_case.fs.unit.scaling_factor[
            mcas_case.fs.unit.material_balances[0.0, "Ca_2+"]
        ] == pytest.approx(104.7, rel=1e-3)
        assert mcas_case.fs.unit.scaling_factor[
            mcas_case.fs.unit.material_balances[0.0, "SO4_2-"]
        ] == pytest.approx(44.94, rel=1e-3)
        assert mcas_case.fs.unit.scaling_factor[
            mcas_case.fs.unit.material_balances[0.0, "Mg_2+"]
        ] == pytest.approx(1.722, rel=1e-3)
        assert mcas_case.fs.unit.scaling_factor[
            mcas_case.fs.unit.material_balances[0.0, "Na_+"]
        ] == pytest.approx(2.068, rel=1e-3)
        assert mcas_case.fs.unit.scaling_factor[
            mcas_case.fs.unit.material_balances[0.0, "Cl_-"]
        ] == pytest.approx(0.6174, rel=1e-3)

        assert mcas_case.fs.unit.scaling_factor[
            mcas_case.fs.unit.separation_constraint[0.0, "Liq", "H2O"]
        ] == pytest.approx(0.0233193, rel=1e-5)
        assert mcas_case.fs.unit.scaling_factor[
            mcas_case.fs.unit.separation_constraint[0.0, "Liq", "Ca_2+"]
        ] == pytest.approx(0.004, rel=1e-5)
        assert mcas_case.fs.unit.scaling_factor[
            mcas_case.fs.unit.separation_constraint[0.0, "Liq", "SO4_2-"]
        ] == pytest.approx(0.00960, rel=1e-5)
        assert mcas_case.fs.unit.scaling_factor[
            mcas_case.fs.unit.separation_constraint[0.0, "Liq", "Mg_2+"]
        ] == pytest.approx(0.00240, rel=1e-5)
        assert mcas_case.fs.unit.scaling_factor[
            mcas_case.fs.unit.separation_constraint[0.0, "Liq", "Na_+"]
        ] == pytest.approx(0.0023, rel=1e-3)

        assert (
            mcas_case.fs.unit.scaling_factor[
                mcas_case.fs.unit.permeate_electronegativity[0.0]
            ]
            == 1
        )
        assert mcas_case.fs.unit.scaling_factor[
            mcas_case.fs.unit.retentate_pressure_balance[0.0]
        ] == pytest.approx(0.0000025, rel=1e-3)
        assert mcas_case.fs.unit.scaling_factor[
            mcas_case.fs.unit.retentate_temperature_equality[0.0]
        ] == pytest.approx(0.003354, rel=1e-3)
        assert mcas_case.fs.unit.scaling_factor[
            mcas_case.fs.unit.permeate_temperature_equality[0.0]
        ] == pytest.approx(0.003354, rel=1e-3)

    @pytest.mark.component
    def test_initialize(self, mcas_case):
        initializer = Nanofiltration0DInitializer()
        initializer.initialize(mcas_case.fs.unit)

        assert (
            initializer.summary[mcas_case.fs.unit]["status"] == InitializationStatus.Ok
        )

    @pytest.mark.component
    def test_solve(self, mcas_case):
        solver = get_solver(
            "ipopt_v2", writer_config={"scale_model": True, "linear_presolve": True}
        )
        results = solver.solve(mcas_case)

        assert_optimal_termination(results)

    @pytest.mark.component
    def test_numerical_issues(self, mcas_case):
        sm = TransformationFactory("core.scale_model").create_using(
            mcas_case, rename=False
        )

        dt = DiagnosticsToolbox(sm.fs.unit)
        dt.assert_no_numerical_warnings()

    @pytest.mark.component
    def test_solution(self, mcas_case):
        # Retentate stream
        assert value(
            mcas_case.fs.unit.retentate.flow_mol_phase_comp[0, "Liq", "H2O"]
        ) == pytest.approx(0.2 * 53.6036, rel=1e-5)
        assert value(
            mcas_case.fs.unit.retentate.flow_mol_phase_comp[0, "Liq", "Ca_2+"]
        ) == pytest.approx(0.00955, rel=1e-5)
        assert value(
            mcas_case.fs.unit.retentate.flow_mol_phase_comp[0, "Liq", "Mg_2+"]
        ) == pytest.approx(0.5808, rel=1e-5)
        assert value(
            mcas_case.fs.unit.retentate.flow_mol_phase_comp[0, "Liq", "SO4_2-"]
        ) == pytest.approx(0.02225, rel=1e-5)
        assert value(
            mcas_case.fs.unit.retentate.flow_mol_phase_comp[0, "Liq", "Cl_-"]
        ) == pytest.approx(1.29167, rel=1e-5)
        assert value(
            mcas_case.fs.unit.retentate.flow_mol_phase_comp[0, "Liq", "Na_+"]
        ) == pytest.approx(0.155472, rel=1e-5)
        assert value(mcas_case.fs.unit.retentate.temperature[0]) == pytest.approx(
            298.15, rel=1e-5
        )
        assert value(mcas_case.fs.unit.retentate.pressure[0]) == pytest.approx(
            3.9e5, rel=1e-5
        )

        # Permeate stream
        assert value(
            mcas_case.fs.unit.permeate.flow_mol_phase_comp[0, "Liq", "H2O"]
        ) == pytest.approx(0.8 * 53.6036, rel=1e-5)
        assert value(
            mcas_case.fs.unit.permeate.flow_mol_phase_comp[0, "Liq", "Ca_2+"]
        ) == pytest.approx(0, abs=1e-7)
        assert value(
            mcas_case.fs.unit.permeate.flow_mol_phase_comp[0, "Liq", "Mg_2+"]
        ) == pytest.approx(0, abs=1e-7)
        assert value(
            mcas_case.fs.unit.permeate.flow_mol_phase_comp[0, "Liq", "SO4_2-"]
        ) == pytest.approx(0, abs=1e-7)
        assert value(
            mcas_case.fs.unit.permeate.flow_mol_phase_comp[0, "Liq", "Cl_-"]
        ) == pytest.approx(0.328098, rel=1e-5)
        assert value(
            mcas_case.fs.unit.permeate.flow_mol_phase_comp[0, "Liq", "Na_+"]
        ) == pytest.approx(0.328098, rel=1e-5)
        assert value(mcas_case.fs.unit.permeate.temperature[0]) == pytest.approx(
            298.15, rel=1e-5
        )
        assert value(mcas_case.fs.unit.permeate.pressure[0]) == pytest.approx(
            101325, rel=1e-5
        )

        # Rejection
        for j in ["Ca_2+", "Mg_2+", "SO4_2-"]:
            assert value(
                mcas_case.fs.unit.properties_permeate[0].conc_mol_phase_comp["Liq", j]
                / mcas_case.fs.unit.properties_in[0].conc_mol_phase_comp["Liq", j]
            ) == pytest.approx(0, abs=1e-8)
        assert value(
            mcas_case.fs.unit.properties_permeate[0].conc_mol_phase_comp["Liq", "Na_+"]
            / mcas_case.fs.unit.properties_in[0].conc_mol_phase_comp["Liq", "Na_+"]
        ) == pytest.approx(0.9, abs=1e-8)

    @pytest.mark.component
    def test_conservation(self, mcas_case):
        for j in mcas_case.fs.properties.component_list:
            assert (
                abs(
                    value(
                        mcas_case.fs.unit.inlet.flow_mol_phase_comp[0, "Liq", j]
                        - mcas_case.fs.unit.retentate.flow_mol_phase_comp[0, "Liq", j]
                        - mcas_case.fs.unit.permeate.flow_mol_phase_comp[0, "Liq", j]
                    )
                )
                <= 1e-6
            )
