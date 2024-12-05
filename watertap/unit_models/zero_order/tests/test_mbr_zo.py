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
Tests for zero-order membrane bioreactor model
"""
import pytest


from pyomo.environ import (
    Block,
    ConcreteModel,
    Constraint,
    value,
    Var,
    assert_optimal_termination,
)
from pyomo.util.check_units import assert_units_consistent

from idaes.core import FlowsheetBlock
from watertap.core.solvers import get_solver
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util.testing import initialization_tester
from idaes.core import UnitModelCostingBlock

from watertap.unit_models.zero_order import MBRZO
from watertap.core.wt_database import Database
from watertap.core.zero_order_properties import WaterParameterBlock
from watertap.costing.zero_order_costing import ZeroOrderCosting

solver = get_solver()


class TestMBRZOdefault:
    @pytest.fixture(scope="class")
    def model(self):
        m = ConcreteModel()
        m.db = Database()

        m.fs = FlowsheetBlock(dynamic=False)
        m.fs.params = WaterParameterBlock(
            solute_list=[
                "tss",
                "nonvolatile_toc",
                "toc",
                "eeq",
                "viruses_enteric",
                "total_coliforms_fecal_ecoli",
                "cryptosporidium",
            ]
        )

        m.fs.unit = MBRZO(property_package=m.fs.params, database=m.db)

        m.fs.unit.inlet.flow_mass_comp[0, "H2O"].fix(1000)
        m.fs.unit.inlet.flow_mass_comp[0, "tss"].fix(1)
        m.fs.unit.inlet.flow_mass_comp[0, "nonvolatile_toc"].fix(1)
        m.fs.unit.inlet.flow_mass_comp[0, "toc"].fix(1)
        m.fs.unit.inlet.flow_mass_comp[0, "eeq"].fix(1)
        m.fs.unit.inlet.flow_mass_comp[0, "viruses_enteric"].fix(1)
        m.fs.unit.inlet.flow_mass_comp[0, "total_coliforms_fecal_ecoli"].fix(1)
        m.fs.unit.inlet.flow_mass_comp[0, "cryptosporidium"].fix(1)

        return m

    @pytest.mark.unit
    def test_build(self, model):
        assert model.fs.unit.config.database == model.db

        assert isinstance(model.fs.unit.elec_coeff_1, Var)
        assert isinstance(model.fs.unit.elec_coeff_2, Var)
        assert isinstance(model.fs.unit.electricity, Var)
        assert isinstance(model.fs.unit.electricity_intensity, Var)
        assert isinstance(model.fs.unit.electricity_intensity_constraint, Constraint)
        assert isinstance(model.fs.unit.electricity_constraint, Constraint)

    @pytest.mark.component
    def test_load_parameters(self, model):
        data = model.db.get_unit_operation_parameters("mbr")

        model.fs.unit.load_parameters_from_database()

        assert model.fs.unit.recovery_frac_mass_H2O[0].fixed
        assert (
            model.fs.unit.recovery_frac_mass_H2O[0].value
            == data["recovery_frac_mass_H2O"]["value"]
        )

        for (t, j), v in model.fs.unit.removal_frac_mass_comp.items():
            assert v.fixed
            assert v.value == data["removal_frac_mass_comp"][j]["value"]

        assert model.fs.unit.elec_coeff_1.fixed
        assert model.fs.unit.elec_coeff_1.value == data["elec_coeff_1"]["value"]

        assert model.fs.unit.elec_coeff_2.fixed
        assert model.fs.unit.elec_coeff_2.value == data["elec_coeff_2"]["value"]

    @pytest.mark.component
    def test_degrees_of_freedom(self, model):
        assert degrees_of_freedom(model.fs.unit) == 0

    @pytest.mark.component
    def test_unit_consistency(self, model):
        assert_units_consistent(model.fs.unit)

    @pytest.mark.component
    def test_initialize(self, model):
        initialization_tester(model)

    @pytest.mark.solver
    @pytest.mark.skipif(solver is None, reason="Solver not available")
    @pytest.mark.component
    def test_solve(self, model):
        results = solver.solve(model)

        # Check for optimal solution
        assert_optimal_termination(results)

    @pytest.mark.solver
    @pytest.mark.skipif(solver is None, reason="Solver not available")
    @pytest.mark.component
    def test_solution(self, model):
        assert pytest.approx(1.007, rel=1e-5) == value(
            model.fs.unit.properties_in[0].flow_vol
        )
        assert pytest.approx(0.99305, rel=1e-5) == value(
            model.fs.unit.properties_in[0].conc_mass_comp["tss"]
        )
        assert pytest.approx(0.99305, rel=1e-5) == value(
            model.fs.unit.properties_in[0].conc_mass_comp["nonvolatile_toc"]
        )
        assert pytest.approx(0.99305, rel=1e-5) == value(
            model.fs.unit.properties_in[0].conc_mass_comp["toc"]
        )
        assert pytest.approx(0.99305, rel=1e-5) == value(
            model.fs.unit.properties_in[0].conc_mass_comp["eeq"]
        )
        assert pytest.approx(0.99305, rel=1e-5) == value(
            model.fs.unit.properties_in[0].conc_mass_comp["viruses_enteric"]
        )
        assert pytest.approx(0.99305, rel=1e-5) == value(
            model.fs.unit.properties_in[0].conc_mass_comp["total_coliforms_fecal_ecoli"]
        )
        assert pytest.approx(0.99305, rel=1e-5) == value(
            model.fs.unit.properties_in[0].conc_mass_comp["cryptosporidium"]
        )
        assert pytest.approx(1.001222762, rel=1e-5) == value(
            model.fs.unit.properties_treated[0].flow_vol
        )
        assert pytest.approx(0.49939, rel=1e-5) == value(
            model.fs.unit.properties_treated[0].conc_mass_comp["tss"]
        )
        assert pytest.approx(0.39951, rel=1e-5) == value(
            model.fs.unit.properties_treated[0].conc_mass_comp["nonvolatile_toc"]
        )
        assert pytest.approx(0.29110405, rel=1e-5) == value(
            model.fs.unit.properties_treated[0].conc_mass_comp["toc"]
        )
        assert pytest.approx(0.11985345, rel=1e-5) == value(
            model.fs.unit.properties_treated[0].conc_mass_comp["eeq"]
        )
        assert pytest.approx(0.0099878, rel=1e-5) == value(
            model.fs.unit.properties_treated[0].conc_mass_comp["viruses_enteric"]
        )
        assert pytest.approx(0.001200532, rel=1e-5) == value(
            model.fs.unit.properties_treated[0].conc_mass_comp[
                "total_coliforms_fecal_ecoli"
            ]
        )
        assert pytest.approx(9.9878e-05, rel=1e-5) == value(
            model.fs.unit.properties_treated[0].conc_mass_comp["cryptosporidium"]
        )
        assert pytest.approx(0.0057772, rel=1e-5) == value(
            model.fs.unit.properties_byproduct[0].flow_vol
        )
        assert pytest.approx(86.547, rel=1e-5) == value(
            model.fs.unit.properties_byproduct[0].conc_mass_comp["tss"]
        )
        assert pytest.approx(103.8558564, rel=1e-5) == value(
            model.fs.unit.properties_byproduct[0].conc_mass_comp["nonvolatile_toc"]
        )
        assert pytest.approx(122.6433808, rel=1e-5) == value(
            model.fs.unit.properties_byproduct[0].conc_mass_comp["toc"]
        )
        assert pytest.approx(152.3219227, rel=1e-5) == value(
            model.fs.unit.properties_byproduct[0].conc_mass_comp["eeq"]
        )
        assert pytest.approx(171.36216303, rel=1e-5) == value(
            model.fs.unit.properties_byproduct[0].conc_mass_comp["viruses_enteric"]
        )
        assert pytest.approx(172.8850361, rel=1e-5) == value(
            model.fs.unit.properties_byproduct[0].conc_mass_comp[
                "total_coliforms_fecal_ecoli"
            ]
        )
        assert pytest.approx(173.0757847, rel=1e-5) == value(
            model.fs.unit.properties_byproduct[0].conc_mass_comp["cryptosporidium"]
        )
        assert pytest.approx(2946.15766, abs=1e-5) == value(
            model.fs.unit.electricity[0]
        )
        assert pytest.approx(0.812688, abs=1e-5) == value(
            model.fs.unit.electricity_intensity[0]
        )

    @pytest.mark.solver
    @pytest.mark.skipif(solver is None, reason="Solver not available")
    @pytest.mark.component
    def test_conservation(self, model):
        for j in model.fs.params.component_list:
            assert 1e-6 >= abs(
                value(
                    model.fs.unit.inlet.flow_mass_comp[0, j]
                    - model.fs.unit.treated.flow_mass_comp[0, j]
                    - model.fs.unit.byproduct.flow_mass_comp[0, j]
                )
            )

    @pytest.mark.component
    def test_report(self, model):

        model.fs.unit.report()


class TestMBRZO_w_default_removal:
    @pytest.fixture(scope="class")
    def model(self):
        m = ConcreteModel()
        m.db = Database()

        m.fs = FlowsheetBlock(dynamic=False)
        m.fs.params = WaterParameterBlock(
            solute_list=[
                "tss",
                "nonvolatile_toc",
                "toc",
                "eeq",
                "viruses_enteric",
                "total_coliforms_fecal_ecoli",
                "cryptosporidium",
                "foo",
            ]
        )

        m.fs.unit = MBRZO(property_package=m.fs.params, database=m.db)

        m.fs.unit.inlet.flow_mass_comp[0, "H2O"].fix(1000)
        m.fs.unit.inlet.flow_mass_comp[0, "tss"].fix(1)
        m.fs.unit.inlet.flow_mass_comp[0, "nonvolatile_toc"].fix(1)
        m.fs.unit.inlet.flow_mass_comp[0, "toc"].fix(1)
        m.fs.unit.inlet.flow_mass_comp[0, "eeq"].fix(1)
        m.fs.unit.inlet.flow_mass_comp[0, "viruses_enteric"].fix(1)
        m.fs.unit.inlet.flow_mass_comp[0, "total_coliforms_fecal_ecoli"].fix(1)
        m.fs.unit.inlet.flow_mass_comp[0, "cryptosporidium"].fix(1)
        m.fs.unit.inlet.flow_mass_comp[0, "foo"].fix(1)

        return m

    @pytest.mark.unit
    def test_build(self, model):
        assert model.fs.unit.config.database == model.db

        assert isinstance(model.fs.unit.elec_coeff_1, Var)
        assert isinstance(model.fs.unit.elec_coeff_2, Var)
        assert isinstance(model.fs.unit.electricity, Var)
        assert isinstance(model.fs.unit.electricity_intensity, Var)
        assert isinstance(model.fs.unit.electricity_intensity_constraint, Constraint)
        assert isinstance(model.fs.unit.electricity_constraint, Constraint)

    @pytest.mark.component
    def test_load_parameters(self, model):
        data = model.db.get_unit_operation_parameters("mbr")

        model.fs.unit.load_parameters_from_database(use_default_removal=True)

        assert model.fs.unit.recovery_frac_mass_H2O[0].fixed
        assert (
            model.fs.unit.recovery_frac_mass_H2O[0].value
            == data["recovery_frac_mass_H2O"]["value"]
        )

        for (t, j), v in model.fs.unit.removal_frac_mass_comp.items():
            assert v.fixed
            if j == "foo":
                assert v.value == data["default_removal_frac_mass_comp"]["value"]
            else:
                assert v.value == data["removal_frac_mass_comp"][j]["value"]

        assert model.fs.unit.elec_coeff_1.fixed
        assert model.fs.unit.elec_coeff_1.value == data["elec_coeff_1"]["value"]

        assert model.fs.unit.elec_coeff_2.fixed
        assert model.fs.unit.elec_coeff_2.value == data["elec_coeff_2"]["value"]

    @pytest.mark.component
    def test_degrees_of_freedom(self, model):
        assert degrees_of_freedom(model.fs.unit) == 0

    @pytest.mark.component
    def test_unit_consistency(self, model):
        assert_units_consistent(model.fs.unit)

    @pytest.mark.component
    def test_initialize(self, model):
        initialization_tester(model)

    @pytest.mark.solver
    @pytest.mark.skipif(solver is None, reason="Solver not available")
    @pytest.mark.component
    def test_solve(self, model):
        results = solver.solve(model)

        # Check for optimal solution
        assert_optimal_termination(results)

    @pytest.mark.solver
    @pytest.mark.skipif(solver is None, reason="Solver not available")
    @pytest.mark.component
    def test_solution(self, model):
        assert pytest.approx(1.008, rel=1e-5) == value(
            model.fs.unit.properties_in[0].flow_vol
        )
        assert pytest.approx(0.99206, rel=1e-5) == value(
            model.fs.unit.properties_in[0].conc_mass_comp["tss"]
        )
        assert pytest.approx(0.99206, rel=1e-5) == value(
            model.fs.unit.properties_in[0].conc_mass_comp["nonvolatile_toc"]
        )
        assert pytest.approx(0.99206, rel=1e-5) == value(
            model.fs.unit.properties_in[0].conc_mass_comp["toc"]
        )
        assert pytest.approx(0.99206, rel=1e-5) == value(
            model.fs.unit.properties_in[0].conc_mass_comp["eeq"]
        )
        assert pytest.approx(0.99206, rel=1e-5) == value(
            model.fs.unit.properties_in[0].conc_mass_comp["viruses_enteric"]
        )
        assert pytest.approx(0.99206, rel=1e-5) == value(
            model.fs.unit.properties_in[0].conc_mass_comp["total_coliforms_fecal_ecoli"]
        )
        assert pytest.approx(0.99206, rel=1e-5) == value(
            model.fs.unit.properties_in[0].conc_mass_comp["cryptosporidium"]
        )
        assert pytest.approx(0.99206, rel=1e-5) == value(
            model.fs.unit.properties_in[0].conc_mass_comp["foo"]
        )
        assert pytest.approx(1.002222762018698, rel=1e-5) == value(
            model.fs.unit.properties_treated[0].flow_vol
        )
        assert pytest.approx(0.49889, rel=1e-5) == value(
            model.fs.unit.properties_treated[0].conc_mass_comp["tss"]
        )
        assert pytest.approx(0.39911, rel=1e-5) == value(
            model.fs.unit.properties_treated[0].conc_mass_comp["nonvolatile_toc"]
        )
        assert pytest.approx(0.290813, rel=1e-5) == value(
            model.fs.unit.properties_treated[0].conc_mass_comp["toc"]
        )
        assert pytest.approx(0.119734, rel=1e-5) == value(
            model.fs.unit.properties_treated[0].conc_mass_comp["eeq"]
        )
        assert pytest.approx(0.00997782, rel=1e-5) == value(
            model.fs.unit.properties_treated[0].conc_mass_comp["viruses_enteric"]
        )
        assert pytest.approx(0.00119934, rel=1e-5) == value(
            model.fs.unit.properties_treated[0].conc_mass_comp[
                "total_coliforms_fecal_ecoli"
            ]
        )
        assert pytest.approx(9.9788e-05, abs=1e-7) == value(
            model.fs.unit.properties_treated[0].conc_mass_comp["cryptosporidium"]
        )
        assert pytest.approx(0.99778, rel=1e-5) == value(
            model.fs.unit.properties_treated[0].conc_mass_comp["foo"]
        )
        assert pytest.approx(0.0057772, rel=1e-5) == value(
            model.fs.unit.properties_byproduct[0].flow_vol
        )
        assert pytest.approx(86.547, rel=1e-5) == value(
            model.fs.unit.properties_byproduct[0].conc_mass_comp["tss"]
        )
        assert pytest.approx(103.8558564, rel=1e-5) == value(
            model.fs.unit.properties_byproduct[0].conc_mass_comp["nonvolatile_toc"]
        )
        assert pytest.approx(122.6433808, rel=1e-5) == value(
            model.fs.unit.properties_byproduct[0].conc_mass_comp["toc"]
        )
        assert pytest.approx(152.3219227, rel=1e-5) == value(
            model.fs.unit.properties_byproduct[0].conc_mass_comp["eeq"]
        )
        assert pytest.approx(171.36216303, rel=1e-5) == value(
            model.fs.unit.properties_byproduct[0].conc_mass_comp["viruses_enteric"]
        )
        assert pytest.approx(172.8850361, rel=1e-5) == value(
            model.fs.unit.properties_byproduct[0].conc_mass_comp[
                "total_coliforms_fecal_ecoli"
            ]
        )
        assert pytest.approx(173.0757847, rel=1e-5) == value(
            model.fs.unit.properties_byproduct[0].conc_mass_comp["cryptosporidium"]
        )
        assert pytest.approx(1.73093e-6, abs=1e-5) == value(
            model.fs.unit.properties_byproduct[0].conc_mass_comp["foo"]
        )
        assert pytest.approx(2948.205338, abs=1e-5) == value(
            model.fs.unit.electricity[0]
        )
        assert pytest.approx(0.812446, abs=1e-5) == value(
            model.fs.unit.electricity_intensity[0]
        )

    @pytest.mark.solver
    @pytest.mark.skipif(solver is None, reason="Solver not available")
    @pytest.mark.component
    def test_conservation(self, model):
        for j in model.fs.params.component_list:
            assert 1e-6 >= abs(
                value(
                    model.fs.unit.inlet.flow_mass_comp[0, j]
                    - model.fs.unit.treated.flow_mass_comp[0, j]
                    - model.fs.unit.byproduct.flow_mass_comp[0, j]
                )
            )

    @pytest.mark.component
    def test_report(self, model):

        model.fs.unit.report()


db = Database()
params = db._get_technology("mbr")


class TestMBRZOsubtype:
    @pytest.fixture(scope="class")
    def model(self):
        m = ConcreteModel()

        m.fs = FlowsheetBlock(dynamic=False)
        m.fs.params = WaterParameterBlock(
            solute_list=[
                "tss",
                "nonvolatile_toc",
                "toc",
                "eeq",
                "viruses_enteric",
                "total_coliforms_fecal_ecoli",
                "cryptosporidium",
            ]
        )

        m.fs.unit = MBRZO(property_package=m.fs.params, database=db)

        return m

    @pytest.mark.parametrize("subtype", [k for k in params.keys()])
    @pytest.mark.component
    def test_load_parameters(self, model, subtype):
        model.fs.unit.config.process_subtype = subtype
        data = db.get_unit_operation_parameters("mbr", subtype=subtype)

        model.fs.unit.load_parameters_from_database()

        assert model.fs.unit.recovery_frac_mass_H2O[0].fixed
        assert (
            model.fs.unit.recovery_frac_mass_H2O[0].value
            == data["recovery_frac_mass_H2O"]["value"]
        )

        for (t, j), v in model.fs.unit.removal_frac_mass_comp.items():
            assert v.fixed
            assert v.value == data["removal_frac_mass_comp"][j]["value"]

        assert model.fs.unit.elec_coeff_1.fixed
        assert model.fs.unit.elec_coeff_1.value == data["elec_coeff_1"]["value"]

        assert model.fs.unit.elec_coeff_2.fixed
        assert model.fs.unit.elec_coeff_2.value == data["elec_coeff_2"]["value"]


@pytest.mark.parametrize("subtype", [k for k in params.keys()])
def test_costing(subtype):
    m = ConcreteModel()
    m.db = Database()

    m.fs = FlowsheetBlock(dynamic=False)

    m.fs.params = WaterParameterBlock(solute_list=["sulfur", "toc", "tds"])

    m.fs.costing = ZeroOrderCosting()

    m.fs.unit1 = MBRZO(
        property_package=m.fs.params, database=m.db, process_subtype=subtype
    )

    m.fs.unit1.inlet.flow_mass_comp[0, "H2O"].fix(10000)
    m.fs.unit1.inlet.flow_mass_comp[0, "sulfur"].fix(1)
    m.fs.unit1.inlet.flow_mass_comp[0, "toc"].fix(2)
    m.fs.unit1.inlet.flow_mass_comp[0, "tds"].fix(3)
    m.fs.unit1.load_parameters_from_database(use_default_removal=True)
    assert degrees_of_freedom(m.fs.unit1) == 0

    m.fs.unit1.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)

    assert isinstance(m.fs.costing.mbr, Block)
    assert isinstance(m.fs.costing.mbr.capital_a_parameter, Var)
    assert isinstance(m.fs.costing.mbr.capital_b_parameter, Var)
    assert isinstance(m.fs.unit1.costing.capital_cost, Var)
    assert isinstance(m.fs.unit1.costing.capital_cost_constraint, Constraint)

    assert_units_consistent(m.fs)
    assert degrees_of_freedom(m.fs.unit1) == 0

    assert m.fs.unit1.electricity[0] in m.fs.costing._registered_flows["electricity"]
