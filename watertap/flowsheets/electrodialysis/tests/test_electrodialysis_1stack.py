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

import pytest
from pyomo.environ import (
    ConcreteModel,
    value,
    Var,
    Objective,
    Block,
)
from pyomo.network import Port

from idaes.core import (
    FlowsheetBlock,
    MaterialBalanceType,
    MomentumBalanceType,
)
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.models.unit_models import Feed, Product, Separator
from watertap.unit_models.electrodialysis_1D import (
    ElectricalOperationMode,
    Electrodialysis1D,
)

from watertap.property_models.multicomp_aq_sol_prop_pack import MCASParameterBlock

import watertap.flowsheets.electrodialysis.electrodialysis_1stack as edfs

__author__ = "Xiangyu Bi"


class TestElectrodialysisVoltageConst:
    @pytest.fixture(scope="class")
    def electrodialysis_1D1stack(self):
        m = edfs.build()
        return m

    @pytest.mark.unit
    def test_build_model(self, electrodialysis_1D1stack):
        m = electrodialysis_1D1stack

        # Test basic build
        assert isinstance(m, ConcreteModel)
        assert isinstance(m.fs, FlowsheetBlock)
        assert isinstance(m.fs.properties, MCASParameterBlock)
        assert isinstance(m.fs.costing, Block)
        assert isinstance(m.fs.feed, Feed)
        assert isinstance(m.fs.separator, Separator)
        assert isinstance(m.fs.EDstack, Electrodialysis1D)
        assert isinstance(m.fs.product, Product)
        assert isinstance(m.fs.disposal, Product)

        # Test port
        assert isinstance(m.fs.feed.outlet, Port)
        assert isinstance(m.fs.separator.inlet, Port)
        assert isinstance(m.fs.separator.inlet_diluate, Port)
        assert isinstance(m.fs.separator.inlet_concentrate, Port)
        assert isinstance(m.fs.EDstack.inlet_diluate, Port)
        assert isinstance(m.fs.EDstack.inlet_concentrate, Port)
        assert isinstance(m.fs.product.inlet, Port)
        assert isinstance(m.fs.disposal.inlet, Port)

        # Test consting
        assert isinstance(m.fs.EDstack.costing, Block)
        assert isinstance(m.fs.EDstack.costing.capital_cost, Var)
        assert isinstance(m.fs.EDstack.costing.fixed_operating_cost, Var)

        var_str_list = [
            "total_capital_cost",
            "total_operating_cost",
        ]
        for var_str in var_str_list:
            var = getattr(m.fs.costing, var_str)
            assert isinstance(var, Var)

        # Test arcs
        arc_dict = {
            m.fs.s01: (m.fs.feed.outlet, m.fs.separator.inlet),
            m.fs.s02: (m.fs.separator.inlet_diluate, m.fs.EDstack.inlet_diluate),
            m.fs.s03: (
                m.fs.separator.inlet_concentrate,
                m.fs.EDstack.inlet_concentrate,
            ),
            m.fs.s04: (m.fs.EDstack.outlet_diluate, m.fs.product.inlet),
            m.fs.s05: (m.fs.EDstack.outlet_concentrate, m.fs.disposal.inlet),
        }
        for arc, port_tpl in arc_dict.items():
            assert arc.source is port_tpl[0]
            assert arc.destination is port_tpl[1]

        # Test the primary EDstack properties
        # test configrations
        assert len(m.fs.EDstack.config) == 21
        assert not m.fs.EDstack.config.dynamic
        assert not m.fs.EDstack.config.has_holdup
        assert (
            m.fs.EDstack.config.operation_mode
            == ElectricalOperationMode.Constant_Voltage
        )
        assert (
            m.fs.EDstack.config.material_balance_type == MaterialBalanceType.useDefault
        )
        assert (
            m.fs.EDstack.config.momentum_balance_type
            == MomentumBalanceType.pressureTotal
        )
        assert m.fs.EDstack.config.property_package is m.fs.properties
        assert "H2O" in m.fs.properties.component_list

    @pytest.mark.component
    def test_specific_operating_conditions(self, electrodialysis_1D1stack):
        m = electrodialysis_1D1stack
        edfs.set_operating_conditions(m)
        assert degrees_of_freedom(m) == 0
        edfs.initialize_system(m)
        edfs.solve(m)
        assert value(m.fs.feed.properties[0].flow_vol_phase["Liq"]) == pytest.approx(
            8.7e-5, abs=1e-6
        )
        assert value(m.fs.product.properties[0].flow_vol_phase["Liq"]) == pytest.approx(
            4.2e-5, abs=1e-6
        )
        assert value(
            m.fs.disposal.properties[0].flow_vol_phase["Liq"]
        ) == pytest.approx(4.5e-5, abs=1e-6)
        assert value(
            sum(
                m.fs.feed.properties[0].conc_mass_phase_comp["Liq", j]
                for j in m.fs.properties.ion_set
            )
        ) == pytest.approx(9.895, rel=1e-3)
        assert value(
            sum(
                m.fs.product.properties[0].conc_mass_phase_comp["Liq", j]
                for j in m.fs.properties.ion_set
            )
        ) == pytest.approx(4.479, rel=1e-3)
        assert value(
            sum(
                m.fs.disposal.properties[0].conc_mass_phase_comp["Liq", j]
                for j in m.fs.properties.ion_set
            )
        ) == pytest.approx(14.942, rel=1e-3)

        assert value(m.fs.EDstack.recovery_mass_H2O[0]) == pytest.approx(
            0.485, rel=1e-3
        )
        assert value(
            m.fs.EDstack.cell_width
            * m.fs.EDstack.cell_length
            * m.fs.EDstack.cell_pair_num
        ) == pytest.approx(7.900, rel=1e-3)
        assert value(m.fs.EDstack.voltage_applied[0]) == 5
        assert value(m.fs.costing.specific_energy_consumption) == pytest.approx(
            0.1973, rel=1e-3
        )
        assert value(m.fs.costing.LCOW) == pytest.approx(1.1142, rel=1e-3)

    @pytest.mark.component
    def test_optimization(self, electrodialysis_1D1stack):
        m = electrodialysis_1D1stack
        edfs.initialize_system(m)
        edfs.optimize_system(m)
        isinstance(m.fs.objective, Objective)
        assert m.fs.objective.expr == m.fs.costing.LCOW
        assert degrees_of_freedom(m) == 1

        assert value(m.fs.feed.properties[0].flow_vol_phase["Liq"]) == pytest.approx(
            8.7e-5, abs=1e-6
        )
        assert value(m.fs.product.properties[0].flow_vol_phase["Liq"]) == pytest.approx(
            4.2e-5, abs=1e-6
        )
        assert value(
            m.fs.disposal.properties[0].flow_vol_phase["Liq"]
        ) == pytest.approx(4.5e-5, abs=1e-6)
        assert value(m.fs.product_salinity) == pytest.approx(0.9996, rel=1e-3)
        assert value(m.fs.disposal_salinity) == pytest.approx(18.1124, rel=1e-3)

        assert value(m.fs.EDstack.recovery_mass_H2O[0]) == pytest.approx(
            0.4846, rel=1e-3
        )
        assert value(m.fs.mem_area) == pytest.approx(1.0980, rel=1e-2)
        assert value(m.fs.EDstack.voltage_applied[0]) == pytest.approx(7.0325, rel=1e-3)
        assert value(m.fs.costing.specific_energy_consumption) == pytest.approx(
            2.3062, rel=1e-3
        )
        assert value(m.fs.costing.LCOW) == pytest.approx(0.42546, rel=1e-3)

    @pytest.mark.unit
    def test_main_fun(self, electrodialysis_1D1stack):
        edfs.main()
