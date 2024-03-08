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

import pytest
import abc

from pyomo.environ import Block, assert_optimal_termination, ComponentMap, value
from pyomo.util.check_units import assert_units_consistent
from idaes.core import (
    declare_process_block_class,
    UnitModelBlockData,
)
from idaes.core.util.model_statistics import (
    degrees_of_freedom,
)
from idaes.core.solvers import get_solver


# -----------------------------------------------------------------------------
@declare_process_block_class("DummyUnitModel")
class DummyUnitModelData(UnitModelBlockData):
    def build(self):
        super().build()


# -----------------------------------------------------------------------------
class UnitAttributeError(AttributeError):
    """
    WaterTAP exception for generic attribute errors arising from unit model testing.
    """


class UnitValueError(ValueError):
    """
    WaterTAP exception for generic value errors arising from unit model testing.
    """


class UnitRuntimeError(RuntimeError):
    """
    WaterTAP exception for generic runtime errors arising from unit model testing.
    """


class UnitCostingTestHarness(abc.ABC):
    def configure_class(self):
        # string for solver, if None use WaterTAP default
        self.solver = None
        # dictionary for solver options, if None use WaterTAP default
        self.optarg = None

        # solution map from var to value
        self.cost_solutions = ComponentMap()

        # arguments for badly scaled variables
        self.default_large = 1e4
        self.default_small = 1e-3
        self.default_zero = 1e-10

        # arguments for solver tolerance
        self.default_absolute_tolerance = 1e-12
        self.default_relative_tolerance = 1e-6

        model = self.configure()
        if not hasattr(self, "unit_model_costing_block"):
            if model.find_component("fs.unit") is None:
                raise RuntimeError(
                    f"The {self.__class__.__name__}.configure method should name "
                    f"the unit model `fs.unit`."
                )
            self.unit_model_costing_block = model.find_component("fs.unit.costing")
            if self.unit_model_costing_block is None:
                raise RuntimeError(
                    f"The {self.__class__.__name__}.configure method should either "
                    "set the attribute `unit_model_costing_block` or name it `fs.unit.costing`."
                )
        # keep the model so it does not get garbage collected
        self._model = model
        blk = self.unit_model_costing_block

        # attaching objects to model to carry through in pytest frame
        # TODO: Consider removing these objects and directly calling self
        assert not hasattr(blk, "_test_objs")
        blk._test_objs = Block()
        blk._test_objs.solver = self.solver
        blk._test_objs.optarg = self.optarg
        blk._test_objs.cost_solutions = self.cost_solutions

    @abc.abstractmethod
    def configure(self):
        """
        Placeholder method to allow user to setup test harness.

        The configure method must set the attributes:
        cost_solutions: ComponentMap of values for the specified variables

        The costing and unit model tested should be named `fs.unit`, or this method
        should set the attribute `unit_model_block`.

        Returns:
            model: the top-level Pyomo model
        """

    @pytest.fixture(scope="class")
    def frame(self):
        self.configure_class()
        return self._model, self.unit_model_costing_block

    @pytest.mark.unit
    def test_units_consistent(self, frame):
        m, unit_model_costing = frame
        assert_units_consistent(unit_model_costing)

    @pytest.mark.unit
    def test_dof(self, frame):
        m, unit_model_costing = frame
        if degrees_of_freedom(unit_model_costing) != 0:
            raise UnitAttributeError(
                "The unit has {dof} degrees of freedom when 0 is required."
                "".format(dof=degrees_of_freedom(unit_model_costing))
            )

    @pytest.mark.component
    def test_initialization(self, frame):
        m, unit_model_costing = frame
        m.fs.unit.costing.initialize()

    @pytest.mark.component
    def test_cost_solutions(self, frame):
        self.configure_class()
        m, blk = frame
        solutions = blk._test_objs.cost_solutions

        # solve unit
        if blk._test_objs.solver is None:
            opt = get_solver()
        else:
            opt = get_solver(
                solver=blk._test_objs.solver, options=blk._test_objs.optarg
            )
        results = opt.solve(blk)

        # check solve
        assert_optimal_termination(results)

        # check results
        for var, val in solutions.items():
            comp_obj = None
            try:
                val = float(val)
            except:
                # expect the same API as pytest.approx
                comp_obj = val
                val = comp_obj.expected
            if comp_obj is None:
                comp_obj = pytest.approx(
                    val,
                    abs=self.default_absolute_tolerance,
                    rel=self.default_relative_tolerance,
                )
            if not comp_obj == value(var):
                raise AssertionError(f"{var}: Expected {val}, got {value(var)} instead")
