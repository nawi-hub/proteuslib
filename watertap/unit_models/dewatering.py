###############################################################################
# WaterTAP Copyright (c) 2021, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory, Oak Ridge National
# Laboratory, National Renewable Energy Laboratory, and National Energy
# Technology Laboratory (subject to receipt of any required approvals from
# the U.S. Dept. of Energy). All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. These files are also available online at the URL
# "https://github.com/watertap-org/watertap/"
#
###############################################################################
"""
Dewatering unit model for BSM2. Based on IDAES separator unit

Model based on 

J. Alex, L. Benedetti, J.B. Copp, K.V. Gernaey, U. Jeppsson,
I. Nopens, M.N. Pons, C. Rosen, J.P. Steyer and
P. A. Vanrolleghem
Benchmark Simulation Model no. 2 (BSM2)
"""

from enum import Enum
from pandas import DataFrame

# Import IDAES cores
from idaes.core import (
    declare_process_block_class,
    MaterialBalanceType,
)
from idaes.models.unit_models.separator import SeparatorData, SplittingType

from idaes.core.util.tables import create_stream_table_dataframe
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.solvers import get_solver
import idaes.core.util.scaling as iscale
import idaes.logger as idaeslog

from pyomo.environ import (
    Constraint,
    Param,
    Block,
    value,
    units as pyunits,
    check_optimal_termination,
    Set,
)

from idaes.core.util.exceptions import (
    ConfigurationError,
    PropertyNotSupportedError,
    InitializationError,
)

__author__ = "Alejandro Garciadiego"


# Set up logger
_log = idaeslog.getLogger(__name__)


@declare_process_block_class("DewateringUnit")
class DewateringData(SeparatorData):
    """
    Dewatering unit block for BSM2
    """

    CONFIG = SeparatorData.CONFIG()
    CONFIG.outlet_list = ["underflow", "overflow"]
    CONFIG.split_basis = SplittingType.componentFlow

    def build(self):
        """
        Begin building model.
        Args:
            None
        Returns:
            None
        """

        # Call UnitModel.build to setup dynamics
        super(DewateringData, self).build()

        if "underflow" and "overflow" not in self.config.outlet_list:
            raise ConfigurationError(
                "{} encountered unrecognised "
                "outlet_list. This should not "
                "occur - please use overflow "
                "and underflow as outlets.".format(self.name)
            )

        self.p_dewat = Param(
            initialize=0.28,
            units=pyunits.dimensionless,
            mutable=True,
            doc="Percentage of suspended solids in the underflow",
        )

        self.TSS_rem = Param(
            initialize=0.98,
            units=pyunits.dimensionless,
            mutable=True,
            doc="Percentage of suspended solids removed",
        )

        @self.Expression(self.flowsheet().time, doc="Suspended solid concentration")
        def TSS(blk, t):
            return 0.75 * (
                blk.inlet.conc_mass_comp[0, "X_I"]
                + blk.inlet.conc_mass_comp[0, "X_P"]
                + blk.inlet.conc_mass_comp[0, "X_BH"]
                + blk.inlet.conc_mass_comp[0, "X_BA"]
                + blk.inlet.conc_mass_comp[0, "X_S"]
            )

        @self.Expression(self.flowsheet().time, doc="Dewatering factor")
        def f_dewat(blk, t):
            return blk.p_dewat * (10 / (blk.TSS[t]))

        @self.Expression(self.flowsheet().time, doc="Remove factor")
        def f_q_du(blk, t):
            return blk.TSS_rem / (pyunits.kg / pyunits.m**3) / 100 / blk.f_dewat[t]

        self.non_particulate_components = Set(
            initialize=[
                "S_I",
                "S_S",
                "S_O",
                "S_NO",
                "S_NH",
                "S_ND",
                "H2O",
                "S_ALK",
            ]
        )

        self.particulate_components = Set(
            initialize=["X_I", "X_S", "X_P", "X_BH", "X_BA", "X_ND"]
        )

        @self.Constraint(
            self.flowsheet().time,
            self.particulate_components,
            doc="particulate fraction",
        )
        def overflow_particulate_fraction(blk, t, i):
            return blk.split_fraction[t, "overflow", i] == 1 - blk.TSS_rem

        @self.Constraint(
            self.flowsheet().time,
            self.non_particulate_components,
            doc="soluble fraction",
        )
        def non_particulate_components(blk, t, i):
            return blk.split_fraction[t, "overflow", i] == 1 - blk.f_q_du[t]

    def _get_performance_contents(self, time_point=0):
        var_dict = {}
        for k in self.split_fraction.keys():
            if k[0] == time_point:
                var_dict[f"Split Fraction [{str(k[1:])}]"] = self.split_fraction[k]
        return {"vars": var_dict}

    def _get_stream_table_contents(self, time_point=0):
        outlet_list = self.create_outlet_list()

        io_dict = {}
        if self.config.mixed_state_block is None:
            io_dict["Inlet"] = self.mixed_state
        else:
            io_dict["Inlet"] = self.config.mixed_state_block

        for o in outlet_list:
            io_dict[o] = getattr(self, o + "_state")

        return create_stream_table_dataframe(io_dict, time_point=time_point)
