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
CSTR unit model for BSM2 and plant-wide wastewater treatment modeling.
This unit inherits from the IDAES CSTR unit.
"""

# Import IDAES cores
from idaes.core import (
    declare_process_block_class,
)
from idaes.models.unit_models.cstr import CSTRData as CSTRIDAESData

import idaes.logger as idaeslog
from idaes.core.scaling import CustomScalerBase, ConstraintScalingScheme

from pyomo.environ import (
    Constraint,
    NonNegativeReals,
    Var,
    units as pyunits,
)

from watertap.costing.unit_models.cstr import cost_cstr

__author__ = "Marcus Holly"


# Set up logger
_log = idaeslog.getLogger(__name__)


class CSTRScaler(CustomScalerBase):
    """
    Default modular scaler for CSTR.

    This Scaler relies on the associated property and reaction packages,
    either through user provided options (submodel_scalers argument) or by default
    Scalers assigned to the packages.
    """

    DEFAULT_SCALING_FACTORS = {
        "volume": 1e-3,
    }

    def variable_scaling_routine(
        self, model, overwrite: bool = False, submodel_scalers: dict = None
    ):
        """
        Routine to apply scaling factors to variables in model.

        Args:
            model: model to be scaled
            overwrite: whether to overwrite existing scaling factors
            submodel_scalers: dict of Scalers to use for sub-models, keyed by submodel local name

        Returns:
            None
        """
        # Call scaling methods for sub-models
        self.call_submodel_scaler_method(
            submodel=model.control_volume.properties_in,
            method="variable_scaling_routine",
            submodel_scalers=submodel_scalers,
            overwrite=overwrite,
        )
        self.propagate_state_scaling(
            target_state=model.control_volume.properties_out,
            source_state=model.control_volume.properties_in,
            overwrite=overwrite,
        )

        self.call_submodel_scaler_method(
            submodel=model.control_volume.properties_out,
            method="variable_scaling_routine",
            submodel_scalers=submodel_scalers,
            overwrite=overwrite,
        )
        self.call_submodel_scaler_method(
            submodel=model.control_volume.reactions,
            method="variable_scaling_routine",
            submodel_scalers=submodel_scalers,
            overwrite=overwrite,
        )

        # Scaling control volume variables
        self.scale_variable_by_default(
            model.control_volume.volume[0], overwrite=overwrite
        )

    def constraint_scaling_routine(
        self, model, overwrite: bool = False, submodel_scalers: dict = None
    ):
        """
        Routine to apply scaling factors to constraints in model.

        Submodel Scalers are called for the property and reaction blocks. All other constraints
        are scaled using the inverse maximum scheme.

        Args:
            model: model to be scaled
            overwrite: whether to overwrite existing scaling factors
            submodel_scalers: dict of Scalers to use for sub-models, keyed by submodel local name

        Returns:
            None
        """
        # Call scaling methods for sub-models
        self.call_submodel_scaler_method(
            submodel=model.control_volume.properties_in,
            method="constraint_scaling_routine",
            submodel_scalers=submodel_scalers,
            overwrite=overwrite,
        )
        self.call_submodel_scaler_method(
            submodel=model.control_volume.properties_out,
            method="constraint_scaling_routine",
            submodel_scalers=submodel_scalers,
            overwrite=overwrite,
        )
        self.call_submodel_scaler_method(
            submodel=model.control_volume.reactions,
            method="constraint_scaling_routine",
            submodel_scalers=submodel_scalers,
            overwrite=overwrite,
        )

        # Scale unit level constraints
        if hasattr(model, "CSTR_retention_time"):
            self.scale_constraint_by_nominal_value(
                model.CSTR_retention_time[0],
                scheme=ConstraintScalingScheme.inverseMaximum,
                overwrite=overwrite,
            )


@declare_process_block_class("CSTR")
class CSTRData(CSTRIDAESData):
    """
    CSTR unit block for BSM2
    """

    default_scaler = CSTRScaler

    CONFIG = CSTRIDAESData.CONFIG()

    def build(self):
        """
        Begin building model.
        Args:
            None
        Returns:
            None
        """

        # Call UnitModel.build to set up dynamics
        super(CSTRData, self).build()

        self.hydraulic_retention_time = Var(
            self.flowsheet().time,
            initialize=4,
            domain=NonNegativeReals,
            units=pyunits.s,
            doc="Hydraulic retention time",
        )

        def CSTR_retention_time_rule(self, t):
            return (
                self.hydraulic_retention_time[t]
                == self.volume[t] / self.control_volume.properties_in[t].flow_vol
            )

        self.CSTR_retention_time = Constraint(
            self.flowsheet().time,
            rule=CSTR_retention_time_rule,
            doc="Total CSTR retention time",
        )

    @property
    def default_costing_method(self):
        return cost_cstr
