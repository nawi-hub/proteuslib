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
Inherits from a modified CSTR model with injection terms. This model assumes oxygen will be injected.
"""

# Import Pyomo libraries
from pyomo.common.config import In

# Import IDAES cores
from idaes.core import (
    declare_process_block_class,
)
from idaes.core.scaling import CustomScalerBase, ConstraintScalingScheme


from watertap.unit_models.cstr_injection import (
    CSTR_InjectionData,
    ElectricityConsumption,
)

__author__ = "Adam Atia"


class AerationTankScaler(CustomScalerBase):
    """
    Default modular scaler for the aeration tank unit model.

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
        if hasattr(model, "eq_hydraulic_retention_time"):
            self.scale_constraint_by_nominal_value(
                model.eq_hydraulic_retention_time[0],
                scheme=ConstraintScalingScheme.inverseMaximum,
                overwrite=overwrite,
            )

        if hasattr(model, "eq_mass_transfer"):
            self.scale_constraint_by_nominal_value(
                model.eq_mass_transfer[0],
                scheme=ConstraintScalingScheme.inverseMaximum,
                overwrite=overwrite,
            )

        if hasattr(model, "eq_electricity_consumption"):
            self.scale_constraint_by_nominal_value(
                model.eq_electricity_consumption[0],
                scheme=ConstraintScalingScheme.inverseMaximum,
                overwrite=overwrite,
            )


@declare_process_block_class("AerationTank")
class AerationTankData(CSTR_InjectionData):
    """
    CSTR Unit Model with Injection Class
    """

    default_scaler = AerationTankScaler

    CONFIG = CSTR_InjectionData.CONFIG()
    CONFIG.electricity_consumption = ElectricityConsumption.calculated
    CONFIG.get("has_aeration")._domain = In([True])
    CONFIG.has_aeration = True
