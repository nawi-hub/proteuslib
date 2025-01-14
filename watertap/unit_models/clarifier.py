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
Clarifier unit model for BSM2 and plant-wide wastewater treatment modeling.
This unit inherits from the IDAES separator unit.
"""

# Import IDAES cores
from idaes.core import (
    declare_process_block_class,
)
from idaes.models.unit_models.separator import SeparatorData, SplittingType

from idaes.core.util.tables import create_stream_table_dataframe
import idaes.logger as idaeslog
from idaes.core.scaling import CustomScalerBase, ConstraintScalingScheme

from pyomo.environ import (
    Constraint,
    Var,
    Param,
    units as pyunits,
)

from idaes.core.util.exceptions import (
    ConfigurationError,
)
from watertap.costing.unit_models.clarifier import cost_clarifier

__author__ = "Chenyu Wang"

# Set up logger
_log = idaeslog.getLogger(__name__)


class ClarifierScaler(CustomScalerBase):
    """
    Default modular scaler for the clarifier unit model.
    This Scaler relies on the associated property and reaction packages,
    either through user provided options (submodel_scalers argument) or by default
    Scalers assigned to the packages.
    """

    DEFAULT_SCALING_FACTORS = {
        "surface_area": 1e-3,
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
            submodel=model.mixed_state,
            method="variable_scaling_routine",
            submodel_scalers=submodel_scalers,
            overwrite=overwrite,
        )
        self.propagate_state_scaling(
            target_state=model.underflow_state,
            source_state=model.mixed_state,
            overwrite=overwrite,
        )
        self.propagate_state_scaling(
            target_state=model.effluent_state,
            source_state=model.mixed_state,
            overwrite=overwrite,
        )

        self.call_submodel_scaler_method(
            submodel=model.underflow_state,
            method="variable_scaling_routine",
            submodel_scalers=submodel_scalers,
            overwrite=overwrite,
        )
        self.call_submodel_scaler_method(
            submodel=model.effluent_state,
            method="variable_scaling_routine",
            submodel_scalers=submodel_scalers,
            overwrite=overwrite,
        )

        # Scale unit level variables
        self.scale_variable_by_default(model.surface_area, overwrite=overwrite)

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
            submodel=model.mixed_state,
            method="constraint_scaling_routine",
            submodel_scalers=submodel_scalers,
            overwrite=overwrite,
        )
        self.call_submodel_scaler_method(
            submodel=model.underflow_state,
            method="constraint_scaling_routine",
            submodel_scalers=submodel_scalers,
            overwrite=overwrite,
        )
        self.call_submodel_scaler_method(
            submodel=model.effluent_state,
            method="constraint_scaling_routine",
            submodel_scalers=submodel_scalers,
            overwrite=overwrite,
        )

        # Scale unit level constraints
        for c in model.component_data_objects(Constraint, descend_into=False):
            self.scale_constraint_by_nominal_value(
                c,
                scheme=ConstraintScalingScheme.inverseMaximum,
                overwrite=overwrite,
            )


@declare_process_block_class("Clarifier")
class ClarifierData(SeparatorData):
    """
    Clarifier unit model for BSM2
    """

    default_scaler = ClarifierScaler

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

        # Call UnitModel.build to set up dynamics
        super(ClarifierData, self).build()

        if "underflow" and "effluent" not in self.config.outlet_list:
            raise ConfigurationError(
                "{} encountered unrecognised "
                "outlet_list. This should not "
                "occur - please use underflow "
                "and effluent as outlets.".format(self.name)
            )

        self.surface_area = Var(
            initialize=1500,
            doc="Cross section surface area of the clarifier",
            units=pyunits.m**2,
            bounds=(0, 3000),
        )

        self.electricity_consumption = Var(
            self.flowsheet().time,
            units=pyunits.kW,
            bounds=(0, None),
            doc="Electricity consumption of unit",
        )
        # The value is taken from Maravelias' data
        self.energy_electric_flow_vol_inlet = Param(
            initialize=0.008,
            units=pyunits.kWh / pyunits.m**3,
            mutable=True,
            doc="Electricity intensity with respect to inlet flow",
        )

        # Electricity constraint
        @self.Constraint(
            self.flowsheet().time,
            doc="Constraint for electricity consumption based on phosphorus removal",
        )
        def rule_electricity_consumption(self, t):
            return self.electricity_consumption[t] == (
                self.energy_electric_flow_vol_inlet
                * pyunits.convert(
                    self.inlet.flow_vol[t],
                    to_units=pyunits.m**3 / pyunits.hr,
                )
            )

    def _get_performance_contents(self, time_point=0):
        if hasattr(self, "split_fraction"):
            var_dict = {}
            for k in self.split_fraction.keys():
                if k[0] == time_point:
                    var_dict[f"Split Fraction [{str(k[1:])}]"] = self.split_fraction[k]
            return {"vars": var_dict}
        else:
            return None

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

    @property
    def default_costing_method(self):
        return cost_clarifier
