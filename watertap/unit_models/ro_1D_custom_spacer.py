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
import sys

from idaes.core.util.env_info import __author__

# Import IDAES cores
from idaes.core import (
    declare_process_block_class,
    useDefault,
)
from idaes.core.util.exceptions import ConfigurationError
import idaes.logger as idaeslog
from pyomo.core import value

from watertap.core.custom_spacer_base import (
    _add_spacer_config,
    CustomSpacer,
    SpacerType,
    SherwoodCorrelation,
    PressureCorrelation,
)
from watertap.unit_models.reverse_osmosis_base import (
    ReverseOsmosisBaseData,
    _add_has_full_reporting,
)
from watertap.core.membrane_channel1d import CONFIG_Template
from idaes.core.util import scaling as iscale
from pyomo.common.formatting import tabular_writer
from idaes.core.util.units_of_measurement import report_quantity

__author__ == "Laxmicharan Samineni"

# Set up logger
_log = idaeslog.getLogger(__name__)
_log.setLevel(idaeslog.DEBUG)


@declare_process_block_class("ReverseOsmosis1DCustomSpacer")
class ReverseOsmosis1DCustomSpacerData(ReverseOsmosisBaseData, CustomSpacer):
    CONFIG = CONFIG_Template
    _add_has_full_reporting(config_obj=CONFIG)
    _add_spacer_config(CONFIG=CONFIG)
    _add_geometry_config(CONFIG=CONFIG)

    def build(self):
        _log.warning("CUSTOM SPACER RO MODEL is being used for the build")
        super().build()
        CustomSpacer.apply_custom_spacer_correlations(self)

    def _set_transformation_methods(self):
        if self.config.transformation_method is useDefault:
            _log.warning(
                "Discretization method was "
                "not specified for the "
                "reverse osmosis module. "
                "Defaulting to finite "
                "difference method."
            )
            self.config.transformation_method = "dae.finite_difference"

        if self.config.transformation_scheme is useDefault:
            _log.warning(
                "Discretization scheme was "
                "not specified for the "
                "reverse osmosis module."
                "Defaulting to backward finite "
                "difference."
            )
            self.config.transformation_scheme = "BACKWARD"
            return

    def _check_spacer_config(self):
        if self.config.spacer_type == SpacerType.no_spacer:
            if self.config.sherwood_correlation not in (
                SherwoodCorrelation.graetz,
                SherwoodCorrelation.sieder,
            ):
                raise ConfigurationError(
                    f"Sherwood correlation: {self.config.sherwood_correlation} is not valid for Spacer type: "
                    f"{self.config.spacer_type}"
                )
            elif self.config.pressure_correlation != PressureCorrelation.no_spacer:
                raise ConfigurationError(
                    f"Pressure correlation: {self.config.pressure_correlation} is not valid for Spacer type:"
                    f" {self.config.spacer_type}"
                )

        elif self.config.spacer_type == SpacerType.net_spacer:
            valid_sherwood_types = (
                SherwoodCorrelation.guillen,
                SherwoodCorrelation.da_costa,
                SherwoodCorrelation.parameterized,
            )
            valid_pressure_types = (
                PressureCorrelation.guillen,
                PressureCorrelation.da_costa,
                PressureCorrelation.parameterized,
            )
            if self.config.sherwood_correlation not in valid_sherwood_types:
                raise ConfigurationError(
                    f"Sherwood correlation: {self.config.sherwood_correlation} is not valid for Spacer type:"
                    f" {self.config.spacer_type}"
                )
            elif self.config.pressure_correlation not in valid_pressure_types:
                raise ConfigurationError(
                    f"Pressure correlation: {self.config.pressure_correlation} is not valid for Spacer type:"
                    f" {self.config.spacer_type}"
                )

        if (
            self.config.sherwood_correlation == SherwoodCorrelation.parameterized
            and self.config.pressure_correlation != PressureCorrelation.parameterized
        ):
            _log.warning(
                f"Only Sherwood correlation is parameterized. Pressure correlation is set to"
                f" {self.config.pressure_correlation}"
            )

        if (
            self.config.pressure_correlation == PressureCorrelation.parameterized
            and self.config.sherwood_correlation != SherwoodCorrelation.parameterized
        ) or (
            self.config.pressure_correlation != PressureCorrelation.parameterized
            and self.config.sherwood_correlation == SherwoodCorrelation.parameterized
        ):
            raise ConfigurationError(
                f"{self.config.sherwood_correlation} and {self.config.pressure_correlation} are not compatible. Either"
                f" both Sh and Pn or none can be set to parameterized"
            )
        return

    def _add_feed_side_membrane_channel_and_geometry(self):
        self._set_transformation_methods()
        self._check_spacer_config()
        CustomSpacer._add_feed_side_membrane_channel_and_geometry(self)

    def _add_deltaP(self):
        CustomSpacer._add_deltaP(self)

    def _add_mass_transfer(self):
        CustomSpacer._add_mass_transfer(self)

    def calculate_scaling_factors(self):
        if iscale.get_scaling_factor(self.dens_solvent) is None:
            sf = iscale.get_scaling_factor(
                self.feed_side.properties[0, 0].dens_mass_phase["Liq"]
            )
            iscale.set_scaling_factor(self.dens_solvent, sf)

        super().calculate_scaling_factors()

        # these variables should have user input, if not there will be a warning
        if iscale.get_scaling_factor(self.width) is None:
            sf = iscale.get_scaling_factor(self.width, default=1, warning=True)
            iscale.set_scaling_factor(self.width, sf)

        if iscale.get_scaling_factor(self.length) is None:
            sf = iscale.get_scaling_factor(self.length, default=10, warning=True)
            iscale.set_scaling_factor(self.length, sf)

        for (t, x, p, j), v in self.mass_transfer_phase_comp.items():
            sf = (
                iscale.get_scaling_factor(
                    self.feed_side.properties[t, x].get_material_flow_terms(p, j)
                )
                / iscale.get_scaling_factor(self.feed_side.length)
            ) * value(self.nfe)
            if iscale.get_scaling_factor(v) is None:
                iscale.set_scaling_factor(v, sf)
            v = self.feed_side.mass_transfer_term[t, x, p, j]
            if iscale.get_scaling_factor(v) is None:
                iscale.set_scaling_factor(v, sf)

        if hasattr(self, "deltaP"):
            for v in self.deltaP.values():
                if iscale.get_scaling_factor(v) is None:
                    iscale.set_scaling_factor(v, 1e-4)

    def _get_performance_contents(self, time_point=0):
        performance_contents = super()._get_performance_contents()
        var_dict = performance_contents.get("vars", {})
        expr_dict = performance_contents.get("exprs", {})
        spacer_dict = {}
        spacer_dict["Channel Height"] = self.feed_side.channel_height
        spacer_dict["Spacer Porosity"] = self.feed_side.spacer_porosity

        if self.config.spacer_type is SpacerType.net_spacer:
            spacer_dict["Filament Diameter"] = self.feed_side.filament_dia
            spacer_dict["Mesh Size"] = self.feed_side.mesh_size
            spacer_dict["Spacer Angle"] = self.feed_side.spacer_angle
        return {"vars": var_dict, "exprs": expr_dict, "spacer_vars": spacer_dict}

    def report(self, time_point=0, dof=False, ostream=None, prefix=""):
        super().report()
        if ostream is None:
            ostream = sys.stdout
        tab = " " * 4
        performance = self._get_performance_contents()
        if "spacer_vars" in performance.keys() and len(performance["spacer_vars"]) > 0:
            ostream.write("\n")
            ostream.write(f"{prefix}{tab}Feed Spacer: \n\n")

            tabular_writer(
                ostream,
                prefix + tab,
                ((k, v) for k, v in performance["spacer_vars"].items()),
                ("Value", "Units", "Fixed", "Bounds"),
                lambda k, v: [
                    "{:#.5g}".format(report_quantity(v).m),
                    report_quantity(v).u,
                    v.fixed,
                    v.bounds,
                ],
            )
