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

# Import Pyomo libraries
from pyomo.environ import (
    Var,
    Set,
    NonNegativeReals,
    NegativeReals,
    Reference,
    units as pyunits,
    exp,
    value,
    check_optimal_termination,
)

from idaes.core import declare_process_block_class, FlowDirection
from idaes.core.util import scaling as iscale
from idaes.core.util.misc import add_object_reference
from watertap.core import (
    MembraneChannel0DBlock,
    ConcentrationPolarizationType,
    MassTransferCoefficient,
    PressureChangeType,
)
from watertap.core.membrane_channel0d import CONFIG_Template
from watertap.unit_models.osmotically_assisted_reverse_osmosis_base import (
    OsmoticallyAssistedReverseOsmosisBaseData,
    _add_has_full_reporting,
)
import idaes.logger as idaeslog


__author__ = "Adam Atia, Chenyu Wang"


@declare_process_block_class("OsmoticallyAssistedReverseOsmosis0D")
class OsmoticallyAssistedReverseOsmosisData(OsmoticallyAssistedReverseOsmosisBaseData):
    """
    Standard OARO Unit Model Class:
    - zero dimensional model
    - steady state only
    - single liquid phase only
    """

    CONFIG = CONFIG_Template()

    _add_has_full_reporting(CONFIG)
    # TODO: add a test for this exception
    def _add_membrane_channel_and_geometry(
        self, side="feed_side", flow_direction=FlowDirection.forward
    ):
        if not isinstance(side, str):
            raise TypeError(f"{side} is not a string. Please provide a string.")

        # Build membrane channel control volume
        setattr(
            self,
            side,
            MembraneChannel0DBlock(
                dynamic=False,
                has_holdup=False,
                property_package=self.config.property_package,
                property_package_args=self.config.property_package_args,
            ),
        )
        mem_side = getattr(self, side)

        if (self.config.pressure_change_type != PressureChangeType.fixed_per_stage) or (
            self.config.mass_transfer_coefficient == MassTransferCoefficient.calculated
        ):
            if not hasattr(self, "length") and not hasattr(self, "width"):
                self._add_length_and_width()
            mem_side.add_geometry(
                length_var=self.length,
                width_var=self.width,
                flow_direction=flow_direction,
            )
            self._add_area(include_constraint=True)
        else:
            mem_side.add_geometry(length_var=None, width_var=None)
            self._add_area(include_constraint=False)

    # def _add_deltaP(self):
    #     add_object_reference(self, "deltaP", self.feed_side.deltaP)

    def _add_mass_transfer(self):

        units_meta = self.config.property_package.get_metadata().get_derived_units

        # not in 1DRO
        @self.Constraint(
            self.flowsheet().config.time, doc="Enthalpy transfer from feed to permeate"
        )
        def eq_connect_enthalpy_transfer(b, t):
            return (
                b.permeate_side.enthalpy_transfer[t]
                == -b.feed_side.enthalpy_transfer[t]
            )

        # mass transfer
        def mass_transfer_phase_comp_initialize(b, t, p, j):
            return value(
                self.feed_side.properties_in[t].get_material_flow_terms("Liq", j)
                * self.recovery_mass_phase_comp[t, "Liq", j]
            )

        self.mass_transfer_phase_comp = Var(
            self.flowsheet().config.time,
            self.config.property_package.phase_list,
            self.config.property_package.component_list,
            initialize=mass_transfer_phase_comp_initialize,
            bounds=(0.0, 1e6),
            domain=NonNegativeReals,
            units=units_meta("mass") * units_meta("time") ** -1,
            doc="Mass transfer to permeate",
        )

        @self.Constraint(
            self.flowsheet().config.time,
            self.config.property_package.phase_list,
            self.config.property_package.component_list,
            doc="Mass transfer term",
        )
        def eq_mass_transfer_term(self, t, p, j):
            return (
                self.mass_transfer_phase_comp[t, p, j]
                == -self.feed_side.mass_transfer_term[t, p, j]
            )

        # Feed and permeate-side connection
        @self.Constraint(
            self.flowsheet().config.time,
            self.config.property_package.phase_list,
            self.config.property_package.component_list,
            doc="Mass transfer from feed to permeate",
        )
        def eq_connect_mass_transfer(b, t, p, j):
            return (
                b.permeate_side.mass_transfer_term[t, p, j]
                == -b.feed_side.mass_transfer_term[t, p, j]
            )

        # Different expression in 1DRO
        @self.Constraint(
            self.flowsheet().config.time,
            self.config.property_package.phase_list,
            self.config.property_package.component_list,
            doc="Permeate production",
        )
        def eq_permeate_production(b, t, p, j):
            return (
                b.mixed_permeate[t].get_material_flow_terms(p, j)
                == b.area * b.flux_mass_phase_comp_avg[t, p, j]
            )

        # ------------------------------------------------------------------------------------------------
        # Not in 1DRO
        @self.Constraint(
            self.flowsheet().config.time,
            self.length_domain,
            self.config.property_package.solute_set,
            doc="Permeate mass fraction",
        )
        def eq_mass_frac_permeate(b, t, x, j):
            return (
                b.permeate_side.properties[t, x].mass_frac_phase_comp["Liq", j]
                * sum(
                    self.flux_mass_phase_comp[t, x, "Liq", jj]
                    for jj in self.config.property_package.component_list
                )
                == self.flux_mass_phase_comp[t, x, "Liq", j]
            )

    def calculate_scaling_factors(self):
        if iscale.get_scaling_factor(self.dens_solvent) is None:
            sf = iscale.get_scaling_factor(
                self.feed_side.properties_in[0].dens_mass_phase["Liq"]
            )
            iscale.set_scaling_factor(self.dens_solvent, sf)

        super().calculate_scaling_factors()

        for (t, p, j), v in self.mass_transfer_phase_comp.items():
            sf = iscale.get_scaling_factor(
                self.feed_side.properties_in[t].get_material_flow_terms(p, j)
            )
            if iscale.get_scaling_factor(v) is None:
                iscale.set_scaling_factor(v, sf)
            v = self.feed_side.mass_transfer_term[t, p, j]
            if iscale.get_scaling_factor(v) is None:
                iscale.set_scaling_factor(v, sf)

        if hasattr(self, "length"):
            if iscale.get_scaling_factor(self.length) is None:
                iscale.set_scaling_factor(self.length, 1)

        if hasattr(self, "width"):
            if iscale.get_scaling_factor(self.width) is None:
                iscale.set_scaling_factor(self.width, 1)
