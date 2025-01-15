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


from idaes.core import (
    declare_process_block_class,
)
from idaes.models.unit_models.heat_exchanger import HeatExchangerData
from watertap.core.solvers import get_solver
import idaes.logger as idaeslog
from watertap.costing.unit_models.heat_exchanger import (
    cost_heat_exchanger,
)
from enum import Enum, auto
from pyomo.common.config import ConfigValue
from idaes.core.util.model_statistics import degrees_of_freedom


_log = idaeslog.getLogger(__name__)


__author__ = "Elmira Shamlou"

"""
This unit model uses is based on the IDAES `feedwater_heater_0D` model.
However, the constraints and properties defined in the IDAES unit model do not align with those in the available WaterTAP property packages.
To address this, alternative constraints have been replaced to ensure full condensation based on the WaterTAP properties. 
In addition, a condenser option and its corresponding initialization routine have been added, allowing the user to switch between the condenser and steam heater models.
Note that additional components like desuperheaters, drain mixers, and coolers are not included. If necessary, these can be modeled separately by adding heat exchangers and mixers to the flowsheet.
To do: Consider incorporating as a modification to IDAES' FeedWaterHeater model directly in the IDAES repo"
"""


class Mode(Enum):
    HEATER = auto()
    CONDENSER = auto()


@declare_process_block_class(
    "SteamHeater0D",
    doc="""Feedwater Heater Condensing Section
The feedwater heater condensing section model is a normal 0D heat exchanger
model with an added constraint to calculate the steam flow such that the outlet
of shell is a saturated liquid.""",
)
class SteamHeater0DData(HeatExchangerData):
    CONFIG = HeatExchangerData.CONFIG()
    CONFIG.declare(
        "mode",
        ConfigValue(
            default=Mode.HEATER,
            domain=Mode,
            description="Mode of operation: heater or condenser",
        ),
    )
    CONFIG.declare(
        "estimate_cooling_water",
        ConfigValue(
            default=False,
            domain=bool,
            description="Estimate cooling water flow rate for condenser mode",
        ),
    )

    def build(self):
        super().build()

        @self.Constraint(
            self.flowsheet().time,
            self.hot_side.config.property_package.component_list,
            doc="Mass balance",
        )
        def outlet_liquid_mass_balance(b, t, j):
            lb = b.hot_side.properties_out[t].flow_mass_phase_comp["Vap", j].lb
            b.hot_side.properties_out[t].flow_mass_phase_comp["Vap", j].fix(lb)
            return (
                b.hot_side.properties_in[t].flow_mass_phase_comp["Vap", j]
                + b.hot_side.properties_in[t].flow_mass_phase_comp["Liq", j]
                == b.hot_side.properties_out[t].flow_mass_phase_comp["Liq", j]
            )

        @self.Constraint(self.flowsheet().time, doc="Saturation pressure constraint")
        def outlet_pressure_sat(b, t):
            return (
                b.hot_side.properties_out[t].pressure
                >= b.hot_side.properties_out[t].pressure_sat
            )
        
       
    def initialize_build(self, *args, **kwargs):
        """
        Initialization routine for both heater and condenser modes. For condenser mode with cooling water estimation, the initialization is performed based on a specified design temperature rise on the cold side.
        """
        solver = kwargs.get("solver", None)
        optarg = kwargs.get("optarg", {})
        outlvl = kwargs.get("outlvl", idaeslog.NOTSET)
        init_log = idaeslog.getInitLogger(self.name, outlvl, tag="unit")
        solve_log = idaeslog.getSolveLogger(self.name, outlvl, tag="unit")

        if self.config.mode == Mode.HEATER:
            self.hot_side_inlet.fix()
            self.cold_side_inlet.fix()
            self.hot_side_outlet.unfix()
            #self.cold_side_outlet.unfix()
            #self.area.fix()

            self.outlet_liquid_mass_balance.deactivate()
            self.outlet_pressure_sat.deactivate()

            # regular heat exchanger initialization
            super().initialize_build(*args, **kwargs)

            for j in self.hot_side.config.property_package.component_list:
                self.hot_side.properties_out[0].flow_mass_phase_comp["Vap", j].fix(0)

            self.outlet_liquid_mass_balance.activate()
            self.outlet_pressure_sat.activate()

            for j in self.hot_side.config.property_package.component_list:
                self.hot_side_inlet.flow_mass_phase_comp[0, "Vap", j].unfix()

            opt = get_solver(solver, optarg)

            with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
                res = opt.solve(self, tee=slc.tee)
            init_log.info(
                "Initialization Complete (w/ extraction calc): {}".format(
                    idaeslog.condition(res)
                )
            )
        elif (
            self.config.mode == Mode.CONDENSER
            and not self.config.estimate_cooling_water
        ):
            
            self.hot_side_inlet.fix()
            self.cold_side_inlet.fix()
            #self.hot_side_outlet.unfix()
            #self.cold_side_outlet.unfix()
            #self.cold_side_outlet.temperature.fix()
            # condenser mode without cooling water estimation
            self.outlet_liquid_mass_balance.deactivate()
            self.outlet_pressure_sat.deactivate()
            super().initialize_build(*args, **kwargs)
            opt = get_solver(solver, optarg)
            #self.cold_side_outlet.temperature.unfix()

            self.outlet_liquid_mass_balance.activate()
            self.outlet_pressure_sat.activate()

            with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
                res = opt.solve(self, tee=slc.tee)
            init_log.info("Initialization Complete {}".format(idaeslog.condition(res)))

        elif self.config.mode == Mode.CONDENSER and self.config.estimate_cooling_water:
            print(f"DOF 1: {degrees_of_freedom(self)}")

            # condenser mode with cooling water estimation
            cold_side_outlet_temperature = self.cold_side_outlet.temperature[0].value
            self.hot_side_inlet.fix()
            self.cold_side_inlet.fix()
            #self.hot_side_outlet.unfix()
            self.cold_side_outlet.temperature.unfix()
            # condenser mode without cooling water estimation
            self.outlet_liquid_mass_balance.deactivate()
            self.outlet_pressure_sat.deactivate()
            print(f"DOF 2: {degrees_of_freedom(self)}")

            super().initialize_build(*args, **kwargs)
            self.area.unfix()
            self.cold_side_outlet.temperature[0].fix(cold_side_outlet_temperature)
            """
            for j in self.cold_side.config.property_package.solute_set:
                self.cold_side.properties_in[0].mass_frac_phase_comp["Liq", j]
                self.cold_side.properties_in[0].mass_frac_phase_comp["Liq", j].fix()
                """

            #for j in self.cold_side.config.property_package.component_list:
            self.cold_side.properties_in[0].flow_mass_phase_comp["Liq", "H2O"].unfix()

            self.outlet_liquid_mass_balance.activate()
            self.outlet_pressure_sat.activate()

            opt = get_solver(solver, optarg)

            with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
                res = opt.solve(self, tee=slc.tee)
            init_log.info(
                "Initialization Complete (w/ cooling water estimation): {}".format(
                    idaeslog.condition(res)
                )
            )

            #for j in self.cold_side.config.property_package.solute_set:
             #   self.cold_side.properties_in[0].mass_frac_phase_comp["Liq", j].unfix()
              #  self.cold_side.properties_in[0].flow_mass_phase_comp["Liq", j].fix()
            print(f"DOF 3: {degrees_of_freedom(self)}")
    """
                opt = get_solver(solver, optarg)

                with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
                    res = opt.solve(self, tee=slc.tee)
                init_log.info(
                    "Initialization Complete (w/ cooling water estimation): {}".format(
                        idaeslog.condition(res)
                    )
                )
    """
    @property
    def default_costing_method(self):
        return cost_heat_exchanger
