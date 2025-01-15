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

from copy import deepcopy

# Import Pyomo libraries
from pyomo.environ import (
    Var,
    check_optimal_termination,
    Param,
    Constraint,
    Suffix,
    units as pyunits,
)
from pyomo.common.config import ConfigBlock, ConfigValue, In

# Import IDAES cores
from idaes.core import (
    declare_process_block_class,
    UnitModelBlockData,
    useDefault,
)
from watertap.core.solvers import get_solver
from idaes.core.util.tables import create_stream_table_dataframe
from idaes.core.util.constants import Constants
from idaes.core.util.config import is_physical_parameter_block

from idaes.core.util.exceptions import InitializationError

import idaes.core.util.scaling as iscale
import idaes.logger as idaeslog

from watertap.core import InitializationMixin
from watertap.costing.unit_models.crystallizer import cost_crystallizer
from watertap.core.util.initialization import interval_initializer

_log = idaeslog.getLogger(__name__)

__author__ = "Oluwamayowa Amusat"


# when using this file the name "Filtration" is what is imported
@declare_process_block_class("Crystallization")
class CrystallizationData(InitializationMixin, UnitModelBlockData):
    """
    Zero order crystallization model
    """

    # CONFIG are options for the unit model, this simple model only has the mandatory config options
    CONFIG = ConfigBlock()

    CONFIG.declare(
        "dynamic",
        ConfigValue(
            domain=In([False]),
            default=False,
            description="Dynamic model flag - must be False",
            doc="""Indicates whether this model will be dynamic or not,
    **default** = False. The filtration unit does not support dynamic
    behavior, thus this must be False.""",
        ),
    )

    CONFIG.declare(
        "has_holdup",
        ConfigValue(
            default=False,
            domain=In([False]),
            description="Holdup construction flag - must be False",
            doc="""Indicates whether holdup terms should be constructed or not.
    **default** - False. The filtration unit does not have defined volume, thus
    this must be False.""",
        ),
    )

    CONFIG.declare(
        "property_package",
        ConfigValue(
            default=useDefault,
            domain=is_physical_parameter_block,
            description="Property package to use for control volume",
            doc="""Property parameter object used to define property calculations,
    **default** - useDefault.
    **Valid values:** {
    **useDefault** - use default package from parent model or flowsheet,
    **PhysicalParameterObject** - a PhysicalParameterBlock object.}""",
        ),
    )

    CONFIG.declare(
        "property_package_args",
        ConfigBlock(
            implicit=True,
            description="Arguments to use for constructing property packages",
            doc="""A ConfigBlock with arguments to be passed to a property block(s)
    and used when constructing these,
    **default** - None.
    **Valid values:** {
    see property package for documentation.}""",
        ),
    )

    def build(self):
        super().build()

        solvent_set = self.config.property_package.solvent_set
        solute_set = self.config.property_package.solute_set

        # this creates blank scaling factors, which are populated later
        self.scaling_factor = Suffix(direction=Suffix.EXPORT)

        # Next, get the base units of measurement from the property definition
        units_meta = self.config.property_package.get_metadata().get_derived_units

        # Add unit variables
        """
        self.approach_temperature_heat_exchanger = Param(
            initialize=4,
            units=pyunits.K,
            doc="Maximum temperature difference between inlet and outlet of a crystallizer heat exchanger.\
            Lewis et al. suggests 1-2 degC but use 5degC in example; Tavare example used 4 degC.\
            Default is 4 degC",
        )
        """

        # ====== Crystallizer sizing parameters ================= #
        self.dimensionless_crystal_length = Param(
            initialize=3.67,  # Parameter from population balance modeling for median crystal size
            units=pyunits.dimensionless,
        )

        self.crystal_median_length = Var(
            initialize=0.5e-3,  # From Mersmann et al., Tavare et al. example
            bounds=(
                0.2e-3,
                0.6e-3,
            ),  # Limits for FC crystallizers based on Bermingham et al.
            units=pyunits.m,
            doc="Desired median crystal size, m",
        )

        self.crystal_growth_rate = Var(
            initialize=3.7e-8,  # From Mersmann et al. for NaCl. Perry has values between 0.5e-8 to 13e-8 for NaCl
            bounds=(1e-9, 1e-6),  # Based on Mersmann and Kind diagram.
            units=pyunits.m / pyunits.s,
            doc="Crystal growth rate, m/s",
        )

        self.souders_brown_constant = Var(
            initialize=0.04,
            units=pyunits.m / pyunits.s,
            doc="Constant for Souders-Brown equation, set at 0.04 m/s based on Dutta et al. \
            Lewis et al suggests 0.024 m/s, while Tavare suggests about 0.06 m/s ",
        )

        # ====== Model variables ================= #
        self.crystallization_yield = Var(
            solute_set,
            initialize=0.5,
            bounds=(0.0, 1),
            units=pyunits.dimensionless,
            doc="Crystallizer solids yield",
        )

        self.product_volumetric_solids_fraction = Var(
            initialize=0.25,
            bounds=(0.0, 1),
            units=pyunits.dimensionless,
            doc="Volumetric fraction of solids in slurry product (i.e. solids-liquid mixture).",
        )

        self.temperature_operating = Var(
            initialize=298.15,
            bounds=(273, 1000),
            units=pyunits.K,
            doc="Crystallizer operating temperature: boiling point of the solution.",
        )

        self.pressure_operating = Var(
            initialize=1e3,
            bounds=(0.001, 2e6),
            units=pyunits.Pa,
            doc="Operating pressure of the crystallizer.",
        )

        self.dens_mass_magma = Var(
            initialize=250,
            bounds=(0.01, 5000),
            units=pyunits.kg / pyunits.m**3,
            doc="Magma density, i.e. mass of crystals per unit volume of suspension",
        )

        self.dens_mass_slurry = Var(
            initialize=1000,
            bounds=(1, 5000),
            units=pyunits.kg / pyunits.m**3,
            doc="Suspension density, i.e. density of solid-liquid mixture before separation",
        )
        """
            self.work_mechanical = Var(
                self.flowsheet().config.time,
                initialize=1e5,
                bounds=(-5e6, 5e6),
                units=pyunits.kJ / pyunits.s,
                doc="Crystallizer thermal energy requirement",
            )
        """
        self.diameter_crystallizer = Var(
            initialize=3,
            bounds=(0, 25),
            units=pyunits.m,
            doc="Diameter of crystallizer",
        )

        self.height_slurry = Var(
            initialize=3,
            bounds=(0, 25),
            units=pyunits.m,
            doc="Slurry height in crystallizer",
        )

        self.height_crystallizer = Var(
            initialize=3, bounds=(0, 30), units=pyunits.m, doc="Crystallizer height"
        )


        """
            self.magma_circulation_flow_vol = Var(
                initialize=1,
                bounds=(0, 100),
                units=pyunits.m**3 / pyunits.s,
                doc="Minimum circulation flow rate through crystallizer heat exchanger",
            )
        """

        self.relative_supersaturation = Var(
            solute_set, initialize=0.1, bounds=(0, 100), units=pyunits.dimensionless
        )

        self.t_res = Var(
            initialize=1,
            bounds=(0, 10),
            units=pyunits.hr,
            doc="Residence time in crystallizer",
        )

        self.volume_suspension = Var(
            initialize=1,
            bounds=(0, None),
            units=pyunits.m**3,
            doc="Crystallizer minimum active volume, i.e. volume of liquid-solid suspension",
        )

        # Add state blocks for inlet, outlet, and waste
        # These include the state variables and any other properties on demand
        # Add inlet block
        tmp_dict = dict(**self.config.property_package_args)
        tmp_dict["has_phase_equilibrium"] = False
        tmp_dict["parameters"] = self.config.property_package
        tmp_dict["defined_state"] = True  # inlet block is an inlet
        self.properties_in = self.config.property_package.state_block_class(
            self.flowsheet().config.time, doc="Material properties of inlet", **tmp_dict
        )

        # Add outlet and waste block
        tmp_dict["defined_state"] = False  # outlet and waste block is not an inlet
        self.properties_out = self.config.property_package.state_block_class(
            self.flowsheet().config.time,
            doc="Material properties of liquid outlet",
            **tmp_dict,
        )

        self.properties_solids = self.config.property_package.state_block_class(
            self.flowsheet().config.time,
            doc="Material properties of solid crystals at outlet",
            **tmp_dict,
        )

        self.properties_vapor = self.config.property_package.state_block_class(
            self.flowsheet().config.time,
            doc="Material properties of water vapour at outlet",
            **tmp_dict,
        )

        # Add ports - oftentimes users interact with these rather than the state blocks
        self.add_port(name="inlet", block=self.properties_in)
        self.add_port(name="outlet", block=self.properties_out)
        self.add_port(name="solids", block=self.properties_solids)
        self.add_port(name="vapor", block=self.properties_vapor)

        # Add constraints
        # 1. Material balances
        @self.Constraint(
            self.config.property_package.component_list,
            doc="Mass balance for components",
        )
        def eq_mass_balance_constraints(b, j):
            return sum(
                b.properties_in[0].flow_mass_phase_comp[p, j]
                for p in self.config.property_package.phase_list
                if (p, j) in b.properties_in[0].phase_component_set
            ) == sum(
                b.properties_out[0].flow_mass_phase_comp[p, j]
                for p in self.config.property_package.phase_list
                if (p, j) in b.properties_out[0].phase_component_set
            ) + sum(
                b.properties_vapor[0].flow_mass_phase_comp[p, j]
                for p in self.config.property_package.phase_list
                if (p, j) in b.properties_vapor[0].phase_component_set
            ) + sum(
                b.properties_solids[0].flow_mass_phase_comp[p, j]
                for p in self.config.property_package.phase_list
                if (p, j) in b.properties_solids[0].phase_component_set
            )

        # 2. Constraint on outlet liquid composition based on solubility requirements
        @self.Constraint(
            self.config.property_package.component_list,
            doc="Solubility vs mass fraction constraint",
        )
        def eq_solubility_massfrac_equality_constraint(b, j):
            if j in solute_set:
                return (
                    b.properties_out[0].mass_frac_phase_comp["Liq", j]
                    - b.properties_out[0].solubility_mass_frac_phase_comp["Liq", j]
                    == 0
                )
            else:
                return Constraint.Skip

        """
            # 3. Performance equations
            # (a) based on yield
            @self.Constraint(
                self.config.property_package.component_list,
                doc="Component salt yield equation",
            )
            def eq_removal_balance(b, j):
                if j in solvent_set:
                    return Constraint.Skip
                else:
                    return (
                        b.properties_in[0].flow_mass_phase_comp["Liq", j]
                        * b.crystallization_yield[j]
                        == b.properties_in[0].flow_mass_phase_comp["Liq", j]
                        - b.properties_out[0].flow_mass_phase_comp["Liq", j]
                    )
        
        # (b) Volumetric fraction constraint
        @self.Constraint(doc="Solid volumetric fraction in outlet: constraint, 1-E")
        def eq_vol_fraction_solids(b):
            return self.product_volumetric_solids_fraction * (
                b.properties_solids[0].flow_vol + b.properties_out[0].flow_vol
            )== b.properties_solids[
                0
            ].flow_vol 

        # (c) Magma density constraint
        @self.Constraint(doc="Slurry magma density")
        def eq_dens_magma(b):
            return (
                self.dens_mass_magma
                == b.properties_solids[0].dens_mass_solute["Sol"]
                * self.product_volumetric_solids_fraction
            )

        # (e) Relative supersaturation
        @self.Constraint(
            solute_set,
            doc="Relative supersaturation created via evaporation, g/g (solution)",
        )

            def eq_relative_supersaturation(b, j):
        # Calculate the mass of solute j after evaporation
        #### change the naming
                mass_after_evap = b.properties_in[0].flow_mass_phase_comp["Liq", j]

                # Calculate the total mass of liquid phase components after evaporation
                total_mass_after_evap = (
                    sum(
                        b.properties_in[0].flow_mass_phase_comp["Liq", k]
                        for k in b.config.property_package.solute_set
                    )
                    + b.properties_in[0].flow_mass_phase_comp["Liq", "H2O"]
                    - b.properties_vapor[0].flow_mass_phase_comp["Vap", "H2O"]
                )

                return (
                    b.relative_supersaturation[j] * b.properties_out[0].solubility_mass_frac_phase_comp["Liq", j] * total_mass_after_evap
                    == (
                        mass_after_evap
                        - b.properties_out[0].solubility_mass_frac_phase_comp["Liq", j] * total_mass_after_evap
                    ))
"""
        # (d) Operating pressure constraint
        @self.Constraint(doc="Operating pressure constraint")
        def eq_operating_pressure_constraint(b):
            return self.pressure_operating  == b.properties_out[0].pressure_sat

        


        # 4. Fix flows of empty solid, liquid and vapour streams
        # (i) Fix solids: liquid and vapour flows must be zero
        for p, j in self.properties_solids[0].phase_component_set:
            if p != "Sol":
                self.properties_solids[0].flow_mass_phase_comp[p, j].fix(1e-8)

        # (ii) Fix liquids: solid and vapour flows must be zero
        for p, j in self.properties_out[0].phase_component_set:
            if p != "Liq":
                self.properties_out[0].flow_mass_phase_comp[p, j].fix(1e-8)

        # (iii) Fix vapor: solid and vapour flows must be zero.
        for p, j in self.properties_vapor[0].phase_component_set:
            if p != "Vap":
                self.properties_vapor[0].flow_mass_phase_comp[p, j].fix(1e-8)

        # 5. Add an energy balance for the system
        ## (iii) Enthalpy balance: based on Lewis et al. Enthalpy is exothermic, and the hC in property package is -ve
        @self.Constraint(doc="Enthalpy balance over crystallization system")
        def eq_enthalpy_balance(b):
            return (
                b.properties_in[0].enth_flow
                - b.properties_out[0].enth_flow
                - b.properties_vapor[0].enth_flow
                - b.properties_solids[0].enth_flow
                #+ self.work_mechanical[0]
                - sum(
                   b.properties_solids[0].flow_mass_phase_comp["Sol", j]
                    * b.properties_solids[0].dh_crystallization_mass_comp[j]
                    for j in solute_set
                )
                == 0
            )

        # 6. Pressure and temperature balances - what is the pressure of the outlet solid and vapour?
        # TO-DO: Figure out actual liquid and solid pressures.
        @self.Constraint()
        def eq_p_con1(b):
            return b.properties_out[0].pressure_sat == b.properties_out[0].pressure

        @self.Constraint()
        def eq_p_con2(b):
            return b.properties_out[0].pressure_sat == b.properties_solids[0].pressure

        @self.Constraint()
        def eq_p_con3(b):
            return b.properties_vapor[0].pressure == self.pressure_operating

        @self.Constraint()
        def eq_T_con1(b):
            return self.temperature_operating == b.properties_solids[0].temperature

        @self.Constraint()
        def eq_T_con2(b):
            return self.temperature_operating == b.properties_vapor[0].temperature

        @self.Constraint()
        def eq_T_con3(b):
            return self.temperature_operating == b.properties_out[0].temperature
        
        """

        # 7. Heat exchanger minimum circulation flow rate calculations - see Lewis et al. or Tavare et al.
        @self.Constraint(
            doc="Constraint on mimimum circulation rate through crystallizer heat exchanger"
        )
        def eq_minimum_hex_circulation_rate_constraint(b):
            dens_cp_avg = self.approach_temperature_heat_exchanger * (
                b.product_volumetric_solids_fraction
                * b.properties_solids[0].dens_mass_solute["Sol"]
                * b.properties_solids[0].cp_mass_solute["Sol"]
                + (1 - b.product_volumetric_solids_fraction)
                * b.properties_out[0].dens_mass_phase["Liq"]
                * b.properties_out[0].cp_mass_phase["Liq"]
            )
            return b.magma_circulation_flow_vol * dens_cp_avg == pyunits.convert(
                b.work_mechanical[0], to_units=pyunits.J / pyunits.s
            )
        

        # 8. Suspension density
        @self.Constraint(doc="Slurry density calculation")
        def eq_dens_mass_slurry(b):
            return (
                self.dens_mass_slurry
                == b.product_volumetric_solids_fraction
                * b.properties_solids[0].dens_mass_solute["Sol"]
                + (1 - b.product_volumetric_solids_fraction)
                * b.properties_out[0].dens_mass_phase["Liq"]
            )

        # 9. Residence time calculation
        @self.Constraint(doc="Residence time")
        def eq_residence_time(b):
            return b.t_res == b.crystal_median_length / (
                b.dimensionless_crystal_length
                * pyunits.convert(
                    b.crystal_growth_rate, to_units=pyunits.m / pyunits.hr
                )
            )

        # 10. Suspension volume calculation
        @self.Constraint(doc="Suspension volume")
        def eq_suspension_volume(b):
            return b.volume_suspension == (
                b.properties_solids[0].flow_vol + b.properties_out[0].flow_vol
            ) * pyunits.convert(b.t_res, to_units=pyunits.s)

        # 11. Minimum diameter of evaporation zone
        @self.Expression(doc="maximum allowable vapour linear velocity in m/s")
        def eq_max_allowable_velocity(b):
            return (
                b.souders_brown_constant
                * (
                    b.properties_out[0].dens_mass_phase["Liq"]
                    / b.properties_vapor[0].dens_mass_solvent["Vap"]
                )
                ** 0.5
            )

        @self.Constraint(
            doc="Crystallizer diameter (based on minimum diameter of evaporation zone)"
        )
        def eq_vapor_head_diameter_constraint(b):
            return (
                self.diameter_crystallizer
                == (
                    4
                    * b.properties_vapor[0].flow_vol_phase["Vap"]
                    / (Constants.pi * b.eq_max_allowable_velocity)
                )
                ** 0.5
            )

        # 12. Minimum crystallizer height
        @self.Constraint(doc="Slurry height based on crystallizer diameter")
        def eq_slurry_height_constraint(b):
            return self.height_slurry * (
                Constants.pi * b.diameter_crystallizer**2
            )== 4 * b.volume_suspension 

        @self.Expression(
            doc="Recommended height of vapor space (0.75*D) based on Tavares et. al."
        )
        def eq_vapor_space_height(b):
            return 0.75 * b.diameter_crystallizer

        @self.Expression(
            doc="Height to diameter ratio constraint for evaporative crystallizers (Wilson et. al.)"
        )
        def eq_minimum_height_diameter_ratio(b):
            return 1.5 * b.diameter_crystallizer

        @self.Constraint(doc="Crystallizer height")
        def eq_crystallizer_height_constraint(b):
            # Height is max(). Manual smooth max implementation used here: max(a,b) = 0.5(a + b + |a-b|)
            a = b.eq_vapor_space_height + b.height_slurry
            b = b.eq_minimum_height_diameter_ratio
            eps = 1e-20 * pyunits.m
            return self.height_crystallizer == 0.5 * (
                a + b + ((a - b) ** 2 + eps**2) ** 0.5
            )

            """

    def initialize_build(
        self,
        state_args=None,
        outlvl=idaeslog.NOTSET,
        solver=None,
        optarg=None,
    ):
        """
        General wrapper for pressure changer initialization routines

        Keyword Arguments:
            state_args : a dict of arguments to be passed to the property
                         package(s) to provide an initial state for
                         initialization (see documentation of the specific
                         property package) (default = {}).
            outlvl : sets output level of initialization routine
            optarg : solver options dictionary object (default=None)
            solver : str indicating which solver to use during
                     initialization (default = None)

        Returns: None
        """
        init_log = idaeslog.getInitLogger(self.name, outlvl, tag="unit")
        solve_log = idaeslog.getSolveLogger(self.name, outlvl, tag="unit")

        opt = get_solver(solver, optarg)

        # ---------------------------------------------------------------------
        # Initialize holdup block
        flags = self.properties_in.initialize(
            outlvl=outlvl,
            optarg=optarg,
            solver=solver,
            state_args=state_args,
            hold_state=True,
        )
        init_log.info_high("Initialization Step 1 Complete.")
        # ---------------------------------------------------------------------
        # Initialize other state blocks
        # Set state_args from inlet state
        if state_args is None:
            state_args = {}
            state_dict = self.properties_in[
                self.flowsheet().config.time.first()
            ].define_port_members()

            for k in state_dict.keys():
                if state_dict[k].is_indexed():
                    state_args[k] = {}
                    for m in state_dict[k].keys():
                        state_args[k][m] = state_dict[k][m].value
                else:
                    state_args[k] = state_dict[k].value

        self.properties_out.initialize(
            outlvl=outlvl,
            optarg=optarg,
            solver=solver,
            state_args=state_args,
        )

        state_args_solids = deepcopy(state_args)
        for p, j in self.properties_solids.phase_component_set:
            if p == "Sol":
                state_args_solids["flow_mass_phase_comp"][p, j] = state_args[
                    "flow_mass_phase_comp"
                ]["Liq", j]
            elif p == "Liq" or p == "Vap":
                state_args_solids["flow_mass_phase_comp"][p, j] = 1e-8
        self.properties_solids.initialize(
            outlvl=outlvl,
            optarg=optarg,
            solver=solver,
            state_args=state_args_solids,
        )

        state_args_vapor = deepcopy(state_args)
        for p, j in self.properties_vapor.phase_component_set:
            if p == "Vap":
                state_args_vapor["flow_mass_phase_comp"][p, j] = state_args[
                    "flow_mass_phase_comp"
                ]["Liq", j]
            elif p == "Liq" or p == "Sol":
                state_args_vapor["flow_mass_phase_comp"][p, j] = 1e-8
        self.properties_vapor.initialize(
            outlvl=outlvl,
            optarg=optarg,
            solver=solver,
            state_args=state_args_vapor,
        )
        init_log.info_high("Initialization Step 2 Complete.")

        interval_initializer(self)
        # ---------------------------------------------------------------------
        # Solve unit
        with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
            res = opt.solve(self, tee=slc.tee)
        init_log.info_high("Initialization Step 3 {}.".format(idaeslog.condition(res)))
        # ---------------------------------------------------------------------
        # Release Inlet state
        self.properties_in.release_state(flags, outlvl=outlvl)
        init_log.info("Initialization Complete: {}".format(idaeslog.condition(res)))

        if not check_optimal_termination(res):
            raise InitializationError(f"Unit model {self.name} failed to initialize")
    def calculate_scaling_factors(self):
        super().calculate_scaling_factors()

        iscale.set_scaling_factor(
            self.crystal_growth_rate, 1e7
        )  # growth rates typically of order 1e-7 to 1e-9 m/s
        iscale.set_scaling_factor(
            self.crystal_median_length, 1e3
        )  # Crystal lengths typically in mm
        iscale.set_scaling_factor(
            self.souders_brown_constant, 1e2
        )  # Typical values are 0.0244, 0.04 and 0.06
        iscale.set_scaling_factor(
            self.diameter_crystallizer, 1
        )  # Crystallizer diameters typically up to about 20 m
        iscale.set_scaling_factor(
            self.height_crystallizer, 1
        )  # H/D ratio maximum is about 1.5, so same scaling as diameter
        iscale.set_scaling_factor(self.height_slurry, 1)  # Same scaling as diameter
        #iscale.set_scaling_factor(self.magma_circulation_flow_vol, 1)
        iscale.set_scaling_factor(self.relative_supersaturation, 10)
        iscale.set_scaling_factor(self.t_res, 1)  # Residence time is in hours
        iscale.set_scaling_factor(
            self.volume_suspension, 0.1
        )  # Suspension volume usually in tens to hundreds range
        iscale.set_scaling_factor(
            self.crystallization_yield, 1
        )  # Yield is between 0 and 1, usually in the 10-60% range
        iscale.set_scaling_factor(self.product_volumetric_solids_fraction, 10)
        iscale.set_scaling_factor(
            self.temperature_operating,
            iscale.get_scaling_factor(self.properties_in[0].temperature),
        )
        iscale.set_scaling_factor(self.pressure_operating, 1e-3)
        iscale.set_scaling_factor(
            self.dens_mass_magma, 1e-3
        )  # scaling factor of dens_mass_phase['Liq']
        iscale.set_scaling_factor(
            self.dens_mass_slurry, 1e-3
        )  # scaling factor of dens_mass_phase['Liq']
        #iscale.set_scaling_factor(
         #   self.work_mechanical[0],
          #  iscale.get_scaling_factor(
           #     self.properties_in[0].flow_mass_phase_comp["Vap", "H2O"]
            #)
            #* iscale.get_scaling_factor(self.properties_in[0].enth_mass_solvent["Vap"]),
        #)
        iscale.set_scaling_factor(self.properties_in[0].enth_flow, 1e-3)
        iscale.set_scaling_factor(self.properties_out[0].enth_flow, 1e-3)
        iscale.set_scaling_factor(self.properties_vapor[0].enth_flow, 1e-3)
        iscale.set_scaling_factor(self.properties_solids[0].enth_flow, 1e-3)
        iscale.set_scaling_factor(self.properties_in[0.0].pressure, 1e3)



        # transforming constraints
        for ind, c in self.eq_T_con1.items():
            sf = iscale.get_scaling_factor(self.properties_in[0].temperature)
            iscale.constraint_scaling_transform(c, sf)

        for ind, c in self.eq_T_con2.items():
            sf = iscale.get_scaling_factor(self.properties_in[0].temperature)
            iscale.constraint_scaling_transform(c, sf)

        for ind, c in self.eq_T_con3.items():
            sf = iscale.get_scaling_factor(self.properties_in[0].temperature)
            iscale.constraint_scaling_transform(c, sf)

        for ind, c in self.eq_p_con1.items():
            sf = iscale.get_scaling_factor(self.properties_in[0].pressure)
            iscale.constraint_scaling_transform(c, sf)

        for ind, c in self.eq_p_con2.items():
            sf = iscale.get_scaling_factor(self.properties_in[0].pressure)
            iscale.constraint_scaling_transform(c, sf)

        for ind, c in self.eq_p_con3.items():
            sf = iscale.get_scaling_factor(self.properties_in[0].pressure)
            iscale.constraint_scaling_transform(c, sf)

        for j, c in self.eq_mass_balance_constraints.items():
            sf = iscale.get_scaling_factor(
                self.properties_in[0].flow_mass_phase_comp["Liq", j]
            )
            iscale.constraint_scaling_transform(c, sf)

        for j, c in self.eq_solubility_massfrac_equality_constraint.items():
            iscale.constraint_scaling_transform(c, 1e0)

        #for j, c in self.eq_dens_magma.items():
          #  iscale.constraint_scaling_transform(
            #    c, iscale.get_scaling_factor(self.dens_mass_magma)
            #)

        #for j, c in self.eq_removal_balance.items():
          #  sf = iscale.get_scaling_factor(
           #     self.properties_in[0].flow_mass_phase_comp["Liq", j]
           # )
          #  iscale.constraint_scaling_transform(c, sf)

        for ind, c in self.eq_enthalpy_balance.items():
            sf = iscale.get_scaling_factor(self.properties_in[0].enth_flow)
            iscale.constraint_scaling_transform(c, sf)


    def _get_stream_table_contents(self, time_point=0):
        return create_stream_table_dataframe(
            {
                "Feed Inlet": self.inlet,
                "Liquid Outlet": self.outlet,
                "Vapor Outlet": self.vapor,
                "Solid Outlet": self.solids,
            },
            time_point=time_point,
        )

    def _get_performance_contents(self, time_point=0):
        var_dict = {}
        var_dict["Operating Temperature (K)"] = self.temperature_operating
        var_dict["Operating Pressure (Pa)"] = self.pressure_operating
        var_dict["Magma density of solution (Kg/m**3)"] = self.dens_mass_magma
        var_dict["Slurry density (Kg/m3)"] = self.dens_mass_slurry
        #var_dict["Heat requirement"] = self.work_mechanical[time_point]
        var_dict["Crystallizer diameter (m)"] = self.diameter_crystallizer
        #var_dict["Magma circulation flow rate (m**3/s)"] = (
           # self.magma_circulation_flow_vol
       # )
        var_dict["Vol. frac. of solids in suspension, 1-E"] = (
            self.product_volumetric_solids_fraction
        )
        var_dict["Residence time"] = self.t_res
        var_dict["Crystallizer minimum active volume (m**3)"] = self.volume_suspension
        var_dict["Suspension height in crystallizer (m)"] = self.height_slurry
        var_dict["Crystallizer height (m)"] = self.height_crystallizer

        for j in self.config.property_package.solute_set:
            yield_mem_name = f"{j} yield (fraction)"
            var_dict[yield_mem_name] = self.crystallization_yield[j]
            supersat_mem_name = f"{j} relative supersaturation (mass fraction basis)"
            var_dict[supersat_mem_name] = self.relative_supersaturation[j]

        return {"vars": var_dict}

    @property
    def default_costing_method(self):
        return cost_crystallizer
