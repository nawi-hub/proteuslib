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
from enum import Enum, auto
from pyomo.core.expr.numeric_expr import sin as pysin
from pyomo.core.expr.numeric_expr import cos as pycos
from math import pi
from idaes.core.util.env_info import __author__
from pyomo.common.config import ConfigValue, In
from pyomo.core import NegativeReals, value
from pyomo.core import Var
from pyomo.environ import PositiveReals, NonNegativeReals, Reals
from pyomo.environ import units as pyunits
from pyomo.util.calc_var_value import calculate_variable_from_constraint
from watertap.core.membrane_channel1d import MembraneChannel1DBlock

__author__ == "Laxmicharan Samineni"


def _add_spacer_config(CONFIG):
    CONFIG.declare(
        "spacer_type",
        ConfigValue(
            default=SpacerType.net_spacer,
            domain=In(SpacerType),
            description="Argument for selecting the type of feed spacer",
            doc="""This argument changes the type of feed spacer by adding the required geometry variables along with 
            the corresponding mass transfer and pressure drop calculations.
    **default** - SpacerType.net_spacer.
    **Valid values:** {
        no_spacer: Converts RO unit model to a case with no spacer
        net_spacer: Adds necessary correlations for net_spacers
        printed_spacer: Adds 3D printed spacer and necessary correlations}""",
        ),
    )

    CONFIG.declare(
        "sherwood_correlation",
        ConfigValue(
            default=SherwoodCorrelation.guillen,
            domain=In(SherwoodCorrelation),
            description="Argument for selecting the Sherwood correlation to be used for mass transfer calculation",
            doc="""This argument selects the Sherwood correlation to be used in the feed side mass transfer coefficient
            calculation. This will effect the concentration polarization modulus.
    **default** - SherwoodCorrelation.guillen.
    **Valid values:** {
        graetz: Graetz-Leveque correlation is used for mass transfer correlation. Exclusive for no spacer case
        sieder: Sieder-Tate correlation is used for mass transfer correlation. Exclusive for no spacer case
        guillen: Guillen & Hoek correlation is used for mass transfer correlation. Standard case in WaterTAP model
        schock: Schock & Miquel correlation is used for mass transfer correlation.
        da_costa: DaCosta correlation is used for mass transfer correlation.
        parameterized: Parameterizes the Sherwood number
        References: https://doi.org/10.1016/j.cej.2008.10.030 - Guillen & Hoek
                    Da Costa, Andre. Fluid flow and mass transfer in spacer-filled
                    channels for ultrafiltration. Diss. UNSW Sydney, 1993. - DaCosta and no_spacer correlations
                    1) Table 5.1 for Graetz-Leveque and Sieder-Tate correlations
                    2) Equation 5.20 for DaCosta Sherwood correlation}""",
        ),
    )

    CONFIG.declare(
        "pressure_correlation",
        ConfigValue(
            default=PressureCorrelation.guillen,
            domain=In(PressureCorrelation),
            description="Argument for selecting the feed side axial pressure drop equation",
            doc="""This argument changes the correlation used to calculate the axial pressure drop on the feed side.
    **default** - PressureCorrelation.guillen.
    **Valid values:** {
    no_spacer: Hagen-Poiseuille correlation is used for pressure drop correlation.
    guillen: Guillen & Hoek correlation is used for pressure drop calculation. Standard case in WaterTAP model
    schock: Schock & Miquel correlation is used for pressure drop calculation.
    da_costa: DaCosta correlation is used for pressure drop calculation.
    parameterized: Parameterizes the Power number 
    References: https://doi.org/10.1016/j.cej.2008.10.030 - Guillen & Hoek
                Da Costa, Andre. Fluid flow and mass transfer in spacer-filled
                channels for ultrafiltration. Diss. UNSW Sydney, 1993. - DaCosta and no_spacer correlations
                1) Equation 4.12 for no spacer pressure drop correlation
                2) Equation 4.27 for DaCosta pressure drop correlation
                    Assumptions:   Cd value is assumed as 10 based on the justification from DaCosta thesis page 116
                    k_theta values in the third term is assumed to be 0.98 according to the assumption in DaCosta
                    thesis for turbulent regime, the values for laminar flow are higher.
                    NF is normalized to length so the overall dp_dx equation gives pressure drop per unit length}""",
        ),
    )


class SpacerType(Enum):
    """
    no_spacer: Converts RO unit model to a case with no spacer
    net_spacer: Adds necessary correlations for net_spacers
    printed_spacer: Adds 3D printed spacer and necessary correlations
    """

    no_spacer = auto()
    net_spacer = auto()
    printed_spacer = auto()


class SherwoodCorrelation(Enum):
    """
    graetz: Graetz-Leveque correlation is used for mass transfer correlation. Exclusive for no spacer case
    sieder: Sieder-Tate correlation is used for mass transfer correlation. Exclusive for no spacer case
    guillen: Guillen & Hoek correlation is used for mass transfer correlation. Standard case in WaterTAP model
    schock: Schock & Miquel correlation is used for mass transfer correlation.
    da_costa: DaCosta correlation is used for mass transfer correlation.
    parameterized: Parameterizes the Sherwood number
    References: https://doi.org/10.1016/j.cej.2008.10.030 - Guillen & Hoek
                Da Costa, Andre. Fluid flow and mass transfer in spacer-filled
                channels for ultrafiltration. Diss. UNSW Sydney, 1993. - DaCosta and no_spacer correlations
                1) Table 5.1 for Graetz-Leveque and Sieder-Tate correlations
                2) Equation 5.20 for DaCosta Sherwood correlation
    """

    graetz = auto()
    sieder = auto()
    guillen = auto()
    schock = auto()
    da_costa = auto()
    parameterized = auto()


class PressureCorrelation(Enum):
    """
    no_spacer: Hagen-Poiseuille correlation is used for pressure drop correlation.
    guillen: Guillen & Hoek correlation is used for pressure drop calculation. Standard case in WaterTAP model
    schock: Schock & Miquel correlation is used for pressure drop calculation.
    da_costa: DaCosta correlation is used for pressure drop calculation.
    parameterized: Parameterizes the Power number
    References: https://doi.org/10.1016/j.cej.2008.10.030 - Guillen & Hoek
                Da Costa, Andre. Fluid flow and mass transfer in spacer-filled
                channels for ultrafiltration. Diss. UNSW Sydney, 1993. - DaCosta and no_spacer correlations
                1) Equation 4.12 for no spacer pressure drop correlation
                2) Equation 4.27 for DaCosta pressure drop correlation
                    Assumptions:   Cd value is assumed as 10 based on the justification from DaCosta thesis page 116
                    k_theta values in the third term is assumed to be 0.98 according to the assumption in DaCosta
                    thesis for turbulent regime, the values for laminar flow are higher.
                    NF is normalized to length so the overall dp_dx equation gives pressure drop per unit length
    """

    no_spacer = auto()
    guillen = auto()
    schock = auto()
    da_costa = auto()
    parameterized = auto()


class CustomSpacer:
    def apply_custom_spacer_correlations(self):
        # Adds the necessary variables based on the choices provided
        self._add_variables()
        # Adds the necessary constraints relevant for the case
        if self.config.spacer_type == SpacerType.no_spacer:
            self._add_no_spacer_constraints()
        elif self.config.spacer_type == SpacerType.net_spacer:
            self._add_net_spacer_constraints()

        # Adds equality constraints needed to calculate the pressure drop and sherwood number for the DaCosta case
        if self.config.sherwood_correlation == SherwoodCorrelation.da_costa:
            self._add_da_costa_constraints()
        elif self.config.pressure_correlation == PressureCorrelation.da_costa:
            self._add_da_costa_constraints()

        # Adds eq_N_Sh_modified and eq_dP_dx_modified to account for the new case and activates them
        self._add_sherwood_correlations()
        self._add_pressure_drop_correlations()

        # Deletes the net_spacer geometry and constraints for porosity calculation to parameterize spacer porosity along
        # with the Sh and Pn values. User needs to fix spacer porosity instead of spacer geometry features
        if (
            self.config.sherwood_correlation == SherwoodCorrelation.parameterized
            and self.config.pressure_correlation == PressureCorrelation.parameterized
        ):
            self._adjust_for_sh_pn_parameterization()

        # self._add_spacer_performance_expressions()
        # TODO: Finish adding the methods for spacer related reporting

    def _add_sherwood_correlations(self):
        # Modifying the Sherwood correlation based on the user input
        solute_set = self.feed_side.config.property_package.solute_set

        @self.feed_side.Constraint(
            self.flowsheet().config.time,
            self.length_domain,
            solute_set,
            doc="Sherwood number equation modified to "
            + str(self.config.sherwood_correlation),
        )
        def eq_N_Sh_comp_modified(b, t, x, j):
            if self.config.spacer_type == SpacerType.no_spacer:
                if self.config.sherwood_correlation == SherwoodCorrelation.sieder:
                    return b.N_Sh_comp[t, x, j] == (
                        1.86
                        * (b.N_Re[t, x] ** 0.33)
                        * (b.N_Sc_comp[t, x, j] ** 0.33)
                        * ((b.dh * self.nfe / self.length) ** 0.33)
                    )
                elif self.config.sherwood_correlation == SherwoodCorrelation.graetz:
                    return b.N_Sh_comp[t, x, j] == (
                        1.62
                        * (b.N_Re[t, x] ** 0.33)
                        * (b.N_Sc_comp[t, x, j] ** 0.33)
                        * ((b.dh * self.nfe / self.length) ** 0.33)
                    )

            elif self.config.spacer_type == SpacerType.net_spacer:
                if self.config.sherwood_correlation == SherwoodCorrelation.guillen:
                    # Not necessary to give an option for guillen as it is the standard case but the spacer porosity
                    # constraint is added when net_spacer is selected in this model so I am duplicating this constraint
                    # with modified tag to make it a general case.
                    return b.N_Sh_comp[t, x, j] == (
                        0.46 * (b.N_Re[t, x] ** 0.36) * (b.N_Sc_comp[t, x, j] ** 0.36)
                    )
                elif self.config.sherwood_correlation == SherwoodCorrelation.schock:
                    return b.N_Sh_comp[t, x, j] == (
                        0.065 * (b.N_Re[t, x] ** 0.875) * (b.N_Sc_comp[t, x, j] ** 0.25)
                    )
                elif self.config.sherwood_correlation == SherwoodCorrelation.da_costa:
                    return b.N_Sh_comp[t, x, j] == (
                        0.664
                        * b.Sh_corr_factor_da_costa
                        * (b.N_Re[t, x] ** 0.5)
                        * (b.N_Sc_comp[t, x, j] ** 0.33)
                        * ((2 * b.dh / b.mesh_size) ** 0.5)
                    )
                elif (
                    self.config.sherwood_correlation
                    == SherwoodCorrelation.parameterized
                ):
                    return b.N_Sh_comp[t, x, j] == b.Sh_param

        # ===============================================================================================================
        self.feed_side.eq_N_Sh_comp.deactivate()
        self.feed_side.eq_N_Sh_comp_modified.activate()

    def _add_pressure_drop_correlations(self):
        # Modifying the Pressure drop correlations based on the user input
        if self.config.pressure_correlation != PressureCorrelation.da_costa:
            self._modify_friction_factor()
            self.feed_side.eq_friction_factor.deactivate()
            self.feed_side.eq_friction_factor_modified.activate()

        elif self.config.pressure_correlation == PressureCorrelation.da_costa:

            @self.feed_side.Constraint(
                self.flowsheet().config.time,
                self.length_domain,
                doc="Modified pressure drop equation"
                + str(self.config.pressure_correlation),
            )
            def eq_dP_dx_modified(b, t, x):
                return (
                    b.dP_dx[t, x]
                    == b.dP_dx_viscous_drag[t, x]
                    + b.dP_dx_wall_viscous_drag[t, x]
                    + b.dP_dx_form_drag[t, x]
                    + b.dP_dx_kinetic_losses[t, x]
                )

            # ===========================================================================================================
            self.feed_side.eq_dP_dx.deactivate()
            self.feed_side.eq_dP_dx_modified.activate()
            # Friction factor is not used for pressure drop calculation in the DaCosta case
            self.feed_side.eq_friction_factor.deactivate()

    def _modify_friction_factor(self):
        # Modifying the friction factor based on the specified case
        @self.feed_side.Constraint(
            self.flowsheet().config.time,
            self.length_domain,
            doc="Modified friction factor" + str(self.config.pressure_correlation),
        )
        def eq_friction_factor_modified(b, t, x):
            if self.config.spacer_type == SpacerType.no_spacer:
                return b.friction_factor_darcy[t, x] * b.N_Re[t, x] == 64
            elif self.config.spacer_type == SpacerType.net_spacer:
                if self.config.pressure_correlation == PressureCorrelation.guillen:
                    return (b.friction_factor_darcy[t, x] - 0.42) * b.N_Re[
                        t, x
                    ] == 189.3
                elif self.config.pressure_correlation == PressureCorrelation.schock:
                    return (
                        b.friction_factor_darcy[t, x] * b.N_Re[t, x] ** -0.3
                    ) == 6.23
                elif (
                    self.config.pressure_correlation
                    == PressureCorrelation.parameterized
                ):
                    return (
                        b.friction_factor_darcy[t, x] * b.N_Re[t, x] ** 3
                        == 2 * b.Pn_param
                    )
            # ===========================================================================================================

    def _add_variables(self):
        # Adds variables necessary for net_type spacers when the class or method is called
        if self.config.spacer_type == SpacerType.net_spacer:
            self.feed_side.spacer_angle = Var(
                initialize=30,
                units=pyunits.degrees,
                bounds=(0, 180),
                doc="spacer_angle with respect to the flow direction",
            )  # Angle is defined in degrees

            self.feed_side.spacer_angle_radians = pyunits.convert(
                self.feed_side.spacer_angle, to_units=pyunits.radians
            )

            self.feed_side.mesh_size = Var(
                initialize=1e-3,
                units=pyunits.m,
                within=PositiveReals,
                doc="spacer mesh size",
            )

            self.feed_side.filament_dia = Var(
                initialize=2e-4,
                units=pyunits.m,
                within=PositiveReals,
                doc="spacer filament diameter",
            )

            self.feed_side.spacer_height = Var(
                initialize=1e-3,
                units=pyunits.m,
                within=PositiveReals,
                doc="spacer height",
            )

        if self.config.sherwood_correlation == SherwoodCorrelation.da_costa:
            self.feed_side.Sh_corr_factor_da_costa = Var(
                initialize=1,
                units=pyunits.dimensionless,
                within=PositiveReals,
                doc="Mass transfer correction factor",
            )

        if self.config.pressure_correlation == PressureCorrelation.da_costa:
            self.feed_side.nf_value = Var(
                initialize=1,
                units=pyunits.dimensionless,
                within=PositiveReals,
                doc="Number of filaments",
            )

            self.feed_side.dP_dx_viscous_drag = Var(
                self.flowsheet().config.time,
                self.length_domain,
                bounds=(-1e5, 0),
                domain=Reals,
                units=pyunits.kg / pyunits.m**2 / pyunits.s**2,
                doc="Pressure drop due to viscous drag (DaCosta)",
            )

            self.feed_side.dP_dx_form_drag = Var(
                self.flowsheet().config.time,
                self.length_domain,
                bounds=(-1e5, 0),
                domain=Reals,
                units=pyunits.kg / pyunits.m**2 / pyunits.s**2,
                doc="Pressure drop due to form drag (DaCosta)",
            )

            self.feed_side.dP_dx_kinetic_losses = Var(
                self.flowsheet().config.time,
                self.length_domain,
                bounds=(-1e5, 0),
                domain=Reals,
                units=pyunits.kg / pyunits.m**2 / pyunits.s**2,
                doc="Pressure drop due to kinetic losses (DaCosta)",
            )

            self.feed_side.dP_dx_wall_viscous_drag = Var(
                self.flowsheet().config.time,
                self.length_domain,
                bounds=(-1e5, 0),
                domain=Reals,
                units=pyunits.kg / pyunits.m**2 / pyunits.s**2,
                doc="Pressure drop due to viscous drag on walls (DaCosta)",
            )

        if self.config.sherwood_correlation == SherwoodCorrelation.parameterized:
            self.feed_side.Sh_param = Var(
                initialize=1,
                bounds=(1e-3, 1e3),
                domain=NonNegativeReals,
                units=pyunits.dimensionless,
                doc="Sherwood parameter in RO",
            )

        if self.config.pressure_correlation == PressureCorrelation.parameterized:
            self.feed_side.Pn_param = Var(
                initialize=1e3,
                bounds=(1e-3, 1e6),
                domain=NonNegativeReals,
                units=pyunits.dimensionless,
                doc="Power number parameter in RO",
            )

    def _add_no_spacer_constraints(self):
        # Modify the hydraulic diameter constraint for the no_spacer case
        @self.feed_side.Constraint(doc="Hydraulic diameter for the no spacer case")
        # for rectangular channel when b>>h, dh = 2h
        def eq_dh_no_spacer(b):
            return b.dh == 2 * b.channel_height

        # ===============================================================================================================
        # Activate the hydraulic diameter calculation for no spacer case and deactivate the usual calculation
        self.feed_side.eq_dh.deactivate()
        self.feed_side.eq_dh_no_spacer.activate()
        # Fixes the spacer porosity to 1
        self.feed_side.spacer_porosity.fix(1)

    def _add_net_spacer_constraints(self):
        # Add a constraint to calculate the spacer porosity based on the spacer geometry
        @self.feed_side.Constraint(doc="Spacer porosity")
        # Porosity calculation from Equation 3.7 in Da Costa, Andre. Fluid flow and mass transfer in spacer-filled
        # channels for ultrafiltration. Diss. UNSW Sydney, 1993.
        def eq_spacer_porosity(b):
            return b.spacer_porosity == 1 - pi * b.filament_dia**2 / (
                2 * b.mesh_size * 2 * b.filament_dia * pysin(b.spacer_angle_radians)
            )

        # Constraint for spacer height
        @self.feed_side.Constraint(doc="Spacer height calculation for net spacer")
        def eq_spacer_height(b):
            return b.spacer_height == 2 * b.filament_dia

        @self.feed_side.Constraint(doc="Channel Height")
        def eq_channel_height(b):
            return b.channel_height == 2 * b.filament_dia

        # ===============================================================================================================
        # Activate the constraint for spacer porosity and channel height constraints

        self.feed_side.eq_spacer_porosity.activate()
        calculate_variable_from_constraint(
            self.feed_side.spacer_porosity, self.feed_side.eq_spacer_porosity
        )
        self.feed_side.eq_channel_height.activate()
        calculate_variable_from_constraint(
            self.feed_side.channel_height, self.feed_side.eq_channel_height
        )
        self.feed_side.eq_spacer_height.activate()
        calculate_variable_from_constraint(
            self.feed_side.spacer_height, self.feed_side.eq_spacer_height
        )

    def _add_da_costa_constraints(self):
        if self.config.sherwood_correlation == SherwoodCorrelation.da_costa:
            # Add the Sherwood correction factor for the DaCosta case
            @self.feed_side.Constraint(
                doc="Sherwood correlation correction factor for DaCosta case"
            )
            def eq_Sh_corr_factor_da_costa(b):
                return (
                    b.Sh_corr_factor_da_costa
                    == 1.654
                    * (b.filament_dia / b.channel_height) ** -0.039
                    * b.spacer_porosity**0.75
                    * pysin(pi * b.spacer_angle_radians / 360) ** 0.086
                )

        if self.config.pressure_correlation == PressureCorrelation.da_costa:
            # Add the number of filaments constraint for the DaCosta case
            @self.feed_side.Constraint(doc="Number of filaments per unit length")
            def eq_nf_value(b):
                # Number of filaments per unit length  calculation from Equation 3.15 in
                # Da Costa, Andre. Fluid flow and mass transfer in spacer-filled
                # channels for ultrafiltration. Diss. UNSW Sydney, 1993.
                return b.nf_value == self.length / (
                    (b.mesh_size + b.filament_dia)
                    * pycos(pi * (b.spacer_angle_radians / 360))
                )

            # Add pressure drop calculations per unit length based on DaCosta correlations
            @self.feed_side.Constraint(
                self.flowsheet().config.time,
                self.length_domain,
                doc="Viscous drag from spacer",
            )
            def eq_dP_dx_viscous_drag(b, t, x):
                # First term in Equation 4.27 of Da Costa, Andre. Fluid flow and mass transfer in spacer-filled
                # channels for ultrafiltration. Diss. UNSW Sydney, 1993.
                return (
                    -b.dP_dx_viscous_drag[t, x]
                    == (
                        (
                            2.107
                            * (b.nf_value / b.area)
                            * (
                                (b.filament_dia / 2) ** 3
                                * b.velocity[t, x] ** 3
                                * pysin(pi * b.spacer_angle_radians / 360) ** 3
                                * b.properties[t, x].dens_mass_phase["Liq"]
                                * b.properties[t, x].visc_d_phase["Liq"]
                            )
                            ** 0.5
                        )
                    )
                    / self.length
                )

            @self.feed_side.Constraint(
                self.flowsheet().config.time,
                self.length_domain,
                doc="Form drag from spacer",
            )
            def eq_dP_dx_form_drag(b, t, x):
                # Second term of pressure drop equation
                return (
                    -b.dP_dx_form_drag[t, x]
                    == (
                        (
                            0.5
                            * 10
                            * b.nf_value
                            * b.mesh_size
                            * b.filament_dia
                            * b.properties[t, x].dens_mass_phase["Liq"]
                            * b.velocity[t, x] ** 2
                            * pysin(pi * b.spacer_angle_radians / 360) ** 2
                        )
                        / b.area
                    )
                    / self.length
                )

            @self.feed_side.Constraint(
                self.flowsheet().config.time,
                self.length_domain,
                doc="Kinetic losses from spacer",
            )
            def eq_dP_dx_kinetic_losses(b, t, x):
                # Third term of pressure drop equation
                return (
                    -b.dP_dx_kinetic_losses[t, x]
                    == (
                        (
                            0.5
                            * b.properties[t, x].dens_mass_phase["Liq"]
                            * 0.98
                            * (b.nf_value - 1)
                            * ((b.mesh_size * b.spacer_height) / (2 * b.area))
                            * (
                                b.velocity[t, x]
                                / pycos(pi * b.spacer_angle_radians / 360)
                            )
                            ** 2
                        )
                    )
                    / self.length
                )

            @self.feed_side.Constraint(
                self.flowsheet().config.time,
                self.length_domain,
                doc="Viscous drag on channel walls",
            )
            def eq_dP_dx_wall_viscous_drag(b, t, x):
                # Fourth term of pressure drop equation
                return -b.dP_dx_wall_viscous_drag[t, x] == (
                    (
                        12
                        * b.velocity[t, x]
                        * b.properties[t, x].visc_d_phase["Liq"]
                        * b.spacer_porosity
                        / (b.channel_height**2)
                    )
                )

        # ===============================================================================================================
        # Activate the new constraints
        # TODO: Group the da_costa constraints to handle them easily
        if self.config.sherwood_correlation == SherwoodCorrelation.da_costa:
            self.feed_side.eq_Sh_corr_factor_da_costa.activate()
        if self.config.pressure_correlation == PressureCorrelation.da_costa:
            self.feed_side.eq_nf_value.activate()
            self.feed_side.eq_dP_dx_viscous_drag.activate()
            self.feed_side.eq_dP_dx_form_drag.activate()
            self.feed_side.eq_dP_dx_kinetic_losses.activate()
            self.feed_side.eq_dP_dx_wall_viscous_drag.activate()

    def _adjust_for_sh_pn_parameterization(self):
        del self.feed_side.mesh_size
        del self.feed_side.filament_dia
        del self.feed_side.spacer_angle
        del self.feed_side.eq_spacer_porosity

    # The following functions mimic the build of RO 1D model
    def _add_feed_side_membrane_channel_and_geometry(self):
        # Mimics the build for a standard RO 1D model
        self.feed_side = MembraneChannel1DBlock(
            dynamic=self.config.dynamic,
            has_holdup=self.config.has_holdup,
            area_definition=self.config.area_definition,
            property_package=self.config.property_package,
            property_package_args=self.config.property_package_args,
            transformation_method=self.config.transformation_method,
            transformation_scheme=self.config.transformation_scheme,
            finite_elements=self.config.finite_elements,
            collocation_points=self.config.collocation_points,
        )
        self._add_length_and_width()
        self.feed_side.add_geometry(length_var=self.length, width_var=self.width)
        self._add_area(include_constraint=True)

    def _add_deltaP(self):
        # Mimics the build for a standard RO 1D model
        units_meta = self.config.property_package.get_metadata().get_derived_units
        self.deltaP = Var(
            self.flowsheet().config.time,
            initialize=-1e5,
            bounds=(-1e6, 0),
            domain=NegativeReals,
            units=units_meta("pressure"),
            doc="Pressure drop across unit",
        )

        @self.Constraint(self.flowsheet().config.time, doc="Pressure drop across unit")
        def eq_pressure_drop(b, t):
            return b.deltaP[t] == sum(
                b.feed_side.dP_dx[t, x] * b.length / b.nfe
                for x in b.difference_elements
            )

    def _add_mass_transfer(self):
        # Mimics the build for a standard RO 1D model
        units_meta = self.config.property_package.get_metadata().get_derived_units

        def mass_transfer_phase_comp_initialize(b, t, x, p, j):
            return value(
                self.feed_side.properties[t, x].get_material_flow_terms("Liq", j)
                * self.recovery_mass_phase_comp[t, "Liq", j]
            )

        self.mass_transfer_phase_comp = Var(
            self.flowsheet().config.time,
            self.difference_elements,
            self.config.property_package.phase_list,
            self.config.property_package.component_list,
            initialize=mass_transfer_phase_comp_initialize,
            bounds=(0.0, 1e6),
            domain=NonNegativeReals,
            units=units_meta("mass")
            * units_meta("time") ** -1
            * units_meta("length") ** -1,
            doc="Mass transfer to permeate",
        )

        # ==========================================================================
        # Mass transfer term equation
        @self.Constraint(
            self.flowsheet().config.time,
            self.difference_elements,
            self.config.property_package.phase_list,
            self.config.property_package.component_list,
            doc="Mass transfer term",
        )
        def eq_mass_transfer_term(b, t, x, p, j):
            return (
                b.mass_transfer_phase_comp[t, x, p, j]
                == -b.feed_side.mass_transfer_term[t, x, p, j]
            )

        # ==========================================================================
        # Feed and permeate-side mass transfer connection --> Mp,j = Mf,transfer = Jj * W * L/n
        @self.Constraint(
            self.flowsheet().config.time,
            self.difference_elements,
            self.config.property_package.phase_list,
            self.config.property_package.component_list,
            doc="Mass transfer from feed to permeate",
        )
        def eq_connect_mass_transfer(b, t, x, p, j):
            return (
                b.permeate_side[t, x].get_material_flow_terms(p, j)
                == -b.feed_side.mass_transfer_term[t, x, p, j] * b.length / b.nfe
            )

        # ==========================================================================
        # Mass flux = feed mass transfer equation
        @self.Constraint(
            self.flowsheet().config.time,
            self.difference_elements,
            self.config.property_package.phase_list,
            self.config.property_package.component_list,
            doc="Mass transfer term",
        )
        def eq_mass_flux_equal_mass_transfer(b, t, x, p, j):
            return (
                b.flux_mass_phase_comp[t, x, p, j] * b.area
                == -b.feed_side.mass_transfer_term[t, x, p, j] * b.length
            )

        # ==========================================================================
        # Final permeate mass flow rate (of solvent and solute) --> Mp,j, final = sum(Mp,j)

        @self.Constraint(
            self.flowsheet().config.time,
            self.config.property_package.phase_list,
            self.config.property_package.component_list,
            doc="Permeate mass flow rates exiting unit",
        )
        def eq_permeate_production(b, t, p, j):
            return b.mixed_permeate[t].get_material_flow_terms(p, j) == sum(
                b.permeate_side[t, x].get_material_flow_terms(p, j)
                for x in b.difference_elements
            )
