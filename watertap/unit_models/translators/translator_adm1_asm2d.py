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
Translator block representing the ADM1/ASM2d interface.
This is copied from the Generic template for a translator block.

Assumptions:
     * Steady-state only

Model formulated from:

Flores-Alsina, X., Solon, K., Mbamba, C.K., Tait, S., Gernaey, K.V., Jeppsson, U. and Batstone, D.J., 2016.
Modelling phosphorus (P), sulfur (S) and iron (Fe) interactions for dynamic simulations of anaerobic digestion processes.
Water Research, 95, pp.370-382.
"""

# Import Pyomo libraries
from pyomo.common.config import ConfigBlock, ConfigValue

# Import IDAES cores
from idaes.core import declare_process_block_class
from idaes.models.unit_models.translator import TranslatorData
from idaes.core.util.config import (
    is_reaction_parameter_block,
)
from idaes.core.util.model_statistics import degrees_of_freedom
from watertap.core.solvers import get_solver
import idaes.logger as idaeslog
import idaes.core.util.scaling as iscale

from idaes.core.scaling import CustomScalerBase, ConstraintScalingScheme


from idaes.core.util.exceptions import InitializationError

from pyomo.environ import (
    Constraint,
    units as pyunits,
    check_optimal_termination,
    Set,
    Param,
)

__author__ = "Chenyu Wang, Marcus Holly, Xinhong Liu"


# Set up logger
_log = idaeslog.getLogger(__name__)


class ADM1ASM2dScaler(CustomScalerBase):
    """
    Default modular scaler for ADM1-ASM2d translator block.
    This Scaler relies on the associated property and reaction packages,
    either through user provided options (submodel_scalers argument) or by default
    Scalers assigned to the packages.
    """

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
            submodel=model.properties_in,
            method="variable_scaling_routine",
            submodel_scalers=submodel_scalers,
            overwrite=overwrite,
        )

        self.call_submodel_scaler_method(
            submodel=model.properties_out,
            method="variable_scaling_routine",
            submodel_scalers=submodel_scalers,
            overwrite=overwrite,
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
            submodel=model.properties_in,
            method="constraint_scaling_routine",
            submodel_scalers=submodel_scalers,
            overwrite=overwrite,
        )
        self.call_submodel_scaler_method(
            submodel=model.properties_out,
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


@declare_process_block_class("Translator_ADM1_ASM2D")
class TranslatorDataADM1ASM2D(TranslatorData):
    """
    Translator block representing the ADM1/ASM2D interface
    """

    default_scaler = ADM1ASM2dScaler

    CONFIG = TranslatorData.CONFIG()

    CONFIG.declare(
        "inlet_reaction_package",
        ConfigValue(
            default=None,
            domain=is_reaction_parameter_block,
            description="Reaction package to use for control volume",
            doc="""Reaction parameter object used to define reaction calculations,
    **default** - None.
    **Valid values:** {
    **None** - no reaction package,
    **ReactionParameterBlock** - a ReactionParameterBlock object.}""",
        ),
    )
    CONFIG.declare(
        "inlet_reaction_package_args",
        ConfigBlock(
            implicit=True,
            description="Arguments to use for constructing reaction packages",
            doc="""A ConfigBlock with arguments to be passed to a reaction block(s)
    and used when constructing these,
    **default** - None.
    **Valid values:** {
    see reaction package for documentation.}""",
        ),
    )
    CONFIG.declare(
        "outlet_reaction_package",
        ConfigValue(
            default=None,
            domain=is_reaction_parameter_block,
            description="Reaction package to use for control volume",
            doc="""Reaction parameter object used to define reaction calculations,
        **default** - None.
        **Valid values:** {
        **None** - no reaction package,
        **ReactionParameterBlock** - a ReactionParameterBlock object.}""",
        ),
    )
    CONFIG.declare(
        "outlet_reaction_package_args",
        ConfigBlock(
            implicit=True,
            description="Arguments to use for constructing reaction packages",
            doc="""A ConfigBlock with arguments to be passed to a reaction block(s)
        and used when constructing these,
        **default** - None.
        **Valid values:** {
        see reaction package for documentation.}""",
        ),
    )

    def build(self):
        """
        Begin building model.
        Args:
            None
        Returns:
            None
        """
        # Call UnitModel.build to setup dynamics
        super(TranslatorDataADM1ASM2D, self).build()

        eps = 0
        mw_p = 31 * pyunits.kg / pyunits.kmol
        mw_n = 14 * pyunits.kg / pyunits.kmol
        mw_c = 12 * pyunits.kg / pyunits.kmol
        mw_XPP = 300.41 * pyunits.kg / pyunits.kmol
        mw_k = 39.1 * pyunits.kg / pyunits.kmol
        mw_mg = 24.3 * pyunits.kg / pyunits.kmol

        @self.Constraint(
            self.flowsheet().time,
            doc="Equality volumetric flow equation",
        )
        def eq_flow_vol_rule(blk, t):
            return blk.properties_out[t].flow_vol == blk.properties_in[t].flow_vol

        @self.Constraint(
            self.flowsheet().time,
            doc="Equality temperature equation",
        )
        def eq_temperature_rule(blk, t):
            return blk.properties_out[t].temperature == blk.properties_in[t].temperature

        @self.Constraint(
            self.flowsheet().time,
            doc="Equality pressure equation",
        )
        def eq_pressure_rule(blk, t):
            return blk.properties_out[t].pressure == blk.properties_in[t].pressure

        # -------------------------------------------Step 1----------------------------------------------------------------#
        @self.Expression(
            self.flowsheet().time,
            doc="Biomass concentration (kgCOD/m3)",
        )
        def biomass(blk, t):
            return (
                blk.properties_in[t].conc_mass_comp["X_su"]
                + blk.properties_in[t].conc_mass_comp["X_aa"]
                + blk.properties_in[t].conc_mass_comp["X_fa"]
                + blk.properties_in[t].conc_mass_comp["X_c4"]
                + blk.properties_in[t].conc_mass_comp["X_pro"]
                + blk.properties_in[t].conc_mass_comp["X_ac"]
                + blk.properties_in[t].conc_mass_comp["X_h2"]
                + blk.properties_in[t].conc_mass_comp["X_PAO"]
            )

        @self.Expression(
            self.flowsheet().time, doc="S_ac concentration (kgCOD/m3) step 1"
        )
        def Sac_AD1(blk, t):
            return (
                blk.properties_in[t].conc_mass_comp["S_ac"]
                + blk.properties_in[t].conc_mass_comp["X_PHA"]
            )

        @self.Expression(
            self.flowsheet().time, doc="S_IC concentration at (kmolC/m3) step 1"
        )
        def SIC_AD1(blk, t):
            return (
                blk.config.inlet_reaction_package.Ci["X_su"] * blk.biomass[t]
                - (
                    blk.config.inlet_reaction_package.f_sI_xc
                    * blk.config.inlet_reaction_package.Ci["S_I"]
                    * blk.biomass[t]
                )
                - (
                    blk.config.inlet_reaction_package.f_ch_xc
                    * blk.biomass[t]
                    * blk.config.inlet_reaction_package.Ci["X_ch"]
                )
                - (
                    blk.config.inlet_reaction_package.f_pr_xc
                    * blk.biomass[t]
                    * blk.config.inlet_reaction_package.Ci["X_pr"]
                )
                - (
                    blk.config.inlet_reaction_package.f_li_xc
                    * blk.biomass[t]
                    * blk.config.inlet_reaction_package.Ci["X_li"]
                )
                - (
                    blk.config.inlet_reaction_package.f_xI_xc
                    * blk.biomass[t]
                    * blk.config.inlet_reaction_package.Ci["X_I"]
                )
                + (
                    blk.properties_in[t].conc_mass_comp["X_PHA"]
                    * blk.config.inlet_reaction_package.Ci["X_PHA"]
                )
                - (
                    blk.properties_in[t].conc_mass_comp["X_PHA"]
                    * blk.config.inlet_reaction_package.Ci["S_ac"]
                )
            )

        @self.Expression(
            self.flowsheet().time, doc="S_IN concentration (kmolN/m3) step 1"
        )
        def SIN_AD1(blk, t):
            return (
                blk.config.inlet_reaction_package.Ni["X_su"] * blk.biomass[t]
                - (
                    blk.config.inlet_reaction_package.f_sI_xc
                    * blk.biomass[t]
                    * blk.config.inlet_reaction_package.Ni["S_I"]
                )
                - (
                    blk.config.inlet_reaction_package.f_pr_xc
                    * blk.biomass[t]
                    * blk.config.inlet_reaction_package.Ni["X_pr"]
                )
                - (
                    blk.config.inlet_reaction_package.f_xI_xc
                    * blk.biomass[t]
                    * blk.config.inlet_reaction_package.Ni["X_I"]
                )
            )

        @self.Expression(
            self.flowsheet().time, doc="S_I concentration (kgCOD/m3) step 1"
        )
        def SI_AD1(blk, t):
            return blk.properties_in[t].conc_mass_comp["S_I"] + (
                blk.config.inlet_reaction_package.f_sI_xc * blk.biomass[t]
            )

        @self.Expression(
            self.flowsheet().time, doc="X_ch concentration (kgCOD/m3) step 1"
        )
        def Xch_AD1(blk, t):
            return blk.properties_in[t].conc_mass_comp["X_ch"] + (
                blk.config.inlet_reaction_package.f_ch_xc * blk.biomass[t]
            )

        @self.Expression(
            self.flowsheet().time, doc="X_pr concentration (kgCOD/m3) step 1"
        )
        def Xpr_AD1(blk, t):
            return blk.properties_in[t].conc_mass_comp["X_pr"] + (
                blk.config.inlet_reaction_package.f_pr_xc * blk.biomass[t]
            )

        @self.Expression(
            self.flowsheet().time, doc="X_li concentration (kgCOD/m3) step 1"
        )
        def Xli_AD1(blk, t):
            return blk.properties_in[t].conc_mass_comp["X_li"] + (
                blk.config.inlet_reaction_package.f_li_xc * blk.biomass[t]
            )

        @self.Expression(
            self.flowsheet().time, doc="X_I concentration (kgCOD/m3) step 1"
        )
        def XI_AD1(blk, t):
            return blk.properties_in[t].conc_mass_comp["X_I"] + (
                blk.config.inlet_reaction_package.f_xI_xc * blk.biomass[t]
            )

        @self.Expression(
            self.flowsheet().time, doc="S_IP concentration at (kmolP/m3) step 1"
        )
        def SIP_AD1(blk, t):
            return (
                blk.properties_in[t].conc_mass_comp["X_PP"] / mw_XPP
                + (blk.config.inlet_reaction_package.Pi["X_su"] * blk.biomass[t])
                - (
                    blk.config.inlet_reaction_package.f_sI_xc
                    * blk.biomass[t]
                    * blk.config.inlet_reaction_package.Pi["S_I"]
                )
                - (
                    blk.config.inlet_reaction_package.f_ch_xc
                    * blk.biomass[t]
                    * blk.config.inlet_reaction_package.P_ch
                )
                - (
                    blk.config.inlet_reaction_package.f_li_xc
                    * blk.biomass[t]
                    * blk.config.inlet_reaction_package.Pi["X_li"]
                )
                - (
                    blk.config.inlet_reaction_package.f_xI_xc
                    * blk.biomass[t]
                    * blk.config.inlet_reaction_package.Pi["X_I"]
                )
            )

        self.XPHA_AD1 = Param(
            initialize=eps,
            units=pyunits.kg / pyunits.m**3,
            mutable=True,
            doc="Mass concentration of X_PHA at step 1",
        )

        self.XPP_AD1 = Param(
            initialize=eps,
            units=pyunits.kmol / pyunits.m**3,
            mutable=True,
            doc="Molar concentration of X_PP at step 1",
        )

        self.XPAO_AD1 = Param(
            initialize=eps,
            units=pyunits.kg / pyunits.m**3,
            mutable=True,
            doc="Mass concentration of X_PAO at step 1",
        )

        # -------------------------------------------Step 2----------------------------------------------------------------#

        @self.Expression(
            self.flowsheet().time, doc="S_IC concentration at (kmolC/m3) step 2"
        )
        def SIC_AD2(blk, t):
            return (
                blk.Xch_AD1[t] * blk.config.inlet_reaction_package.Ci["X_ch"]
                + (blk.Xpr_AD1[t] * blk.config.inlet_reaction_package.Ci["X_pr"])
                + (blk.Xli_AD1[t] * blk.config.inlet_reaction_package.Ci["X_li"])
                - blk.config.outlet_reaction_package.i_CXS
                / mw_c
                * (blk.Xch_AD1[t] + blk.Xpr_AD1[t] + blk.Xli_AD1[t])
                + blk.properties_in[t].conc_mass_comp["S_su"]
                * blk.config.inlet_reaction_package.Ci["S_su"]
                + blk.properties_in[t].conc_mass_comp["S_aa"]
                * blk.config.inlet_reaction_package.Ci["S_aa"]
                + blk.properties_in[t].conc_mass_comp["S_fa"]
                * blk.config.inlet_reaction_package.Ci["S_fa"]
                - blk.config.outlet_reaction_package.i_CSF
                / mw_c
                * (
                    blk.properties_in[t].conc_mass_comp["S_su"]
                    + blk.properties_in[t].conc_mass_comp["S_aa"]
                    + blk.properties_in[t].conc_mass_comp["S_fa"]
                )
                + blk.properties_in[t].conc_mass_comp["S_va"]
                * blk.config.inlet_reaction_package.Ci["S_va"]
                + blk.properties_in[t].conc_mass_comp["S_bu"]
                * blk.config.inlet_reaction_package.Ci["S_bu"]
                + blk.properties_in[t].conc_mass_comp["S_pro"]
                * blk.config.inlet_reaction_package.Ci["S_pro"]
                + blk.Sac_AD1[t] * blk.config.inlet_reaction_package.Ci["S_ac"]
                - blk.config.outlet_reaction_package.i_CSA
                / mw_c
                * (
                    blk.properties_in[t].conc_mass_comp["S_va"]
                    + blk.properties_in[t].conc_mass_comp["S_bu"]
                    + blk.properties_in[t].conc_mass_comp["S_pro"]
                    + blk.Sac_AD1[t]
                )
            )

        @self.Expression(
            self.flowsheet().time, doc="S_IN concentration at (kmolN/m3) step 2"
        )
        def SIN_AD2(blk, t):
            return (
                blk.Xpr_AD1[t] * blk.config.inlet_reaction_package.Ni["X_pr"]
                - blk.config.outlet_reaction_package.i_NXS
                / mw_n
                * (blk.Xch_AD1[t] + blk.Xpr_AD1[t] + blk.Xli_AD1[t])
                + blk.properties_in[t].conc_mass_comp["S_aa"]
                * blk.config.inlet_reaction_package.Ni["S_aa"]
                - blk.config.outlet_reaction_package.i_NSF
                / mw_n
                * (
                    blk.properties_in[t].conc_mass_comp["S_su"]
                    + blk.properties_in[t].conc_mass_comp["S_fa"]
                    + blk.properties_in[t].conc_mass_comp["S_va"]
                )
            )

        @self.Expression(
            self.flowsheet().time, doc="S_IP concentration at (kmolP/m3) step 2"
        )
        def SIP_AD2(blk, t):
            return (
                self.XPP_AD1
                + blk.Xch_AD1[t] * blk.config.inlet_reaction_package.P_ch
                + blk.Xli_AD1[t] * blk.config.inlet_reaction_package.Pi["X_li"]
                - blk.config.outlet_reaction_package.i_PXS
                / mw_p
                * (blk.Xch_AD1[t] + blk.Xpr_AD1[t] + blk.Xli_AD1[t])
                - blk.config.outlet_reaction_package.i_PSF
                / mw_p
                * (
                    blk.properties_in[t].conc_mass_comp["S_su"]
                    + blk.properties_in[t].conc_mass_comp["S_fa"]
                    + blk.properties_in[t].conc_mass_comp["S_va"]
                )
            )

        @self.Expression(
            self.flowsheet().time, doc="S_K concentration at (kmolK/m3) step 2"
        )
        def SK_AD2(blk, t):
            return blk.properties_in[t].conc_mass_comp["S_K"] / mw_k + self.XPP_AD1 / 3

        @self.Expression(
            self.flowsheet().time, doc="S_Mg concentration at (kmolMg/m3) step 2"
        )
        def SMg_AD2(blk, t):
            return (
                blk.properties_in[t].conc_mass_comp["S_Mg"] / mw_mg + self.XPP_AD1 / 3
            )

        @self.Constraint(
            self.flowsheet().time, doc="S_F concentration output (kgCOD/m3)"
        )
        def SF_output(blk, t):
            return (
                blk.properties_out[t].conc_mass_comp["S_F"]
                == blk.properties_in[t].conc_mass_comp["S_su"]
                + blk.properties_in[t].conc_mass_comp["S_aa"]
                + blk.properties_in[t].conc_mass_comp["S_fa"]
            )

        @self.Constraint(
            self.flowsheet().time, doc="S_A concentration output (kgCOD/m3)"
        )
        def SA_output(blk, t):
            return (
                blk.properties_out[t].conc_mass_comp["S_A"]
                == blk.properties_in[t].conc_mass_comp["S_va"]
                + blk.properties_in[t].conc_mass_comp["S_bu"]
                + blk.properties_in[t].conc_mass_comp["S_pro"]
                + blk.Sac_AD1[t]
            )

        @self.Constraint(
            self.flowsheet().time, doc="S_I concentration output (kgCOD/m3)"
        )
        def SI_output(blk, t):
            return blk.properties_out[t].conc_mass_comp["S_I"] == blk.SI_AD1[t]

        @self.Constraint(
            self.flowsheet().time, doc="S_NH4 concentration output (kgN/m3)"
        )
        def SNH4_output(blk, t):
            return blk.properties_out[t].conc_mass_comp["S_NH4"] == mw_n * (
                blk.properties_in[t].conc_mass_comp["S_IN"] / mw_n
                + blk.SIN_AD1[t]
                + blk.SIN_AD2[t]
            )

        @self.Constraint(
            self.flowsheet().time, doc="S_PO4 concentration output (kgP/m3)"
        )
        def SPO4_output(blk, t):
            return blk.properties_out[t].conc_mass_comp["S_PO4"] == mw_p * (
                blk.properties_in[t].conc_mass_comp["S_IP"] / mw_p
                + blk.SIP_AD1[t]
                + blk.SIP_AD2[t]
            )

        @self.Constraint(
            self.flowsheet().time, doc="S_IC concentration output (kgC/m3)"
        )
        def SIC_output(blk, t):
            return blk.properties_out[t].conc_mass_comp["S_IC"] == mw_c * (
                blk.properties_in[t].conc_mass_comp["S_IC"] / mw_c
                + blk.SIC_AD1[t]
                + blk.SIC_AD2[t]
            )

        @self.Constraint(
            self.flowsheet().time, doc="X_I concentration output (kgCOD/m3)"
        )
        def XI_output(blk, t):
            return blk.properties_out[t].conc_mass_comp["X_I"] == blk.XI_AD1[t]

        @self.Constraint(
            self.flowsheet().time, doc="X_S concentration output (kgCOD/m3)"
        )
        def XS_output(blk, t):
            return (
                blk.properties_out[t].conc_mass_comp["X_S"]
                == blk.Xch_AD1[t] + blk.Xpr_AD1[t] + blk.Xli_AD1[t]
            )

        @self.Constraint(
            self.flowsheet().time, doc="S_K concentration output (kgCOD/m3)"
        )
        # TODO: Need to add precipitation term for X_Kstruv
        def SK_output(blk, t):
            return blk.properties_out[t].conc_mass_comp["S_K"] == blk.SK_AD2[t] * mw_k

        @self.Constraint(
            self.flowsheet().time, doc="S_Mg concentration output (kgCOD/m3)"
        )
        def SMg_output(blk, t):
            return (
                blk.properties_out[t].conc_mass_comp["S_Mg"] == blk.SMg_AD2[t] * mw_mg
            )

        self.zero_flow_components = Set(
            initialize=[
                "S_O2",
                "S_N2",
                "S_NO3",
                "X_H",
                "X_PAO",
                "X_PP",
                "X_PHA",
                "X_AUT",
            ]
        )

        @self.Constraint(
            self.flowsheet().time,
            self.zero_flow_components,
            doc="Components with no flow equation",
        )
        def return_zero_flow_comp(blk, t, i):
            return (
                blk.properties_out[t].conc_mass_comp[i]
                == 1e-10 * pyunits.kg / pyunits.m**3
            )

        iscale.set_scaling_factor(self.properties_out[0].flow_vol, 1e5)

    def initialize_build(
        self,
        state_args_in=None,
        state_args_out=None,
        outlvl=idaeslog.NOTSET,
        solver=None,
        optarg=None,
    ):
        """
        This method calls the initialization method of the state blocks.

        Keyword Arguments:
            state_args_in : a dict of arguments to be passed to the inlet
                property package (to provide an initial state for
                initialization (see documentation of the specific
                property package) (default = None).
            state_args_out : a dict of arguments to be passed to the outlet
                property package (to provide an initial state for
                initialization (see documentation of the specific
                property package) (default = None).
            outlvl : sets output level of initialization routine
            optarg : solver options dictionary object (default=None, use
                     default solver options)
            solver : str indicating which solver to use during
                     initialization (default = None, use default solver)

        Returns:
            None
        """
        init_log = idaeslog.getInitLogger(self.name, outlvl, tag="unit")

        # Create solver
        opt = get_solver(solver, optarg)

        # ---------------------------------------------------------------------
        # Initialize state block
        flags = self.properties_in.initialize(
            outlvl=outlvl,
            optarg=optarg,
            solver=solver,
            state_args=state_args_in,
            hold_state=True,
        )

        self.properties_out.initialize(
            outlvl=outlvl,
            optarg=optarg,
            solver=solver,
            state_args=state_args_out,
        )

        if degrees_of_freedom(self) != 0:
            raise Exception(
                f"{self.name} degrees of freedom were not 0 at the beginning "
                f"of initialization. DoF = {degrees_of_freedom(self)}"
            )

        with idaeslog.solver_log(init_log, idaeslog.DEBUG) as slc:
            res = opt.solve(self, tee=slc.tee)

        self.properties_in.release_state(flags=flags, outlvl=outlvl)

        init_log.info(f"Initialization Complete: {idaeslog.condition(res)}")

        if not check_optimal_termination(res):
            raise InitializationError(
                f"{self.name} failed to initialize successfully. Please check "
                f"the output logs for more information."
            )
