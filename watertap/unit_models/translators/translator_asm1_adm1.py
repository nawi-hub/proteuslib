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
Translator block representing the ASM1/ADM1 interface.

Assumptions:
     * Steady-state only

Model formulated from:

Copp J. and Jeppsson, U., Rosen, C., 2006.
Towards an ASM1 - ADM1 State Variable Interface for Plant-Wide Wastewater Treatment Modeling.
Proceedings of the Water Environment Federation, 2003, pp 498-510.
"""

# Import Pyomo libraries
from pyomo.common.config import ConfigBlock, ConfigValue

# Import IDAES cores
from idaes.core.util.math import smooth_max
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

from pyomo.environ import (
    Constraint,
    Param,
    NonNegativeReals,
    Var,
    units as pyunits,
    check_optimal_termination,
    Set,
    Expr_if,
)

from idaes.core.util.exceptions import InitializationError

__author__ = "Alejandro Garciadiego, Xinhong Liu"


# Set up logger
_log = idaeslog.getLogger(__name__)


class ASM1ADM1Scaler(CustomScalerBase):
    """
    Default modular scaler for the ASM1-ADM1 translator block.
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


@declare_process_block_class("Translator_ASM1_ADM1")
class TranslatorDataASM1ADM1(TranslatorData):
    """
    Translator block representing the ASM1/ADM1 interface
    """

    default_scaler = ASM1ADM1Scaler

    CONFIG = TranslatorData.CONFIG()
    CONFIG.declare(
        "reaction_package",
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
        "reaction_package_args",
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
        super(TranslatorDataASM1ADM1, self).build()

        self.i_xe = Param(
            initialize=0.06,
            units=pyunits.dimensionless,
            mutable=True,
            doc="Fraction nitrogen in particulate products",
        )

        self.i_xb = Param(
            initialize=0.08,
            units=pyunits.dimensionless,
            mutable=True,
            doc="Fraction nitrogen in biomass",
        )

        self.f_xI = Param(
            initialize=0.05,
            units=pyunits.dimensionless,
            mutable=True,
            doc="Anaerobic degradable fraction of X_I and X_P",
        )

        self.eps = Param(
            initialize=1e-10,
            units=pyunits.kg / pyunits.m**3,
            mutable=True,
            doc="Smoothing factor",
        )

        mw_n = 14 * pyunits.kg / pyunits.kmol
        mw_c = 12 * pyunits.kg / pyunits.kmol

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

        @self.Expression(self.flowsheet().time, doc="COD demand")
        def CODd(blk, t):
            return (
                blk.properties_in[t].conc_mass_comp["S_O"]
                + blk.properties_in[t].conc_mass_comp["S_NO"] * 2.86
            )

        self.inter_S_S = Var(
            self.flowsheet().time,
            initialize=self.properties_in[0].conc_mass_comp["S_S"],
            units=pyunits.kg / pyunits.m**3,
            domain=NonNegativeReals,
            doc="Readily biodegradable substrate remaining",
        )

        @self.Constraint(
            self.flowsheet().time,
            doc="COD demand minus readily biodegradable substrate",
        )
        def CODd_step1(blk, t):
            return blk.inter_S_S[t] == smooth_max(
                0 * pyunits.kg / pyunits.m**3,
                blk.properties_in[t].conc_mass_comp["S_S"] - blk.CODd[t],
                blk.eps,
            )

        @self.Expression(self.flowsheet().time, doc="COD demand after S_S")
        # Includes smooth formulation for max(x, 0) to avoid having negative values
        # formulation included in all subsequent values for COD requirements and
        # COD remains
        def CODd2(blk, t):
            return smooth_max(
                0 * pyunits.kg / pyunits.m**3,
                blk.CODd[t] - blk.properties_in[t].conc_mass_comp["S_S"],
                blk.eps,
            )

        self.inter_X_S = Var(
            self.flowsheet().time,
            initialize=self.properties_in[0].conc_mass_comp["X_S"],
            units=pyunits.kg / pyunits.m**3,
            domain=NonNegativeReals,
            doc="Slowly biodegradable substrate remaining",
        )

        @self.Constraint(
            self.flowsheet().time,
            doc="COD demand minus slowly biodegradable substrate",
        )
        def CODd_step2(blk, t):
            return blk.inter_X_S[t] == smooth_max(
                0 * pyunits.kg / pyunits.m**3,
                blk.properties_in[t].conc_mass_comp["X_S"] - blk.CODd2[t],
                blk.eps,
            )

        @self.Expression(self.flowsheet().time, doc="COD demand after X_S")
        def CODd3(blk, t):
            return smooth_max(
                0 * pyunits.kg / pyunits.m**3,
                blk.CODd2[t] - blk.properties_in[t].conc_mass_comp["X_S"],
                blk.eps,
            )

        self.inter_X_BH = Var(
            self.flowsheet().time,
            initialize=self.properties_in[0].conc_mass_comp["X_BH"],
            units=pyunits.kg / pyunits.m**3,
            domain=NonNegativeReals,
            doc="Active heterotrophic biomass remaining",
        )

        @self.Constraint(
            self.flowsheet().time,
            doc="COD demand minus active heterotrophic biomass",
        )
        def CODd_step3(blk, t):
            return blk.inter_X_BH[t] == smooth_max(
                0 * pyunits.kg / pyunits.m**3,
                blk.properties_in[t].conc_mass_comp["X_BH"] - blk.CODd3[t],
                blk.eps,
            )

        @self.Expression(self.flowsheet().time, doc="COD demand after X_BH")
        def CODd4(blk, t):
            return smooth_max(
                0 * pyunits.kg / pyunits.m**3,
                blk.CODd3[t] - blk.properties_in[t].conc_mass_comp["X_BH"],
                blk.eps,
            )

        self.inter_X_BA = Var(
            self.flowsheet().time,
            initialize=self.properties_in[0].conc_mass_comp["X_BA"],
            units=pyunits.kg / pyunits.m**3,
            domain=NonNegativeReals,
            doc="Active autotrophic biomass remaining",
        )

        @self.Constraint(
            self.flowsheet().time,
            doc="COD demand minus active autotrophic biomass",
        )
        def CODd_step4(blk, t):
            return blk.inter_X_BA[t] == smooth_max(
                0 * pyunits.kg / pyunits.m**3,
                blk.properties_in[t].conc_mass_comp["X_BA"] - blk.CODd4[t],
                blk.eps,
            )

        @self.Expression(self.flowsheet().time, doc="COD demand after X_BA")
        def CODd5(blk, t):
            return smooth_max(
                0 * pyunits.kg / pyunits.m**3,
                blk.CODd4[t] - blk.properties_in[t].conc_mass_comp["X_BA"],
                blk.eps,
            )

        @self.Expression(self.flowsheet().time, doc="Soluble COD")
        def CODs(blk, t):
            return blk.properties_in[t].conc_mass_comp["S_I"] + blk.inter_S_S[t]

        @self.Expression(self.flowsheet().time, doc="Particulate COD")
        def CODp(blk, t):
            return (
                blk.properties_in[t].conc_mass_comp["X_I"]
                + blk.inter_X_S[t]
                + blk.inter_X_BH[t]
                + blk.inter_X_BA[t]
                + blk.properties_in[t].conc_mass_comp["X_P"]
            )

        @self.Expression(self.flowsheet().time, doc="Total COD")
        def CODt(blk, t):
            return blk.CODs[t] + blk.CODp[t]

        self.TKN_in = Var(
            self.flowsheet().time,
            initialize=self.properties_in[0].conc_mass_comp["S_NH"]
            + self.properties_in[0].conc_mass_comp["S_ND"]
            + self.properties_in[0].conc_mass_comp["X_ND"]
            + self.i_xb
            * (
                self.properties_in[0].conc_mass_comp["X_BH"]
                + self.properties_in[0].conc_mass_comp["X_BA"]
            )
            + self.i_xe
            * (
                self.properties_in[0].conc_mass_comp["X_I"]
                + self.properties_in[0].conc_mass_comp["X_P"]
            ),
            units=pyunits.kg / pyunits.m**3,
            domain=NonNegativeReals,
            doc="Total Kjeldahl nitrogen",
        )

        @self.Constraint(
            self.flowsheet().time,
            doc="Total Kjeldahl nitrogen",
        )
        def TKNin(blk, t):
            return blk.TKN_in[t] == blk.properties_in[t].conc_mass_comp[
                "S_NH"
            ] + blk.properties_in[t].conc_mass_comp["S_ND"] + blk.properties_in[
                t
            ].conc_mass_comp[
                "X_ND"
            ] + self.i_xb * (
                blk.inter_X_BH[t] + blk.inter_X_BA[t]
            ) + self.i_xe * (
                blk.properties_in[t].conc_mass_comp["X_I"]
                + blk.properties_in[t].conc_mass_comp["X_P"]
            )

        @self.Expression(self.flowsheet().time, doc="Required soluble COD")
        def ReqCODs(blk, t):
            return (
                blk.properties_in[t].conc_mass_comp["S_ND"]
                / blk.config.reaction_package.N_aa
                / mw_n
            )

        @self.Expression(self.flowsheet().time, doc="Amino acids mapping")
        def saa_mapping(blk, t):
            return Expr_if(
                blk.inter_S_S[t] > blk.ReqCODs[t],
                blk.ReqCODs[t],
                blk.inter_S_S[t],
            )

        @self.Expression(self.flowsheet().time, doc="Monosaccharides mapping step A")
        def ssu_mapping_A(blk, t):
            return Expr_if(
                blk.inter_S_S[t] > blk.ReqCODs[t],
                blk.inter_S_S[t] - blk.ReqCODs[t],
                1e-10 * pyunits.kg / pyunits.m**3,
            )

        @self.Constraint(
            self.flowsheet().time,
            doc="Amino acids balance",
        )
        def Saa_balance(blk, t):
            return blk.properties_out[t].conc_mass_comp["S_aa"] == blk.saa_mapping[t]

        @self.Expression(self.flowsheet().time, doc="COD remaining from step A")
        def COD_remain_a(blk, t):
            return smooth_max(
                0 * pyunits.kg / pyunits.m**3, blk.CODt[t] - blk.inter_S_S[t], blk.eps
            )

        @self.Expression(self.flowsheet().time, doc="Organic nitrogen pool from step A")
        def ORGN_remain_a(blk, t):
            return (
                blk.TKN_in[t]
                - (
                    blk.properties_out[t].conc_mass_comp["S_aa"]
                    * blk.config.reaction_package.N_aa
                    * mw_n
                )
                - blk.properties_in[t].conc_mass_comp["S_NH"]
            )

        @self.Expression(
            self.flowsheet().time, doc="Required soluble inert organic nitrogen"
        )
        def ReqOrgNS(blk, t):
            return (
                blk.properties_in[t].conc_mass_comp["S_I"]
                * blk.config.reaction_package.N_I
                * mw_n
            )

        @self.Expression(self.flowsheet().time, doc="Soluble inert mapping step B")
        def si_mapping_B(blk, t):
            return Expr_if(
                blk.ORGN_remain_a[t] > blk.ReqOrgNS[t],
                blk.properties_in[t].conc_mass_comp["S_I"],
                blk.ORGN_remain_a[t] / blk.config.reaction_package.N_I / mw_n,
            )

        @self.Expression(self.flowsheet().time, doc="Monosaccharides mapping step B")
        def ssu_mapping_B(blk, t):
            return Expr_if(
                blk.ORGN_remain_a[t] > blk.ReqOrgNS[t],
                blk.ssu_mapping_A[t],
                blk.ssu_mapping_A[t]
                + blk.properties_in[t].conc_mass_comp["S_I"]
                - blk.si_mapping_B[t],
            )

        @self.Constraint(
            self.flowsheet().time,
            doc="Inert organic balance",
        )
        def Si_balance(blk, t):
            return blk.properties_out[t].conc_mass_comp["S_I"] == blk.si_mapping_B[t]

        @self.Constraint(
            self.flowsheet().time,
            doc="Monosacharides balance",
        )
        def Ssu_balance(blk, t):
            return blk.properties_out[t].conc_mass_comp["S_su"] == blk.ssu_mapping_B[t]

        @self.Expression(self.flowsheet().time, doc="COD remaining from step B")
        def COD_remain_b(blk, t):
            return smooth_max(
                0 * pyunits.kg / pyunits.m**3,
                blk.COD_remain_a[t] - blk.properties_in[t].conc_mass_comp["S_I"],
                blk.eps,
            )

        @self.Expression(self.flowsheet().time, doc="Organic nitrogen pool from step B")
        def ORGN_remain_b(blk, t):
            return blk.ORGN_remain_a[t] - (
                blk.properties_out[t].conc_mass_comp["S_I"]
                * blk.config.reaction_package.N_I
                * mw_n
            )

        @self.Expression(
            self.flowsheet().time, doc="Required particulate inert material"
        )
        def ReqOrgNx(blk, t):
            return (
                blk.f_xI
                * (
                    blk.properties_in[t].conc_mass_comp["X_P"]
                    + blk.properties_in[t].conc_mass_comp["X_I"]
                )
                * blk.config.reaction_package.N_I
                * mw_n
            )

        @self.Expression(
            self.flowsheet().time,
            doc="Inert particulate organic material mapping step C",
        )
        def xi_mapping(blk, t):
            return Expr_if(
                blk.ORGN_remain_b[t] > blk.ReqOrgNx[t],
                (
                    blk.f_xI
                    * (
                        blk.properties_in[t].conc_mass_comp["X_P"]
                        + blk.properties_in[t].conc_mass_comp["X_I"]
                    )
                ),
                blk.ORGN_remain_b[t] / blk.config.reaction_package.N_I / mw_n,
            )

        @self.Constraint(
            self.flowsheet().time,
            doc="Organic nitrogen balance",
        )
        def Xi_balance(blk, t):
            return blk.properties_out[t].conc_mass_comp["X_I"] == blk.xi_mapping[t]

        @self.Expression(self.flowsheet().time, doc="COD remaining from step C")
        def COD_remain_c(blk, t):
            return smooth_max(
                0 * pyunits.kg / pyunits.m**3,
                blk.COD_remain_b[t] - blk.properties_out[t].conc_mass_comp["X_I"],
                blk.eps,
            )

        @self.Expression(self.flowsheet().time, doc="Organic nitrogen pool from step C")
        def ORGN_remain_c(blk, t):
            return blk.ORGN_remain_b[t] - (
                blk.properties_out[t].conc_mass_comp["X_I"]
                * blk.config.reaction_package.N_I
                * mw_n
            )

        @self.Expression(self.flowsheet().time, doc="Required Composites")
        def ReqCODXc(blk, t):
            return blk.ORGN_remain_c[t] / blk.config.reaction_package.N_xc / mw_n

        @self.Expression(self.flowsheet().time, doc="Composites mapping")
        def xc_mapping(blk, t):
            return Expr_if(
                blk.COD_remain_c[t] > blk.ReqCODXc[t],
                blk.ReqCODXc[t],
                blk.COD_remain_c[t],
            )

        @self.Constraint(
            self.flowsheet().time,
            doc="Composites balance",
        )
        def Xc_balance(blk, t):
            return blk.properties_out[t].conc_mass_comp["X_c"] == blk.xc_mapping[t]

        @self.Expression(self.flowsheet().time, doc="Carbohydrates mapping")
        def xch_mapping(blk, t):
            return Expr_if(
                blk.COD_remain_c[t] > blk.ReqCODXc[t],
                blk.config.reaction_package.f_ch_xc
                / (
                    blk.config.reaction_package.f_ch_xc
                    + blk.config.reaction_package.f_li_xc
                )
                * (blk.COD_remain_c[t] - blk.properties_out[t].conc_mass_comp["X_c"]),
                1e-10 * pyunits.kg / pyunits.m**3,
            )

        @self.Constraint(
            self.flowsheet().time,
            doc="Carbohydrates balance",
        )
        def xch_balance(blk, t):
            return blk.properties_out[t].conc_mass_comp["X_ch"] == blk.xch_mapping[t]

        @self.Expression(self.flowsheet().time, doc="Lipids mapping")
        def xli_mapping(blk, t):
            return Expr_if(
                blk.COD_remain_c[t] > blk.ReqCODXc[t],
                blk.config.reaction_package.f_li_xc
                / (
                    blk.config.reaction_package.f_ch_xc
                    + blk.config.reaction_package.f_li_xc
                )
                * (blk.COD_remain_c[t] - blk.properties_out[t].conc_mass_comp["X_c"]),
                1e-10 * pyunits.kg / pyunits.m**3,
            )

        @self.Constraint(
            self.flowsheet().time,
            doc="Lipids balance",
        )
        def xli_balance(blk, t):
            return blk.properties_out[t].conc_mass_comp["X_li"] == blk.xli_mapping[t]

        @self.Expression(self.flowsheet().time, doc="Inorganic nitrogen mapping")
        def sin_mapping(blk, t):
            return Expr_if(
                blk.COD_remain_c[t] > blk.ReqCODXc[t],
                blk.properties_in[t].conc_mass_comp["S_NH"],
                blk.properties_in[t].conc_mass_comp["S_NH"]
                + (
                    blk.ORGN_remain_c[t]
                    - blk.properties_out[t].conc_mass_comp["X_c"]
                    * blk.config.reaction_package.N_xc
                    * mw_n
                ),
            )

        @self.Constraint(
            self.flowsheet().time,
            doc="Inorganic nitrogen balance",
        )
        def SIN_balance(blk, t):
            return blk.properties_out[t].conc_mass_comp["S_IN"] == blk.sin_mapping[t]

        @self.Constraint(
            self.flowsheet().time,
            doc="Anions balance",
        )
        def return_an(blk, t):
            return (
                blk.properties_out[t].anions
                == blk.properties_out[t].conc_mass_comp["S_IN"] / mw_n
            )

        @self.Constraint(
            self.flowsheet().time,
            doc="Cations balance",
        )
        def return_cat(blk, t):
            return (
                blk.properties_out[t].cations
                == blk.properties_out[t].conc_mass_comp["S_IC"] / mw_c
            )

        self.zero_flow_components = Set(
            initialize=[
                "S_fa",
                "S_va",
                "S_bu",
                "S_pro",
                "S_ac",
                "S_h2",
                "S_ch4",
                "X_pr",
                "X_su",
                "X_aa",
                "X_fa",
                "X_c4",
                "X_pro",
                "X_ac",
                "X_h2",
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

        @self.Constraint(
            self.flowsheet().time,
            doc="Equality alkalinity equation",
        )
        def return_Salk(blk, t):
            return (
                blk.properties_in[t].alkalinity
                == blk.properties_out[t].conc_mass_comp["S_IC"] / mw_c
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
