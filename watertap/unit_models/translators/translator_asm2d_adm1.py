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
Translator block representing the ASM2d/ADM1 interface.

Assumptions:
     * Steady-state only

Model formulated from:

Flores-Alsina, X., Solon, K., Mbamba, C.K., Tait, S., Gernaey, K.V., Jeppsson, U. and Batstone, D.J., 2016.
Modelling phosphorus (P), sulfur (S) and iron (Fe) interactions for dynamic simulations of anaerobic digestion processes.
Water Research, 95, pp.370-382. https://github.com/wwtmodels/Plant-Wide-Models
"""

# Import Pyomo libraries
from pyomo.common.config import Bool, ConfigBlock, ConfigValue

# Import IDAES cores
from idaes.core import declare_process_block_class
from idaes.core.util.math import smooth_max, smooth_min
from idaes.models.unit_models.translator import TranslatorData
from idaes.core.util.config import (
    is_reaction_parameter_block,
)
from idaes.core.util.model_statistics import degrees_of_freedom
from watertap.core.solvers import get_solver
import idaes.logger as idaeslog
from idaes.core.scaling import CustomScalerBase, ConstraintScalingScheme

from pyomo.environ import (
    Constraint,
    Param,
    units as pyunits,
    check_optimal_termination,
    Set,
    Expr_if,
    value,
)

from idaes.core.util.exceptions import InitializationError

__author__ = "Marcus Holly"


# Set up logger
_log = idaeslog.getLogger(__name__)


class ASM2dADM1Scaler(CustomScalerBase):
    """
    Default modular scaler for ASM2d-ADM1 translator block.
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


@declare_process_block_class("Translator_ASM2d_ADM1")
class TranslatorDataASM2dADM1(TranslatorData):
    """
    Translator block representing the ASM2d/ADM1 interface
    """

    default_scaler = ASM2dADM1Scaler

    CONFIG = TranslatorData.CONFIG()

    # TODO: Change the default to False
    CONFIG.declare(
        "bio_P",
        ConfigValue(
            default=False,
            domain=Bool,
            description="Switching function for phosphorus biomass",
            doc="""Switching function for handling the transformation of phosphorus biomass,
**default** - True.
**Valid values:** {
**False** - the BioP variables are kinetically described within the ADM,
**True** - the BioP variables are supposed to be transformed in the interface""",
        ),
    )

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
        super(TranslatorDataASM2dADM1, self).build()

        eps = 0
        mw_p = 31 * pyunits.kg / pyunits.kmol
        mw_n = 14 * pyunits.kg / pyunits.kmol
        mw_c = 12 * pyunits.kg / pyunits.kmol

        # TODO: Move these parameter values to the modified ADM1 rxn package
        self.f_XPHA_Sva = Param(
            initialize=0.1,
            units=pyunits.dimensionless,
            mutable=True,
            doc="Valerate from polyhydroxyalkanoates",
        )

        self.f_XPHA_Sbu = Param(
            initialize=0.1,
            units=pyunits.dimensionless,
            mutable=True,
            doc="Butyrate from polyhydroxyalkanoates",
        )

        self.f_XPHA_Spro = Param(
            initialize=0.4,
            units=pyunits.dimensionless,
            mutable=True,
            doc="Propionate from polyhydroxyalkanoates",
        )

        self.f_XPHA_Sac = Param(
            initialize=0.4,
            units=pyunits.dimensionless,
            mutable=True,
            doc="Acetate from polyhydroxyalkanoates",
        )

        self.C_PHA = Param(
            initialize=0.3 / 12,
            units=pyunits.dimensionless,
            mutable=True,
            doc="Carbon content of polyhydroxyalkanoates",
        )
        self.i_PSI = Param(
            initialize=0.00649,
            units=pyunits.dimensionless,
            mutable=True,
            doc="P content of inert soluble COD S_I, [kg P/kg COD]",
        )
        self.eps_smooth = Param(
            initialize=1e-4,
            units=pyunits.kg / pyunits.m**3,
            mutable=True,
            doc="Smoothing factor",
        )

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
            doc="Concentration (kgCOD/m3) of COD demanding compounds in S_O2",
        )
        def COD_SO2(blk, t):
            return blk.properties_in[t].conc_mass_comp["S_O2"] / (
                (1 - blk.config.inlet_reaction_package.Y_H)
                / blk.config.inlet_reaction_package.Y_H
            )

        @self.Expression(
            self.flowsheet().time, doc="S_O2 concentration (kgCOD/m3) at step 1"
        )
        def SO2_AS1(blk, t):
            return (
                blk.properties_in[t].conc_mass_comp["S_O2"]
                - (1 - blk.config.inlet_reaction_package.Y_H)
                / blk.config.inlet_reaction_package.Y_H
                * self.COD_SO2[t]
            )

        @self.Expression(
            self.flowsheet().time, doc="S_A concentration (kgCOD/m3) at step 1"
        )
        def SA_AS1(blk, t):
            return (
                blk.properties_in[t].conc_mass_comp["S_A"]
                - (1 / blk.config.inlet_reaction_package.Y_H) * self.COD_SO2[t]
            )

        @self.Expression(
            self.flowsheet().time, doc="S_NH4 concentration (kgN/m3) at step 1"
        )
        def SNH4_AS1(blk, t):
            return (
                blk.properties_in[t].conc_mass_comp["S_NH4"]
                - blk.config.inlet_reaction_package.i_NBM * self.COD_SO2[t]
            )

        @self.Expression(
            self.flowsheet().time, doc="S_PO4 concentration (kgP/m3) at step 1"
        )
        def SPO4_AS1(blk, t):
            return (
                blk.properties_in[t].conc_mass_comp["S_PO4"]
                - blk.config.inlet_reaction_package.i_PBM * self.COD_SO2[t]
            )

        @self.Expression(
            self.flowsheet().time, doc="S_IC concentration (kgC/m3) at step 1"
        )
        def SIC_AS1(blk, t):
            return (
                blk.properties_in[t].conc_mass_comp["S_IC"]
                - (-1 / blk.config.inlet_reaction_package.Y_H)
                * self.COD_SO2[t]
                * blk.config.inlet_reaction_package.i_CSA
                + self.COD_SO2[t] * blk.config.inlet_reaction_package.i_CXB
            )

        @self.Expression(
            self.flowsheet().time, doc="X_H concentration (kgCOD/m3) at step 1"
        )
        def XH_AS1(blk, t):
            return blk.properties_in[t].conc_mass_comp["X_H"] + self.COD_SO2[t]

        # -------------------------------------------Step 2----------------------------------------------------------------#
        @self.Expression(
            self.flowsheet().time,
            doc="Concentration of COD demanding compounds in S_NO3",
        )
        def COD_SNO3(blk, t):
            return blk.properties_in[t].conc_mass_comp["S_NO3"] / (
                (1 - blk.config.inlet_reaction_package.Y_H)
                / (
                    blk.config.inlet_reaction_package.i_NOx_N2
                    * blk.config.inlet_reaction_package.Y_H
                )
            )

        @self.Expression(
            self.flowsheet().time, doc="S_A concentration (kgCOD/m3) at step 2"
        )
        def SA_AS2(blk, t):
            return (
                blk.SA_AS1[t]
                - (1 / blk.config.inlet_reaction_package.Y_H) * self.COD_SNO3[t]
            )

        @self.Expression(
            self.flowsheet().time, doc="S_NH4 concentration (kgN/m3) at step 2"
        )
        def SNH4_AS2(blk, t):
            return (
                blk.SNH4_AS1[t]
                - self.COD_SNO3[t] * blk.config.inlet_reaction_package.i_NBM
            )

        @self.Expression(
            self.flowsheet().time, doc="S_N2 concentration (kgN/m3) at step 2"
        )
        def SN2_AS2(blk, t):
            return (
                blk.properties_in[t].conc_mass_comp["S_N2"]
                + (
                    (1 - blk.config.inlet_reaction_package.Y_H)
                    / (
                        blk.config.inlet_reaction_package.i_NOx_N2
                        * blk.config.inlet_reaction_package.Y_H
                    )
                )
                * self.COD_SNO3[t]
            )

        @self.Expression(
            self.flowsheet().time, doc="S_NO3 concentration (kgN/m3) at step 2"
        )
        def SNO3_AS2(blk, t):
            return (
                blk.properties_in[t].conc_mass_comp["S_NO3"]
                - (
                    (1 - blk.config.inlet_reaction_package.Y_H)
                    / (
                        blk.config.inlet_reaction_package.i_NOx_N2
                        * blk.config.inlet_reaction_package.Y_H
                    )
                )
                * self.COD_SNO3[t]
            )

        @self.Expression(
            self.flowsheet().time, doc="S_PO4 concentration (kgP/m3) at step 2"
        )
        def SPO4_AS2(blk, t):
            return (
                blk.SPO4_AS1[t]
                - self.COD_SNO3[t] * blk.config.inlet_reaction_package.i_PBM
            )

        @self.Expression(
            self.flowsheet().time, doc="S_IC concentration (kgC/m3) at step 2"
        )
        def SIC_AS2(blk, t):
            return (
                blk.SIC_AS1[t]
                - (-1 / blk.config.inlet_reaction_package.Y_H)
                * self.COD_SNO3[t]
                * blk.config.inlet_reaction_package.i_CSA
                + self.COD_SNO3[t] * blk.config.inlet_reaction_package.i_CXB
            )

        @self.Expression(
            self.flowsheet().time, doc="X_H concentration (kgCOD/m3) at step 2"
        )
        def XH_AS2(blk, t):
            return blk.XH_AS1[t] + self.COD_SNO3[t]

        # -------------------------------------------Step 3----------------------------------------------------------------#
        @self.Expression(
            self.flowsheet().time, doc="Nitrogen demand for soluble inerts"
        )
        def S_ND(blk, t):
            return (
                blk.properties_in[t].conc_mass_comp["S_F"]
                * blk.config.inlet_reaction_package.i_NSF
            )

        @self.Expression(
            self.flowsheet().time, doc="Phosphorus demand for soluble inerts"
        )
        def S_PD(blk, t):
            return (
                blk.properties_in[t].conc_mass_comp["S_F"]
                * blk.config.inlet_reaction_package.i_PSF
            )

        @self.Expression(
            self.flowsheet().time, doc="Organic nitrogen from soluble inerts"
        )
        def SN_org(blk, t):
            return blk.S_ND[t] / (blk.config.outlet_reaction_package.Ni["S_aa"] * mw_n)

        @self.Expression(self.flowsheet().time, doc="Monosaccharides mapping")
        def Ssu_mapping(blk, t):
            return smooth_max(
                0 * pyunits.kg / pyunits.m**3,
                blk.properties_in[t].conc_mass_comp["S_F"] - blk.SN_org[t],
                blk.eps_smooth,
            )

        @self.Expression(self.flowsheet().time, doc="Amino acids mapping")
        def Saa_mapping(blk, t):
            return smooth_min(
                blk.SN_org[t],
                blk.properties_in[t].conc_mass_comp["S_F"],
                blk.eps_smooth,
            )

        @self.Constraint(
            self.flowsheet().time,
            doc="Monosaccharides output (kgCOD/m3)",
        )
        def Ssu_output(blk, t):
            return blk.properties_out[t].conc_mass_comp["S_su"] == blk.Ssu_mapping[t]

        @self.Constraint(
            self.flowsheet().time,
            doc="Amino acids output (kgCOD/m3)",
        )
        def Saa_output(blk, t):
            return blk.properties_out[t].conc_mass_comp["S_aa"] == blk.Saa_mapping[t]

        @self.Expression(
            self.flowsheet().time, doc="S_F concentration (kgCOD/m3) at step 3"
        )
        def SF_AS3(blk, t):
            return (
                blk.properties_in[t].conc_mass_comp["S_F"]
                - blk.Ssu_mapping[t]
                - blk.Saa_mapping[t]
            )

        @self.Expression(
            self.flowsheet().time, doc="S_NH4 concentration (kgN/m3) at step 3"
        )
        def SNH4_AS3(blk, t):
            return (
                blk.SNH4_AS2[t]
                + blk.properties_in[t].conc_mass_comp["S_F"]
                * blk.config.inlet_reaction_package.i_NSF
                - blk.Saa_mapping[t]
                * blk.config.outlet_reaction_package.Ni["S_aa"]
                * mw_n
            )

        @self.Expression(
            self.flowsheet().time, doc="S_PO4 concentration (kgP/m3) at step 3"
        )
        def SPO4_AS3(blk, t):
            return (
                blk.SPO4_AS2[t]
                + blk.properties_in[t].conc_mass_comp["S_F"]
                * blk.config.inlet_reaction_package.i_PSF
            )

        @self.Expression(
            self.flowsheet().time, doc="S_IC concentration (kgC/m3) at step 3"
        )
        def SIC_AS3(blk, t):
            return (
                blk.SIC_AS2[t]
                + blk.properties_in[t].conc_mass_comp["S_F"]
                * blk.config.inlet_reaction_package.i_CSF
                - blk.Ssu_mapping[t]
                * blk.config.outlet_reaction_package.Ci["S_su"]
                * mw_c
                - blk.Saa_mapping[t]
                * blk.config.outlet_reaction_package.Ci["S_aa"]
                * mw_c
            )

        # ------------------------------------------BioP = True------------------------------------------------------------#

        # -------------------------------------------Step 4----------------------------------------------------------------#
        if self.config.bio_P:

            @self.Expression(
                self.flowsheet().time, doc="Biomass concentration (kgCOD/m3) at step 4"
            )
            # TODO: include sulfur (X_SO) in this expression after implementing sulfur-extension
            def biomass(blk, t):
                return (
                    blk.XH_AS2[t]
                    + blk.properties_in[t].conc_mass_comp["X_PAO"]
                    + blk.properties_in[t].conc_mass_comp["X_AUT"]
                )

            @self.Expression(
                self.flowsheet().time, doc="S_I concentration (kgCOD/m3) at step 4"
            )
            def SI_AS4(blk, t):
                return (
                    blk.properties_in[t].conc_mass_comp["S_I"]
                    + blk.biomass[t] * blk.config.outlet_reaction_package.f_sI_xc
                )

            @self.Constraint(
                self.flowsheet().time, doc="S_I concentration (kgCOD/m3) output"
            )
            def SI_output(blk, t):
                return blk.properties_out[t].conc_mass_comp["S_I"] == blk.SI_AS4[t]

            @self.Expression(
                self.flowsheet().time, doc="S_NH4 concentration (kgN/m3) at step 4"
            )
            def SNH4_AS4(blk, t):
                return (
                    blk.SNH4_AS3[t]
                    + blk.biomass[t] * blk.config.inlet_reaction_package.i_NBM
                    - blk.biomass[t]
                    * blk.config.outlet_reaction_package.f_sI_xc
                    * blk.config.inlet_reaction_package.i_NSI
                    - blk.biomass[t]
                    * blk.config.outlet_reaction_package.f_xI_xc
                    * blk.config.inlet_reaction_package.i_NSI
                    - blk.biomass[t]
                    * blk.config.outlet_reaction_package.f_pr_xc
                    * blk.config.outlet_reaction_package.Ni["X_pr"]
                    * mw_n
                )

            @self.Expression(
                self.flowsheet().time, doc="S_PO4 concentration (kgP/m3) at step 4"
            )
            def SPO4_AS4(blk, t):
                return (
                    blk.SPO4_AS3[t]
                    + blk.biomass[t] * blk.config.inlet_reaction_package.i_PBM
                    - blk.biomass[t]
                    * blk.config.outlet_reaction_package.f_sI_xc
                    * self.i_PSI
                    - blk.biomass[t]
                    * blk.config.outlet_reaction_package.f_xI_xc
                    * blk.config.inlet_reaction_package.i_PXI
                    - blk.biomass[t]
                    * blk.config.outlet_reaction_package.f_ch_xc
                    * blk.config.outlet_reaction_package.P_ch
                    * mw_p
                    - blk.biomass[t]
                    * blk.config.outlet_reaction_package.f_li_xc
                    * blk.config.outlet_reaction_package.Pi["X_li"]
                    * mw_p
                )

            @self.Expression(
                self.flowsheet().time, doc="S_PO4 concentration (kgC/m3) at step 4"
            )
            def SIC_AS4(blk, t):
                return (
                    blk.SIC_AS3[t]
                    + blk.biomass[t] * blk.config.inlet_reaction_package.i_CXB
                    - blk.biomass[t]
                    * blk.config.outlet_reaction_package.f_sI_xc
                    * blk.config.inlet_reaction_package.i_CSI
                    - blk.biomass[t]
                    * blk.config.outlet_reaction_package.f_xI_xc
                    * blk.config.inlet_reaction_package.i_CXI
                    - blk.biomass[t]
                    * blk.config.outlet_reaction_package.f_pr_xc
                    * blk.config.outlet_reaction_package.Ci["X_pr"]
                    * mw_c
                    - blk.biomass[t]
                    * blk.config.outlet_reaction_package.f_ch_xc
                    * blk.config.outlet_reaction_package.Ci["X_ch"]
                    * mw_c
                    - blk.biomass[t]
                    * blk.config.outlet_reaction_package.f_li_xc
                    * blk.config.outlet_reaction_package.Ci["X_li"]
                    * mw_c
                )

            @self.Expression(
                self.flowsheet().time, doc="X_I concentration (kgCOD/m3) at step 4"
            )
            def XI_AS4(blk, t):
                return (
                    blk.properties_in[t].conc_mass_comp["X_I"]
                    + blk.biomass[t] * blk.config.outlet_reaction_package.f_xI_xc
                )

            @self.Constraint(
                self.flowsheet().time, doc="X_I concentration (kgCOD/m3) output"
            )
            def XI_output(blk, t):
                return blk.properties_out[t].conc_mass_comp["X_I"] == blk.XI_AS4[t]

            # TODO: Consider removing cases where vars are set to 0 and they are not being used
            # TODO: Many were left in just to match Flores-Alsina as closely as possible

            self.XH_AS4 = Param(
                initialize=eps,
                units=pyunits.kg / pyunits.m**3,
                mutable=True,
                doc="Mass concentration of X_H at step 4",
            )

            self.XPAO_AS4 = Param(
                initialize=eps,
                units=pyunits.kg / pyunits.m**3,
                mutable=True,
                doc="Mass concentration of X_PAO at step 4",
            )

            @self.Constraint(
                self.flowsheet().time, doc="X_PAO concentration (kgCOD/m3) output"
            )
            def XPAO_output(blk, t):
                return blk.properties_out[t].conc_mass_comp["X_PAO"] == self.XPAO_AS4

            self.XAUT_AS4 = Param(
                initialize=eps,
                units=pyunits.kg / pyunits.m**3,
                mutable=True,
                doc="Mass concentration of X_AUT at step 4",
            )

            # -----------------------------------------Step 5--------------------------------------------------------------#
            @self.Expression(
                self.flowsheet().time, doc="Nitrogen demand for particulate inerts"
            )
            def X_ND(blk, t):
                return (
                    blk.properties_in[t].conc_mass_comp["X_S"]
                    * blk.config.inlet_reaction_package.i_NXS
                )

            @self.Expression(
                self.flowsheet().time, doc="Phosphorus demand for particulate inerts"
            )
            def X_PD(blk, t):
                return (
                    blk.properties_in[t].conc_mass_comp["X_S"]
                    * blk.config.inlet_reaction_package.i_PXS
                )

            @self.Expression(
                self.flowsheet().time, doc="Organic nitrogen from particulate inerts"
            )
            def XN_org(blk, t):
                return blk.X_ND[t] / (
                    blk.config.outlet_reaction_package.Ni["X_pr"] * mw_n
                )

            @self.Expression(self.flowsheet().time, doc="Carbohydrates mapping")
            def Xch_mapping(blk, t):
                return smooth_max(
                    0 * pyunits.kg / pyunits.m**3,
                    (blk.properties_in[t].conc_mass_comp["X_S"] - blk.XN_org[t]) * 0.4,
                    blk.eps_smooth,
                )

            # TODO: Can this be replaced with smooth_max or smooth_min?
            @self.Expression(self.flowsheet().time, doc="Protein mapping")
            def Xpr_mapping(blk, t):
                return Expr_if(
                    blk.XN_org[t] >= blk.properties_in[t].conc_mass_comp["X_S"],
                    blk.SF_AS3[t],
                    blk.XN_org[t],
                )

            @self.Expression(self.flowsheet().time, doc="Lipids mapping")
            def Xli_mapping(blk, t):
                return smooth_max(
                    0 * pyunits.kg / pyunits.m**3,
                    (blk.properties_in[t].conc_mass_comp["X_S"] - blk.XN_org[t]) * 0.6,
                    blk.eps_smooth,
                )

            @self.Constraint(
                self.flowsheet().time,
                doc="Carbohydrates output",
            )
            def Xch_output(blk, t):
                return (
                    blk.properties_out[t].conc_mass_comp["X_ch"]
                    == blk.Xch_mapping[t]
                    + blk.biomass[t] * blk.config.outlet_reaction_package.f_ch_xc
                )

            @self.Constraint(
                self.flowsheet().time,
                doc="Protein output",
            )
            def Xpr_output(blk, t):
                return (
                    blk.properties_out[t].conc_mass_comp["X_pr"]
                    == blk.Xpr_mapping[t]
                    + blk.biomass[t] * blk.config.outlet_reaction_package.f_pr_xc
                )

            @self.Constraint(
                self.flowsheet().time,
                doc="Lipids output",
            )
            def Xli_output(blk, t):
                return (
                    blk.properties_out[t].conc_mass_comp["X_li"]
                    == blk.Xli_mapping[t]
                    + blk.biomass[t] * blk.config.outlet_reaction_package.f_li_xc
                )

            @self.Expression(
                self.flowsheet().time, doc="S_NH4 concentration (kgN/m3) at step 5"
            )
            def SNH4_AS5(blk, t):
                return (
                    blk.SNH4_AS4[t]
                    + blk.properties_in[t].conc_mass_comp["X_S"]
                    * blk.config.inlet_reaction_package.i_NXS
                    - blk.Xpr_mapping[t]
                    * blk.config.outlet_reaction_package.Ni["X_pr"]
                    * mw_n
                )

            @self.Constraint(
                self.flowsheet().time, doc="S_NH4 concentration (kgN/m3) output"
            )
            def SIN_output(blk, t):
                return blk.properties_out[t].conc_mass_comp["S_IN"] == blk.SNH4_AS5[t]

            @self.Expression(
                self.flowsheet().time, doc="S_PO4 concentration (kgP/m3) at step 5"
            )
            def SPO4_AS5(blk, t):
                return (
                    blk.SPO4_AS4[t]
                    + blk.properties_in[t].conc_mass_comp["X_S"]
                    * blk.config.inlet_reaction_package.i_PXS
                    - blk.Xch_mapping[t]
                    * blk.config.outlet_reaction_package.P_ch
                    * mw_p
                    - blk.Xli_mapping[t]
                    * blk.config.outlet_reaction_package.Pi["X_li"]
                    * mw_p
                )

            @self.Expression(
                self.flowsheet().time, doc="S_IC concentration (kgC/m3) at step 5"
            )
            def SIC_AS5(blk, t):
                return (
                    blk.SIC_AS4[t]
                    + blk.properties_in[t].conc_mass_comp["X_S"]
                    * blk.config.inlet_reaction_package.i_CXS
                    - blk.Xch_mapping[t]
                    * blk.config.outlet_reaction_package.Ci["X_ch"]
                    * mw_c
                    - blk.Xpr_mapping[t]
                    * blk.config.outlet_reaction_package.Ci["X_pr"]
                    * mw_c
                    - blk.Xli_mapping[t]
                    * blk.config.outlet_reaction_package.Ci["X_li"]
                    * mw_c
                )

            self.XS_AS5 = Param(
                initialize=eps,
                units=pyunits.kg / pyunits.m**3,
                mutable=True,
                doc="Mass concentration of X_AS at step 5",
            )

            # -----------------------------------------Step 6--------------------------------------------------------------#
            self.XPP_AS6 = Param(
                initialize=eps,
                units=pyunits.kg / pyunits.m**3,
                mutable=True,
                doc="Mass concentration of X_PP at step 6",
            )

            @self.Constraint(
                self.flowsheet().time, doc="X_PP concentration (kgP/m3) output"
            )
            def XPP_output(blk, t):
                return blk.properties_out[t].conc_mass_comp["X_PP"] == self.XPP_AS6

            self.XPHA_AS6 = Param(
                initialize=eps,
                units=pyunits.kg / pyunits.m**3,
                mutable=True,
                doc="Mass concentration of X_PHA at step 6",
            )

            @self.Constraint(
                self.flowsheet().time, doc="X_PHA concentration (kgCOD/m3) output"
            )
            def XPHA_output(blk, t):
                return blk.properties_out[t].conc_mass_comp["X_PHA"] == self.XPHA_AS6

            self.Sva_AS6 = Param(
                initialize=value(self.XPHA_AS6) * value(self.f_XPHA_Sva),
                units=pyunits.kg / pyunits.m**3,
                mutable=True,
                doc="Mass concentration of S_va at step 6",
            )

            @self.Constraint(
                self.flowsheet().time,
                doc="Total valerate concentration (kgCOD/m3) output",
            )
            def Sva_output(blk, t):
                return blk.properties_out[t].conc_mass_comp["S_va"] == self.Sva_AS6

            self.Sbu_AS6 = Param(
                initialize=value(self.XPHA_AS6) * value(self.f_XPHA_Sbu),
                units=pyunits.kg / pyunits.m**3,
                mutable=True,
                doc="Mass concentration of S_bu at step 6",
            )

            @self.Constraint(
                self.flowsheet().time,
                doc="Total butyrate concentration (kgCOD/m3) output",
            )
            def Sbu_output(blk, t):
                return blk.properties_out[t].conc_mass_comp["S_bu"] == self.Sbu_AS6

            self.Spro_AS6 = Param(
                initialize=value(self.XPHA_AS6) * value(self.f_XPHA_Spro),
                units=pyunits.kg / pyunits.m**3,
                mutable=True,
                doc="Mass concentration of S_pro at step 6",
            )

            @self.Constraint(
                self.flowsheet().time,
                doc="Total propionate concentration (kgCOD/m3) output",
            )
            def Spro_output(blk, t):
                return blk.properties_out[t].conc_mass_comp["S_pro"] == self.Spro_AS6

            self.Sac_AS6 = Param(
                initialize=value(self.XPHA_AS6) * value(self.f_XPHA_Sac),
                units=pyunits.kg / pyunits.m**3,
                mutable=True,
                doc="Mass concentration of S_ac at step 6",
            )

            @self.Constraint(
                self.flowsheet().time,
                doc="Total acetate concentration (kgCOD/m3) output",
            )
            def Sac_output(blk, t):
                return (
                    blk.properties_out[t].conc_mass_comp["S_ac"]
                    == self.Sac_AS6 + blk.SA_AS2[t]
                )

            @self.Expression(
                self.flowsheet().time, doc="S_PO4 concentration (kgP/m3) at step 6"
            )
            def SPO4_AS6(blk, t):
                return blk.SPO4_AS5[t] + blk.properties_in[t].conc_mass_comp["X_PP"]

            @self.Constraint(
                self.flowsheet().time, doc="S_IP concentration (kgP/m3) output"
            )
            def SIP_output(blk, t):
                return blk.properties_out[t].conc_mass_comp["S_IP"] == blk.SPO4_AS6[t]

            @self.Expression(
                self.flowsheet().time, doc="S_IC concentration (kgC/m3) at step 6"
            )
            def SIC_AS6(blk, t):
                return (
                    blk.SIC_AS5[t]
                    + blk.properties_in[t].conc_mass_comp["X_PHA"] * self.C_PHA
                    - self.Sva_AS6
                    * blk.config.outlet_reaction_package.Ci["S_va"]
                    * mw_c
                    - self.Sbu_AS6
                    * blk.config.outlet_reaction_package.Ci["S_bu"]
                    * mw_c
                    - self.Spro_AS6
                    * blk.config.outlet_reaction_package.Ci["S_pro"]
                    * mw_c
                    - self.Sac_AS6
                    * blk.config.outlet_reaction_package.Ci["S_ac"]
                    * mw_c
                )

            @self.Constraint(
                self.flowsheet().time, doc="S_IC concentration (kgC/m3) output"
            )
            def SIC_output(blk, t):
                return blk.properties_out[t].conc_mass_comp["S_IC"] == blk.SIC_AS6[t]

            @self.Expression(
                self.flowsheet().time, doc="S_K concentration (kgK/m3) at step 6"
            )
            def SK_AS6(blk, t):
                return (
                    blk.properties_in[t].conc_mass_comp["S_K"]
                    + blk.config.outlet_reaction_package.K_XPP
                    * blk.properties_in[t].conc_mass_comp["X_PP"]
                )

            @self.Constraint(
                self.flowsheet().time,
                doc="Total potassium concentration (kgK/m3) output",
            )
            def SK_output(blk, t):
                return blk.properties_out[t].conc_mass_comp["S_K"] == blk.SK_AS6[t]

            @self.Expression(
                self.flowsheet().time, doc="S_Mg concentration (kgMg/m3) at step 6"
            )
            def SMg_AS6(blk, t):
                return (
                    blk.properties_in[t].conc_mass_comp["S_Mg"]
                    + blk.config.outlet_reaction_package.Mg_XPP
                    * blk.properties_in[t].conc_mass_comp["X_PP"]
                )

            @self.Constraint(
                self.flowsheet().time,
                doc="Total magnesium concentration (kgMg/m3) output",
            )
            def SMg_output(blk, t):
                return blk.properties_out[t].conc_mass_comp["S_Mg"] == blk.SMg_AS6[t]

        # -------------------------------------------BioP = False----------------------------------------------------------#

        # -------------------------------------------Step 4----------------------------------------------------------------#
        else:

            @self.Expression(
                self.flowsheet().time, doc="Biomass concentration (kgCOD/m3) at step 4"
            )
            # TODO: include sulfur (X_SO) in this expression after implementing sulfur-extension
            def biomass(blk, t):
                return blk.XH_AS2[t] + blk.properties_in[t].conc_mass_comp["X_AUT"]

            @self.Expression(
                self.flowsheet().time, doc="S_I concentration (kgCOD/m3) at step 4"
            )
            def SI_AS4(blk, t):
                return (
                    blk.properties_in[t].conc_mass_comp["S_I"]
                    + blk.biomass[t] * blk.config.outlet_reaction_package.f_sI_xc
                )

            @self.Constraint(
                self.flowsheet().time, doc="S_I concentration (kgCOD/m3) output"
            )
            def SI_output(blk, t):
                return blk.properties_out[t].conc_mass_comp["S_I"] == blk.SI_AS4[t]

            @self.Expression(
                self.flowsheet().time, doc="S_NH4 concentration (kgN/m3) at step 4"
            )
            def SNH4_AS4(blk, t):
                return (
                    blk.SNH4_AS3[t]
                    + blk.biomass[t] * blk.config.inlet_reaction_package.i_NBM
                    - blk.biomass[t]
                    * blk.config.outlet_reaction_package.f_sI_xc
                    * blk.config.inlet_reaction_package.i_NSI
                    - blk.biomass[t]
                    * blk.config.outlet_reaction_package.f_xI_xc
                    * blk.config.inlet_reaction_package.i_NSI
                    - blk.biomass[t]
                    * blk.config.outlet_reaction_package.f_pr_xc
                    * blk.config.outlet_reaction_package.Ni["X_pr"]
                    * mw_n
                )

            @self.Expression(
                self.flowsheet().time, doc="S_PO4 concentration (kgP/m3) at step 4"
            )
            def SPO4_AS4(blk, t):
                return (
                    blk.SPO4_AS3[t]
                    + blk.biomass[t] * blk.config.inlet_reaction_package.i_PBM
                    - blk.biomass[t]
                    * blk.config.outlet_reaction_package.f_sI_xc
                    * self.i_PSI
                    - blk.biomass[t]
                    * blk.config.outlet_reaction_package.f_xI_xc
                    * blk.config.inlet_reaction_package.i_PXI
                    - blk.biomass[t]
                    * blk.config.outlet_reaction_package.f_ch_xc
                    * blk.config.outlet_reaction_package.P_ch
                    * mw_p
                    - blk.biomass[t]
                    * blk.config.outlet_reaction_package.f_li_xc
                    * blk.config.outlet_reaction_package.Pi["X_li"]
                    * mw_p
                )

            @self.Expression(
                self.flowsheet().time, doc="S_PO4 concentration (kgC/m3) at step 4"
            )
            def SIC_AS4(blk, t):
                return (
                    blk.SIC_AS3[t]
                    + blk.biomass[t] * blk.config.inlet_reaction_package.i_CXB
                    - blk.biomass[t]
                    * blk.config.outlet_reaction_package.f_sI_xc
                    * blk.config.inlet_reaction_package.i_CSI
                    - blk.biomass[t]
                    * blk.config.outlet_reaction_package.f_xI_xc
                    * blk.config.inlet_reaction_package.i_CXI
                    - blk.biomass[t]
                    * blk.config.outlet_reaction_package.f_pr_xc
                    * blk.config.outlet_reaction_package.Ci["X_pr"]
                    * mw_c
                    - blk.biomass[t]
                    * blk.config.outlet_reaction_package.f_ch_xc
                    * blk.config.outlet_reaction_package.Ci["X_ch"]
                    * mw_c
                    - blk.biomass[t]
                    * blk.config.outlet_reaction_package.f_li_xc
                    * blk.config.outlet_reaction_package.Ci["X_li"]
                    * mw_c
                )

            @self.Expression(
                self.flowsheet().time, doc="X_I concentration (kgCOD/m3) at step 4"
            )
            def XI_AS4(blk, t):
                return (
                    blk.properties_in[t].conc_mass_comp["X_I"]
                    + blk.biomass[t] * blk.config.outlet_reaction_package.f_xI_xc
                )

            @self.Constraint(
                self.flowsheet().time, doc="X_I concentration (kgCOD/m3) output"
            )
            def XI_output(blk, t):
                return blk.properties_out[t].conc_mass_comp["X_I"] == blk.XI_AS4[t]

            self.XH_AS4 = Param(
                initialize=eps,
                units=pyunits.kg / pyunits.m**3,
                mutable=True,
                doc="Mass concentration of X_H at step 4",
            )

            @self.Expression(
                self.flowsheet().time, doc="X_PAO concentration (kgP/m3) at step 4"
            )
            def XPAO_AS4(blk, t):
                return blk.properties_in[t].conc_mass_comp["X_PAO"]

            @self.Expression(
                self.flowsheet().time, doc="X_PP concentration (kgP/m3) at step 4"
            )
            def XPP_AS4(blk, t):
                return blk.properties_in[t].conc_mass_comp["X_PP"]

            @self.Expression(
                self.flowsheet().time, doc="X_PHA concentration (kgP/m3) at step 4"
            )
            def XPHA_AS4(blk, t):
                return blk.properties_in[t].conc_mass_comp["X_PHA"]

            self.XAUT_AS4 = Param(
                initialize=eps,
                units=pyunits.kg / pyunits.m**3,
                mutable=True,
                doc="Mass concentration of X_AUT at step 4",
            )

            # -----------------------------------------Step 5--------------------------------------------------------------#
            @self.Expression(
                self.flowsheet().time, doc="Nitrogen demand for particulate inerts"
            )
            def X_ND(blk, t):
                return (
                    blk.properties_in[t].conc_mass_comp["X_S"]
                    * blk.config.inlet_reaction_package.i_NXS
                )

            @self.Expression(
                self.flowsheet().time, doc="Phosphorus demand for particulate inerts"
            )
            def X_PD(blk, t):
                return (
                    blk.properties_in[t].conc_mass_comp["X_S"]
                    * blk.config.inlet_reaction_package.i_PXS
                )

            @self.Expression(
                self.flowsheet().time, doc="Organic nitrogen from particulate inerts"
            )
            def XN_org(blk, t):
                return blk.X_ND[t] / (
                    blk.config.outlet_reaction_package.Ni["X_pr"] * mw_n
                )

            @self.Expression(self.flowsheet().time, doc="Carbohydrates mapping")
            def Xch_mapping(blk, t):
                return smooth_max(
                    0 * pyunits.kg / pyunits.m**3,
                    (blk.properties_in[t].conc_mass_comp["X_S"] - blk.XN_org[t]) * 0.4,
                    blk.eps_smooth,
                )

            # TODO: Can this be replaced with smooth_max or smooth_min?
            @self.Expression(self.flowsheet().time, doc="Protein mapping")
            def Xpr_mapping(blk, t):
                return Expr_if(
                    blk.XN_org[t] >= blk.properties_in[t].conc_mass_comp["X_S"],
                    blk.SF_AS3[t],
                    blk.XN_org[t],
                )

            @self.Expression(self.flowsheet().time, doc="Lipids mapping")
            def Xli_mapping(blk, t):
                return smooth_max(
                    0 * pyunits.kg / pyunits.m**3,
                    (blk.properties_in[t].conc_mass_comp["X_S"] - blk.XN_org[t]) * 0.6,
                    blk.eps_smooth,
                )

            @self.Constraint(
                self.flowsheet().time,
                doc="Carbohydrates output",
            )
            def Xch_output(blk, t):
                return (
                    blk.properties_out[t].conc_mass_comp["X_ch"]
                    == blk.Xch_mapping[t]
                    + blk.biomass[t] * blk.config.outlet_reaction_package.f_ch_xc
                )

            @self.Constraint(
                self.flowsheet().time,
                doc="Protein output",
            )
            def Xpr_output(blk, t):
                return (
                    blk.properties_out[t].conc_mass_comp["X_pr"]
                    == blk.Xpr_mapping[t]
                    + blk.biomass[t] * blk.config.outlet_reaction_package.f_pr_xc
                )

            @self.Constraint(
                self.flowsheet().time,
                doc="Lipids output",
            )
            def Xli_output(blk, t):
                return (
                    blk.properties_out[t].conc_mass_comp["X_li"]
                    == blk.Xli_mapping[t]
                    + blk.biomass[t] * blk.config.outlet_reaction_package.f_li_xc
                )

            @self.Expression(
                self.flowsheet().time, doc="S_NH4 concentration (kgN/m3) at step 5"
            )
            def SNH4_AS5(blk, t):
                return (
                    blk.SNH4_AS4[t]
                    + blk.properties_in[t].conc_mass_comp["X_S"]
                    * blk.config.inlet_reaction_package.i_NXS
                    - blk.Xpr_mapping[t]
                    * blk.config.outlet_reaction_package.Ni["X_pr"]
                    * mw_n
                )

            @self.Constraint(
                self.flowsheet().time, doc="S_NH4 concentration (kgN/m3) output"
            )
            def SIN_output(blk, t):
                return blk.properties_out[t].conc_mass_comp["S_IN"] == blk.SNH4_AS5[t]

            @self.Expression(
                self.flowsheet().time, doc="S_PO4 concentration (kgP/m3) at step 5"
            )
            def SPO4_AS5(blk, t):
                return (
                    blk.SPO4_AS4[t]
                    + blk.properties_in[t].conc_mass_comp["X_S"]
                    * blk.config.inlet_reaction_package.i_PXS
                    - blk.Xch_mapping[t]
                    * blk.config.outlet_reaction_package.P_ch
                    * mw_p
                    - blk.Xli_mapping[t]
                    * blk.config.outlet_reaction_package.Pi["X_li"]
                    * mw_p
                )

            @self.Constraint(
                self.flowsheet().time, doc="S_IP concentration (kgP/m3) output"
            )
            def SIP_output(blk, t):
                return blk.properties_out[t].conc_mass_comp["S_IP"] == blk.SPO4_AS5[t]

            @self.Expression(
                self.flowsheet().time, doc="S_IC concentration (kgC/m3) at step 5"
            )
            def SIC_AS5(blk, t):
                return (
                    blk.SIC_AS4[t]
                    + blk.properties_in[t].conc_mass_comp["X_S"]
                    * blk.config.inlet_reaction_package.i_CXS
                    - blk.Xch_mapping[t]
                    * blk.config.outlet_reaction_package.Ci["X_ch"]
                    * mw_c
                    - blk.Xpr_mapping[t]
                    * blk.config.outlet_reaction_package.Ci["X_pr"]
                    * mw_c
                    - blk.Xli_mapping[t]
                    * blk.config.outlet_reaction_package.Ci["X_li"]
                    * mw_c
                )

            @self.Constraint(
                self.flowsheet().time, doc="S_IC concentration (kgC/m3) output"
            )
            def SIC_output(blk, t):
                return blk.properties_out[t].conc_mass_comp["S_IC"] == blk.SIC_AS5[t]

            self.XS_AS5 = Param(
                initialize=eps,
                units=pyunits.kg / pyunits.m**3,
                mutable=True,
                doc="Mass concentration of X_S at step 5",
            )

            self.XPAO_AS5 = Param(
                initialize=self.properties_in[0].conc_mass_comp["X_PAO"],
                units=pyunits.kg / pyunits.m**3,
                mutable=True,
                doc="Mass concentration of X_PAO at step 5",
            )

            self.XPP_AS5 = Param(
                initialize=self.properties_in[0].conc_mass_comp["X_PP"],
                units=pyunits.kg / pyunits.m**3,
                mutable=True,
                doc="Mass concentration of X_PP at step 5",
            )

            self.XPHA_AS5 = Param(
                initialize=self.properties_in[0].conc_mass_comp["X_PHA"],
                units=pyunits.kg / pyunits.m**3,
                mutable=True,
                doc="Mass concentration of X_PHA at step 5",
            )

            # -----------------------------------------Step 6--------------------------------------------------------------#
            @self.Constraint(
                self.flowsheet().time,
                doc="Total valerate concentration",
            )
            def Sva_output(blk, t):
                return (
                    blk.properties_out[t].conc_mass_comp["S_va"]
                    == eps * pyunits.kg / pyunits.m**3
                )

            @self.Constraint(
                self.flowsheet().time,
                doc="Total butyrate concentration",
            )
            def Sbu_output(blk, t):
                return (
                    blk.properties_out[t].conc_mass_comp["S_bu"]
                    == eps * pyunits.kg / pyunits.m**3
                )

            @self.Constraint(
                self.flowsheet().time,
                doc="Total propionate concentration",
            )
            def Spro_output(blk, t):
                return (
                    blk.properties_out[t].conc_mass_comp["S_pro"]
                    == eps * pyunits.kg / pyunits.m**3
                )

            @self.Constraint(
                self.flowsheet().time,
                doc="Total acetate concentration",
            )
            def Sac_output(blk, t):
                return (
                    blk.properties_out[t].conc_mass_comp["S_ac"]
                    == eps * pyunits.kg / pyunits.m**3 + blk.SA_AS2[t]
                )

            @self.Constraint(
                self.flowsheet().time, doc="X_PAO concentration (kgCOD/m3) output"
            )
            def XPAO_output(blk, t):
                return (
                    blk.properties_out[t].conc_mass_comp["X_PAO"]
                    == blk.properties_in[0].conc_mass_comp["X_PAO"]
                )

            @self.Constraint(
                self.flowsheet().time, doc="X_PP concentration (kgP/m3) output"
            )
            def XPP_output(blk, t):
                return (
                    blk.properties_out[t].conc_mass_comp["X_PP"]
                    == blk.properties_in[0].conc_mass_comp["X_PP"]
                )

            @self.Constraint(
                self.flowsheet().time, doc="X_PHA concentration (kgCOD/m3) output"
            )
            def XPHA_output(blk, t):
                return (
                    blk.properties_out[t].conc_mass_comp["X_PHA"]
                    == blk.properties_in[0].conc_mass_comp["X_PHA"]
                )

            @self.Constraint(
                self.flowsheet().time,
                doc="Total magnesium concentration (kgK/m3) output",
            )
            def SK_output(blk, t):
                return (
                    blk.properties_out[t].conc_mass_comp["S_K"]
                    == blk.properties_in[t].conc_mass_comp["S_K"]
                )

            @self.Constraint(
                self.flowsheet().time,
                doc="Total magnesium concentration (kgMg/m3) output",
            )
            def SMg_output(blk, t):
                return (
                    blk.properties_out[t].conc_mass_comp["S_Mg"]
                    == blk.properties_in[t].conc_mass_comp["S_Mg"]
                )

        self.zero_flow_components = Set(
            initialize=[
                "S_fa",
                "S_h2",
                "S_ch4",
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
                == eps * pyunits.kg / pyunits.m**3
            )

        # TODO: Relationship carried over from ASM1/ADM1 - not clear how Flores-Alsina handles ions (see perf_plant_AD_ss.m?)
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
