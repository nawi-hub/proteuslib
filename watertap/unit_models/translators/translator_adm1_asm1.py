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
Translator block representing the ADM1/ASM1 interface.

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
    Param,
    units as pyunits,
    check_optimal_termination,
    Set,
)

__author__ = "Alejandro Garciadiego, Andrew Lee, Xinhong Liu"


# Set up logger
_log = idaeslog.getLogger(__name__)


class ADM1ASM1Scaler(CustomScalerBase):
    """
    Default modular scaler for the ADM1-ASM1 translator block.
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


@declare_process_block_class("Translator_ADM1_ASM1")
class TranslatorDataADM1ASM1(TranslatorData):
    """
    Translator block representing the ADM1/ASM1 interface
    """

    default_scaler = ADM1ASM1Scaler

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
        super(TranslatorDataADM1ASM1, self).build()

        self.i_xe = Param(
            initialize=0.06,
            units=pyunits.dimensionless,
            mutable=True,
            doc="Nitrogen inert content",
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

        self.unchanged_component = Set(initialize=["S_I", "X_I"])

        @self.Constraint(
            self.flowsheet().time,
            self.unchanged_component,
            doc="Equality equation for unchanged components",
        )
        def eq_unchanged_conc(blk, t, i):
            return (
                blk.properties_out[t].conc_mass_comp[i]
                == blk.properties_in[t].conc_mass_comp[i]
            )

        self.readily_biodegradable = Set(
            initialize=["S_su", "S_aa", "S_fa", "S_va", "S_bu", "S_pro", "S_ac"]
        )

        self.slowly_biodegradable = Set(
            initialize=[
                "X_c",
                "X_ch",
                "X_pr",
                "X_li",
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
            doc="Equality S_S equation",
        )
        def eq_SS_conc(blk, t):
            return blk.properties_out[t].conc_mass_comp["S_S"] == sum(
                blk.properties_in[t].conc_mass_comp[i]
                for i in blk.readily_biodegradable
            )

        @self.Constraint(
            self.flowsheet().time,
            doc="Equality X_S equation",
        )
        def eq_XS_conc(blk, t):
            return blk.properties_out[t].conc_mass_comp["X_S"] == sum(
                blk.properties_in[t].conc_mass_comp[i] for i in blk.slowly_biodegradable
            )

        @self.Constraint(
            self.flowsheet().time,
            doc="Equality S_NH equation",
        )
        def eq_Snh_conc(blk, t):
            return (
                blk.properties_out[t].conc_mass_comp["S_NH"]
                == blk.properties_in[t].conc_mass_comp["S_IN"]
            )

        @self.Constraint(
            self.flowsheet().time,
            doc="Equality S_ND equation",
        )
        def eq_Snd_conc(blk, t):
            return blk.properties_out[t].conc_mass_comp["S_ND"] == mw_n * (
                (
                    blk.properties_in[t].conc_mass_comp["S_I"]
                    * blk.config.reaction_package.N_I
                )
                + (
                    blk.properties_in[t].conc_mass_comp["S_aa"]
                    * blk.config.reaction_package.N_aa
                )
            )

        @self.Constraint(
            self.flowsheet().time,
            doc="Equality Xnd equation",
        )
        def eq_Xnd_conc(blk, t):
            return blk.properties_out[t].conc_mass_comp["X_ND"] == (
                mw_n
                * (
                    (
                        blk.config.reaction_package.N_bac
                        * (
                            blk.properties_in[t].conc_mass_comp["X_su"]
                            + blk.properties_in[t].conc_mass_comp["X_aa"]
                            + blk.properties_in[t].conc_mass_comp["X_fa"]
                            + blk.properties_in[t].conc_mass_comp["X_c4"]
                            + blk.properties_in[t].conc_mass_comp["X_pro"]
                            + blk.properties_in[t].conc_mass_comp["X_ac"]
                            + blk.properties_in[t].conc_mass_comp["X_h2"]
                        )
                    )
                    + (
                        blk.properties_in[t].conc_mass_comp["X_I"]
                        * blk.config.reaction_package.N_I
                    )
                    + (
                        blk.properties_in[t].conc_mass_comp["X_c"]
                        * blk.config.reaction_package.N_xc
                    )
                    + (
                        blk.properties_in[t].conc_mass_comp["X_pr"]
                        * blk.config.reaction_package.N_aa
                    )
                )
                - (blk.properties_in[t].conc_mass_comp["X_I"] * blk.i_xe)
            )

        @self.Constraint(
            self.flowsheet().time,
            doc="Equality alkalinity equation",
        )
        def return_Salk(blk, t):
            return (
                blk.properties_out[t].alkalinity
                == blk.properties_in[t].conc_mass_comp["S_IC"] / mw_c
            )

        self.zero_flow_components = Set(
            initialize=["X_BH", "X_BA", "X_P", "S_O", "S_NO"]
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
