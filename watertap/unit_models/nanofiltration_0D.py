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
Simple zero-dimensional Nanofiltration unit model.
"""

from pyomo.environ import Block, Constraint, Set, value, Var
from pyomo.common.config import ConfigDict, ConfigValue, In

from idaes.core import (
    declare_process_block_class,
    EnergyBalanceType,
    MomentumBalanceType,
    useDefault,
    UnitModelBlockData,
)
from idaes.core.initialization import ModularInitializerBase
from idaes.core.scaling import CustomScalerBase, ConstraintScalingScheme
from idaes.core.util.config import is_physical_parameter_block
from idaes.core.util.exceptions import ConfigurationError
import idaes.logger as idaeslog


_log = idaeslog.getLogger(__name__)


class Nanofiltration0DInitializer(ModularInitializerBase):
    """
    Initializer for 0D Nanofiltration models.

    """

    CONFIG = ModularInitializerBase.CONFIG()

    def initialize_main_model(
        self,
        model: Block,
    ):
        """
        Initialization routine for main Nanofiltration0D models.

        Args:
            model: Pyomo Block to be initialized.
            copy_inlet_state: bool (default=False). Whether to copy inlet state to other states or not.
                Copying will generally be faster, but inlet states may not contain
                all properties required elsewhere.

        Returns:
            Pyomo solver results object.

        """
        # Get loggers
        init_log = idaeslog.getInitLogger(
            model.name, self.get_output_level(), tag="unit"
        )
        solve_log = idaeslog.getSolveLogger(
            model.name, self.get_output_level(), tag="unit"
        )

        # Create solver
        solver = self._get_solver()

        # ---------------------------------------------------------------------
        # Initialize inlet state
        prop_init = self.get_submodel_initializer(model.properties_in)

        if prop_init is not None:
            prop_init.initialize(
                model=model.properties_in,
                output_level=self.get_output_level(),
            )
        else:
            raise ValueError(
                "No Initializer found for property package. Please set provide "
                "sub-model initializers or assign a default Initializer."
            )

        init_log.info_high("Initialization Step 1 (Inlet State) Complete.")

        # ---------------------------------------------------------------------
        # Estimate outlet states from inlet and initialize
        for t, in_state in model.properties_in.items():
            state_vars = in_state.define_state_vars()
            state_vars_ret = model.properties_retentate[t].define_state_vars()
            state_vars_per = model.properties_permeate[t].define_state_vars()

            for n, sv in state_vars.items():
                sv_ret = state_vars_ret[n]
                sv_per = state_vars_per[n]

                if sv.local_name == "pressure" and not sv_ret.fixed:
                    # For pressure, only retentate is linked to inlet
                    if hasattr(model, "deltaP"):
                        sv_ret.set_value(sv + model.deltaP[t])
                    else:
                        sv_ret.set_value(sv)
                elif "flow" in sv.local_name:
                    self._init_flow(model, t, in_state, sv, sv_ret, sv_per)
                elif any(
                    sv.local_name.startswith(i) for i in ["mass_frac", "mole_frac"]
                ):
                    self._init_frac(model, t, in_state, sv, sv_ret, sv_per)
                elif sv.local_name.startswith("conc"):
                    self._init_conc(model, t, in_state, sv, sv_ret, sv_per)
                else:
                    # For everything else, assume outlet similar to inlet
                    for k, svd in sv.items():
                        if not sv_ret[k].fixed:
                            sv_ret[k].set_value(svd)
                        if not sv_per[k].fixed:
                            sv_per[k].set_value(svd)

        prop_init.initialize(
            model=model.properties_retentate,
            output_level=self.get_output_level(),
        )
        prop_init.initialize(
            model=model.properties_permeate,
            output_level=self.get_output_level(),
        )

        init_log.info_high("Initialization Step 2 (Outlet States) Complete.")

        # ---------------------------------------------------------------------
        # Solve full model
        with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
            res = solver.solve(model, tee=slc.tee)

        init_log.info("Initialization Completed, {}".format(idaeslog.condition(res)))

        return res

    def _init_conc(self, model, t, in_state, sv, sv_ret, sv_per):
        # Component indexed flow - need to apply split fractions
        for k, svd in sv.items():
            if isinstance(k, str):
                # Single indexing set, component is the index
                j = k
            else:
                # Component is always the last index
                j = k[-1]

            # Determine split based on type
            comp = in_state.params.get_component(j)

            if comp.is_solvent():
                # Assume solvent concentrations unchanged
                sv_ret[k].set_value(svd)
                sv_per[k].set_value(svd)
            else:
                if j == model.config.electroneutrality_ion:
                    # Guess electroneutrality ion will be based on solvent recovery
                    split = model.solvent_recovery[t]
                else:
                    # For initialization, assume density is roughly constant and rejection can be used as split fraction
                    split = 1 - model.rejection_comp[t, j]

                if not sv_ret[k].fixed:
                    sv_ret[k].set_value(svd * (1 - split))
                if not sv_per[k].fixed:
                    sv_per[k].set_value(svd * split)

    def _init_flow(self, model, t, in_state, sv, sv_ret, sv_per):
        # Determine if it is a component flow or not
        if sv.local_name.endswith("_comp"):
            # Component indexed flow - need to apply split fractions
            for k, svd in sv.items():
                if isinstance(k, str):
                    # Single indexing set, component is the index
                    j = k
                else:
                    # Component is always the last index
                    j = k[-1]

                # Determine split based on type
                comp = in_state.params.get_component(j)

                if comp.is_solvent() or j == model.config.electroneutrality_ion:
                    # Assume electroneutrality ion will have similar separation to solvents
                    split = model.solvent_recovery[t]
                else:
                    # For initialization, assume density is roughly constant and rejection can be used as split fraction
                    split = 1 - model.rejection_comp[t, j]

                if not sv_ret[k].fixed:
                    sv_ret[k].set_value(svd * (1 - split))
                if not sv_per[k].fixed:
                    sv_per[k].set_value(svd * split)
        else:
            # Assume a total flow basis, and use solvent recovery
            split = model.solvent_recovery[t]
            for k, svd in sv.items():
                if not sv_ret[k].fixed:
                    sv_ret[k].set_value(svd * (1 - split))
                if not sv_per[k].fixed:
                    sv_per[k].set_value(svd * split)

    def _init_frac(self, model, t, in_state, sv, sv_ret, sv_per):
        # First need to iterate over all indices to collect normalized flows
        # Assume a basis of 1 mass or mole unit
        # Also, only single phase property packages supported, so we only need to
        # worry about the component index
        nom_ret_comp_flow = {}
        nom_per_comp_flow = {}

        for k, svd in sv.items():
            if isinstance(k, str):
                # Single indexing set, component is the index
                j = k
            else:
                # Component is always the last index
                j = k[-1]

            # Determine split based on type
            comp = in_state.params.get_component(j)

            if comp.is_solvent() or j == model.config.electroneutrality_ion:
                # Assume electroneutrality ion will have similar separation to solvents
                split = model.solvent_recovery[t]
            else:
                # For initialization, assume density is roughly constant and rejection can be used as split fraction
                split = 1 - model.rejection_comp[t, j]

            nom_ret_comp_flow[j] = value(svd * (1 - split))
            nom_per_comp_flow[j] = value(svd * split)

        nom_ret_tot_flow = sum(f for f in nom_ret_comp_flow.values())
        nom_per_tot_flow = sum(f for f in nom_per_comp_flow.values())

        # # Next set value based on fraction of nominal total flow
        for k, svd in sv.items():
            if isinstance(k, str):
                # Single indexing set, component is the index
                j = k
            else:
                # Component is always the last index
                j = k[-1]

            if not sv_ret[k].fixed:
                sv_ret[k].set_value(nom_ret_comp_flow[j] / nom_ret_tot_flow)
            if not sv_per[k].fixed:
                sv_per[k].set_value(nom_per_comp_flow[j] / nom_per_tot_flow)


class Nanofiltration0DScaler(CustomScalerBase):
    """
    Scaler class for Nanofiltration0D models.
    """

    DEFAULT_SCALING_FACTORS = {
        "deltaP": 1e-4,
        "rejection_comp": 1e2,
        "solvent_recovery": 10,
    }

    def variable_scaling_routine(
        self, model, overwrite: bool = False, submodel_scalers: dict = None
    ):
        """
        Apply variable scaling to model.

        Args:
            model: model to be scaled
            overwrite: whether to overwrite existing scaling factors
            submodel_scalers: ComponentMap of Scaler objects to be applied to sub-models

        Returns:
            None

        """
        # Scale inlet state
        self.call_submodel_scaler_method(
            submodel=model.properties_in,
            method="variable_scaling_routine",
            submodel_scalers=submodel_scalers,
            overwrite=overwrite,
        )

        # Propagate scaling from inlet state
        self.propagate_state_scaling(
            target_state=model.properties_retentate,
            source_state=model.properties_in,
            overwrite=overwrite,
        )
        self.propagate_state_scaling(
            target_state=model.properties_permeate,
            source_state=model.properties_in,
            overwrite=overwrite,
        )

        # Scale remaining states
        self.call_submodel_scaler_method(
            submodel=model.properties_retentate,
            method="variable_scaling_routine",
            submodel_scalers=submodel_scalers,
            overwrite=overwrite,
        )
        self.call_submodel_scaler_method(
            submodel=model.properties_permeate,
            method="variable_scaling_routine",
            submodel_scalers=submodel_scalers,
            overwrite=overwrite,
        )

        # Scale remaining variables
        # Variables with default scaling factors
        for n in self.DEFAULT_SCALING_FACTORS.keys():
            c = model.find_component(n)
            for v in c.values():
                self.scale_variable_by_default(v, overwrite=overwrite)

    def constraint_scaling_routine(
        self, model, overwrite: bool = False, submodel_scalers: dict = None
    ):
        """
        Apply constraint scaling to model.

        Args:
            model: model to be scaled
            overwrite: whether to overwrite existing scaling factors
            submodel_scalers: ComponentMap of Scaler objects to be applied to sub-models

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
            submodel=model.properties_retentate,
            method="constraint_scaling_routine",
            submodel_scalers=submodel_scalers,
            overwrite=overwrite,
        )
        self.call_submodel_scaler_method(
            submodel=model.properties_permeate,
            method="constraint_scaling_routine",
            submodel_scalers=submodel_scalers,
            overwrite=overwrite,
        )

        # Scale remaining constraints
        for c in model.component_data_objects(Constraint, descend_into=False):
            self.scale_constraint_by_nominal_value(
                c,
                scheme=ConstraintScalingScheme.inverseMaximum,
                overwrite=overwrite,
            )


@declare_process_block_class("Nanofiltration0D")
class Nanofiltration0DData(UnitModelBlockData):
    """
    Nanofiltration model.
    """

    default_initializer = Nanofiltration0DInitializer
    default_scaler = Nanofiltration0DScaler

    CONFIG = ConfigDict()
    CONFIG.declare(
        "dynamic",
        ConfigValue(
            domain=In([False]),
            default=False,
            description="Dynamic model flag - must be False",
            doc="""Indicates whether this model will be dynamic or not,
    **default** = False. Product blocks are always steady-state.""",
        ),
    )
    CONFIG.declare(
        "has_holdup",
        ConfigValue(
            default=False,
            domain=In([False]),
            description="Holdup construction flag - must be False",
            doc="""Product blocks do not contain holdup, thus this must be
    False.""",
        ),
    )
    CONFIG.declare(
        "property_package",
        ConfigValue(
            default=useDefault,
            domain=is_physical_parameter_block,
            description="Property package to use for mixer",
            doc="""Property parameter object used to define property
    calculations,
    **default** - useDefault.
    **Valid values:** {
    **useDefault** - use default package from parent model or flowsheet,
    **PropertyParameterObject** - a PropertyParameterBlock object.}""",
        ),
    )
    CONFIG.declare(
        "property_package_args",
        ConfigDict(
            implicit=True,
            description="Arguments to use for constructing property packages",
            doc="""A ConfigDict with arguments to be passed to a property
    block(s) and used when constructing these,
    **default** - None.
    **Valid values:** {
    see property package for documentation.}""",
        ),
    )
    CONFIG.declare(
        "momentum_balance_type",
        ConfigValue(
            default=MomentumBalanceType.pressureTotal,
            domain=In(MomentumBalanceType),
            description="Retentate momentum balance construction flag",
            doc="""Indicates what type of momentum balance should be constructed
    for the retentate side. Only  MomentumBalanceType.none and
    MomentumBalanceType.pressureTotal (default) are supported.
    **Valid values:** {
    **MomentumBalanceType.none** - exclude momentum balances,
    **MomentumBalanceType.pressureTotal** - single pressure balance for material}""",
        ),
    )
    CONFIG.declare(
        "has_pressure_change",
        ConfigValue(
            default=False,
            domain=In([True, False]),
            description="Retentate pressure change term construction flag",
            doc="""Indicates whether terms for retentate pressure change should be
    constructed,
    **default** - False.
    **Valid values:** {
    **True** - include pressure change terms,
    **False** - exclude pressure change terms.}""",
        ),
    )
    CONFIG.declare(
        "energy_balance_type",
        ConfigValue(
            default=EnergyBalanceType.isothermal,
            domain=In(EnergyBalanceType),
            description="Indicated what type of energy balance should be constructed.",
            doc="""Indicates what type of momentum balance should be constructed
    for the retentate side. Only  EnergyBalanceType.none and
    EnergyBalanceType.isothermal (default) are supported.""",
        ),
    )
    # TODO: Support multiple electroneutrality ions
    # This needs some rule for prioritizing each ion.
    CONFIG.declare(
        "electroneutrality_ion",
        ConfigValue(
            default="Cl_-",
            domain=str,
            description="Balancing ion to use to maintain electroneutrality",
            doc="""Name of ion to use to maintain electroneutrality in
            permeate stream. If an ion is provided, the split fraction
            for this ion will be solved implicitly to ensure electroneutrality
            is maintained in the permeate stream. If None, user must provide
            spl;it fractions for all monovalent ions and electroneutrality is
            not guaranteed.""",
        ),
    )
    CONFIG.declare(
        "passing_species_list",
        ConfigValue(
            default=None,
            domain=list,
            description="List of solutes with low rejection",
            doc="""List of solute species which pass through the nanofiltration membrane
            with low rejection. If not provided (None), solutes will be classified based
            on their charge, with monovalent and uncharged species considered passing.""",
        ),
    )
    CONFIG.declare(
        "default_passing_rejection",
        ConfigValue(
            default=0.1,
            domain=float,
            description="Default value for rejection of passing species",
            doc="""Default value to be assigned as the rejection fraction for species
            identified as passing through the membrane with low rejection.""",
        ),
    )
    CONFIG.declare(
        "default_excluded_rejection",
        ConfigValue(
            default=1 - 1e-10,
            domain=float,
            description="Default value for rejection of excluded species",
            doc="""Default value to be assigned as the rejection fraction for species
            identified as be excluded by membrane (high rejection).""",
        ),
    )

    def build(self):
        super().build()

        prop_params = self.config.property_package

        # Check that property package only supports single phase
        # TODO: Extend the model to handle phase equilibrium
        if len(prop_params.phase_list) > 1:
            raise ConfigurationError(
                "Nanofiltration0D model only supports single phase "
                "property packages."
            )

        # Check that the electroneutrality ion is a valid passing species
        # I.e. it must be in the passing species list or a monovalent ion
        if self.config.electroneutrality_ion is not None:
            try:
                bal_ion = prop_params.get_component(self.config.electroneutrality_ion)
            except AttributeError:
                raise ConfigurationError(
                    f"electroneutrality_ion ({self.config.electroneutrality_ion}) "
                    "is not a valid component in property package."
                )

            if self.config.passing_species_list is not None:
                if (
                    self.config.electroneutrality_ion
                    not in self.config.passing_species_list
                ):
                    raise ConfigurationError(
                        f"electroneutrality_ion ({self.config.electroneutrality_ion}) "
                        "must be a member of the passing species list."
                    )
            else:
                # Check that balancing ion is monovalent
                if (
                    not hasattr(bal_ion.config, "charge")
                    or abs(bal_ion.config.charge) > 1
                ):
                    raise ConfigurationError(
                        f"electroneutrality_ion ({self.config.electroneutrality_ion}) "
                        "must be a monovalent ion."
                    )

        # Check energy balance type is supported
        if self.config.energy_balance_type not in (
            EnergyBalanceType.none,
            EnergyBalanceType.isothermal,
        ):
            raise ConfigurationError(
                f"Nanofiltration0D model only supports isothermal operation or no energy balance "
                f"(assigned {self.config.energy_balance_type})"
            )

        # Check that pressure balance arguments are consistent
        if self.config.momentum_balance_type not in (
            MomentumBalanceType.none,
            MomentumBalanceType.pressureTotal,
        ):
            raise ConfigurationError(
                f"Nanofiltration0D model only supports total pressure balance or no pressure balance "
                f"(assigned {self.config.momentum_balance_type})"
            )
        elif (
            self.config.has_pressure_change
            and self.config.momentum_balance_type == MomentumBalanceType.none
        ):
            raise ConfigurationError(
                "Inconsistent configuration arguments. has_pressure_change=True "
                "requires that momentum_balance_type not equal MomentumBalanceType.none."
            )

        tmp_dict_in = dict(**self.config.property_package_args)
        tmp_dict_in["has_phase_equilibrium"] = False
        tmp_dict_in["defined_state"] = True

        self.properties_in = self.config.property_package.build_state_block(
            self.flowsheet().time, doc="Material properties at inlet", **tmp_dict_in
        )

        tmp_dict_out = dict(**self.config.property_package_args)
        tmp_dict_out["has_phase_equilibrium"] = False
        tmp_dict_out["defined_state"] = False

        self.properties_retentate = self.config.property_package.build_state_block(
            self.flowsheet().time, doc="Material properties at inlet", **tmp_dict_out
        )

        self.properties_permeate = self.config.property_package.build_state_block(
            self.flowsheet().time, doc="Material properties at inlet", **tmp_dict_out
        )

        self.add_port("inlet", self.properties_in, doc="Inlet Port")
        self.add_port("retentate", self.properties_retentate, doc="Retentate Port")
        self.add_port("permeate", self.properties_permeate, doc="Permeate Port")

        # NF separation variables
        self.solvent_recovery = Var(self.flowsheet().time, initialize=0.8)

        self._solute_set = Set(
            initialize=[
                j
                for j in prop_params.component_list
                if not prop_params.get_component(j).is_solvent()
                and j != self.config.electroneutrality_ion
            ]
        )

        def _init_rejection(b, _, j):
            if b.config.passing_species_list is not None:
                if j in b.config.passing_species_list:
                    return b.config.default_passing_rejection
                return b.config.default_excluded_rejection
            else:
                if j in prop_params.ion_set:
                    # Check charge
                    comp = prop_params.get_component(j)
                    if abs(comp.config.charge) == 1:
                        # Monovalent - include in set
                        return b.config.default_passing_rejection
                    else:
                        return b.config.default_excluded_rejection
                else:
                    # Non-ionic solute - assume passing species
                    return b.config.default_passing_rejection

        self.rejection_comp = Var(
            self.flowsheet().time,
            self._solute_set,
            initialize=_init_rejection,
            bounds=(1e-10, 1),
            doc="Solute rejection fractions",
        )

        units = self.config.property_package.get_metadata().derived_units
        if self.config.has_pressure_change:
            self.deltaP = Var(
                self.flowsheet().time,
                initialize=0,
                units=units.PRESSURE,
                doc="Retentate side pressure change",
            )

        # Material balance
        @self.Constraint(self.flowsheet().time, prop_params.component_list)
        def material_balances(b, t, j):
            pset = b.properties_in[t].phase_list
            pcset = b.properties_in[t].phase_component_set
            return sum(
                b.properties_in[t].get_material_flow_terms(p, j)
                for p in pset
                if (p, j) in pcset
            ) == sum(
                b.properties_retentate[t].get_material_flow_terms(p, j)
                for p in pset
                if (p, j) in pcset
            ) + sum(
                b.properties_permeate[t].get_material_flow_terms(p, j)
                for p in pset
                if (p, j) in pcset
            )

        @self.Constraint(self.flowsheet().time, self.properties_in.phase_component_set)
        def separation_constraint(b, t, p, j):
            comp = prop_params.get_component(j)

            if comp.is_solvent():
                # Permeate flows equal to recovery * inlet flow
                return b.solvent_recovery[t] * b.properties_in[
                    t
                ].get_material_flow_terms(p, j) == b.properties_permeate[
                    t
                ].get_material_flow_terms(
                    p, j
                )
            elif j == self.config.electroneutrality_ion:
                # Rejection of electroneutrality ion will be solved using electroneutrality constraint
                return Constraint.Skip

            # else: Rejection = 1 - C_permeate/C_feed
            return (
                (1 - b.rejection_comp[t, j])
                * b.properties_in[t].conc_mol_phase_comp[p, j]
            ) == b.properties_permeate[t].conc_mol_phase_comp[p, j]

        if self.config.electroneutrality_ion is not None:
            # Add electroneutrality constraint for permeate stream
            @self.Constraint(self.flowsheet().time)
            def permeate_electronegativity(b, t):
                perm_state = b.properties_permeate[t]
                return 0 == sum(
                    perm_state.params.get_component(j).config.charge
                    * perm_state.flow_mol_phase_comp[p, j]
                    for p, j in perm_state.phase_component_set
                    if j in perm_state.params.ion_set
                )

        # Retentate pressure balance
        if self.config.momentum_balance_type == MomentumBalanceType.pressureTotal:

            @self.Constraint(self.flowsheet().time, doc="Retentate pressure balance")
            def retentate_pressure_balance(b, t):
                expr = b.properties_retentate[t].pressure
                if b.config.has_pressure_change:
                    expr += -b.deltaP[t]
                return b.properties_in[t].pressure == expr

        # Temperature equalities
        if self.config.energy_balance_type == EnergyBalanceType.isothermal:

            @self.Constraint(
                self.flowsheet().time, doc="Retentate temperature equality"
            )
            def retentate_temperature_equality(b, t):
                return (
                    b.properties_in[t].temperature
                    == b.properties_retentate[t].temperature
                )

            @self.Constraint(self.flowsheet().time, doc="Permeate temperature equality")
            def permeate_temperature_equality(b, t):
                return (
                    b.properties_in[t].temperature
                    == b.properties_permeate[t].temperature
                )
