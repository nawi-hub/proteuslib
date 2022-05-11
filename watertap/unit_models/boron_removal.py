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
    Block,
    Set,
    Var,
    Param,
    Expression,
    Suffix,
    NonNegativeReals,
    PositiveIntegers,
    Reference,
    value,
    exp,
    log10,
    units as pyunits,
)

from pyomo.common.config import ConfigBlock, ConfigValue, In

# Import IDAES cores
from idaes.core import (
    ControlVolume0DBlock,
    declare_process_block_class,
    MaterialBalanceType,
    EnergyBalanceType,
    MomentumBalanceType,
    UnitModelBlockData,
    useDefault,
    MaterialFlowBasis,
)
from idaes.core.util.constants import Constants
from idaes.core.util import get_solver
from idaes.core.util.tables import create_stream_table_dataframe
from idaes.core.util.config import is_physical_parameter_block
from idaes.core.util.exceptions import ConfigurationError
import idaes.core.util.scaling as iscale
import idaes.logger as idaeslog

__author__ = "Austin Ladshaw"

_log = idaeslog.getLogger(__name__)

# Name of the unit model
@declare_process_block_class("BoronRemoval")
class BoronRemovalData(UnitModelBlockData):
    """
    0D Boron Removal model for after 1st Stage of RO

    This model is an approximate equilibrium reactor wherein it is
    assumed that:
        (1) All reactions and activities are assumed ideal
        (2) Only major reactions are water dissociation
                and boron dissociation
        (3) Only 1 caustic chemical is being added to raise pH
        (4) The caustic additive will always completely dissociate into
                a cation and some amount hydroxide anions
                (e.g., NaOH --> Na+ + OH-, Ca(OH2) --> Ca2+ + 2 OH-, etc)
        (5) Any other ions remaining in solution do not significantly
                change with pH changes, but do help act as buffers to
                changes in pH (i.e., will absorb/contribute protons
                proportional to their charge).
    """

    # CONFIG are options for the unit model
    CONFIG = ConfigBlock()

    CONFIG.declare(
        "dynamic",
        ConfigValue(
            domain=In([False]),
            default=False,
            description="Dynamic model flag; must be False",
            doc="""Indicates whether this model will be dynamic or not,
    **default**: False. The filtration unit does not support dynamic
    behavior, thus this must be False.""",
        ),
    )

    CONFIG.declare(
        "has_holdup",
        ConfigValue(
            default=False,
            domain=In([False]),
            description="Holdup construction flag; must be False",
            doc="""Indicates whether holdup terms should be constructed or not.
    **default**: False. The filtration unit does not have defined volume, thus
    this must be False.""",
        ),
    )

    CONFIG.declare(
        "material_balance_type",
        ConfigValue(
            default=MaterialBalanceType.useDefault,
            domain=In(MaterialBalanceType),
            description="Material balance construction flag",
            doc="""Indicates what type of mass balance should be constructed,
    **default** - MaterialBalanceType.useDefault.
    **Valid values:** {
    **MaterialBalanceType.useDefault - refer to property package for default
    balance type
    **MaterialBalanceType.none** - exclude material balances,
    **MaterialBalanceType.componentPhase** - use phase component balances,
    **MaterialBalanceType.componentTotal** - use total component balances,
    **MaterialBalanceType.elementTotal** - use total element balances,
    **MaterialBalanceType.total** - use total material balance.}""",
        ),
    )

    # NOTE: This option is temporarily disabled
    '''
    CONFIG.declare("energy_balance_type", ConfigValue(
        default=EnergyBalanceType.useDefault,
        domain=In(EnergyBalanceType),
        description="Energy balance construction flag",
        doc="""Indicates what type of energy balance should be constructed,
    **default** - EnergyBalanceType.useDefault.
    **Valid values:** {
    **EnergyBalanceType.useDefault - refer to property package for default
    balance type
    **EnergyBalanceType.none** - exclude energy balances,
    **EnergyBalanceType.enthalpyTotal** - single enthalpy balance for material,
    **EnergyBalanceType.enthalpyPhase** - enthalpy balances for each phase,
    **EnergyBalanceType.energyTotal** - single energy balance for material,
    **EnergyBalanceType.energyPhase** - energy balances for each phase.}"""))
    '''

    CONFIG.declare(
        "momentum_balance_type",
        ConfigValue(
            default=MomentumBalanceType.pressureTotal,
            domain=In(MomentumBalanceType),
            description="Momentum balance construction flag",
            doc="""Indicates what type of momentum balance should be constructed,
    **default** - MomentumBalanceType.pressureTotal.
    **Valid values:** {
    **MomentumBalanceType.none** - exclude momentum balances,
    **MomentumBalanceType.pressureTotal** - single pressure balance for material,
    **MomentumBalanceType.pressurePhase** - pressure balances for each phase,
    **MomentumBalanceType.momentumTotal** - single momentum balance for material,
    **MomentumBalanceType.momentumPhase** - momentum balances for each phase.}""",
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

    CONFIG.declare(
        "chemical_mapping_data",
        ConfigValue(
            default={},
            domain=dict,
            description="Dictionary of chemical species names and their mapping",
            doc="""
        Dictionary of chemical species names from the property
        package and how they map to specific species needed for solving a simple
        boron speciation problem in an equilibrium reactor. This dictionary must
        have the following format [Required]::

            {'boron_name': 'name_of_species_representing_boron', #[is required]
            'borate_name': 'name_of_species_representing_borate', #[is required]
            'proton_name': 'name_of_species_representing_protons',  #[is optional]
            'hydroxide_name': 'name_of_species_representing_hydroxides', #[is optional]
            'caustic_additive':
                {
                    'additive_name': 'name_of_the_actual_chemical', #[is optional]
                    'cation_name': 'name_of_cation_species_in_additive', #[is optional]
                    'mw_additive': (value, units), #[is required]
                    'charge_additive': value, #[is required]
                },
            }

            """,
        ),
    )

    def build(self):
        # build always starts by calling super().build()
        # This triggers a lot of boilerplate in the background for you
        super().build()

        # this creates blank scaling factors, which are populated later
        self.scaling_factor = Suffix(direction=Suffix.EXPORT)

        # Next, get the base units of measurement from the property definition
        units_meta = self.config.property_package.get_metadata().get_derived_units

        # Check configs for errors
        common_msg = (
            "The 'chemical_mapping_data' dict MUST contain a dict of names that map \n"
            + "to each chemical name in the property package for boron and borate. \n"
            + "Optionally, user may provide names that also map to protons and hydroxide, \n"
            + "as well as to the cation from the caustic additive. The 'caustic_additive' \n"
            + "must be a dict that contains molecular weight and charge.\n\n"
            + "Example:\n"
            + "-------\n"
            + "{'boron_name': 'B[OH]3',    #[is required]\n"
            + " 'borate_name': 'B[OH]4_-', #[is required]\n"
            + " 'proton_name': 'H_+',      #[is OPTIONAL]\n"
            + " 'hydroxide_name': 'OH_-',  #[is OPTIONAL]\n"
            + " 'caustic_additive': \n"
            + "     {'additive_name': 'NaOH',                    #[is OPTIONAL]\n"
            + "      'cation_name': 'Na_+',                      #[is OPTIONAL]\n"
            + "      'mw_additive': (40, pyunits.g/pyunits.mol), #[is required]\n"
            + "      'charge_additive': 1,                       #[is required]\n"
            + "     }, \n"
            + "}\n\n"
        )
        if type(self.config.chemical_mapping_data) != dict or self.config.chemical_mapping_data=={}:
            raise ConfigurationError(
                "\n\n Did not provide a 'dict' for 'chemical_mapping_data' \n" + common_msg
            )
        if ('boron_name' not in self.config.chemical_mapping_data or
            'borate_name' not in self.config.chemical_mapping_data or
            'caustic_additive' not in self.config.chemical_mapping_data):
            raise ConfigurationError(
                "\n\n Missing some required information in 'chemical_mapping_data' \n" + common_msg
            )
        if ('mw_additive' not in self.config.chemical_mapping_data['caustic_additive'] or
            'charge_additive' not in self.config.chemical_mapping_data['caustic_additive']):
            raise ConfigurationError(
                "\n\n Missing some required information in 'chemical_mapping_data' \n" + common_msg
            )
        if (
            type(self.config.chemical_mapping_data['caustic_additive']['mw_additive'])
            != tuple
        ):
            raise ConfigurationError(
                "\n Did not provide a tuple for 'mw_additive' \n" + common_msg
            )

        # Assign name IDs locally for reference later when building constraints
        self.boron_name_id = self.config.chemical_mapping_data['boron_name']
        self.borate_name_id = self.config.chemical_mapping_data['borate_name']
        if 'proton_name' in self.config.chemical_mapping_data:
            self.proton_name_id = self.config.chemical_mapping_data['proton_name']
        else:
            self.proton_name_id = None
        if 'hydroxide_name' in self.config.chemical_mapping_data:
            self.hydroxide_name_id = self.config.chemical_mapping_data['hydroxide_name']
        else:
            self.hydroxide_name_id = None
        if 'cation_name' in self.config.chemical_mapping_data['caustic_additive']:
            self.cation_name_id = self.config.chemical_mapping_data['caustic_additive']['cation_name']
        else:
            self.cation_name_id = None
        if 'additive_name' in self.config.chemical_mapping_data['caustic_additive']:
            self.caustic_chem_name = self.config.chemical_mapping_data['caustic_additive']['additive_name']
        else:
            self.caustic_chem_name = None

        # Cross reference and check given names with set of valid names
        if self.boron_name_id not in self.config.property_package.component_list:
            raise ConfigurationError(
                "\n Given 'boron_name' {" + self.boron_name_id + "} does not match " +
                "any species name from the property package \n{}".format(
                    [c for c in self.config.property_package.component_list])
            )
        if self.borate_name_id not in self.config.property_package.component_list:
            raise ConfigurationError(
                "\n Given 'borate_name' {" + self.borate_name_id + "} does not match " +
                "any species name from the property package \n{}".format(
                    [c for c in self.config.property_package.component_list])
            )
        if self.proton_name_id != None:
            if self.proton_name_id not in self.config.property_package.component_list:
                raise ConfigurationError(
                    "\n Given 'proton_name' {" + self.proton_name_id + "} does not match " +
                    "any species name from the property package \n{}".format(
                        [c for c in self.config.property_package.component_list])
                )
        if self.hydroxide_name_id != None:
            if self.hydroxide_name_id not in self.config.property_package.component_list:
                raise ConfigurationError(
                    "\n Given 'hydroxide_name' {" + self.hydroxide_name_id + "} does not match " +
                    "any species name from the property package \n{}".format(
                        [c for c in self.config.property_package.component_list])
                )
        if self.cation_name_id != None:
            if self.cation_name_id not in self.config.property_package.component_list:
                raise ConfigurationError(
                    "\n Given 'cation_name' {" + self.cation_name_id + "} does not match " +
                    "any species name from the property package \n{}".format(
                        [c for c in self.config.property_package.component_list])
                )

        # check for existence of inherent reactions
        #   This is to ensure that no degeneracy could be introduced
        #   in the system of equations (may not need this explicit check)
        if hasattr(self.config.property_package, 'inherent_reaction_idx'):
            raise ConfigurationError(
                "\n Property Package CANNOT contain 'inherent_reactions' \n"
            )

        # cation set reference
        cation_set = self.config.property_package.cation_set

        # anion set reference
        anion_set = self.config.property_package.anion_set

        # Add param to store all charges of ions for convenience
        self.ion_charge = Param(
            anion_set | cation_set,
            initialize=1,
            mutable=True,
            units=pyunits.dimensionless,
            doc="Ion charge",
        )

        # Loop through full set and try to assign charge
        for j in self.config.property_package.component_list:
            if j in anion_set or j in cation_set:
                self.ion_charge[j] = self.config.property_package.get_component(
                    j
                ).config.charge

        # Add unit variables and parameters
        mw_add = pyunits.convert_value(
            self.config.chemical_mapping_data['caustic_additive']['mw_additive'][0],
            from_units=self.config.chemical_mapping_data['caustic_additive']['mw_additive'][1],
            to_units=pyunits.mg / pyunits.mol,
        )
        self.caustic_mw = Param(
            mutable=True,
            initialize=mw_add,
            domain=NonNegativeReals,
            units=pyunits.mg / pyunits.mol,
            doc="Molecular weight of the caustic additive",
        )
        self.caustic_cation_charge = Param(
            mutable=True,
            initialize=self.config.chemical_mapping_data['caustic_additive']['charge_additive'],
            domain=NonNegativeReals,
            units=pyunits.dimensionless,
            doc="Charge of the caustic additive",
        )
        self.caustic_dose = Var(
            self.flowsheet().config.time,
            initialize=0,
            bounds=(0, None),
            domain=NonNegativeReals,
            units=pyunits.mg / pyunits.L,
            doc="Dosages of the set of caustic additive",
        )

        # Reaction parameters
        self.Kw_0 = Param(
            mutable=True,
            initialize=60.91,
            domain=NonNegativeReals,
            units=pyunits.mol**2 / pyunits.m**6,
            doc="Water dissociation pre-exponential constant",
        )
        self.dH_w = Param(
            mutable=True,
            initialize=55830,
            domain=NonNegativeReals,
            units=pyunits.J / pyunits.mol,
            doc="Water dissociation enthalpy",
        )
        self.Ka_0 = Param(
            mutable=True,
            initialize=0.000163,
            domain=NonNegativeReals,
            units=pyunits.mol / pyunits.m**3,
            doc="Boron dissociation pre-exponential constant",
        )
        self.dH_a = Param(
            mutable=True,
            initialize=13830,
            domain=NonNegativeReals,
            units=pyunits.J / pyunits.mol,
            doc="Boron dissociation enthalpy",
        )

        # molarity vars (for approximate boron speciation)
        #       Used to establish the mass transfer rates by
        #       first solving a coupled equilibrium system
        #
        #   ENE: [H+] = [OH-] + [A-] + (Alk - n*[base])
        #   MB:  TB = [HA] + [A-]
        #   rw:  Kw = [H+][OH-]
        #   ra:  Ka[HA] = [H+][A-]
        #
        #       Alk = sum(n*Anions) - sum(n*Cations) (from props)
        #       [base] = (Dose/MW)
        #       TB = [HA]_inlet + [A-]_inlet (from props)

        self.mol_H = Var(
            self.flowsheet().config.time,
            initialize=1e-4,
            bounds=(0, None),
            domain=NonNegativeReals,
            units=pyunits.mol / pyunits.m**3,
            doc="Resulting molarity of protons",
        )
        self.mol_OH = Var(
            self.flowsheet().config.time,
            initialize=1e-4,
            bounds=(0, None),
            domain=NonNegativeReals,
            units=pyunits.mol / pyunits.m**3,
            doc="Resulting molarity of hydroxide",
        )
        self.mol_Boron = Var(
            self.flowsheet().config.time,
            initialize=1e-2,
            bounds=(0, None),
            domain=NonNegativeReals,
            units=pyunits.mol / pyunits.m**3,
            doc="Resulting molarity of Boron",
        )
        self.mol_Borate = Var(
            self.flowsheet().config.time,
            initialize=1e-2,
            bounds=(0, None),
            domain=NonNegativeReals,
            units=pyunits.mol / pyunits.m**3,
            doc="Resulting molarity of Borate",
        )

        # Build control volume for feed side
        self.control_volume = ControlVolume0DBlock(
            default={
                "dynamic": False,
                "has_holdup": False,
                "property_package": self.config.property_package,
                "property_package_args": self.config.property_package_args,
                "reaction_package": None,
                "reaction_package_args": None,
            }
        )

        self.control_volume.add_state_blocks(has_phase_equilibrium=False)

        self.control_volume.add_material_balances(
            balance_type=self.config.material_balance_type,
            has_mass_transfer=True,
            has_rate_reactions=False,
            has_equilibrium_reactions=False,
        )

        # NOTE: This checks for if an energy_balance_type is defined
        if hasattr(self.config, "energy_balance_type"):
            self.control_volume.add_energy_balances(
                balance_type=self.config.energy_balance_type,
                has_enthalpy_transfer=False,
            )

        self.control_volume.add_momentum_balances(
            balance_type=self.config.momentum_balance_type, has_pressure_change=False
        )

        # Add ports
        self.add_inlet_port(name="inlet", block=self.control_volume)
        self.add_outlet_port(name="outlet", block=self.control_volume)

        # -------- Add constraints ---------
        # Adds isothermal constraint if no energy balance present
        if not hasattr(self.config, "energy_balance_type"):

            @self.Constraint(self.flowsheet().config.time, doc="Isothermal condition")
            def eq_isothermal(self, t):
                return (
                    self.control_volume.properties_out[t].temperature
                    == self.control_volume.properties_in[t].temperature
                )

        # Constraints for mass transfer terms
        @self.Constraint(
            self.flowsheet().config.time,
            doc="Electroneutrality condition",
        )
        def eq_electroneutrality(self, t):
            Alk = 0
            for j in self.ion_charge:
                conc = self.control_volume.properties_out[t].conc_mol_phase_comp["Liq", j]
                if (j == self.boron_name_id or j == self.borate_name_id
                    or j == self.proton_name_id or j == self.hydroxide_name_id):
                    Alk += 0.0
                else:
                    Alk += -self.ion_charge[j]*conc
            mol_H = pyunits.convert(self.mol_H[t],
                to_units=units_meta("amount") * units_meta("length") ** -3,
            )
            mol_OH = pyunits.convert(self.mol_OH[t],
                to_units=units_meta("amount") * units_meta("length") ** -3,
            )
            mol_Borate = pyunits.convert(self.mol_Borate[t],
                to_units=units_meta("amount") * units_meta("length") ** -3,
            )
            mol_Base = pyunits.convert(self.caustic_cation_charge*self.caustic_dose[t]/self.caustic_mw,
                to_units=units_meta("amount") * units_meta("length") ** -3,
            )
            return mol_H == mol_OH + mol_Borate + Alk - mol_Base


        @self.Constraint(
            self.flowsheet().config.time,
            doc="Total boron balance",
        )
        def eq_total_boron(self, t):
            inlet_Boron = self.control_volume.properties_in[t].conc_mol_phase_comp["Liq", self.boron_name_id]
            inlet_Borate = self.control_volume.properties_in[t].conc_mol_phase_comp["Liq", self.borate_name_id]
            mol_Borate = pyunits.convert(self.mol_Borate[t],
                to_units=units_meta("amount") * units_meta("length") ** -3,
            )
            mol_Boron = pyunits.convert(self.mol_Boron[t],
                to_units=units_meta("amount") * units_meta("length") ** -3,
            )
            return inlet_Boron + inlet_Borate == mol_Borate + mol_Boron


        @self.Constraint(
            self.flowsheet().config.time,
            doc="Water dissociation",
        )
        def eq_water_dissociation(self, t):
            return (self.Kw_0 * exp(-self.dH_w/Constants.gas_constant/self.control_volume.properties_out[t].temperature)) == self.mol_H[t] * self.mol_OH[t]


        @self.Constraint(
            self.flowsheet().config.time,
            doc="Boron dissociation",
        )
        def eq_boron_dissociation(self, t):
            return (self.Ka_0 * exp(-self.dH_a/Constants.gas_constant/self.control_volume.properties_out[t].temperature)) * self.mol_Boron[t] == self.mol_H[t] * self.mol_Borate[t]

        # Add constraints for mass transfer terms
        @self.Constraint(
            self.flowsheet().config.time,
            self.config.property_package.phase_list,
            self.config.property_package.component_list,
            doc="Mass transfer term",
        )
        def eq_mass_transfer_term(self, t, p, j):
            if j == self.boron_name_id:
                boron_out = pyunits.convert(self.mol_Boron[t],
                    to_units=units_meta("amount") * units_meta("length") ** -3,
                )
                input_rate = self.control_volume.properties_in[t].flow_mol_phase_comp[p, self.boron_name_id]
                exit_rate = (
                    self.control_volume.properties_out[t].flow_vol_phase[p] * boron_out
                )

                loss_rate = input_rate - exit_rate
                return self.control_volume.mass_transfer_term[t, p, j] == -loss_rate
            elif j == self.borate_name_id:
                borate_out = pyunits.convert(self.mol_Borate[t],
                    to_units=units_meta("amount") * units_meta("length") ** -3,
                )
                input_rate = self.control_volume.properties_in[t].flow_mol_phase_comp[p, self.borate_name_id]
                exit_rate = (
                    self.control_volume.properties_out[t].flow_vol_phase[p] * borate_out
                )

                loss_rate = input_rate - exit_rate
                return self.control_volume.mass_transfer_term[t, p, j] == -loss_rate
            elif j == self.proton_name_id:
                p_out = pyunits.convert(self.mol_H[t],
                    to_units=units_meta("amount") * units_meta("length") ** -3,
                )
                input_rate = self.control_volume.properties_in[t].flow_mol_phase_comp[p, self.proton_name_id]
                exit_rate = (
                    self.control_volume.properties_out[t].flow_vol_phase[p] * p_out
                )

                loss_rate = input_rate - exit_rate
                return self.control_volume.mass_transfer_term[t, p, j] == -loss_rate
            elif j == self.hydroxide_name_id:
                h_out = pyunits.convert(self.mol_OH[t],
                    to_units=units_meta("amount") * units_meta("length") ** -3,
                )
                input_rate = self.control_volume.properties_in[t].flow_mol_phase_comp[p, self.hydroxide_name_id]
                exit_rate = (
                    self.control_volume.properties_out[t].flow_vol_phase[p] * h_out
                )

                loss_rate = input_rate - exit_rate
                return self.control_volume.mass_transfer_term[t, p, j] == -loss_rate
            elif j == self.cation_name_id:
                c_out = pyunits.convert(self.caustic_cation_charge*self.caustic_dose[t]/self.caustic_mw,
                    to_units=units_meta("amount") * units_meta("length") ** -3,
                )
                exit_rate = (
                    self.control_volume.properties_out[t].flow_vol_phase[p] * c_out
                )
                return self.control_volume.mass_transfer_term[t, p, j] == exit_rate
            else:
                return self.control_volume.mass_transfer_term[t, p, j] == 0.0

    # initialize method
    def initialize_build(
        blk, state_args=None, outlvl=idaeslog.NOTSET, solver=None, optarg=None
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
        init_log = idaeslog.getInitLogger(blk.name, outlvl, tag="unit")
        solve_log = idaeslog.getSolveLogger(blk.name, outlvl, tag="unit")
        # Set solver options
        opt = get_solver(solver, optarg)

        # ---------------------------------------------------------------------
        # Initialize holdup block
        flags = blk.control_volume.initialize(
            outlvl=outlvl,
            optarg=optarg,
            solver=solver,
            state_args=state_args,
        )
        init_log.info_high("Initialization Step 1 Complete.")

        # Apply guess for unit model vars
        for t in blk.flowsheet().config.time:
            # Naive guess (pH = 7)
            blk.mol_H[t].set_value(1e-4)
            blk.mol_OH[t].set_value(10**-14*1000/blk.mol_H[t].value)
            TB = value(blk.control_volume.properties_in[t].conc_mol_phase_comp["Liq",blk.boron_name_id]) + value(blk.control_volume.properties_in[t].conc_mol_phase_comp["Liq",blk.borate_name_id])
            Ratio = 10**-9.21*1000/blk.mol_H[t].value
            blk.mol_Boron[t].set_value(TB/(1+Ratio))
            blk.mol_Borate[t].set_value(TB*Ratio/(1+Ratio))
        # ---------------------------------------------------------------------

        # ---------------------------------------------------------------------
        # Solve unit
        with idaeslog.solver_log(solve_log, idaeslog.DEBUG) as slc:
            res = opt.solve(blk, tee=slc.tee)
        init_log.info_high("Initialization Step 2 {}.".format(idaeslog.condition(res)))

        # ---------------------------------------------------------------------
        # Release Inlet state
        blk.control_volume.release_state(flags, outlvl + 1)
        init_log.info("Initialization Complete: {}".format(idaeslog.condition(res)))

        # Rescale internal variables
        for t in blk.flowsheet().config.time:
            iscale.set_scaling_factor(blk.mol_OH[t], 100/blk.mol_OH[t].value)
            iscale.set_scaling_factor(blk.mol_H[t], 100/blk.mol_H[t].value)
            iscale.set_scaling_factor(blk.mol_Boron[t], 100/blk.mol_Boron[t].value)
            iscale.set_scaling_factor(blk.mol_Borate[t], 100/blk.mol_Borate[t].value)

    def outlet_pH(self, time=0):
        return -log10(value(self.mol_H[time])/1000)

    def outlet_pOH(self, time=0):
        return -log10(value(self.mol_OH[time])/1000)

    def propogate_initial_state(self):
        units_meta = self.config.property_package.get_metadata().get_derived_units
        i = 0
        for t in self.control_volume.properties_in:

            # Should check 'define_state_vars' to see if user has provided
            #   state vars that are outside of the checks in this function
            if (
                "flow_mol_phase_comp"
                not in self.control_volume.properties_in[t].define_state_vars()
                and "flow_mass_phase_comp"
                not in self.control_volume.properties_in[t].define_state_vars()
            ):
                raise ConfigurationError(
                    "BoronRemoval unit model requires "
                    "either a 'flow_mol_phase_comp' or 'flow_mass_phase_comp' "
                    "state variable basis to apply the 'propogate_initial_state' method"
                )

            if i == 0:
                t0 = t

            if "pressure" in self.control_volume.properties_in[t].define_state_vars():
                self.control_volume.properties_out[t].pressure = value(
                    self.control_volume.properties_in[t0].pressure
                )

            if "temperature" in self.control_volume.properties_in[t].define_state_vars():
                self.control_volume.properties_out[t].temperature = value(
                    self.control_volume.properties_in[t0].temperature
                )

            # If flow_mol_phase_comp is state var
            if (
                "flow_mol_phase_comp"
                in self.control_volume.properties_in[t].define_state_vars()
            ):
                for ind in self.control_volume.properties_in[t].flow_mol_phase_comp:
                    self.control_volume.properties_out[t].flow_mol_phase_comp[ind] = value(
                        self.control_volume.properties_in[t0].flow_mol_phase_comp[ind]
                    )

                # Check to see if 'flow_mass_phase_comp' is constructed
                if self.control_volume.properties_out[t].is_property_constructed(
                    "flow_mass_phase_comp"
                ):
                    for ind in self.control_volume.properties_in[t].flow_mass_phase_comp:
                        self.control_volume.properties_out[t].flow_mass_phase_comp[
                            ind
                        ] = value(
                            self.control_volume.properties_in[t0].flow_mol_phase_comp[
                                ind
                            ] * self.control_volume.properties_in[t0].mw_comp[ind[1]]
                        )

                        self.control_volume.properties_in[t].flow_mass_phase_comp[
                            ind
                        ] = value(
                            self.control_volume.properties_in[t0].flow_mol_phase_comp[
                                ind
                            ] * self.control_volume.properties_in[t0].mw_comp[ind[1]]
                        )

                # Check to see if 'mole_frac_phase_comp' is constructed
                if self.control_volume.properties_out[t].is_property_constructed(
                    "mole_frac_phase_comp"
                ):
                    for ind in self.control_volume.properties_in[t].mole_frac_phase_comp:
                        self.control_volume.properties_out[t].mole_frac_phase_comp[
                            ind
                        ] = value(
                            self.control_volume.properties_in[t0].flow_mol_phase_comp[
                                ind
                            ] /
                            sum(self.control_volume.properties_in[t0].flow_mol_phase_comp[ind[0],j]
                                for j in self.config.property_package.component_list)
                        )

                    for ind in self.control_volume.properties_in[t].mole_frac_phase_comp:
                        self.control_volume.properties_in[t].mole_frac_phase_comp[
                            ind
                        ] = value(
                            self.control_volume.properties_in[t0].flow_mol_phase_comp[
                                ind
                            ] /
                            sum(self.control_volume.properties_in[t0].flow_mol_phase_comp[ind[0],j]
                                for j in self.config.property_package.component_list)
                        )

                # Check to see if 'mass_frac_phase_comp' is constructed
                if self.control_volume.properties_out[t].is_property_constructed(
                    "mass_frac_phase_comp"
                ):
                    for ind in self.control_volume.properties_in[t].mass_frac_phase_comp:
                        self.control_volume.properties_out[t].mass_frac_phase_comp[
                            ind
                        ] = value(
                            self.control_volume.properties_in[t0].flow_mol_phase_comp[
                                ind
                            ] * self.control_volume.properties_in[t0].mw_comp[ind[1]] /
                            sum(self.control_volume.properties_in[t0].flow_mol_phase_comp[ind[0],j] *
                                self.control_volume.properties_in[t0].mw_comp[j]
                                for j in self.config.property_package.component_list)
                        )

                    for ind in self.control_volume.properties_in[t].mass_frac_phase_comp:
                        self.control_volume.properties_in[t].mass_frac_phase_comp[
                            ind
                        ] = value(
                            self.control_volume.properties_in[t0].flow_mol_phase_comp[
                                ind
                            ] * self.control_volume.properties_in[t0].mw_comp[ind[1]] /
                            sum(self.control_volume.properties_in[t0].flow_mol_phase_comp[ind[0],j] *
                                self.control_volume.properties_in[t0].mw_comp[j]
                                for j in self.config.property_package.component_list)
                        )

                # Check to see if 'conc_mol_phase_comp' is constructed
                if self.control_volume.properties_out[t].is_property_constructed(
                    "conc_mol_phase_comp"
                ):
                    approx_dens = pyunits.convert(55000 * pyunits.mol / pyunits.m**3,
                        to_units=units_meta("amount") * units_meta("length") ** -3,
                    )
                    for ind in self.control_volume.properties_in[t].conc_mol_phase_comp:
                        self.control_volume.properties_out[t].conc_mol_phase_comp[
                            ind
                        ] = approx_dens * value(
                            self.control_volume.properties_in[t0].flow_mol_phase_comp[
                                ind
                            ] /
                            sum(self.control_volume.properties_in[t0].flow_mol_phase_comp[ind[0],j]
                                for j in self.config.property_package.component_list)
                        )

                    for ind in self.control_volume.properties_in[t].conc_mol_phase_comp:
                        self.control_volume.properties_in[t].conc_mol_phase_comp[
                            ind
                        ] = approx_dens * value(
                            self.control_volume.properties_in[t0].flow_mol_phase_comp[
                                ind
                            ] /
                            sum(self.control_volume.properties_in[t0].flow_mol_phase_comp[ind[0],j]
                                for j in self.config.property_package.component_list)
                        )

                # Check to see if 'conc_mass_phase_comp' is constructed
                if self.control_volume.properties_out[t].is_property_constructed(
                    "conc_mass_phase_comp"
                ):
                    approx_dens = pyunits.convert(1000 * pyunits.kg / pyunits.m**3,
                        to_units=units_meta("mass") * units_meta("length") ** -3,
                    )
                    for ind in self.control_volume.properties_in[t].conc_mass_phase_comp:
                        self.control_volume.properties_out[t].conc_mass_phase_comp[
                            ind
                        ] = approx_dens * value(
                            self.control_volume.properties_in[t0].flow_mol_phase_comp[
                                ind
                            ] * self.control_volume.properties_in[t0].mw_comp[ind[1]] /
                            sum(self.control_volume.properties_in[t0].flow_mol_phase_comp[ind[0],j] *
                                self.control_volume.properties_in[t0].mw_comp[j]
                                for j in self.config.property_package.component_list)
                        )

                    for ind in self.control_volume.properties_in[t].conc_mass_phase_comp:
                        self.control_volume.properties_in[t].conc_mass_phase_comp[
                            ind
                        ] = approx_dens * value(
                            self.control_volume.properties_in[t0].flow_mol_phase_comp[
                                ind
                            ] * self.control_volume.properties_in[t0].mw_comp[ind[1]] /
                            sum(self.control_volume.properties_in[t0].flow_mol_phase_comp[ind[0],j] *
                                self.control_volume.properties_in[t0].mw_comp[j]
                                for j in self.config.property_package.component_list)
                        )

            i += 1
        #End Loop

    def calculate_scaling_factors(self):
        super().calculate_scaling_factors()

        units_meta = self.config.property_package.get_metadata().get_derived_units

        # Provide some intial guess values before scaling
        self.propogate_initial_state()

        # Add scaling for unit model vars (with user input)
        if iscale.get_scaling_factor(self.caustic_dose) is None:
            sf = iscale.get_scaling_factor(self.caustic_dose, default=1, warning=True)
            iscale.set_scaling_factor(self.caustic_dose, sf)

        # Add scaling for unit model vars (without user input)
        if iscale.get_scaling_factor(self.mol_Boron) is None:
            sf = iscale.get_scaling_factor(self.control_volume.properties_in[0].conc_mol_phase_comp["Liq", self.boron_name_id],
                default=1, warning=False)
            iscale.set_scaling_factor(self.mol_Boron, sf/10)

        if iscale.get_scaling_factor(self.mol_Borate) is None:
            sf = iscale.get_scaling_factor(self.control_volume.properties_in[0].conc_mol_phase_comp["Liq", self.borate_name_id],
                default=1, warning=False)
            iscale.set_scaling_factor(self.mol_Borate, sf/10)

        # Scaling for H and OH
        if iscale.get_scaling_factor(self.mol_H) is None:
            if self.proton_name_id in self.config.property_package.component_list:
                sf = iscale.get_scaling_factor(self.control_volume.properties_in[0].conc_mol_phase_comp["Liq", self.proton_name_id],
                    default=1, warning=False)
            else:
                sf = 10
            iscale.set_scaling_factor(self.mol_H, sf)

        if iscale.get_scaling_factor(self.mol_OH) is None:
            if self.hydroxide_name_id in self.config.property_package.component_list:
                sf = iscale.get_scaling_factor(self.control_volume.properties_in[0].conc_mol_phase_comp["Liq", self.hydroxide_name_id],
                    default=1, warning=False)
            else:
                sf = 10
            iscale.set_scaling_factor(self.mol_OH, sf)

        # Scale isothermal condition
        sf = iscale.get_scaling_factor(
            self.control_volume.properties_in[0].temperature
        )
        for t in self.control_volume.properties_in:
            iscale.constraint_scaling_transform(
                    self.eq_isothermal[t], sf
                )

        # Scaling for water dissociation and boron dissociation
        for t in self.control_volume.properties_in:
            sf = iscale.get_scaling_factor(self.mol_H) * iscale.get_scaling_factor(self.mol_OH)
            iscale.constraint_scaling_transform(self.eq_water_dissociation[t], sf)
        for t in self.control_volume.properties_in:
            sf = iscale.get_scaling_factor(self.mol_Borate) * iscale.get_scaling_factor(self.mol_H)
            iscale.constraint_scaling_transform(self.eq_boron_dissociation[t], sf)

        # Scaling for total boron
        for t in self.control_volume.properties_in:
            sf = iscale.get_scaling_factor(self.mol_Boron)
            iscale.constraint_scaling_transform(self.eq_total_boron[t], sf)

        # Scaling for electroneutrality
        for t in self.control_volume.properties_in:
            sf = iscale.get_scaling_factor(self.mol_H) + iscale.get_scaling_factor(self.mol_Borate)
            iscale.constraint_scaling_transform(self.eq_electroneutrality[t], sf)

        # Scaling for mass_transfer_term
        for t in self.control_volume.properties_in:
            for j in self.config.property_package.component_list:
                sf = iscale.get_scaling_factor(self.mol_Borate)
                iscale.constraint_scaling_transform(self.eq_mass_transfer_term[t, "Liq", j], sf)
