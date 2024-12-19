from pyomo.environ import (
    ConcreteModel,
    Var,
    Param,
    Objective,
    Constraint,
    Block,
    units as pyunits,
)
import pyomo.environ as pyo
from pyomo.util.calc_var_value import calculate_variable_from_constraint
from pyomo.common.modeling import unique_component_name
import idaes.core.util.scaling as iscale
from idaes.core.util.model_statistics import degrees_of_freedom
from watertap.core.solvers import get_solver
import reaktoro
from reaktoro_pse.reaktoro_block import ReaktoroBlock


def build_model(m, input_ions, input_dict, ion_wts, output_dict, pH_feed=7):
    # TO-DOs:
    # (1) Translation dict must either be supplied by user or build automatically in some way
    # (2) Vapor pressure - check actual variable used in constraint!

    m.rkt_block = Block()
    mm = m.rkt_block
    database = reaktoro.PhreeqcDatabase("pitzer.dat")
    input_ions = input_ions + ["H2O"]  # MCAS does not have H2O on list

    translation_dict = {
        "H2O": "H2O",
        "Mg_2+": "Mg+2",
        "Na_+": "Na+",
        "Cl_-": "Cl-",
        "SO4_2-": "SO4-2",
        "Ca_2+": "Ca+2",
        "HCO3_-": "HCO3-",
        "K_+": "K+",
    }

    # Define pH
    mm.feed_pH = Var(initialize=pH_feed, bounds=(4, 12), units=pyunits.dimensionless)
    mm.feed_pH.fix()

    # Define list of solids
    mm.output_solid_list = [
        key for key, inner_dict in output_dict.items() if inner_dict["Solid"] is True
    ]

    # Define molar feed
    mm.feed_composition_molar = Var(
        input_ions, initialize=1, units=pyunits.mol / pyunits.s
    )
    mm.inlet_temperature = Var(
        initialize=pyo.value(m.inlet.temperature[0]),
        units=pyunits.get_units(m.inlet.temperature[0]),
    )
    mm.inlet_temperature.fix()
    mm.inlet_pressure = Var(
        initialize=pyo.value(m.inlet.pressure[0]),
        units=pyunits.get_units(m.inlet.pressure[0]),
    )
    mm.inlet_pressure.fix()

    # Create molecular weight variable
    mm.mol_wt_ions = Param(input_ions, units=pyunits.g / pyunits.mol)
    for i in input_ions:
        mm.mol_wt_ions[i].set_value(ion_wts[i])

    @mm.Constraint(input_ions)
    def eq_molar_flows(b, key):
        return b.feed_composition_molar[key] * b.mol_wt_ions[key] == pyunits.convert(
            m.inlet.flow_mass_phase_comp[0, "Liq", key], to_units=pyunits.g / pyunits.s
        )

    # Add a variable for the vapor phase
    mm.vapor_molar_flow = Var(["H2O"], initialize=1, units=pyunits.mol / pyunits.s)

    @mm.Constraint()
    def eq_vapor_flow(b):
        return b.vapor_molar_flow["H2O"] == b.feed_composition_molar["H2O"] * (
            input_dict["Evaporation percent"]["flowsheet_var"] / 100
        )

    # Reaktoro feed block computations
    mm.feed_properties = Var(
        [("molarEnthalpy", None), ("specificHeatCapacityConstP", None)], initialize=1
    )
    mm.eq_feed_properties = ReaktoroBlock(
        system_state={
            "temperature": mm.inlet_temperature,
            "pressure": mm.inlet_pressure,
            "pH": mm.feed_pH,
        },
        aqueous_phase={
            "composition": mm.feed_composition_molar,
            "convert_to_rkt_species": True,
            "species_to_rkt_species_dict": translation_dict,
            "activity_model": "ActivityModelPitzer",
        },
        outputs={"speciesAmount": True},
        dissolve_species_in_reaktoro=True,
        database=database,  # need to specify new database to use
    )

    # Modified case block
    chem_modified_liq_prop_list = [
        ("specificHeatCapacityConstP", None),
        ("molarEnthalpy", None),
        ("vaporPressure", "H2O(g)"),
        ("pH", None),
    ]
    for solid in mm.output_solid_list:
        chem_modified_liq_prop_list.append(("speciesAmount", solid))
    mm.chem_modified_liq_properties = Var(chem_modified_liq_prop_list, initialize=1e-5)

    mm.eq_chem_modified_liq_properties = ReaktoroBlock(
        aqueous_phase={
            "composition": mm.eq_feed_properties.outputs,
            "convert_to_rkt_species": False,
            "species_to_rkt_species_dict": translation_dict,
            "activity_model": "ActivityModelPitzer",
        },
        system_state={
            "temperature": input_dict["Temperature"]["flowsheet_var"],
            "pressure": mm.inlet_pressure,
        },
        exact_speciation=True,
        outputs=mm.chem_modified_liq_properties,
        mineral_phase={"phase_components": mm.output_solid_list},
        gas_phase={
            "phase_components": ["H2O(g)", "Ntg(g)"],
            "activity_model": "ActivityModelRedlichKwong",
        },
        database=database,
        chemistry_modifier={
            "H2O_evaporation": mm.vapor_molar_flow["H2O"],
        },
        register_new_chemistry_modifiers={"H2O_evaporation": {"H": -2, "O": -1}},
        dissolve_species_in_reaktoro=True,
        build_speciation_block=True,
        reaktoro_solve_options={"open_species_on_property_block": ["H+", "OH-"]},
        jacobian_options={
            "user_scaling": {
                ("molarEnthalpy", None): 1,
                ("specificHeatCapacityConstP", None): 1,
            },
        },
    )

    # Define solids in moles
    mm.molar_solids = Var(
        mm.output_solid_list, initialize=1, units=pyunits.mol / pyunits.s
    )

    @mm.Constraint(mm.output_solid_list)
    def eq_molar_solids_composition(b, key):
        return (
            b.molar_solids[key]
            == b.chem_modified_liq_properties[("speciesAmount", key)]
        )

    # Create molecular weight variable for solids
    mm.mol_wt_solids = Param(mm.output_solid_list, units=pyunits.g / pyunits.mol)
    for solid in mm.output_solid_list:
        mm.mol_wt_solids[solid].set_value(output_dict[solid]["mw"])

    scale_model(mm, mm.output_solid_list)
    initialize(mm)

    return m


def scale_model(m, solids_list):
    for key in m.feed_composition_molar:
        iscale.set_scaling_factor(
            m.feed_composition_molar[key], 1 / m.feed_composition_molar[key].value
        )
    iscale.set_scaling_factor(m.feed_properties[("molarEnthalpy", None)], 1 / 1e4)
    iscale.set_scaling_factor(
        m.chem_modified_liq_properties[("molarEnthalpy", None)], 1 / 1e4
    )
    iscale.set_scaling_factor(m.inlet_temperature, 1 / 100)
    for solid in solids_list:
        iscale.set_scaling_factor(
            m.chem_modified_liq_properties[("speciesAmount", solid)], 1e5
        )


def initialize(m):
    """prop feed to precipitation comp"""
    for key in m.feed_composition_molar:
        calculate_variable_from_constraint(
            m.feed_composition_molar[key], m.eq_molar_flows[key]
        )
        # calculate_variable_from_constraint(m.sat_liquid_molar[key], m.eq_sat_liquid_composition[key])
    for key in m.molar_solids:
        calculate_variable_from_constraint(
            m.molar_solids[key], m.eq_molar_solids_composition[key]
        )
    calculate_variable_from_constraint(m.vapor_molar_flow["H2O"], m.eq_vapor_flow)
    # """ initialize feed and precipitaiton properties
    m.eq_feed_properties.initialize()
    m.eq_chem_modified_liq_properties.initialize()
    # return m


def add_reaktoro_model(
    model, input_ions, input_dict, mol_wt_ions, solids_list, output_dict, pH_feed
):
    # Check solid keys, before anything
    assert output_dict.keys() == solids_list.keys()

    # build reaktoro model/block
    m = build_model(
        m=model,
        input_ions=input_ions,
        input_dict=input_dict,
        output_dict=output_dict,
        pH_feed=pH_feed,
        ion_wts=mol_wt_ions,
    )

    # Link output to specific crystallizer variables
    @m.Constraint(m.rkt_block.output_solid_list)
    def eq_crystallizer_reaktoro_linking(b, key):
        return m.flow_mass_sol_comp_apparent[key] == pyunits.convert(
            m.rkt_block.molar_solids[key] * m.rkt_block.mol_wt_solids[key],
            to_units=pyunits.kg / pyunits.s,
        )

    # Define pressure
    @m.Constraint()
    def eq_vapor_pressure_constraint(b):
        return (
            m.pressure_operating
            == m.rkt_block.chem_modified_liq_properties[("vaporPressure", "H2O(g)")]
        )
