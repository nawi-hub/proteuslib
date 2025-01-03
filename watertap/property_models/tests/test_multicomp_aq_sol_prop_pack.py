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
import re

import pytest

from pyomo.environ import (
    ConcreteModel,
    assert_optimal_termination,
    Set,
    Param,
    units as pyunits,
    Suffix,
    value,
    Var,
    Constraint,
)
from pyomo.util.check_units import assert_units_consistent, assert_units_equivalent

# Imports from idaes core
from idaes.core.base.components import Solvent, Solute, Cation, Anion
from idaes.core.base.phases import PhaseType as PT
from idaes.core import (
    FlowsheetBlock,
    MaterialFlowBasis,
    MaterialBalanceType,
    AqueousPhase,
    EnergyBalanceType,
)
from idaes.core.util.scaling import calculate_scaling_factors, get_scaling_factor
from idaes.core.util.model_statistics import (
    number_variables,
    number_total_constraints,
    number_unused_variables,
)
from idaes.core.util.scaling import (
    unscaled_variables_generator,
    unscaled_constraints_generator,
    badly_scaled_var_generator,
    set_scaling_factor,
)
from idaes.core.util.exceptions import ConfigurationError
from idaes.core.scaling import set_scaling_factor
from idaes.core.initialization import (
    BlockTriangularizationInitializer,
    InitializationStatus,
)

# Imports from idaes generic models
from idaes.models.properties.modular_properties.pure.ConstantProperties import Constant
from idaes.models.properties.modular_properties.state_definitions import FpcTP
from idaes.models.properties.modular_properties.eos.ideal import Ideal

# Import the idaes objects for Generic Properties
from idaes.models.properties.modular_properties.base.generic_property import (
    GenericParameterBlock,
)

# Import idaes mixer to check compatibility in absence of get_enthalpy_flow_terms()
from idaes.models.unit_models import Mixer
from idaes.models.unit_models.mixer import MixingType
import idaes.logger as idaeslog

from watertap.property_models.multicomp_aq_sol_prop_pack import (
    MCASParameterBlock,
    MCASStateBlock,
    ActivityCoefficientModel,
    DensityCalculation,
    DiffusivityCalculation,
    ElectricalMobilityCalculation,
    EquivalentConductivityCalculation,
    TransportNumberCalculation,
    MCASScaler,
)
from watertap.core.util.initialization import check_dof
from watertap.property_models.tests.property_test_harness import PropertyAttributeError
from watertap.core.solvers import get_solver

solver = get_solver()


# -----------------------------------------------------------------------------
@pytest.mark.unit
def test_default_initializer_scaler():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.properties = MCASParameterBlock(
        solute_list=["A"],
        mw_data={"H2O": 0.018, "A": 0.01},
        charge={"A": 0},
    )

    m.fs.state = m.fs.properties.build_state_block(m.fs.time)

    assert m.fs.state.default_initializer is BlockTriangularizationInitializer
    assert m.fs.state.default_scaler is MCASScaler


@pytest.mark.unit
def test_config():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.properties = MCASParameterBlock(
        solute_list=["A"],
        mw_data={"H2O": 0.018, "A": 0.01},
        charge={"A": 0},
    )

    assert len(m.fs.properties.config) == 18
    config_options = [
        "default_arguments",
        "solute_list",
        "stokes_radius_data",
        "diffusivity_data",
        "molar_volume_data",
        "mw_data",
        "elec_mobility_data",
        "trans_num_data",
        "equiv_conductivity_phase_data",
        "charge",
        "ignore_neutral_charge",
        "activity_coefficient_model",
        "density_calculation",
        "diffus_calculation",
        "elec_mobility_calculation",
        "trans_num_calculation",
        "equiv_conductivity_calculation",
        "material_flow_basis",
    ]
    for i in m.fs.properties.config:
        assert i in config_options
    assert not m.fs.properties.config.ignore_neutral_charge
    assert (
        m.fs.properties.config.activity_coefficient_model
        == ActivityCoefficientModel.ideal
    )
    assert m.fs.properties.config.density_calculation == DensityCalculation.constant
    assert m.fs.properties.config.diffus_calculation == DiffusivityCalculation.none
    assert (
        m.fs.properties.config.elec_mobility_calculation
        == ElectricalMobilityCalculation.none
    )
    assert (
        m.fs.properties.config.trans_num_calculation
        == TransportNumberCalculation.ElectricalMobility
    )
    assert (
        m.fs.properties.config.equiv_conductivity_calculation
        == EquivalentConductivityCalculation.ElectricalMobility
    )
    assert m.fs.properties.config.material_flow_basis == MaterialFlowBasis.molar


@pytest.fixture(scope="module")
def model():
    m = ConcreteModel()

    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.properties = MCASParameterBlock(
        solute_list=["A", "B", "C", "D"],
        diffusivity_data={
            ("Liq", "A"): 1e-09,
            ("Liq", "B"): 1e-10,
            ("Liq", "C"): 1e-07,
            ("Liq", "D"): 1e-11,
        },
        mw_data={"H2O": 0.018, "A": 0.01, "B": 0.025, "C": 0.1, "D": 0.125},
        elec_mobility_data={
            ("Liq", "A"): 5.19e-08,
            ("Liq", "B"): 8.29e-08,
            ("Liq", "C"): 6.17e-08,
            ("Liq", "D"): 7.92e-08,
        },
        stokes_radius_data={"A": 1e-09, "B": 1e-09, "C": 1e-09, "D": 1e-10},
        charge={"A": 1, "B": -2, "C": 2, "D": -1},
    )

    return m


@pytest.mark.unit
def test_parameter_block(model):
    assert isinstance(model.fs.properties.component_list, Set)
    for j in model.fs.properties.component_list:
        assert j in ["H2O", "A", "B", "C", "D"]
    assert isinstance(model.fs.properties.solvent_set, Set)
    for j in model.fs.properties.solvent_set:
        assert j in ["H2O"]
    # solute set and ion set are identical when all components in component_list are ions given hierarchy
    assert isinstance(model.fs.properties.solute_set, Set)
    for j in model.fs.properties.solute_set:
        assert j in ["A", "B", "C", "D"]
    assert isinstance(model.fs.properties.ion_set, Set)
    for j in model.fs.properties.ion_set:
        assert j in ["A", "B", "C", "D"]
    assert isinstance(model.fs.properties.anion_set, Set)
    for j in model.fs.properties.anion_set:
        assert j in ["B", "D"]
    assert isinstance(model.fs.properties.cation_set, Set)
    for j in model.fs.properties.cation_set:
        assert j in ["A", "C"]
    # neutral set is always constructed, even when no neutrals are present in component_list
    assert isinstance(model.fs.properties.neutral_set, Set)
    assert model.fs.properties.neutral_set == []

    assert isinstance(model.fs.properties.phase_list, Set)
    for j in model.fs.properties.phase_list:
        assert j in ["Liq"]

    assert model.fs.properties._state_block_class is MCASStateBlock

    assert isinstance(model.fs.properties.mw_comp, Param)
    assert model.fs.properties.mw_comp["A"].value == 10e-3
    assert model.fs.properties.mw_comp["B"].value == 25e-3
    assert model.fs.properties.mw_comp["C"].value == 100e-3
    assert model.fs.properties.mw_comp["D"].value == 125e-3
    assert model.fs.properties.mw_comp["H2O"].value == 18e-3

    assert isinstance(model.fs.properties.radius_stokes_comp, Param)
    assert model.fs.properties.radius_stokes_comp["A"].value == 1e-9
    assert model.fs.properties.radius_stokes_comp["B"].value == 1e-9
    assert model.fs.properties.radius_stokes_comp["C"].value == 1e-9
    assert model.fs.properties.radius_stokes_comp["D"].value == 1e-10

    assert (
        model.fs.properties.config.activity_coefficient_model
        == ActivityCoefficientModel.ideal
    )
    assert model.fs.properties.config.material_flow_basis == MaterialFlowBasis.molar


@pytest.mark.component
def test_property_ions(model):
    m = model
    m.fs.stream = m.fs.properties.build_state_block([0], defined_state=True)

    m.fs.stream[0].flow_mol_phase_comp["Liq", "A"].fix(0.000407)
    m.fs.stream[0].flow_mol_phase_comp["Liq", "B"].fix(0.010479)
    m.fs.stream[0].flow_mol_phase_comp["Liq", "C"].fix(0.010479)
    m.fs.stream[0].flow_mol_phase_comp["Liq", "D"].fix(0.000407)
    m.fs.stream[0].flow_mol_phase_comp["Liq", "H2O"].fix(0.99046)
    m.fs.stream[0].temperature.fix(298.15)
    m.fs.stream[0].pressure.fix(101325)

    m.fs.stream[0].assert_electroneutrality(defined_state=True)

    m.fs.stream[0].mole_frac_phase_comp
    m.fs.stream[0].flow_mass_phase_comp
    m.fs.stream[0].molality_phase_comp
    m.fs.stream[0].pressure_osm_phase
    m.fs.stream[0].diffus_phase_comp
    m.fs.stream[0].elec_cond_phase
    m.fs.stream[0].dens_mass_phase
    m.fs.stream[0].conc_mol_phase_comp
    m.fs.stream[0].act_coeff_phase_comp
    m.fs.stream[0].trans_num_phase_comp
    m.fs.stream[0].total_hardness
    m.fs.stream[0].total_dissolved_solids

    calculate_scaling_factors(m.fs)

    assert_units_consistent(m)

    check_dof(m, fail_flag=True)

    m.fs.stream.initialize()

    results = solver.solve(m)
    assert_optimal_termination(results)

    assert model.fs.stream[0].diffus_phase_comp["Liq", "A"].value == 1e-9
    assert model.fs.stream[0].diffus_phase_comp["Liq", "B"].value == 1e-10
    assert model.fs.stream[0].diffus_phase_comp["Liq", "C"].value == 1e-7
    assert model.fs.stream[0].diffus_phase_comp["Liq", "D"].value == 1e-11
    assert value(m.fs.stream[0].conc_mass_phase_comp["Liq", "A"]) == pytest.approx(
        2.1206e-1, rel=1e-3
    )

    assert value(m.fs.stream[0].conc_mol_phase_comp["Liq", "A"]) == pytest.approx(
        21.2055, rel=1e-3
    )
    assert value(m.fs.stream[0].molality_phase_comp["Liq", "A"]) == pytest.approx(
        2.2829e-2, rel=1e-3
    )
    assert value(m.fs.stream[0].elec_cond_phase["Liq"]) == pytest.approx(15.5, rel=1e-3)
    assert value(m.fs.stream[0].pressure_osm_phase["Liq"]) == pytest.approx(
        2.812e6, rel=1e-3
    )
    assert value(m.fs.stream[0].dens_mass_phase["Liq"]) == pytest.approx(
        1000.0, rel=1e-3
    )
    assert value(m.fs.stream[0].act_coeff_phase_comp["Liq", "A"]) == 1
    assert value(m.fs.stream[0].trans_num_phase_comp["Liq", "B"]) == pytest.approx(
        0.563, rel=1e-3
    )
    assert isinstance(model.fs.properties.diffus_phase_comp, Var)
    for v in model.fs.properties.diffus_phase_comp.values():
        assert v.fixed
    assert value(m.fs.stream[0].total_hardness) == pytest.approx(
        value(m.fs.stream[0].conc_mol_phase_comp["Liq", "C"] * 100.0869), rel=1e-3
    )
    assert_units_equivalent(m.fs.stream[0].total_hardness, pyunits.mg / pyunits.L)
    assert value(m.fs.stream[0].total_dissolved_solids) == pytest.approx(
        71109.93, rel=1e-3
    )
    assert_units_equivalent(
        m.fs.stream[0].total_dissolved_solids, pyunits.mg / pyunits.L
    )


@pytest.fixture(scope="module")
def model2():
    m = ConcreteModel()

    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.properties = MCASParameterBlock(
        solute_list=["A", "B", "C", "D"],
        charge={"A": 1, "B": -2, "C": 2, "D": -1},
        mw_data={"A": 10e-3, "B": 25e-3, "C": 100e-3, "D": 25e-3},
    )

    return m


@pytest.mark.component
def test_property_ions_2(model2):
    m = model2

    stream = m.fs.stream = m.fs.properties.build_state_block([0], defined_state=True)

    stream[0].flow_mol_phase_comp["Liq", "A"].fix(0.000407)
    stream[0].flow_mol_phase_comp["Liq", "B"].fix(0.010479)
    stream[0].flow_mol_phase_comp["Liq", "C"].fix(0.010479)
    stream[0].flow_mol_phase_comp["Liq", "D"].fix(0.000407)
    stream[0].flow_mol_phase_comp["Liq", "H2O"].fix(0.99046)
    stream[0].temperature.fix(298.15)
    stream[0].pressure.fix(101325)

    stream[0].radius_stokes_comp["A"] = 1e-9
    stream[0].radius_stokes_comp["B"] = 1e-9
    stream[0].radius_stokes_comp["C"] = 1e-9
    stream[0].radius_stokes_comp["D"] = 1e-10

    stream[0].assert_electroneutrality(defined_state=True, tol=1e-8)

    stream[0].mole_frac_phase_comp

    stream[0].flow_mass_phase_comp

    stream[0].molality_phase_comp
    stream[0].pressure_osm_phase
    stream[0].dens_mass_phase
    stream[0].conc_mol_phase_comp
    stream[0].flow_vol
    calculate_scaling_factors(m.fs)

    assert_units_consistent(m)

    check_dof(m, fail_flag=True)

    stream.initialize()

    results = solver.solve(m)
    assert_optimal_termination(results)
    assert value(stream[0].flow_vol_phase["Liq"]) == pytest.approx(1.91524e-5, rel=1e-3)


@pytest.fixture(scope="module")
def model3():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)

    m.fs.properties = MCASParameterBlock(
        solute_list=["Ca_2+", "SO4_2-", "Na_+", "Cl_-", "Mg_2+"],
        diffusivity_data={
            ("Liq", "Ca_2+"): 7.92e-10,
            ("Liq", "SO4_2-"): 1.06e-09,
            ("Liq", "Na_+"): 1.33e-09,
            ("Liq", "Cl_-"): 2.03e-09,
            ("Liq", "Mg_2+"): 7.06e-10,
        },
        elec_mobility_calculation=ElectricalMobilityCalculation.EinsteinRelation,
        charge={"Ca_2+": 2, "SO4_2-": -2, "Na_+": 1, "Cl_-": -1, "Mg_2+": 2},
        mw_data={
            "Ca_2+": 40e-3,
            "SO4_2-": 97e-3,
            "Na_+": 23e-3,
            "Cl_-": 35e-3,
            "Mg_2+": 24e-3,
        },
    )

    m.fs.stream = m.fs.properties.build_state_block([0], defined_state=True)

    return m


@pytest.mark.unit
def test_build(model3):
    m = model3

    # test scaling factor
    assert hasattr(m.fs.stream[0], "scaling_factor")
    assert isinstance(m.fs.stream[0].scaling_factor, Suffix)

    # test state variables
    state_vars_list = ["flow_mol_phase_comp", "temperature", "pressure"]
    state_vars_dict = m.fs.stream[0].define_state_vars()
    assert len(state_vars_dict) == len(state_vars_list)
    for sv in state_vars_list:
        assert sv in state_vars_dict
        assert hasattr(m.fs.stream[0], sv)
        var = getattr(m.fs.stream[0], sv)
        assert isinstance(var, Var)

    metadata = m.fs.properties.get_metadata().properties

    # check that properties are not built if not demanded
    for v in metadata.list_supported_properties():
        if metadata[v.name].method is not None:
            if m.fs.stream[0].is_property_constructed(v.name):
                if (
                    m.fs.properties.config.material_flow_basis
                    == MaterialFlowBasis.molar
                    and v.name == "flow_mol_phase_comp"
                ):
                    continue
                else:
                    raise PropertyAttributeError(
                        "Property {v_name} is an on-demand property, but was found "
                        "on the stateblock without being demanded".format(v_name=v.name)
                    )

    # check that properties are built if demanded
    for v in metadata.list_supported_properties():
        if metadata[v.name].method is not None:
            if not hasattr(m.fs.stream[0], v.name):
                raise PropertyAttributeError(
                    "Property {v_name} is an on-demand property, but was not built "
                    "when demanded".format(v_name=v.name)
                )

    assert m.fs.stream[0].is_property_constructed("act_coeff_phase_comp")

    var_list = [
        "mass_frac_phase_comp",
        "dens_mass_phase",
        "flow_vol_phase",
        "conc_mass_phase_comp",
        "flow_mass_phase_comp",
        "mole_frac_phase_comp",
        "molality_phase_comp",
        "elec_mobility_phase_comp",
        "trans_num_phase_comp",
        "equiv_conductivity_phase",
        "elec_cond_phase",
        "pressure_osm_phase",
        "act_coeff_phase_comp",
        "total_hardness",
        "total_dissolved_solids",
        "enth_mass_phase",
        "pressure_sat",
    ]

    # test on demand constraints
    for v in var_list:
        assert hasattr(m.fs.stream[0], "_" + v)  # check method
        assert hasattr(m.fs.stream[0], "eq_" + v)  # check constraint
        c = getattr(m.fs.stream[0], "eq_" + v)
        assert isinstance(c, Constraint)

    assert number_variables(m) == 126
    assert number_total_constraints(m) == 73
    assert number_unused_variables(m) == 5


@pytest.mark.unit
def test_general_methods(model3):
    m = model3

    assert hasattr(m.fs.stream[0], "get_material_flow_terms")

    assert hasattr(m.fs.stream[0], "default_material_balance_type")
    assert (
        m.fs.stream[0].default_material_balance_type()
        is MaterialBalanceType.componentTotal
    )
    assert m.fs.stream[0].default_energy_balance_type() is EnergyBalanceType.none

    assert hasattr(m.fs.stream[0], "get_material_flow_basis")
    assert m.fs.stream[0].get_material_flow_basis() is MaterialFlowBasis.molar


@pytest.mark.unit
def test_default_scaling(model3):
    m = model3

    assert hasattr(m.fs.properties, "default_scaling_factor")
    default_scaling_var_dict = {
        ("temperature", None): 1e-2,
        ("pressure", None): 1e-4,
        ("dens_mass_phase", "Liq"): 1e-3,
        ("visc_d_phase", "Liq"): 1e3,
        ("diffus_phase_comp", "Liq"): 1e10,
        ("visc_k_phase", "Liq"): 1e6,
        ("enth_mass_phase", "Liq"): 1e-5,
        ("pressure_sat", None): 1e-5,
    }

    assert len(default_scaling_var_dict) == len(m.fs.properties.default_scaling_factor)
    for t, sf in default_scaling_var_dict.items():
        assert t in m.fs.properties.default_scaling_factor.keys()
        assert m.fs.properties.default_scaling_factor[t] == sf


@pytest.mark.unit
def test_scaling(model3):
    m = model3
    metadata = m.fs.properties.get_metadata().properties

    for v in metadata.list_supported_properties():
        getattr(m.fs.stream[0], v.name)

    calculate_scaling_factors(m)

    # check that all variables have scaling factors
    unscaled_var_list = list(unscaled_variables_generator(m))
    [print(i) for i in unscaled_var_list]
    assert len(unscaled_var_list) == 0

    # check that all constraints have been scaled
    unscaled_constraint_list = list(unscaled_constraints_generator(m))
    [print(i) for i in unscaled_constraint_list]

    assert len(unscaled_constraint_list) == 0

    for j in m.fs.properties.config.solute_list:
        assert get_scaling_factor(m.fs.stream[0].mw_comp[j]) is not None
        assert (
            get_scaling_factor(m.fs.stream[0].diffus_phase_comp["Liq", j]) is not None
        )
        assert (
            get_scaling_factor(m.fs.stream[0].act_coeff_phase_comp["Liq", j])
            is not None
        )
    assert get_scaling_factor(m.fs.stream[0].dens_mass_phase["Liq"]) is not None
    assert get_scaling_factor(m.fs.stream[0].visc_d_phase["Liq"]) is not None


@pytest.mark.component
def test_seawater_data():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.properties = MCASParameterBlock(
        solute_list=["Ca_2+", "SO4_2-", "Na_+", "Cl_-", "Mg_2+"],
        diffusivity_data={
            ("Liq", "Ca_2+"): 7.92e-10,
            ("Liq", "SO4_2-"): 1.06e-09,
            ("Liq", "Na_+"): 1.33e-09,
            ("Liq", "Cl_-"): 2.03e-09,
            ("Liq", "Mg_2+"): 7.06e-10,
        },
        mw_data={
            "H2O": 0.018,
            "Na_+": 0.023,
            "Ca_2+": 0.04,
            "Mg_2+": 0.024,
            "Cl_-": 0.035,
            "SO4_2-": 0.096,
        },
        stokes_radius_data={
            "Na_+": 1.84e-10,
            "Ca_2+": 3.09e-10,
            "Mg_2+": 3.47e-10,
            "Cl_-": 1.21e-10,
            "SO4_2-": 2.3e-10,
        },
        charge={"Na_+": 1, "Ca_2+": 2, "Mg_2+": 2, "Cl_-": -1, "SO4_2-": -2},
        elec_mobility_calculation=ElectricalMobilityCalculation.EinsteinRelation,
        density_calculation=DensityCalculation.seawater,
        activity_coefficient_model=ActivityCoefficientModel.davies,
    )

    m.fs.stream = stream = m.fs.properties.build_state_block([0], defined_state=True)

    mass_flow_in = 1 * pyunits.kg / pyunits.s
    feed_mass_frac = {
        "Na_+": 11122e-6,
        "Ca_2+": 382e-6,
        "Mg_2+": 1394e-6,
        "SO4_2-": 2136e-6,
        "Cl_-": 20300e-6,
    }
    for ion, x in feed_mass_frac.items():
        mol_comp_flow = (
            x * pyunits.kg / pyunits.kg * mass_flow_in / stream[0].mw_comp[ion]
        )

        stream[0].flow_mol_phase_comp["Liq", ion].fix(mol_comp_flow)

    H2O_mass_frac = 1 - sum(x for x in feed_mass_frac.values())
    H2O_mol_comp_flow = (
        H2O_mass_frac
        * pyunits.kg
        / pyunits.kg
        * mass_flow_in
        / stream[0].mw_comp["H2O"]
    )

    stream[0].flow_mol_phase_comp["Liq", "H2O"].fix(H2O_mol_comp_flow)
    stream[0].temperature.fix(298.15)
    stream[0].pressure.fix(101325)

    stream[0].assert_electroneutrality(
        defined_state=True, tol=1e-2, adjust_by_ion="Cl_-"
    )

    metadata = m.fs.properties.get_metadata().properties
    for v in metadata.list_supported_properties():
        getattr(stream[0], v.name)

    assert_units_consistent(m)

    check_dof(m, fail_flag=True)

    m.fs.properties.set_default_scaling(
        "flow_mol_phase_comp", 1e-1, index=("Liq", "H2O")
    )
    m.fs.properties.set_default_scaling(
        "flow_mol_phase_comp", 1e1, index=("Liq", "Na_+")
    )
    m.fs.properties.set_default_scaling(
        "flow_mol_phase_comp", 1e1, index=("Liq", "Cl_-")
    )
    m.fs.properties.set_default_scaling(
        "flow_mol_phase_comp", 1e2, index=("Liq", "Ca_2+")
    )
    m.fs.properties.set_default_scaling(
        "flow_mol_phase_comp", 1e2, index=("Liq", "SO4_2-")
    )
    m.fs.properties.set_default_scaling(
        "flow_mol_phase_comp", 1e2, index=("Liq", "Mg_2+")
    )

    calculate_scaling_factors(m)

    stream.initialize()

    # check if any variables are badly scaled
    badly_scaled_var_list = list(
        badly_scaled_var_generator(m, large=100, small=0.01, zero=1e-10)
    )
    [print(i[0], i[1]) for i in badly_scaled_var_list]
    assert len(badly_scaled_var_list) == 0

    results = solver.solve(m)
    assert_optimal_termination(results)

    assert value(stream[0].flow_vol_phase["Liq"]) == pytest.approx(9.767e-4, rel=1e-3)
    assert value(stream[0].flow_mol_phase_comp["Liq", "H2O"]) == pytest.approx(
        53.59256, rel=1e-3
    )
    assert value(stream[0].flow_mol_phase_comp["Liq", "Na_+"]) == pytest.approx(
        0.4836, rel=1e-3
    )
    assert value(stream[0].flow_mol_phase_comp["Liq", "Ca_2+"]) == pytest.approx(
        0.00955, rel=1e-3
    )
    assert value(stream[0].flow_mol_phase_comp["Liq", "Mg_2+"]) == pytest.approx(
        0.05808, rel=1e-3
    )
    assert value(stream[0].flow_mol_phase_comp["Liq", "Cl_-"]) == pytest.approx(
        0.57443, rel=1e-3
    )
    assert value(stream[0].flow_mol_phase_comp["Liq", "SO4_2-"]) == pytest.approx(
        0.02225, rel=1e-3
    )

    assert value(stream[0].dens_mass_phase["Liq"]) == pytest.approx(1023.816, rel=1e-3)
    assert value(stream[0].pressure_osm_phase["Liq"]) == pytest.approx(
        29.132e5, rel=1e-3
    )
    assert value(stream[0].elec_cond_phase["Liq"]) == pytest.approx(8.066, rel=1e-3)
    assert value(stream[0].flow_vol) == pytest.approx(9.767e-4, rel=1e-3)

    assert value(
        sum(
            stream[0].conc_mass_phase_comp["Liq", j]
            for j in m.fs.properties.ion_set | m.fs.properties.solute_set
        )
    ) == pytest.approx(35.9744, rel=1e-3)
    assert value(
        sum(
            stream[0].mass_frac_phase_comp["Liq", j]
            for j in m.fs.properties.ion_set | m.fs.properties.solute_set
        )
    ) == pytest.approx(0.035142, rel=1e-3)
    assert value(
        sum(
            stream[0].mass_frac_phase_comp["Liq", j]
            for j in m.fs.properties.component_list
        )
    ) == pytest.approx(1, rel=1e-3)
    assert value(
        sum(
            stream[0].mole_frac_phase_comp["Liq", j]
            for j in m.fs.properties.component_list
        )
    ) == pytest.approx(1, rel=1e-3)

    assert value(stream[0].conc_mol_phase_comp["Liq", "Na_+"]) == pytest.approx(
        495.082, rel=1e-3
    )
    assert value(stream[0].conc_mol_phase_comp["Liq", "Cl_-"]) == pytest.approx(
        588.0431, rel=1e-3
    )
    assert value(stream[0].conc_mol_phase_comp["Liq", "Ca_2+"]) == pytest.approx(
        9.777, rel=1e-3
    )
    assert value(stream[0].conc_mol_phase_comp["Liq", "SO4_2-"]) == pytest.approx(
        22.780, rel=1e-3
    )
    assert value(stream[0].conc_mol_phase_comp["Liq", "Mg_2+"]) == pytest.approx(
        59.467, rel=1e-3
    )

    assert value(stream[0].conc_mass_phase_comp["Liq", "Na_+"]) == pytest.approx(
        11.387, rel=1e-3
    )
    assert value(stream[0].conc_mass_phase_comp["Liq", "Cl_-"]) == pytest.approx(
        20.5815, rel=1e-3
    )
    assert value(stream[0].conc_mass_phase_comp["Liq", "Ca_2+"]) == pytest.approx(
        0.391, rel=1e-3
    )
    assert value(stream[0].conc_mass_phase_comp["Liq", "SO4_2-"]) == pytest.approx(
        2.187, rel=1e-3
    )
    assert value(stream[0].conc_mass_phase_comp["Liq", "Mg_2+"]) == pytest.approx(
        1.427, rel=1e-3
    )

    assert value(stream[0].mole_frac_phase_comp["Liq", "Na_+"]) == pytest.approx(
        8.833e-3, rel=1e-3
    )
    assert value(stream[0].mole_frac_phase_comp["Liq", "Cl_-"]) == pytest.approx(
        1.049e-2, rel=1e-3
    )
    assert value(stream[0].mole_frac_phase_comp["Liq", "Ca_2+"]) == pytest.approx(
        1.744e-4, rel=1e-3
    )
    assert value(stream[0].mole_frac_phase_comp["Liq", "SO4_2-"]) == pytest.approx(
        4.064e-4, rel=1e-3
    )
    assert value(stream[0].mole_frac_phase_comp["Liq", "Mg_2+"]) == pytest.approx(
        1.061e-3, rel=1e-3
    )

    assert value(stream[0].mass_frac_phase_comp["Liq", "Na_+"]) == pytest.approx(
        1.112e-2, rel=1e-3
    )
    assert value(stream[0].mass_frac_phase_comp["Liq", "Cl_-"]) == pytest.approx(
        2.01e-2, rel=1e-3
    )
    assert value(stream[0].mass_frac_phase_comp["Liq", "Ca_2+"]) == pytest.approx(
        3.82e-4, rel=1e-3
    )
    assert value(stream[0].mass_frac_phase_comp["Liq", "SO4_2-"]) == pytest.approx(
        2.136e-3, rel=1e-3
    )
    assert value(stream[0].mass_frac_phase_comp["Liq", "Mg_2+"]) == pytest.approx(
        1.394e-3, rel=1e-3
    )

    assert value(stream[0].elec_mobility_phase_comp["Liq", "Na_+"]) == pytest.approx(
        5.177e-8, rel=1e-3
    )
    assert value(stream[0].elec_mobility_phase_comp["Liq", "Cl_-"]) == pytest.approx(
        7.901e-8, rel=1e-3
    )
    assert value(stream[0].elec_mobility_phase_comp["Liq", "Ca_2+"]) == pytest.approx(
        6.165e-8, rel=1e-3
    )
    assert value(stream[0].elec_mobility_phase_comp["Liq", "SO4_2-"]) == pytest.approx(
        8.251e-8, rel=1e-3
    )
    assert value(stream[0].elec_mobility_phase_comp["Liq", "Mg_2+"]) == pytest.approx(
        5.496e-8, rel=1e-3
    )
    assert value(stream[0].trans_num_phase_comp["Liq", "Na_+"]) == pytest.approx(
        0.3066, rel=1e-3
    )
    assert value(stream[0].trans_num_phase_comp["Liq", "Cl_-"]) == pytest.approx(
        0.5558, rel=1e-3
    )
    assert value(stream[0].trans_num_phase_comp["Liq", "Ca_2+"]) == pytest.approx(
        0.01442, rel=1e-3
    )
    assert value(stream[0].trans_num_phase_comp["Liq", "SO4_2-"]) == pytest.approx(
        0.04497, rel=1e-3
    )
    assert value(stream[0].trans_num_phase_comp["Liq", "Mg_2+"]) == pytest.approx(
        0.07820, rel=1e-3
    )

    assert value(stream[0].debye_huckel_constant) == pytest.approx(0.01554, rel=1e-3)
    assert value(stream[0].ionic_strength_molal) == pytest.approx(0.73467, rel=1e-3)
    assert value(stream[0].total_hardness) == pytest.approx(
        value(
            (
                stream[0].conc_mol_phase_comp["Liq", "Ca_2+"]
                + stream[0].conc_mol_phase_comp["Liq", "Mg_2+"]
            )
            * 100.0869
        )
    )
    assert value(stream[0].total_dissolved_solids) == pytest.approx(35974.42, rel=1e-3)
    assert value(stream[0].enth_mass_phase["Liq"]) == pytest.approx(98938.56, rel=1e-3)
    assert value(stream[0].enth_flow) == pytest.approx(98918.931, rel=1e-3)
    assert value(stream[0].pressure_sat) == pytest.approx(3110.73, rel=1e-3)


@pytest.mark.component
def test_assert_electroneutrality_get_property():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.properties = MCASParameterBlock(
        solute_list=["Ca_2+", "SO4_2-", "Na_+", "Cl_-", "Mg_2+"],
        diffusivity_data={
            ("Liq", "Ca_2+"): 7.92e-10,
            ("Liq", "SO4_2-"): 1.06e-09,
            ("Liq", "Na_+"): 1.33e-09,
            ("Liq", "Cl_-"): 2.03e-09,
            ("Liq", "Mg_2+"): 7.06e-10,
        },
        mw_data={
            "H2O": 0.018,
            "Na_+": 0.023,
            "Ca_2+": 0.04,
            "Mg_2+": 0.024,
            "Cl_-": 0.035,
            "SO4_2-": 0.096,
        },
        stokes_radius_data={
            "Na_+": 1.84e-10,
            "Ca_2+": 3.09e-10,
            "Mg_2+": 3.47e-10,
            "Cl_-": 1.21e-10,
            "SO4_2-": 2.3e-10,
        },
        charge={"Na_+": 1, "Ca_2+": 2, "Mg_2+": 2, "Cl_-": -1, "SO4_2-": -2},
        density_calculation=DensityCalculation.seawater,
        activity_coefficient_model=ActivityCoefficientModel.davies,
    )

    m.fs.stream = stream = m.fs.properties.build_state_block([0], defined_state=True)

    mass_flow_in = 1 * pyunits.kg / pyunits.s
    feed_mass_frac = {
        "Na_+": 11122e-6,
        "Ca_2+": 382e-6,
        "Mg_2+": 1394e-6,
        "SO4_2-": 2136e-6,
        "Cl_-": 20300e-6,
    }
    for ion, x in feed_mass_frac.items():
        mol_comp_flow = (
            x * pyunits.kg / pyunits.kg * mass_flow_in / stream[0].mw_comp[ion]
        )

        stream[0].flow_mol_phase_comp["Liq", ion].fix(mol_comp_flow)

    H2O_mass_frac = 1 - sum(x for x in feed_mass_frac.values())
    H2O_mol_comp_flow = (
        H2O_mass_frac
        * pyunits.kg
        / pyunits.kg
        * mass_flow_in
        / stream[0].mw_comp["H2O"]
    )

    stream[0].flow_mol_phase_comp["Liq", "H2O"].fix(H2O_mol_comp_flow)
    stream[0].temperature.fix(298.15)
    stream[0].pressure.fix(101325)

    assert not stream[0].is_property_constructed("pressure_osm_phase")
    assert not stream[0].is_property_constructed("mass_frac_phase_comp")

    stream[0].assert_electroneutrality(
        defined_state=True,
        adjust_by_ion="Cl_-",
        get_property=["mass_frac_phase_comp", "pressure_osm_phase"],
    )
    assert stream[0].is_property_constructed("mass_frac_phase_comp")
    assert stream[0].is_property_constructed("pressure_osm_phase")
    assert not hasattr(stream, "charge_balance")

    assert not stream[0].is_property_constructed("flow_vol")
    stream[0].assert_electroneutrality(
        defined_state=True, adjust_by_ion="Cl_-", get_property="flow_vol"
    )
    assert stream[0].is_property_constructed("flow_vol")
    assert not hasattr(stream, "charge_balance")

    # check that charge_balance constraint is deleted after failed solve
    stream[0].flow_vol_phase.fix(1e5)
    with pytest.raises(
        ValueError,
        match="The stateblock failed to solve while computing "
        "concentrations to check the charge balance.",
    ):
        stream[0].assert_electroneutrality(defined_state=True, adjust_by_ion="Cl_-")
    assert not hasattr(stream, "charge_balance")


@pytest.mark.component
def test_assert_electroneutrality_get_property_2():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.properties = MCASParameterBlock(
        solute_list=["Ca_2+", "SO4_2-", "Na_+", "Cl_-", "Mg_2+"],
        mw_data={
            "H2O": 0.018,
            "Na_+": 0.023,
            "Ca_2+": 0.04,
            "Mg_2+": 0.024,
            "Cl_-": 0.035,
            "SO4_2-": 0.096,
        },
        charge={"Na_+": 1, "Ca_2+": 2, "Mg_2+": 2, "Cl_-": -1, "SO4_2-": -2},
        density_calculation=DensityCalculation.seawater,
        activity_coefficient_model=ActivityCoefficientModel.davies,
    )
    m.fs.stream = stream = m.fs.properties.build_state_block([0], defined_state=True)

    mass_flow_in = 1 * pyunits.kg / pyunits.s
    feed_mass_frac = {
        "Na_+": 11122e-6,
        "Ca_2+": 382e-6,
        "Mg_2+": 1394e-6,
        "SO4_2-": 2136e-6,
        "Cl_-": 20300e-6,
    }
    for ion, x in feed_mass_frac.items():
        mol_comp_flow = (
            x * pyunits.kg / pyunits.kg * mass_flow_in / stream[0].mw_comp[ion]
        )

        stream[0].flow_mol_phase_comp["Liq", ion].fix(mol_comp_flow)

    H2O_mass_frac = 1 - sum(x for x in feed_mass_frac.values())
    H2O_mol_comp_flow = (
        H2O_mass_frac
        * pyunits.kg
        / pyunits.kg
        * mass_flow_in
        / stream[0].mw_comp["H2O"]
    )

    stream[0].flow_mol_phase_comp["Liq", "H2O"].fix(H2O_mol_comp_flow)
    stream[0].temperature.fix(298.15)
    stream[0].pressure.fix(101325)

    # check get_property works with tuple
    stream[0].assert_electroneutrality(
        defined_state=True,
        adjust_by_ion="Cl_-",
        get_property=("mass_frac_phase_comp", "pressure_osm_phase"),
    )
    # check error when adjust_by_ion is not in solute list
    with pytest.raises(
        ValueError,
        match="adjust_by_ion must be set to the name of an ion in the ion_set.",
    ):
        stream[0].assert_electroneutrality(defined_state=True, adjust_by_ion="foo")

    # check error when get_property is not None and defined_state is NOT true
    with pytest.raises(
        ValueError,
        match="Set defined_state to true if get_property" " = flow_mass_phase_comp",
    ):
        stream[0].assert_electroneutrality(
            defined_state=False, get_property="flow_mass_phase_comp"
        )

    # check error when state vars are fixed but defined_state is set to False
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "fs.stream[0].flow_mol_phase_comp[Liq,Ca_2+] was fixed. "
            "Either set defined_state=True or unfix fs.stream[0].flow_mol_phase_comp "
            "for each solute to check that electroneutrality is satisfied."
        ),
    ):
        stream[0].assert_electroneutrality(defined_state=False)

    # check error when get_property receives anything other than str, list, or tuple of strings
    with pytest.raises(
        TypeError,
        match=re.escape("get_property must be a string or list/tuple of strings."),
    ):
        stream[0].assert_electroneutrality(
            defined_state=True, adjust_by_ion="Cl_-", get_property=1
        )

    # check error when electroneutrality condition violated for stringent tolerance
    #   Changed the error message to look for the correct pattern instead of
    #   exact match of the numeric value in the string
    stream[0].flow_mol_phase_comp.unfix()
    with pytest.raises(
        AssertionError,
        match=re.escape("Electroneutrality condition violated in fs.stream[0]. "),
    ):
        stream[0].assert_electroneutrality(defined_state=False, tol=1e-25)


@pytest.fixture(scope="module")
def model4():
    m4 = ConcreteModel()

    m4.fs = FlowsheetBlock(dynamic=False)
    m4.fs.properties = MCASParameterBlock(
        solute_list=["A", "B", "C", "D", "E"],
        diffusivity_data={
            ("Liq", "A"): 1e-09,
            ("Liq", "B"): 1e-10,
            ("Liq", "C"): 1e-07,
            ("Liq", "D"): 1e-11,
            ("Liq", "E"): 1e-11,
        },
        mw_data={"H2O": 0.018, "A": 0.01, "B": 0.025, "C": 0.1, "D": 0.025, "E": 0.025},
        elec_mobility_data={
            ("Liq", "A"): 5.19e-08,
            ("Liq", "B"): 8.29e-08,
            ("Liq", "C"): 6.17e-08,
            ("Liq", "D"): 7.92e-08,
        },
        stokes_radius_data={"A": 1e-09, "B": 1e-09, "C": 1e-09, "D": 1e-10, "E": 1e-10},
        charge={"A": 1, "B": -2, "C": 2, "D": -1, "E": 0},
    )

    # config
    thermo_config = {
        "components": {
            "H2O": {
                "type": Solvent,
                "valid_phase_types": PT.aqueousPhase,
                "dens_mol_liq_comp": Constant,
                "enth_mol_liq_comp": Constant,
                "cp_mol_liq_comp": Constant,
                "entr_mol_liq_comp": Constant,
                # Parameter data is always associated with the methods defined above
                "parameter_data": {
                    "mw": (18.0153, pyunits.g / pyunits.mol),
                    "dens_mol_liq_comp_coeff": (55.2, pyunits.kmol * pyunits.m**-3),
                    "cp_mol_liq_comp_coeff": (
                        75.312,
                        pyunits.J / pyunits.mol / pyunits.K,
                    ),
                    "enth_mol_form_liq_comp_ref": (0, pyunits.kJ / pyunits.mol),
                    "entr_mol_form_liq_comp_ref": (
                        0,
                        pyunits.J / pyunits.K / pyunits.mol,
                    ),
                },
                # End parameter_data
            },
            "A": {
                "type": Cation,
                "charge": 1,
                "dens_mol_liq_comp": Constant,
                "enth_mol_liq_comp": Constant,
                "cp_mol_liq_comp": Constant,
                "entr_mol_liq_comp": Constant,
                "parameter_data": {
                    "mw": (10, pyunits.g / pyunits.mol),
                    "elec_mobility_phase_comp": (
                        5.19e-8,
                        pyunits.meter**2 * pyunits.volt**-1 * pyunits.second**-1,
                    ),
                    "dens_mol_liq_comp_coeff": (55.2, pyunits.kmol * pyunits.m**-3),
                    "cp_mol_liq_comp_coeff": (
                        75.312,
                        pyunits.J / pyunits.mol / pyunits.K,
                    ),
                    "enth_mol_form_liq_comp_ref": (0, pyunits.kJ / pyunits.mol),
                    "entr_mol_form_liq_comp_ref": (
                        0,
                        pyunits.J / pyunits.K / pyunits.mol,
                    ),
                },
            },
            "B": {
                "type": Anion,
                "charge": -2,
                "dens_mol_liq_comp": Constant,
                "enth_mol_liq_comp": Constant,
                "cp_mol_liq_comp": Constant,
                "entr_mol_liq_comp": Constant,
                "parameter_data": {
                    "mw": (25, pyunits.g / pyunits.mol),
                    "elec_mobility_phase_comp": (
                        8.29e-8,
                        pyunits.meter**2 * pyunits.volt**-1 * pyunits.second**-1,
                    ),
                    "dens_mol_liq_comp_coeff": (55.2, pyunits.kmol * pyunits.m**-3),
                    "cp_mol_liq_comp_coeff": (
                        75.312,
                        pyunits.J / pyunits.mol / pyunits.K,
                    ),
                    "enth_mol_form_liq_comp_ref": (0, pyunits.kJ / pyunits.mol),
                    "entr_mol_form_liq_comp_ref": (
                        0,
                        pyunits.J / pyunits.K / pyunits.mol,
                    ),
                },
            },
            "C": {
                "type": Cation,
                "charge": 2,
                "dens_mol_liq_comp": Constant,
                "enth_mol_liq_comp": Constant,
                "cp_mol_liq_comp": Constant,
                "entr_mol_liq_comp": Constant,
                "parameter_data": {
                    "mw": (100, pyunits.g / pyunits.mol),
                    "elec_mobility_phase_comp": (
                        6.17e-8,
                        pyunits.meter**2 * pyunits.volt**-1 * pyunits.second**-1,
                    ),
                    "dens_mol_liq_comp_coeff": (55.2, pyunits.kmol * pyunits.m**-3),
                    "cp_mol_liq_comp_coeff": (
                        75.312,
                        pyunits.J / pyunits.mol / pyunits.K,
                    ),
                    "enth_mol_form_liq_comp_ref": (0, pyunits.kJ / pyunits.mol),
                    "entr_mol_form_liq_comp_ref": (
                        0,
                        pyunits.J / pyunits.K / pyunits.mol,
                    ),
                },
            },
            "D": {
                "type": Anion,
                "charge": -1,
                "dens_mol_liq_comp": Constant,
                "enth_mol_liq_comp": Constant,
                "cp_mol_liq_comp": Constant,
                "entr_mol_liq_comp": Constant,
                "parameter_data": {
                    "mw": (25, pyunits.g / pyunits.mol),
                    "elec_mobility_phase_comp": (
                        7.92e-8,
                        pyunits.meter**2 * pyunits.volt**-1 * pyunits.second**-1,
                    ),
                    "dens_mol_liq_comp_coeff": (55.2, pyunits.kmol * pyunits.m**-3),
                    "cp_mol_liq_comp_coeff": (
                        75.312,
                        pyunits.J / pyunits.mol / pyunits.K,
                    ),
                    "enth_mol_form_liq_comp_ref": (0, pyunits.kJ / pyunits.mol),
                    "entr_mol_form_liq_comp_ref": (
                        0,
                        pyunits.J / pyunits.K / pyunits.mol,
                    ),
                },
            },
            "E": {
                "type": Solute,
                "valid_phase_types": PT.aqueousPhase,
                "dens_mol_liq_comp": Constant,
                "enth_mol_liq_comp": Constant,
                "cp_mol_liq_comp": Constant,
                "entr_mol_liq_comp": Constant,
                "parameter_data": {
                    "mw": (25, pyunits.g / pyunits.mol),
                    "dens_mol_liq_comp_coeff": (55.2, pyunits.kmol * pyunits.m**-3),
                    "cp_mol_liq_comp_coeff": (
                        75.312,
                        pyunits.J / pyunits.mol / pyunits.K,
                    ),
                    "enth_mol_form_liq_comp_ref": (0, pyunits.kJ / pyunits.mol),
                    "entr_mol_form_liq_comp_ref": (
                        0,
                        pyunits.J / pyunits.K / pyunits.mol,
                    ),
                },
            },
        },
        # End Component list
        "phases": {
            "Liq": {"type": AqueousPhase, "equation_of_state": Ideal},
        },
        "state_definition": FpcTP,
        "state_bounds": {
            "temperature": (273.15, 300, 650),
            "pressure": (5e4, 1e5, 1e6),
        },
        "pressure_ref": 1e5,
        "temperature_ref": 300,
        "base_units": {
            "time": pyunits.s,
            "length": pyunits.m,
            "mass": pyunits.kg,
            "amount": pyunits.mol,
            "temperature": pyunits.K,
        },
    }
    # End thermo_config definition

    m5 = ConcreteModel()
    m5.fs = FlowsheetBlock(dynamic=False)
    m5.fs.properties = GenericParameterBlock(**thermo_config)

    return (m4, m5)


@pytest.mark.unit
def test_parameter_block_comparison(model4):
    m_ion = model4[0]
    m_generic = model4[1]

    assert isinstance(m_ion.fs.properties.component_list, Set)
    assert isinstance(m_generic.fs.properties.component_list, Set)
    assert len(m_ion.fs.properties.component_list) == len(
        m_generic.fs.properties.component_list
    )
    for j in m_ion.fs.properties.component_list:
        assert j in ["H2O", "A", "B", "C", "D", "E"]

    assert isinstance(m_ion.fs.properties.cation_set, Set)
    assert isinstance(m_generic.fs.properties.cation_set, Set)
    assert len(m_ion.fs.properties.cation_set) == len(
        m_generic.fs.properties.cation_set
    )
    for j in m_ion.fs.properties.cation_set:
        assert j in ["A", "C"]

    assert isinstance(m_ion.fs.properties.anion_set, Set)
    assert isinstance(m_generic.fs.properties.anion_set, Set)
    assert len(m_ion.fs.properties.anion_set) == len(m_generic.fs.properties.anion_set)
    for j in m_ion.fs.properties.anion_set:
        assert j in ["B", "D"]

    assert isinstance(m_ion.fs.properties.ion_set, Set)
    assert isinstance(m_generic.fs.properties.ion_set, Set)
    assert len(m_ion.fs.properties.ion_set) == len(m_generic.fs.properties.ion_set)
    for j in m_ion.fs.properties.ion_set:
        assert j in ["A", "B", "C", "D"]

    assert isinstance(m_ion.fs.properties.neutral_set, Set)
    for j in m_ion.fs.properties.neutral_set:
        assert j in ["E"]

    assert isinstance(m_ion.fs.properties.solute_set, Set)
    assert isinstance(m_generic.fs.properties.solute_set, Set)
    for j in m_ion.fs.properties.solute_set:
        assert j in ["A", "B", "C", "D", "E"]
        assert m_ion.fs.properties.get_component(j).is_solute()

    assert m_ion.fs.properties.charge_comp["B"].value == -2
    # NOTE: Below is how you grab charge from the generic package
    assert (
        m_ion.fs.properties.charge_comp["B"].value
        == m_generic.fs.properties.get_component("B").config.charge
    )


@pytest.fixture(scope="module")
def model5():
    dic0 = {
        "solute_list": ["Na_+", "Cl_-", "N"],
        "mw_data": {"H2O": 18e-3, "Na_+": 23e-3, "Cl_-": 35.5e-3, "N": 10e-3},
        "charge": {"Na_+": 1, "Cl_-": -1, "N": 0},
        "diffusivity_data": {
            ("Liq", "Na_+"): 1.33e-9,
            ("Liq", "Cl_-"): 2.03e-9,
            ("Liq", "N"): 1.5e-9,
        },
        "elec_mobility_calculation": ElectricalMobilityCalculation.EinsteinRelation,
    }
    dic_transnum = {"trans_num_data": {("Liq", "Na_+"): 0.4, ("Liq", "Cl_-"): 0.6}}
    dic_equivcond = {"equiv_conductivity_phase_data": {"Liq": 0.01}}
    dic_config = {
        "equiv_conductivity_calculation": EquivalentConductivityCalculation.none,
        "trans_num_calculation": TransportNumberCalculation.none,
    }
    dic1 = dic0.copy()
    dic1.update(**dic_transnum, **dic_equivcond, **dic_config)
    m1 = ConcreteModel()
    m2 = ConcreteModel()
    m1.fs = FlowsheetBlock(dynamic=False)
    m2.fs = FlowsheetBlock(dynamic=False)
    m1.fs.properties = MCASParameterBlock(**dic0)
    m2.fs.properties = MCASParameterBlock(**dic1)
    m1.fs.stream = m1.fs.properties.build_state_block([0], defined_state=True)
    m2.fs.stream = m2.fs.properties.build_state_block([0], defined_state=True)
    for m in (m1, m2):
        m.fs.stream[0].pressure.fix(101325)
        m.fs.stream[0].temperature.fix(298.15)
        m.fs.stream[0].flow_mol_phase_comp["Liq", "H2O"].fix(2.40e-1)
        m.fs.stream[0].flow_mol_phase_comp["Liq", "Na_+"].fix(7.38e-4)
        m.fs.stream[0].flow_mol_phase_comp["Liq", "Cl_-"].fix(7.38e-4)
        m.fs.stream[0].flow_mol_phase_comp["Liq", "N"].fix(7.38e-5)
        m.fs.properties.set_default_scaling(
            "flow_mol_phase_comp", 1e1, index=("Liq", "H2O")
        )
        m.fs.properties.set_default_scaling(
            "flow_mol_phase_comp", 1e3, index=("Liq", "Na_+")
        )
        m.fs.properties.set_default_scaling(
            "flow_mol_phase_comp", 1e3, index=("Liq", "Cl_-")
        )
        m.fs.properties.set_default_scaling(
            "flow_mol_phase_comp", 1e4, index=("Liq", "N")
        )
        m.fs.stream[0].elec_mobility_phase_comp
        m.fs.stream[0].trans_num_phase_comp
        m.fs.stream[0].elec_cond_phase
        m.fs.stream.initialize()
        calculate_scaling_factors(m)
    return (m1, m2)


@pytest.mark.unit
def test_elec_properties_robust(model5):
    m = model5
    badly_scaled_var_values_0 = {
        var.name: val for (var, val) in badly_scaled_var_generator(m[0])
    }
    assert not badly_scaled_var_values_0
    badly_scaled_var_values_1 = {
        var.name: val for (var, val) in badly_scaled_var_generator(m[1])
    }
    assert not badly_scaled_var_values_1
    assert value(
        m[0].fs.stream[0].elec_mobility_phase_comp["Liq", "Na_+"]
    ) == pytest.approx(5.177e-8, rel=1e-3)
    assert value(
        m[0].fs.stream[0].elec_mobility_phase_comp["Liq", "Cl_-"]
    ) == pytest.approx(7.901e-8, rel=1e-3)
    assert value(
        m[0].fs.stream[0].trans_num_phase_comp["Liq", "Na_+"]
    ) == pytest.approx(0.396, rel=1e-3)
    assert value(
        m[0].fs.stream[0].trans_num_phase_comp["Liq", "Cl_-"]
    ) == pytest.approx(0.604, rel=1e-3)
    assert value(m[0].fs.stream[0].elec_cond_phase["Liq"]) == pytest.approx(
        2.134, rel=1e-3
    )

    assert value(
        m[1].fs.stream[0].elec_mobility_phase_comp["Liq", "Na_+"]
    ) == pytest.approx(5.177e-8, rel=1e-3)
    assert value(
        m[1].fs.stream[0].elec_mobility_phase_comp["Liq", "Cl_-"]
    ) == pytest.approx(7.901e-8, rel=1e-3)
    assert value(
        m[1].fs.stream[0].trans_num_phase_comp["Liq", "Na_+"]
    ) == pytest.approx(0.4, rel=1e-3)
    assert value(
        m[1].fs.stream[0].trans_num_phase_comp["Liq", "Cl_-"]
    ) == pytest.approx(0.6, rel=1e-3)
    assert value(m[1].fs.stream[0].elec_cond_phase["Liq"]) == pytest.approx(
        1.691, rel=1e-3
    )


@pytest.fixture(scope="module")
def model6():
    dic0 = {
        "solute_list": ["Na_+", "Cl_-", "N"],
        "mw_data": {"H2O": 18e-3, "Na_+": 23e-3, "Cl_-": 35.5e-3, "N": 10e-3},
    }

    dic_charge = {"charge": {"Na_+": 1, "Cl_-": -1, "N": 0}}
    dic_diffus = {
        "diffusivity_data": {
            ("Liq", "Na_+"): 1.33e-9,
            ("Liq", "Cl_-"): 2.03e-9,
            ("Liq", "N"): 1.5e-9,
        }
    }
    dic_config = {
        "elec_mobility_calculation": ElectricalMobilityCalculation.EinsteinRelation,
        "equiv_conductivity_calculation": EquivalentConductivityCalculation.none,
        "trans_num_calculation": TransportNumberCalculation.none,
    }
    dic2 = dic0.copy()
    dic2.update(**dic_charge)
    dic3 = dic0.copy()
    dic3.update(**dic_charge, **dic_diffus, **dic_config)
    m1 = ConcreteModel()
    m2 = ConcreteModel()
    m3 = ConcreteModel()
    m1.fs = FlowsheetBlock(dynamic=False)
    m2.fs = FlowsheetBlock(dynamic=False)
    m3.fs = FlowsheetBlock(dynamic=False)
    m2.fs.properties = MCASParameterBlock(**dic2)
    m3.fs.properties = MCASParameterBlock(**dic3)
    m2.fs.stream = m2.fs.properties.build_state_block([0], defined_state=True)
    m3.fs.stream = m3.fs.properties.build_state_block([0], defined_state=True)
    for m in (m2, m3):
        m.fs.stream[0].pressure.fix(101325)
        m.fs.stream[0].temperature.fix(298.15)
        m.fs.stream[0].flow_mol_phase_comp["Liq", "H2O"].fix(2.40e-1)
        m.fs.stream[0].flow_mol_phase_comp["Liq", "Na_+"].fix(7.38e-4)
        m.fs.stream[0].flow_mol_phase_comp["Liq", "Cl_-"].fix(7.38e-4)
        m.fs.stream[0].flow_mol_phase_comp["Liq", "N"].fix(7.38e-5)
        m.fs.properties.set_default_scaling(
            "flow_mol_phase_comp", 1e1, index=("Liq", "H2O")
        )
        m.fs.properties.set_default_scaling(
            "flow_mol_phase_comp", 1e3, index=("Liq", "Na_+")
        )
        m.fs.properties.set_default_scaling(
            "flow_mol_phase_comp", 1e3, index=("Liq", "Cl_-")
        )
        m.fs.properties.set_default_scaling(
            "flow_mol_phase_comp", 1e4, index=("Liq", "N")
        )
        m.fs.stream.initialize()
        calculate_scaling_factors(m)
    return (m1, m2, m3)


@pytest.mark.unit
def test_elec_properties_errormsg(model6):
    m = model6
    m[0].fs.properties = MCASParameterBlock(
        solute_list=["Na_+", "Cl_-", "N"],
        mw_data={"H2O": 0.018, "Na_+": 0.023, "Cl_-": 0.0355, "N": 0.01},
        charge={"Na_+": 1, "Cl_-": -1},
        elec_mobility_calculation=ElectricalMobilityCalculation.EinsteinRelation,
        ignore_neutral_charge=True,
    )
    m[0].fs.stream = m[0].fs.properties.build_state_block([0], defined_state=True)

    with pytest.raises(
        ConfigurationError,
        match="""Missing a valid diffusivity_data configuration to use EinsteinRelation 
                        to compute the "elec_mobility_phase_comp" """,
    ):
        m[0].fs.stream[0].elec_mobility_phase_comp

    with pytest.raises(
        ConfigurationError,
        match="""Missing the "elec_mobility_data" configuration to build the elec_mobility_phase_comp 
                        and/or its derived variables""",
    ):
        m[1].fs.stream[0].elec_mobility_phase_comp
    with pytest.raises(
        ConfigurationError,
        match="""Missing a valid equiv_conductivity_phase_data configuration to build 
                        "equiv_conductivity_phase" and its derived variables""",
    ):
        m[2].fs.stream[0].equiv_conductivity_phase
    with pytest.raises(
        ConfigurationError,
        match="""Missing a valid trans_num_data configuration to build "trans_num_phase_comp" """,
    ):
        m[2].fs.stream[0].trans_num_phase_comp


@pytest.mark.unit
def test_solute_list_errormsg():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    with pytest.raises(
        ConfigurationError,
        match="'H2O'is reserved as the default solvent and cannot be a solute.",
    ):
        m.fs.properties = MCASParameterBlock(
            solute_list=["H2O", "Na_+", "Cl_-", "N"],
            mw_data={"H2O": 0.018, "Na_+": 0.023, "Cl_-": 0.0355, "N": 0.01},
            charge={"Na_+": 1, "Cl_-": -1},
            ignore_neutral_charge=True,
        )


@pytest.fixture(scope="module")
def model7():
    m_hl = ConcreteModel()

    m_hl.fs = FlowsheetBlock(dynamic=False)

    m_hl.fs.properties = MCASParameterBlock(
        solute_list=["A", "B", "C", "D", "E", "F", "G"],
        charge={"E": 1, "F": -1},
        ignore_neutral_charge=True,
        diffus_calculation=DiffusivityCalculation.HaydukLaudie,
        molar_volume_data={
            ("Liq", "A"): 96e-6,  # tested for benzene
            ("Liq", "B"): 100e-6,  # arbitrary
            ("Liq", "C"): 60e-6,  # arbitrary
            ("Liq", "D"): 200e-6,  # arbitrary
        },
        diffusivity_data={("Liq", "E"): 1.33e-9, ("Liq", "F"): 2.03e-9},
        mw_data={"A": 1, "B": 1, "C": 1, "D": 1, "E": 1, "F": 1, "G": 1},  # arbitrary
    )
    # build state block
    m_hl.fs.sb = m_hl.fs.properties.build_state_block([0], defined_state=True)
    # touch on demand var when hayduklaudie is selected
    m_hl.fs.sb[0].diffus_phase_comp
    m_hl.fs.properties.visc_d_phase["Liq"] = 1.0e-3

    return m_hl


@pytest.mark.unit
def test_diffus_hl(model7):
    m = model7

    assert (
        m.fs.properties.config.diffus_calculation == DiffusivityCalculation.HaydukLaudie
    )

    check_dof(m, fail_flag=True)
    assert_units_consistent(m)

    # init and solve
    m.fs.sb.initialize()
    calculate_scaling_factors(m.fs)
    results = solver.solve(m)
    assert_optimal_termination(results)

    assert isinstance(m.fs.properties.molar_volume_phase_comp, Param)
    assert m.fs.properties.molar_volume_phase_comp["Liq", "A"].value == 96e-6
    assert m.fs.properties.molar_volume_phase_comp["Liq", "B"].value == 100e-6
    assert m.fs.properties.molar_volume_phase_comp["Liq", "C"].value == 60e-6
    assert m.fs.properties.molar_volume_phase_comp["Liq", "D"].value == 200e-6

    sb = m.fs.sb[0]

    assert sb.diffus_phase_comp["Liq", "A"].value == pytest.approx(
        9.015e-10, rel=1e-3
    )  # tested for benzene
    assert sb.diffus_phase_comp["Liq", "B"].value == pytest.approx(8.801e-10, rel=1e-3)
    assert sb.diffus_phase_comp["Liq", "C"].value == pytest.approx(1.189e-09, rel=1e-3)
    assert sb.diffus_phase_comp["Liq", "D"].value == pytest.approx(5.851e-10, rel=1e-3)
    assert sb.diffus_phase_comp["Liq", "E"].value == pytest.approx(1.33e-09, rel=1e-3)
    assert sb.diffus_phase_comp["Liq", "F"].value == pytest.approx(2.03e-09, rel=1e-3)


@pytest.mark.unit
def test_diffus_properties_errormsg(model7):
    m = model7
    with pytest.raises(
        KeyError,
        match=("Index"),
    ):
        m.fs.sb[0].diffus_phase_comp["Liq", "G"]


@pytest.mark.component
def test_solute_list_longer_than_diff_data():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.properties = MCASParameterBlock(
        solute_list=["A", "B", "C", "D", "E", "F", "G", "H"],
        charge={"E": 1, "F": -1},
        ignore_neutral_charge=True,
        mw_data={"A": 1, "B": 1, "C": 1, "D": 1, "E": 1, "F": 1, "G": 1, "H": 1},
        diffus_calculation=DiffusivityCalculation.none,
        molar_volume_data={
            ("Liq", "A"): 96e-6,  # tested for benzene
            ("Liq", "B"): 100e-6,  # arbitrary
            ("Liq", "C"): 60e-6,  # arbitrary
            ("Liq", "D"): 200e-6,  # arbitrary
        },
        diffusivity_data={
            ("Liq", "A"): 1e-11,
            ("Liq", "E"): 1.33e-9,
            ("Liq", "F"): 2.03e-9,
        },
    )
    m.fs.properties.visc_d_phase["Liq"] = 1.0e-3

    m.fs.sb = m.fs.properties.build_state_block([0], defined_state=True)
    calculate_scaling_factors(m.fs)


@pytest.mark.component
def test_flow_mass_basis():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.properties = MCASParameterBlock(
        solute_list=["A"],
        material_flow_basis=MaterialFlowBasis.mass,
        mw_data={"A": 1},
        charge={"A": 0},
    )

    m.fs.sb = m.fs.properties.build_state_block([0], defined_state=True)

    assert m.fs.properties.config.material_flow_basis == MaterialFlowBasis.mass

    m.fs.sb[0].assert_electroneutrality(defined_state=False)
    assert hasattr(m.fs.sb[0], "get_material_flow_terms")

    A = m.fs.sb[0].get_material_flow_terms("Liq", "A")
    print(A)
    A.pprint()
    assert_units_equivalent(
        m.fs.sb[0].get_material_flow_terms("Liq", "A"), pyunits.kg / pyunits.s
    )
    assert (
        m.fs.sb[0].get_material_flow_terms("Liq", "A")
        == m.fs.sb[0].flow_mass_phase_comp["Liq", "A"]
    )


@pytest.mark.unit
def test_compatibility_with_mixer():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.properties = MCASParameterBlock(
        solute_list=["Na_+", "Cl_-"],
        mw_data={"Na_+": 23, "Cl_-": 35},
        charge={"Na_+": 1, "Cl_-": -1},
    )

    m.fs.mixer1 = Mixer(
        property_package=m.fs.properties, energy_mixing_type=MixingType.none
    )

    with pytest.raises(
        NotImplementedError,
        match="property package has not implemented the get_enthalpy_flow_terms method. Please contact the property package developer.",
    ):
        m.fs.mixer2 = Mixer(
            property_package=m.fs.properties,
        )


c_list = [10e-10, 10e-9, 10e-8, 10e-7, 10e-6, 10e-5]


@pytest.mark.parametrize("c", c_list)
@pytest.mark.component
def test_calculate_state_with_flow_mol_stateVar(c):
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    mw = 0.50013 * pyunits.kg / pyunits.mol
    m.fs.properties = MCASParameterBlock(
        solute_list=["target_ion"],
        mw_data={"target_ion": mw, "H2O": 0.018 * pyunits.kg / pyunits.mol},
        ignore_neutral_charge=True,
    )
    m.fs.stream = m.fs.properties.build_state_block([0], defined_state=True)
    m.fs.stream2 = m.fs.properties.build_state_block([0], defined_state=True)

    flow_in = 0.04381 * pyunits.m**3 / pyunits.s
    conc_mass = c * pyunits.kg / pyunits.m**3
    flow_mol = pyunits.convert(
        conc_mass / mw * flow_in, to_units=pyunits.mol / pyunits.s
    )

    set_scaling_factor(m.fs.stream[0].flow_mol_phase_comp["Liq", "H2O"], 1)
    set_scaling_factor(
        m.fs.stream[0].flow_mol_phase_comp["Liq", "target_ion"], value(1 / flow_mol)
    )
    m.fs.stream[0].conc_mol_phase_comp
    m.fs.stream[0].flow_vol_phase
    calculate_scaling_factors(m.fs.stream[0])
    m.fs.stream.calculate_state(
        var_args={
            ("flow_vol_phase", "Liq"): flow_in,
            ("conc_mol_phase_comp", ("Liq", "target_ion")): conc_mass / mw,
            ("pressure", None): 101325,
            ("temperature", None): 298,
        },
        hold_state=True,
    )
    assert pytest.approx(
        value(m.fs.stream[0].flow_mol_phase_comp["Liq", "target_ion"]), rel=1e-3
    ) == value(flow_mol)

    set_scaling_factor(m.fs.stream2[0].flow_mol_phase_comp["Liq", "H2O"], 1)
    set_scaling_factor(
        m.fs.stream2[0].flow_mol_phase_comp["Liq", "target_ion"], value(1 / flow_mol)
    )
    m.fs.stream2[0].conc_mass_phase_comp
    m.fs.stream2[0].flow_vol_phase
    calculate_scaling_factors(m.fs.stream2[0])
    m.fs.stream2.calculate_state(
        var_args={
            ("flow_vol_phase", "Liq"): flow_in,
            ("conc_mass_phase_comp", ("Liq", "target_ion")): conc_mass,
            ("pressure", None): 101325,
            ("temperature", None): 298,
        },
        hold_state=True,
    )
    assert pytest.approx(
        value(m.fs.stream2[0].flow_mol_phase_comp["Liq", "target_ion"]), rel=1e-3
    ) == value(flow_mol)


@pytest.mark.parametrize("c", c_list)
@pytest.mark.component
def test_calculate_state_with_flow_mass_stateVar(c):
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    mw = 0.50013 * pyunits.kg / pyunits.mol
    m.fs.properties = MCASParameterBlock(
        solute_list=["target_ion"],
        mw_data={"target_ion": mw, "H2O": 0.018 * pyunits.kg / pyunits.mol},
        material_flow_basis=MaterialFlowBasis.mass,
        ignore_neutral_charge=True,
    )
    m.fs.stream = m.fs.properties.build_state_block([0], defined_state=True)
    m.fs.stream2 = m.fs.properties.build_state_block([0], defined_state=True)

    flow_in = 0.04381 * pyunits.m**3 / pyunits.s
    conc_mass = c * pyunits.kg / pyunits.m**3
    conc_mol = conc_mass / mw
    flow_mass = pyunits.convert(conc_mass * flow_in, to_units=pyunits.kg / pyunits.s)
    set_scaling_factor(m.fs.stream[0].flow_mass_phase_comp["Liq", "H2O"], 1)
    set_scaling_factor(
        m.fs.stream[0].flow_mass_phase_comp["Liq", "target_ion"], value(1 / flow_mass)
    )
    m.fs.stream[0].conc_mol_phase_comp
    calculate_scaling_factors(m.fs.stream[0])
    m.fs.stream.calculate_state(
        var_args={
            ("flow_vol_phase", "Liq"): flow_in,
            ("conc_mol_phase_comp", ("Liq", "target_ion")): conc_mol,
            ("pressure", None): 101325,
            ("temperature", None): 298,
        },
        hold_state=True,
    )

    assert pytest.approx(
        value(m.fs.stream[0].flow_mass_phase_comp["Liq", "target_ion"]), rel=1e-3
    ) == value(flow_mass)

    set_scaling_factor(m.fs.stream2[0].flow_mass_phase_comp["Liq", "H2O"], 1)
    set_scaling_factor(
        m.fs.stream2[0].flow_mass_phase_comp["Liq", "target_ion"], value(1 / flow_mass)
    )
    m.fs.stream2[0].conc_mass_phase_comp
    calculate_scaling_factors(m.fs.stream2[0])
    m.fs.stream2.calculate_state(
        var_args={
            ("flow_vol_phase", "Liq"): flow_in,
            ("conc_mass_phase_comp", ("Liq", "target_ion")): conc_mass,
            ("pressure", None): 101325,
            ("temperature", None): 298,
        },
        hold_state=True,
    )

    assert pytest.approx(
        value(m.fs.stream2[0].flow_mass_phase_comp["Liq", "target_ion"]), rel=1e-3
    ) == value(flow_mass)


@pytest.mark.component
def test_seawater_data_with_flow_mass_basis():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.properties = MCASParameterBlock(
        solute_list=["Ca_2+", "SO4_2-", "Na_+", "Cl_-", "Mg_2+"],
        material_flow_basis=MaterialFlowBasis.mass,
        diffusivity_data={
            ("Liq", "Ca_2+"): 7.92e-10,
            ("Liq", "SO4_2-"): 1.06e-09,
            ("Liq", "Na_+"): 1.33e-09,
            ("Liq", "Cl_-"): 2.03e-09,
            ("Liq", "Mg_2+"): 7.06e-10,
        },
        mw_data={
            "H2O": 0.018,
            "Na_+": 0.023,
            "Ca_2+": 0.04,
            "Mg_2+": 0.024,
            "Cl_-": 0.035,
            "SO4_2-": 0.096,
        },
        stokes_radius_data={
            "Na_+": 1.84e-10,
            "Ca_2+": 3.09e-10,
            "Mg_2+": 3.47e-10,
            "Cl_-": 1.21e-10,
            "SO4_2-": 2.3e-10,
        },
        charge={"Na_+": 1, "Ca_2+": 2, "Mg_2+": 2, "Cl_-": -1, "SO4_2-": -2},
        elec_mobility_calculation=ElectricalMobilityCalculation.EinsteinRelation,
        density_calculation=DensityCalculation.seawater,
        activity_coefficient_model=ActivityCoefficientModel.davies,
    )

    m.fs.stream = stream = m.fs.properties.build_state_block([0], defined_state=True)

    mass_flow_in = 1 * pyunits.kg / pyunits.s
    feed_mass_frac = {
        "Na_+": 11122e-6,
        "Ca_2+": 382e-6,
        "Mg_2+": 1394e-6,
        "SO4_2-": 2136e-6,
        "Cl_-": 20300e-6,
    }
    for ion, x in feed_mass_frac.items():
        mass_comp_flow = x * pyunits.kg / pyunits.kg * mass_flow_in

        stream[0].flow_mass_phase_comp["Liq", ion].fix(mass_comp_flow)

    H2O_mass_frac = 1 - sum(x for x in feed_mass_frac.values())

    stream[0].flow_mass_phase_comp["Liq", "H2O"].fix(H2O_mass_frac)
    stream[0].temperature.fix(298.15)
    stream[0].pressure.fix(101325)

    stream[0].assert_electroneutrality(
        defined_state=True, tol=1e-2, adjust_by_ion="Cl_-"
    )

    metadata = m.fs.properties.get_metadata().properties
    for v in metadata.list_supported_properties():
        getattr(stream[0], v.name)
        assert stream[0].is_property_constructed(v.name)

    assert_units_consistent(m)

    check_dof(m, fail_flag=True)

    m.fs.properties.set_default_scaling("flow_mass_phase_comp", 1, index=("Liq", "H2O"))
    m.fs.properties.set_default_scaling(
        "flow_mass_phase_comp", 1e2, index=("Liq", "Na_+")
    )
    m.fs.properties.set_default_scaling(
        "flow_mass_phase_comp", 1e2, index=("Liq", "Cl_-")
    )
    m.fs.properties.set_default_scaling(
        "flow_mass_phase_comp", 1e3, index=("Liq", "Ca_2+")
    )
    m.fs.properties.set_default_scaling(
        "flow_mass_phase_comp", 1e3, index=("Liq", "SO4_2-")
    )
    m.fs.properties.set_default_scaling(
        "flow_mass_phase_comp", 1e3, index=("Liq", "Mg_2+")
    )

    calculate_scaling_factors(m)

    stream.initialize()

    badly_scaled_var_list = list(
        badly_scaled_var_generator(m, large=100, small=0.01, zero=1e-10)
    )

    assert len(badly_scaled_var_list) == 0
    results = solver.solve(m)
    assert_optimal_termination(results)

    assert value(stream[0].flow_vol_phase["Liq"]) == pytest.approx(9.767e-4, rel=1e-3)
    assert value(stream[0].flow_mol_phase_comp["Liq", "H2O"]) == pytest.approx(
        53.59256, rel=1e-3
    )
    assert value(stream[0].flow_mol_phase_comp["Liq", "Na_+"]) == pytest.approx(
        0.4836, rel=1e-3
    )
    assert value(stream[0].flow_mol_phase_comp["Liq", "Ca_2+"]) == pytest.approx(
        0.00955, rel=1e-3
    )
    assert value(stream[0].flow_mol_phase_comp["Liq", "Mg_2+"]) == pytest.approx(
        0.05808, rel=1e-3
    )
    assert value(stream[0].flow_mol_phase_comp["Liq", "Cl_-"]) == pytest.approx(
        0.57443, rel=1e-3
    )
    assert value(stream[0].flow_mol_phase_comp["Liq", "SO4_2-"]) == pytest.approx(
        0.02225, rel=1e-3
    )

    assert value(stream[0].dens_mass_phase["Liq"]) == pytest.approx(1023.816, rel=1e-3)
    assert value(stream[0].pressure_osm_phase["Liq"]) == pytest.approx(
        29.132e5, rel=1e-3
    )
    assert value(stream[0].elec_cond_phase["Liq"]) == pytest.approx(8.066, rel=1e-3)
    assert value(stream[0].flow_vol) == pytest.approx(9.767e-4, rel=1e-3)

    assert value(
        sum(
            stream[0].conc_mass_phase_comp["Liq", j]
            for j in m.fs.properties.ion_set | m.fs.properties.solute_set
        )
    ) == pytest.approx(35.9744, rel=1e-3)
    assert value(
        sum(
            stream[0].mass_frac_phase_comp["Liq", j]
            for j in m.fs.properties.ion_set | m.fs.properties.solute_set
        )
    ) == pytest.approx(0.035142, rel=1e-3)
    assert value(
        sum(
            stream[0].mass_frac_phase_comp["Liq", j]
            for j in m.fs.properties.component_list
        )
    ) == pytest.approx(1, rel=1e-3)
    assert value(
        sum(
            stream[0].mole_frac_phase_comp["Liq", j]
            for j in m.fs.properties.component_list
        )
    ) == pytest.approx(1, rel=1e-3)

    assert value(stream[0].conc_mol_phase_comp["Liq", "Na_+"]) == pytest.approx(
        495.082, rel=1e-3
    )
    assert value(stream[0].conc_mol_phase_comp["Liq", "Cl_-"]) == pytest.approx(
        588.0431, rel=1e-3
    )
    assert value(stream[0].conc_mol_phase_comp["Liq", "Ca_2+"]) == pytest.approx(
        9.777, rel=1e-3
    )
    assert value(stream[0].conc_mol_phase_comp["Liq", "SO4_2-"]) == pytest.approx(
        22.780, rel=1e-3
    )
    assert value(stream[0].conc_mol_phase_comp["Liq", "Mg_2+"]) == pytest.approx(
        59.467, rel=1e-3
    )

    assert value(stream[0].conc_mass_phase_comp["Liq", "Na_+"]) == pytest.approx(
        11.387, rel=1e-3
    )
    assert value(stream[0].conc_mass_phase_comp["Liq", "Cl_-"]) == pytest.approx(
        20.5815, rel=1e-3
    )
    assert value(stream[0].conc_mass_phase_comp["Liq", "Ca_2+"]) == pytest.approx(
        0.391, rel=1e-3
    )
    assert value(stream[0].conc_mass_phase_comp["Liq", "SO4_2-"]) == pytest.approx(
        2.187, rel=1e-3
    )
    assert value(stream[0].conc_mass_phase_comp["Liq", "Mg_2+"]) == pytest.approx(
        1.427, rel=1e-3
    )

    assert value(stream[0].mole_frac_phase_comp["Liq", "Na_+"]) == pytest.approx(
        8.833e-3, rel=1e-3
    )
    assert value(stream[0].mole_frac_phase_comp["Liq", "Cl_-"]) == pytest.approx(
        1.049e-2, rel=1e-3
    )
    assert value(stream[0].mole_frac_phase_comp["Liq", "Ca_2+"]) == pytest.approx(
        1.744e-4, rel=1e-3
    )
    assert value(stream[0].mole_frac_phase_comp["Liq", "SO4_2-"]) == pytest.approx(
        4.064e-4, rel=1e-3
    )
    assert value(stream[0].mole_frac_phase_comp["Liq", "Mg_2+"]) == pytest.approx(
        1.061e-3, rel=1e-3
    )

    assert value(stream[0].mass_frac_phase_comp["Liq", "Na_+"]) == pytest.approx(
        1.112e-2, rel=1e-3
    )
    assert value(stream[0].mass_frac_phase_comp["Liq", "Cl_-"]) == pytest.approx(
        2.01e-2, rel=1e-3
    )
    assert value(stream[0].mass_frac_phase_comp["Liq", "Ca_2+"]) == pytest.approx(
        3.82e-4, rel=1e-3
    )
    assert value(stream[0].mass_frac_phase_comp["Liq", "SO4_2-"]) == pytest.approx(
        2.136e-3, rel=1e-3
    )
    assert value(stream[0].mass_frac_phase_comp["Liq", "Mg_2+"]) == pytest.approx(
        1.394e-3, rel=1e-3
    )

    assert value(stream[0].elec_mobility_phase_comp["Liq", "Na_+"]) == pytest.approx(
        5.177e-8, rel=1e-3
    )
    assert value(stream[0].elec_mobility_phase_comp["Liq", "Cl_-"]) == pytest.approx(
        7.901e-8, rel=1e-3
    )
    assert value(stream[0].elec_mobility_phase_comp["Liq", "Ca_2+"]) == pytest.approx(
        6.165e-8, rel=1e-3
    )
    assert value(stream[0].elec_mobility_phase_comp["Liq", "SO4_2-"]) == pytest.approx(
        8.251e-8, rel=1e-3
    )
    assert value(stream[0].elec_mobility_phase_comp["Liq", "Mg_2+"]) == pytest.approx(
        5.496e-8, rel=1e-3
    )
    assert value(stream[0].trans_num_phase_comp["Liq", "Na_+"]) == pytest.approx(
        0.3066, rel=1e-3
    )
    assert value(stream[0].trans_num_phase_comp["Liq", "Cl_-"]) == pytest.approx(
        0.5558, rel=1e-3
    )
    assert value(stream[0].trans_num_phase_comp["Liq", "Ca_2+"]) == pytest.approx(
        0.01442, rel=1e-3
    )
    assert value(stream[0].trans_num_phase_comp["Liq", "SO4_2-"]) == pytest.approx(
        0.04497, rel=1e-3
    )
    assert value(stream[0].trans_num_phase_comp["Liq", "Mg_2+"]) == pytest.approx(
        0.07820, rel=1e-3
    )

    assert value(stream[0].debye_huckel_constant) == pytest.approx(0.01554, rel=1e-3)
    assert value(stream[0].ionic_strength_molal) == pytest.approx(0.73467, rel=1e-3)
    assert value(stream[0].total_hardness) == pytest.approx(
        value(
            pyunits.convert(
                (
                    stream[0].conc_mol_phase_comp["Liq", "Ca_2+"]
                    + stream[0].conc_mol_phase_comp["Liq", "Mg_2+"]
                )
                * 100.0869
                * pyunits.g
                / pyunits.mol,
                to_units=pyunits.mg / pyunits.L,
            )
        ),
        rel=1e-3,
    )
    assert value(stream[0].total_dissolved_solids) == pytest.approx(35974.42, rel=1e-3)


@pytest.mark.component
def test_total_hardness_and_TDS():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)

    m.fs.properties = MCASParameterBlock(
        solute_list=["Ca_2+", "SO4_2-", "Na_+", "Cl_-", "Mg_2+", "Al_3+"],
        charge={
            "Ca_2+": 2,
            "SO4_2-": -2,
            "Na_+": 1,
            "Cl_-": -1,
            "Mg_2+": 2,
            "Al_3+": 3,
        },
        mw_data={
            "Ca_2+": 40e-3,
            "SO4_2-": 97e-3,
            "Na_+": 23e-3,
            "Cl_-": 35e-3,
            "Mg_2+": 24e-3,
            "Al_3+": 27e-3,
        },
        material_flow_basis=MaterialFlowBasis.mass,
    )

    m.fs.stream = stream = m.fs.properties.build_state_block([0], defined_state=True)

    mass_flow_in = 1 * pyunits.kg / pyunits.s
    feed_mass_frac = {
        "Na_+": 11122e-6,
        "Ca_2+": 382e-6,
        "Mg_2+": 1394e-6,
        "SO4_2-": 2136e-6,
        "Cl_-": 20300e-6,
        "Al_3+": 10e-6,
    }
    for ion, x in feed_mass_frac.items():
        mass_comp_flow = x * pyunits.kg / pyunits.kg * mass_flow_in

        stream[0].flow_mass_phase_comp["Liq", ion].fix(mass_comp_flow)

    H2O_mass_frac = 1 - sum(x for x in feed_mass_frac.values())

    stream[0].flow_mass_phase_comp["Liq", "H2O"].fix(H2O_mass_frac)
    stream[0].temperature.fix(298.15)
    stream[0].pressure.fix(101325)

    stream[0].assert_electroneutrality(
        defined_state=True, tol=1e-6, adjust_by_ion="Cl_-"
    )

    stream[0].total_hardness
    stream[0].total_dissolved_solids
    stream[0].conc_mol_phase_comp
    assert_units_consistent(m)

    check_dof(m, fail_flag=True)

    for j in m.fs.properties.component_list:
        m.fs.properties.set_default_scaling(
            "flow_mass_phase_comp",
            value(1 / stream[0].flow_mass_phase_comp["Liq", j]),
            index=("Liq", j),
        )

    calculate_scaling_factors(m)

    stream.initialize()

    results = solver.solve(m)
    assert_optimal_termination(results)

    assert value(stream[0].total_hardness) == pytest.approx(
        value(
            pyunits.convert(
                (
                    stream[0].conc_mol_phase_comp["Liq", "Ca_2+"]
                    + stream[0].conc_mol_phase_comp["Liq", "Mg_2+"]
                    + 3.0 / 2.0 * stream[0].conc_mol_phase_comp["Liq", "Al_3+"]
                )
                * 100.0869
                * pyunits.g
                / pyunits.mol,
                to_units=pyunits.mg / pyunits.L,
            )
        ),
        rel=1e-3,
    )

    assert value(stream[0].total_dissolved_solids) == pytest.approx(35205.61, rel=1e-3)


@pytest.mark.component
def test_no_total_hardness(caplog):
    caplog.set_level(
        idaeslog.INFO, logger="watertap.property_models.multicomp_aq_sol_prop_pack."
    )
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)

    m.fs.properties = MCASParameterBlock(
        solute_list=["Na_+", "Cl_-"],
        charge={"Na_+": 1, "Cl_-": -1},
        mw_data={
            "Na_+": 23e-3,
            "Cl_-": 35e-3,
        },
        material_flow_basis=MaterialFlowBasis.mass,
    )

    m.fs.stream = stream = m.fs.properties.build_state_block([0], defined_state=True)

    mass_flow_in = 1 * pyunits.kg / pyunits.s
    feed_mass_frac = {
        "Na_+": 11122e-6,
        "Cl_-": 20300e-6,
    }
    for ion, x in feed_mass_frac.items():
        mass_comp_flow = x * pyunits.kg / pyunits.kg * mass_flow_in

        stream[0].flow_mass_phase_comp["Liq", ion].fix(mass_comp_flow)

    H2O_mass_frac = 1 - sum(x for x in feed_mass_frac.values())

    stream[0].flow_mass_phase_comp["Liq", "H2O"].fix(H2O_mass_frac)
    stream[0].temperature.fix(298.15)
    stream[0].pressure.fix(101325)

    stream[0].assert_electroneutrality(
        defined_state=True, tol=1e-6, adjust_by_ion="Cl_-"
    )

    stream[0].total_hardness

    assert_units_consistent(m)

    check_dof(m, fail_flag=True)

    for j in m.fs.properties.component_list:
        m.fs.properties.set_default_scaling(
            "flow_mass_phase_comp",
            value(1 / stream[0].flow_mass_phase_comp["Liq", j]),
            index=("Liq", j),
        )

    calculate_scaling_factors(m)

    stream.initialize()

    results = solver.solve(m)
    assert_optimal_termination(results)

    assert value(stream[0].total_hardness) == 0

    assert (
        "Since no multivalent cations were specified in solute_list, total_hardness need not be created. total_hardness has been fixed to 0."
        in caplog.text
    )


@pytest.mark.component
def test_no_total_hardness_not_TDS_with_apparent_species_only(caplog):
    caplog.set_level(
        idaeslog.INFO, logger="watertap.property_models.multicomp_aq_sol_prop_pack."
    )
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)

    m.fs.properties = MCASParameterBlock(
        solute_list=["NaCl"],
        ignore_neutral_charge=True,
        material_flow_basis=MaterialFlowBasis.mass,
    )

    m.fs.stream = stream = m.fs.properties.build_state_block([0], defined_state=True)

    mass_flow_in = 1 * pyunits.kg / pyunits.s
    feed_mass_frac = {
        "NaCl": 11122e-6 + 20300e-6,
    }
    for ion, x in feed_mass_frac.items():
        mass_comp_flow = x * pyunits.kg / pyunits.kg * mass_flow_in

        stream[0].flow_mass_phase_comp["Liq", ion].fix(mass_comp_flow)

    H2O_mass_frac = 1 - sum(x for x in feed_mass_frac.values())

    stream[0].flow_mass_phase_comp["Liq", "H2O"].fix(H2O_mass_frac)
    stream[0].temperature.fix(298.15)
    stream[0].pressure.fix(101325)

    stream[0].total_hardness

    stream[0].total_dissolved_solids
    # stream[0].total_hardness

    assert_units_consistent(m)

    check_dof(m, fail_flag=True)

    assert value(stream[0].total_dissolved_solids) == 0
    assert value(stream[0].total_hardness) == 0

    assert (
        "Since no ions were specified in solute_list, total_dissolved_solids has been fixed to 0. The  total_dissolved_solids calculation does not currently account for apparent species (e.g., NaCl)."
        in caplog.text
    )
    assert (
        "Since no multivalent cations were specified in solute_list, total_hardness need not be created. total_hardness has been fixed to 0."
        in caplog.text
    )


@pytest.mark.component
def test_flow_mass_basis_with_RO_unit():
    m = ConcreteModel()
    m.fs = FlowsheetBlock()
    m.fs.properties = MCASParameterBlock(
        solute_list=["NaCl"],
        diffusivity_data={("Liq", "NaCl"): 1e-9},
        mw_data={"NaCl": 0.058, "H2O": 0.018},
        material_flow_basis=MaterialFlowBasis.mass,
        ignore_neutral_charge=True,
    )

    # Import RO model
    from watertap.unit_models.reverse_osmosis_0D import (
        ReverseOsmosis0D,
        ConcentrationPolarizationType,
        MassTransferCoefficient,
    )

    m.fs.unit = ReverseOsmosis0D(
        property_package=m.fs.properties,
        concentration_polarization_type=ConcentrationPolarizationType.calculated,
        mass_transfer_coefficient=MassTransferCoefficient.fixed,
        has_pressure_change=True,
    )

    feed_flow_mass = 1
    feed_mass_frac_NaCl = 0.035
    feed_pressure = 50e5
    feed_temperature = 273.15 + 25
    membrane_pressure_drop = 3e5
    membrane_area = 50
    A = 4.2e-12
    B = 3.5e-8
    pressure_atmospheric = 101325
    kf = 2e-5

    feed_mass_frac_H2O = 1 - feed_mass_frac_NaCl
    m.fs.unit.inlet.flow_mass_phase_comp[0, "Liq", "NaCl"].fix(
        feed_flow_mass * feed_mass_frac_NaCl
    )
    m.fs.unit.inlet.flow_mass_phase_comp[0, "Liq", "H2O"].fix(
        feed_flow_mass * feed_mass_frac_H2O
    )
    m.fs.unit.inlet.pressure[0].fix(feed_pressure)
    m.fs.unit.inlet.temperature[0].fix(feed_temperature)
    m.fs.unit.deltaP.fix(-membrane_pressure_drop)
    m.fs.unit.area.fix(membrane_area)
    m.fs.unit.A_comp.fix(A)
    m.fs.unit.B_comp.fix(B)
    m.fs.unit.permeate.pressure[0].fix(pressure_atmospheric)
    m.fs.unit.feed_side.K[0, 0.0, "NaCl"].fix(kf)
    m.fs.unit.feed_side.K[0, 1.0, "NaCl"].fix(kf)
    m.fs.properties.set_default_scaling("flow_mass_phase_comp", 1, index=("Liq", "H2O"))
    m.fs.properties.set_default_scaling(
        "flow_mass_phase_comp", 1e2, index=("Liq", "NaCl")
    )
    calculate_scaling_factors(m)
    m.fs.unit.initialize()
    results = solver.solve(m)
    assert_optimal_termination(results)


@pytest.mark.unit
def test_no_solute_list_provided():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    with pytest.raises(
        ConfigurationError,
        match=re.escape(
            "The solute_list argument was not provided while instantiating the MCAS property model. Provide a list of solutes to solute_list (as a list of strings).",
        ),
    ):
        m.fs.properties = MCASParameterBlock()

    with pytest.raises(
        ConfigurationError,
        match=re.escape(
            "The solute_list argument was not provided while instantiating the MCAS property model. Provide a list of solutes to solute_list (as a list of strings).",
        ),
    ):
        m.fs.properties = MCASParameterBlock(solute_list=[])


@pytest.mark.unit
def test_no_mw_data_provided():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    msg = "Molecular weight data could not be obtained for the following solutes and no data were provided\n: {'foo': OSError('Molecular weight data could not be found for foo.')}."
    with pytest.raises(ConfigurationError, match=re.escape(msg)):
        m.fs.properties = MCASParameterBlock(
            solute_list=["foo"], ignore_neutral_charge=True
        )
    msg = "Molecular weight data could not be obtained for the following solutes and no data were provided\n: {'blahblah': OSError('Molecular weight data could not be found for blahblah.'), 'booboo': OSError('Molecular weight data could not be found for booboo.')}."
    with pytest.raises(ConfigurationError, match=re.escape(msg)):
        m.fs.properties2 = MCASParameterBlock(
            solute_list=["blahblah", "booboo"], ignore_neutral_charge=True
        )
    msg = "Molecular weight data could not be obtained for the following solutes and no data were provided\n: {'foo': OSError('Molecular weight data could not be found for foo.')}."
    with pytest.raises(ConfigurationError, match=re.escape(msg)):
        m.fs.properties = MCASParameterBlock(
            solute_list=["foo", "Na_+", "Cl_-"], ignore_neutral_charge=True
        )
    # Choose capitalized solute names that won't be found in periodic table and omit mw_data
    # Because of helper function logic, an example like following would result in an empty result
    msg = "The symbol 'T' from the component name 'TDS' could not be found in the periodic table."
    with pytest.raises(ConfigurationError, match=re.escape(msg)):
        m.fs.mcas_props = MCASParameterBlock(solute_list=["TDS"])


@pytest.mark.unit
def test_no_h2o_mw_data():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.properties = MCASParameterBlock(
        solute_list=["NaCl"],
        mw_data={"NaCl": 58e-3},
        ignore_neutral_charge=True,
    )

    m.fs.stream = m.fs.properties.build_state_block([0], defined_state=True)

    assert isinstance(m.fs.properties.mw_comp, Param)
    assert m.fs.properties.mw_comp["NaCl"].value == 58e-3
    assert m.fs.properties.mw_comp["H2O"].value == 18e-3
    assert m.fs.stream[0].mw_comp["NaCl"].value == 58e-3
    assert m.fs.stream[0].mw_comp["H2O"].value == 18e-3


@pytest.mark.unit
def test_no_h2o_mw_data_overwrite():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.properties = MCASParameterBlock(
        solute_list=["NaCl"],
        mw_data={"H2O": 18.1e-3, "NaCl": 58e-3},
        ignore_neutral_charge=True,
    )

    m.fs.stream = m.fs.properties.build_state_block([0], defined_state=True)

    assert isinstance(m.fs.properties.mw_comp, Param)
    assert m.fs.properties.mw_comp["NaCl"].value == 58e-3
    assert m.fs.properties.mw_comp["H2O"].value == 18.1e-3
    assert m.fs.stream[0].mw_comp["NaCl"].value == 58e-3
    assert m.fs.stream[0].mw_comp["H2O"].value == 18.1e-3


@pytest.mark.unit
def test_get_neutral_charge():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.properties = MCASParameterBlock(
        solute_list=["Na_+", "Cl_-", "N"],
        mw_data={"H2O": 0.018, "Na_+": 0.023, "Cl_-": 0.0355, "N": 0.01},
        charge={"Na_+": 1, "Cl_-": -1},
    )
    m.fs.stream = m.fs.properties.build_state_block([0], defined_state=True)
    m.fs.stream[0].charge_comp["N"] == 0


@pytest.mark.unit
def test_get_charge2():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.properties = MCASParameterBlock(
        solute_list=["Na_+", "Cl_-", "N"],
        mw_data={"H2O": 0.018, "Na_+": 0.023, "Cl_-": 0.0355, "N": 0.01},
    )
    m.fs.stream = m.fs.properties.build_state_block([0], defined_state=True)
    m.fs.stream[0].charge_comp["Na_+"] == 1
    m.fs.stream[0].charge_comp["Cl_-"] == -1
    m.fs.stream[0].charge_comp["N"] == 0


@pytest.mark.unit
def test_get_charge_not_obtained():
    msg = "Charge data could not be obtained for the following solutes and no data were provided\n: {'target_ion': OSError(\"Charge sign could not be determined from the string 'target_ion'\")}"
    with pytest.raises(ConfigurationError, match=re.escape(msg)):
        m = ConcreteModel()
        m.fs = FlowsheetBlock(dynamic=False)
        m.fs.properties = MCASParameterBlock(
            solute_list=["target_ion"],
            mw_data={"target_ion": 0.023},
        )

    msg = "Charge data could not be obtained for the following solutes and no data were provided\n: {'target_ion': OSError(\"Charge sign could not be determined from the string 'target_ion'\"), 'my_target_ion_': OSError(\"Charge could not be determined from the string 'my_target_ion_'\")}"
    with pytest.raises(ConfigurationError, match=re.escape(msg)):
        m = ConcreteModel()
        m.fs = FlowsheetBlock(dynamic=False)
        m.fs.properties = MCASParameterBlock(
            solute_list=["target_ion", "my_target_ion_"],
            mw_data={"target_ion": 0.023, "my_target_ion_": 1},
        )
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.properties = MCASParameterBlock(
        solute_list=["target_ion"],
        mw_data={"target_ion": 0.023},
        ignore_neutral_charge=True,
    )


@pytest.mark.unit
def test_automatic_charge_mw_population():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.properties = MCASParameterBlock(
        solute_list=[
            "Na_+",
            "Cl_-",
            "Ca_2+",
            "K_+",
            "HCO3_-",
            "SO4_2-",
            "Mg_2+",
        ],
    )

    test_vals = {
        "Na_+": (0.02299, 1),
        "Cl_-": (0.03545, -1),
        "Ca_2+": (0.04008, 2),
        "K_+": (0.039098, 1),
        "HCO3_-": (0.061016, -1),
        "SO4_2-": (0.096066, -2),
        "Mg_2+": (0.024305, 2),
    }
    m.fs.stream = m.fs.properties.build_state_block([0], defined_state=True)
    for comp, val in test_vals.items():
        assert value(m.fs.stream[0].mw_comp[comp]) == val[0]
        assert value(m.fs.stream[0].charge_comp[comp]) == val[1]

    m.fs.properties2 = MCASParameterBlock(
        solute_list=[
            "target_ion",
            "Na_+",
            "Cl_-",
            "Ca_2+",
            "K_+",
            "HCO3_-",
            "SO4_2-",
            "Mg_2+",
        ],
        mw_data={"target_ion": 0.023},
        charge={"target_ion": 0},
    )

    test_vals = {
        "target_ion": (0.023, 0),
        "Na_+": (0.02299, 1),
        "Cl_-": (0.03545, -1),
        "Ca_2+": (0.04008, 2),
        "K_+": (0.039098, 1),
        "HCO3_-": (0.061016, -1),
        "SO4_2-": (0.096066, -2),
        "Mg_2+": (0.024305, 2),
    }
    m.fs.stream2 = m.fs.properties2.build_state_block([0], defined_state=True)
    for comp, val in test_vals.items():
        assert value(m.fs.stream2[0].mw_comp[comp]) == val[0]
        assert value(m.fs.stream2[0].charge_comp[comp]) == val[1]


class TestMCASInitializer:
    @pytest.mark.component
    def test_initialize(self):
        m = ConcreteModel()
        m.fs = FlowsheetBlock(dynamic=False)
        m.fs.properties = MCASParameterBlock(
            solute_list=["Ca_2+", "SO4_2-", "Na_+", "Cl_-", "Mg_2+"],
            material_flow_basis=MaterialFlowBasis.mass,
            diffusivity_data={
                ("Liq", "Ca_2+"): 7.92e-10,
                ("Liq", "SO4_2-"): 1.06e-09,
                ("Liq", "Na_+"): 1.33e-09,
                ("Liq", "Cl_-"): 2.03e-09,
                ("Liq", "Mg_2+"): 7.06e-10,
            },
            mw_data={
                "H2O": 0.018,
                "Na_+": 0.023,
                "Ca_2+": 0.04,
                "Mg_2+": 0.024,
                "Cl_-": 0.035,
                "SO4_2-": 0.096,
            },
            stokes_radius_data={
                "Na_+": 1.84e-10,
                "Ca_2+": 3.09e-10,
                "Mg_2+": 3.47e-10,
                "Cl_-": 1.21e-10,
                "SO4_2-": 2.3e-10,
            },
            charge={"Na_+": 1, "Ca_2+": 2, "Mg_2+": 2, "Cl_-": -1, "SO4_2-": -2},
            elec_mobility_calculation=ElectricalMobilityCalculation.EinsteinRelation,
            density_calculation=DensityCalculation.seawater,
            activity_coefficient_model=ActivityCoefficientModel.davies,
        )

        m.fs.stream = stream = m.fs.properties.build_state_block(
            [0], defined_state=True
        )

        mass_flow_in = 1 * pyunits.kg / pyunits.s
        feed_mass_frac = {
            "Na_+": 11122e-6,
            "Ca_2+": 382e-6,
            "Mg_2+": 1394e-6,
            "SO4_2-": 2136e-6,
            "Cl_-": 20300e-6,
        }
        for ion, x in feed_mass_frac.items():
            mass_comp_flow = x * pyunits.kg / pyunits.kg * mass_flow_in

            stream[0].flow_mass_phase_comp["Liq", ion].fix(mass_comp_flow)

        H2O_mass_frac = 1 - sum(x for x in feed_mass_frac.values())

        stream[0].flow_mass_phase_comp["Liq", "H2O"].fix(H2O_mass_frac)
        stream[0].temperature.fix(298.15)
        stream[0].pressure.fix(101325)

        stream[0].assert_electroneutrality(
            defined_state=True, tol=1e-2, adjust_by_ion="Cl_-"
        )

        metadata = m.fs.properties.get_metadata().properties
        for v in metadata.list_supported_properties():
            getattr(stream[0], v.name)
            assert stream[0].is_property_constructed(v.name)

        initializer = stream.default_initializer()
        initializer.initialize(stream)
        assert initializer.summary[stream]["status"] == InitializationStatus.Ok


class TestMCASScaler:
    @pytest.mark.component
    def test_scaler(self):
        m = ConcreteModel()
        m.fs = FlowsheetBlock(dynamic=False)
        m.fs.properties = MCASParameterBlock(
            solute_list=["Ca_2+", "SO4_2-", "Na_+", "Cl_-", "Mg_2+"],
            material_flow_basis=MaterialFlowBasis.mass,
            diffusivity_data={
                ("Liq", "Ca_2+"): 7.92e-10,
                ("Liq", "SO4_2-"): 1.06e-09,
                ("Liq", "Na_+"): 1.33e-09,
                ("Liq", "Cl_-"): 2.03e-09,
                ("Liq", "Mg_2+"): 7.06e-10,
            },
            mw_data={
                "H2O": 0.018,
                "Na_+": 0.023,
                "Ca_2+": 0.04,
                "Mg_2+": 0.024,
                "Cl_-": 0.035,
                "SO4_2-": 0.096,
            },
            stokes_radius_data={
                "Na_+": 1.84e-10,
                "Ca_2+": 3.09e-10,
                "Mg_2+": 3.47e-10,
                "Cl_-": 1.21e-10,
                "SO4_2-": 2.3e-10,
            },
            charge={"Na_+": 1, "Ca_2+": 2, "Mg_2+": 2, "Cl_-": -1, "SO4_2-": -2},
            elec_mobility_calculation=ElectricalMobilityCalculation.EinsteinRelation,
            density_calculation=DensityCalculation.seawater,
            activity_coefficient_model=ActivityCoefficientModel.davies,
        )

        m.fs.stream = stream = m.fs.properties.build_state_block(
            [0], defined_state=True
        )

        mass_flow_in = 1 * pyunits.kg / pyunits.s
        feed_mass_frac = {
            "Na_+": 11122e-6,
            "Ca_2+": 382e-6,
            "Mg_2+": 1394e-6,
            "SO4_2-": 2136e-6,
            "Cl_-": 20300e-6,
        }
        for ion, x in feed_mass_frac.items():
            mass_comp_flow = x * pyunits.kg / pyunits.kg * mass_flow_in

            stream[0].flow_mass_phase_comp["Liq", ion].fix(mass_comp_flow)

        H2O_mass_frac = 1 - sum(x for x in feed_mass_frac.values())

        stream[0].flow_mass_phase_comp["Liq", "H2O"].fix(H2O_mass_frac)
        stream[0].temperature.fix(298.15)
        stream[0].pressure.fix(101325)

        stream[0].assert_electroneutrality(
            defined_state=True, tol=1e-2, adjust_by_ion="Cl_-"
        )

        metadata = m.fs.properties.get_metadata().properties
        for v in metadata.list_supported_properties():
            getattr(stream[0], v.name)
            assert stream[0].is_property_constructed(v.name)

        initializer = stream.default_initializer()
        initializer.initialize(stream)
        assert initializer.summary[stream]["status"] == InitializationStatus.Ok

        scaler = stream.default_scaler()

        set_scaling_factor(stream[0].flow_mass_phase_comp["Liq", "H2O"], 1)
        set_scaling_factor(stream[0].flow_mass_phase_comp["Liq", "Ca_2+"], 1e4)
        set_scaling_factor(stream[0].flow_mass_phase_comp["Liq", "SO4_2-"], 1e3)
        set_scaling_factor(stream[0].flow_mass_phase_comp["Liq", "Na_+"], 1e2)
        set_scaling_factor(stream[0].flow_mass_phase_comp["Liq", "Cl_-"], 1e2)
        set_scaling_factor(stream[0].flow_mass_phase_comp["Liq", "Mg_2+"], 1e3)

        scaler.scale_model(stream[0])

        assert len(m.fs.stream[0].scaling_factor) == 154

        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].temperature
        ] == pytest.approx(0.01, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[m.fs.stream[0].pressure] == pytest.approx(
            0.00001, rel=1e-3
        )
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].flow_mass_phase_comp["Liq", "H2O"]
        ] == pytest.approx(1, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].flow_mass_phase_comp["Liq", "Ca_2+"]
        ] == pytest.approx(10000, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].flow_mass_phase_comp["Liq", "SO4_2-"]
        ] == pytest.approx(1000, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].flow_mass_phase_comp["Liq", "Na_+"]
        ] == pytest.approx(100, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].flow_mass_phase_comp["Liq", "Cl_-"]
        ] == pytest.approx(100, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].flow_mass_phase_comp["Liq", "Mg_2+"]
        ] == pytest.approx(1000, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].conc_mol_phase_comp["Liq", "H2O"]
        ] == pytest.approx(0.0001, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].conc_mol_phase_comp["Liq", "Ca_2+"]
        ] == pytest.approx(0.4, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].conc_mol_phase_comp["Liq", "SO4_2-"]
        ] == pytest.approx(0.096, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].conc_mol_phase_comp["Liq", "Na_+"]
        ] == pytest.approx(0.0023, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].conc_mol_phase_comp["Liq", "Cl_-"]
        ] == pytest.approx(0.0035, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].conc_mol_phase_comp["Liq", "Mg_2+"]
        ] == pytest.approx(0.024, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].conc_mass_phase_comp["Liq", "H2O"]
        ] == pytest.approx(0.001, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].conc_mass_phase_comp["Liq", "Ca_2+"]
        ] == pytest.approx(10, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].conc_mass_phase_comp["Liq", "SO4_2-"]
        ] == pytest.approx(1, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].conc_mass_phase_comp["Liq", "Na_+"]
        ] == pytest.approx(0.1, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].conc_mass_phase_comp["Liq", "Cl_-"]
        ] == pytest.approx(0.1, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].conc_mass_phase_comp["Liq", "Mg_2+"]
        ] == pytest.approx(1, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].dens_mass_phase["Liq"]
        ] == pytest.approx(0.001, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].mass_frac_phase_comp["Liq", "H2O"]
        ] == pytest.approx(1, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].mass_frac_phase_comp["Liq", "Ca_2+"]
        ] == pytest.approx(10000, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].mass_frac_phase_comp["Liq", "SO4_2-"]
        ] == pytest.approx(1000, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].mass_frac_phase_comp["Liq", "Na_+"]
        ] == pytest.approx(100, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].mass_frac_phase_comp["Liq", "Cl_-"]
        ] == pytest.approx(100, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].mass_frac_phase_comp["Liq", "Mg_2+"]
        ] == pytest.approx(1000, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].dens_mass_solvent
        ] == pytest.approx(0.001, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].act_coeff_phase_comp["Liq", "Ca_2+"]
        ] == pytest.approx(1, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].act_coeff_phase_comp["Liq", "SO4_2-"]
        ] == pytest.approx(1, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].act_coeff_phase_comp["Liq", "Na_+"]
        ] == pytest.approx(1, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].act_coeff_phase_comp["Liq", "Cl_-"]
        ] == pytest.approx(1, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].act_coeff_phase_comp["Liq", "Mg_2+"]
        ] == pytest.approx(1, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].ionic_strength_molal
        ] == pytest.approx(2.571, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].molality_phase_comp["Liq", "Ca_2+"]
        ] == pytest.approx(18, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].molality_phase_comp["Liq", "SO4_2-"]
        ] == pytest.approx(18, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].molality_phase_comp["Liq", "Na_+"]
        ] == pytest.approx(18, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].molality_phase_comp["Liq", "Cl_-"]
        ] == pytest.approx(18, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].molality_phase_comp["Liq", "Mg_2+"]
        ] == pytest.approx(18, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].flow_mol_phase_comp["Liq", "H2O"]
        ] == pytest.approx(0.1, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].flow_mol_phase_comp["Liq", "Ca_2+"]
        ] == pytest.approx(100, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].flow_mol_phase_comp["Liq", "SO4_2-"]
        ] == pytest.approx(100, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].flow_mol_phase_comp["Liq", "Na_+"]
        ] == pytest.approx(100, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].flow_mol_phase_comp["Liq", "Cl_-"]
        ] == pytest.approx(100, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].flow_mol_phase_comp["Liq", "Mg_2+"]
        ] == pytest.approx(100, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].debye_huckel_constant
        ] == pytest.approx(100, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].enth_mass_phase["Liq"]
        ] == pytest.approx(0.00001, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].total_dissolved_solids
        ] == pytest.approx(0.0000278, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].flow_vol_phase["Liq"]
        ] == pytest.approx(5556, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].mole_frac_phase_comp["Liq", "H2O"]
        ] == pytest.approx(1, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].mole_frac_phase_comp["Liq", "Ca_2+"]
        ] == pytest.approx(1000, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].mole_frac_phase_comp["Liq", "SO4_2-"]
        ] == pytest.approx(1000, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].mole_frac_phase_comp["Liq", "Na_+"]
        ] == pytest.approx(1000, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].mole_frac_phase_comp["Liq", "Cl_-"]
        ] == pytest.approx(1000, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].mole_frac_phase_comp["Liq", "Mg_2+"]
        ] == pytest.approx(1000, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].pressure_osm_phase["Liq"]
        ] == pytest.approx(0.00000129, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].pressure_sat
        ] == pytest.approx(0.001, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].visc_k_phase["Liq"]
        ] == pytest.approx(1000000, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].flow_equiv_phase_comp["Liq", "Ca_2+"]
        ] == pytest.approx(10000, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].flow_equiv_phase_comp["Liq", "SO4_2-"]
        ] == pytest.approx(1000, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].flow_equiv_phase_comp["Liq", "Na_+"]
        ] == pytest.approx(100, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].flow_equiv_phase_comp["Liq", "Cl_-"]
        ] == pytest.approx(100, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].flow_equiv_phase_comp["Liq", "Mg_2+"]
        ] == pytest.approx(1000, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].conc_equiv_phase_comp["Liq", "Ca_2+"]
        ] == pytest.approx(0.4, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].conc_equiv_phase_comp["Liq", "SO4_2-"]
        ] == pytest.approx(0.096, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].conc_equiv_phase_comp["Liq", "Na_+"]
        ] == pytest.approx(0.0023, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].conc_equiv_phase_comp["Liq", "Cl_-"]
        ] == pytest.approx(0.0035, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].conc_equiv_phase_comp["Liq", "Mg_2+"]
        ] == pytest.approx(0.024, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].equiv_conductivity_phase["Liq"]
        ] == pytest.approx(1545, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].elec_mobility_phase_comp["Liq", "Ca_2+"]
        ] == pytest.approx(250000000, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].elec_mobility_phase_comp["Liq", "SO4_2-"]
        ] == pytest.approx(250000000, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].elec_mobility_phase_comp["Liq", "Na_+"]
        ] == pytest.approx(250000000, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].elec_mobility_phase_comp["Liq", "Cl_-"]
        ] == pytest.approx(250000000, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].elec_mobility_phase_comp["Liq", "Mg_2+"]
        ] == pytest.approx(250000000, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].elec_cond_phase["Liq"]
        ] == pytest.approx(3.225, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].trans_num_phase_comp["Liq", "Ca_2+"]
        ] == pytest.approx(10, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].trans_num_phase_comp["Liq", "SO4_2-"]
        ] == pytest.approx(10, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].trans_num_phase_comp["Liq", "Na_+"]
        ] == pytest.approx(10, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].trans_num_phase_comp["Liq", "Cl_-"]
        ] == pytest.approx(10, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].trans_num_phase_comp["Liq", "Mg_2+"]
        ] == pytest.approx(10, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].total_hardness
        ] == pytest.approx(0.0001443, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].eq_conc_mol_phase_comp["Liq", "H2O"]
        ] == pytest.approx(0.001, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].eq_conc_mol_phase_comp["Liq", "Ca_2+"]
        ] == pytest.approx(10, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].eq_conc_mol_phase_comp["Liq", "SO4_2-"]
        ] == pytest.approx(1, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].eq_conc_mol_phase_comp["Liq", "Na_+"]
        ] == pytest.approx(0.1, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].eq_conc_mol_phase_comp["Liq", "Cl_-"]
        ] == pytest.approx(0.1, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].eq_conc_mol_phase_comp["Liq", "Mg_2+"]
        ] == pytest.approx(1, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].eq_conc_mass_phase_comp["Liq", "H2O"]
        ] == pytest.approx(0.001, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].eq_conc_mass_phase_comp["Liq", "Ca_2+"]
        ] == pytest.approx(10, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].eq_conc_mass_phase_comp["Liq", "SO4_2-"]
        ] == pytest.approx(1, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].eq_conc_mass_phase_comp["Liq", "Na_+"]
        ] == pytest.approx(0.1, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].eq_conc_mass_phase_comp["Liq", "Cl_-"]
        ] == pytest.approx(0.1, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].eq_conc_mass_phase_comp["Liq", "Mg_2+"]
        ] == pytest.approx(1, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].eq_dens_mass_phase["Liq"]
        ] == pytest.approx(0.001, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].eq_mass_frac_phase_comp["Liq", "H2O"]
        ] == pytest.approx(1, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].eq_mass_frac_phase_comp["Liq", "Ca_2+"]
        ] == pytest.approx(2617, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].eq_mass_frac_phase_comp["Liq", "SO4_2-"]
        ] == pytest.approx(468.1, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].eq_mass_frac_phase_comp["Liq", "Na_+"]
        ] == pytest.approx(89.89, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].eq_mass_frac_phase_comp["Liq", "Cl_-"]
        ] == pytest.approx(49.74, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].eq_mass_frac_phase_comp["Liq", "Mg_2+"]
        ] == pytest.approx(717.2, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].eq_dens_mass_solvent
        ] == pytest.approx(0.001, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].eq_act_coeff_phase_comp["Liq", "Ca_2+"]
        ] == pytest.approx(65.09, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].eq_act_coeff_phase_comp["Liq", "SO4_2-"]
        ] == pytest.approx(65.09, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].eq_act_coeff_phase_comp["Liq", "Na_+"]
        ] == pytest.approx(260.4, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].eq_act_coeff_phase_comp["Liq", "Cl_-"]
        ] == pytest.approx(260.4, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].eq_act_coeff_phase_comp["Liq", "Mg_2+"]
        ] == pytest.approx(65.09, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].eq_ionic_strength_molal
        ] == pytest.approx(2.571, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].eq_molality_phase_comp["Liq", "Ca_2+"]
        ] == pytest.approx(18, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].eq_molality_phase_comp["Liq", "SO4_2-"]
        ] == pytest.approx(18, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].eq_molality_phase_comp["Liq", "Na_+"]
        ] == pytest.approx(18, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].eq_molality_phase_comp["Liq", "Cl_-"]
        ] == pytest.approx(18, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].eq_molality_phase_comp["Liq", "Mg_2+"]
        ] == pytest.approx(18, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].eq_flow_mol_phase_comp["Liq", "H2O"]
        ] == pytest.approx(1.037, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].eq_flow_mol_phase_comp["Liq", "Ca_2+"]
        ] == pytest.approx(2500, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].eq_flow_mol_phase_comp["Liq", "SO4_2-"]
        ] == pytest.approx(468.2, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].eq_flow_mol_phase_comp["Liq", "Na_+"]
        ] == pytest.approx(89.91, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].eq_flow_mol_phase_comp["Liq", "Cl_-"]
        ] == pytest.approx(49.75, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].eq_flow_mol_phase_comp["Liq", "Mg_2+"]
        ] == pytest.approx(717.4, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].eq_debye_huckel_constant
        ] == pytest.approx(64.32, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].eq_enth_mass_phase["Liq"]
        ] == pytest.approx(0.0000007982, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].eq_total_dissolved_solids
        ] == pytest.approx(0.0000278, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].eq_flow_vol_phase["Liq"]
        ] == pytest.approx(5489, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].eq_mole_frac_phase_comp["Liq", "H2O"]
        ] == pytest.approx(1, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].eq_mole_frac_phase_comp["Liq", "Ca_2+"]
        ] == pytest.approx(1000, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].eq_mole_frac_phase_comp["Liq", "SO4_2-"]
        ] == pytest.approx(1000, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].eq_mole_frac_phase_comp["Liq", "Na_+"]
        ] == pytest.approx(1000, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].eq_mole_frac_phase_comp["Liq", "Cl_-"]
        ] == pytest.approx(1000, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].eq_mole_frac_phase_comp["Liq", "Mg_2+"]
        ] == pytest.approx(1000, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].eq_pressure_osm_phase["Liq"]
        ] == pytest.approx(0.0000009278, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].eq_pressure_sat
        ] == pytest.approx(0.0003216, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].eq_visc_k_phase["Liq"]
        ] == pytest.approx(1000, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].eq_flow_equiv_phase_comp["Liq", "Ca_2+"]
        ] == pytest.approx(50, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].eq_flow_equiv_phase_comp["Liq", "SO4_2-"]
        ] == pytest.approx(50, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].eq_flow_equiv_phase_comp["Liq", "Na_+"]
        ] == pytest.approx(100, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].eq_flow_equiv_phase_comp["Liq", "Cl_-"]
        ] == pytest.approx(100, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].eq_flow_equiv_phase_comp["Liq", "Mg_2+"]
        ] == pytest.approx(50, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].eq_conc_equiv_phase_comp["Liq", "Ca_2+"]
        ] == pytest.approx(0.2, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].eq_conc_equiv_phase_comp["Liq", "SO4_2-"]
        ] == pytest.approx(0.048, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].eq_conc_equiv_phase_comp["Liq", "Na_+"]
        ] == pytest.approx(0.0023, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].eq_conc_equiv_phase_comp["Liq", "Cl_-"]
        ] == pytest.approx(0.0035, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].eq_conc_equiv_phase_comp["Liq", "Mg_2+"]
        ] == pytest.approx(0.012, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].eq_equiv_conductivity_phase["Liq"]
        ] == pytest.approx(1545, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].eq_elec_mobility_phase_comp["Liq", "Ca_2+"]
        ] == pytest.approx(16220000, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].eq_elec_mobility_phase_comp["Liq", "SO4_2-"]
        ] == pytest.approx(12120000, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].eq_elec_mobility_phase_comp["Liq", "Na_+"]
        ] == pytest.approx(19320000, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].eq_elec_mobility_phase_comp["Liq", "Cl_-"]
        ] == pytest.approx(12660000, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].eq_elec_mobility_phase_comp["Liq", "Mg_2+"]
        ] == pytest.approx(18200000, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].eq_elec_cond_phase["Liq"]
        ] == pytest.approx(3.225, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].eq_trans_num_phase_comp["Liq", "Ca_2+"]
        ] == pytest.approx(10, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].eq_trans_num_phase_comp["Liq", "SO4_2-"]
        ] == pytest.approx(10, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].eq_trans_num_phase_comp["Liq", "Na_+"]
        ] == pytest.approx(1.908, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].eq_trans_num_phase_comp["Liq", "Cl_-"]
        ] == pytest.approx(2.904, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].eq_trans_num_phase_comp["Liq", "Mg_2+"]
        ] == pytest.approx(9.956, rel=1e-3)
        assert m.fs.stream[0].scaling_factor[
            m.fs.stream[0].eq_total_hardness
        ] == pytest.approx(0.0001443, rel=1e-3)
