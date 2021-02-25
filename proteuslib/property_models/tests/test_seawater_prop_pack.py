import pytest
import proteuslib.property_models.seawater_prop_pack as props
from idaes.generic_models.properties.tests.test_harness import \
    PropertyTestHarness
from pyomo.environ import (ConcreteModel,
                           Constraint,
                           TerminationCondition,
                           SolverStatus,
                           value,
                           SolverFactory,
                           Suffix,
                           Param,
                           Expression,
                           Var)
from idaes.core import (FlowsheetBlock,
                        MaterialFlowBasis,
                        MaterialBalanceType,
                        EnergyBalanceType)
from idaes.core.components import Solute, Solvent
from idaes.core.phases import LiquidPhase
from idaes.core.util.model_statistics import (degrees_of_freedom,
                                              number_variables,
                                              number_total_constraints,
                                              fixed_variables_set,
                                              activated_constraints_set,
                                              unused_variables_set,
                                              number_unused_variables)
from idaes.core.util.testing import (get_default_solver,
                                     initialization_tester)
from idaes.core.util.scaling import (calculate_scaling_factors,
                                     set_scaling_factor,
                                     unscaled_variables_generator,
                                     unscaled_constraints_generator,
                                     badly_scaled_var_generator)
from pyomo.util.check_units import assert_units_consistent

# -----------------------------------------------------------------------------
# Get default solver for testing
solver = get_default_solver()
solver.options["nlp_scaling_method"] = "user-scaling"

# -----------------------------------------------------------------------------
@pytest.mark.unit
class TestBasicMix(PropertyTestHarness):
    def configure(self):
        self.prop_pack = props.SeawaterParameterBlock
        self.param_args = {}
        self.prop_args = {}
        self.has_density_terms = False

class TestSeawaterPropPack():
    @pytest.fixture(scope="class")
    def frame(self):
        m = ConcreteModel()
        m.fs = FlowsheetBlock(default={"dynamic": False})
        m.fs.properties = props.SeawaterParameterBlock()
        m.fs.stream = m.fs.properties.build_state_block([0], default={})
        
        # specify conditions
        mass_flow = 1
        x_TDS = 0.035
        m.fs.stream[0].flow_mass_phase_comp['Liq', 'TDS'].fix(x_TDS * mass_flow)
        m.fs.stream[0].flow_mass_phase_comp['Liq', 'H2O'].fix((1 - x_TDS) * mass_flow)
        m.fs.stream[0].temperature.fix(273.15 + 25)
        m.fs.stream[0].pressure.fix(101325)
        return m

    @pytest.mark.unit
    def test_components_phases(self, frame):
        m = frame

        # test components, package only supports H2O and TDS
        assert hasattr(m.fs.properties, 'component_list')
        assert len(m.fs.properties.component_list) == 2
        assert 'H2O' in m.fs.properties.component_list
        assert hasattr(m.fs.properties, 'H2O')
        assert isinstance(m.fs.properties.H2O, Solvent)
        assert 'TDS' in m.fs.properties.component_list
        assert hasattr(m.fs.properties, 'TDS')
        assert isinstance(m.fs.properties.TDS, Solute)

        # test phase, package only supports Liquid
        assert hasattr(m.fs.properties, 'phase_list')
        assert len(m.fs.properties.phase_list) == 1
        assert hasattr(m.fs.properties, 'Liq')
        assert isinstance(m.fs.properties.Liq, LiquidPhase)

    @pytest.mark.unit
    def test_parameters(self, frame):
        m = frame

        assert hasattr(m.fs.properties, 'mw_comp')
        assert isinstance(m.fs.properties.mw_comp, Param)

        dens_mass_param_end_list = ['A1', 'A2', 'A3', 'A4', 'A5',
                                    'B1', 'B2', 'B3', 'B4', 'B5']
        for e in dens_mass_param_end_list:
            assert hasattr(m.fs.properties, 'dens_mass_param_' + e)
            v = getattr(m.fs.properties, 'dens_mass_param_' + e)
            assert isinstance(v, Var)

        visc_d_param_end_list = ['muw_A', 'muw_B', 'muw_C', 'muw_D',
                                 'A_1', 'A_2', 'A_3', 'B_1', 'B_2', 'B_3']
        for e in visc_d_param_end_list:
            assert hasattr(m.fs.properties, 'visc_d_param_' + e)
            v = getattr(m.fs.properties, 'visc_d_param_' + e)
            assert isinstance(v, Var)

        osm_coeff_param_end_list = [str(i) for i in range(1, 10 + 1)]
        for e in osm_coeff_param_end_list:
            assert hasattr(m.fs.properties, 'osm_coeff_param_' + e)
            v = getattr(m.fs.properties, 'osm_coeff_param_' + e)
            assert isinstance(v, Var)

        enth_mass_param_end_list = ['A1', 'A2', 'A3', 'A4', 'B1', 'B2']
        for e in enth_mass_param_end_list:
            assert hasattr(m.fs.properties, 'enth_mass_param_' + e)
            v = getattr(m.fs.properties, 'enth_mass_param_' + e)
            assert isinstance(v, Var)

        # test that the parameter variables are fixed
        for v in m.fs.properties.component_objects(Var):
            assert v.is_fixed()

    @pytest.mark.unit
    def test_default_scaling(self, frame):
        m = frame

        assert hasattr(m.fs.properties, 'default_scaling_factor')
        default_scaling_var_dict = {('temperature', None): 1e-2,
                                    ('pressure', None): 1e-6,
                                    ('dens_mass_phase', 'Liq'): 1e-3,
                                    ('visc_d_phase', 'Liq'): 1e3,
                                    ('osm_coeff', None): 1e0,
                                    ('enth_mass_phase', 'Liq'): 1e-5}
        assert len(default_scaling_var_dict) == len(m.fs.properties.default_scaling_factor)
        for t, sf in default_scaling_var_dict.items():
            assert t in m.fs.properties.default_scaling_factor.keys()
            assert m.fs.properties.default_scaling_factor[t] == sf

    @pytest.mark.unit
    def test_metadata_exists(self, frame):
        m = frame
        assert hasattr(m.fs.properties, 'define_metadata')

    @pytest.mark.unit
    def test_build(self, frame):
        m = frame

        # test scaling factor
        assert hasattr(m.fs.stream[0], 'scaling_factor')
        assert isinstance(m.fs.stream[0].scaling_factor, Suffix)

        # test state variables
        state_vars_list = ['flow_mass_phase_comp', 'temperature', 'pressure']
        state_vars_dict = m.fs.stream[0].define_state_vars()
        assert len(state_vars_dict) == len(state_vars_list)
        for sv in state_vars_list:
            assert sv in state_vars_dict
            assert hasattr(m.fs.stream[0], sv)
            var = getattr(m.fs.stream[0], sv)
            assert isinstance(var, Var)

        # test on demand variables
        var_list = ['mass_frac_phase_comp', 'dens_mass_phase', 'flow_vol_phase', 'flow_vol',
                    'conc_mass_phase_comp', 'flow_mol_phase_comp', 'mole_frac_phase_comp',
                    'molality_comp', 'visc_d_phase', 'osm_coeff', 'pressure_osm',
                    'enth_mass_phase']
        for v in var_list:  # test that they are not built when not demanded
            assert not m.fs.stream[0].is_property_constructed(v)
        for v in var_list:  # test they are built on demand
            assert hasattr(m.fs.stream[0], v)
            var = getattr(m.fs.stream[0], v)
            assert isinstance(var, Var)

        # test on demand constraints
        for v in var_list:
            assert hasattr(m.fs.stream[0], '_' + v)  # check method
            assert hasattr(m.fs.stream[0], 'eq_' + v)  # check constraint
            c = getattr(m.fs.stream[0], 'eq_' + v)
            assert isinstance(c, Constraint)

        # test on demand expressions
        exp_list = ['enth_flow']
        for e in exp_list:  # test that they are not built when not demanded
            assert not m.fs.stream[0].is_property_constructed(e)
        for e in exp_list:  # test they are built on demand
            assert hasattr(m.fs.stream[0], e)
            exp = getattr(m.fs.stream[0], e)
            assert isinstance(exp, Expression)

    @pytest.mark.unit
    def test_statistics(self, frame):
        m = frame

        assert number_variables(m) == 56
        assert number_total_constraints(m) == 16
        assert number_unused_variables(m) == 1  # pressure is unused

    @pytest.mark.unit
    def test_general_methods(self, frame):
        m = frame

        assert hasattr(m.fs.stream[0], 'get_material_flow_terms')
        assert (m.fs.stream[0].get_material_flow_terms('Liq', 'H2O')
               is m.fs.stream[0].flow_mass_phase_comp['Liq', 'H2O'])
        assert (m.fs.stream[0].get_material_flow_terms('Liq', 'TDS')
                is m.fs.stream[0].flow_mass_phase_comp['Liq', 'TDS'])

        assert hasattr(m.fs.stream[0], 'get_enthalpy_flow_terms')
        assert (m.fs.stream[0].get_enthalpy_flow_terms(None)
                is m.fs.stream[0].enth_flow)

        assert hasattr(m.fs.stream[0], 'default_material_balance_type')
        assert (m.fs.stream[0].default_material_balance_type()
                is MaterialBalanceType.componentTotal)

        assert hasattr(m.fs.stream[0], 'default_energy_balance_type')
        assert (m.fs.stream[0].default_energy_balance_type()
                is EnergyBalanceType.enthalpyTotal)

        assert hasattr(m.fs.stream[0], 'get_material_flow_basis')
        assert (m.fs.stream[0].get_material_flow_basis()
                is MaterialFlowBasis.mass)

    @pytest.mark.unit
    def test_units(self, frame):
        m = frame

        # check all variables have assigned units
        for v in m.component_data_objects(Var, descend_into=True):
            assert v.get_units() is not None

        assert_units_consistent(m)

    @pytest.mark.unit
    def test_scaling(self, frame):
        m = frame

        set_scaling_factor(m.fs.stream[0].flow_mass_phase_comp['Liq', 'H2O'], 1)
        set_scaling_factor(m.fs.stream[0].flow_mass_phase_comp['Liq', 'TDS'], 1e2)
        calculate_scaling_factors(m.fs)

        # check that all variables have scaling factors
        unscaled_var_list = list(unscaled_variables_generator(m))
        assert len(unscaled_var_list) == 0
        # check that all constraints have been scaled
        unscaled_constraint_list = list(unscaled_constraints_generator(m))
        assert len(unscaled_constraint_list) == 0

        # check if any variables are badly scaled
        badly_scaled_var_list = list(badly_scaled_var_generator(m))
        assert len(badly_scaled_var_list) == 0

    @pytest.mark.unit
    def test_dof(self,frame):
        m = frame
        assert (degrees_of_freedom(m) == 0)

    @pytest.mark.component
    def test_solve(self, frame):
        m = frame

        results = solver.solve(m)

        # Check for optimal solution
        assert results.solver.termination_condition == \
               TerminationCondition.optimal
        assert results.solver.status == SolverStatus.ok

    @pytest.mark.component
    def test_solution(self, frame):
        m = frame

        # specified conditions
        assert pytest.approx(0.035, rel=1e-3) == value(m.fs.stream[0].flow_mass_phase_comp['Liq', 'TDS'])
        assert pytest.approx(0.965, rel=1e-3) == value(m.fs.stream[0].flow_mass_phase_comp['Liq', 'H2O'])
        assert pytest.approx(273.15 + 25, rel=1e-3) == value(m.fs.stream[0].temperature)
        assert pytest.approx(101325, rel=1e-3) == value(m.fs.stream[0].pressure)

        # calculated properties
        assert pytest.approx(0.035, rel=1e-3) == value(m.fs.stream[0].mass_frac_phase_comp['Liq', 'TDS'])
        assert pytest.approx(0.965, rel=1e-3) == value(m.fs.stream[0].mass_frac_phase_comp['Liq', 'H2O'])
        assert pytest.approx(1023.6, rel=1e-3) == value(m.fs.stream[0].dens_mass_phase['Liq'])
        assert pytest.approx(9.770e-4, rel=1e-3) == value(m.fs.stream[0].flow_vol_phase['Liq'])
        assert pytest.approx(9.770e-4, rel=1e-3) == value(m.fs.stream[0].flow_vol)
        assert pytest.approx(987.7, rel=1e-3) == value(m.fs.stream[0].conc_mass_phase_comp['Liq', 'H2O'])
        assert pytest.approx(35.82, rel=1e-3) == value(m.fs.stream[0].conc_mass_phase_comp['Liq', 'TDS'])
        assert pytest.approx(53.57, rel=1e-3) == value(m.fs.stream[0].flow_mol_phase_comp['Liq', 'H2O'])
        assert pytest.approx(0.5989, rel=1e-3) == value(m.fs.stream[0].flow_mol_phase_comp['Liq', 'TDS'])
        assert pytest.approx(0.9889, rel=1e-3) == value(m.fs.stream[0].mole_frac_phase_comp['Liq', 'H2O'])
        assert pytest.approx(1.106e-2, rel=1e-3) == value(m.fs.stream[0].mole_frac_phase_comp['Liq', 'TDS'])
        assert pytest.approx(0.6206, rel=1e-3) == value(m.fs.stream[0].molality_comp['TDS'])
        assert pytest.approx(9.588e-4, rel=1e-3) == value(m.fs.stream[0].visc_d_phase['Liq'])
        assert pytest.approx(0.9068, rel=1e-3) == value(m.fs.stream[0].osm_coeff)
        assert pytest.approx(2.790e6, rel=1e-3) == value(m.fs.stream[0].pressure_osm)
        assert pytest.approx(9.974e4, rel=1e-3) == value(m.fs.stream[0].enth_mass_phase['Liq'])
        assert pytest.approx(9.974e4, rel=1e-3) == value(m.fs.stream[0].enth_flow)
