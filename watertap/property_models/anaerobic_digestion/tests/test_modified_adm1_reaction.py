#################################################################################
# WaterTAP Copyright (c) 2020-2023, The Regents of the University of California,
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
Tests for ADM1 reaction package.

Verified against results from:

X. Flores-Alsina, K. Solon, C.K. Mbamba, S. Tait, K.V. Gernaey, U. Jeppsson, D.J. Batstone,
Modelling phosphorus (P), sulfur (S) and iron (Fe) interactions for dynamic simulations of anaerobic digestion processes,
Water Research. 95 (2016) 370-382. https://www.sciencedirect.com/science/article/pii/S0043135416301397

Authors: Chenyu Wang, Marcus Holly
"""

import pytest

from pyomo.environ import (
    check_optimal_termination,
    ConcreteModel,
    Constraint,
    units,
    value,
    Var,
    Param,
    Objective,
    SolverFactory,
)
from pyomo.util.check_units import assert_units_consistent
from idaes.core import FlowsheetBlock
from watertap.unit_models.anaerobic_digestor import AD
from idaes.core import MaterialFlowBasis
from idaes.core.solvers import get_solver
import idaes.logger as idaeslog
import idaes.core.util.scaling as iscale
from idaes.core.util.model_statistics import degrees_of_freedom, large_residuals_set
from idaes.core.util.model_diagnostics import DegeneracyHunter

from watertap.property_models.anaerobic_digestion.modified_adm1_properties import (
    ModifiedADM1ParameterBlock,
)
from watertap.property_models.anaerobic_digestion.adm1_properties_vapor import (
    ADM1_vaporParameterBlock,
)
from watertap.property_models.anaerobic_digestion.modified_adm1_reactions import (
    ModifiedADM1ReactionParameterBlock,
    ModifiedADM1ReactionBlock,
)
from watertap.core.util.model_diagnostics.infeasible import *
from idaes.core.util.testing import initialization_tester

# -----------------------------------------------------------------------------
# Get default solver for testing
solver = get_solver()

# TODO: Fix and unfix all variables and run at max_iter:0 to see which constraints fail using large_residual_set
# TODO: Double check the values for X_PP, X_fa, and X_c4 in the Petersen matrix and reaction rates (double check signs)
# TODO: Play around with scaling in property file


class TestParamBlock(object):
    @pytest.fixture(scope="class")
    def model(self):
        model = ConcreteModel()
        model.pparams = ModifiedADM1ParameterBlock()
        model.rparams = ModifiedADM1ReactionParameterBlock(
            property_package=model.pparams
        )

        return model

    @pytest.mark.unit
    def test_build(self, model):
        assert model.rparams.reaction_block_class is ModifiedADM1ReactionBlock

        assert len(model.rparams.rate_reaction_idx) == 25
        for i in model.rparams.rate_reaction_idx:
            assert i in [
                "R1",
                "R2",
                "R3",
                "R4",
                "R5",
                "R6",
                "R7",
                "R8",
                "R9",
                "R10",
                "R11",
                "R12",
                "R13",
                "R14",
                "R15",
                "R16",
                "R17",
                "R18",
                "R19",
                "R20",
                "R21",
                "R22",
                "R23",
                "R24",
                "R25",
            ]

        # Expected non-zero stoichiometries
        # Values from table 11 in reference

        mw_n = 14
        mw_c = 12
        mw_p = 31

        stoic = {
            # R1: Hydrolysis of carbohydrates
            ("R1", "Liq", "S_su"): 1,
            ("R1", "Liq", "X_ch"): -1,
            # R2:  Hydrolysis of proteins
            ("R2", "Liq", "S_aa"): 1,
            ("R2", "Liq", "X_pr"): -1,
            # R3:  Hydrolysis of lipids
            ("R3", "Liq", "S_su"): 0.05,
            ("R3", "Liq", "S_fa"): 0.95,
            ("R3", "Liq", "X_li"): -1,
            ("R3", "Liq", "S_IC"): 0.000029 * mw_c,
            ("R3", "Liq", "S_IP"): 0.0003441 * mw_p,
            # R4:  Uptake of sugars
            ("R4", "Liq", "S_su"): -1,
            ("R4", "Liq", "S_bu"): 0.1195,
            ("R4", "Liq", "S_pro"): 0.2422,
            ("R4", "Liq", "S_ac"): 0.3668,
            ("R4", "Liq", "S_h2"): 0.1715,
            ("R4", "Liq", "S_IC"): 0.00726 * mw_c,
            ("R4", "Liq", "S_IN"): -0.000615 * mw_n,
            ("R4", "Liq", "S_IP"): -0.000069 * mw_p,
            ("R4", "Liq", "X_su"): 0.10,
            # R5:  Uptake of amino acids
            ("R5", "Liq", "S_aa"): -1,
            ("R5", "Liq", "S_va"): 0.2116,
            ("R5", "Liq", "S_bu"): 0.2392,
            ("R5", "Liq", "S_pro"): 0.0460,
            ("R5", "Liq", "S_ac"): 0.3680,
            ("R5", "Liq", "S_h2"): 0.0552,
            ("R5", "Liq", "S_IC"): 0.00450 * mw_c,
            ("R5", "Liq", "S_IN"): 0.00741 * mw_n,
            ("R5", "Liq", "S_IP"): -0.0000556 * mw_p,
            ("R5", "Liq", "X_aa"): 0.08,
            # R6:  Uptake of long chain fatty acids (LCFAs)
            ("R6", "Liq", "S_fa"): -1,
            ("R6", "Liq", "S_ac"): 0.6580,
            ("R6", "Liq", "S_h2"): 0.2820,
            ("R6", "Liq", "S_IC"): -0.000989 * mw_c,
            ("R6", "Liq", "S_IN"): -0.000369 * mw_n,
            ("R6", "Liq", "S_IP"): -0.0000417 * mw_p,
            ("R6", "Liq", "X_fa"): 0.06,
            # R7:  Uptake of valerate
            ("R7", "Liq", "S_va"): -1,
            ("R7", "Liq", "S_pro"): 0.5076,
            ("R7", "Liq", "S_ac"): 0.2914,
            ("R7", "Liq", "S_h2"): 0.1410,
            ("R7", "Liq", "S_IC"): -0.000495 * mw_c,
            ("R7", "Liq", "S_IN"): -0.000369 * mw_n,
            ("R7", "Liq", "S_IP"): -0.0000417 * mw_p,
            ("R7", "Liq", "X_c4"): 0.06,
            # R8:  Uptake of butyrate
            ("R8", "Liq", "S_bu"): -1,
            ("R8", "Liq", "S_ac"): 0.7520,
            ("R8", "Liq", "S_h2"): 0.1880,
            ("R8", "Liq", "S_IC"): -0.000331 * mw_c,
            ("R8", "Liq", "S_IN"): -0.000369 * mw_n,
            ("R8", "Liq", "S_IP"): -0.0000417 * mw_p,
            ("R8", "Liq", "X_c4"): 0.06,
            # R9: Uptake of propionate
            ("R9", "Liq", "S_pro"): -1,
            ("R9", "Liq", "S_ac"): 0.5472,
            ("R9", "Liq", "S_h2"): 0.4128,
            ("R9", "Liq", "S_IC"): 0.008465 * mw_c,
            ("R9", "Liq", "S_IN"): -0.000246 * mw_n,
            ("R9", "Liq", "S_IP"): -0.0000278 * mw_p,
            ("R9", "Liq", "X_pro"): 0.04,
            # R10: Uptake of acetate
            ("R10", "Liq", "S_ac"): -1,
            ("R10", "Liq", "S_ch4"): 0.95,
            ("R10", "Liq", "S_IC"): 0.014881 * mw_c,
            ("R10", "Liq", "S_IN"): -0.000308 * mw_n,
            ("R10", "Liq", "S_IP"): -0.0000347 * mw_p,
            ("R10", "Liq", "X_ac"): 0.05,
            # R11: Uptake of hydrogen
            ("R11", "Liq", "S_h2"): -1,
            ("R11", "Liq", "S_ch4"): 0.94,
            ("R11", "Liq", "S_IC"): -0.016518 * mw_c,
            ("R11", "Liq", "S_IN"): -0.000369 * mw_n,
            ("R11", "Liq", "S_IP"): -0.0000417 * mw_p,
            ("R11", "Liq", "X_h2"): 0.06,
            # R12: Decay of X_su
            ("R12", "Liq", "S_IC"): 0.002773 * mw_c,
            ("R12", "Liq", "S_IN"): 0.003551 * mw_n,
            ("R12", "Liq", "S_IP"): 0.0005534 * mw_p,
            ("R12", "Liq", "X_ch"): 0.275,
            ("R12", "Liq", "X_pr"): 0.275,
            ("R12", "Liq", "X_li"): 0.35,
            ("R12", "Liq", "X_su"): -1,
            ("R12", "Liq", "X_I"): 0.1,
            # R13: Decay of X_aa
            ("R13", "Liq", "S_IC"): 0.002773 * mw_c,
            ("R13", "Liq", "S_IN"): 0.003551 * mw_n,
            ("R13", "Liq", "S_IP"): 0.0005534 * mw_p,
            ("R13", "Liq", "X_ch"): 0.275,
            ("R13", "Liq", "X_pr"): 0.275,
            ("R13", "Liq", "X_li"): 0.35,
            ("R13", "Liq", "X_aa"): -1,
            ("R13", "Liq", "X_I"): 0.1,
            # R14: Decay of X_fa
            ("R14", "Liq", "S_IC"): 0.002773 * mw_c,
            ("R14", "Liq", "S_IN"): 0.003551 * mw_n,
            ("R14", "Liq", "S_IP"): 0.0005534 * mw_p,
            ("R14", "Liq", "X_ch"): 0.275,
            ("R14", "Liq", "X_pr"): 0.275,
            ("R14", "Liq", "X_li"): 0.35,
            ("R14", "Liq", "X_fa"): -1,
            ("R14", "Liq", "X_I"): 0.1,
            # R15: Decay of X_c4
            ("R15", "Liq", "S_IC"): 0.002773 * mw_c,
            ("R15", "Liq", "S_IN"): 0.003551 * mw_n,
            ("R15", "Liq", "S_IP"): 0.0005534 * mw_p,
            ("R15", "Liq", "X_ch"): 0.275,
            ("R15", "Liq", "X_pr"): 0.275,
            ("R15", "Liq", "X_li"): 0.35,
            ("R15", "Liq", "X_c4"): -1,
            ("R15", "Liq", "X_I"): 0.1,
            # R16: Decay of X_pro
            ("R16", "Liq", "S_IC"): 0.002773 * mw_c,
            ("R16", "Liq", "S_IN"): 0.003551 * mw_n,
            ("R16", "Liq", "S_IP"): 0.0005534 * mw_p,
            ("R16", "Liq", "X_ch"): 0.275,
            ("R16", "Liq", "X_pr"): 0.275,
            ("R16", "Liq", "X_li"): 0.35,
            ("R16", "Liq", "X_pro"): -1,
            ("R16", "Liq", "X_I"): 0.1,
            # R17: Decay of X_ac
            ("R17", "Liq", "S_IC"): 0.002773 * mw_c,
            ("R17", "Liq", "S_IN"): 0.003551 * mw_n,
            ("R17", "Liq", "S_IP"): 0.0005534 * mw_p,
            ("R17", "Liq", "X_ch"): 0.275,
            ("R17", "Liq", "X_pr"): 0.275,
            ("R17", "Liq", "X_li"): 0.35,
            ("R17", "Liq", "X_ac"): -1,
            ("R17", "Liq", "X_I"): 0.1,
            # R18: Decay of X_h2
            ("R18", "Liq", "S_IC"): 0.002773 * mw_c,
            ("R18", "Liq", "S_IN"): 0.003551 * mw_n,
            ("R18", "Liq", "S_IP"): 0.0005534 * mw_p,
            ("R18", "Liq", "X_ch"): 0.275,
            ("R18", "Liq", "X_pr"): 0.275,
            ("R18", "Liq", "X_li"): 0.35,
            ("R18", "Liq", "X_h2"): -1,
            ("R18", "Liq", "X_I"): 0.1,
            # R19: Storage of S_va in X_PHA
            ("R19", "Liq", "S_va"): -1,
            ("R19", "Liq", "S_IC"): -0.000962 * mw_c,
            ("R19", "Liq", "S_IP"): 0.012903 * mw_p,
            ("R19", "Liq", "X_PHA"): 1,
            ("R19", "Liq", "X_PP"): -0.012903,
            ("R19", "Liq", "S_K"): 0.004301,
            ("R19", "Liq", "S_Mg"): 0.004301,
            # R20: Storage of S_bu in X_PHA
            ("R20", "Liq", "S_bu"): -1,
            ("R20", "Liq", "S_IP"): 0.012903 * mw_p,
            ("R20", "Liq", "X_PHA"): 1,
            ("R20", "Liq", "X_PP"): -0.012903,
            ("R20", "Liq", "S_K"): 0.004301,
            ("R20", "Liq", "S_Mg"): 0.004301,
            # R21: Storage of S_pro in X_PHA
            ("R21", "Liq", "S_pro"): -1,
            ("R21", "Liq", "S_IC"): 0.001786 * mw_c,
            ("R21", "Liq", "S_IP"): 0.012903 * mw_p,
            ("R21", "Liq", "X_PHA"): 1,
            ("R21", "Liq", "X_PP"): -0.012903,
            ("R21", "Liq", "S_K"): 0.004301,
            ("R21", "Liq", "S_Mg"): 0.004301,
            # R22: Storage of S_ac in X_PHA
            ("R22", "Liq", "S_ac"): -1,
            ("R22", "Liq", "S_IC"): 0.006250 * mw_c,
            ("R22", "Liq", "S_IP"): 0.012903 * mw_p,
            ("R22", "Liq", "X_PHA"): 1,
            ("R22", "Liq", "X_PP"): -0.012903,
            ("R22", "Liq", "S_K"): 0.004301,
            ("R22", "Liq", "S_Mg"): 0.004301,
            # R23: Lysis of X_PAO
            ("R23", "Liq", "S_IC"): 0.002773 * mw_c,
            ("R23", "Liq", "S_IN"): 0.003551 * mw_n,
            ("R23", "Liq", "S_IP"): 0.000553 * mw_p,
            ("R23", "Liq", "X_ch"): 0.275,
            ("R23", "Liq", "X_pr"): 0.275,
            ("R23", "Liq", "X_li"): 0.35,
            ("R23", "Liq", "X_I"): 0.1,
            ("R23", "Liq", "X_PAO"): -1,
            # R24: Lysis of X_PP
            ("R24", "Liq", "S_IP"): 1 * mw_p,
            ("R24", "Liq", "X_PP"): -1,
            ("R24", "Liq", "S_K"): 1 / 3,
            ("R24", "Liq", "S_Mg"): 1 / 3,
            # R25: Lysis of X_PAO
            ("R25", "Liq", "S_va"): 0.1,
            ("R25", "Liq", "S_bu"): 0.1,
            ("R25", "Liq", "S_pro"): 0.4,
            ("R25", "Liq", "S_ac"): 0.4,
            ("R25", "Liq", "S_IC"): -0.003118 * mw_c,
            ("R25", "Liq", "X_PHA"): -1,
        }

        assert len(model.rparams.rate_reaction_stoichiometry) == 25 * 32
        for i, v in model.rparams.rate_reaction_stoichiometry.items():

            assert i[0] in [
                "R1",
                "R2",
                "R3",
                "R4",
                "R5",
                "R6",
                "R7",
                "R8",
                "R9",
                "R10",
                "R11",
                "R12",
                "R13",
                "R14",
                "R15",
                "R16",
                "R17",
                "R18",
                "R19",
                "R20",
                "R21",
                "R22",
                "R23",
                "R24",
                "R25",
            ]
            if i in stoic:
                assert pytest.approx(stoic[i], rel=1e-2) == value(v)
            else:
                assert pytest.approx(value(v), rel=1e-2) == 0

        assert isinstance(model.rparams.Z_h2s, Param)
        assert value(model.rparams.Z_h2s) == 0
        assert isinstance(model.rparams.f_sI_xc, Var)
        assert value(model.rparams.f_sI_xc) == 0.1
        assert isinstance(model.rparams.f_xI_xc, Var)
        assert value(model.rparams.f_xI_xc) == 0.20
        assert isinstance(model.rparams.f_ch_xc, Var)
        assert value(model.rparams.f_ch_xc) == 0.2
        assert isinstance(model.rparams.f_pr_xc, Var)
        assert value(model.rparams.f_pr_xc) == 0.2
        assert isinstance(model.rparams.f_li_xc, Var)
        assert value(model.rparams.f_li_xc) == 0.30

        assert isinstance(model.rparams.f_fa_li, Var)
        assert value(model.rparams.f_fa_li) == 0.95
        assert isinstance(model.rparams.f_h2_su, Var)
        assert value(model.rparams.f_h2_su) == 0.1906
        assert isinstance(model.rparams.f_bu_su, Var)
        assert value(model.rparams.f_bu_su) == 0.1328
        assert isinstance(model.rparams.f_pro_su, Var)
        assert value(model.rparams.f_pro_su) == 0.2691
        assert isinstance(model.rparams.f_ac_su, Var)
        assert value(model.rparams.f_ac_su) == 0.4076
        assert isinstance(model.rparams.f_h2_aa, Var)
        assert value(model.rparams.f_h2_aa) == 0.06
        assert isinstance(model.rparams.f_va_aa, Var)
        assert value(model.rparams.f_va_aa) == 0.23
        assert isinstance(model.rparams.f_bu_aa, Var)
        assert value(model.rparams.f_bu_aa) == 0.26
        assert isinstance(model.rparams.f_pro_aa, Var)
        assert value(model.rparams.f_pro_aa) == 0.05
        assert isinstance(model.rparams.f_ac_aa, Var)
        assert value(model.rparams.f_ac_aa) == 0.40

        assert isinstance(model.rparams.Y_su, Var)
        assert value(model.rparams.Y_su) == 0.1
        assert isinstance(model.rparams.Y_aa, Var)
        assert value(model.rparams.Y_aa) == 0.08
        assert isinstance(model.rparams.Y_fa, Var)
        assert value(model.rparams.Y_fa) == 0.06
        assert isinstance(model.rparams.Y_c4, Var)
        assert value(model.rparams.Y_c4) == 0.06
        assert isinstance(model.rparams.Y_pro, Var)
        assert value(model.rparams.Y_pro) == 0.04
        assert isinstance(model.rparams.Y_ac, Var)
        assert value(model.rparams.Y_ac) == 0.05
        assert isinstance(model.rparams.Y_h2, Var)
        assert value(model.rparams.Y_h2) == 0.06

        assert isinstance(model.rparams.k_hyd_ch, Var)
        assert value(model.rparams.k_hyd_ch) == 10
        assert isinstance(model.rparams.k_hyd_pr, Var)
        assert value(model.rparams.k_hyd_pr) == 10
        assert isinstance(model.rparams.k_hyd_li, Var)
        assert value(model.rparams.k_hyd_li) == 10
        assert isinstance(model.rparams.K_S_IN, Var)
        assert value(model.rparams.K_S_IN) == 1e-4
        assert isinstance(model.rparams.k_m_su, Var)
        assert value(model.rparams.k_m_su) == 30
        assert isinstance(model.rparams.K_S_su, Var)
        assert value(model.rparams.K_S_su) == 0.5

        assert isinstance(model.rparams.pH_UL_aa, Var)
        assert value(model.rparams.pH_UL_aa) == 5.5
        assert isinstance(model.rparams.pH_LL_aa, Var)
        assert value(model.rparams.pH_LL_aa) == 4

        assert isinstance(model.rparams.k_m_aa, Var)
        assert value(model.rparams.k_m_aa) == 50
        assert isinstance(model.rparams.K_S_aa, Var)
        assert value(model.rparams.K_S_aa) == 0.3
        assert isinstance(model.rparams.k_m_fa, Var)
        assert value(model.rparams.k_m_fa) == 6
        assert isinstance(model.rparams.K_S_fa, Var)
        assert value(model.rparams.K_S_fa) == 0.4
        assert isinstance(model.rparams.K_I_h2_fa, Var)
        assert value(model.rparams.K_I_h2_fa) == 5e-6
        assert isinstance(model.rparams.k_m_c4, Var)
        assert value(model.rparams.k_m_c4) == 20
        assert isinstance(model.rparams.K_S_c4, Var)
        assert value(model.rparams.K_S_c4) == 0.2
        assert isinstance(model.rparams.K_I_h2_c4, Var)
        assert value(model.rparams.K_I_h2_c4) == 1e-5
        assert isinstance(model.rparams.k_m_pro, Var)
        assert value(model.rparams.k_m_pro) == 13
        assert isinstance(model.rparams.K_S_pro, Var)
        assert value(model.rparams.K_S_pro) == 0.1
        assert isinstance(model.rparams.K_I_h2_pro, Var)
        assert value(model.rparams.K_I_h2_pro) == 3.5e-6
        assert isinstance(model.rparams.k_m_ac, Var)
        assert value(model.rparams.k_m_ac) == 8
        assert isinstance(model.rparams.K_S_ac, Var)
        assert value(model.rparams.K_S_ac) == 0.15
        assert isinstance(model.rparams.K_I_nh3, Var)
        assert value(model.rparams.K_I_nh3) == 0.0018

        assert isinstance(model.rparams.pH_UL_ac, Var)
        assert value(model.rparams.pH_UL_ac) == 7
        assert isinstance(model.rparams.pH_LL_ac, Var)
        assert value(model.rparams.pH_LL_ac) == 6

        assert isinstance(model.rparams.k_m_h2, Var)
        assert value(model.rparams.k_m_h2) == 35
        assert isinstance(model.rparams.K_S_h2, Var)
        assert value(model.rparams.K_S_h2) == 7e-6

        assert isinstance(model.rparams.pH_UL_h2, Var)
        assert value(model.rparams.pH_UL_h2) == 6
        assert isinstance(model.rparams.pH_LL_h2, Var)
        assert value(model.rparams.pH_LL_h2) == 5

        assert isinstance(model.rparams.k_dec_X_su, Var)
        assert value(model.rparams.k_dec_X_su) == 0.02
        assert isinstance(model.rparams.k_dec_X_aa, Var)
        assert value(model.rparams.k_dec_X_aa) == 0.02
        assert isinstance(model.rparams.k_dec_X_fa, Var)
        assert value(model.rparams.k_dec_X_fa) == 0.02
        assert isinstance(model.rparams.k_dec_X_c4, Var)
        assert value(model.rparams.k_dec_X_c4) == 0.02
        assert isinstance(model.rparams.k_dec_X_pro, Var)
        assert value(model.rparams.k_dec_X_pro) == 0.02
        assert isinstance(model.rparams.k_dec_X_ac, Var)
        assert value(model.rparams.k_dec_X_ac) == 0.02
        assert isinstance(model.rparams.k_dec_X_h2, Var)
        assert value(model.rparams.k_dec_X_h2) == 0.02

        assert isinstance(model.rparams.K_a_va, Var)
        assert value(model.rparams.K_a_va) == 1.38e-5
        assert isinstance(model.rparams.K_a_bu, Var)
        assert value(model.rparams.K_a_bu) == 1.5e-5
        assert isinstance(model.rparams.K_a_pro, Var)
        assert value(model.rparams.K_a_pro) == 1.32e-5
        assert isinstance(model.rparams.K_a_ac, Var)
        assert value(model.rparams.K_a_ac) == 1.74e-5


class TestReactionBlock(object):
    @pytest.fixture(scope="class")
    def model(self):
        model = ConcreteModel()
        model.pparams = ModifiedADM1ParameterBlock()
        model.vparams = ADM1_vaporParameterBlock()
        model.rparams = ModifiedADM1ReactionParameterBlock(
            property_package=model.pparams
        )

        model.props = model.pparams.build_state_block([1])
        model.props_vap = model.vparams.build_state_block([1])
        model.rxns = model.rparams.build_reaction_block([1], state_block=model.props)

        return model

    @pytest.mark.unit
    def test_build(self, model):
        assert model.rxns[1].conc_mass_comp_ref is model.props[1].conc_mass_comp

    @pytest.mark.unit
    def test_rxn_rate(self, model):
        assert isinstance(model.rxns[1].reaction_rate, Var)
        assert len(model.rxns[1].reaction_rate) == 25
        assert isinstance(model.rxns[1].rate_expression, Constraint)
        assert len(model.rxns[1].rate_expression) == 25

    @pytest.mark.unit
    def test_get_reaction_rate_basis(self, model):
        assert model.rxns[1].get_reaction_rate_basis() == MaterialFlowBasis.mass

    @pytest.mark.component
    def test_initialize(self, model):
        assert model.rxns.initialize() is None

    @pytest.mark.component
    def check_units(self, model):
        assert_units_consistent(model)


class TestReactor:
    @pytest.fixture(scope="class")
    def model(self):
        m = ConcreteModel()

        m.fs = FlowsheetBlock(dynamic=False)

        m.fs.props = ModifiedADM1ParameterBlock()
        m.fs.props_vap = ADM1_vaporParameterBlock()
        m.fs.rxn_props = ModifiedADM1ReactionParameterBlock(property_package=m.fs.props)

        m.fs.unit = AD(
            liquid_property_package=m.fs.props,
            vapor_property_package=m.fs.props_vap,
            reaction_package=m.fs.rxn_props,
            has_heat_transfer=True,
            has_pressure_change=False,
        )

        # Feed conditions based on mass balance in Flores-Alsina, where 0 terms are expressed as 1e-9
        m.fs.unit.inlet.flow_vol.fix(0.001967593)  # Double check this value
        m.fs.unit.inlet.temperature.fix(308.15)
        m.fs.unit.inlet.pressure.fix(101325)

        m.fs.unit.inlet.conc_mass_comp[0, "S_su"].fix(0.034597)
        m.fs.unit.inlet.conc_mass_comp[0, "S_aa"].fix(0.015037)
        m.fs.unit.inlet.conc_mass_comp[0, "S_fa"].fix(1e-6)
        m.fs.unit.inlet.conc_mass_comp[0, "S_va"].fix(1e-6)
        m.fs.unit.inlet.conc_mass_comp[0, "S_bu"].fix(1e-6)
        m.fs.unit.inlet.conc_mass_comp[0, "S_pro"].fix(1e-6)
        m.fs.unit.inlet.conc_mass_comp[0, "S_ac"].fix(0.025072)
        m.fs.unit.inlet.conc_mass_comp[0, "S_h2"].fix(1e-6)
        m.fs.unit.inlet.conc_mass_comp[0, "S_ch4"].fix(1e-6)
        m.fs.unit.inlet.conc_mass_comp[0, "S_IC"].fix(0.34628)
        m.fs.unit.inlet.conc_mass_comp[0, "S_IN"].fix(0.60014)
        m.fs.unit.inlet.conc_mass_comp[0, "S_IP"].fix(0.22677)
        m.fs.unit.inlet.conc_mass_comp[0, "S_I"].fix(0.026599)

        m.fs.unit.inlet.conc_mass_comp[0, "X_ch"].fix(7.3687)
        m.fs.unit.inlet.conc_mass_comp[0, "X_pr"].fix(7.7308)
        m.fs.unit.inlet.conc_mass_comp[0, "X_li"].fix(10.3288)
        m.fs.unit.inlet.conc_mass_comp[0, "X_su"].fix(1e-6)
        m.fs.unit.inlet.conc_mass_comp[0, "X_aa"].fix(1e-6)
        m.fs.unit.inlet.conc_mass_comp[0, "X_fa"].fix(1e-6)
        m.fs.unit.inlet.conc_mass_comp[0, "X_c4"].fix(1e-6)
        m.fs.unit.inlet.conc_mass_comp[0, "X_pro"].fix(1e-6)
        m.fs.unit.inlet.conc_mass_comp[0, "X_ac"].fix(1e-6)
        m.fs.unit.inlet.conc_mass_comp[0, "X_h2"].fix(1e-6)
        m.fs.unit.inlet.conc_mass_comp[0, "X_I"].fix(12.7727)
        m.fs.unit.inlet.conc_mass_comp[0, "X_PHA"].fix(0.0022493)
        m.fs.unit.inlet.conc_mass_comp[0, "X_PP"].fix(1.04110)
        m.fs.unit.inlet.conc_mass_comp[0, "X_PAO"].fix(3.4655)
        m.fs.unit.inlet.conc_mass_comp[0, "S_K"].fix(0.02268)
        m.fs.unit.inlet.conc_mass_comp[0, "S_Mg"].fix(0.02893)

        m.fs.unit.inlet.cations[0].fix(0.04)  # Double check this value
        m.fs.unit.inlet.anions[0].fix(0.02)  # Double check this value

        m.fs.unit.volume_liquid.fix(3400)
        m.fs.unit.volume_vapor.fix(300)
        m.fs.unit.liquid_outlet.temperature.fix(308.15)

        return m

    @pytest.mark.component
    def test_dof(self, model):
        assert degrees_of_freedom(model) == 0

    @pytest.mark.component
    def test_unit_consistency(self, model):
        assert_units_consistent(model) == 0

    @pytest.mark.component
    def test_scaling_factors(self, model):
        m = model
        iscale.calculate_scaling_factors(m)

        # check that all variables have scaling factors
        unscaled_var_list = list(iscale.unscaled_variables_generator(m))
        assert len(unscaled_var_list) == 0

        # TODO: resolving "badly scaled vars" in this case doesn't help resolve; revisit scaling
        # for _ in iscale.badly_scaled_var_generator(m):
        #     assert False

    @pytest.mark.solver
    @pytest.mark.skipif(solver is None, reason="Solver not available")
    @pytest.mark.component
    def test_initialize(self, model):
        initialization_tester(model)

    @pytest.mark.solver
    @pytest.mark.skipif(solver is None, reason="Solver not available")
    @pytest.mark.component
    def test_solve(self, model):
        solver = get_solver(options={"bound_push": 1e-8})
        results = solver.solve(model, tee=True)
        model.display()
        print(large_residuals_set(model))

        assert check_optimal_termination(results)

    # TO DO: retest after conversion changes
    @pytest.mark.component
    def test_solution(self, model):
        assert value(model.fs.unit.liquid_outlet.flow_vol[0]) == pytest.approx(
            0.0019675, rel=1e-2
        )
        assert value(model.fs.unit.liquid_outlet.temperature[0]) == pytest.approx(
            308.15, rel=1e-2
        )
        assert value(model.fs.unit.liquid_outlet.pressure[0]) == pytest.approx(
            101325, rel=1e-2
        )
        assert value(
            model.fs.unit.liquid_outlet.conc_mass_comp[0, "S_su"]
        ) == pytest.approx(8.7441, rel=1e-2)
        assert value(
            model.fs.unit.liquid_outlet.conc_mass_comp[0, "S_aa"]
        ) == pytest.approx(5.31e-3, rel=1e-2)
        assert value(
            model.fs.unit.liquid_outlet.conc_mass_comp[0, "S_fa"]
        ) == pytest.approx(10.7452, rel=1e-2)
        assert value(
            model.fs.unit.liquid_outlet.conc_mass_comp[0, "S_va"]
        ) == pytest.approx(0.015746, rel=1e-2)
        assert value(
            model.fs.unit.liquid_outlet.conc_mass_comp[0, "S_bu"]
        ) == pytest.approx(0.016398, rel=1e-2)
        assert value(
            model.fs.unit.liquid_outlet.conc_mass_comp[0, "S_pro"]
        ) == pytest.approx(0.035679, rel=1e-2)
        assert value(
            model.fs.unit.liquid_outlet.conc_mass_comp[0, "S_ac"]
        ) == pytest.approx(0.043118, rel=1e-2)
        assert value(
            model.fs.unit.liquid_outlet.conc_mass_comp[0, "S_h2"]
        ) == pytest.approx(0.0112896, rel=1e-2)
        assert value(
            model.fs.unit.liquid_outlet.conc_mass_comp[0, "S_ch4"]
        ) == pytest.approx(2.72537e-7, rel=1e-2)
        assert value(
            model.fs.unit.liquid_outlet.conc_mass_comp[0, "S_IC"]
        ) == pytest.approx(0.093719 * 12, rel=1e-2)
        assert value(
            model.fs.unit.liquid_outlet.conc_mass_comp[0, "S_IN"]
        ) == pytest.approx(0.11650 * 14, rel=1e-2)
        assert value(
            model.fs.unit.liquid_outlet.conc_mass_comp[0, "S_IP"]
        ) == pytest.approx(29.11478, rel=1e-2)
        assert value(
            model.fs.unit.liquid_outlet.conc_mass_comp[0, "S_I"]
        ) == pytest.approx(0.02660, rel=1e-2)
        assert value(
            model.fs.unit.liquid_outlet.conc_mass_comp[0, "X_ch"]
        ) == pytest.approx(0.0407196, rel=1e-2)
        assert value(
            model.fs.unit.liquid_outlet.conc_mass_comp[0, "X_pr"]
        ) == pytest.approx(0.042521, rel=1e-2)
        assert value(
            model.fs.unit.liquid_outlet.conc_mass_comp[0, "X_li"]
        ) == pytest.approx(0.05655, rel=1e-2)
        assert value(
            model.fs.unit.liquid_outlet.conc_mass_comp[0, "X_su"]
        ) == pytest.approx(9.182936e-10, rel=1e-2)
        assert value(
            model.fs.unit.liquid_outlet.conc_mass_comp[0, "X_aa"]
        ) == pytest.approx(0.486516, rel=1e-2)
        assert value(
            model.fs.unit.liquid_outlet.conc_mass_comp[0, "X_fa"]
        ) == pytest.approx(4.35411e-6, rel=1e-2)
        assert value(
            model.fs.unit.liquid_outlet.conc_mass_comp[0, "X_c4"]
        ) == pytest.approx(4.34946e-6, rel=1e-2)
        assert value(
            model.fs.unit.liquid_outlet.conc_mass_comp[0, "X_pro"]
        ) == pytest.approx(4.34719e-6, rel=1e-2)
        assert value(
            model.fs.unit.liquid_outlet.conc_mass_comp[0, "X_ac"]
        ) == pytest.approx(4.584885e-6, rel=1e-2)
        assert value(
            model.fs.unit.liquid_outlet.conc_mass_comp[0, "X_h2"]
        ) == pytest.approx(8.88796e-10, rel=1e-2)
        assert value(
            model.fs.unit.liquid_outlet.conc_mass_comp[0, "X_I"]
        ) == pytest.approx(13.0695, rel=1e-2)
        assert value(
            model.fs.unit.liquid_outlet.conc_mass_comp[0, "X_PHA"]
        ) == pytest.approx(7.2792, rel=1e-2)
        assert value(
            model.fs.unit.liquid_outlet.conc_mass_comp[0, "X_PP"]
        ) == pytest.approx(0.11430, rel=1e-2)
        assert value(
            model.fs.unit.liquid_outlet.conc_mass_comp[0, "X_PAO"]
        ) == pytest.approx(0.69310, rel=1e-2)
        assert value(
            model.fs.unit.liquid_outlet.conc_mass_comp[0, "S_K"]
        ) == pytest.approx(0.33162, rel=1e-2)
        assert value(
            model.fs.unit.liquid_outlet.conc_mass_comp[0, "S_Mg"]
        ) == pytest.approx(0.33787, rel=1e-2)
        assert value(model.fs.unit.liquid_outlet.anions[0]) == pytest.approx(
            2e-2, rel=1e-2
        )
        assert value(model.fs.unit.liquid_outlet.cations[0]) == pytest.approx(
            4e-2, rel=1e-2
        )

        assert value(
            model.fs.unit.liquid_phase.reactions[0].conc_mass_va
        ) == pytest.approx(0.0157435, rel=1e-2)
        assert value(
            model.fs.unit.liquid_phase.reactions[0].conc_mass_bu
        ) == pytest.approx(0.01639589, rel=1e-2)
        assert value(
            model.fs.unit.liquid_phase.reactions[0].conc_mass_ac
        ) == pytest.approx(0.04311, rel=1e-2)
        assert value(
            model.fs.unit.liquid_phase.reactions[0].conc_mass_pro
        ) == pytest.approx(0.035674, rel=1e-2)
        assert value(
            model.fs.unit.liquid_phase.reactions[0].conc_mol_hco3
        ) == pytest.approx(0.093347, rel=1e-2)
        assert value(
            model.fs.unit.liquid_phase.reactions[0].conc_mol_nh3
        ) == pytest.approx(0.0419747, rel=1e-2)
        assert value(
            model.fs.unit.liquid_phase.reactions[0].conc_mol_co2
        ) == pytest.approx(0.0003727, rel=1e-2)
        assert value(
            model.fs.unit.liquid_phase.reactions[0].conc_mol_nh4
        ) == pytest.approx(0.074527, rel=1e-2)

        assert value(model.fs.unit.liquid_phase.reactions[0].S_H) == pytest.approx(
            1.97135e-9, rel=1e-2
        )
        assert value(model.fs.unit.liquid_phase.reactions[0].S_OH) == pytest.approx(
            1.05449e-5, rel=1e-2
        )
        assert value(model.fs.unit.liquid_phase.reactions[0].KW, Var) == pytest.approx(
            2.08e-14, rel=1e-2
        )
        assert value(
            model.fs.unit.liquid_phase.reactions[0].K_a_co2, Var
        ) == pytest.approx(4.94e-7, rel=1e-2)
        assert value(
            model.fs.unit.liquid_phase.reactions[0].K_a_IN, Var
        ) == pytest.approx(1.11e-9, rel=1e-2)
