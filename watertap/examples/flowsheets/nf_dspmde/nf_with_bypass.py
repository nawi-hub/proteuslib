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

from pyomo.environ import (
    units as pyunits,
)
from pyomo.network import Arc
from idaes.core import (
    FlowsheetBlock,
)

from idaes.core.solvers import get_solver
from idaes.core.util.initialization import propagate_state
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.models.unit_models import (
    Mixer,
    Separator,
    Product,
    Feed,
    StateJunction,
)
from idaes.models.unit_models.mixer import MomentumMixingType, MixingType

from watertap.unit_models.nanofiltration_DSPMDE_0D import (
    NanofiltrationDSPMDE0D,
)

from watertap.unit_models.pressure_changer import Pump

from pyomo.environ import ConcreteModel, TransformationFactory

import math

from watertap.property_models.multicomp_aq_sol_prop_pack import (
    MCASParameterBlock,
    ActivityCoefficientModel,
    DensityCalculation,
)
import idaes.core.util.scaling as iscale

from watertap.examples.flowsheets.nf_dspmde import nf

from idaes.core import UnitModelCostingBlock
from watertap.costing import WaterTAPCosting


def main():
    solver = get_solver()
    m = build()
    # Define composition in kg/m3 == g/l

    initialize(m, solver)
    unfix_opt_vars(m)
    nf.add_objective(m)
    optimize(m, solver)
    print("Optimal cost", m.fs.costing.LCOW.value)
    print("Optimal NF pressure (Bar)", m.fs.NF.pump.outlet.pressure[0].value / 1e5)
    print("Optimal area (m2)", m.fs.NF.nfUnit.area.value)
    print(
        "Optimal nf recovery (%)",
        m.fs.NF.nfUnit.recovery_vol_phase[0.0, "Liq"].value * 100,
    )
    print("bypass (%)", m.fs.by_pass_splitter.split_fraction[0, "bypass"].value * 100)

    print("Feed hardness (mg/L as CaCO3)", m.fs.feed.total_hardness.value)
    print("Product hardness (mg/L as CaCO3)", m.fs.product.total_hardness.value)
    print("Disposal hardness (mg/L as CaCO3)", m.fs.disposal.total_hardness.value)
    return m


def build():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.costing = WaterTAPCosting()
    default = nf.define_feed_comp()
    m.fs.properties = MCASParameterBlock(**default)
    m.fs.feed = Feed(property_package=m.fs.properties)
    m.fs.feed.properties[0].conc_mass_phase_comp[...]
    m.fs.feed.properties[0].flow_mass_phase_comp[...]
    # building conc mass phase component as its not build by default

    m.fs.product = Product(property_package=m.fs.properties)
    m.fs.disposal = Product(property_package=m.fs.properties)

    nf.add_hardness_constraint(m.fs.feed)
    nf.add_hardness_constraint(m.fs.product)
    nf.add_hardness_constraint(m.fs.disposal)

    m.fs.by_pass_splitter = Separator(
        property_package=m.fs.properties,
        outlet_list=["nf_stage", "bypass"],
    )
    # NF UNIT BLOCK
    m.fs.NF = FlowsheetBlock(dynamic=False)

    nf.build_nf_block(m, m.fs.NF)

    m.fs.total_product_mixer = Mixer(
        property_package=m.fs.properties,
        inlet_list=["bypass", "nf_stage"],
        energy_mixing_type=MixingType.none,
        momentum_mixing_type=MomentumMixingType.minimize,
    )
    m.fs.total_product_mixer.mixed_state[0.0].temperature.fix(293.15)
    m.fs.feed_to_splitter = Arc(
        source=m.fs.feed.outlet, destination=m.fs.by_pass_splitter.inlet
    )

    m.fs.splitter_to_nfUnit_feed = Arc(
        source=m.fs.by_pass_splitter.nf_stage, destination=m.fs.NF.feed.inlet
    )

    m.fs.splitter_to_mixer = Arc(
        source=m.fs.by_pass_splitter.bypass, destination=m.fs.total_product_mixer.bypass
    )

    m.fs.nfUnit_product_to_mixer = Arc(
        source=m.fs.NF.product.outlet,
        destination=m.fs.total_product_mixer.nf_stage,
    )

    m.fs.nfUnit_retentate_to_disposal = Arc(
        source=m.fs.NF.retentate.outlet,
        destination=m.fs.disposal.inlet,
    )
    m.fs.mixer_to_product = Arc(
        source=m.fs.total_product_mixer.outlet, destination=m.fs.product.inlet
    )
    m.fs.costing.cost_process()
    m.fs.costing.add_annual_water_production(m.fs.product.properties[0].flow_vol)
    m.fs.costing.add_LCOW(m.fs.product.properties[0].flow_vol)
    m.fs.costing.add_specific_energy_consumption(m.fs.product.properties[0].flow_vol)

    TransformationFactory("network.expand_arcs").apply_to(m)
    return m


def fix_init_vars(m):
    # fix intial guess for splitter
    m.fs.by_pass_splitter.split_fraction[0, "bypass"].fix(0.5)

    # apply defualts ofr normal NF init
    nf.fix_init_vars(m)


def initialize(m, solver=None, **kwargs):
    # use standard nf default feed
    nf.set_default_feed(m, solver)
    fix_init_vars(m)

    init_system(m, solver)

    # solve box problem
    print("initalized, DOFS:", degrees_of_freedom(m))
    assert degrees_of_freedom(m) == 0
    solver.solve(m, tee=True)
    print("Solved box problem")


def init_system(m, solver):
    if solver == None:
        solver = get_solver()
    m.fs.feed.initialize(optarg=solver.options)

    propagate_state(m.fs.feed_to_splitter)

    m.fs.by_pass_splitter.mixed_state.initialize(optarg=solver.options)
    m.fs.by_pass_splitter.initialize(optarg=solver.options)
    propagate_state(m.fs.splitter_to_mixer)
    propagate_state(m.fs.splitter_to_nfUnit_feed)

    nf.init_nf_block(m, m.fs.NF, solver)

    propagate_state(m.fs.nfUnit_product_to_mixer)

    m.fs.total_product_mixer.mixed_state.initialize(optarg=solver.options)
    m.fs.total_product_mixer.initialize(optarg=solver.options)

    propagate_state(m.fs.nfUnit_retentate_to_disposal)
    propagate_state(m.fs.mixer_to_product)
    m.fs.NF.product.initialize(optarg=solver.options)
    m.fs.NF.retentate.initialize(optarg=solver.options)


def unfix_opt_vars(m):
    nf.unfix_opt_vars(m)
    m.fs.by_pass_splitter.split_fraction[0, "bypass"].unfix()


def optimize(m, solver, **kwargs):
    result = nf.optimize(m, solver)
    return result


if __name__ == "__main__":
    main()
