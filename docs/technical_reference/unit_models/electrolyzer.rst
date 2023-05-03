Electrolyzer
============
.. note::

  The development of the electrolyzer model is ongoing.

This is a simplified electrolyzer unit model used to approximate electrolysis performance.  With the current build, the model is simulated under the following assumptions:
   * simulation of this unit model is only supported with the Multi-Component Aqueous Solution (MCAS) property package
   * supports liquid phase only, vapor components are modeled in the liquid phase
   * supports steady-state only
   * assumes isothermal conditions and performance is not temperature dependent
   * does not determine equilibrium of electrolysis products in solution
   * does not consider undesired reactions
   * does not account for mass transfer limitations of electrolyte volume to electrode surface area ratios

.. index::
   pair: watertap.unit_models.electrolyzer;electrolyzer

.. currentmodule:: watertap.unit_models.electrolyzer

Introduction
------------
This model was primarily motivated to simulate the chlor-alkali membrane electrolysis process, a conventional electrolyzer for the production of chlorine gas and hydroxide (Bommaraju, 2015) (Kent, 2007). The model has been demonstrated for the chlor-alkali configuration in the electrolyzer testing file and may be generalizable to other electrolysis mechanisms, but has not been validated for generalized processes.

Given a membrane electrolyzer, the catholyte and anolyte are separated by an ion exchange membrane. This provides two distinct control volumes to perform model calculations. The basis for determining electrolyzer performance is accounting for the Faradaic conversion of species with respect to the electrolysis reactions at the cathode and anode. Faradaic conversion considers equating the supplied electrical current to the amount of electrons available for electrochemical reaction by Faraday's law. The calculation of Faradaic conversion is described in "electrons passed between anode and cathode contributing to reactions" found in the :ref:`Equations <electrolyzer_equations>`.

Degrees of Freedom
------------------
For a given electrolyzer design, there are 8 degrees of freedom in addition to the inlet state variables (i.e., temperature, pressure, component flowrates) that should be fixed for the model to be fully specified. In typical design cases the following 8 variables are fixed:

   * membrane current density, :math:`i_{mem}`
   * anode current density, :math:`i_{ano}`
   * anode overpotential, :math:`\eta_{ano}`
   * cathode current density, :math:`i_{cat}`
   * cathode overpotential, :math:`\eta_{cat}`
   * current, :math:`I`
   * current efficiency, :math:`\theta_{current}`
   * voltage efficiency, :math:`\theta_{voltage}` _or_ resistance, :math:`R`

Additionally, the electrolysis reactions at the anode and cathode must be specified with stoichiometry and electrochemical potential (Phillips, 2017). By default, the following stoichiometric variables will be fixed to 0 when the electrolyzer model block is constructed. Therefore, only the nonzero stoichiometry must be additionally specified. As an example for the chlor-alkali process, the following reactions occur with sodium ions permeating from the anolyte to catholyte. The stoichiometry is normalized to 1 mole of electrons as intended in the model framework.

Anode:

    .. math::

        & Cl^-\to{\small\frac{1}{2}}Cl_2+e^-\hspace{4em}E=1.21V \\\\

Cathode:

    .. math::

        & H_2O+e^-\to{\small\frac{1}{2}}H_2+OH^-\hspace{4em}E=-0.99V \\\\

The following variables should then be fixed:

.. code-block::

   # membrane properties
   m.fs.electrolyzer.membrane_ion_transport_number["Liq", "NA+"].fix(1)

   # anode properties
   # Cl- --> 0.5 Cl2 + e-
   m.fs.electrolyzer.anode_electrochem_potential.fix(1.21)
   m.fs.electrolyzer.anode_stoich["Liq", "CL-"].fix(-1)
   m.fs.electrolyzer.anode_stoich["Liq", "CL2-v"].fix(0.5)

   # cathode properties
   # H20 + e- --> 0.5 H2 + OH-
   m.fs.electrolyzer.cathode_electrochem_potential.fix(-0.99)
   m.fs.electrolyzer.cathode_stoich["Liq", "H2O"].fix(-1)
   m.fs.electrolyzer.cathode_stoich["Liq", "H2-v"].fix(0.5)
   m.fs.electrolyzer.cathode_stoich["Liq", "OH-"].fix(1)

Model Structure
------------------
The electrolyzer model consists of 2 ``ControlVolume0DBlocks``, one for the anolyte and another for the catholyte. Currently, the generation of species via electrolysis reactions is handled by the ``custom_molar_term`` within the ``ControlVolume0DBlock``. Coupling this custom molar term with the Faradaic conversion, the material balance may be written by the following and calculated within the ``ControlVolume0DBlock``. Here, membrane flux is handled by relating the stoichiometric coefficient required to satisfy the charge balance. The direction of membrane flow is calculated from anode to cathode in the unit model, lending to the signage of the membrane stoichiometric coefficient shown.

Anode:

    .. math::

        & \dot{n}_{in,j}-\dot{n}_{out,j}+\frac{\left(\varepsilon_{ano,j}-\varepsilon_{mem,j}\right)I\theta_{current}}{F}=0 \\\\

Cathode:

    .. math::

        & \dot{n}_{in,j}-\dot{n}_{out,j}+\frac{\left(\varepsilon_{cat,j}+\varepsilon_{mem,j}\right)I\theta_{current}}{F}=0 \\\\

Considering the reaction block and reaction package are omitted, no temperature dependence and rigorous energy balance are considered.

Scaling of unit model variables should first be performed using ``calculate_scaling_factors()``. For the case that the model is poorly scaled, the ``current`` variable should be rescaled. Estimated scaling factors are propagated from the ``current`` scaling factor for other unit model variables which are dependent on process scale. An example of the methodology is the provided below.

.. code-block::

   import idaes.core.util.scaling as iscale

   # fix the current for given design
   m.fs.electrolyzer.current.fix(2e8)

   # adjust the current's scaling factor to the inverse of the variable's magnitude
   iscale.set_scaling_factor(m.fs.electrolyzer.current, 1e-8)

   # run calculate scaling factor utility on model that propagates estimated scaling factors for other variables
   iscale.calculate_scaling_factors(m)

Sets
----
.. csv-table::
   :header: "Description", "Symbol", "Indices"

   "time", ":math:`t`", "[0]"
   "phases", ":math:`p`", "['Liq']"
   "components", ":math:`j`", "['H2O', solutes]"

.. _electrolyzer_variables:

Variables
----------

.. csv-table::
   :header: "Description", "Symbol", "Variable Name", "Index", "Units"

   "membrane area", ":math:`A_{mem}`", "membrane_area", "None", ":math:`m^2`"
   "membrane current density", ":math:`i_{mem}`", "membrane_current_density", "None", ":math:`\frac{A}{m^2}`"
   "ion transport number of species passing through the membrane :math:`^{ab}`", ":math:`t_{mem,j}`", "membrane_ion_transport_number", "[p, j]", ":math:`\text{dimensionless}`"
   "anode area", ":math:`A_{ano}`", "anode_area", "None", ":math:`m^2`"
   "anode current density", ":math:`i_{ano}`", "anode_current_density", "None", ":math:`\frac{A}{m^2}`"
   "anode electrochemical potential :math:`^a`", ":math:`E_{ano}`", "anode_electrochem_potential", "None", ":math:`V`"
   "anode overpotential", ":math:`\eta_{ano}`", "anode_overpotential", "None", ":math:`V`"
   "anode stoichiometry of the reaction :math:`^{ab}`", ":math:`\varepsilon_{ano,j}`", "anode_stoich", "[p, j]", ":math:`\text{dimensionless}`"
   "cathode area", ":math:`A_{cat}`", "cathode_area", "None", ":math:`m^2`"
   "cathode current density", ":math:`i_{cat}`", "cathode_current_density", "None", ":math:`\frac{A}{m^2}`"
   "cathode electrochemical potential :math:`^{ab}`", ":math:`E_{cat}`", "cathode_electrochem_potential", "None", ":math:`V`"
   "cathode overpotential", ":math:`\eta_{cat}`", "cathode_overpotential", "None", ":math:`V`"
   "cathode stoichiometry of the reaction :math:`^a`", ":math:`\varepsilon_{cat,j}`", "cathode_stoich", "[p, j]", ":math:`\text{dimensionless}`"
   "current", ":math:`I`", "current", "None", ":math:`A`"
   "cell voltage", ":math:`V_{cell}`", "voltage_cell", "None", ":math:`V`"
   "ohmic resistance", ":math:`R`", "resistance", "None", ":math:`\Omega`"
   "power", ":math:`P`", "power", "None", ":math:`W`"
   "reversible voltage", ":math:`V_{rev}`", "voltage_reversible", "None", ":math:`V`"
   "electrons contributing to electrolysis reactions", ":math:`\dot{n}_{e^-}`", "electron_flow", "None", ":math:`\frac{mol}{s}`"
   "current efficiency", ":math:`\theta_{current}`", "efficiency_current", "None", ":math:`\text{dimensionless}`"
   "voltage efficiency", ":math:`\theta_{voltage}`", "efficiency_voltage", "None", ":math:`\text{dimensionless}`"
   "power efficiency", ":math:`\theta_{power}`", "efficiency_power", "None", ":math:`\text{dimensionless}`"
   "molar flow of species j across the membrane from anolyte to catholyte", ":math:`\dot{n}_{mem,j}`", "mass_transfer_term", "[t, p, j]", ":math:`\frac{mol}{s}`"
   "molar generation of species j by electrolysis at the anode", ":math:`\dot{n}_{ano,j}`", "custom_reaction_anode", "[t, j]", ":math:`\frac{mol}{s}`"
   "molar generation of species j by electrolysis at the cathode", ":math:`\dot{n}_{cat,j}`", "custom_reaction_cathode", "[t, j]", ":math:`\frac{mol}{s}`"

| :math:`^a` Variable intended to be move to a interchangeable component block, callable to the base electrolyzer model
| :math:`^b` Value is normalized to 1 electron in electrolysis stoichiometry


For non-fixed efficiencies, electrochemical potentials, and other relevant variables, custom constraints  may be constructed at the flowsheet level. These may allow for predicative performance as a function of temperature, concentration, overpotential, and other variables.

.. _electrolyzer_equations:

Equations
-----------
.. csv-table::
   :header: "Description", "Equation"

   "membrane current density", ":math:`I = i_{mem}A_{mem}`"
   "ion permeation through the membrane", ":math:`\dot{n}_{mem,j} = -t_{mem,j}\dot{n}_{e^-}`"
   "anode current density", ":math:`I = i_{ano}A_{ano}`"
   "molar generation of species according the anode electrolysis reaction", ":math:`\dot{n}_{ano,j} = \varepsilon_{ano,j}\dot{n}_{e^-}`"
   "cathode current density", ":math:`I = i_{cat}A_{cat}`"
   "molar generation of species according the cathode electrolysis reaction", ":math:`\dot{n}_{cat,j} = \varepsilon_{cat,j}\dot{n}_{e^-}`"
   "reversible voltage", ":math:`V_{rev} = E_{ano}-E_{cat}`"
   "cell voltage", ":math:`V_{cell} = V_{rev}+\eta_{ano}+\eta_{cat}+IR`"
   "power", ":math:`P = IV_{cell}`"
   "electrons contributing to reactions :math:`^c`", ":math:`\dot{n}_{e^-} = \frac{I\theta_{current}}{F}`"
   "voltage efficiency", ":math:`V_{rev} = V\theta_{voltage}`"
   "power efficiency", ":math:`\theta_{power} = \theta_{current}\theta_{voltage}`"

\ :math:`^c` is Faraday's constant from ``idaes.core.util.constants``, 96,485 C/mol

Costing Method
---------------

Costing Method Variables
+++++++++++++++++++++++++

The following parameters are constructed when applying the electrolyzer costing method in the ``watertap_costing_package``:

.. csv-table::
   :header: "Description", "Symbol", "Variable Name", "Default Value", "Units"

   "membrane replacement factor (fraction of membrane replaced/year)", ":math:`f_{memreplace}`", "factor_membrane_replacement", "0.33", ":math:`\text{dimensionless}`"
   "membrane unit cost", ":math:`c_{membrane}`", "membrane_unit_cost", "25", ":math:`\frac{$}{m^2}`"
   "anode unit cost", ":math:`c_{anode}`", "anode_unit_cost", "300", ":math:`\frac{$}{m^2}`"
   "cathode unit cost", ":math:`c_{cathode}`", "cathode_unit_cost", "600", ":math:`\frac{$}{m^2}`"
   "membrane, anode, and cathode fraction of total capital", ":math:`f_{material}`", "fraction_material_cost", "0.65", ":math:`\text{dimensionless}`"

The following variables are constructed when applying the electrolyzer costing method in the ``watertap_costing_package``:

.. csv-table::
   :header: "Description", "Symbol", "Variable Name", "Units"

   "membrane capital cost", ":math:`C_{membrane}`", "membrane_cost", ":math:`$`"
   "anode capital cost", ":math:`C_{anode}`", "anode_cost", ":math:`$`"
   "cathode capital cost", ":math:`C_{cathode}`", "cathode_cost", ":math:`$`"
   "membrane replacement cost", ":math:`C_{memreplace}`", "membrane_replacement_cost", ":math:`\frac{$}{yr}`"

Capital Cost Calculations
+++++++++++++++++++++++++

Capital costs are contributing to the majority of material costs for the anode cathode and membrane. Each material cost is calculated individually then summed (O’Brien, 2005).

    .. math::

        & C_{membrane} = c_{membrane}A_{membrane} \\\\
        & C_{anode} = c_{anode}A_{anode} \\\\
        & C_{cathode} = c_{cathode}A_{cathode} \\\\
        & C_{cap,total} = \frac{C_{membrane}+C_{anode}+C_{cathode}}{f_{memreplace}}

Operating Cost Calculations
+++++++++++++++++++++++++++

Operating costs for the electrolyzer are the electricity requirements and membrane replacement costs. Electricity is costed using ``cost_flow`` applied to the ``power`` variable on the unit model. Currently, replacement costs for the anode and cathode are not considered in the costing function.

    .. math::

        C_{op,tot} = C_{membrane replace} = f_{memreplace}c_{membrane}A_{membrane}

Code Documentation
-------------------

* :mod:`watertap.unit_models.electrolyzer`
* :mod:`watertap.costing.units.electrolyzer`

References
-----------
Bommaraju, T. V., & O’Brien, T. F. (2015). Brine electrolysis. Electrochemistry Encyclopedia. https://knowledge.electrochem.org/encycl/art-b01-brine.htm


Kent, J. A. (Ed.). (2007). Kent and Riegel’s Handbook of Industrial Chemistry and Biotechnology. Springer US. https://doi.org/10.1007/978-0-387-27843-8


O’Brien, T., Bommaraju, T. V., & Hine, F. (2005). Handbook of chlor-alkali technology. Springer.

Phillips, R., Edwards, A., Rome, B., Jones, D. R., & Dunnill, C. W. (2017). Minimising the ohmic resistance of an alkaline electrolysis cell through effective cell design. International Journal of Hydrogen Energy, 42(38), 23986–23994. https://doi.org/10.1016/j.ijhydene.2017.07.184

