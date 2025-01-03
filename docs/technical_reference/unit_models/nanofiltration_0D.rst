Nanofiltration (0D)
====================
This nanofiltration (NF) unit model is suitable for non-predictive case studies where the user wishes to specify, constant rejection fractions
whilst preserving electroneutrality in the permeate stream (optional). A single recovery fraction is assumed for all solvents, whilst
individual rejection fractions can be set for all solutes, with the exception of one ion specified for maintaining
electroneutrality. Solutes are assigned default rejection value by grouping them into either two categories, passing and excluded, with
default rejection values for each category. Solutes are categorised using either a user-provided list of species which freely pass the
membrane, or by assuming all neutral and monovalent species pass the membrane.

Retentate pressure is assumed to be related to feed pressure with an optional pressure drop whilst permeate pressure is assumed to be
a degree of freedom, and temperature equality is assumed. Temperature and pressure constraints can be removed with configuration arguments.

This model assumes supports a single liquid phase only and assumes steady-state.

.. index::
   pair: watertap.unit_models.nanofiltration_0D;nanofiltration_0D

.. currentmodule:: watertap.unit_models.nanofiltration_0D

Degrees of Freedom
------------------
The ``Nanofiltration0D`` model has the following degrees of freedom

   * inlet state
   * permeate pressure
   * solvent recovery fraction (``solvent_recovery``)
   * solute rejection fractions (``rekection_comp``) for all solutes EXCEPT the one identified as the electroneutrality species.

The following additional degrees of freedom  may exist depending on configuration options

   * retentate pressure drop (``deltaP``)

Model Structure
---------------
The Nanofiltration0D model consists of three state blocks (``properties_in``, ``properties_retentate`` and ``properties_permeate``).

.. _NF0D_variables:

Variables
---------

.. csv-table::
   :header: "Description", "Symbol", "Variable Name", "Index", "Units"

   "Solvent recovery", ":math:`Q_{solvent}`", "solvent_recovery", None, ":math:`\text{dimensionless}`"
   "Solute rejection", ":math:`R_j`", "rejection_comp", [j], ":math:`\text{dimensionless}`"
   "Retentate pressure drop", ":math:`\deltaP`", "deltaP", [t], ":math:`\text{pressure}`"

.. _NF0D_equations:

Equations
---------

Here :math:`F` represents component flowrate and :math:`Z_j` is the charge on species j.

.. csv-table::
   :header: "Description", "Equation"

   "Component material balances", ":math:`F_{in, j} = F_{retentate, j} + F_{permeate, j}`"
   "Solvent recovery",":math:`Q_{solvent} \times F_{in, j} = F_{permeate_j}`"
   "Solute rejection",":math:`R_j = 1 - \frac{C_{permeate, j}}{C_{feed_j}}`"
   "Permeate electroneutrality",":math:`0 = F_{permeate, j} \times Z_{j}`"
   "Retentate pressure balance",":math:`P_{in} = P_{retentate} - \deltaP`"
   "Retentate temperature equality", ":math:`T_{in} = T_{retentate}`"
   "Permeate temperature equality", ":math:`T_{in} = T_{permeate}`"

Class Documentation
-------------------

* :mod:`watertap.unit_models.nanofiltration_0D`
