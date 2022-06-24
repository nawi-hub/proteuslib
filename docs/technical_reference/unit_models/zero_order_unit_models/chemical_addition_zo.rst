Chemical Addition (ZO)
======================

Model Type
----------
This unit model is formulated as a pass-through model form.
See documentation for :ref:`pass-through Helper Methods<pt_methods>`.

Electricity Consumption
-----------------------
The constraint used to calculate energy consumption is described in the Additional Constraints section below. More details can be found in the unit model class.

Costing Method
--------------
Costing is calculated using the cost_chemical_addition method in the zero-order costing package.
See documentation for the :ref:`zero-order costing package<zero_order_costing>`.

Additional Variables
--------------------

.. csv-table::
   :header: "Description", "Variable Name", "Units"

   "Dosing rate of chemical", "chemical_dosage", "mg/l"
   "Mass density of chemical solution", "solution_density", "kg/m**3"
   "Mass fraction of chemical in solution", "ratio_in_solution", "None"
   "Volumetric flow rate of chemical solution", "chemical_flow_vol", "m**3/s"
   "Electricity consumption of unit", "electricity", "kW"

Additional Constraints
----------------------

.. csv-table::
   :header: "Description", "Constraint Name"

   "None", "chemical_flow_constraint"
   "Constraint for electricity consumption based on pump flowrate.", "electricity_consumption"

.. index::
   pair: watertap.unit_models.zero_order.chemical_addition_zo;chemical_addition_zo

.. currentmodule:: watertap.unit_models.zero_order.chemical_addition_zo

Class Documentation
-------------------

.. automodule:: watertap.unit_models.zero_order.chemical_addition_zo
    :members:
    :noindex:
