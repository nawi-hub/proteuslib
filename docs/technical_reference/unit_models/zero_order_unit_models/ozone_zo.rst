Ozone  (ZO)
===========

Model Type
----------
This unit model is formulated as a single-input, single-output model form.
See documentation for :ref:`single-input, single-output Helper Methods<siso_methods>`.

Electricity Consumption
-----------------------
The constraint used to calculate energy consumption is described in the Additional Constraints section below. More details can be found in the unit model class.

Costing Method
--------------
Costing is calculated using the cost_ozonation method in the zero-order costing package.
See documentation for the :ref:`zero-order costing package<zero_order_costing>`.

Additional Variables
--------------------

.. csv-table::
   :header: "Description", "Variable Name", "Units"

   "Ozone contact time", "contact_time", "min"
   "CT value for ozone contactor", "concentration_time", "mg*min/l"
   "Ozone mass transfer efficiency", "mass_transfer_efficiency", "None"
   "Specific energy consumption for ozone generation", "specific_energy_coeff", "kWh/lb"
   "Mass flow rate of ozone", "ozone_flow_mass", "lb/hr"
   "Ozone consumption", "ozone_consumption", "mg/l"
   "Ozone generation power demand", "electricity", "kW"

Additional Constraints
----------------------

.. csv-table::
   :header: "Description", "Constraint Name"

   "Ozone consumption constraint", "ozone_consumption_constraint"
   "Ozone mass flow constraint", "ozone_flow_mass_constraint"
   "Ozone power constraint", "electricity_constraint"

.. index::
   pair: watertap.unit_models.zero_order.ozone_zo;ozone_zo

.. currentmodule:: watertap.unit_models.zero_order.ozone_zo

Class Documentation
-------------------

.. automodule:: watertap.unit_models.zero_order.ozone_zo
    :members:
    :noindex:
