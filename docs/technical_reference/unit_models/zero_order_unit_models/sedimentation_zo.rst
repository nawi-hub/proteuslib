Sedimentation  (ZO)
===================

Model Type
----------
This unit model is formulated as a single-input, double-output model form.
See documentation for :ref:`single-input, double-output Helper Methods<sido_methods>`.

Electricity Consumption
-----------------------
Electricity consumption is calculated using the constant_intensity helper function.
See documentation for :ref:`Helper Methods for Electricity Demand<electricity_methods>`.

Costing Method
--------------
Costing is calculated using the cost_sedimentation method in the zero-order costing package.
See documentation for the :ref:`zero-order costing package<zero_order_costing>`.

Additional Variables
--------------------

.. csv-table::
   :header: "Description", "Variable Name", "Units"

   "Surface area of sedimentation tank", "basin_surface_area", "ft**2"
   "Particle settling velocity", "settling_velocity", "m/s"

Additional Constraints
----------------------

.. csv-table::
   :header: "Description", "Constraint Name"

   "None", "basin_surface_area_constraint"

.. index::
   pair: watertap.unit_models.zero_order.sedimentation_zo;sedimentation_zo

.. currentmodule:: watertap.unit_models.zero_order.sedimentation_zo

Class Documentation
-------------------

.. automodule:: watertap.unit_models.zero_order.sedimentation_zo
    :members:
    :noindex:
