How to scale a model
--------------------

In ProteusLib, models are scaled in two ways: 1) passing scaling factors for each variable to the solver, and 2) transforming each constraint on the model. 
The method to pass scaling factors for each variable to the solver depends on the solver, but the methods are generally similar. 
Here we describe how to scale models for the solver IPOPT, which is the supported and default solver for ProteusLib.

Scaling a model in four steps
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Set scaling factors for case specific extensive variables
2. Use the ``calculate_scaling_factors`` function to automatically set scaling factors for the other model variables and transform the model constraints
3. If the user created custom variables and constraints, set the scaling factors for the custom variables and transform the custom constraints
4. Set the solver options to include user scaling


Setting scaling factors for variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Scaling factors should be the inverse the expected order of magnitude of the variable, such that if a variable was multiplied by the scaling factor the value 
would be between 0.01 and 100. For example, if a variable is expected to have a value between 1e5-1e6, a scaling factor of 1e-5 or 1e-6 would be appropriate. 

Scaling factors are located in the ``scaling_factor`` object on each block of a model. 
This object is a Pyomo Suffix and can be displayed to show what variables have scaling factors and their values as shown below:

.. code-block:: python

    # where blk is the block of interest (e.g. m.fs.unit)
    blk.scaling_factor.display()

For most variables, the scaling factors are set by default or calculated with the ``calculate_scaling_factors`` function (described later). 
However, several extensive variables (e.g. flowrates, work, area) are case specific and should be set by the user. 
A warning will be provided if users do not set the scaling factors for extensive variables. Users can set a scaling factor for a variable with the following:

.. code-block:: python

    from idaes.core.util.scaling import set_scaling_factor
    # var is the variable, and sf is the scaling factor
    # when setting the scaling factor for a non-indexed variable or for all indices of an indexed variable
    set_scaling_factor(var, sf)

    # when setting the scaling factor for a specific index of a variable
    set_scaling_factor(var[ind], sf)

Typically, a user would like to set the scaling factor for extensive variables on all property StateBlocks on a flowsheet, rather than for each individual StateBlock. 
A user can achieve this by setting the default scaling at the property ParameterBlock as shown below:

.. code-block:: python

    # where prop is the property parameter block, var_name is the variable string name, ind is the variable index as a tuple, and sf is the scaling factor
    prop.set_default_scaling(var_name, sf, index=ind)


Transforming constraints
^^^^^^^^^^^^^^^^^^^^^^^^

Constraints should be scaled such that 1e-8 is a reasonable criterion for convergence. 
Generally, this can be achieved by ensuring that the terms within a constraint are scaled similarly to variables, where the scaled value ranges between 0.01 and 100. 
For example, a mass balance constraint should be scaled with the scaling factor for the mass flowrate.

Unlike the variables, constraints are directly transformed before passing the model to the IPOPT solver. Users can transform the constraints with the following:

.. code-block:: python

    from idaes.core.util.scaling import constraint_scaling_transform
    # where con is the constraint, ind is the constraint index, and sf is the scaling factor
    constraint_scaling_transform(con[ind], sf)

The scaling factor used to transform each constraint is recorded in the ``constraint_transformed_scaling_factor`` Pyomo Suffix. 
Users can observe these values by displaying the suffix as follows:

.. code-block:: python

    # where blk is the block of interest (e.g. m.fs.unit)
    blk.constraint_transformed_scaling_factor.display()

.. note::

    Whenever a user scales a constraint that was already transformed, the constraint is transformed to its original state before applying the new scaling factor.

Using the calculate_scaling_factors function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``calculate_scaling_factors`` function determines and sets scaling factors for all standard variables and transforms all standard constraints by using 
developer provided default values and user provided case specific scaling factors. If the expected case specific scaling factors are not set by the user, 
the function will use a non-case specific default value and provide a warning that states what scaling factor is missing. The function can be used as follows: 


.. code-block:: python

    from idaes.core.util.scaling import calculate_scaling_factors
    calculate_scaling_factors(m)

.. note::

    Each unit, property, reaction, and costing model has a method called calculate_scaling_factors. When the ``calculate_scaling_factors`` function is called 
    on the flowsheet, these methods are recursively called for all blocks.

Passing scaling factors to the solver
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In order to pass scaling factors to IPOPT, the user-scaling option must be specified as shown below.

.. code-block:: python

    from pyomo.environ import SolverFactory
    # Create IPOPT solver object
    opt = SolverFactory('ipopt')
    # Set user-scaling option
    opt.options = {'nlp_scaling_method': 'user-scaling'}
    # Solve model m
    opt.solve(m)

