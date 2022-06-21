.. _install:

Installing WaterTAP
===================

Introduction
------------

.. _about-conda:

Using Conda environments
^^^^^^^^^^^^^^^^^^^^^^^^

Conda environments are a way to create and manage multiple sets of packages and/or Python versions on the same system without incurring conflicts.
Each Conda environment is a dedicated directory, separate from other Conda environments and the operating system's own directories, containing its own collection of packages, executables, and Python installation, including the Python interpreter.
Once a Conda environment is *activated*, using the ``conda activate`` command in a terminal/console, the environment's own version of Python will be used to run commands or interactive sessions for the remainder of the session.

For these reasons, Conda environments are especially useful to install and manage multiple projects (and/or multiple *versions* of the same project) on the same computer with minimal effort,
as they provide a way to seamlessly switch between different projects without conflicts.

Using Conda environments is not mandatory to be able to install and use WaterTAP; however, it is strongly recommended.

To use Conda environments, the ``conda`` package manager is required. Refer to the `Conda installation guide <https://idaes-pse.readthedocs.io/en/stable/tutorials/getting_started/index.html#installation>`_ for detailed steps on how to install Conda for your operating system.

General installation
--------------------

If you are going to use WaterTAP's functionality, but *do not* plan to modify or contribute to WaterTAP's codebase, choose this option.

#. Create a Conda environment (in this example, named ``watertap``) where WaterTAP and its runtime dependencies will be installed:

	.. code-block:: shell

		conda create --name watertap --yes python=3.8 pip=21.1

#. Activate the ``watertap`` environment:

	.. code-block:: shell

		conda activate watertap
	
	To verify that the correct environment is active, run the following command:

	.. code-block:: shell

		python -c "import sys; print(sys.executable)"
	
	If the environment was activated correctly, its name should be contained in the path displayed by the above command.

	.. important:: The ``conda activate`` command described above must be run each time a new terminal/console session is started.

#. Install WaterTAP using ``pip``:

	.. code-block:: shell

		pip install watertap

#. (Optional) If you want to verify that the installation was successful by running the WaterTAP test suite, see :ref:`Running the WaterTAP test suite`.

.. _install-idaes-ext:

Installing solvers
^^^^^^^^^^^^^^^^^^

Windows and Linux Users: solvers distributed through IDAES Extensions
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

After installing WaterTAP, the ``idaes get-extensions`` command can be used to automatically install the solvers distributed as part of the IDAES Extensions.

.. important:: Depending on your operating system, additional steps might be needed. For more information, refer to the `IDAES installation guide <https://idaes-pse.readthedocs.io/en/stable/tutorials/getting_started/index.html#installation>`_.

From the same environment where WaterTAP was installed, run:

    .. code-block:: shell

        idaes get-extensions

.. note:: Typically, the ``idaes get-extensions`` command only needs to be run once for each system, as it will install the required files into a common, system-wide location.

macOS: solvers from conda-forge (experimental)
++++++++++++++++++++++++++++++++++++++++++++++

After installing WaterTAP, we need to ensure we have the Xcode toolkit, build the PyNumero Pyomo extensions, and obtain solvers from conda-forge.

To install Xcode, run:

    .. code-block:: shell

        xcode-select --install


To build PyNumero, from the same environment where WaterTAP was installed, run:

    .. code-block:: shell

        conda install --yes cmake
        pyomo build-extensions

The output of the second command should be something like:

    .. code-block:: shell

        INFO: Finished building Pyomo extensions.
        INFO: The following extensions were built:
                [FAIL]  appsi
                [FAIL]  mcpp
                [ OK ]  pynumero

Finally, we can obtain Ipopt and CBC from conda-forge:

    .. code-block:: shell

        conda install --yes -c conda-forge ipopt coincbc

.. note:: The ``pyomo build-extensions`` command only needs to be run once for each system as it builds and installs the required libraries into a common, system-wide location.

.. note:: After building PyNumero, you should not need cmake. You can remove it by running ``conda uninstall cmake``.

.. _install-edb:

Installing the Electrolyte Database (EDB)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To install the EDB, follow these steps:

1. **Install MongoDB**. The EDB uses `MongoDB <https://www.mongodb.com/>`_ as its storage engine. MongoDB is a third-party
   application that must be installed separately. The "community edition" of MongoDB is free, and has all the
   functionality needed for the EDB. To download and install it, go to the `MongoDB homepage <https://www.mongodb.com/>`_,
   or directly to the
   `MongoDB Community Server download page <https://www.mongodb.com/try/download/community>`_ (see
   :ref:`screenshot <screenshot-mongodb-download>`). On that page,
   select the correct operating system and follow the instructions to install the server.


2. **Load data**. Some electrolyte data is distributed with WaterTAP to "bootstrap" the EDB.
   To load it, use the ``edb load`` command --- part of the :ref:`EDB command-line tools <edb-cli>` ---
   with the bootstrap option, from a shell or command window::

    # Load the standard data into the default MongoDB database, running locally
    edb load -b

3. **Verify the installation**. If the above command works, the MongoDB server is running and the data should
   be loaded. You can verify this in a couple of ways:

    a. `Use the command-line program` to dump out the 'base' collection (which is small) to the console. In a
       shell environment where the Python package has been installed, run the following command::

           edb dump -f '-' -t base

       The result should be a bunch of text that resembles the following::

           Wrote 2 record(s) from collection 'base' to file '<stdout>'
           [{"phases": {"Liq": {"type": "AqueousPhase", "equation_of_state": "Ideal"}}, "state_definition":
           "FTPx", "state_bounds": {"flow_mol": [0, 50, 100], "temperature": [273.15, 300, 650], "pressure":
           [50000, 100000, 1000000]}, "pressure_ref": 100000, "temperature_ref": 300, "base_units": {"time": "s",
           "length": "m", "mass": "kg", "amount": "mol", "temperature": "K"}, "name": "thermo"}, {"base_units":
           {"time": "s", "length": "m", "mass": "kg", "amount": "mol", "temperature": "K"}, "phases": {"Liq":
           {"type": "AqueousPhase", "equation_of_state": "Ideal"}}, "state_definition": "FTPx", "state_bounds":
           {"flow_mol": [0, 50, 100], "temperature": [273.15, 300, 650], "pressure": [50000.0, 100000.0, 1000000.0]},
           "pressure_ref": 100000.0, "temperature_ref": 300, "name": "water_reaction"}]

    b. `Use MongoDB's graphical user interface`, "MongoDB Compass", to browse the data. To do this, find and start
       the application called "MongoDB Compass", which should have been installed when you installed the rest of the
       MongoDB application. Run it, and choose to connect to the server at URL ``mongodb://localhost:27017`` (this
       should be the default). You will get a screen like :ref:`this one <screenshot-mongodb-compass-initial>` (with the
       database you are going to click on next circled).
       Then, select the "electrolytedb" database. The result should show three collections with some records loaded in
       each, as in :ref:`this screen <screenshot-mongodb-compass-edb>` .


.. _Running the WaterTAP test suite:

Running the WaterTAP test suite
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#. To run the WaterTAP test suite, first install the optional testing dependencies using pip:

    .. code-block:: shell

        pip install "watertap[testing]"

#. Then, run the following command to run the complete WaterTAP test suite:

    .. code-block:: shell

        pytest --pyargs watertap

#. (Optional) To see a list of available command-line options, run:

    .. code-block:: shell
    
        pytest --pyargs watertap --help

.. note:: Some tests will be skipped (denoted by an ``s`` symbol). This is to be expected, as some of the tests are only applicable within a developer environment.

For WaterTAP developers
-----------------------

If you plan to modify or contribute to WaterTAP's codebase, choose this option.

.. note:: Typically, *contributing to WaterTAP* will involve opening a Pull Request (PR) in WaterTAP's repository. For more information, refer to :ref:`developer-guide`.

#. Create a Conda environment (in this example, named ``watertap-dev``) where WaterTAP and all dependendencies needed for development will be installed, then activate it:

	.. code-block:: shell

		conda create --name watertap-dev --yes python=3.8 pip=21.1 && conda activate watertap-dev

	.. note:: For more information about using Conda environments, refer to the ":ref:`about-conda`" section above.

#. Clone the WaterTAP repository to your local development machine using ``git clone``, then enter the newly created ``watertap`` subdirectory:

	.. code-block:: shell

		git clone https://github.com/watertap-org/watertap && cd watertap

#. Install WaterTAP and the development dependencies using ``pip`` and the ``requirements-dev.txt`` file:

	.. code-block:: shell

		pip install -r requirements-dev.txt

#. If needed, follow the steps described in the ":ref:`install-idaes-ext`" section above to install solvers distributed through IDAES Extensions.

#. (Optional but recommended) `Pre-commit hooks <https://git-scm.com/book/en/v2/Customizing-Git-Git-Hooks>`_ are scripts that are automatically run by Git "client-side" (i.e. on a developer's local machine)
   whenever `git commit` is run.
   WaterTAP uses the `pre-commit <https://pre-commit.com/>`_ framework to manage a few hooks that are useful for WaterTAP developers.
   To install the WaterTAP pre-commit hooks, run:

    .. code-block:: shell

        pre-commit install

#. To verify that the installation was successful, try running the WaterTAP test suite using ``pytest``:

	.. code-block:: shell

		pytest

#. To view/change the generated documentation, see the :ref:`documentation-mini-guide` section

----

.. rubric:: Screenshots

.. _screenshot-mongodb-download:

.. figure:: ../_static/mongodb-download-page.*

    Download page for MongoDB community server (9/2021)


.. _screenshot-mongodb-compass-initial:

.. figure:: ../_static/mongodb-compass-initial.*

    MongoDB Compass Initial Screen (9/2021)


.. _screenshot-mongodb-compass-edb:

.. figure:: ../_static/mongodb-compass-electrolytedb.*

    MongoDB Compass electrolytedb Collections (9/2021)
