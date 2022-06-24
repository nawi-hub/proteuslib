Electrodialysis (1D)
====================

Introduction
------------

Electrodialysis, an electrochemical separation technology, has been utilized to desalinate water for decades.
Compared with other technologies, such as reverse osmosis (RO),
electrodialysis shows advantages in desalinating brackish waters with
moderate salinity by its less intense energy consumption, higher water recovery, and robust
tolerance for adverse non-ionic components (e.g., silica and biological substances) in water.
Between the electrodes of an electrodialysis stack (a reactor module), multiple anion- and
cation-exchange membranes are alternately positioned and separated by spacers to form individual
cells. When operated, electrodialysis converts electrical current to ion flux and, assisted by
the opposite ion selectivity of cation- and anion-exchange membranes (cem and aem), moves ions from
one cell to its adjacent cell in a cell-pair treatment unit (Figure 1). The ion-departing cell is called a **diluate
channel** and the ion-entering cell a **concentrate channel**. Recovered (desalinated) water is
collected from diluate channels of all cell pairs while the concentrate product can be disposed of as brine
or retreated. More overview of the electrodialysis technology can be found in the *References*.

.. figure:: ../../_static/unit_models/EDdiagram.png
    :width: 600
    :align: center

    Figure 1. Schematic representation of an electrodialysis cell pair


One cell pair in an electrodialysis stack can thus be treated as a modeling unit that can multiply to
larger-scale systems.  The presented electrodialysis 1D model establishes mathematical descriptions of
ion and water transport in a cell pair and expands to simulate a stack with a specified cell-pair number.
Modeled mass transfer mechanisms include electrical migration and diffusion of ions and osmosis and electroosmosis
of water. By its name, "1D" suggests the variation of mass transfer and solution properties over length (l in Figure 1)
is mathematically described.  The model relies on the following key assumptions:

* The concentrate and diluate channels have identical geometry.
* For each channel, component fluxes in the bulk solution are uniform in b and s directions in Figure 1
  but dependent on the l directions (the 1-dimensional assumption).
* Steady state: all variables are independent of time.
* Co-current flow operation. 
* Electrical current is operated below the limiting current. 
* Ideality assumptions: activity, osmotic, and van't Hoff coefficients are set at one.
* All ion-exchange membrane properties (ion and water transport number, resistance, permeability) are
  constant.
* Detailed concentration gradient effect at membrane-water interfaces is neglected. 
* Constant pressure and temperature through each channel. 

Control Volumes
---------------

This model has two 1D control volumes for the concentrate and diluate channels.

* Diluate
* Concentrate

Ports
-----

On the two control volumes, this model provides four ports (Pyomo notation in parenthesis):

* inlet_diluate (inlet)
* outlet_diluate (outlet)
* inlet_concentrate (inlet)
* outlet_concentrate (outlet)

Sets
----
This model can simulate electrodialysis for desalination of a water solution containing multiple species
(neutral or ionic). All solution components ( H\ :sub:`2`\ O, neutral solutes, and ions) form a Pyomo set in the model.
For a clear model demonstration, **this document uses an NaCl water solution as an instance hereafter.**  The user can
nevertheless expand the component set as needed to represent other feed water conditions.

.. csv-table:: **Table 1.** List of Set
   :header: "Description", "Symbol", "Indices"


   "Time", ":math:`t`", "[t] ([0])\ :sup:`1`"
   "Length_domain", ":math:`x`", ":math:`l \times(0, 1)` \ :sup:`2`"
   "Phase", ":math:`p`", "['Liq']"
   "Component", ":math:`j`", "['H2O', 'Na_+', '\Cl_-']"
   "Ion", ":math:`j`", "['Na_+', '\Cl_-'] \  :sup:`3`"
   "Membrane", "n/a", "['cem', 'aem']"

**Notes**
 :sup:`1` The time set index is set as [0] in this steady-state model and is reserved for the future extension
 to a dynamic model.

 :sup:`2` By the IDAES convention, the index of length_domain is normalized to a continuous set of (0, 1), which is discretized 
 when differential equations in the model are solved by numerical methods such as "finite difference" discretization. In this
 documentation, :math:`x` refers to the length dimension before normalization and carries a unit of [m].

 :sup:`3` "Ion" is a subset of "Component" and uses the same symbol j.


Degrees of Freedom
------------------
Applying this model to an NaCl solution yields 33 degrees of freedom (**Table 2**), among which
temperature, pressure, and component molar flow rate are state variables that are fixed as initial conditions. The rest
are parameters that should be provided in order to fully solve the model.

.. csv-table:: **Table 2.** List of Degree of Freedom (DOF)
   :header: "Description", "Symbol", "Variable Name", "Index", "Units", "DOF Number \ :sup:`1`"

   "Temperature, inlet_diluate", ":math:`T^D`", "temperature", "None", ":math:`K`", 1
   "Temperature, inlet_concentrate", ":math:`T^C`", "temperature", "None", ":math:`K`", 1
   "Pressure, inlet_diluate",":math:`p^D`", "temperature", "None", ":math:`Pa`", 1
   "Pressure, inlet_concentrate",":math:`p^C`", "temperature", "None", ":math:`Pa`", 1
   "Component molar flow rate, inlet_diluate", ":math:`N_{j, in}^D`", "flow_mol_phase_comp", "[t], ['Liq'], ['H2O', 'Na_+', '\Cl_-']", ":math:`mol s^{-1}`", 3
   "Component molar flow rate, inlet_concentrate", ":math:`N_{j, in}^C`", "flow_mol_phase_comp", "[t], ['Liq'], ['H2O', 'Na_+', '\Cl_-']", ":math:`mol s^{-1}`", 3
   "Water transport number", ":math:`t_w`", "water_trans_number_membrane", "['cem', 'aem']", "dimensionless", 2
   "Water permeability", ":math:`L`", "water_permeability_membrane", "['cem', 'aem']", ":math:`m^{-1}s^{-1}Pa^{-1}`", 2
   "Voltage or Current \ :sup:`2`", ":math:`U` or :math:`I`", "voltage_applied or current_applied", "[t]", ":math:`\text{V}` or :math:`A`", 1
   "Electrode areal resistance", ":math:`r_{el}`", "electrodes_resistance", "[t]", ":math:`\Omega m^2`", 1
   "Cell pair number", ":math:`n`", "cell_pair_num", "None", "dimensionless", 1
   "Current utilization coefficient", ":math:`\xi`", "current_utilization", "None", "dimensionless", 1
   "Spacer thickness", ":math:`s`", "spacer_thickness", "none", ":math:`m` ", 1
   "Membrane areal resistance", ":math:`r`", "membrane_areal_resistance", "['cem', 'aem']", ":math:`\Omega m^2`", 2
   "Cell width", ":math:`b`", "cell_width", "None", ":math:`\text{m}`", 1
   "Cell length", ":math:`l`", "cell_length", "None", ":math:`\text{m}`", 1
   "Thickness of ion exchange membranes", ":math:`\delta`", "membrane_thickness", "['cem', 'aem']", ":math:`m`", 2
   "diffusivity of solute in the membrane phase", ":math:`D`", "solute_diffusivity_membrane", "['cem', 'aem'], ['Na_+', '\Cl_-']", ":math:`m^2 s^{-1}`", 4
   "transport number of ions in the membrane phase", ":math:`t_j`", "ion_trans_number_membrane", "['cem', 'aem'], ['Na_+', '\Cl_-']", "dimensionless", 4

**Note**
 :sup:`1` DOF number accounts for indices of corresponding variables.

 :sup:`2` A user should provide either current or voltage as the electrical input, in correspondence to the "Constant_Current"
 or "Constant_Voltage" treatment mode (configured in this model). The user also should provide an electrical magnitude
 that ensures an operational current *below the limiting current* of the feed solution.


Solution component information
------------------------------
To fully construct solution properties, users need to provide basic component information of the feed solution to use
this model, including identity of all solute species (i.e., Na :sup:`+`, and \Cl :sup:`-` for a
NaCl solution), molecular weight of all component species (i.e., H\ :sub:`2`\ O, Na :sup:`+`, and \Cl :sup:`-`), and charge
and electrical mobility of all ionic species (i.e., Na :sup:`+`, and \Cl :sup:`-`). This can be provided as a solution
dictionary in the following format (instantiated by a NaCl solution).

.. code-block::

   ion_dict = {
            "solute_list": ["Na_+", "Cl_-"],
            "mw_data": {"H2O": 18e-3, "Na_+": 23e-3, "Cl_-": 35.5e-3},
            "electrical_mobility_data": {"Na_+": 5.19e-8, "Cl_-": 7.92e-8},
            "charge": {"Na_+": 1, "Cl_-": -1},
        }

This model, by default, uses H\ :sub:`2`\ O  as the solvent of the feed solution.

Information regarding the property package this unit model relies on can be found here: 

:py:mod:`watertap.property_models.ion_DSPMDE_prop_pack`

Equations
---------

This model solves mass balances of all solution components (H\ :sub:`2`\ O, Na :sup:`+`, and \Cl :sup:`-` for an NaCl
solution) on two control volumes (concentrate and diluate channels). Under the 1D treatment, balance equations are expressed 
as differential algebraic equations (DAE) when concerned variables are functions of length (x). The DAEs are solved in a 
discretization manner using the "finite difference" or "collocation" method implemented in **Pyomo.DAE**. 

Mass balance equations are summarized in **Table 3**. Mass transfer mechanisms account for solute electrical migration, diffusion,
water osmosis, and electroosmosis. Theoretical principles, e.g., continuity equation, Fick's law, and Ohm's law,
to simulate these processes are well developed and some good summaries for the electrodialysis scenario can be found in the *References*.

.. csv-table:: **Table 3** Mass Balance Equations
   :header: "Description", "Equation", "Index set"

   "Component mass balance", ":math:`\left(\frac{\partial N_j (x)}{\partial x}\right)^{C\: or\:  D}+J_j(x)^{C\: or\:  D} b=0`", ":math:`j \in \left['H_2 O', '{Na^{+}} ', '{Cl^{-}} '\right]`"
   "mass transfer flux, concentrate, solute", ":math:`J_j^{C} = \left(t_j^{cem}-t_j^{aem} \right)\frac{\xi i(x)}{ z_j F}-\left(\frac{D_j^{cem}}{\delta ^{cem}} +\frac{D_j^{aem}}{\delta ^{aem}}\right)\left(c_j(x)^C-c_j(x)^D \right)`", ":math:`j \in \left['{Na^{+}} ', '{Cl^{-}} '\right]`"
   "mass transfer flux, diluate, solute", ":math:`J_j^{D} = -\left(t_j^{cem}-t_j^{aem} \right)\frac{\xi i(x)}{ z_j F}+\left(\frac{D_j^{cem}}{\delta ^{cem}} +\frac{D_j^{aem}}{\delta ^{aem}}\right)\left(c_j(x)^C-c_j(x)^D \right)`", ":math:`j \in \left['{Na^{+}} ', '{Cl^{-}} '\right]`"
   "mass transfer flux, concentrate, H\ :sub:`2`\ O", ":math:`J_j^{C} = \left(t_w^{cem}+t_w^{aem} \right)\frac{i(x)}{F}+\left(L^{cem}+L^{aem} \right)\left(p_{osm}(x)^C-p_{osm}(x)^D \right)\frac{\rho_w}{M_w}`", ":math:`j \in \left['H_2 O'\right]`"
   "mass transfer flux, diluate, H\ :sub:`2`\ O", ":math:`J_j^{D} = -\left(t_w^{cem}+t_w^{aem} \right)\frac{i(x)}{F}-\left(L^{cem}+L^{aem} \right)\left(p_{osm}(x)^C-p_{osm}(x)^D \right)\frac{\rho_w}{M_w}`", ":math:`j \in \left['H_2 O'\right]`"

Additionally, several other equations are built to describe the electrochemical principles and electrodialysis performance.

.. csv-table:: **Table 4** Electrical and Performance Equations
   :header: "Description", "Equation"

   "Electrical input condition", ":math:`i(x) = \frac{I}{bl}`, for 'Constant_Current';  :math:`u(x) =U` for 'Constant_Voltage'"
   "Ohm's law", ":math:`u(x) =  i(x) r_{tot}(x)`"
   "Resistance calculation", ":math:`r_{tot}(x)=n\left(r^{cem}+r^{aem}+\frac{s}{\kappa(x)^C}+\frac{s}{\kappa(x)^D}\right)+r_{el}`"
   "Electrical power consumption", ":math:`P(x)=b\int _0 ^l u(x)i(x) dx`"
   "Water-production-specific power consumption", ":math:`P_Q=\frac{P(x=l)}{3.6\times 10^6 nQ_{out}^D}`"
   "Current efficiency for desalination", ":math:`bi(x)\eta(x)=-\sum_{j \in[cation]}{\left[\left(\frac{\partial N_j (x)}{\partial x}\right)^D z_j F\right]}`"

All equations are coded as "constraints" (Pyomo). Isothermal and isobaric conditions apply.

Nomenclature
------------
.. csv-table:: **Table 5.** Nomenclature
   :header: "Symbol", "Description", "Unit"
   :widths: 10, 20, 10

   "**Parameters**"
   ":math:`\rho_w`", "Mass density of water", ":math:`kg\  m^{-3}`"
   ":math:`M_w`", "Molecular weight of water", ":math:`kg\  mol^{-1}`"
   "**Variables**"
   ":math:`N`", "Molar flow rate of a component", ":math:`mol\  s^{-1}`"
   ":math:`J`", "Molar flux of a component", ":math:`mol\  m^{-2}s^{-1}`"
   ":math:`b`", "Cell/membrane width", ":math:`m`"
   ":math:`l`", "Cell/membrane length", ":math:`m`"
   ":math:`t`", "Ion transport number", "dimensionless"
   ":math:`I`", "Current input", ":math:`A`"
   ":math:`i`", "Current density", ":math:`A m^{-2}`"
   ":math:`U`", "Voltage input over a stack", ":math:`V`"
   ":math:`u`", "x-dependent voltage over a stack", ":math:`V`"
   ":math:`n`", "Cell pair number", "dimensionless"
   ":math:`\xi`", "Current utilization coefficient (including ion diffusion and water electroosmosis)", "dimensionless"
   ":math:`z`", "Ion charge", "dimensionless"
   ":math:`F`", "Faraday constant", ":math:`C\ mol^{-1}`"
   ":math:`D`", "Ion Diffusivity", ":math:`m^2 s^{-1}`"
   ":math:`\delta`", "Membrane thickness", ":math:`m`"
   ":math:`c`", "Solute concentration", ":math:`mol\ m^{-3}`"
   ":math:`t_w`", "Water electroosmotic transport number", "dimensionless"
   ":math:`L`", "Water permeability (osmosis)", ":math:`ms^{-1}Pa^{-1}`"
   ":math:`p_{osm}`", "Osmotic pressure", ":math:`Pa`"
   ":math:`r_{tot}`", "Total areal resistance", ":math:`\Omega m^2`"
   ":math:`r`", "Membrane areal resistance", ":math:`\Omega m^2`"
   ":math:`r_{el}`", "Electrode areal resistance", ":math:`\Omega m^2`"
   ":math:`s`", "Spacer thickness", ":math:`m`"
   ":math:`\kappa`", "Solution conductivity", ":math:`S m^{-1}\ or\  \Omega^{-1} m^{-1}`"
   ":math:`\eta`", "Current efficiency for desalination", "dimensionless"
   ":math:`P`", "Power consumption", ":math:`W`"
   ":math:`P_Q`", "Specific power consumption", ":math:`kW\ h\  m^{-3}`"
   ":math:`Q`", "Volume flow rate", ":math:`m^3s^{-1}`"
   "**Subscripts and superscripts**"
   ":math:`C`", "Concentrate channel",
   ":math:`D`", "Diluate channel",
   ":math:`j`", "Component index",
   ":math:`in`", "Inlet",
   ":math:`out`", "Outlet",
   ":math:`cem`", "Cation exchange membrane",
   ":math:`aem`", "Anion exchange membrane",

References
----------
Strathmann, H. (2010). Electrodialysis, a mature technology with a multitude of new applications.
Desalination, 264(3), 268-288.

Strathmann, H. (2004). Ion-exchange membrane separation processes. Elsevier. Ch. 4.

Campione, A., Cipollina, A., Bogle, I. D. L., Gurreri, L., Tamburini, A., Tedesco, M., & Micale, G. (2019).
A hierarchical model for novel schemes of electrodialysis desalination. Desalination, 465, 79-93.