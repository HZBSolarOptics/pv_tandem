Model basics
============

pv_tandem is a collection of modelling functions to enable the performance evaluation of single junction and tandem solar cells.
The core of pv_tandem is made one (single junction cell) or two (tandem cell) 1-diode models to calculate the JV (Current density vs Voltage). The formular of the 1-diode model ist given by:

.. math::

	J(V) = J_\text{ph}-J_\text{0}\left[\exp\left(\frac{V+J\cdot R_\text{series}}{kT/e}\right)-1\right]-\frac{V+J \cdot R_\text{series}}{R_\text{shunt}}

with J as cell current density, V as cell voltage, J\ :sub:`ph` as phototcurrent desity, *J_0* as dark saturation current, *R_series*

.. figure:: _static/model_chain_overview.svg
	:align: center
	
	Basic modelchain of pv_tandem


