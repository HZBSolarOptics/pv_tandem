Model basics
============

pv_tandem is a collection of modelling functions to enable the performance evaluation of single junction and tandem solar cells. The outdoor performance simulation of a solar cell typically requires a combination of several models, chained together. The core data flow of pv_tandem is shown in the follwoing figure.

.. figure:: _static/model_chain_overview.svg
	:align: center
	
	Basic modelchain of pv_tandem

The basic requirment for a solar cell simulation is the photocurrent desity, cell temperature and the solar cell characteristics. Based on these inputs pv_tandem uses one (single junction cell) or two (tandem cell) 1-diode models to calculate the JV (Current density vs Voltage) curve, which can be used to calculate the characteristic points, such as maximum power point (MPP) or open circuit voltage (V\ :sub:`OC`)
The formular of the 1-diode model ist given by:

.. math::

	J(V) = J_\text{ph}-J_\text{0}\left[\exp\left(\frac{V+J\cdot R_\text{series}}{kT/e}\right)-1\right]-\frac{V+J \cdot R_\text{series}}{R_\text{shunt}}

with J as cell current density, V as cell voltage, J\ :sub:`ph` as phototcurrent desity, J\ :sub:`0` as dark saturation current, R\ :sub:`series` as series resistance and R :sub:`shunt` as shunt resistance.
For an energy yield calculation a time series of photocurrent density and cell temperaure is required, that should representation an average of the expected conditions (e.g. typically metrological year (TMY) datasets). For each timestemp the full JV-curve is calculated and the maximum power is evaluated. For the energy yield, the MPP is integrated over time for the total electricity production in the simulated timeframe.

Calculating photocurrent density
________________________________

The photocurrent density of a solar cell depends on two quantities, the spectral irradiance on the solar cell on the one hand and the spectral response of the solar cell on the other hand (often refered to as external quantum efficiency (EQE)). From the spectral irradiance and the EQE the photocurrent density can be calculated according to:

.. math::

 J_\text{ph} = e\int_{\text{300 nm}}^{\text{1200 nm}} EQE(\lambda)E(\lambda)\frac{\lambda}{hc} \mathrm{d}\lambda

with e as elementary charge and E as spectral irradiance.

One of the most accessable data sources for spectral irradiance is the `National Solar Radiation Database (NSRDB) <https://nsrdb.nrel.gov/>`_ provided by National Renewable Energy Laboratory (NREL), a US govermental organization. With their dataproduct `"spectral on-demand" <https://nsrdb.nrel.gov/data-sets/spectral-on-demand-data>`_ year-wise data can be downloaded on a plane defined by a tilt angle. The disadvantage of the dataset is the missing components of direct and diffuse spectral irradiance, which makes it hard to combine with pv_tandem's bifacial illumination model.

For examples of how to use pv_tandem to calculate the photocurrent density, see the gallery :ref:`sphx_glr_auto_examples_plot_jph_from_spec.py`.

While spectral effects can have a significant impact on the performance of tandem solar cells (due to current matching constrains), silicon single junction cells are less effected. Additionaly, non-spectral irradiance data and short circuit current ratings for standart test conditions are much more available then EQE and spectral data it can be sensible to calculate the photocurrent (density) from 



