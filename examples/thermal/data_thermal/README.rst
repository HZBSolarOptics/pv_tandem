Data for the thermal examples
=============================

This folder contains the input data for the examples in ``examples/thermal``,
which reproduce the results of "How do sub-bandgap reflectors affect the
performance of PV modules?" (arXiv:2604.20757). The data are of two kinds:

* weather and irradiance data from the National Solar Radiation Database
  (NSRDB), and
* optical and electrical data of a PV module with passivated emitter and rear
  contact (PERC) solar cells, taken from the literature.

The MIT license of this repository covers the code only. The data are subject
to the terms given in the sections below.


NSRDB weather and irradiance data
---------------------------------

Files:

* ``princeton_1223393_40.33_-74.66_tmy-2024.csv``: typical meteorological year
  (TMY) data without spectral information (broadband irradiance, ambient
  temperature, wind speed, ...) for Princeton, NJ, USA.
* ``princeton-2024-1223393-fixed_tilt_35_spectral.zip``: spectral irradiance
  data for one year (2024) for Princeton, NJ, USA, with zip file containing one
  CSV file and was compressed to reduce the size of the repository. The
  examples extract the CSV file automatically into a temporary folder of the
  operating system and delete it at the end of the run. About 160 MB of free
  disk space are needed there.

Source and terms of use:

* The data were downloaded from the NSRDB (https://nsrdb.nrel.gov) on 2025-09-05.
  No modifications were made.
* The NSRDB is provided by the National Laboratory of the Rockies (formerly
  the National Renewable Energy Laboratory) and is available under a Creative
  Commons Attribution license (CC BY 4.0), see
  https://data.openei.org/submissions/1.
* If you use these data, please cite:

  M. Sengupta, Y. Xie, A. Lopez, A. Habte, G. Maclaurin, and J. Shelby,
  "The National Solar Radiation Data Base (NSRDB)", Renewable and Sustainable
  Energy Reviews 89, 51-60 (2018), https://doi.org/10.1016/j.rser.2018.03.003


PERC module reflectance and quantum efficiency
----------------------------------------------

The reflectance :math:`R` and external quantum efficiency (EQE) are for a
typical PV module with PERC solar cells and are based on the publication by
Subedi et al. (see reference below).

Files and their origin in the publication:

* ``SEMSC_Subedi_ea_PERC_UVVisNIR_sim_2020_Fig5b_sim.csv``: simulated reflectance :math:`R_\mathrm{enc}` of the
  encapsulated PERC B module, with scattering included (Fig. 5b).
* ``SEMSC_Subedi_ea_PERC_UVVisNIR_sim_2020_Fig6_B_sim_enc.csv``: external quantum efficiency of the encapsulated module,
  :math:`\mathrm{EQE}_\mathrm{enc}`, estimated from the EQE of the bare cell (Fig. 6) as described below.

The data were digitized from the figures of the publication. 

Columns: wavelength in nm and the respective quantity.

Because the publication gives the EQE only for the bare cell, the EQE of the
encapsulated module is estimated from the reflectance data as::

    EQE_enc(wavelength) = EQE_bare(wavelength)
                          * (1 - R_enc(wavelength)) / (1 - R_bare(wavelength))

:math:`\mathrm{EQE}_\mathrm{enc}` is the input for the short-circuit current
calculations in the paper (spectral and AM1.5 conditions).

Reference:

I. Subedi, T. J. Silverman, M. G. Deceglie, and N. J. Podraza, "PERC silicon
PV infrared to ultraviolet optical model", Solar Energy Materials and Solar
Cells 215, 110655 (2020), https://doi.org/10.1016/j.solmat.2020.110655

<Statement on the terms of reuse of these data.>