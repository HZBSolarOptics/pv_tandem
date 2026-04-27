Examples for thermal modelling
==============================

This folder contains **working examples** and **notebooks** demonstrating how to use the core functionality of the models implemented in ``pv_tandem/thermal``. These models allow to calculate the annual energy yield of PV modules with and without a *sub-bandgap reflector* (SBR). The models are used for the work presented in K. Jäger, J. Mandal, B. P. Rand, F. Meggers, and C. Becker, “How do sub-bandgap reflectors affect the performance of PV modules?,” arXiv 2604.20757 (2026).


Examples
--------

1. ``example_01_sbr_princeton_non_spectral.py`` A script to calculate the energy yield of a PV module with and without an SBR when the NSRDB data has no spectral data.
2. ``example_02_sbr_princeton_spectral.py`` A script to calculate the energy yield of a PV module with and without an SBR when the NSRDB data has spectral data.

Notebooks
--------

These notebooks are linked to K. Jäger, J. Mandal, B. P. Rand, F. Meggers, and C. Becker. "How do sub-bandgap reflectors affect the performance of photovoltaic modules?" arXiv: 0000.0000 (2026)

1. ``notebook_01_color_maps.ipynb`` A Jupyter notebook to recreate Fig. 4 of the manuscript: Colormaps for the change in the annual energy yield as a function of the temperature coefficient for power and the loss factor.
2. ``notebook_02_degradation.ipynb`` A Jupyter notebook to perform the calculations on degradation and to recreate Figs. 5 and 6 of the manuscript.