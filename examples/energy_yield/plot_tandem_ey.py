"""
Energy Yield for Tandem Solar Cell
==================================
Using spectral on-demand data from NREL and simulated EQE data from GENPRO4.
"""

# %%
# This example shows how to model the performance of a tandem solar cell. It
# uses "spectral-on-demand" data from the NSRDB providded by NREL.
# For the absorptances of the subcells GENPRO4 simulated EQE curves are used
# originally createf for the following publication:
# Reference
# ----------
# .. [1] P. Tillmann, K. Jäger, A. Karsenti, L. Kreinin, C. Becker (2022)
#    “Model-Chain Validation for Estimating the Energy Yield of Bifacial 
#    Perovskite/Silicon Tandem Solar Cells,” Solar RRL 2200079, 
#    DOI: 10.1002/solr.202200079

# %%
# First, we load the preprocessed spectral and meta-data (for temperature)
# The spectral data has to be converted from W/µm/m2 to W/nm/m2 (by deviding by 1000)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pv_tandem import utils, solarcell_models
import pvlib

plt.rcParams['figure.dpi'] = 140

spec_irrad_ts = pd.read_csv(
    "../data/spec_poa_dallas_2020.csv", index_col=0, parse_dates=True
)
spec_irrad_ts.columns = spec_irrad_ts.columns.astype(float)
spec_irrad_ts = spec_irrad_ts.clip(lower=0)/1000

meta_ts = pd.read_csv(
    "../data/meta_ts_dallas_2020.csv", index_col=0, parse_dates=True
)

# %%
# Note that the airmass and zenith values do not exactly match the values in
# the technical report; this is because airmass is estimated from solar
# position and the solar position calculation in the technical report does not
# exactly match the one used here.  However, the differences are minor enough
# to not materially change the spectra.

eqe = pd.read_csv('../data/eqe_tandem_2t.csv', index_col=0)

eqe = utils.interp_eqe_to_spec(eqe, spec_irrad_ts)

electrical_parameters = {
    "Rsh": {"pero": 1000, "si": 3000},
    "RsTandem": 3,
    "j0": {"pero": 2.7e-18, "si": 1e-12},
    "n": {"pero": 1.1, "si": 1},
    "tcJsc": {"pero": 0.0002, "si": 0.00032},
    "tcVoc": {"pero": -0.002, "si": -0.0041},
}

temperature = pvlib.temperature.noct_sam(spec_irrad_ts.sum(axis=1)*1.15,
                                         meta_ts['Temperature'],
                                         meta_ts['Wind Speed'],
                                         noct=45,
                                         module_efficiency=0.25)


temperature = pd.DataFrame({'pero':temperature,
                            'si':temperature})

tandem = solarcell_models.TandemSimulator2T(
    eqe=eqe,
    electrical_parameters=electrical_parameters,
    subcell_names=["pero", "si"],
)

power = tandem.calc_power(spec_irrad=spec_irrad_ts,
                          cell_temps=temperature)

power.index = spec_irrad_ts.index

ax = (power.groupby(power.index.dayofyear).sum() * 10 / 1000).plot()
ax.set_xlabel('Day of year')
ax.set_ylabel('Daily yield (kWh/m2)')

print(f"Yearly yield: {(power * 10 /1000).sum():.1f} kWh/m2")
# %%