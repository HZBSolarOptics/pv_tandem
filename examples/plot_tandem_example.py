"""
Modeling Monofacial Tandem Solar Cell 
============================
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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pv_tandem import utils

plt.rcParams['figure.dpi'] = 140

spec_irrad_ts = pd.read_csv(
    "./data/spec_poa_dallas_2020.csv", index_col=0, parse_dates=True
)
spec_irrad_ts.columns = spec_irrad_ts.columns.astype(float)
spec_irrad_ts = spec_irrad_ts.clip(lower=0)

meta_ts = pd.read_csv(
    "./data/meta_ts_dallas_2020.csv", index_col=0, parse_dates=True
)

ax = meta_ts.groupby(meta_ts.index.dayofyear)["Temperature"].mean().plot()
ax.set_xlabel("Day of year")
ax.set_ylabel("Avg. daily temperature (°C)")
plt.show()

# %%
# Note that the airmass and zenith values do not exactly match the values in
# the technical report; this is because airmass is estimated from solar
# position and the solar position calculation in the technical report does not
# exactly match the one used here.  However, the differences are minor enough
# to not materially change the spectra.

example_eqe = pd.read_csv('./data/eqe_tandem_2t.csv', index_col=0)
ax = (example_eqe*100).plot()
ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel("Absorptance (%)")
ax.legend(['Perovskite cell', 'Silicon cell'], loc='lower right')
plt.show()

eqe = utils.interp_eqe_to_spec(example_eqe, spec_irrad_ts)
eqe.plot()

j_ph = pd.concat(
    [
        utils.calc_current(spec_irrad_ts / 1000, eqe["pero"]),
        utils.calc_current(spec_irrad_ts / 1000, eqe["si"]),
    ],
    axis=1,
)

ax = (j_ph.groupby(j_ph.index.dayofyear).sum()*3.6/1000).plot()
ax.set_xlabel("Day of year")
ax.set_ylabel("Daily generated Charge (MC/day)")
ax.legend(['Perovskite cell', 'Silicon cell'], loc='upper right')
plt.show()


# %%
# Note that the airmass and zenith values do not exactly match the values in
# the technical report; this is because airmass is estimated from solar
# position and the solar position calculation in the technical report does not
# exactly match the one used here.  However, the differences are minor enough
# to not materially change the spectra.

