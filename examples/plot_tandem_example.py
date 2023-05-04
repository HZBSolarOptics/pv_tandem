"""
Modeling Monofacial Tandem Solar Cell 
============================
Using spectral on-demand data from NREL and simulated EQE data from GENPRO4.
"""

# %%
# This example shows how to model the spectral distribution of irradiance
# based on atmospheric conditions. The spectral distribution of irradiance is
# the power content at each wavelength band in the solar spectrum and is
# affected by various scattering and absorption mechanisms in the atmosphere.
# This example recreates an example figure from the SPECTRL2 NREL Technical
# Report [1]_. The figure shows modeled spectra at hourly intervals across
# a single morning.
#
# References
# ----------
# .. [1] Bird, R, and Riordan, C., 1984, "Simple solar spectral model for
#    direct and diffuse irradiance on horizontal and tilted planes at the
#    earth's surface for cloudless atmospheres", NREL Technical Report
#    TR-215-2436 doi:10.2172/5986936.

# %%
# The SPECTRL2 model has several inputs; some can be calculated with pvlib,
# but other must come from a weather dataset. In this case, these weather
# parameters are example assumptions taken from the technical report.

import pandas as pd
import matplotlib.pyplot as plt

spec_irrad_ts = pd.read_csv('./data/spec_poa_dallas_2020.csv', index_col=0, parse_dates=True)
meta_ts = pd.read_csv('./data/meta_ts_dallas_2020.csv', index_col=0, parse_dates=True)

ax =  meta_ts.groupby(meta_ts.index.dayofyear)['Temperature'].mean().plot()
ax.set_xlabel('Day of year')
ax.set_ylabel('Avg. daily temperature (°C)')
plt.show()

# %%
# Note that the airmass and zenith values do not exactly match the values in
# the technical report; this is because airmass is estimated from solar
# position and the solar position calculation in the technical report does not
# exactly match the one used here.  However, the differences are minor enough
# to not materially change the spectra.

    #example_eqe = pd.read_csv('./data/')