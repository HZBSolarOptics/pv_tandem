"""
Energy Yield for Silicon Solar Cell
==================================
Using non-spectral data from NREL
"""

# %%
# This example shows how to model the performance of a single junction silicon
# solar cell. It uses irradiance data from the NSRDB providded by NREL. The
# irradiance data is part of the meta data of the sepctral on demand data product
# but can also be obtained from NSRDB as a seperate data product.
# For the absorptances of the cell it is assumed that the cell produces 39 mA/cm²
# short-circuit current density.


# %%
# First, we load the required libraries and the preprocessed meta-data from spectral
# data for temperature and non-spectral irradiance data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pv_tandem.solarcell_models import OneDiodeModel
import pvlib

plt.rcParams['figure.dpi'] = 140

meta_ts = pd.read_csv(
    "../data/meta_ts_dallas_2020.csv", index_col=0, parse_dates=True
)

# %%
# Next, we calculate the draction of DNI radiating on the tilted module plane 
# (with tilt angle 20 deg) and a fixed fraction of the DHI of 80 %

# Convert angles from degrees to radians
solar_zenith = np.radians(meta_ts['Solar Zenith Angle'])
solar_azimuth = np.radians(meta_ts['Solar Azimuth Angle'])
tilt = np.radians(20)
plane_azimuth = np.radians(180)

# Calculate the cosine of the angle of incidence
cos_theta_i = np.cos(solar_zenith) * np.cos(tilt) + np.sin(solar_zenith) * np.sin(tilt) * np.cos(solar_azimuth - plane_azimuth)

# Calculate the DNI absorbed
dni_plane = meta_ts['DNI'] * cos_theta_i

# Make sure the absorbed DNI is not negative (i.e., the sun is behind the plane)
dni_plane[dni_plane < 0] = 0

# assuming 80% of the DHI are reaching the 20° tilted module plane

irrad_plane = dni_plane + meta_ts['DHI'] * 0.8

# %%
# Next, we define the one-diode model of the solar cell and estiamte the cell
# temperature with the ross model of pvlib

one_diode = OneDiodeModel(
    tcJsc=0.0003, tcVoc=-0.004, R_shunt=3000, R_series=1.5, n=1, j0=1e-12
)

cell_temp = pvlib.temperature.ross(poa_global = irrad_plane,
                                         temp_air = meta_ts['Temperature'],
                                         noct=45
                                         )

# %%
# Finally, the power output of the cell is calculated, assuming a linear relationship
# between irradaince and short circuit density, with Jsc of 39 mA/cm² for 1000
# Watts irradiance

Jsc = irrad_plane/1000*39

iv_paras = one_diode.calc_iv_params(Jsc=Jsc, cell_temp = cell_temp)

power_max = iv_paras['Pmax']

# Converting from W/cm² to kW/m²

power_max = power_max * 10 / 1000

ax = power_max.groupby(power_max.index.dayofyear).sum().plot()
ax.set_xlabel('Day of year')
ax.set_ylabel('Daily yield (kWh/m2)')

print(f"Yearly yield: {(power_max).sum():.1f} kWh/m2")