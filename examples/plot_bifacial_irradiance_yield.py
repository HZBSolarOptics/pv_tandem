# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 08:35:53 2023

@author: Peter
"""

from pv_tandem import bif_yield
import matplotlib.pyplot as plt
import numpy as np
import pvlib
import pandas as pd
import seaborn as sns

meta_ts = pd.read_csv(
    "./data/meta_ts_dallas_2020.csv", index_col=0, parse_dates=True
)

coord_dallas = dict(latitude = 32.8, longitude = -96.8)

solar_pos = pvlib.solarposition.get_solarposition(meta_ts.index, **coord_dallas)

illumination_df = meta_ts
illumination_df['zenith'] = solar_pos['zenith']
illumination_df['azimuth'] = solar_pos['azimuth']

illumination_df = illumination_df[['DNI','DHI','zenith','azimuth']]

simulator = bif_yield.IrradianceSimulator(illumination_df,
        albedo=0.3,
        module_length=1.96,
        module_height=0.5,
    )

irrad_poa = simulator.simulate(spacing=6, tilt=25, simple_results=True)

irrad_poa.groupby(irrad_poa.index.dayofyear).sum().plot()
plt.show()

irrad_poa.eval('front/back').groupby(irrad_poa.index.dayofyear).mean().plot()
plt.show()

tilt_angles = np.arange(10,54,4)
spacings = np.arange(3,11,1)

scan_res = []

for tilt_angle in tilt_angles:
    for spacing in spacings:
        res_tmp = simulator.simulate(spacing=spacing, tilt=tilt_angle, simple_results=True)

        # sum over the year and convert from Wh to kWh
        res_tmp = res_tmp.sum()/1000
        res_tmp['total'] = res_tmp.sum()
        res_tmp['tilt'] = tilt_angle
        res_tmp['spacing'] = spacing
        scan_res.append(res_tmp)
        
scan_res = pd.concat(scan_res, axis=1).T
scan_res = scan_res.set_index(['tilt', 'spacing'], drop=True)

sns.heatmap(scan_res['total'].unstack('spacing').sort_index(ascending=False))