# -*- coding: utf-8 -*-

from angular import AngularAnalysis
import pandas as pd
import pvlib

meta_ts = pd.read_csv(
    "../examples/data/meta_ts_dallas_2020.csv", index_col=0, parse_dates=True
)

coord_dallas = dict(latitude=32.8, longitude=-96.8)

solar_pos = pvlib.solarposition.get_solarposition(
    meta_ts.index, **coord_dallas
)

illumination_df = meta_ts
illumination_df["zenith"] = solar_pos["zenith"]
illumination_df["azimuth"] = solar_pos["azimuth"]

illumination_df = illumination_df[["DNI", "DHI", "zenith", "azimuth"]]

angleSimulator = AngularAnalysis(
    illumination_df,
    albedo=0.3,
    module_length=1.96,
    mount_height=0.5,
    module_spacing=6, module_tilt=25
)

matrix_name = 'testMatrix'
