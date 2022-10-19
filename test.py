# -*- coding: utf-8 -*-


import scipy.io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import lambertw
from scipy import constants
import pvlib
import bifacial_illumination as bi

from sim_functions import spectral_illumination, TandemSimulator

def load_EYcalc_data(mat_file, lat, long):
    mat = scipy.io.loadmat(mat_file)
    ambient_temperature = mat["irradiance"]["Data_TMY3"][0][0][:, 13]
    dt = mat["irradiance"]["Data_TMY3"][0][0]
    dt = pd.to_datetime(
        pd.DataFrame(
            dt[:, :6].astype(int),
            columns=["year", "month", "day", "hour", "minute", "second"],
        )
    ).dt.tz_localize("Etc/GMT+5")

    dt_sp = dt - pd.Timedelta("30 min")
    solarposition = pvlib.solarposition.get_solarposition(
        dt_sp, latitude=lat, longitude=long
    )
    solarposition = solarposition[["zenith", "azimuth"]]

    spec_irrad_diff = mat["irradiance"]["Irr_spectra_clouds_diffuse_horizontal"][0][0]
    spec_irrad_dir = mat["irradiance"]["Irr_spectra_clouds_direct_horizontal"][0][0]
    wl_arr = mat["irradiance"]["Irr_spectra_clouds_wavelength"][0][0][0]

    spec_irrad_dir = pd.DataFrame(spec_irrad_dir, columns=wl_arr, index=dt)
    spec_irrad_diff = pd.DataFrame(spec_irrad_diff, columns=wl_arr, index=dt)

    spec_irrad_diff = spec_irrad_diff.loc[:, 300:1200]
    spec_irrad_dir = spec_irrad_dir.loc[:, 300:1200]

    spec_irrad_dir = (
        spec_irrad_dir
        / np.cos(np.deg2rad(solarposition["zenith"])).clip(0.1, 1).values[:, None]
    )
    spec_irrad_dir = spec_irrad_dir.fillna(0)

    eqe = mat["optics"]["A"][0][0]
    wl_eqe = mat["lambdaTMM"][0]
    eqe = pd.DataFrame(eqe, index=wl_eqe, columns=["pero", "si"])

    spectral_irrad = pd.concat(
        [spec_irrad_dir, spec_irrad_diff], keys=["dni", "dhi"], axis=1
    )

    eqe = pd.DataFrame(
        {
            "pero": np.interp(spectral_irrad["dni"].columns, eqe.index, eqe["pero"]),
            "si": np.interp(spectral_irrad["dni"].columns, eqe.index, eqe["si"]),
        },
        index=spec_irrad_dir.columns,
    )

    return ambient_temperature, spectral_irrad, solarposition, eqe


if __name__ == "__main__":

    lat, long = 25.73, -80.21
    ambient_temp, spectral_irrad, solarposition, eqe = load_EYcalc_data(
        mat_file="eycalc_complete_dump_miami.mat", lat=lat, long=long,
    )

    spec_irrad_inplane = spectral_illumination(
        spectral_irradiance=spectral_irrad, solarposition=solarposition
    ).calc_spectral_inplane()
    
    electrical_parameters = {
        "RshTandem": 1000,
        "RsTandem": 3,
        "j0": {"pero": 2.7e-18, "si": 1e-12},
        "n": {"pero": 1.1, "si": 1},
        "Temp": {"pero": 25, "si": 25},
        "noct": {"pero": 48, "si": 48},
        "tcJsc": {"pero": 0.0002, "si": 0.00032},
        "tcVoc": {"pero": -0.002, "si": -0.0041},
    }
    
    subcells = ['pero', 'si']

    tandem = TandemSimulator(spec_irrad_inplane, eqe, electrical_parameters, subcells, ambient_temp)
    power_HZB = tandem.calc_power()*10
    
    mat_file="eycalc_complete_dump_miami.mat"
    mat = scipy.io.loadmat(mat_file)
    tandem_P_kit = pd.Series(mat['EY'][0]['Power_Tandem'][0][:,0], index = spec_irrad_inplane.index)

    
    fig, ax = plt.subplots(dpi=300)
    power_HZB.groupby(tandem_P_kit.index.dayofyear).sum().plot(ax=ax, label='HZB_reimplement')
    tandem_P_kit.groupby(tandem_P_kit.index.dayofyear).sum().plot(ax=ax, label='KIT')
    ax.legend()
    ax.set_ylim([0, 2500])
    ax.set_xlabel('Day of year')
    ax.set_ylabel('Daily energy yield (Wh/m²)')
    
    fig, ax = plt.subplots(dpi=300)
    power_HZB.groupby(tandem_P_kit.index.dayofyear).sum().plot(ax=ax, label='HZB_reimplement')
    tandem_P_kit.groupby(tandem_P_kit.index.dayofyear).sum().plot(ax=ax, label='KIT')
    ax.legend()
    ax.set_ylim([0, 2500])
    ax.set_xlabel('Day of year')
    ax.set_ylabel('Daily energy yield (Wh/m²)')
    
