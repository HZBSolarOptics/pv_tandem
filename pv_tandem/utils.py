# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from scipy import constants


def calc_current(spec: pd.DataFrame, eqe: pd.DataFrame) -> np.ndarray:
    """
    Calculates the photocurrent from timeseries of impinging spectrum (W/nm/m²) and eqe.

    Parameters
    ----------
    spec : pandas.Dataframe
         Time series of spectral irradiance in the plane of the solar cell. The
         names columns of the DataFrame have to be the wavelength of the incidenting
         light in nm.

    eqe : pandas.Dataframe
        External quantum efficiency of the solar cell.

    Returns
    -------
    current : numpy.array
        Current generated in the solar cell in A/m²
    """
    if len(spec.shape) > 1:
        # spec is timeseries of spectral illumination
        norm_absorbtion = spec.multiply(eqe, axis=1)
        wl_arr = norm_absorbtion.columns.to_series()
        photon_flux = (norm_absorbtion / constants.h / constants.c).multiply(
            wl_arr * 1e-9, axis=1
        )
        current = pd.Series(
            np.trapz(photon_flux, x=wl_arr) * constants.e, index=spec.index
        )
        try:
            current.rename(eqe.name, inplace=True)
        except:
            pass

    else:
        # spec is single spectrum
        norm_absorbtion = eqe.multiply(spec, axis=0)
        wl_arr = norm_absorbtion.index.to_series()
        photon_flux = (norm_absorbtion / constants.h / constants.c).multiply(
            wl_arr * 1e-9, axis=0
        )
        current = np.trapz(photon_flux, x=wl_arr, axis=0) * constants.e

    return current
    # return photo_flux.apply(np.trapz, x=wl_arr, axis=1) * constants.e


def calc_temp_from_NOCT(noct, ambient_temp, irrad_poa):
    """
    Calculates the cell temperature based on the irradiance in the plane of array,
    ambient temperature and NOCT (nominal operating cell temperature)

    Parameters
    ----------
    noct : numeric or array-like
        NOCT (nominal operating cell temperature) of the cell
    ambient_temp : numeric or array-like
        DESCRIPTION.
    irrad_poa : numeric or array-like
        irradiance in the plane of the pv module

    Returns
    -------
    cell_temp : numeric or array-like
        operating temperaure of the cell

    """
    cell_temp = ambient_temp + (noct - 20) / 800 * irrad_poa
    return cell_temp


def interp_eqe_to_spec(eqe: pd.DataFrame, spec: pd.DataFrame) -> pd.DataFrame:
    try:
        f = interp.interp1d(eqe.index, eqe, axis=0)
        new_eqe = f(spec.columns)
        new_eqe = pd.DataFrame(
            new_eqe, index=spec.columns, columns=eqe.columns
        )
        return new_eqe

    except ValueError as v_error:
        if str(v_error) == "object arrays are not supported":
            raise ValueError(
                "It seems that either the wavelength of the spectral or eqe df are not numeric. Try to convert them to a numeric type, e.g.\ndf_spec.columns = df_spec.columns.astype(float)"
            )

        else:
            print("Failed raise")
            raise v_error


def interp_spec_to_eqe(eqe: pd.DataFrame, spec: pd.DataFrame) -> pd.DataFrame:
    try:
        f = interp.interp1d(spec.columns, spec, axis=1)
        new_spec = f(eqe.index)
        new_spec = pd.DataFrame(new_spec, index=spec.index, columns=eqe.index)
        return new_spec

    except ValueError as v_error:
        if str(v_error) == "object arrays are not supported":
            print(
                "It seems that either the wavelength of the spectral or eqe df are not numeric. Try to convert them to a numeric type, e.g.\ndf_spec.columns = df_spec.columns.astype(float)"
            )
        else:
            raise v_error


from scipy import interpolate as interp

if __name__ == "__main__":

    spec = pd.read_csv("./data/tiny_spec.csv", index_col=0)
    spec.columns = spec.columns.astype(float)
    eqe = pd.read_csv("./data/eqe_tandem_2t.csv", index_col=0)

    eqe_new = interp_eqe_to_spec(eqe, spec)
    eqe.plot()

    new_spec = interp_spec_to_eqe(eqe.loc[:1200], spec)

    new_spec.iloc[50].plot()
    spec.iloc[50].plot()

    j_ph = pd.concat(
        [
            calc_current(spec / 1000, eqe_new["pero"]),
            calc_current(spec / 1000, eqe_new["si"]),
        ],
        axis=1,
    )

    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    ax = j_ph.plot()
    ax.tick_params(axis="x", labelrotation=-45)

    # myFmt = mdates.DateFormatter('%Y-%m') # here you can format your datetick labels as desired
    # ax.xaxis.set_major_formatter(myFmt)
    plt.show()
