# -*- coding: utf-8 -*-

import scipy.io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import lambertw
from scipy import constants
import pvlib
import bifacial_illumination as bi

# import cell_sim as cs


def calc_current(spec, eqe):
    """
    Calculates the photocurrent from timeseries of impinging spectrum (W/nm/m²) and eqe.

    Parameters
    ----------
    spec : pandas.Dataframe
         Time series of spectral irradiance in the plane of the solar cell. The
         names columns of the DataFrame have to eb the wavelength of the incidenting
         light in nm.
         
    eqe : pandas.Dataframe
        External quantum efficiency of the solar cell.

    Returns
    -------
    current : numpy.array
        Current generated in the solar cell in A/m²
    """
    norm_absorbtion = spec.multiply(eqe, axis=1)
    wl_arr = norm_absorbtion.columns.to_series()
    photon_flux = (norm_absorbtion / constants.h / constants.c).multiply(
        wl_arr * 1e-9, axis=1
    )
    current = np.trapz(photon_flux, x=wl_arr) * constants.e
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


class OneDiodeModel:
    def __init__(self, tcJsc, tcVoc, R_shunt, R_series, n, j0):
        self.tcJsc = tcJsc
        self.tcVoc = tcVoc
        self.R_shunt = R_shunt
        self.R_series = R_series
        self.n = n
        self.j0 = j0

    def calc_iv(self, Jsc, cell_temp, j_arr):

        # Thermal voltage at room temperature in V
        Vth = 0.02569

        factorJsc = 1 + self.tcJsc * (cell_temp - 25)
        factorVoc = 1 + self.tcVoc * (cell_temp - 25)

        Jsc = Jsc * factorJsc

        Voc_rt = (
            Jsc / 1000 * self.R_shunt
            - self.n
            * Vth
            * lambertw(
                np.exp(
                    np.log(self.j0 / 1000 * self.R_shunt)
                    + self.R_shunt * (Jsc + self.j0) / (1000 * self.n * Vth)
                    - np.log(self.n * Vth)
                )
            )
            + self.j0 / 1000 * self.R_shunt
        )

        Voc_rt[Jsc < 0.1] = np.nan
        Voc = Voc_rt * factorVoc

        lambw = (
            np.log(self.j0 / 1000 * self.R_shunt)
            + (
                self.R_shunt
                * ((np.subtract.outer(Jsc, j_arr) + self.j0))
                / (1000 * self.n * Vth)
            )
            - np.log(self.n * Vth)
        )

        V = (
            np.subtract.outer(
                Jsc / 1000 * self.R_shunt,
                j_arr / 1000 * (self.R_shunt + self.R_series),
            )
            - self.n * Vth * lambertw(np.exp(lambw))
            + (self.j0 / 1000 * self.R_shunt - Voc_rt + Voc)[:, None]
        )

        V[(V < 0) | (np.isnan(V))] = 0

        return V


class TandemSimulator:
    def __init__(
        self,
        spec_irrad,
        eqe,
        electrical_parameters,
        subcells,
        ambient_temp,
        min_Jsc_both_cells=True,
    ):
        self.electrics = electrical_parameters
        self.subcells = subcells
        self.spec_irrad = spec_irrad
        self.eqe = eqe
        self.Jsc = self.calculate_Jsc(min_Jsc_both_cells)

        self.electrical_models = {}
        self.j_arr = np.linspace(-5, 35, 401)
        self.cell_temps = {
            subcell: calc_temp_from_NOCT(
                self.electrics["noct"][subcell], ambient_temp, spec_irrad.sum(axis=1),
            )
            * 1.15  # correction for limited range of spectrum (onyl until 1200 nm)
            for subcell in subcells
        }

        for subcell in subcells:
            self.electrical_models[subcell] = OneDiodeModel(
                tcJsc=self.electrics["tcJsc"][subcell],
                tcVoc=self.electrics["tcVoc"][subcell],
                R_shunt=self.electrics["RshTandem"],
                R_series=self.electrics["RsTandem"],
                n=self.electrics["n"][subcell],
                j0=self.electrics["j0"][subcell],
            )

    def calculate_Jsc(self, min_Jsc_both_cells):

        Jsc = []
        for subcell in self.subcells:
            Jsc_loop = pd.Series(
                calc_current(self.spec_irrad, self.eqe[subcell]) / 10, name=subcell
            )
            Jsc.append(Jsc_loop)
        Jsc = pd.concat(Jsc, axis=1)

        if min_Jsc_both_cells:
            Jsc_min = Jsc.min(axis=1)
            for subcell in self.subcells:
                Jsc[subcell] = Jsc_min

        return Jsc

    def calc_power(self):
        V = []
        for subcell in self.subcells:
            V.append(
                pd.DataFrame(
                    self.electrical_models[subcell].calc_iv(
                        self.Jsc[subcell].values,
                        self.cell_temps[subcell].values,
                        self.j_arr,
                    )
                )
            )
        V = pd.concat(V, axis=1, keys=self.subcells)
        V_tandem = V.groupby(level=1, axis=1).sum()

        P = V_tandem.values * self.j_arr[None, :]
        P_max = P.max(axis=1)
        P_max = pd.Series(P_max, index=self.spec_irrad.index)
        return P_max


class spectral_illumination(bi.YieldSimulator):
    def __init__(
        self,
        spectral_irradiance,
        solarposition,
        bifacial=True,
        albedo=0.2,
        module_length=1.96,
        front_eff=0.2,
        back_eff=0.18,
        module_height=0.5,
        spacing=20,
        tilt_angle=20,
        tmy_data=True,
    ):
        """
        Stil needs docstring
        """
        self.bifacial = bifacial
        self.front_eff = front_eff
        self.back_eff = back_eff
        self.geo_instance = None

        # whether the underlying data is representing a tmy
        self.tmy_data = tmy_data

        # whether the perez model should be used to determine the components of diffuse irradiance
        # self.perez_diffuse = perez_diffuse

        placeholder = solarposition.zenith.copy()
        placeholder[:] = 1

        self.dni = placeholder
        self.dhi = placeholder

        self.spectral_irradiance = spectral_irradiance
        self.albedo = albedo
        self.spacing = spacing
        self.tilt_angle = tilt_angle

        self.input_parameter = dict(
            module_length=module_length, mount_height=module_height
        )
        # self.input_parameter.update(kw_parameter)
        self.input_parameter["zenith_sun"] = solarposition.zenith
        self.input_parameter["azimuth_sun"] = solarposition.azimuth

    def calc_spectral_inplane(self):

        factors = self.simulate(spacing=self.spacing, tilt=self.tilt_angle)
        factors = factors.groupby(axis=1, level=0).mean()

        factors = factors.loc[:, factors.columns.to_series().str.contains("front")]

        spec_irrad_dir = (
            factors.loc[:, factors.columns.to_series().str.contains("direct")]
            .sum(axis=1)
            .values[:, None]
            * self.spectral_irradiance["dni"]
        )
        spec_irrad_diff = (
            factors.loc[:, factors.columns.to_series().str.contains("diffuse")]
            .sum(axis=1)
            .values[:, None]
            * self.spectral_irradiance["dhi"]
        )

        spec_irrad_inplane = spec_irrad_dir + spec_irrad_diff

        return spec_irrad_inplane
