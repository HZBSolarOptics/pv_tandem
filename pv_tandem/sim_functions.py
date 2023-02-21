# -*- coding: utf-8 -*-

import scipy.io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import lambertw
from scipy import constants
import pvlib
import bifacial_illumination as bi
import os

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
    if len(spec.shape) > 1:
        #spec is timeseries of spectral illumination
        norm_absorbtion = spec.multiply(eqe, axis=1)
        wl_arr = norm_absorbtion.columns.to_series()
        photon_flux = (norm_absorbtion / constants.h / constants.c).multiply(
            wl_arr * 1e-9, axis=1
        )
        current = np.trapz(photon_flux, x=wl_arr) * constants.e
    else:
        #spec is single spectrum
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


class OneDiodeModel:
    def __init__(self, tcJsc, tcVoc, R_shunt, R_series, n, j0):
        self.tcJsc = tcJsc
        self.tcVoc = tcVoc
        self.R_shunt = R_shunt
        self.R_series = R_series
        self.n = n
        self.j0 = j0

    def calc_iv(self, Jsc, cell_temp, j_arr):

        def lambertw_exp_large(x):
            result = x-np.log(x)+np.log(x)/x
            return result        

        def lambertwlog(x):
            large_x = x.copy()
            large_x_mask = x>20
            small_x = lambertw(np.exp(np.clip(x, a_min=None, a_max=20)))
            large_x = lambertw_exp_large(x)
            
            x = np.where(large_x_mask, large_x, small_x)
            
            return x
                        
        # Thermal voltage at room temperature in V
        Vth = 0.02569

        factorJsc = 1 + self.tcJsc * (cell_temp - 25)
        factorVoc = 1 + self.tcVoc * (cell_temp - 25)

        Jsc = Jsc * factorJsc
        
        Voc_rt = (
            Jsc / 1000 * self.R_shunt
            - self.n
            * Vth
            * lambertwlog(
                (
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
            - self.n * Vth * lambertwlog((lambw))
            + (self.j0 / 1000 * self.R_shunt - Voc_rt + Voc)[:, None]
        )

        #V[(V < 0) | (np.isnan(V))] = 0
        
        #V_oc = V.max(axis=1)
        #P = V*j_arr[None, :]
        #P_max = P.max(axis=1)
        
        #V_mpp = V[np.arange(0,8760),P.argmax(axis=1)]

        return V


class TandemSimulator:
    def __init__(
        self,
        spec_irrad,
        eqe,
        electrical_parameters,
        subcell_names,
        ambient_temp,
        temp_model='noct',
        min_Jsc_both_cells=True,
        eqe_back=None,
        bifacial=False,
    ):
        self.electrics = electrical_parameters
        self.subcell_names = subcell_names
        self.spec_irrad = spec_irrad
        self.eqe = eqe
        self.eqe_back = eqe_back
        self.bifacial = bifacial
        self.Jsc = self.calculate_Jsc(min_Jsc_both_cells)

        self.electrical_models = {}
        self.j_arr = np.linspace(0, 45, 451)
        if temp_model is None:
            self.cell_temps = {
                subcell: pd.Series(25)
                for subcell in subcell_names
            }
        else:
            self.cell_temps = {
                subcell: calc_temp_from_NOCT(
                    self.electrics["noct"][subcell],
                    ambient_temp,
                    spec_irrad["front"].sum(axis=-1),
                )
                # correction for limited range of spectrum (onyl until 1200 nm)
                * 1.15
                for subcell in subcell_names
            }

        for subcell in subcell_names:
            self.electrical_models[subcell] = OneDiodeModel(
                tcJsc=self.electrics["tcJsc"][subcell],
                tcVoc=self.electrics["tcVoc"][subcell],
                R_shunt=self.electrics["Rsh"][subcell],
                R_series=self.electrics["RsTandem"]/2,
                n=self.electrics["n"][subcell],
                j0=self.electrics["j0"][subcell],
            )

    def calculate_Jsc(self, min_Jsc_both_cells):

        Jsc = []
        for subcell in self.subcell_names:
            Jsc_loop = pd.Series(
                calc_current(self.spec_irrad["front"], self.eqe[subcell]) / 10,
                name=subcell,
            )
            Jsc.append(Jsc_loop)
        Jsc = pd.concat(Jsc, axis=1)

        if self.bifacial is True:
            for subcell in self.subcell_names:
                Jsc_backside = pd.Series(
                    calc_current(self.spec_irrad["back"], self.eqe_back[subcell]) / 10,
                    name=subcell,
                )
                Jsc[subcell] = Jsc[subcell] + Jsc_backside

        if min_Jsc_both_cells:
            Jsc_min = Jsc.min(axis=1)
            for subcell in self.subcells:
                Jsc[subcell] = Jsc_min

        return Jsc
    
    def calc_IV(self, Jsc, return_subsells=False):
        V = []
        
        for subcell in self.subcell_names:
            #if type(cell_temps) == pd.Series:
            #    cell_temps = self.cell_temps[subcell].values
            
            V.append(
                pd.DataFrame(
                    self.electrical_models[subcell].calc_iv(
                        Jsc[subcell].values,
                        self.cell_temps[subcell].values,
                        self.j_arr,
                    )
                )
            )
            
        V = pd.concat(V, axis=1, keys=self.subcell_names)
        V = V.astype(float)
        V[V<0] = np.nan
        V_tandem = V.groupby(level=1, axis=1).aggregate(lambda x: np.sum(x.values))#.apply(lambda x: np.sum(x))

        if return_subsells:
            return V_tandem, V
        else:
            return V_tandem

    def calc_power(self):
        V = []
        
        for subcell in self.subcell_names:
            #if type(cell_temps) == pd.Series:
            #    cell_temps = self.cell_temps[subcell].values
            
            V.append(
                pd.DataFrame(
                    self.electrical_models[subcell].calc_iv(
                        self.Jsc[subcell].values,
                        self.cell_temps[subcell].values,
                        self.j_arr,
                    )
                )
            )
        V = pd.concat(V, axis=1, keys=self.subcell_names)
        V = V.astype(float)
        V_tandem = V.groupby(level=1, axis=1).sum()

        P = V_tandem.values * self.j_arr[None, :]
        P_max = P.max(axis=1)
        P_max = pd.Series(P_max)
        return P_max

class AM15g():
    def __init__(
            self):
        csv_file_path = os.path.join(os.path.dirname(__file__),
                                     'data', 'ASTMG173.csv')
        self.spec = pd.read_csv(csv_file_path, sep=';')
        self.spec.columns = ['wavelength', 'extra_terra', 'global', 'direct']
        self.spec = self.spec.set_index('wavelength')['global']
        
    def interpolate(self, wavelengths):
        spec_return = pd.Series(
            np.interp(wavelengths, self.spec.index, self.spec),
            index = wavelengths
            )
        return spec_return

class spectral_illumination(bi.YieldSimulator):
    def __init__(
        self,
        spectral_irradiance,
        solarposition,
        bifacial=True,
        albedo=0.3,
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

        def scale_irradiance(dni, dhi, factors):
            spec_irrad_dir = (
                factors.loc[:, factors.columns.to_series().str.contains("direct")]
                .sum(axis=1)
                .values[:, None]
                * dni
            )
            spec_irrad_diff = (
                factors.loc[:, factors.columns.to_series().str.contains("diffuse")]
                .sum(axis=1)
                .values[:, None]
                * dhi
            )
            spec_irrad_inplane = spec_irrad_dir + spec_irrad_diff
            return spec_irrad_inplane

        factors_front = factors.loc[
            :, factors.columns.to_series().str.contains("front")
        ]
        factors_back = factors.loc[:, factors.columns.to_series().str.contains("back")]

        spec_irrad_inplane_front = scale_irradiance(
            self.spectral_irradiance["dni"],
            self.spectral_irradiance["dhi"],
            factors_front,
        )

        spec_irrad_inplane_back = scale_irradiance(
            self.spectral_irradiance["dni"],
            self.spectral_irradiance["dhi"],
            factors_back,
        )

        spec_irrad_inplane = pd.concat(
            [spec_irrad_inplane_front, spec_irrad_inplane_back],
            axis=1,
            keys=["front", "back"],
        )

        return spec_irrad_inplane
