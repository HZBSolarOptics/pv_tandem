# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from pv_tandem.utils import (
    calc_temp_from_NOCT,
    calc_current,
    interp_eqe_to_spec,
)

from typing import List, Dict, Optional
from pv_tandem import irradiance_models

from scipy.special import lambertw

class OneDiodeModel:
    """
    A class to calculate the performance of a solar cell with a one diode model.

    Parameters
    ----------
    tcJsc : float or np.ndarray
        The temperature coefficient of the short-circuit current.
    tcVoc : float or np.ndarray
        The temperature coefficient of the open-circuit voltage.
    R_shunt : float or np.ndarray
        The shunt resistance.
    R_series : float or np.ndarray
        The series resistance.
    n : float or np.ndarray
        The diode ideality factor.
    j0 : float or np.ndarray
        The reverse saturation current.

    Returns
    -------
    None

    """
    def __init__(self, tcJsc, tcVoc, R_shunt, R_series, n, j0):       
        self.tcJsc = tcJsc
        self.tcVoc = tcVoc
        
        # severeal parameters are converted to 1d arrays if are initilized to a
        # scalar, in order to ensure that broadcasting works in the IV functions

        if type(R_shunt) == np.ndarray:
            self.R_shunt = R_shunt
        else:
            self.R_shunt = np.array([R_shunt])

        if type(R_series) == np.ndarray:
            self.R_series = R_series
        else:
            self.R_series = np.array([R_series])

        if type(n) == np.ndarray:
            self.n = n
        else:
            self.n = np.array([n])

        if type(j0) == np.ndarray:
            self.j0 = j0
        else:
            self.j0 = np.array([j0])

    def calc_iv(self, Jsc, cell_temp, j_arr):
        def lambertw_exp_large(x):
            result = x - np.log(x) + np.log(x) / x
            return result
        
        def lambertwlog(x):
            res = np.zeros_like(x)
            large_x_mask = x > 10
            small_x = np.real(lambertw(np.exp(x[~large_x_mask]), tol=1e-8))
            large_x = lambertw_exp_large(x[large_x_mask])

            #x = np.where(large_x_mask, large_x, small_x)
            
            res[~large_x_mask] = small_x
            res[large_x_mask] = large_x

            return res

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
            np.log(self.j0 / 1000 * self.R_shunt)[:, None]
            + (
                self.R_shunt[:, None]
                * ((np.subtract.outer(Jsc, j_arr) + self.j0[:, None]))
                / (1000 * self.n[:, None] * Vth)
            )
            - np.log(self.n * Vth)[:, None]
        )

        try:
            V = (
                np.subtract.outer(
                    Jsc / 1000 * self.R_shunt,
                    j_arr / 1000 * (self.R_shunt + self.R_series),
                )
                - (self.n * Vth)[:, None] * lambertwlog((lambw))
                + (self.j0 / 1000 * self.R_shunt - Voc_rt + Voc)[:, None]
            )
        except:
            V = (
                (
                    (Jsc / 1000 * self.R_shunt)[:, None]
                    - (j_arr / 1000 * (self.R_shunt + self.R_series)[:, None])
                )
                - (self.n * Vth)[:, None] * lambertwlog((lambw))
                + (self.j0 / 1000 * self.R_shunt - Voc_rt + Voc)[:, None]
            )
            
        if hasattr(Jsc, "__len__"):
            return np.real(V)
        else:
            return np.real(V)[0]
            

    def calc_iv_params(self, Jsc, cell_temp, j_arr=np.linspace(0, 45, 451)):

        index = None        

        if hasattr(Jsc, "values"):
            index = Jsc.index
            Jsc = Jsc.values

        if hasattr(cell_temp, "values"):
            cell_temp = cell_temp.values

        V = self.calc_iv(Jsc, cell_temp, j_arr)
        P = V * j_arr
        idx_mpp = np.nanargmax(P, axis=1)
        Vmpp = V[np.arange(0, len(V)), idx_mpp]
        Voc = V[:, 0]
        Pmax = np.nanmax(P, axis=1)
        Jmpp = j_arr[idx_mpp]

        Jsc = j_arr[(np.argmax((V < 0), axis=1) - 1).clip(min=0)]

        FF = abs(Pmax) / abs(Jsc * Voc)

        res = pd.DataFrame(
            {
                "Voc": Voc,
                "Vmpp": Vmpp,
                "Pmax": Pmax,
                "FF": FF,
                "Jsc": Jsc,
                "Jmpp": Jmpp,
            }
        )
        
        # set index from Jsc if it was a pd.series
        
        if index is not None:
            res.index = index

        return res

class _TandemSimulator:
    """
    A class to represent a Tandem Simulator.


    Parameters
    ----------
    electrical_parameters : dict
        Electrical parameters of the One Diode Models.
    subcell_names : list
        Names of the subcells.
    eqe : pandas.Dataframe
        External quantum efficiency. The index of the DataFrame has to represent
        the wavelenghts in nm and the columns have to be named with the names used
        in the subcell_names list
    eqe_back : pandas.Dataframe, optional
        Backside external quantum efficiency if bifacial illumination is to be considered.
    bifacial : bool
        Flag to represent if the simulator is bifacial.
    electrical_models : dict
        Electrical models for each subcell.
    j_arr : ndarray
        Array that specifies for which current densities (mA/cm2) the voltage is
        evaluated.

    Examples
    --------
    >>> eqe = pd.DataFrame(index=np.arange(300,1205,5))
    >>> # Unphysical EQE, just for demonstration
    >>> eqe[['pero','si']] = 0.4
    >>> electrical_parameters = {
    >>> 	"Rsh": {"pero": 1000, "si": 3000},
    >>> 	"RsTandem": 3,
    >>> 	"j0": {"pero": 2.7e-18, "si": 1e-12},
    >>> 	"n": {"pero": 1.1, "si": 1},
    >>> 	"Temp": {"pero": 25, "si": 25},
    >>> 	"noct": {"pero": 48, "si": 48},
    >>> 	"tcJsc": {"pero": 0.0002, "si": 0.00032},
    >>> 	"tcVoc": {"pero": -0.002, "si": -0.0041},
    >>> }
    >>> tandem = TandemSimulator(eqe=eqe,
    >>> 						 electrical_parameters=electrical_parameters,
    >>> 						 subcell_names=["pero", "si"])
    >>> iv_df = tandem.calc_IV_stc()
    >>> power = iv_df.tandem * iv_df.index
    >>> idx_power_max = power.idxmax()
    >>> print(f'''
    >>> 	  Maximum efficiency: {power.max():.1f} %,
    >>> 	  Vmpp tandem: {iv_df.loc[idx_power_max,"tandem"]:.2f} V
    >>> 	  Vmpp perovskite: {iv_df.loc[idx_power_max,"pero"]:.2f} V
    >>> 	  Vmpp silicon: {iv_df.loc[idx_power_max,"si"]:.2f} V
    >>> 	  ''')
    Maximum efficiency: 30.4 %,
    Vmpp tandem: 1.78 V
    Vmpp perovskite: 1.09 V
    Vmpp silicon: 0.69 V
    """

    def __init__(
        self,
        eqe,
        electrical_parameters: Dict,
        subcell_names: List[str],
        eqe_back: Optional[float] = None,
        bifacial: bool = False,
    ) -> None:
        self.electrics = electrical_parameters
        self.subcell_names = subcell_names
        self.eqe = eqe
        self.eqe_back = eqe_back
        self.bifacial = bifacial

        self.electrical_models = {}
        self.j_arr = np.linspace(0, 45, 451)
        
        # R_series is initilized with a placeholder because it has to be individually
        # set by the __init__ function of the child classes for 2T and 4T

    def calculate_Jsc(
        self,
        spec_irrad_front: pd.DataFrame,
        spec_irrad_back: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Calculates the photocurrent from timeseries of impinging spectrum (W/nm/m²)
        and the EQE defined at class initiation.

        Parameters
        ----------
        spec_irrad_front : pandas.Dataframe
             Time series of spectral irradiance in the plane of the solar cell
             front side. The names columns of the DataFrame have to be the wavelength
             of the incidenting light in nm.

        spec_irrad_back : pandas.Dataframe
            Time series of spectral irradiance in the plane of the solar cell
            back side in case of a bifacial tandem. The names columns of the
            DataFrame have to be the wavelength of the incidenting light in nm.

        Returns
        -------
        current : pd.DataFrame
            Current generated in the subcells of the solar cell in mA/m²

        Notes
        -----
        See :ref:'plot_jph_from_spec.py'
        """
        Jsc = []
        for subcell in self.subcell_names:
            Jsc_loop = pd.Series(
                calc_current(spec_irrad_front, self.eqe[subcell]) / 10,
                name=subcell,
            )
            Jsc.append(Jsc_loop)
        Jsc = pd.concat(Jsc, axis=1)

        if self.bifacial is True:
            for subcell in self.subcell_names:
                Jsc_backside = pd.Series(
                    calc_current(spec_irrad_back, self.eqe_back[subcell]) / 10,
                    name=subcell,
                )
                Jsc[subcell] = Jsc[subcell] + Jsc_backside

        return Jsc

    def calc_IV_individual(self, Jsc, cell_temps):
        """
        Calculates the IV curves on the grid spcified by j_arr from the photocurrent
        of the individual cells.

        Parameters
        ----------
        Jsc : pandas.Dataframe
            Time series of spectral irradiance in the plane of the solar cell
            back side in case of a bifacial tandem. The names columns of the
            DataFrame have to be the wavelength of the incidenting light in nm.

        cell_temps : pandas.Dataframe
             Time series of the cell temperatures. The columns of the dataframe
             expected to be named like the subcell_names.

        return_subsells : Bool
            Defines if the voltage of the subcells should be returned alongside
            the tandem voltage.

        Returns
        -------
        If return_subsells is False:
        V_tandem : pd.DataFrame
            Dataframe containing IV data where the rows represent the timestamps
            of the Jsc timeseries and the columns the current generated by the
            cells (in mA/cm2)

        If return_subsells is True:
        V_tandem : pd.DataFrame
            Dataframe containing IV data where the rows represent the timestamps
            of the Jsc timeseries and the columns the current generated by the
            cells (in mA/cm2)
        V : pd.DataFrame
            Dataframe containing IV data where the rows represent the timestamps
            of the Jsc timeseries and the columns are a multiindex, where the
            first level represents the subcells and the second level the current
            generated by the cells (in mA/cm2)

        Example
        -------
        See :ref:`sphx_glr_auto_examples_plot_tandem_ey.py` for a usage example.
        """

        V = []

        for subcell in self.subcell_names:
            # if type(cell_temps) == pd.Series:
            #    cell_temps = self.cell_temps[subcell].values

            V.append(
                pd.DataFrame(
                    self.electrical_models[subcell].calc_iv(
                        Jsc[subcell].values,
                        cell_temps[subcell].values,
                        self.j_arr,
                    ),
                    columns=self.j_arr,
                )
            )

        V = pd.concat(V, axis=1, keys=self.subcell_names)
        return V


    def calc_IV_stc(self, backside_current=None):
        """
        Calculates the voltage for the IV curve of the tandem solar cell under
        standart test conditions at the respective current density defined by
        j_arr.

        Parameters
        ----------
        backside_current : dict
             Manual backside current contribution (in mA/cm2) for bifacial tandem.

        Returns
        -------
        Voltage : pd.DataFrame
            DataFrame with the subcell and combined tandem voltage at the respective
            current density (mA/cm2) as index.

        Example
        -------
        Examples
        --------
        >>> eqe = pd.DataFrame(index=np.arange(300,1205,5))
        >>> # Unphysical EQE, just for demonstration
        >>> eqe[['pero','si']] = 0.4
        >>> electrical_parameters = {
        >>> 	"Rsh": {"pero": 1000, "si": 3000},
        >>> 	"RsTandem": 3,
        >>> 	"j0": {"pero": 2.7e-18, "si": 1e-12},
        >>> 	"n": {"pero": 1.1, "si": 1},
        >>> 	"Temp": {"pero": 25, "si": 25},
        >>> 	"noct": {"pero": 48, "si": 48},
        >>> 	"tcJsc": {"pero": 0.0002, "si": 0.00032},
        >>> 	"tcVoc": {"pero": -0.002, "si": -0.0041},
        >>> }
        >>> tandem = TandemSimulator(eqe=eqe,
        >>> 						 electrical_parameters=electrical_parameters,
        >>> 						 subcell_names=["pero", "si"])
        >>> iv_df = tandem.calc_IV_stc()
        >>> power = iv_df.tandem * iv_df.index
        >>> idx_power_max = power.idxmax()
        >>> print(f'''
        >>> 	  Maximum efficiency: {power.max():.1f} %,
        >>> 	  Vmpp tandem: {iv_df.loc[idx_power_max,"tandem"]:.2f} V
        >>> 	  Vmpp perovskite: {iv_df.loc[idx_power_max,"pero"]:.2f} V
        >>> 	  Vmpp silicon: {iv_df.loc[idx_power_max,"si"]:.2f} V
        >>> 	  ''')
        Maximum efficiency: 30.4 %,
        Vmpp tandem: 1.78 V
        Vmpp perovskite: 1.09 V
        Vmpp silicon: 0.69 V
        """

        j_ph_stc = irradiance_models.AM15g().calc_jph(self.eqe) / 10

        if backside_current is not None:
            j_ph_stc = j_ph_stc + backside_current

        V = []

        for subcell in self.subcell_names:
            V.append(
                pd.DataFrame(
                    self.electrical_models[subcell].calc_iv(
                        np.array([j_ph_stc[subcell]]),
                        25,
                        self.j_arr,
                    )
                )
            )

        V_tandem = pd.concat(V, axis=1, keys=self.subcell_names)
        V_tandem = V_tandem.stack().reset_index(drop=True).astype(float)

        V_tandem["tandem"] = V_tandem.sum(axis=1)
        V_tandem.index = self.j_arr
        V_tandem.index = V_tandem.index.rename("current")

        return V_tandem


class TandemSimulator2T(_TandemSimulator):
    """
    A class to represent a Tandem Simulator.


    Parameters
    ----------
    electrical_parameters : dict
        Electrical parameters of the One Diode Models.
    subcell_names : list
        Names of the subcells.
    eqe : pandas.Dataframe
        External quantum efficiency. The index of the DataFrame has to represent
        the wavelenghts in nm and the columns have to be named with the names used
        in the subcell_names list
    eqe_back : pandas.Dataframe, optional
        Backside external quantum efficiency if bifacial illumination is to be considered.
    bifacial : bool
        Flag to represent if the simulator is bifacial.
    electrical_models : dict
        Electrical models for each subcell.
    j_arr : ndarray
        Array that specifies for which current densities (mA/cm2) the voltage is
        evaluated.

    Examples
    --------
    >>> eqe = pd.DataFrame(index=np.arange(300,1205,5))
    >>> # Unphysical EQE, just for demonstration
    >>> eqe[['pero','si']] = 0.4
    >>> electrical_parameters = {
    >>> 	"Rsh": {"pero": 1000, "si": 3000},
    >>> 	"RsTandem": 3,
    >>> 	"j0": {"pero": 2.7e-18, "si": 1e-12},
    >>> 	"n": {"pero": 1.1, "si": 1},
    >>> 	"Temp": {"pero": 25, "si": 25},
    >>> 	"noct": {"pero": 48, "si": 48},
    >>> 	"tcJsc": {"pero": 0.0002, "si": 0.00032},
    >>> 	"tcVoc": {"pero": -0.002, "si": -0.0041},
    >>> }
    >>> tandem = TandemSimulator(eqe=eqe,
    >>> 						 electrical_parameters=electrical_parameters,
    >>> 						 subcell_names=["pero", "si"])
    >>> iv_df = tandem.calc_IV_stc()
    >>> power = iv_df.tandem * iv_df.index
    >>> idx_power_max = power.idxmax()
    >>> print(f'''
    >>> 	  Maximum efficiency: {power.max():.1f} %,
    >>> 	  Vmpp tandem: {iv_df.loc[idx_power_max,"tandem"]:.2f} V
    >>> 	  Vmpp perovskite: {iv_df.loc[idx_power_max,"pero"]:.2f} V
    >>> 	  Vmpp silicon: {iv_df.loc[idx_power_max,"si"]:.2f} V
    >>> 	  ''')
    Maximum efficiency: 30.4 %,
    Vmpp tandem: 1.78 V
    Vmpp perovskite: 1.09 V
    Vmpp silicon: 0.69 V
    """

    def __init__(
        self,
        eqe,
        electrical_parameters: Dict,
        subcell_names: List[str],
        eqe_back: Optional[float] = None,
        bifacial: bool = False,
    ) -> None:
        super().__init__(
            eqe, electrical_parameters, subcell_names, eqe_back, bifacial
        )

        for subcell in subcell_names:
            self.electrical_models[subcell] = OneDiodeModel(
                tcJsc=self.electrics["tcJsc"][subcell],
                tcVoc=self.electrics["tcVoc"][subcell],
                R_shunt=self.electrics["Rsh"][subcell],
                R_series=self.electrics["RsTandem"]/2,
                n=self.electrics["n"][subcell],
                j0=self.electrics["j0"][subcell],
            )

    def calc_IV(self, Jsc, cell_temps, return_subsells=False):
        """
        Calculates the IV curves on the grid spcified by j_arr from the photocurrent
        of the individual cells.

        Parameters
        ----------
        Jsc : pandas.Dataframe
            Time series of spectral irradiance in the plane of the solar cell
            back side in case of a bifacial tandem. The names columns of the
            DataFrame have to be the wavelength of the incidenting light in nm.

        cell_temps : pandas.Dataframe
             Time series of the cell temperatures. The columns of the dataframe
             expected to be named like the subcell_names.

        return_subsells : Bool
            Defines if the voltage of the subcells should be returned alongside
            the tandem voltage.

        Returns
        -------
        If return_subsells is False:
        V_tandem : pd.DataFrame
            Dataframe containing IV data where the rows represent the timestamps
            of the Jsc timeseries and the columns the current generated by the
            cells (in mA/cm2)

        If return_subsells is True:
        V_tandem : pd.DataFrame
            Dataframe containing IV data where the rows represent the timestamps
            of the Jsc timeseries and the columns the current generated by the
            cells (in mA/cm2)
        V : pd.DataFrame
            Dataframe containing IV data where the rows represent the timestamps
            of the Jsc timeseries and the columns are a multiindex, where the
            first level represents the subcells and the second level the current
            generated by the cells (in mA/cm2)

        Example
        -------
        See :ref:`sphx_glr_auto_examples_plot_tandem_ey.py` for a usage example.
        """

        V = self.calc_IV_individual(Jsc, cell_temps)

        V_tandem = V.groupby(level=1, axis=1).sum()

        if return_subsells:
            return V_tandem, V
        else:
            return V_tandem

    def calc_power(
        self,
        spec_irrad: pd.DataFrame,
        cell_temps: pd.DataFrame,
        backside_current: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """
        Calculates the maximum power output density from timeseries of impinging spectrum
        (W/nm/m²) and the cell temperature.

        Parameters
        ----------
        spec_irrad_front : pandas.Dataframe
             Time series of spectral irradiance in the plane of the solar cell front side. The
             names columns of the DataFrame have to be the wavelength of the incidenting
             light in nm.

        cell_temps : pandas.Dataframe
             Time series of the cell temperatures. The columns of the dataframe
             expected to be named like the subcell_names.

        backside_current : pandas.Dataframe
             Manual backside current contribution (in mA/cm2) for bifacial tandem.
             The DataFrame needs to contain a column for each subcell in the tandem.

        Returns
        -------
        Power : pd.Series
            Time series of power output density at the maximum power point.

        Example
        -------
        See :ref:`sphx_glr_auto_examples_plot_tandem_ey.py` for a usage example.
        """

        if self.bifacial is True:
            Jsc = self.calculate_Jsc(spec_irrad["front"], spec_irrad["back"])
        else:
            Jsc = self.calculate_Jsc(spec_irrad)

        if backside_current is not None:
            Jsc = Jsc + backside_current

        V_tandem = self.calc_IV(Jsc, cell_temps)

        P = V_tandem.values * self.j_arr[None, :]
        P_max = P.max(axis=1)
        P_max = pd.Series(P_max, index=spec_irrad.index)
        return P_max

    def calc_IV_stc(self, backside_current=None):
        """
        Calculates the voltage for the IV curve of the tandem solar cell under
        standart test conditions at the respective current density defined by
        j_arr.

        Parameters
        ----------
        backside_current : dict
             Manual backside current contribution (in mA/cm2) for bifacial tandem.

        Returns
        -------
        Voltage : pd.DataFrame
            DataFrame with the subcell and combined tandem voltage at the respective
            current density (mA/cm2) as index.

        Example
        -------
        >>> eqe = pd.DataFrame(index=np.arange(300,1205,5))
        >>> # Unphysical EQE, just for demonstration
        >>> eqe[['pero','si']] = 0.4
        >>> electrical_parameters = {
        >>> 	"Rsh": {"pero": 1000, "si": 3000},
        >>> 	"RsTandem": 3,
        >>> 	"j0": {"pero": 2.7e-18, "si": 1e-12},
        >>> 	"n": {"pero": 1.1, "si": 1},
        >>> 	"Temp": {"pero": 25, "si": 25},
        >>> 	"noct": {"pero": 48, "si": 48},
        >>> 	"tcJsc": {"pero": 0.0002, "si": 0.00032},
        >>> 	"tcVoc": {"pero": -0.002, "si": -0.0041},
        >>> }
        >>> tandem = TandemSimulator(eqe=eqe,
        >>> 						 electrical_parameters=electrical_parameters,
        >>> 						 subcell_names=["pero", "si"])
        >>> iv_df = tandem.calc_IV_stc()
        >>> power = iv_df.tandem * iv_df.index
        >>> idx_power_max = power.idxmax()
        >>> print(f'''
        >>> 	  Maximum efficiency: {power.max():.1f} %,
        >>> 	  Vmpp tandem: {iv_df.loc[idx_power_max,"tandem"]:.2f} V
        >>> 	  Vmpp perovskite: {iv_df.loc[idx_power_max,"pero"]:.2f} V
        >>> 	  Vmpp silicon: {iv_df.loc[idx_power_max,"si"]:.2f} V
        >>> 	  ''')
        Maximum efficiency: 30.4 %,
        Vmpp tandem: 1.78 V
        Vmpp perovskite: 1.09 V
        Vmpp silicon: 0.69 V
        """

        j_ph_stc = irradiance_models.AM15g().calc_jph(self.eqe) / 10

        if backside_current is not None:
            j_ph_stc = j_ph_stc + backside_current

        _, V_tandem = self.calc_IV(
            j_ph_stc.to_frame().T,
            pd.Series([25, 25], index=self.subcell_names).to_frame().T,
            return_subsells=True,
        )
        V_tandem = V_tandem.stack().reset_index(drop=True).astype(float)

        V_tandem["tandem"] = V_tandem.sum(axis=1)
        V_tandem.index = self.j_arr
        V_tandem.index = V_tandem.index.rename("current")

        return V_tandem


class TandemSimulator4T(_TandemSimulator):
    """
    A class to represent a 4 terminal Tandem Simulator.


    Parameters
    ----------
    electrical_parameters : dict
        Electrical parameters of the One Diode Models.
    subcell_names : list
        Names of the subcells.
    eqe : pandas.Dataframe
        External quantum efficiency. The index of the DataFrame has to represent
        the wavelenghts in nm and the columns have to be named with the names used
        in the subcell_names list
    eqe_back : pandas.Dataframe, optional
        Backside external quantum efficiency if bifacial illumination is to be considered.
    bifacial : bool
        Flag to represent if the simulator is bifacial.
    electrical_models : dict
        Electrical models for each subcell.
    j_arr : ndarray
        Array that specifies for which current densities (mA/cm2) the voltage is
        evaluated.

    Examples
    --------
    >>> eqe = pd.DataFrame(index=np.arange(300,1205,5))
    >>> # Unphysical EQE, just for demonstration
    >>> eqe[['pero','si']] = 0.4
    >>> electrical_parameters = {
    >>> 	"Rsh": {"pero": 1000, "si": 3000},
    >>> 	"RsTandem": 3,
    >>> 	"j0": {"pero": 2.7e-18, "si": 1e-12},
    >>> 	"n": {"pero": 1.1, "si": 1},
    >>> 	"Temp": {"pero": 25, "si": 25},
    >>> 	"noct": {"pero": 48, "si": 48},
    >>> 	"tcJsc": {"pero": 0.0002, "si": 0.00032},
    >>> 	"tcVoc": {"pero": -0.002, "si": -0.0041},
    >>> }
    >>> tandem = TandemSimulator(eqe=eqe,
    >>> 						 electrical_parameters=electrical_parameters,
    >>> 						 subcell_names=["pero", "si"])
    >>> iv_df = tandem.calc_IV_stc()
    >>> power = iv_df.tandem * iv_df.index
    >>> idx_power_max = power.idxmax()
    >>> print(f'''
    >>> 	  Maximum efficiency: {power.max():.1f} %,
    >>> 	  Vmpp tandem: {iv_df.loc[idx_power_max,"tandem"]:.2f} V
    >>> 	  Vmpp perovskite: {iv_df.loc[idx_power_max,"pero"]:.2f} V
    >>> 	  Vmpp silicon: {iv_df.loc[idx_power_max,"si"]:.2f} V
    >>> 	  ''')
    Maximum efficiency: 30.4 %,
    Vmpp tandem: 1.78 V
    Vmpp perovskite: 1.09 V
    Vmpp silicon: 0.69 V
    """

    def __init__(
        self,
        eqe,
        electrical_parameters: Dict,
        subcell_names: List[str],
        eqe_back: Optional[float] = None,
        bifacial: bool = False,
    ) -> None:
        super().__init__(
            eqe, electrical_parameters, subcell_names, eqe_back, bifacial
        )

        for subcell in subcell_names:
            self.electrical_models[subcell] = OneDiodeModel(
                tcJsc=self.electrics["tcJsc"][subcell],
                tcVoc=self.electrics["tcVoc"][subcell],
                R_shunt=self.electrics["Rsh"][subcell],
                R_series=self.electrics["Rs"][subcell],
                n=self.electrics["n"][subcell],
                j0=self.electrics["j0"][subcell],
            )

    def calc_IV(self, Jsc, cell_temps, return_subsells=False):
        """
        Calculates the IV curves on the grid spcified by j_arr from the photocurrent
        of the individual cells.

        Parameters
        ----------
        Jsc : pandas.Dataframe
            Time series of spectral irradiance in the plane of the solar cell
            back side in case of a bifacial tandem. The names columns of the
            DataFrame have to be the wavelength of the incidenting light in nm.

        cell_temps : pandas.Dataframe
             Time series of the cell temperatures. The columns of the dataframe
             expected to be named like the subcell_names.

        Returns
        -------
        If return_subsells is False:
        V_tandem : pd.DataFrame
            Dataframe containing IV data where the rows represent the timestamps
            of the Jsc timeseries and the columns the current generated by the
            cells (in mA/cm2)

        If return_subsells is True:
        V_tandem : pd.DataFrame
            Dataframe containing IV data where the rows represent the timestamps
            of the Jsc timeseries and the columns the current generated by the
            cells (in mA/cm2)
        V : pd.DataFrame
            Dataframe containing IV data where the rows represent the timestamps
            of the Jsc timeseries and the columns are a multiindex, where the
            first level represents the subcells and the second level the current
            generated by the cells (in mA/cm2)

        Example
        -------
        See :ref:`sphx_glr_auto_examples_plot_tandem_ey.py` for a usage example.
        """
        
        V = self.calc_IV_individual(Jsc, cell_temps)

        return V

    def calc_power(
        self,
        spec_irrad: pd.DataFrame,
        cell_temps: pd.DataFrame,
        backside_current: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """
        Calculates the maximum power output density from timeseries of impinging spectrum
        (W/nm/m²) and the cell temperature.

        Parameters
        ----------
        spec_irrad_front : pandas.Dataframe
             Time series of spectral irradiance in the plane of the solar cell front side. The
             names columns of the DataFrame have to be the wavelength of the incidenting
             light in nm.

        cell_temps : pandas.Dataframe
             Time series of the cell temperatures. The columns of the dataframe
             expected to be named like the subcell_names.

        backside_current : pandas.Dataframe
             Manual backside current contribution (in mA/cm2) for bifacial tandem.
             The DataFrame needs to contain a column for each subcell in the tandem.

        Returns
        -------
        Power : pd.Series
            Time series of power output density at the maximum power point.

        Example
        -------
        See :ref:`sphx_glr_auto_examples_plot_tandem_ey.py` for a usage example.
        """

        if self.bifacial is True:
            Jsc = self.calculate_Jsc(spec_irrad["front"], spec_irrad["back"])
        else:
            Jsc = self.calculate_Jsc(spec_irrad)

        if backside_current is not None:
            Jsc = Jsc + backside_current

        V_tandem = self.calc_IV(Jsc, cell_temps)

        P = V_tandem.values * self.j_arr[None, :]
        P_max = P.max(axis=1)
        P_max = pd.Series(P_max, index=spec_irrad.index)
        return P_max

    def calc_IV_stc(self, backside_current=None):
        """
        Calculates the voltage for the IV curve of the tandem solar cell under
        standart test conditions at the respective current density defined by
        j_arr.

        Parameters
        ----------
        backside_current : dict
             Manual backside current contribution (in mA/cm2) for bifacial tandem.

        Returns
        -------
        Voltage : pd.DataFrame
            DataFrame with the subcell and combined tandem voltage at the respective
            current density (mA/cm2) as index.

        Example
        -------
        >>> eqe = pd.DataFrame(index=np.arange(300,1205,5))
        >>> # Unphysical EQE, just for demonstration
        >>> eqe[['pero','si']] = 0.4
        >>> electrical_parameters = {
        >>> 	"Rsh": {"pero": 1000, "si": 3000},
        >>> 	"RsTandem": 3,
        >>> 	"j0": {"pero": 2.7e-18, "si": 1e-12},
        >>> 	"n": {"pero": 1.1, "si": 1},
        >>> 	"Temp": {"pero": 25, "si": 25},
        >>> 	"noct": {"pero": 48, "si": 48},
        >>> 	"tcJsc": {"pero": 0.0002, "si": 0.00032},
        >>> 	"tcVoc": {"pero": -0.002, "si": -0.0041},
        >>> }
        >>> tandem = TandemSimulator(eqe=eqe,
        >>> 						 electrical_parameters=electrical_parameters,
        >>> 						 subcell_names=["pero", "si"])
        >>> iv_df = tandem.calc_IV_stc()
        >>> power = iv_df.tandem * iv_df.index
        >>> idx_power_max = power.idxmax()
        >>> print(f'''
        >>> 	  Maximum efficiency: {power.max():.1f} %,
        >>> 	  Vmpp tandem: {iv_df.loc[idx_power_max,"tandem"]:.2f} V
        >>> 	  Vmpp perovskite: {iv_df.loc[idx_power_max,"pero"]:.2f} V
        >>> 	  Vmpp silicon: {iv_df.loc[idx_power_max,"si"]:.2f} V
        >>> 	  ''')
        Maximum efficiency: 30.4 %,
        Vmpp tandem: 1.78 V
        Vmpp perovskite: 1.09 V
        Vmpp silicon: 0.69 V
        """

        j_ph_stc = irradiance_models.AM15g().calc_jph(self.eqe) / 10

        if backside_current is not None:
            j_ph_stc = j_ph_stc + backside_current

        V_tandem = self.calc_IV(
            j_ph_stc.to_frame().T,
            pd.Series([25, 25], index=self.subcell_names).to_frame().T,
            return_subsells=True,
        )
        V_tandem = V_tandem.stack().reset_index(drop=True).astype(float)
        V_tandem.index = self.j_arr

        return V_tandem


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    spec = pd.read_csv("../examples/data/tiny_spec.csv", index_col=0)
    spec.columns = spec.columns.astype(float)
    spec = spec / 1000
    eqe = pd.read_csv("../examples/data/eqe_tandem_2t.csv", index_col=0)

    eqe_new = interp_eqe_to_spec(eqe, spec)

    eqe = pd.DataFrame(index=np.arange(300, 1205, 5))
    # Unphysical EQE, just for demonstration
    eqe[["pero", "si"]] = 0.4

    electrical_parameters = {
        "Rsh": {"pero": 1000, "si": 3000},
        "RsTandem": 3,
        "j0": {"pero": 2.7e-18, "si": 1e-12},
        "n": {"pero": 1.1, "si": 1},
        "Temp": {"pero": 25, "si": 25},
        "noct": {"pero": 48, "si": 48},
        "tcJsc": {"pero": 0.0002, "si": 0.00032},
        "tcVoc": {"pero": -0.002, "si": -0.0041},
    }

    tandem = TandemSimulator2T(
        eqe=eqe,
        electrical_parameters=electrical_parameters,
        subcell_names=["pero", "si"],
    )

    iv_df = tandem.calc_IV_stc()
    power = iv_df.tandem * iv_df.index
    idx_power_max = power.idxmax()

    print(
        f"""
          Maximum efficiency: {power.max():.1f} %,
          Vmpp tandem: {iv_df.loc[idx_power_max,'tandem']:.2f} V
          Vmpp perovskite: {iv_df.loc[idx_power_max,'pero']:.2f} V
          Vmpp silicon: {iv_df.loc[idx_power_max,'si']:.2f} V
          """
    )

    eqe = pd.DataFrame(index=np.arange(300, 1205, 5))
    # Unphysical EQE, just for demonstration
    eqe[["pero", "si"]] = 0.4

    electrical_parameters = {
        "Rsh": {"pero": 1000, "si": 3000},
        "Rs": {"pero":1.5,"si":1.5},
        "j0": {"pero": 2.7e-18, "si": 1e-12},
        "n": {"pero": 1.1, "si": 1},
        "Temp": {"pero": 25, "si": 25},
        "noct": {"pero": 48, "si": 48},
        "tcJsc": {"pero": 0.0002, "si": 0.00032},
        "tcVoc": {"pero": -0.002, "si": -0.0041},
    }

    tandem = TandemSimulator4T(
        eqe=eqe,
        electrical_parameters=electrical_parameters,
        subcell_names=["pero", "si"],
    )

    iv_df = tandem.calc_IV_stc()
    
    
    power = iv_df.multiply(iv_df.index, axis=0)
    
    idx_power_max = power.idxmax()
    max_power = power.max().sum()
    V_mpp_pero = iv_df.loc[idx_power_max['pero'], 'pero']
    V_mpp_si = iv_df.loc[idx_power_max['si'], 'si']

    print(
        f"""
          Maximum efficiency: {max_power:.1f} %,
          Vmpp perovskite: {V_mpp_pero:.2f} V
          Vmpp silicon: {V_mpp_si:.2f} V
          """
    )

    asdf
    plt.plot(power)
