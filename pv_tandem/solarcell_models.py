# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from pv_tandem.utils import (
    calc_temp_from_NOCT,
    calc_current,
    interp_eqe_to_spec,
)
from pv_tandem.electrical_models import OneDiodeModel
from typing import List, Dict, Optional
from pv_tandem import electrical_models, irradiance_models, utils

class TandemSimulator:
    """
    A class to represent a Tandem Simulator.


    Parameters
    ----------
    electrics : dict
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

        for subcell in subcell_names:
            self.electrical_models[subcell] = OneDiodeModel(
                tcJsc=self.electrics["tcJsc"][subcell],
                tcVoc=self.electrics["tcVoc"][subcell],
                R_shunt=self.electrics["Rsh"][subcell],
                R_series=self.electrics["RsTandem"] / 2,
                n=self.electrics["n"][subcell],
                j0=self.electrics["j0"][subcell],
            )


    def calculate_Jsc(self, spec_irrad_front: pd.DataFrame, spec_irrad_back:Optional[pd.DataFrame] = None)-> pd.DataFrame:
        """
        Calculates the photocurrent from timeseries of impinging spectrum (W/nm/m²) and the EQE defined at class initiation.
        
        Parameters
        ----------
        spec_irrad_front : pandas.Dataframe
             Time series of spectral irradiance in the plane of the solar cell front side. The
             names columns of the DataFrame have to be the wavelength of the incidenting
             light in nm.
        
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
                    calc_current(
                        spec_irrad_back, self.eqe_back[subcell]
                    )
                    / 10,
                    name=subcell,
                )
                Jsc[subcell] = Jsc[subcell] + Jsc_backside

        return Jsc

    def calc_IV(self, Jsc, cell_temps, return_subsells=False):
        """
        Calculates the IV curves on the grid spcified by j_arr from the photocurrent of the individual cells.
        
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
            
        Notes
        -----
        See :ref:`sphx_glr_auto_examples_plot_tandem_ey.py`
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
                    columns = self.j_arr
                )
            )

        V = pd.concat(V, axis=1, keys=self.subcell_names)
        V_tandem = V.groupby(level=1, axis=1).sum()

        if return_subsells:
            return V_tandem, V
        else:
            return V_tandem

    def calc_power(self, spec_irrad, cell_temps):
        
        
        
        if self.bifacial is True:
            Jsc = self.calculate_Jsc(spec_irrad['front'], spec_irrad['back'])
        else:
            Jsc = self.calculate_Jsc(spec_irrad)
        
        V_tandem = self.calc_IV(Jsc, cell_temps)
        
        P = V_tandem.values * self.j_arr[None, :]
        P_max = P.max(axis=1)
        P_max = pd.Series(P_max)
        return P_max
    
    def calc_IV_stc(self):
        
        j_ph_stc = irradiance_models.AM15g().calc_jph(self.eqe) / 10
        
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
        
        V_tandem['tandem'] = V_tandem.sum(axis=1)
        V_tandem.index = self.j_arr
        
        return V_tandem


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    spec = pd.read_csv("../examples/data/tiny_spec.csv", index_col=0)
    spec.columns = spec.columns.astype(float)
    spec = spec / 1000
    eqe = pd.read_csv("../examples/data/eqe_tandem_2t.csv", index_col=0)

    eqe_new = interp_eqe_to_spec(eqe, spec)
    
    eqe = pd.DataFrame(index=np.arange(300,1205,5))
    # Unphysical EQE, just for demonstration
    eqe[['pero','si']] = 0.4

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

    tandem = TandemSimulator(eqe=eqe,
                             electrical_parameters=electrical_parameters,
                             subcell_names=["pero", "si"])


    iv_df = tandem.calc_IV_stc()
    power = iv_df.tandem * iv_df.index
    idx_power_max = power.idxmax()
    
    print(f"""
          Maximum efficiency: {power.max():.1f} %,
          Vmpp tandem: {iv_df.loc[idx_power_max,'tandem']:.2f} V
          Vmpp perovskite: {iv_df.loc[idx_power_max,'pero']:.2f} V
          Vmpp silicon: {iv_df.loc[idx_power_max,'si']:.2f} V
          """)
          
    
    asdf
    plt.plot(power)
