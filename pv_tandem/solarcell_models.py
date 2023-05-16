# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from pv_tandem.utils import (
    calc_temp_from_NOCT,
    calc_current,
    interp_eqe_to_spec,
)
from pv_tandem.electrical_models import OneDiodeModel

from pv_tandem import electrical_models, irradiance_models, utils

class TandemSimulator:
    def __init__(
        self,
        spec_irrad,
        eqe,
        electrical_parameters,
        subcell_names,
        ambient_temp=None,
        cell_temps=None,
        temp_model=None,
        min_Jsc_both_cells=False,
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
        if cell_temps is None:
            if temp_model is None:
                self.cell_temps = {
                    subcell: pd.Series(25) for subcell in subcell_names
                }
            else:
                self.cell_temps = {
                    subcell: calc_temp_from_NOCT(
                        self.electrics["noct"][subcell],
                        ambient_temp,
                        spec_irrad["front"].values.sum(axis=-1),
                    )
                    # correction for limited range of spectrum (onyl until 1200 nm)
                    * 1.15
                    for subcell in subcell_names
                }
        else:
            self.cell_temps = cell_temps

        for subcell in subcell_names:
            self.electrical_models[subcell] = OneDiodeModel(
                tcJsc=self.electrics["tcJsc"][subcell],
                tcVoc=self.electrics["tcVoc"][subcell],
                R_shunt=self.electrics["Rsh"][subcell],
                R_series=self.electrics["RsTandem"] / 2,
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
                    calc_current(
                        self.spec_irrad["back"], self.eqe_back[subcell]
                    )
                    / 10,
                    name=subcell,
                )
                Jsc[subcell] = Jsc[subcell] + Jsc_backside

        if min_Jsc_both_cells:
            Jsc_min = Jsc.min(axis=1)
            for subcell in self.subcell_names:
                Jsc[subcell] = Jsc_min

        return Jsc

    def calc_IV(self, Jsc, return_subsells=False):
        V = []

        for subcell in self.subcell_names:
            # if type(cell_temps) == pd.Series:
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
        V[V < 0] = np.nan
        V_tandem = V.groupby(level=1, axis=1).aggregate(
            lambda x: np.sum(x.values)
        )  # .apply(lambda x: np.sum(x))

        if return_subsells:
            return V_tandem, V
        else:
            return V_tandem

    def calc_power(self):
        V = []

        for subcell in self.subcell_names:
            # if type(cell_temps) == pd.Series:
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
        V_tandem[V_tandem<0] = np.nan
        
        iv = pd.Series(V_tandem.values.sum(axis=1)).rename('tandem').to_frame()
        iv = pd.concat([iv, V_tandem], axis=1)
        
        iv[iv.isna()] = 0
        iv.index = self.j_arr
        
        return iv


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    spec = pd.read_csv("../examples/data/tiny_spec.csv", index_col=0)
    spec.columns = spec.columns.astype(float)
    spec = spec / 1000
    eqe = pd.read_csv("../examples/data/eqe_tandem_2t.csv", index_col=0)

    eqe_new = interp_eqe_to_spec(eqe, spec)

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

    tandem = TandemSimulator(
        {"front": spec},
        eqe_new,
        electrical_parameters,
        ["pero", "si"],
        ambient_temp=25,
    )

    power = tandem.calc_power()
    plt.plot(power)
