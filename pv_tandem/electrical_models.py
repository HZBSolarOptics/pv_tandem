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
            result = x - np.log(x) + np.log(x) / x
            return result

        def lambertwlog(x):
            large_x = x.copy()
            large_x_mask = x > 20
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

        return V
    
    def calc_iv_params(self, Jsc, cell_temp, j_arr=np.linspace(0,450,451)):
        V = self.calc_iv(Jsc, cell_temp, j_arr)
        P = V*j_arr
        idx_mpp = np.nanargmax(P, axis=1)
        Vmpp = V[np.arange(0,len(V)),idx_mpp]
        Voc = V[:,0]
        Pmax = np.nanmax(P, axis=1)
        
        Jsc = j_arr[(np.argmax((V<0), axis=1)-1).clip(min=0)]
        
        FF = abs(Pmax)/abs(Jsc*Voc)
        
        res = pd.DataFrame({'Voc':Voc, 'Vmpp':Vmpp, 'Pmax':Pmax, 'FF':FF,
                            'Jsc':Jsc})
        
        return res