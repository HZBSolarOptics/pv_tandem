from scipy import interpolate
from scipy.constants import e, h, hbar, k, c
from scipy import constants
from scipy.integrate import quad

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import jit

from numba.core.errors import TypingError

import sys

elementary_charge = e

data = pd.read_hdf("dataset_stage1.h5", "table")
data = data.loc["20190825":"20200216"]

illu_components = pd.read_hdf("spec_data.h5", "illu_components").loc[
    "20190825":"20200216"
]
spec_dni_normalized = pd.read_hdf("spec_data.h5", "spec_dni_normalized").loc[
    "20190825":"20200216"
]
spec_dhi_normalized = pd.read_hdf("spec_data.h5", "spec_dhi_normalized").loc[
    "20190825":"20200216"
]

data, illu_components = data.align(illu_components, join="inner", axis=0)


def read_eqe(filename, extract_column_nbs, column_names=None):
    wl_genpro = np.arange(300, 1250, 10)
    eqe = pd.read_csv(filename, header=None).T
    eqe.index = wl_genpro

    eqe = eqe.iloc[:, extract_column_nbs]
    if column_names is not None:
        eqe.columns = column_names
    return eqe


def calc_current(spec, eqe):
    """
    Calculates the photocurrent from timeseries of impinging spectrum (W/nm/m²) and eqe.
    """
    norm_absorbtion = spec.multiply(eqe, axis=1)
    wl_arr = norm_absorbtion.columns.to_series()
    photo_flux = (norm_absorbtion / constants.h / constants.c).multiply(
        wl_arr * 1e-9, axis=1
    )
    return np.trapz(photo_flux, x=wl_arr) * constants.e
    # return photo_flux.apply(np.trapz, x=wl_arr, axis=1) * constants.e

def calc_bandgap_silicon(T=300):
    return 1.179 - 9e-5 * T - 3e-7 * T ** 2


def calc_bandgap_perovskite(T=300, bg_rt=1.64):
    delta_bg = 0.025 / 60 * (T - 293)
    return bg_rt + delta_bg


@jit(nopython=True)
def find_nearest(arr_1, arr_2):
    dim_1 = arr_1.shape[0]
    idx_nearest = np.zeros_like(arr_2)
    for i in range(dim_1):
        idx_nearest[i] = np.searchsorted(arr_1[i], arr_2[i])

    idx_nearest = np.where(
        idx_nearest < arr_1.shape[1], idx_nearest, arr_1.shape[1] - 1
    )
    return idx_nearest


class OneDiodeCell:
    def __init__(
        self,
        bandgap,
        T=300,
        lqe_ele=1,
        ideality=1,
        eqe_front=None,
        eqe_back=None,
        r_shunt=2000,
        bandgap_shift_func=calc_bandgap_silicon,
        r_series=None,
    ):
        self.bandgap = bandgap
        self.temperature = T
        self.lqe_ele = lqe_ele
        self.ideality = ideality
        self.eqe_front = eqe_front
        self.eqe_back = eqe_back
        self.r_series = r_series

        self.bandgap_shift = bandgap_shift_func(300) / bandgap_shift_func(T)

        self.wl_bg = h * c / self.bandgap / elementary_charge

        self.j0 = self.calc_j0()

        self.r_shunt = r_shunt
        self.current_mesh = np.linspace(0, 50, 501)
        self.V_grid = np.arange(0, bandgap * 1000, 1) / 1000

    def calc_j0(self):  # use wl on the x-axis
        """
        Function to calculate J0 for the detailed balance limit.
        E_bg: Bandgap of model materials
        T: Temperatur (K)
        lqe_eqe: Electroluminescent emission efficiency. For Shockley–Queisser equals 1.
        return: J0 (dark current) (mA/cm²)
        """
        # E_ev = np.linspace(E_bg,10,1000)
        # t = time.process_time()
        bbr = lambda wl, T: np.divide(
            (2 * c / wl ** 4), (np.exp(h * c / (wl * k * T) - 1.0))
        )  # original

        if self.eqe_front is not None:
            eqe_bbr = self.eqe_front
            try:
                pass
                # eqe_bbr = eqe_bbr + self.eqe_back
            except:
                pass

            if self.bandgap_shift is not None:
                eqe_wl_array = np.multiply.outer(eqe_bbr.index, self.bandgap_shift)
            else:
                eqe_wl_array = eqe_bbr.index
            product = (
                eqe_bbr.values.flatten() * bbr(eqe_wl_array * 1e-9, self.temperature).T
            )
            temp = np.trapz(y=product, x=eqe_wl_array.T * 1e-9)
        else:
            temp = quad(bbr, 0, self.wl_bg)[0]

        temp = (
            temp * 2 * np.pi * elementary_charge / self.lqe_ele * 0.1
        )  # Factor 0.1 for conversion from A/m² to mA/cm²!
        # print (time.process_time()-t, temp)
        return temp

    def cur_eq(self, V, j_ph=None):
        if j_ph is None:
            j_ph = self.get_j_ph_stc()

        temp = (
            np.exp(
                np.divide.outer(
                    elementary_charge * V / k / self.ideality, self.temperature
                )
            )
            - 1
        )
        try:
            return self.j0 * temp - j_ph
        except:
            return np.subtract.outer(self.j0 * temp, j_ph)

    def get_V_from_dark_curr_interp(self, dark_I):

        j_dark = self.cur_eq(self.V_grid, 0)
        j_sheet = self.V_grid / self.r_shunt * 1000
        j_rec_grid = j_dark.T + j_sheet

        try:
            idx_nearest = find_nearest(j_rec_grid, dark_I).astype(np.int32)
        except TypingError:
            idx_nearest = np.searchsorted(j_rec_grid, dark_I).astype(np.int32)
        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise
        # idx_nearest = (np.abs(np.subtract.inner(j_rec_grid, dark_I))).argmin(axis=0)

        V = self.V_grid[idx_nearest]
        V[dark_I <= 0] = np.nan

        return V

    def grided_IV(self, j_ph):
        dark_I = np.subtract.outer(j_ph, self.current_mesh)
        V = self.get_V_from_dark_curr_interp(dark_I)
        if self.r_series is not None:
            V = V - self.current_mesh / 1000 * self.r_series
        return self.current_mesh, V

    def calc_yield(self, bif=True, filter_timediff=True):
        curr, V = self.grided_IV(j_ph=self.get_photocurrent(bif=bif))

        power = curr * V
        power = pd.Series(np.nanmax(power, axis=1))

        if filter_timediff is True:
            delta_t = illu_components.index.to_series()
            delta_t = (delta_t - delta_t.shift()).dt.seconds
            delta_t_filter = (delta_t == 60).reset_index(drop=True)

            power = power[delta_t_filter]
            V = V[delta_t_filter]

        return power, V

    @classmethod
    def perovskite_cell(
        cls,
        bandgap=1.64,
        T=300,
        lqe_eqe=0.0012,
        r_shunt=1000,
        r_series=None,
        eqe=None,
        eqe_back=None,
    ):
        if eqe is None:
            eqe = read_eqe(
                "./eqe_output/jost_outdoor.csv",
                extract_column_nbs=[4],
                column_names=["pero"],
            )[["pero"]]

        if eqe_back is not None:
            eqe_back

        return cls(
            bandgap=bandgap,
            T=T,
            lqe_ele=lqe_eqe,
            eqe_front=eqe,
            r_shunt=r_shunt,
            r_series=r_series,
            bandgap_shift_func=calc_bandgap_perovskite,
        )

    def stc_IV(self, front=True):
        am15g = pd.read_csv("am15g.csv")
        am15g = np.interp(self.eqe_front.index, am15g["wl"], am15g["spec"])

        if front is True:
            j_ph = calc_current(self.eqe_front.T, am15g) / 10
        else:
            j_ph = calc_current(self.eqe_back.T, am15g) / 10
        # j_ph = j_ph[:,0]

        try:
            j_ph = np.ones(len(self.temperature)) * j_ph
        except:
            pass

        curr, V = self.grided_IV(j_ph=j_ph)
        return curr, V

    @classmethod
    def silicon_cell(
        cls,
        r_series=0,
        lqe_eqe=0.0016,
        r_shunt=1000,
        T=300,
        eqe_front=None,
        eqe_back=None,
    ):

        if eqe_front is None:
            eqe_front = read_eqe(
                "./eqe_output/si_front_illu.csv",
                extract_column_nbs=[7],
                column_names=["si"],
            )

        if eqe_back is None:
            eqe_back = read_eqe(
                "./eqe_output/si_back_illu.csv",
                extract_column_nbs=[6],
                column_names=["si"],
            )

        return cls(
            eqe_front=eqe_front,
            eqe_back=eqe_back,
            lqe_ele=lqe_eqe,
            r_shunt=r_shunt,
            T=T,
            bandgap=1.12,
            r_series=r_series,
            bandgap_shift_func=calc_bandgap_silicon,
        )

    # 4 / 6


class Tandem2T:
    def __init__(self, bandgap_top, eqe_front=None, eqe_back=None, T=300, r_series=6):
        if eqe_front is None:
            self.eqe_front = read_eqe(
                "./eqe_output/tandem_2t_front_illu.csv",
                extract_column_nbs=[7, 11],
                column_names=["pero", "si"],
            )
        else:
            self.eqe_front = eqe_front
        if eqe_back is None:
            self.eqe_back = read_eqe(
                "./eqe_output/tandem_2t_back_illu.csv",
                extract_column_nbs=[10, 6],
                column_names=["pero", "si"],
            )
        else:
            self.eqe_back = eqe_back

        self.top_cell = OneDiodeCell.perovskite_cell(
            eqe=self.eqe_front["pero"], bandgap=bandgap_top, T=T,
        )
        self.bot_cell = OneDiodeCell.silicon_cell(
            eqe_front=self.eqe_front[["si"]],
            eqe_back=self.eqe_back[["si"]],
            T=T,
        )
        self.r_series = r_series

        self.photocurrent_bif = calc_photocurrent(
            self.eqe_front, eqe_back=self.eqe_back
        )
        self.photocurrent_mono = calc_photocurrent(self.eqe_front, eqe_back=None)

    # =============================================================================
    #     def get_photocurrent(self, bif=True):
    #         if bif:
    #             return calc_photocurrent(
    #        		         self.eqe_front, eqe_back=self.eqe_back
    #         ).iloc[:,0].values / 10
    #         else:
    #             return calc_photocurrent(self.eqe_front, eqe_back=None).iloc[:,0].values / 10
    #
    # =============================================================================
    def grided_IV(self, j_ph_top, j_ph_bot, T=None, T_coef_si=None, T_coef_pero=None):
        curr, V_top = self.top_cell.grided_IV(j_ph_top)
        _, V_bot = self.bot_cell.grided_IV(j_ph_bot)

        if T is not None:
            V_top = V_top * (1 - T_coef_pero * (T - 25))[:, None]
            V_bot = V_bot * (1 - T_coef_si * (T - 25))[:, None]

        V = V_top + V_bot
        V = V - curr / 1000 * self.r_series
        power = curr * V
        return curr, V, power

    def stc_IV(self):
        am15g = pd.read_csv("am15g.csv")
        am15g = np.interp(self.eqe_front.index, am15g["wl"], am15g["spec"])
        curr = calc_current(self.eqe_front.T, am15g) / 10

        curr, V, power = self.grided_IV(j_ph_top=curr[0], j_ph_bot=curr[1])

        return curr, V, power

    def simulate_yield(self, bif=True, T=None, T_coef_si=None, T_coef_pero=None):
        if bif is True:
            curr, V, power = self.grided_IV(
                j_ph_top=self.photocurrent_bif["pero"].values / 10,
                j_ph_bot=self.photocurrent_bif["si"].values / 10,
                T=T,
                T_coef_si=T_coef_si,
                T_coef_pero=T_coef_pero,
            )
        else:
            curr, V, power = self.grided_IV(
                j_ph_top=self.photocurrent_mono["pero"].values / 10,
                j_ph_bot=self.photocurrent_mono["si"].values / 10,
                T=T,
                T_coef_si=T_coef_si,
                T_coef_pero=T_coef_pero,
            )

        power = pd.Series(np.nanmax(power, axis=1))

        delta_t = self.photocurrent_bif.index.to_series()
        delta_t = (delta_t - delta_t.shift()).dt.seconds
        delta_t_filter = (delta_t == 60).reset_index(drop=True)

        power = power[delta_t_filter]

        return power

    @classmethod
    def from_bg(cls, bandgap, T=300):
        eqe_front_file = (
            f"./eqe_output/tandem_2t/tandem_2t_front_illu_{bandgap:.2f}.csv"
        )
        eqe_back_file = f"./eqe_output/tandem_2t/tandem_2t_back_illu_{bandgap:.2f}.csv"
        eqe_front = read_eqe(
            eqe_front_file, extract_column_nbs=[7, 11], column_names=["pero", "si"],
        )

        eqe_back = read_eqe(
            eqe_back_file, extract_column_nbs=[10, 6], column_names=["pero", "si"],
        )
        return cls(bandgap_top=bandgap, eqe_front=eqe_front, eqe_back=eqe_back, T=T)


class Tandem4T(Tandem2T):
    def __init__(
        self,
        bandgap_top,
        eqe_front=None,
        eqe_back=None,
        T=300,
        r_series_top=6,
        r_series_bot=1.9,
    ):

        if eqe_front is None:
            self.eqe_front = read_eqe(
                "./eqe_output/tandem_4t_front_illu.csv",
                extract_column_nbs=[7, 11],
                column_names=["pero", "si"],
            )
        else:
            self.eqe_front = eqe_front
        if eqe_back is None:
            self.eqe_back = read_eqe(
                "./eqe_output/tandem_4t_back_illu.csv",
                extract_column_nbs=[10, 6],
                column_names=["pero", "si"],
            )
        else:
            self.eqe_back = eqe_back

        self.top_cell = OneDiodeCell.perovskite_cell(
            eqe=self.eqe_front["pero"], bandgap=bandgap_top, T=T,
        )
        self.bot_cell = OneDiodeCell.silicon_cell(
            eqe_front=self.eqe_front["si"],
            eqe_back=self.eqe_back["si"],
            T=T,
        )
        self.r_series_top = r_series_top
        self.r_series_bot = r_series_bot


    def grided_IV(self, j_ph_top, j_ph_bot, T=None):
        curr, V_top = self.top_cell.grided_IV(j_ph_top)
        _, V_bot = self.bot_cell.grided_IV(j_ph_bot)

        return curr, V_top, V_bot

    def stc_IV(self):
        am15g = pd.read_csv("am15g.csv")
        am15g = np.interp(self.eqe_front.index, am15g["wl"], am15g["spec"])
        curr = calc_current(self.eqe_front.T, am15g) / 10

        curr, V_top, V_bot = self.grided_IV(j_ph_top=curr[0], j_ph_bot=curr[1])

        V_top = V_top - curr / 1000 * self.r_series_top
        V_bot = V_bot - curr / 1000 * self.r_series_bot

        return curr, V_top, V_bot

    def stc_power(self):
        curr, V_top, V_bot = self.stc_IV()
        power_mpp_top = np.nanmax(V_top * curr)
        power_mpp_bot = np.nanmax(V_bot * curr)

        return power_mpp_bot + power_mpp_top

    def simulate_yield(self, bif=True):

        if bif is True:
            curr, V_top, V_bot = self.grided_IV(
                j_ph_top=self.photocurrent_bif["pero"].values / 10,
                j_ph_bot=self.photocurrent_bif["si"].values / 10,
            )

        else:
            curr, V_top, V_bot = self.grided_IV(
                j_ph_top=self.photocurrent_mono["pero"].values / 10,
                j_ph_bot=self.photocurrent_mono["si"].values / 10,
            )

        V_top = V_top - curr / 1000 * self.r_series_top
        V_bot = V_bot - curr / 1000 * self.r_series_bot

        power_top = np.nanmax(curr * V_top, axis=1)
        power_bot = np.nanmax(curr * V_bot, axis=1)
        power = power_top + power_bot

        delta_t = self.photocurrent_bif.index.to_series()
        delta_t = (delta_t - delta_t.shift()).dt.seconds
        delta_t_filter = (delta_t == 60).reset_index(drop=True)

        power = pd.Series(power[delta_t_filter])

        return power

    @classmethod
    def from_bg(cls, bandgap, T=300):
        eqe_front_file = (
            f"./eqe_output/tandem_4t/tandem_4t_front_illu_{bandgap:.2f}.csv"
        )
        eqe_back_file = f"./eqe_output/tandem_4t/tandem_4t_back_illu_{bandgap:.2f}.csv"
        eqe_front = read_eqe(
            eqe_front_file, extract_column_nbs=[4, 12], column_names=["pero", "si"],
        )
        eqe_back = read_eqe(
            eqe_back_file, extract_column_nbs=[14, 6], column_names=["pero", "si"],
        )
        return cls(bandgap_top=bandgap, eqe_front=eqe_front, eqe_back=eqe_back, T=T)
    