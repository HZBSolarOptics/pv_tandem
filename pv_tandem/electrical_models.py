# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from scipy.special import lambertw


class OneDiodeModel:
    def __init__(self, tcJsc, tcVoc, R_shunt, R_series, n, j0):
        self.tcJsc = tcJsc
        self.tcVoc = tcVoc

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

        return np.real(V)

    def calc_iv_params(self, Jsc, cell_temp, j_arr=np.linspace(0, 45, 451)):

        if hasattr(Jsc, "values"):
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

        return res


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    pero_iv = pd.read_csv("../examples/data/pero_iv.csv")

    pero_cell = OneDiodeModel(
        tcJsc=-0.001, tcVoc=-0.0013, R_shunt=1000, R_series=4, n=1.1, j0=2e-16
    )

    res = pero_cell.calc_iv_params(np.array([22]), np.array([25]))

    iv = pero_cell.calc_iv(
        np.array([22]), np.array([25]), j_arr=np.linspace(0, 45, 451)
    )

    df_iv = pd.DataFrame(
        {"voltage": iv[0], "current": np.linspace(0, 45, 451)}
    )

    df_iv = df_iv.loc[df_iv["voltage"] > 0]

    fig, ax = plt.subplots(dpi=150)
    (pero_iv.set_index("voltage")["current"] * -1).plot(ax=ax)
    df_iv.set_index("voltage")["current"].plot(ax=ax)

    exp_data = pd.read_csv("../examples/data/pero_voc.csv")
    exp_data["Intensity"] = (
        exp_data["Intensity"].astype(float) / 10
    ).round() * 10
    exp_data = exp_data.set_index("Intensity").iloc[:, ::-1]

    int_factor = np.repeat([0.1, 0.3, 0.6, 1, 1.2], 3)
    int_factors = np.array([0.1, 0.3, 0.6, 1, 1.2])

    int_factors_repeated = np.repeat(int_factors, 3)

    T_arr = np.array([25, 55, 85])  # +273#np.arange(280, 340, 10)
    T_arr = np.tile(T_arr, 5)

    j_ph = 22.4 * int_factor

    V = pero_cell.calc_iv(
        Jsc=j_ph, cell_temp=T_arr, j_arr=np.linspace(0, 45, 451)
    )

    V = V.reshape(5, 3, -1)

    fig, ax = plt.subplots(dpi=200)

    labels = []
    cycle = ["C0", "C1", "C3"]
    for i in range(3):
        ax.scatter(
            int_factors * 100, V[:, i, 0], marker="x", s=80, color=cycle[i]
        )

    exp_data.plot(style="+", ax=ax, markersize=10, color=cycle)

    ax.legend()

    ax.set_ylabel("Open curcuit voltage (V)")
    ax.set_xlabel("Light intensity (% of AM1.5g)")

    plt.show()
