"""
Energy Yield for Bifacial Tandem Solar Cell
===========================================
Using spectral on-demand data from NREL and simulated EQE data from GENPRO4.
"""

# %%
# This example shows how to model the performance of a bifacial tandem solar cell,
# with different bandgaps and how it compares to a monofacial counterpart. It
# uses "spectral-on-demand" data from the NSRDB providded by NREL.
# For the absorptances of the subcells GENPRO4 simulated EQE curves are used
# originally createf for the following publication:
# .. [1] P. Tillmann, K. Jäger, A. Karsenti, L. Kreinin, C. Becker (2022)
#    “Model-Chain Validation for Estimating the Energy Yield of Bifacial
#    Perovskite/Silicon Tandem Solar Cells,” Solar RRL 2200079,
#    :doi:`10.1002/solr.202200079`
#
# First, the nessesary libraries and datasets (spectral irradiance for the front side,
# diffuse and direct irradiance and eqe data) are loaded. The meta data of the spectral
# irradiance data contains the direct and diffuse irradiance data needed for the
# bifacial model


from pv_tandem import (
    utils,
    bifacial,
    solarcell_models,
    irradiance_models,
)
import matplotlib.pyplot as plt
import numpy as np
import pvlib
import pandas as pd
import seaborn as sns

spec_irrad_ts = pd.read_csv(
    "../data/spec_poa_dallas_2020.csv", index_col=0, parse_dates=True
)

spec_irrad_ts.columns = spec_irrad_ts.columns.astype(float)
# converting spectral data from W/µ/m² to W/nm/m²
spec_irrad_ts = spec_irrad_ts.clip(lower=0) / 1000

eqe = pd.read_csv("../data/eqe_tandem_2t.csv", index_col=0)
eqe = utils.interp_eqe_to_spec(eqe, spec_irrad_ts)

meta_ts = pd.read_csv(
    "../data/meta_ts_dallas_2020.csv", index_col=0, parse_dates=True
)

coord_dallas = dict(latitude=32.8, longitude=-96.8)

# %%
# Next we use the python library pvlib to calculate the solar position (zenith
# and azimuth angle), then the geometry of the bifacial soalr cell array is
# defined and the backside irradiance is calculated in the plane-of-array (poa).
# Further, the noct (nominal operating cell temperature) model of pvlib is used
# to calculated the cell temperature from irradiance, ambient temperature and wind
# speed

solar_pos = pvlib.solarposition.get_solarposition(
    meta_ts.index, **coord_dallas
)

illumination_df = meta_ts
illumination_df["zenith"] = solar_pos["zenith"]
illumination_df["azimuth"] = solar_pos["azimuth"]

illumination_df = illumination_df[["DNI", "DHI", "zenith", "azimuth"]]

# The

simulator = bifacial.IrradianceSimulator(
    illumination_df,
    albedo=0.3,
    module_length=1.96,
    mount_height=0.5,
    module_spacing=6, module_tilt=25
)

irrad_poa = simulator.simulate(simple_results=True)

# pvlib is used to calculate the solar cell temperature

temperature = pvlib.temperature.noct_sam(
    spec_irrad_ts.sum(axis=1) * 1.15 + irrad_poa["back"],
    meta_ts["Temperature"],
    meta_ts["Wind Speed"],
    noct=45,
    module_efficiency=0.25,
)

cell_temps = pd.DataFrame({"pero": temperature, "si": temperature})


electrical_parameters = {
    "Rsh": {"pero": 1000, "si": 3000},
    "RsTandem": 3,
    "j0": {"pero": 2.7e-18, "si": 1e-12},
    "n": {"pero": 1.1, "si": 1},
    "tcJsc": {"pero": 0.0002, "si": 0.00032},
    "tcVoc": {"pero": -0.002, "si": -0.0041},
}

tandem = solarcell_models.TandemSimulator2T(
    eqe=eqe,
    electrical_parameters=electrical_parameters,
    subcell_names=["pero", "si"],
)

Jsc_backside = (irrad_poa["back"] / 1000 * 35).rename("si").to_frame()
Jsc_backside["pero"] = 0

Jsc = tandem.calculate_Jsc(spec_irrad_ts)

Jsc["si"] = Jsc["si"] + irrad_poa["back"]

V_tandem = tandem.calc_IV(Jsc, cell_temps)

P = V_tandem.values * tandem.j_arr[None, :]
P_max = P.max(axis=1)
P_max = pd.Series(P_max, index=spec_irrad_ts.index)


# A dataset with simulated eqe data for several different perovskite bandgaps is
# loaded to scan the energy yield for the optimal bandgap

eqe_all_bgs = pd.read_csv("../data/eqe_tandem_all_bgs.csv")

P_stc = []
energy_yield_bif = []
energy_yield_mono = []
j_ph = []

for bandgap in eqe_all_bgs["bandgap"].sort_values().unique():
    eqe = eqe_all_bgs.loc[
        eqe_all_bgs["bandgap"] == bandgap, ["pero", "si", "wl"]
    ]
    eqe = eqe.set_index("wl").sort_index()
    eqe = utils.interp_eqe_to_spec(eqe, spec_irrad_ts)

    j0 = utils.calc_j0_RT(eqe["pero"], lqe_ele=0.01)

    electrical_parameters["j0"]["pero"] = j0

    tandem = solarcell_models.TandemSimulator2T(
        eqe=eqe,
        electrical_parameters=electrical_parameters,
        subcell_names=["pero", "si"],
    )

    j_ph.append(pd.Series(irradiance_models.AM15g().calc_jph(eqe) / 10))

    V_stc = tandem.calc_IV_stc()
    P_max_stc = V_stc.reset_index().eval("current*tandem").max()

    P_stc.append(pd.Series({bandgap: P_max_stc}))

    ey_bif = tandem.calc_power(
        spec_irrad_ts, cell_temps=cell_temps, backside_current=Jsc_backside
    )
    ey_bif = ey_bif.sum() / 1000 * 10

    energy_yield_bif.append(pd.Series({bandgap: ey_bif}))

    ey_mono = tandem.calc_power(spec_irrad_ts, cell_temps=cell_temps)
    ey_mono = ey_mono.sum() / 1000 * 10

    energy_yield_mono.append(pd.Series({bandgap: ey_mono}))


energy_yield_bif = pd.concat(energy_yield_bif)
energy_yield_mono = pd.concat(energy_yield_mono)
P_stc = pd.concat(P_stc)

fig, ax = plt.subplots(dpi=150)

ax2 = ax.twinx()

ax = energy_yield_bif.plot(ax=ax, c="C0", label="bif")
ax = energy_yield_mono.plot(ax=ax, c="C1", label="mono")
ax2 = P_stc.plot(ax=ax2, c="C2", label="stc")

ax.set_xlabel("Perovskite Bandgap (eV)")
ax.set_ylabel("Annual energy yield (kWh/m²)")
ax2.set_ylabel("Power density (kWh/m2)", color="C2")
ax2.tick_params(axis="y", colors="C2")

handles, _ = ax.get_legend_handles_labels()
handles2, _ = ax2.get_legend_handles_labels()

# Combine the handles from both axes
handles += handles2

ax.legend(
    handles,
    [
        "Energy yield bifacial",
        "Energy yield monofacial",
        "Standart test conditions",
    ],
)
