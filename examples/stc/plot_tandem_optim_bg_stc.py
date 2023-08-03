"""
Tandem Solar Cell under STC
===========================
Simulating the IV curve of a Tandem Solar Cell with a 1-Diode model.
"""

# %%
# This example shows how to model the performance of a tandem solar cell. It
# uses "spectral-on-demand" data from the NSRDB providded by NREL.
# For the absorptances of the subcells GENPRO4 simulated EQE curves are used
# originally createf for the following publication:
# Reference
# ----------
# .. [1] P. Tillmann, K. Jäger, A. Karsenti, L. Kreinin, C. Becker (2022)
#    “Model-Chain Validation for Estimating the Energy Yield of Bifacial
#    Perovskite/Silicon Tandem Solar Cells,” Solar RRL 2200079,
#    DOI: 10.1002/solr.202200079

# %%
# First, we load the preprocessed spectral and meta-data (for temperature)
# The spectral data has to be converted from W/µm/m2 to W/nm/m2 (by deviding by 1000)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pv_tandem import utils, solarcell_models
import pvlib

plt.rcParams["figure.dpi"] = 140

eqe = pd.read_csv("../data/eqe_tandem_2t.csv", index_col=0)

# %%
# Note that the airmass and zenith values do not exactly match the values in
# the technical report; this is because airmass is estimated from solar
# position and the solar position calculation in the technical report does not
# exactly match the one used here.  However, the differences are minor enough
# to not materially change the spectra.




electrical_parameters_4T = {
    "Rsh": {"pero": 2000, "si": 5000},
    "Rs": {"pero":2, "si":1},
    "j0": {"pero": 2.7e-18, "si": 1e-12},
    "n": {"pero": 1.1, "si": 1},
    "tcJsc": {"pero": 0.0002, "si": 0.00032},
    "tcVoc": {"pero": -0.002, "si": -0.0041},
}

electrical_parameters_2T = {
    "Rsh": {"pero": 2000, "si": 5000},
    "RsTandem": 3,
    "j0": {"pero": 2.7e-18, "si": 1e-12},
    "n": {"pero": 1.1, "si": 1},
    "tcJsc": {"pero": 0.0002, "si": 0.00032},
    "tcVoc": {"pero": -0.002, "si": -0.0041},
}

tandem_4T = solarcell_models.TandemSimulator4T(
    eqe=eqe,
    electrical_parameters=electrical_parameters_4T,
    subcell_names=["pero", "si"],
)

eqe_all_bgs = pd.read_csv('../data/eqe_tandem_all_bgs.csv')

eff_2T = []
eff_4T = []

bandgaps = eqe_all_bgs['bandgap'].sort_values().unique()

for bandgap in bandgaps:
    eqe = eqe_all_bgs.loc[eqe_all_bgs['bandgap']==bandgap, ['pero', 'si','wl']]
    eqe = eqe.set_index('wl').sort_index()

    j0 = utils.calc_j0_RT(eqe['pero'], lqe_ele=0.01)
    
    electrical_parameters_2T['j0']['pero'] = j0
    electrical_parameters_4T['j0']['pero'] = j0
    
    tandem_2T = solarcell_models.TandemSimulator2T(
        eqe=eqe,
        electrical_parameters=electrical_parameters_2T,
        subcell_names=["pero", "si"],
    )
    
    iv_df = tandem_2T.calc_IV_stc()
    
    eff_2T.append((iv_df.tandem * iv_df.index).max())
    
    tandem_4T = solarcell_models.TandemSimulator4T(
        eqe=eqe,
        electrical_parameters=electrical_parameters_4T,
        subcell_names=["pero", "si"],
    )
    
    iv_df = tandem_4T.calc_IV_stc()
    
    eff_4T.append(iv_df.multiply(iv_df.index, axis=0).max().sum())
    

eff_2T = pd.Series(eff_2T, index=bandgaps)
eff_4T = pd.Series(eff_4T, index=bandgaps)

fig, ax = plt.subplots(dpi=150)



ax = eff_2T.plot(ax=ax, c="C0", label='bif')
ax = eff_4T.plot(ax=ax, c="C1", label='mono')

ax.set_xlabel('Perovskite Bandgap (eV)')
ax.set_ylabel('Efficiency (%)')

#ax.legend()
#ax2.legend()

handles, _ = ax.get_legend_handles_labels()


# Combine the handles from both axes

ax.legend(handles, ['2 Terminal', '4 Terminal'])