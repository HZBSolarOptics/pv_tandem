import matplotlib.pyplot as plt
import numpy as np
from pv_tandem.solarcell_models import OneDiodeModel

one_diode = OneDiodeModel(
    tcJsc=-0.001, tcVoc=-0.0013, R_shunt=5000, R_series=1.5, n=1, j0=1e-12
)

j_arr = np.linspace(0,40,401)
voltage = one_diode.calc_iv(Jsc = 39, cell_temp = 25, j_arr = j_arr)
fig, ax = plt.subplots()

ax.plot(voltage, j_arr)

ax.set_xlabel("Voltage (V)")
ax.set_ylabel("Current density (mA/cm2)")
ax.set_xlim(0,0.9)
ax.set_ylim(0)
plt.show()