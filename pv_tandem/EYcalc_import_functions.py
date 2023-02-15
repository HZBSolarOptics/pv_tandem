import scipy.io
from scipy import constants, integrate
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pvlib

from sim_functions import spectral_illumination, TandemSimulator


def load_EYcalc_irrad_data(location_name, lat, long, timezone="Etc/GMT+5"):

    mat_temp = scipy.io.loadmat(f"Spectra_{location_name}/TMY3_{location_name}.mat")
    ambient_temperature = mat_temp["Data_TMY3"][:, 13]

    mat = scipy.io.loadmat(f"Spectra_{location_name}/Irr_spectra_clouds.mat")
    dt = mat_temp["Data_TMY3"]
    dt = pd.to_datetime(
        pd.DataFrame(
            dt[:, :6].astype(int),
            columns=["year", "month", "day", "hour", "minute", "second"],
        )
    ).dt.tz_localize(timezone)

    dt_sp = dt - pd.Timedelta("30 min")
    solarposition = pvlib.solarposition.get_solarposition(
        dt_sp, latitude=lat, longitude=long
    )
    solarposition = solarposition[["zenith", "azimuth"]]

    spec_irrad_diff = mat["Irr_spectra_clouds_diffuse_horizontal"]
    spec_irrad_dir = mat["Irr_spectra_clouds_direct_horizontal"]
    wl_arr = mat["Irr_spectra_clouds_wavelength"][0]

    spec_irrad_dir = pd.DataFrame(spec_irrad_dir, columns=wl_arr, index=dt)
    spec_irrad_diff = pd.DataFrame(spec_irrad_diff, columns=wl_arr, index=dt)

    spec_irrad_diff = spec_irrad_diff.loc[:, 300:1200]
    spec_irrad_dir = spec_irrad_dir.loc[:, 300:1200]

    spec_irrad_dir = (
        spec_irrad_dir
        / np.cos(np.deg2rad(solarposition["zenith"])).clip(0.1, 1).values[:, None]
    )
    spec_irrad_dir = spec_irrad_dir.fillna(0)

    spectral_irrad = pd.concat(
        [spec_irrad_dir, spec_irrad_diff], keys=["dni", "dhi"], axis=1
    )

    return ambient_temperature, spectral_irrad, solarposition


def load_eqe_EYcalc(eqe_filename, wavelengths):
    mat = scipy.io.loadmat(eqe_filename)
    eqe = mat["optics"]["A"][0][0]
    wl_eqe = np.linspace(300, 1200, 181)
    eqe = pd.DataFrame(eqe, index=wl_eqe, columns=["pero", "si"])

    eqe = pd.DataFrame(
        {
            "pero": np.interp(wavelengths, eqe.index, eqe["pero"]),
            "si": np.interp(wavelengths, eqe.index, eqe["si"]),
        },
        index=wavelengths,
    )
    return eqe


def load_eqe_backside(wavelengths):
    mat_file = "optics_backside_illumination.mat"
    mat = scipy.io.loadmat(mat_file)
    eqe = mat["optics"]["A"][0][0].T
    wl_eqe = np.linspace(300, 1200, 181)
    eqe = pd.DataFrame(eqe, index=wl_eqe, columns=["si"])
    eqe["pero"] = 0

    eqe = pd.DataFrame(
        {
            "pero": np.interp(wavelengths, eqe.index, eqe["pero"]),
            "si": np.interp(wavelengths, eqe.index, eqe["si"]),
        },
        index=wavelengths,
    )
    return eqe

def calc_j0(eqe, temperature=300, bandgap_shift = None, lqe_ele = 0.01):  # use wl on the x-axis
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
        (2 * constants.c / wl ** 4), (np.exp(constants.h * constants.c / (wl * constants.k * T) - 1.0))
    )  # original

    
    if bandgap_shift is not None:
        eqe_wl_array = np.multiply.outer(eqe.index, bandgap_shift)
    else:
        eqe_wl_array = eqe.index
        
    product = (
        eqe.values.flatten() * bbr(eqe_wl_array * 1e-9, temperature).T
    )
    integral = np.trapz(y=product, x=eqe_wl_array.T * 1e-9)
    
    j0 = (
        integral * 2 * np.pi * constants.e / lqe_ele * 0.1
    )  # Factor 0.1 for conversion from A/m² to mA/cm²!
    return j0


if __name__ == "__main__":

    location_name = "722780TYA_Phoenix"

    lat, long = 33.44277, -112.072754

    ambient_temp, spectral_irrad, solarposition = load_EYcalc_irrad_data(
        location_name, lat, long, timezone="Etc/GMT+7"
    )
    eqe_backside = load_eqe_backside(wavelengths=spectral_irrad["dni"].columns)

    energy_yield = []
    energy_yield_bif = []

    bandgap_array = [1.6, 1.65, 1.7, 1.75, 1.8]

    Jsc_mono = []
    Jsc_bif = []

    for bandgap in bandgap_array:

        eqe = load_eqe_EYcalc(
            f"optics_{bandgap}eV_1000nm.mat", wavelengths=spectral_irrad["dni"].columns
        )
        
        spec_irrad_inplane = spectral_illumination(
            spectral_irradiance=spectral_irrad, solarposition=solarposition,
            albedo=0.3,
            spacing=20,
            tilt_angle=20,
        ).calc_spectral_inplane()
        
        j0_pero = calc_j0(eqe['pero'])
        
        electrical_parameters = {
            "RshTandem": 1000,
            "RsTandem": 3,
            "j0": {"pero": j0_pero, "si": 1e-12},
            "n": {"pero": 1.1, "si": 1},
            "Temp": {"pero": 25, "si": 25},
            "noct": {"pero": 48, "si": 48},
            "tcJsc": {"pero": 0.0002, "si": 0.00032},
            "tcVoc": {"pero": -0.002, "si": -0.0041},
        }
    
        subcells = ["pero", "si"]
    
        tandem = TandemSimulator(
            spec_irrad_inplane, eqe, electrical_parameters, subcells, ambient_temp,
            min_Jsc_both_cells = False
        )
        
        Jsc_mono.append(tandem.Jsc.sum())
        
        #energy_yield.append(tandem.calc_power().sum().real * 10)
    
        tandem_bif = TandemSimulator(
            spec_irrad_inplane,
            eqe,
            electrical_parameters,
            subcells,
            ambient_temp,
            eqe_back=eqe_backside,
            bifacial=True,
            min_Jsc_both_cells = False
        )
        Jsc_bif.append(tandem_bif.Jsc.sum())

        #energy_yield_bif.append(tandem_bif.calc_power().sum().real * 10)
      
        
    Jsc_mono = pd.concat(Jsc_mono, axis=1).T
    Jsc_mono.index = bandgap_array
    
    Jsc_bif = pd.concat(Jsc_bif, axis=1).T
    Jsc_bif.index = bandgap_array
    
    fig, ax = plt.subplots(dpi=200)
    
    (Jsc_mono * 10 / 1000).plot(ax=ax)
    (Jsc_bif * 10 / 1000).plot(ax=ax)
    
    ax.legend(["Pero mono", "Si mono", 'Pero bif', 'Si bif'])
    ax.set_xlabel('Bandgap (nm)')
    ax.set_ylabel('Time integrated Jsc (kAh/m²/year)')
        
    asdf
        

    for bandgap in bandgap_array:

        eqe = load_eqe_EYcalc(
            f"optics_{bandgap}eV_1000nm.mat", wavelengths=spectral_irrad["dni"].columns
        )
        
        spec_irrad_inplane = spectral_illumination(
            spectral_irradiance=spectral_irrad, solarposition=solarposition,
            albedo=0.3,
            spacing=20,
            tilt_angle=20,
        ).calc_spectral_inplane()
        
        j0_pero = calc_j0(eqe['pero'])
        
        electrical_parameters = {
            "RshTandem": 1000,
            "RsTandem": 3,
            "j0": {"pero": j0_pero, "si": 1e-12},
            "n": {"pero": 1.1, "si": 1},
            "Temp": {"pero": 25, "si": 25},
            "noct": {"pero": 48, "si": 48},
            "tcJsc": {"pero": 0.0002, "si": 0.00032},
            "tcVoc": {"pero": -0.002, "si": -0.0041},
        }
    
        subcells = ["pero", "si"]
    
        tandem = TandemSimulator(
            spec_irrad_inplane, eqe, electrical_parameters, subcells, ambient_temp,
            min_Jsc_both_cells = True
        )
        
        Jsc_mono.append(tandem.Jsc.sum())
        
        energy_yield.append(tandem.calc_power().sum().real * 10)
    
        tandem_bif = TandemSimulator(
            spec_irrad_inplane,
            eqe,
            electrical_parameters,
            subcells,
            ambient_temp,
            eqe_back=eqe_backside,
            bifacial=True,
            min_Jsc_both_cells = False
        )
        Jsc_mono.append(tandem_bif.Jsc.sum())

        energy_yield_bif.append(tandem_bif.calc_power().sum().real * 10)
        
    energy_yield = pd.Series(energy_yield, index=bandgap_array).rename('monofacial')
    energy_yield_bif = pd.Series(energy_yield_bif, index=bandgap_array).rename('bifacial')
    
    fig, ax = plt.subplots(dpi=300)
    (energy_yield/1000).plot(ax=ax)
    (energy_yield_bif/1000).plot(ax=ax)
    ax.legend()
    ax.set_ylim([500, 800])
    ax.set_xlabel("Bandgap (eV)")
    ax.set_ylabel("Yearly energy yield (kWh/m²/year)")
        

# =============================================================================
#     mat_file = "eycalc_complete_dump_miami.mat"
#     mat = scipy.io.loadmat(mat_file)
#     tandem_P_kit = pd.Series(
#         mat["EY"][0]["Power_Tandem"][0][:, 0], index=spec_irrad_inplane.index
#     )
# 
#     fig, ax = plt.subplots(dpi=300)
#     power_HZB.groupby(tandem_P_kit.index.dayofyear).sum().plot(
#         ax=ax, label="HZB_reimplement"
#     )
#     power_HZB_bif.groupby(tandem_P_kit.index.dayofyear).sum().plot(
#         ax=ax, label="HZB_reimplement_bif"
#     )
#     tandem_P_kit.groupby(tandem_P_kit.index.dayofyear).sum().plot(ax=ax, label="KIT")
#     ax.legend()
#     ax.set_ylim([0, 2500])
#     ax.set_xlabel("Day of year")
#     ax.set_ylabel("Daily energy yield (Wh/m²)")
# =============================================================================
