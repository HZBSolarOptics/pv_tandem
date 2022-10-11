# -*- coding: utf-8 -*-

import scipy.io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import lambertw
from scipy import constants
import pvlib
import bifacial_illumination as bi

#import cell_sim as cs


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

class TandemKit():
    def __init__(self, Jsc, electrical_parameters, subcells):
        # Thermal voltage at room temperature in V
        self.Vth = 0.02569;
        self.electrics = electrical_parameters
        self.subcells = subcells
        self.Jsc = Jsc

    def calc_tandem_kit(self):
        # Thermal voltage at room temperature in V
        Vth = 0.02569;
        
        factorJsc = {};
        factorVoc = {};
        Jsc = {};
        Voc_rt = {};
        Voc = {};
        P = {};
        
        j_arr = np.linspace(0, 35, 351)
        
        for cell in self.subcells:
            factorJsc = (1 + self.electrics['tcJsc'][cell] * (self.electrics['Temp'][cell] - 25))
            factorVoc = (1 + self.electrics['tcVoc'][cell] * (self.electrics['Temp'][cell] - 25))
            
            Jsc[cell] = self.Jsc[cell] * factorJsc
        
            Voc_rt[cell] = Jsc[cell]/1000 * self.electrics['RshTandem'] - \
                self.electrics['n'][cell] * Vth * lambertw(np.exp(
                    np.log(self.electrics['j0'][cell]/1000 * self.electrics['RshTandem']) + \
                    self.electrics['RshTandem'] * (Jsc[cell] + self.electrics['j0'][cell])/(1000 * self.electrics['n'][cell] * Vth)  - \
                np.log(self.electrics['n'][cell] * Vth) )) + self.electrics['j0'][cell]/1000 * self.electrics['RshTandem']
            
            Voc_rt[cell][Jsc[cell]<0.1] = np.nan
            Voc[cell] = Voc_rt[cell] * factorVoc
            
            lambw = np.log(self.electrics['j0'][cell]/1000 * self.electrics['RshTandem']) + \
                    (self.electrics['RshTandem'] * ((np.subtract.outer(Jsc[cell], j_arr) +
                    self.electrics['j0'][cell]))/(1000 * self.electrics['n'][cell] * Vth))  - \
                    np.log(self.electrics['n'][cell] * Vth) 
                
            V = np.subtract.outer(Jsc[cell]/1000 * self.electrics['RshTandem'], j_arr/1000 * (self.electrics['RshTandem'] + self.electrics['RsTandem'])) - \
                self.electrics['n'][cell] * Vth *  lambertw(np.exp(lambw)) + \
                (self.electrics['j0'][cell]/1000 * self.electrics['RshTandem'] - Voc_rt[cell] + Voc[cell])[:,None]
                
            V[V<0] = 0
            
            P[cell] = V * j_arr[None, :]
            P[cell] = np.nanmax(P[cell], axis=1)
            
            
        tandem_Voc = np.vstack([Voc_rt[cell] for cell in self.subcells]).sum(axis=0)
        tandem_Voc[np.isnan(tandem_Voc)] = 0
        
        tandem_P = np.vstack([P[cell] for cell in self.subcells]).sum(axis=0)
        tandem_P[np.isnan(tandem_P)] = 0
        
        


mat = scipy.io.loadmat('eycalc_complete_dump_miami.mat')

dt = mat['irradiance']['Data_TMY3'][0][0]
dt = pd.to_datetime(
    pd.DataFrame(dt[:,:6].astype(int), columns=['year', 'month', 'day', 'hour', 'minute', 'second'])
                 ).dt.tz_localize('Etc/GMT+4')

#dt = dt - pd.Timedelta('4h')

lat, long = 25.73, -80.21

solarposition = pvlib.solarposition.get_solarposition(dt, latitude=lat, longitude=long)

print(solarposition['zenith'].groupby(solarposition.index.hour).min())

asdf

df_bif_illum = solarposition[['zenith', 'azimuth']]
df_bif_illum[['DHI', 'DNI']] = 1

#df_berlin = df_berlin.query('zenith<90')

simulator = bi.YieldSimulator(df_bif_illum,
                              tmy_data=False,
                              module_height=0.5)

factors = simulator.simulate(spacing=20, tilt=20)
factors = factors.groupby(axis=1, level=0).mean()

spec_irrad_diff = mat['irradiance']['Irr_spectra_clouds_diffuse_horizontal'][0][0]
spec_irrad_dir = mat['irradiance']['Irr_spectra_clouds_direct_horizontal'][0][0]
wl_arr = mat['irradiance']['Irr_spectra_clouds_wavelength'][0][0][0]

eqe = mat['optics']['A'][0][0]
wl_eqe = mat['lambdaTMM'][0]

eqe = pd.DataFrame(eqe, index=wl_eqe, columns=['pero', 'si'])
eqe.plot()

spec_irrad_dir = pd.DataFrame(spec_irrad_dir, columns=wl_arr, index = dt)
spec_irrad_diff = pd.DataFrame(spec_irrad_diff, columns=wl_arr, index = dt)

spec_irrad_dir = spec_irrad_dir.loc[:,300:1200]



spec_irrad_dir = spec_irrad_dir / np.cos(np.deg2rad(solarposition['zenith'])).clip(0.01,1).values[:,None]
spec_irrad_dir = spec_irrad_dir.fillna(0)

spec_irrad_dir.sum(axis=1).clip(0,1500).hist()

print(spec_irrad_dir.sum(axis=1).groupby(spec_irrad_dir.index.hour).max())

asdf

spec_irrad_diff = spec_irrad_diff.loc[:,300:1200]

spec_irrad_dir = factors.loc[:, factors.columns.to_series().str.contains('direct')].sum(axis=1)*spec_irrad_dir
spec_irrad_diff = factors.loc[:, factors.columns.to_series().str.contains('diffuse')].sum(axis=1)*spec_irrad_diff


eqe = pd.DataFrame({'pero': np.interp(spec_irrad_dir.columns, eqe.index, eqe['pero']),
                    'si': np.interp(spec_irrad_dir.columns, eqe.index, eqe['si'])},
                    index = spec_irrad_dir.columns)

spec_irrad = spec_irrad_dir + spec_irrad_diff

i_ph = {'pero':calc_current(spec_irrad, eqe['pero'])/10,
                    'si':calc_current(spec_irrad_dir, eqe['si'])/10}
                    

electrical_parameters = {
    'RshTandem': 1000,
    'RsTandem': 3,
    'j0': {'pero': 2.7e-18, 'si':1e-12},
    'n': {'pero':1.1, 'si':1},
    'Temp': {'pero': 25, 'si':25},
    'noct': {'pero': 48, 'si': 48},
    'tcJsc': {'pero': 0.0002, 'si': 0.00032},
    'tcVoc': {'pero': -0.002, 'si': -0.0041}
    }


tandem = TandemKit(i_ph, electrical_parameters, subcells=['pero', 'si'])
        
        
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            


