import os
import sys
import pandas as pd
import numpy as np
from scipy.interpolate import pchip_interpolate
from pv_tandem.solarcell_models import OneDiodeModel
from pv_tandem.bifacial import IrradianceSimulator
import pvlib

class GetLocData:
    def __init__(self,nsrdb_data_filepath,pv_reflection_filepath,spectral_boolean):
        self.pv_reflection = pd.read_csv(pv_reflection_filepath, skiprows = 2)
        if spectral_boolean:
            self.nsrdb_data_filepath = nsrdb_data_filepath+'not_spectral.csv'
            self.nsrdb_data_filepath_spectral = nsrdb_data_filepath+'spectral.csv'
            self.get_loc_data()
            self.get_spec_fracs()
        else:
            self.nsrdb_data_filepath = nsrdb_data_filepath
            self.get_loc_data()
        self.get_am15_fracs()

    # Function for importing location data and add solar position.
    def get_loc_data(self):
        nsrdb_data = pvlib.iotools.read_nsrdb_psm4(self.nsrdb_data_filepath)
        solar_pos = pvlib.solarposition.get_solarposition(nsrdb_data[0].index,
                                                          latitude = nsrdb_data[1]['latitude'],
                                                          longitude = nsrdb_data[1]['longitude'],
                                                          altitude = nsrdb_data[1]['altitude'],
                                                          pressure = nsrdb_data[0]['pressure'],
                                                          temperature = nsrdb_data[0]['temp_air']
                                                         )
        loc_data = nsrdb_data[0].copy()
        loc_data['zenith'] = solar_pos['zenith']
        loc_data['apparent_zenith'] = solar_pos['apparent_zenith']
        loc_data['azimuth'] = solar_pos['azimuth']
        # Adjust time series so that all values are in one year to have Fuentes model working
        loc_data.index = pd.date_range(loc_data.index[0],periods = len(loc_data), freq = 'h')
        self.data = loc_data

    # Function for some standard AM1.5 fractions and values
    def get_am15_fracs(self):
        # AM15 data
        am15_file_path = os.path.join(
            os.path.dirname(__file__), "data", "ASTMG173.csv"
        )
        am15 = pd.read_csv(am15_file_path,delimiter = ';')
        wl_array = am15['Wvlgth nm']
        am15_array = am15['Global tilt  W*m-2*nm-1']
        # Adapt reflectance data of PV module to wavelenth range
        r_perc_int = np.ones(len(wl_array))
        # The R spectrum from Subedi only has values > 300 nm. Hence, interpolation starts at [43].
        r_perc_int[43:1702] = pchip_interpolate(self.pv_reflection['wl'],
                                                self.pv_reflection['R'],
                                                wl_array[43:1702])
        # Calculate different power values
        # Total irradiance by AM1.5 spectrum
        irrad_am15 = np.trapezoid(am15_array,wl_array)
        idx_bg = 940 # index of bandgap wavelength (1100 nm for Si)
        irrad_pv = np.trapezoid(am15_array[:idx_bg]*(1-r_perc_int[:idx_bg]),wl_array[:idx_bg])
        irrad_ir = np.trapezoid(am15_array[idx_bg:]*(1-r_perc_int[idx_bg:]),wl_array[idx_bg:])
        self.data['irrad_am15'] = irrad_am15
        self.data['ir_frac_am15'] = irrad_ir/irrad_am15
        self.data['pv_frac_am15'] = irrad_pv/irrad_am15
        self.data['irrad_ref'] = irrad_pv

    # Function for extracting spectral fractions
    def get_spec_fracs(self):
        spectral = pd.read_csv(self.nsrdb_data_filepath_spectral,skiprows = 2)
        # Assuming that first wavelength column is in [32]
        wl_array = spectral.columns[32:].map(lambda each:float(each[:-3]))
        r_perc_int = np.ones(len(wl_array))
        # The R spectrum from Subedi only has values > 300 nm. Hence, interpolation starts at [43].
        r_perc_int[43:1702] = pchip_interpolate(self.pv_reflection['wl']/1000,
                                                self.pv_reflection['R'],
                                                wl_array[43:1702])
        #calculate irradiances
        irr_tot = np.trapezoid(spectral[spectral.columns[32:]],wl_array) # total
        # Fraction absorbed by PV module for wavelengths < Si bandgap (column 940 refers to 1100 nm)
        irr_pv_abs = np.trapezoid(spectral[spectral.columns[32:972]]*(1-r_perc_int[:940]),
                                  wl_array[:940])
        # Fraction absorbed by PV module for wavelengths longer than Si bandgap
        irr_ir_abs = np.trapezoid(spectral[spectral.columns[972:]]*(1-r_perc_int[940:]),
                                  wl_array[940:])
        # Calculate fractions (done such that np.divide does not cause strange behavior)
        pv_frac = np.zeros(len(irr_tot))
        ir_frac = np.zeros(len(irr_tot))
        idx = np.where(irr_tot > 0)
        pv_frac[idx] = np.divide(irr_pv_abs[idx], irr_tot[idx])
        ir_frac[idx] = np.divide(irr_ir_abs[idx], irr_tot[idx])
        self.data['pv_frac'] = pv_frac
        self.data['ir_frac'] = ir_frac

# Class for calculating EY
class CalcEY:
    # def __init__(self,nsrdb_data,pv_dict,config_dict):
    def __init__(self,loc_data,pv_dict,config_dict):
        self.loc_data = loc_data
        self.pv_dict = pv_dict
        self.config_dict = config_dict
        self.irrad_poa = None
        self.irrad_therm = None
        self.j_sc = None
        self.coating = None
        self.ctstr = None
        self.loss_factor = None
        self.res = {}

    # get radiant exposure (accumulated irradiance) over the year with simple model
    def get_irrad_simple(self):
        # Convert angles from degrees to radians
        solar_zenith = np.radians(self.loc_data['zenith'])
        solar_azimuth = np.radians(self.loc_data['azimuth'])
        tilt = np.radians(self.pv_dict['module_tilt_deg'])
        plane_azimuth = np.radians(self.pv_dict['module_plane_azimuth_deg'])

        # Calculate the cosine of the angle of incidence
        cos_theta_i = (np.cos(solar_zenith) * np.cos(tilt)
                       + np.sin(solar_zenith) * np.sin(tilt) * np.cos(solar_azimuth - plane_azimuth)
                       )

        # Calculate the DNI absorbed
        dni_plane = self.loc_data['dni'] * cos_theta_i

        # Make sure the absorbed DNI is not negative (i.e., the sun is behind the plane)
        dni_plane[dni_plane < 0] = 0

        # assuming 80% of the DHI are reaching the 20° tilted module plane
        irrad_poa_simple = dni_plane + self.loc_data['dhi'] * 0.8
        self.res['irrad_poa_simple'] = irrad_poa_simple
        self.res['radexp_poa_simple'] = irrad_poa_simple.sum()

    # get radiant exposure (accumulated irradiance) over the year with better irradiance model
    def get_irrad_vf(self):
        illum_full = self.loc_data.copy()
        illum = illum_full[illum_full['zenith'] < 90]
        illum = illum.rename(columns={'dhi':'DHI','dni':'DNI'})
        #illum['DHI'] = illum['dhi']
        #illum['DNI'] = illum['dni']
        illum = illum[['DNI','DHI','zenith','azimuth']]
        simulator = IrradianceSimulator(illum,
                                        albedo=self.pv_dict['albedo'],
                                        module_length=self.pv_dict['module_length'],
                                        mount_height=self.pv_dict['module_height'],
                                        module_tilt=self.pv_dict['module_tilt_deg'],
                                        module_spacing=self.pv_dict['module_spacing'],
                                        ground_steps = self.pv_dict['ground_steps']
                                       )
        irrad_poa_vf = simulator.simulate(simple_results=True)
        irrad_poa_vf = irrad_poa_vf.reindex(index = illum_full.index, fill_value = 0)
        self.res['irrad_poa_vf'] = irrad_poa_vf
        self.res['radexp_poa_vf'] = irrad_poa_vf['front'].sum()
        self.res['radexp_poa_vf_back'] = irrad_poa_vf['back'].sum()

    def get_irrad_poa(self):
        #Get the irradiation
        match self.config_dict['irrad_model']:
            case "simple":
                self.get_irrad_simple()
                irrad_poa = self.res['irrad_poa_simple']
            case "vf":
                self.get_irrad_vf()
                irrad_poa = self.res['irrad_poa_vf']['front']
            case _:
                print('Please use a valid irradiation model ("simple" or "vf").')
                sys.exit()
        self.irrad_poa = irrad_poa

    def get_irrad_therm(self):
        if self.coating:
            if self.config_dict['spectral']:
                irrad_factor = 1-self.loc_data['ir_frac']
            else:
                irrad_factor = 1-self.loc_data['ir_frac_am15']
            self.irrad_therm = self.irrad_poa*irrad_factor*self.loss_factor
        else:
            self.irrad_therm = self.irrad_poa

    def get_temp(self):
        match self.config_dict['temp_model']:
            case "Ross":
                cell_temp = pvlib.temperature.ross(
                    poa_global = self.irrad_therm,
                    temp_air = self.loc_data['temp_air'],
                    noct=45
                    )
            case "Fuentes":
                cell_temp = pvlib.temperature.fuentes(
                    poa_global = self.irrad_therm,
                    temp_air = self.loc_data['temp_air'],
                    wind_speed = self.loc_data['wind_speed'],
                    module_height = self.pv_dict['module_height'], # default 5.0
                    module_width = self.pv_dict['module_width'], # default: 0.31579
                    module_length = self.pv_dict['module_length'], # default: 1.2
                    surface_tilt = self.pv_dict['module_tilt_deg'], # default: 30
                    noct_installed = self.pv_dict['noct'],
                    emissivity = self.pv_dict['fuentes_module_emissivity'] # default: 0.84
                    )
            case "SAPM":
                j = self.pv_dict['sapm_type']
                cell_temp = pvlib.temperature.sapm_cell(
                    poa_global = self.irrad_therm,
                    temp_air = self.loc_data['temp_air'],
                    wind_speed = self.loc_data['wind_speed'],
                    a = self.pv_dict['sapm_a'][j],
                    b = self.pv_dict['sapm_b'][j],
                    deltaT = self.pv_dict['sapm_deltaT'][j]
                    )
                module_temp = pvlib.temperature.sapm_module(
                    poa_global = self.irrad_therm,
                    temp_air = self.loc_data['temp_air'],
                    wind_speed = self.loc_data['wind_speed'],
                    a = self.pv_dict['sapm_a'][j],
                    b = self.pv_dict['sapm_b'][j],
                    )
                self.res['module_temp_'+self.ctstr] = module_temp
                # Values for daytime only
                self.res['module_temp_day_'+self.ctstr] = module_temp[self.loc_data['zenith']<90]

            case _:
                print('Please use a valid temperature model (Ross, Fuentes or SAPM).')
                sys.exit()

        self.res['cell_temp_'+self.ctstr] = cell_temp
        self.res['cell_temp_day_'+self.ctstr] = cell_temp[self.loc_data['zenith']<90] # Daytime only

    def get_j_sc(self):
        if self.config_dict['spectral']:
            # Power reaching PV module < 1100 nm for AM1.5 spectrum, accounting for PERC reflection
            # only consider light < 1100 nm
            self.j_sc = (self.irrad_poa
                         *self.loc_data['pv_frac']
                         /self.loc_data['irrad_ref']
                         *self.loss_factor
                         *self.pv_dict['j_sc_STC']
                         )
        else:
            self.j_sc = self.irrad_poa/1000*self.loss_factor*self.pv_dict['j_sc_STC']

    def get_p_max(self):
        # Parameters for one-diode model (consider making this part of pv_dict.
        one_diode = OneDiodeModel(tcJsc=0.0003,
                                  tcVoc=-0.004,
                                  R_shunt=3000,
                                  R_series=1.5,
                                  n=1,
                                  j0=1e-12
                                  )
        j_sc = self.j_sc
        cell_temp = self.res['cell_temp_'+self.ctstr]
        iv_paras = one_diode.calc_iv_params(Jsc=j_sc[j_sc > 0.12],
                                            cell_temp = cell_temp[j_sc > 0.12])
        power_max = iv_paras['Pmax'] * 10 / 1000
        self.res['power_max_'+self.ctstr] = power_max
        self.res['ey_'+self.ctstr] = (power_max).sum()

    def get_p_tc(self):
        cell_temp = self.res['cell_temp_'+self.ctstr]
        if self.config_dict['spectral']:
            # Power reachingPV module < 1100 nm for AM1.5 spectrum, accounting for PERC reflection
            power_tc = (self.pv_dict['PCE']/100
                        *self.irrad_poa
                        *self.loc_data['pv_frac']
                        /self.loc_data['irrad_ref']
                        *self.loss_factor
                        *(1+self.pv_dict['t_coeff']/100*(cell_temp - 25))
                        /1000 # the last /1000 converts from W -> kW
                        )
        else:
            power_tc = (self.pv_dict['PCE']/100
                        *self.irrad_poa
                        *self.loss_factor
                        *(1+self.pv_dict['t_coeff']/100*(cell_temp - 25))
                        /1000 # the last /1000 converts from W -> kW
                        )
        self.res['power_tc_'+self.ctstr] = power_tc
        self.res['ey_tc_'+self.ctstr] = (power_tc).sum()

    def get_ey(self):
        self.get_irrad_poa()
        for coating in (True,False):
            self.coating = coating
            if coating:
                self.ctstr = 'irr'
                self.loss_factor = 1.0-self.pv_dict['loss']
            else:
                self.ctstr = 'ref'
                self.loss_factor = 1.0
            self.get_irrad_therm()
            self.get_temp()
            self.get_j_sc()
            self.get_p_max()
            self.get_p_tc()

    def print_res(self):
        temp_ref = self.res['cell_temp_day_ref']
        temp_irr = self.res['cell_temp_day_irr']
        ey_ref = self.res['ey_ref']
        ey_irr = self.res['ey_irr']
        ey_tc_ref = self.res['ey_tc_ref']
        ey_tc_irr = self.res['ey_tc_irr']
        irrad = self.irrad_poa
        print(f"Avg. temp ref: {np.mean(temp_ref):.1f} \N{DEGREE SIGN}C")
        print(f"Avg. temp irr: {np.mean(temp_irr):.1f} \N{DEGREE SIGN}C")
        print(f"Max. ΔT: {max(temp_ref-temp_irr):.1f} \N{DEGREE SIGN}C")
        print(f"Min. ΔT: {min(temp_ref-temp_irr):.1f} \N{DEGREE SIGN}C")
        print(f"Avg. ΔT: {np.mean(temp_ref-temp_irr):.1f} \N{DEGREE SIGN}C")
        print(f"Avg. ΔT: {np.mean((temp_ref-temp_irr)[irrad > 0]):.1f} \N{DEGREE SIGN}C (when irrad > 0)")
        print(f"EY ref: {ey_ref:.0f} kWh/m2")
        print(f"EY irr: {ey_irr:.0f} kWh/m2")
        print(f"Abs. change: {ey_irr-ey_ref:.1f} kWh/m2")
        print(f"Rel. change: {(ey_irr-ey_ref)/ey_ref*100:.1f} %")
        print(f"EY tc ref: {ey_tc_ref:.0f} kWh/m2")
        print(f"EY tc irr: {ey_tc_irr:.0f} kWh/m2")
        print(f"Abs. change: {ey_tc_irr-ey_tc_ref:.1f} kWh/m2")
        print(f"Rel. change: {(ey_tc_irr-ey_tc_ref)/ey_tc_ref*100:.1f} %\n")

    def print_res_table(self):
        temp_ref = self.res['cell_temp_day_ref']
        temp_irr = self.res['cell_temp_day_irr']
        ey_ref = self.res['ey_ref']
        ey_irr = self.res['ey_irr']
        # Get irradiation
        irrad = self.irrad_poa
        print(f"{np.mean(temp_ref):.1f}\t"
              f"{np.mean(temp_irr):.1f}\t"
              f"{max(temp_ref-temp_irr):.1f}\t"
              f"{np.mean((temp_ref-temp_irr)[irrad > 0]):.1f}\t"
              f"{ey_ref:.0f}\t{ey_irr:.0f}\t"
              f"{ey_irr-ey_ref:.1f}\t"
              f"{(ey_irr-ey_ref)/ey_ref*100:.1f}%")
