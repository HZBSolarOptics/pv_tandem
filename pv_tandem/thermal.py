""" 
Two classes for calculating annual energy yield of photovoltaic (PV) modules
with and without sub-bandgap reflecting coatings.
"""

import os
import sys
import csv
import pandas as pd
import numpy as np
from scipy.interpolate import pchip_interpolate
from pv_tandem.solarcell_models import OneDiodeModel
from pv_tandem.bifacial import IrradianceSimulator
import pvlib

class GetLocData:
    """ Assembles location-dependent data required for energy yield calculations"""

    def __init__(self,nsrdb_data_filepath,pv_reflection_filepath,spectral_boolean):
        """
        Initialize the object and populate the ``self.data`` dictionary.

        The initialization follows one of two processing routes depending on the
        value of ``spectral_boolean``, corresponding to spectral or non-spectral
        data handling.

        Parameters
        ----------
        nsrdb_data_filepath : str
            Path to CSV file with NSRDB data.
        pv_reflection_filepath : str
            Path to CSV file with spectral reflectance data of PV module.
        spectral_boolean : bool
            If True, spectral data handling methods are used; otherwise, non-spectral
                methods are applied.

        Return
        ------
        None
        
        """
        self.pv_reflection = pd.read_csv(pv_reflection_filepath, skiprows = 2)
        if spectral_boolean:
            nsrdb_data_filepath_spectral = nsrdb_data_filepath
            nsrdb_data_filepath_non_spectral = "NSRDB_data_temp.csv"
            self.truncate_csv_columns_after_header(nsrdb_data_filepath_spectral,
                                                   nsrdb_data_filepath_non_spectral
                                                   )
            self.get_loc_data(nsrdb_data_filepath_non_spectral)
            self.get_spec_fracs(nsrdb_data_filepath_spectral)
            os.remove(nsrdb_data_filepath_non_spectral)
        else:
            self.get_loc_data(nsrdb_data_filepath)
        self.get_am15_fracs()

    def truncate_csv_columns_after_header(self,input_path, output_path,keep_cols=32, header_rows=2):
        """
        Truncate columns in a CSV file containing spectral NSRDB after the header rows
        and store data in a new file at output_path.
    
        Parameters
        ----------
        input_path : str
            Path to the input CSV file with spectral NSRDB data.
        output_path : str
            Path to the output CSV file.
        keep_cols : int, optional
            Number of columns to keep for rows below the header.
            The spectral data is beyond column 32
        header_rows : int, optional
            Number of top rows to preserve unchanged.
            NSRDB data files contain 2 header rows
    
        Returns
        -------
        None

        """
        with open(input_path, newline="", encoding="utf-8") as infile, \
             open(output_path, "w", newline="", encoding="utf-8") as outfile:

            reader = csv.reader(infile)
            writer = csv.writer(outfile)

            for row_index, row in enumerate(reader, start=1):
                if row_index <= header_rows:
                    # Keep header rows exactly as they are
                    writer.writerow(row)
                else:
                    # NSRDB spectral data has cells with "," which cannot be handeled by read_psm4
                    if row_index == header_rows + 1:
                        row = [cell.replace(',', '') for cell in row]
                    writer.writerow(row[:keep_cols])

    def get_loc_data(self,nsrdb_data_filepath):
        """
        Import NSRDB data and compute solar position information.

        The imported data and derived solar position quantities are stored in
        the ``self.data`` dictionary.

        Parameters
        ----------
        nsrdb_data_filepath : str
            Path to the input CSV file with NSRDB data.
            
        Returns
        -------
        None
        
        """
        nsrdb_data = pvlib.iotools.read_nsrdb_psm4(nsrdb_data_filepath)
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

    def get_am15_fracs(self):
        """ 
        Calculate spectral fractions for AM1.5 solar spectrum
        
        Spectral fractions in the infrared `ir_frac_am15` (wavelength > 1100 nm)
        and in the PV-usable band `pv_frac_am15` (wavelength < 1100 nm) are calculated
        and stored in the ``self.data`` dictionary.
        
        Returns
        -------
        None
        
        """
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

    def get_spec_fracs(self,nsrdb_data_filepath_spectral):
        """ 
        Calculate spectral fraactions when spectral NSRDB data is given.
        
        Spectral fractions in the infrared `ir_frac` (wavelength > 1100 nm)
        and in the PV-usable band `pv_frac` (wavelength < 1100 nm) are calculated
        and stored in the ``self.data`` dictionary.
        
        Parameters
        ----------
        nsrdb_data_filepath_spectral : str
            Path to the input CSV file with spectral NSRDB data.
            
        Returns
        -------
        None.
        
        """
        spectral = pd.read_csv(nsrdb_data_filepath_spectral,skiprows = 2)
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
    """Contains all methods required for calculating the annual energy yield"""

    def __init__(self,loc_data,pv_dict,config_dict):
        """
        Initialize the object.
        
        Parameters
        ----------
        loc_data : dict
            Dictionary containing location-dependent data.
            This dictionary is populated in the class ``GetLocData``.
        pv_dict : dict
            Dictionary with parameters of the PV module.
            To be populated by the user before initialization.
        config_dict : dict
            Dictionary with configuration parameters.
            To be populated by user before initialization.
            
        Returns
        -------
        None

        """
        self.loc_data = loc_data
        self.pv_dict = pv_dict
        self.config_dict = config_dict
        self.coating = None
        self.ctstr = None
        self.loss_factor = None
        self.res = {}

    def get_irrad_simple(self):
        """
        Calculate irradiance on PV module plane of array (POA) using a simple model.
        
        Calculated arrays `irrad_poa_simple` and `radexp_poa_simple` are added to 
        ``self.res`` dictionary.
        
        Returns
        -------
        None.
        
        """

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
        # get radiant exposure (accumulated irradiance) over the year
        self.res['radexp_poa_simple'] = irrad_poa_simple.sum()

    def get_irrad_vf(self):
        """
        Calculate irradiance on PV module plane of array (POA) using a view-factor model [1].
        
        Calculated arrays `irrad_poa_vf`, `radexp_poa_vf` and `radexp_poa_vf` are added to 
        ``self.res`` dictionary.
        
        Returns
        -------
        None.
        
        References
        ---------        
        [1] K. Jäger, P. Tillmann, and C. Becker, Optics Express 28, 4751 (2020).
        """
        illum_full = self.loc_data.copy()
        illum = illum_full[illum_full['zenith'] < 90]
        illum = illum.rename(columns={'dhi':'DHI','dni':'DNI'})
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
        """
        Set the variable ``self.dict['irrad_poa']`` according to the irradiation model set in 
        ``self.config_dict['irrad_model']``.

        Returns
        -------
        None.

        """
        match self.config_dict['irrad_model']:
            case "simple":
                self.get_irrad_simple()
                self.res['irrad_poa'] = self.res['irrad_poa_simple']
            case "vf":
                self.get_irrad_vf()
                self.res['irrad_poa'] = self.res['irrad_poa_vf']['front']
            case _:
                print('Please use a valid irradiation model ("simple" or "vf").')
                sys.exit()

    def get_irrad_therm(self):
        """
        Calculate the irradiance used as input for the temperature models.
        
        If ``self.coating`` is True, calculate irradiance assuming a sub-bandgap reflecting coating,
        else otherwise.
        
        The array `irrad_therm` is added to ``self.res``.

        Returns
        -------
        None.

        """
        if self.coating:
            if self.config_dict['spectral']:
                irrad_factor = 1-self.loc_data['ir_frac']
            else:
                irrad_factor = 1-self.loc_data['ir_frac_am15']
            self.res['irrad_therm'] = self.res['irrad_poa']*irrad_factor*self.loss_factor
        else:
            self.res['irrad_therm'] = self.res['irrad_poa']

    def get_temp(self):
        """
        Calculate the solar cell temperature and store results in ``self.res``.
        
        Depending on the value of ``self.config_dict['temp_model']``, one of the
        following temperature models provided by pvlib is used:
        
        - Ross model
        - Fuentes model
        - SAPM model (when SAPM is used, the module temperature is also calculated)
        
        This method supports PV modules with and without a sub-bandgap reflecting
        coating (SBR), depending on the value of ``self.res['irrad_therm']``. Based on the
        value of ``self.ctstring``, the following entries are added to ``self.res``:
        
        - ``cell_temp_ref`` for PV modules without a SBR
        - ``cell_temp_sbr`` for PV modules with a SBR
        
        Returns
        -------
        None
        
        References
        ----------
        pvlib documentation on temperature models:
        https://pvlib-python.readthedocs.io/en/stable/reference/pv_modeling/temperature.html
        """
        match self.config_dict['temp_model']:
            case "Ross":
                cell_temp = pvlib.temperature.ross(
                    poa_global = self.res['irrad_therm'],
                    temp_air = self.loc_data['temp_air'],
                    noct=45
                    )
            case "Fuentes":
                cell_temp = pvlib.temperature.fuentes(
                    poa_global = self.res['irrad_therm'],
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
                match self.pv_dict['sapm_module_mount']:
                    case 0:
                        sapm_module_config = 'open_rack_glass_glass'
                    case 1:
                        sapm_module_config = 'close_mount_glass_glass'
                    case 2:
                        sapm_module_config = 'open_rack_glass_polymer'
                    case 3:
                        sapm_module_config = 'insulated_back_glass_polymer'
                    case _:
                        print('Please use a valid onfiguration for the SAPM model (0,1,2 or 3).')
                        sys.exit()
                sapm_params = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS
                sapm_params_dict = sapm_params['sapm'][sapm_module_config]
                #j = self.pv_dict['sapm_type']
                cell_temp = pvlib.temperature.sapm_cell(
                    poa_global = self.res['irrad_therm'],
                    temp_air = self.loc_data['temp_air'],
                    wind_speed = self.loc_data['wind_speed'],
                    **sapm_params_dict
                    )
                module_temp = pvlib.temperature.sapm_module(
                    poa_global = self.res['irrad_therm'],
                    temp_air = self.loc_data['temp_air'],
                    wind_speed = self.loc_data['wind_speed'],
                    a = sapm_params_dict['a'],
                    b = sapm_params_dict['b']
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
        """
        Estimate the short-circuit current using linear regression and store results in
        ``self.res['j_sc']``
        
        Methods selects one of two different approaches
        depending on whether spectral irradiance data is provided.

        Returns
        -------
        None.

        """
        if self.config_dict['spectral']:
            # Power reaching PV module < 1100 nm for AM1.5 spectrum, accounting for PERC reflection
            # only consider light < 1100 nm
            self.res['j_sc'] = (self.res['irrad_poa']
                         *self.loc_data['pv_frac']
                         /self.loc_data['irrad_ref']
                         *self.loss_factor
                         *self.pv_dict['j_sc_STC']
                         )
        else:
            self.res['j_sc'] = self.res['irrad_poa']/1000*self.loss_factor*self.pv_dict['j_sc_STC']

    def get_p_max(self):
        """
        Calculate current-voltage characteristics of PV module and extract maximum power.
        
        This method supports PV modules with and without a sub-bandgap reflecting coating (SBR).
        Based on the value of ``self.ctstring``, the following entries are added to ``self.res``:
        
        - ``power_max_ref`` and ``ey_ref`` for PV modules without a SBR
        - ``power_max_sbr`` and ``ey_sbr`` for PV modules with a SBR

        Returns
        -------
        None.
        """
        
        #Parameters for one-diode model (consider making this part of pv_dict.
        one_diode = OneDiodeModel(tcJsc=self.pv_dict['tc_j_sc']/100,
                                  tcVoc=self.pv_dict['tc_V_oc']/100,
                                  R_shunt=self.pv_dict['R_shunt'],
                                  R_series=self.pv_dict['R_series'],
                                  n=self.pv_dict['n'],
                                  j0=self.pv_dict['j_0']
                                  )
        j_sc = self.res['j_sc']
        cell_temp = self.res['cell_temp_'+self.ctstr]
        iv_paras = one_diode.calc_iv_params(Jsc=j_sc[j_sc > 0.12],
                                            cell_temp = cell_temp[j_sc > 0.12])
        power_max = iv_paras['Pmax'] * 10 / 1000
        self.res['power_max_'+self.ctstr] = power_max
        self.res['ey_'+self.ctstr] = (power_max).sum()

    def get_p_tc(self):
        """
        Calculate power using linear regression depending on irradiance and cell temperature
        using a power temperature coefficient.
        
        This method supports PV modules with and without a sub-bandgap reflecting coating (SBR).
        Based on the value of ``self.ctstring``, the following entries are added to ``self.res``:
        
        - ``power_tc_ref`` and ``ey_tc_ref`` for PV modules without a SBR
        - ``power_tc_sbr`` and ``ey_tc_sbr`` for PV modules with a SBR

        Returns
        -------
        None.

        """
        cell_temp = self.res['cell_temp_'+self.ctstr]
        if self.config_dict['spectral']:
            # Power reachingPV module < 1100 nm for AM1.5 spectrum, accounting for PERC reflection
            power_tc = (self.pv_dict['PCE']/100
                        *self.res['irrad_poa']
                        *self.loc_data['pv_frac']
                        /self.loc_data['pv_frac_am15']
                        *self.loss_factor
                        *(1+self.pv_dict['tc_P']/100*(cell_temp - 25))
                        /1000 # the last /1000 converts from W -> kW
                        )
        else:
            power_tc = (self.pv_dict['PCE']/100
                        *self.res['irrad_poa']
                        *self.loss_factor
                        *(1+self.pv_dict['tc_P']/100*(cell_temp - 25))
                        /1000 # the last /1000 converts from W -> kW
                        )
        self.res['power_tc_'+self.ctstr] = power_tc
        self.res['ey_tc_'+self.ctstr] = (power_tc).sum()

    def get_ey(self):
        """
        Conduct all methods requrired to calculate annual energy yield.
        
        This method uses the one-diode model implemented in the method ``self.get_p_max()``.
        It performs calculation for PV modules with and without sub_bandgap reflector.

        Returns
        -------
        None.

        """
        self.get_irrad_poa()
        for self.coating in (True,False):
            if self.coating:
                self.ctstr = 'sbr'
                self.loss_factor = 1.0-self.pv_dict['loss']
            else:
                self.ctstr = 'ref'
                self.loss_factor = 1.0
            self.get_irrad_therm()
            self.get_temp()
            self.get_j_sc()
            self.get_p_max()

    def get_ey_tc(self):
        """
        Conduct all methods requrired to calculate annual energy yield.
        
        This method uses linear regression to estimate the power implemented 
        in the method ``self.get_p_tc()``.
        It performs calculation for PV modules with and without sub_bandgap reflector.

        Returns
        -------
        None.

        """
        self.get_irrad_poa()
        for self.coating in (True,False):
            if self.coating:
                self.ctstr = 'sbr'
                self.loss_factor = 1.0-self.pv_dict['loss']
            else:
                self.ctstr = 'ref'
                self.loss_factor = 1.0
            self.get_irrad_therm()
            self.get_temp()
            self.get_p_tc()

    def print_res(self):
        """
        Print a selection of results.
        
        Requires prior execution of ``self.get_ey()`` and ``self.get_ey_tc()``, which
        populate the results printed by this method

        """
        print("\nRESULTS\n=======")
        temp_ref = self.res['cell_temp_day_ref']
        temp_sbr = self.res['cell_temp_day_sbr']
        ey_ref = self.res['ey_ref']
        ey_sbr = self.res['ey_sbr']
        ey_tc_ref = self.res['ey_tc_ref']
        ey_tc_sbr = self.res['ey_tc_sbr']
        print(f"Avg. temp ref: {np.mean(temp_ref):.1f} \N{DEGREE SIGN}C")
        print(f"Avg. temp irr: {np.mean(temp_sbr):.1f} \N{DEGREE SIGN}C")
        print(f"Max. ΔT: {max(temp_ref-temp_sbr):.1f} \N{DEGREE SIGN}C")
        print(f"Min. ΔT: {min(temp_ref-temp_sbr):.1f} \N{DEGREE SIGN}C")
        print(f"Avg. ΔT: {np.mean(temp_ref-temp_sbr):.1f} \N{DEGREE SIGN}C")
        print("\nRESULTS ONE-DIODE MODEL WITH tc_V and tc_J")
        print(f"EY ref: {ey_ref:.0f} kWh/m2")
        print(f"EY SBR: {ey_sbr:.0f} kWh/m2")
        print(f"Abs. change: {ey_sbr-ey_ref:.1f} kWh/m2")
        print(f"Rel. change: {(ey_sbr-ey_ref)/ey_ref*100:.1f} %")
        print("\nRESULTS SIMPLE MODEL WITH tc_P")
        print(f"EY ref: {ey_tc_ref:.0f} kWh/m2")
        print(f"EY SBR: {ey_tc_sbr:.0f} kWh/m2")
        print(f"Abs. change: {ey_tc_sbr-ey_tc_ref:.1f} kWh/m2")
        print(f"Rel. change: {(ey_tc_sbr-ey_tc_ref)/ey_tc_ref*100:.1f} %\n")

    def print_res_table(self):
        """
        Print a selection of results separated by tabulators.
        

        Requires prior execution of ``self.get_ey()``, which
        populates the results printed by this method
        """
        temp_ref = self.res['cell_temp_day_ref']
        temp_sbr = self.res['cell_temp_day_sbr']
        ey_ref = self.res['ey_ref']
        ey_sbr = self.res['ey_sbr']
        # Get irradiation
        irrad = self.res['irrad_poa']
        print(f"{np.mean(temp_ref):.1f}\t"
              f"{np.mean(temp_sbr):.1f}\t"
              f"{max(temp_ref-temp_sbr):.1f}\t"
              f"{np.mean((temp_ref-temp_sbr)[irrad > 0]):.1f}\t"
              f"{ey_ref:.0f}\t{ey_sbr:.0f}\t"
              f"{ey_sbr-ey_ref:.1f}\t"
              f"{(ey_sbr-ey_ref)/ey_ref*100:.1f}%")
