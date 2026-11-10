"""
Example for for calculating annual energy yield of photovoltaic (PV) modules
with and without sub-bandgap reflecting coatings.

This example is for a NSRDB* dataset with spectral irradiance data
* National Solar Radiation Database

Example by Klaus Jaeger, 2026-10-14
"""

import tempfile
import zipfile

from pv_tandem.thermal import GetAM15Data, GetLocData, CalcEY

# Initialize PV module dictionary
pv_dict = {
    # Geometrical parameters
    'module_height': 1.5,               # in m
    'module_width': 1.1,                # in m
    'module_length': 1.7,               # in m
    'module_spacing': 100,              # in m
    'module_tilt_deg': 34.5,            # optimal tilt for TMY 2024 data in Princeton
    'module_plane_azimuth_deg': 180,    # PV module is facing South
    'ground_steps': 101,                # steps in view factor model
    
    # Environmental parameter
    'albedo': 0.3,
    
    # Parameters for temperature models (Ross, Fuentes or SAPM, from pvlib)
    'fuentes_module_emissivity': 0.84,  # Default: 0.84, used in Fuentes model
    'noct': 45,                         # Used in Ross model     
    'sapm_module_mount': 0,             # Used in SAPM model
                                        # 0: glass/glass, open rack
                                        # 1: glass/glass, close roof
                                        # 2: glass/polymer, open rack
                                        # 3: glass/polymer, insulated back
    
    # Parameters for sub-bandgap reflector (SBR) and J_sc calculation
    'j_sc_STC_no_EQE': 39,              # Short circuit current density at STC when no EQE data is available
    'use_EQE': True,                    # Use EQE data from an input file
    'wl_j_sc_min_nm': 300,              # lower bound for J_sc calcuation when EQE data is available
    'wl_j_sc_max_ref_nm': 1200,         # upper bound for J_sc calcuation for reference case when EQE data is available
    'wl_SBR_cutoff_nm': 1150,           # cutoff wavelength for the SBR
    
    # Parameters for electrical one-diode model
    'PCE': 24.76,                       # Power conversion efficiency of PV module at STC in %
    'tc_P': -0.40,                      # Temperature coefficient for power in %/K
    'tc_j_sc': 0.03,                    # Temperature coefficient for short-circuit current density in %/K
    'tc_V_oc': -0.4,                    # Temperature coefficient for open-circuit voltage in %/K
    'R_shunt': 3000,                    # Shunt resistance
    'R_series': 1.5,                    # Series resistance
    'n': 1,                             # Diode ideality factor
    'j_0': 1e-12,                       # Dark current density
    
    # Thermal coating parameter
    'loss': 0                           # Loss of thermal coating
    }

# Some general configuration
config_dict = {'irrad_model': 'vf',     # Irradiation model. "vf" or "simple"
               'temp_model': 'SAPM',    # Temperature model "Ross", "Fuentes", or "SAPM"
               'spectral': True,        # Set, whether NSRDB data is spectral or not
               'calc_j_sc': True        # J_sc calculated for each spectrum (instead for AM1.5 only)
              }          

# File paths:
reflection_filepath = 'data_thermal/SEMSC_Subedi_ea_PERC_UVVisNIR_sim_2020_Fig5b_sim.csv'
EQE_filepath = 'data_thermal/SEMSC_Subedi_ea_PERC_UVVisNIR_sim_2020_Fig6_B_sim_enc.csv'
# The NSRDB spectral data are stored as a zip file to keep the repository small
loc_zip_filepath = 'data_thermal/princeton-2024-1223393-fixed_tilt_35_spectral.zip'


am15_data = GetAM15Data(pv_reflection_filepath = reflection_filepath,
                        pv_eqe_filepath = EQE_filepath,
                        wl_jsc_max_ref_nm=pv_dict['wl_j_sc_max_ref_nm'], #
                        wl_sbr_cutoff_nm=pv_dict['wl_SBR_cutoff_nm'] #lambda_cutoff
                        ) 

# Extract the csv file to a temporary folder, which is deleted automatically
# (including the extracted file) at the end of the with block
with tempfile.TemporaryDirectory() as tmp_dir:
    with zipfile.ZipFile(loc_zip_filepath) as zf:
        csv_name = next(n for n in zf.namelist() if n.endswith('.csv'))
        loc_filepath = zf.extract(csv_name, tmp_dir)
 
    loc_data = GetLocData(nsrdb_data_filepath = loc_filepath,
                          pv_reflection_filepath = reflection_filepath,
                          pv_eqe_filepath = EQE_filepath,
                          spectral_boolean = True,
                          wl_jsc_max_ref_nm = pv_dict['wl_j_sc_max_ref_nm'], #
                          wl_sbr_cutoff_nm = pv_dict['wl_SBR_cutoff_nm'] #lambda_cutoff
                         )        

ey_test = CalcEY(am15_data_obj = am15_data,
                 loc_data_obj = loc_data,
                 pv_dict = pv_dict,
                 config_dict = config_dict
                )

ey_test.get_ey()    # Calculate annual energy yield with one-diode model
ey_test.get_ey_tc() # Calculate annual energy yield with simple model
ey_test.print_res() # Print some results