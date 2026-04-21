"""
Example for for calculating annual energy yield of photovoltaic (PV) modules
with and without sub-bandgap reflecting coatings.

This example is for a NSRDB* dataset with spectral irradiance data
* National Solar Radiation Database

Example by Klaus Jaeger, 2026-04-20
"""

from pv_tandem.thermal import GetLocData, CalcEY

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

    # Parameters for electrical one-diode model
    'j_sc_STC': 39,                     # Short circuit current density at STC
    'PCE': 24.76,                       # Power conversion efficiency of PV module at STC in %
    'tc_P': -0.40,                      # Temperature coefficient for power in %/K
    'tc_j_sc': 0.03,                    # Temperature coefficient for short-circuit current density in %/K
    'tc_V_oc': -0.4,                    # Temperature coefficient for open-circuit voltage in %/K
    'R_shunt': 3000,                    # Shunt resistance
    'R_series': 1.5,                    # Series resistance
    'n': 1,                             # Diode ideality factor
    'j_0': 1e-12,                       # Dark current density

    # Thermal coating parameter
    'loss': 0,                          # Loss of thermal coating

    # Filepath to csv file containing PV module reflectance
    'reflection_filepath': 'data_thermal/SEMSC_Subedi_ea_PERC_UVVisNIR_sim_2020_Fig5b.csv'
    }

# General configuration variables
config_dict = {'irrad_model': 'vf',     # Irradiation model. "vf" or "simple"
               'temp_model': 'SAPM',    # Temperature model "Ross", "Fuentes", or "SAPM"
               'spectral': True         # Set, whether NSRDB data is spectral or not
              }

# Load NSRDB data and populate the "loc_data" dictionary
nsrdb_data_filepath = 'data_thermal/princeton-2024-1223393-fixed_tilt_35_spectral.csv'
loc_data = GetLocData(nsrdb_data_filepath = nsrdb_data_filepath,
                      pv_reflection_filepath = pv_dict['reflection_filepath'],
                      spectral_boolean =config_dict['spectral']
                      )

# Initialize Class "CalcEY"
ey_test = CalcEY(loc_data = loc_data.data,
                 pv_dict = pv_dict,
                 config_dict = config_dict)
ey_test.get_ey()    # Calculate annual energy yield with one-diode model
ey_test.get_ey_tc() # Calculate annual energy yield with simple model
ey_test.print_res() # Print some results
