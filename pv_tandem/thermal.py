"""
Annual energy yield (AEY) modelling of silicon PV modules with and without an
ideal sub-bandgap reflector (SBR).

The module provides three classes:

``GetAM15Data``
    Location-independent AM1.5 quantities: band fractions, usable-band
    irradiances and EQE-integrated short-circuit current densities, each
    evaluated for the reference ('ref') and SBR ('sbr') case.
``GetLocData``
    Location-dependent quantities from NSRDB data: weather, solar position and
    (optionally) hourly spectral band fractions.
``CalcEY``
    Combines both data sources to simulate module temperature, current and
    annual energy yield for the 'ref' and 'sbr' cases.

Two wavelength bounds control the model
---------------------------------------
``wl_jsc_max_ref_nm``
    Collection edge of the bare cell (e.g. 1200 nm). Fixed. Determines up to
    which wavelength the REFERENCE module collects current.
``wl_sbr_cutoff_nm``
    Cutoff of the ideal SBR. The reflector is assumed to transmit perfectly
    below and reflect perfectly above this wavelength, so the SBR module
    collects current only up to this value and reflects everything beyond it.

Because the SBR reflects everything above its cutoff, a cutoff below the
collection edge removes weakly collected near-band-edge current. The cutoff
therefore acts on BOTH the current side and the thermal (cooling) side, and
j_sc differs between the 'ref' and 'sbr' cases. Setting
``wl_sbr_cutoff_nm == wl_jsc_max_ref_nm`` must make the two cases identical on
the current side; this is the primary invariant to test after any change.

Units
-----
All wavelength arguments are given in NANOMETRES. Internally the ASTM-G173
grid is in nm (``wl_array_nm``) while the NSRDB spectral column labels are in
micrometres (``wl_array_um``); conversions are made explicit at the point of
use.
"""
import os
import sys
import csv
import warnings
import pandas as pd
import numpy as np
from scipy.interpolate import pchip_interpolate
from scipy.constants import e, h, c
from pv_tandem.solarcell_models import OneDiodeModel
from pv_tandem.bifacial import IrradianceSimulator
import pvlib

class GetAM15Data:
    """ Assembles AM1.5-dependent data required for energy yield calculations"""
    def __init__(self, pv_reflection_filepath, pv_eqe_filepath = None,
                 wl_jsc_max_ref_nm=1200.0, wl_sbr_cutoff_nm=1150.0):
        """
        Initialize the object and populate the ``self.data`` dictionary.

        Parameters
        ----------

        pv_reflection_filepath : str
            Path to CSV file with spectral reflectance data of PV module.
        pv_eqe_filepath : str, optional
            Path to CSV file with the external quantum efficiency (EQE) of the
            PV module. May be given either as a fraction (0-1) or in percent
            (0-100); the scale is detected automatically. If omitted, the
            EQE-based AM1.5 current densities are not computed and
            ``use_EQE=True`` cannot be used in ``CalcEY``.
        wl_jsc_max_ref_nm : float, optional
            Wavelength until where external quantum efficiency (EQE) is
            integrated when no sub-bandgap reflector (SBR) is present.
            Default 1200.0. Given in nm.
        wl_sbr_cutoff_nm : float, optional
            Cutoff wavelength of the ideal SBR [nm], separating the PV-usable
            band (<= cutoff, collected as current) from the sub-bandgap band
            (> cutoff, reflected by the SBR). Default 1150.0.

        Return
        ------
        None
        """

        self.wl_jsc_max_ref_nm = wl_jsc_max_ref_nm
        self.wl_sbr_cutoff_nm = wl_sbr_cutoff_nm

        self.pv_reflection = pd.read_csv(pv_reflection_filepath, skiprows=2)
        self.pv_eqe = pd.read_csv(pv_eqe_filepath, skiprows=2) if pv_eqe_filepath is not None else None

        self.data = {}

        self.read_am15_file()
        self.get_am15_fracs()
        if  self.pv_eqe is not None:
            self.get_am15_currents()

    def read_am15_file(self):
        """
        Read the ASTM-G173 reference spectrum from ``data/ASTMG173.csv``.

        Sets
        ----
        self.wl_array_nm : pandas.Series
            Wavelength grid of the ASTM-G173 spectrum [nm].
        self.am15_array : pandas.Series
            Global tilt spectral irradiance [W m^-2 nm^-1].

        Returns
        -------
        None
        """
        am15_file_path = os.path.join(
            os.path.dirname(__file__), "data", "ASTMG173.csv"
        )
        am15 = pd.read_csv(am15_file_path, delimiter=';')

        self.wl_array_nm = am15['Wvlgth nm']            # ASTM-G173 grid, in nm
        self.am15_array = am15['Global tilt  W*m-2*nm-1']


    def get_am15_fracs(self):
        """
        Calculate AM1.5 band fractions for the 'ref' and 'sbr' cases.

        For each case ctstr in ('ref', 'sbr'), the infrared fraction
        `ir_frac_am15_{ctstr}` (wavelength > cutoff) and the PV-usable
        fraction `pv_frac_am15_{ctstr}` (wavelength <= cutoff) are computed
        and stored in ``self.data``, together with the absolute usable-band
        irradiance `irrad_am15_pv_{ctstr}`. The cutoff is case-dependent:
        `wl_jsc_max_ref_nm` (e.g. 1200 nm) for 'ref' and `wl_sbr_cutoff_nm`
        for 'sbr'. The total AM1.5 irradiance `irrad_am15` is cutoff
        independent and therefore stored once, outside the case loop.

        Returns
        -------
        None
        """

        # Total irradiance by AM1.5 spectrum
        irrad_am15 = np.trapezoid(self.am15_array, self.wl_array_nm)


        # Adapt reflectance data of PV module to wavelenth range
        r_perc_int = np.ones(len(self.wl_array_nm))
        # The R spectrum from Subedi only has values > 300 nm. Hence, interpolation starts at [43].
        r_perc_int[43:1702] = pchip_interpolate(self.pv_reflection['wl'],
                                                self.pv_reflection['R'],
                                                self.wl_array_nm[43:1702])

        # Calculate different irradiance values
        wl_max = {'ref': self.wl_jsc_max_ref_nm, 'sbr': self.wl_sbr_cutoff_nm}
        for ctstr in ['ref', 'sbr']:
            # side='left': the cutoff wavelength itself falls into the
            # IR / sub-bandgap band, consistent with get_am15_currents.
            idx_bg = int(np.searchsorted(np.asarray(self.wl_array_nm, dtype=float),
                                         wl_max[ctstr], side='left'))
            irrad_pv = np.trapezoid(self.am15_array[:idx_bg] * (1 - r_perc_int[:idx_bg]), self.wl_array_nm[:idx_bg])
            irrad_ir = np.trapezoid(self.am15_array[idx_bg:] * (1 - r_perc_int[idx_bg:]), self.wl_array_nm[idx_bg:])
            self.data[f'ir_frac_am15_{ctstr}'] = irrad_ir / irrad_am15
            self.data[f'pv_frac_am15_{ctstr}'] = irrad_pv / irrad_am15
            self.data[f'irrad_am15_pv_{ctstr}'] = irrad_pv
        self.data['irrad_am15'] = irrad_am15


    def get_am15_currents(self, wl_min_nm = 300, theta_in = 0):
        """
        Integrate the EQE against the AM1.5 spectrum to obtain the AM1.5
        short-circuit current density for both cases.

        The lower bound is ``wl_min_nm``; the upper bound is case dependent,
        ``wl_jsc_max_ref_nm`` for 'ref' and ``wl_sbr_cutoff_nm`` for 'sbr'.
        Because the ideal SBR reflects everything above its cutoff, the 'sbr'
        current is reduced by the near-band-edge current between the cutoff and
        the collection edge. With
        ``wl_sbr_cutoff_nm == wl_jsc_max_ref_nm`` both results are identical.

        The EQE scale (0-1 or 0-100) is detected automatically.

        Parameters
        ----------
        wl_min_nm : float, optional
            Lower integration bound [nm]. Default 300.
        theta_in : float, optional
            Angle of incidence [rad]; the current is scaled by cos(theta_in).
            Default 0.

        Sets
        ----
        self.data['j_sc_am15_ref'], self.data['j_sc_am15_sbr'] : float
            AM1.5 short-circuit current density [mA/cm^2].

        Returns
        -------
        None
        """

        # spectral photon flux -> current density prefactor  [mA/cm^2 per (W m^-2 nm^-1)]
        dj_dlambda = self.am15_array * self.wl_array_nm * 1e-9 * e / (h * c) / 10.0
        dj_dlambda*= np.cos(theta_in)        # oblique-incidence correction

        # Convert EQE data to the [0, 1] range if it is given in percent.
        # EQE cannot physically exceed 1, so a maximum above 1.5 is unambiguous.
        eqe = self.pv_eqe['EQE'].copy()
        if eqe.max() > 1.5:
            eqe = eqe / 100

        # The EQE data must cover the integration window; pchip_interpolate
        # would otherwise silently extrapolate.
        wl_max_all = max(self.wl_jsc_max_ref_nm, self.wl_sbr_cutoff_nm)
        if (self.pv_eqe['wl'].min() > wl_min_nm) or (self.pv_eqe['wl'].max() < wl_max_all):
            warnings.warn(
                f"EQE data span {self.pv_eqe['wl'].min():.0f}-{self.pv_eqe['wl'].max():.0f} nm "
                f"but the integration window is {wl_min_nm:.0f}-{wl_max_all:.0f} nm. "
                "pchip_interpolate will extrapolate outside the measured EQE range.",
                stacklevel=2)

        # Interpolate EQE onto the AM1.5 wavelength grid
        abs_fine = pchip_interpolate(self.pv_eqe['wl'],
                                     eqe,
                                     self.wl_array_nm)

        integrand_full = dj_dlambda * abs_fine
        wl_max_nm = {'ref': self.wl_jsc_max_ref_nm, 'sbr': self.wl_sbr_cutoff_nm}
        for ctstr in ['ref', 'sbr']:
            # zero outside the integration window (vectorized, replaces the loop)
            integrand = np.where((self.wl_array_nm >= wl_min_nm) & (self.wl_array_nm <= wl_max_nm[ctstr]), integrand_full, 0.0)
            self.data[f'j_sc_am15_{ctstr}'] = np.trapezoid(integrand, self.wl_array_nm)


class GetLocData:
    """ Assembles location-dependent data required for energy yield calculations"""

    def __init__(self, nsrdb_data_filepath, pv_reflection_filepath, spectral_boolean,
                 pv_eqe_filepath = None, wl_jsc_max_ref_nm=1200.0, wl_sbr_cutoff_nm=1150.0):
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
        pv_eqe_filepath : str, optional
            Path to CSV file with the external quantum efficiency (EQE) of the
            PV module. May be given either as a fraction (0-1) or in percent
            (0-100); the scale is detected automatically. If omitted, the
            EQE-based AM1.5 current densities are not computed and
            ``use_EQE=True`` cannot be used in ``CalcEY``.
        spectral_boolean : bool
            If True, spectral data handling methods are used; otherwise,
            non-spectral methods are applied.
        wl_jsc_max_ref_nm : float, optional
            Wavelength until where external quantum efficiency (EQE) is
            integrated when no sub-bandgap reflector (SBR) is present.
            Default 1200.0. Given in nm.
        wl_sbr_cutoff_nm : float, optional
            Cutoff wavelength of the ideal SBR [nm], separating the PV-usable
            band (<= cutoff, collected as current) from the sub-bandgap band
            (> cutoff, reflected by the SBR). Default 1150.0.

        Return
        ------
        None
        """
        # Both wavelength bounds are stored so every fraction method uses one value.
        self.wl_jsc_max_ref_nm = wl_jsc_max_ref_nm
        self.wl_sbr_cutoff_nm = wl_sbr_cutoff_nm

        if spectral_boolean:
            nsrdb_data_filepath_spectral = nsrdb_data_filepath
            nsrdb_data_filepath_non_spectral = "NSRDB_data_temp.csv"

            self.pv_reflection = pd.read_csv(pv_reflection_filepath, skiprows=2)
            self.pv_eqe = pd.read_csv(pv_eqe_filepath, skiprows=2) if pv_eqe_filepath is not None else None

            self.load_spectral_data(nsrdb_data_filepath_spectral,
                                    nsrdb_data_filepath_non_spectral
                                    )
            self.get_loc_data(nsrdb_data_filepath_non_spectral)
            self.get_spec_fracs()
            if self.pv_eqe is not None:
                self.get_spec_currents()
            os.remove(nsrdb_data_filepath_non_spectral)
        else:
            self.get_loc_data(nsrdb_data_filepath)

    def load_spectral_data(self, input_path, output_path, keep_cols=32, header_rows=2):
        """
        Read an NSRDB spectral CSV file once, write a column-truncated copy to
        ``output_path``, and cache the full data and wavelength grid for reuse by
        other methods.
    
        The first ``header_rows`` metadata lines are copied unchanged. The
        following column-name header row has any embedded commas removed (NSRDB
        spectral files contain a comma inside one header cell, which the
        non-quote-aware parser used by ``pvlib.iotools.read_nsrdb_psm4`` cannot
        handle) and is used both for the output file and to name the columns of
        the cached DataFrame. Only the first ``keep_cols`` columns are written to
        ``output_path``; all columns are kept in the cached data.
    
        Parameters
        ----------
        input_path : str
            Path to the input CSV file with spectral NSRDB data.
        output_path : str
            Path to the truncated output CSV file.
        keep_cols : int, optional
            Number of columns to keep in the truncated output file. The spectral
            irradiance columns begin at this column index. Default 32.
        header_rows : int, optional
            Number of metadata header rows preceding the column-name header row.
            NSRDB data files contain 2 metadata header rows. Default 2.
    
        Returns
        -------
        None
    
        Notes
        -----
        Sets the following attributes for reuse by other methods:
    
        self.spectral : pandas.DataFrame
            Full spectral data (all columns), read once and cached for reuse.
        self.cols : pandas.Index
            Column labels of the spectral-irradiance columns
            (``self.spectral.columns[keep_cols:]``).
        self.wl_array_um : pandas.Index
            Wavelengths corresponding to ``self.cols``, in micrometres.
        """

        with open(input_path, newline="", encoding="utf-8") as infile:
            reader = csv.reader(infile)
            header_line_1 = next(reader)
            header_line_2 = next(reader)
            # Row 3 (column-name header): NSRDB spectral data has a cell containing
            # "," which read_psm4's own (non-quote-aware) parser cannot handle.
            col_header_row = [cell.replace(',', '') for cell in next(reader)]

        # Single full read of the data body (pandas), using the cleaned header.
        spectral = pd.read_csv(input_path, skiprows=header_rows + 1,
                               header=None, names=col_header_row)

        with open(output_path, "w", newline="", encoding="utf-8") as outfile:
            writer = csv.writer(outfile)
            writer.writerow(header_line_1)
            writer.writerow(header_line_2)
            writer.writerow(col_header_row[:keep_cols])
        spectral.iloc[:, :keep_cols].to_csv(output_path, mode='a', index=False, header=False)

        # --- cache for reuse by other methods ---
        self.spectral = spectral
        self.cols = spectral.columns[keep_cols:]
        self.wl_array_um = np.array([float(c[:-3]) for c in self.cols])

    def get_loc_data(self, nsrdb_data_filepath):
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
                                                          latitude=nsrdb_data[1]['latitude'],
                                                          longitude=nsrdb_data[1]['longitude'],
                                                          altitude=nsrdb_data[1]['altitude'],
                                                          pressure=nsrdb_data[0]['pressure'],
                                                          temperature=nsrdb_data[0]['temp_air']
                                                          )

        loc_data = nsrdb_data[0].copy()
        loc_data['zenith'] = solar_pos['zenith']
        loc_data['apparent_zenith'] = solar_pos['apparent_zenith']
        loc_data['azimuth'] = solar_pos['azimuth']

        # Adjust time series so that all values are in one year to have Fuentes model working
        loc_data.index = pd.date_range(loc_data.index[0], periods=len(loc_data), freq='h')

        self.data = loc_data

    def get_spec_fracs(self):
        """
        Calculate hourly spectral fractions from spectral NSRDB data.

        For each case ctstr in ('ref', 'sbr'), the infrared fraction
        `ir_frac_{ctstr}` (wavelength > cutoff) and the PV-usable fraction
        `pv_frac_{ctstr}` (wavelength <= cutoff) are computed per hour and
        stored in ``self.data``. The cutoff is case-dependent:
        `wl_jsc_max_ref_nm` (e.g. 1200 nm) for 'ref' and `wl_sbr_cutoff_nm`
        for 'sbr'.

        Returns
        -------
        None.
        """

        cols = self.cols                 # cached from load_spectral_data
        wl_array_um = self.wl_array_um   # cached from load_spectral_data

        r_perc_int = np.ones(len(wl_array_um))
        # The R spectrum from Subedi only has values 300 nm < lambda < 2500 nm,
        # hence interpolation is between [43:1702].
        # NSRDB column labels are in um while pv_reflection['wl'] is in nm,
        # so the reflectance grid is converted to um here (/1000).
        r_perc_int[43:1702] = pchip_interpolate(self.pv_reflection['wl'] / 1000,
                                                self.pv_reflection['R'],
                                                wl_array_um[43:1702])

        # The cutoff index is derived from the wavelength bounds, and the
        # dataframe-column slices are derived from that index, so the
        # wavelength-array index and the raw-column index cannot drift apart.

        # calculate irradiances
        irr_tot = np.trapezoid(self.spectral[cols], wl_array_um)      # total

        wl_max = {'ref': self.wl_jsc_max_ref_nm, 'sbr': self.wl_sbr_cutoff_nm}
        for ctstr in ['ref', 'sbr']:
            idx_bg = int(np.searchsorted(wl_array_um, wl_max[ctstr] /1000, side='left'))

            cols_pv = cols[:idx_bg]    # wavelengths <= cutoff (PV-usable band)
            cols_ir = cols[idx_bg:]    # wavelengths >  cutoff (sub-bandgap / SBR band)

            # Fraction absorbed by PV module for wavelengths <= cutoff (PV-usable band)
            irr_pv_abs = np.trapezoid(self.spectral[cols_pv] * (1 - r_perc_int[:idx_bg]),
                                      wl_array_um[:idx_bg])
            # Fraction absorbed by PV module for wavelengths > cutoff (sub-bandgap band)
            irr_ir_abs = np.trapezoid(self.spectral[cols_ir] * (1 - r_perc_int[idx_bg:]),
                                      wl_array_um[idx_bg:])

            # Calculate fractions (done such that np.divide does not cause strange behavior)
            pv_frac = np.zeros(len(irr_tot))
            ir_frac = np.zeros(len(irr_tot))
            idx = np.where(irr_tot > 0)
            pv_frac[idx] = np.divide(irr_pv_abs[idx], irr_tot[idx])
            ir_frac[idx] = np.divide(irr_ir_abs[idx], irr_tot[idx])

            self.data[f'pv_frac_{ctstr}'] = pv_frac
            self.data[f'ir_frac_{ctstr}'] = ir_frac
        self.data['irrad_spec'] = irr_tot

    def get_spec_currents(self, wl_min_nm = 300, theta_in = 0):
        """
        Integrate the EQE against the hourly NSRDB spectra to obtain the
        short-circuit current density for each timestamp, for both cases.
    
        The current density is obtained by direct integration of the measured
        external quantum efficiency against the time-dependent spectral
        irradiance, without recourse to a reference spectrum or a regression
        approach. The lower integration bound is ``wl_min_nm``; the upper bound
        is case dependent, ``wl_jsc_max_ref_nm`` for 'ref' and
        ``wl_sbr_cutoff_nm`` for 'sbr'. Because the ideal SBR reflects everything
        above its cutoff, the 'sbr' current is reduced by the near-band-edge
        current between the cutoff and the collection edge. With
        ``wl_sbr_cutoff_nm == wl_jsc_max_ref_nm`` both results are identical.
    
        The EQE scale (0-1 or 0-100) is detected automatically. A warning is
        issued if the EQE data do not cover the full integration window, since
        the interpolation would then extrapolate.
    
        Parameters
        ----------
        wl_min_nm : float, optional
            Lower integration bound [nm]. Default 300.
        theta_in : float, optional
            Angle of incidence [rad]; the current is scaled by cos(theta_in).
            Default 0.
    
        Returns
        -------
        None
    
        Notes
        -----
        Uses the cached ``self.spectral``, ``self.cols`` and ``self.wl_array_um``
        set by ``load_spectral_data``, and sets the following entries of
        ``self.data``, each an array with one value per timestamp:
    
        j_sc_spec_ref, j_sc_spec_sbr : ndarray
            Short-circuit current density [mA/cm^2].
        j_sc_norm_spec_ref, j_sc_norm_spec_sbr : ndarray
            Current density normalised by the total incident irradiance;
            zero where the irradiance vanishes.
        """

        cols = self.cols                 # cached from load_spectral_data
        wl_array_um = self.wl_array_um   # cached from load_spectral_data
        wl_array_nm = wl_array_um * 1000


        # spectral photon flux -> current density prefactor  [mA/cm^2 per (W m^-2 nm^-1)]
        dj_dlambda = self.spectral[cols].to_numpy() * wl_array_um[np.newaxis, :] * 1e-6 * e / (h * c) / 10.0
        dj_dlambda*= np.cos(theta_in)        # oblique-incidence correction

        # Convert EQE data to the [0, 1] range if it is given in percent.
        # EQE cannot physically exceed 1, so a maximum above 1.5 is unambiguous.
        eqe = self.pv_eqe['EQE'].copy()
        if eqe.max() > 1.5:
            eqe = eqe / 100

        # The EQE data must cover the integration window; pchip_interpolate
        # would otherwise silently extrapolate.
        wl_max_all = max(self.wl_jsc_max_ref_nm, self.wl_sbr_cutoff_nm)
        if (self.pv_eqe['wl'].min() > wl_min_nm) or (self.pv_eqe['wl'].max() < wl_max_all):
            warnings.warn(
                f"EQE data span {self.pv_eqe['wl'].min():.0f}-{self.pv_eqe['wl'].max():.0f} nm "
                f"but the integration window is {wl_min_nm:.0f}-{wl_max_all:.0f} nm. "
                "pchip_interpolate will extrapolate outside the measured EQE range.",
                stacklevel=2)

        # Interpolate EQE onto the AM1.5 wavelength grid
        abs_fine = pchip_interpolate(self.pv_eqe['wl'],
                                     eqe,
                                     wl_array_nm)

        integrand_full = dj_dlambda * abs_fine
        wl_max_nm = {'ref': self.wl_jsc_max_ref_nm, 'sbr': self.wl_sbr_cutoff_nm}
        #idx_irrad = np.where(self.data['irrad_spec'] > 0)[0]
        for ctstr in ['ref', 'sbr']:
            # zero outside the integration window (integration vectorized)
            integrand = np.where((wl_array_nm >= wl_min_nm) & (wl_array_nm <= wl_max_nm[ctstr]), integrand_full, 0.0)
            j_sc_spec = np.trapezoid(integrand, wl_array_um, axis = 1) # note that wl here is in um!
            irr = np.asarray(self.data['irrad_spec'], dtype=float)
            j_sc_norm_spec = np.divide(j_sc_spec, irr, out=np.zeros_like(j_sc_spec, dtype=float), where=irr > 0) # normalized by total irradiance
            self.data[f'j_sc_spec_{ctstr}'] = j_sc_spec
            self.data[f'j_sc_norm_spec_{ctstr}'] = j_sc_norm_spec


# Class for calculating EY
class CalcEY:
    """Contains all methods required for calculating the annual energy yield"""

    def __init__(self, am15_data_obj, loc_data_obj, pv_dict, config_dict):
        """
        Initialize the object.

        Parameters
        ----------
        am15_data_obj : GetAM15Data
            Instance of ``GetAM15Data``; its ``.data`` dict and wavelength
            bounds are used.
        loc_data_obj : GetLocData
            Instance of ``GetLocData``; its ``.data`` dict and wavelength
            bounds are used.
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
        # --- Consistency checks:
        # 1) both data sources must use the same
        # wavelength bounds, since the SBR/reference split has to agree
        # between the AM1.5 and location-dependent fraction calculations.
        for attr in ('wl_jsc_max_ref_nm', 'wl_sbr_cutoff_nm'):
            v_am15 = getattr(am15_data_obj, attr)
            v_loc = getattr(loc_data_obj, attr)
            if v_am15 != v_loc:
                raise ValueError(
                    f"{attr} differs between GetAM15Data ({v_am15}) and "
                    f"GetLocData ({v_loc}). Both must use the same wavelength bounds.")
        # 2) calc_j_sc=True requires EQE-derived current data to be available:
        #    both use_EQE=True and an EQE file actually loaded into loc_data_obj.
        if config_dict.get('calc_j_sc'):
            if not config_dict.get('spectral'):
                raise ValueError(
                    "calc_j_sc=True requires spectral weather data (spectral=True), "
                    "since per-spectrum current integration needs hourly spectra.")
            if not pv_dict.get('use_EQE'):
                raise ValueError(
                    "calc_j_sc=True requires EQE data (use_EQE=True), since the "
                    "per-spectrum current densities are obtained by integrating the EQE.")
            if getattr(loc_data_obj, 'pv_eqe', None) is None:
                raise ValueError(
                    "calc_j_sc=True requires EQE data, but GetLocData was constructed "
                    "without pv_eqe_filepath.")
        # --- End consistency checks.

        self.am15_data = am15_data_obj.data
        self.loc_data = loc_data_obj.data
        self.pv_dict = pv_dict
        self.config_dict = config_dict
        # Per-case loop state, set together in get_ey() / get_ey_tc()
        self.coating = None      # True for the SBR case, False for the reference
        self.ctstr = None        # 'sbr' or 'ref'; used to key all result entries
        self.loss_factor = None  # optical loss of a non-ideal SBR
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

        # assuming 80% of the DHI are reaching the 20 deg tilted module plane
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
        [1] K. Jaeger, P. Tillmann, and C. Becker, Optics Express 28, 4751 (2020).
        """
        illum_full = self.loc_data.copy()
        illum = illum_full[illum_full['zenith'] < 90]
        illum = illum.rename(columns={'dhi': 'DHI', 'dni': 'DNI'})
        illum = illum[['DNI', 'DHI', 'zenith', 'azimuth']]

        simulator = IrradianceSimulator(illum,
                                        albedo=self.pv_dict['albedo'],
                                        module_length=self.pv_dict['module_length'],
                                        mount_height=self.pv_dict['module_height'],
                                        module_tilt=self.pv_dict['module_tilt_deg'],
                                        module_spacing=self.pv_dict['module_spacing'],
                                        ground_steps=self.pv_dict['ground_steps']
                                        )

        irrad_poa_vf = simulator.simulate(simple_results=True)
        irrad_poa_vf = irrad_poa_vf.reindex(index=illum_full.index, fill_value=0)

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
        else otherwise. The array `irrad_therm` is added to ``self.res``.

        Returns
        -------
        None.
        """
        if self.coating:
            if self.config_dict['spectral']:
                irrad_factor = 1 - self.loc_data[f'ir_frac_{self.ctstr}']
            else:
                irrad_factor = 1 - self.am15_data[f'ir_frac_am15_{self.ctstr}']
            self.res['irrad_therm'] = self.res['irrad_poa'] * irrad_factor * self.loss_factor
        else:
            self.res['irrad_therm'] = self.res['irrad_poa']

    def get_temp(self):
        """
        Calculate the solar cell temperature and store results in ``self.res``.

        Depending on the value of ``self.config_dict['temp_model']``, one of the
        following temperature models provided by pvlib is used:
        Ross model, Fuentes model, or SAPM model (when SAPM is used, the module
        temperature is also calculated).

        Returns
        -------
        None
        """
        match self.config_dict['temp_model']:
            case "Ross":
                cell_temp = pvlib.temperature.ross(
                    poa_global=self.res['irrad_therm'],
                    temp_air=self.loc_data['temp_air'],
                    noct=45
                )
            case "Fuentes":
                cell_temp = pvlib.temperature.fuentes(
                    poa_global=self.res['irrad_therm'],
                    temp_air=self.loc_data['temp_air'],
                    wind_speed=self.loc_data['wind_speed'],
                    module_height=self.pv_dict['module_height'],
                    module_width=self.pv_dict['module_width'],
                    module_length=self.pv_dict['module_length'],
                    surface_tilt=self.pv_dict['module_tilt_deg'],
                    noct_installed=self.pv_dict['noct'],
                    emissivity=self.pv_dict['fuentes_module_emissivity']
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
                cell_temp = pvlib.temperature.sapm_cell(
                    poa_global=self.res['irrad_therm'],
                    temp_air=self.loc_data['temp_air'],
                    wind_speed=self.loc_data['wind_speed'],
                    **sapm_params_dict
                )

                module_temp = pvlib.temperature.sapm_module(
                    poa_global=self.res['irrad_therm'],
                    temp_air=self.loc_data['temp_air'],
                    wind_speed=self.loc_data['wind_speed'],
                    a=sapm_params_dict['a'],
                    b=sapm_params_dict['b']
                )
                self.res['module_temp_' + self.ctstr] = module_temp
                # Values for daytime only
                self.res['module_temp_day_' + self.ctstr] = module_temp[self.loc_data['zenith'] < 90]
            case _:
                print('Please use a valid temperature model (Ross, Fuentes or SAPM).')
                sys.exit()

        self.res['cell_temp_' + self.ctstr] = cell_temp
        self.res['cell_temp_day_' + self.ctstr] = cell_temp[self.loc_data['zenith'] < 90]  # Daytime only

    def get_j_sc(self):
        """
        Calculate the short-circuit current density for the 'ref' module (no
        coating) and the 'sbr' module (with ideal SBR), and store the result in
        ``self.res[f'j_sc_{self.ctstr}']``.
    
        Three paths are available, selected by ``config_dict``:
    
        * spectral data with ``calc_j_sc=True``: the current is obtained from the
          per-timestamp current densities ``j_sc_norm_spec_{ctstr}``, which were
          computed by direct integration of the EQE against the hourly spectra
          (see ``GetLocData.get_spec_currents``).
        * spectral data with ``calc_j_sc=False``: a linear-regression approach is
          used instead, scaling the AM1.5 reference current by the ratio of the
          usable-band irradiance to its AM1.5 counterpart.
        * no spectral data: the AM1.5 spectrum is assumed throughout and the
          current scales linearly with the plane-of-array irradiance.
    
        The SBR reflects all light above its cutoff wavelength, so the 'sbr'
        module collects current only up to ``wl_sbr_cutoff_nm``, while the 'ref'
        module collects up to ``wl_jsc_max_ref_nm`` (e.g. 1200 nm). The cutoff
        therefore affects the SBR case on both the current side and the thermal
        (cooling) side.
    
        When ``use_EQE`` is False, both cases use the cutoff-independent anchor
        ``j_sc_STC_no_EQE``, so no current penalty is applied and the SBR differs
        from the reference only through cooling.
    
        Returns
        -------
        None
    """

        if self.pv_dict['use_EQE']:
            j_sc_am15 = self.am15_data[f'j_sc_am15_{self.ctstr}']
        else:
            j_sc_am15 = self.pv_dict['j_sc_STC_no_EQE']


        if self.config_dict['spectral']:
            if self.config_dict['calc_j_sc']: # Jsc calculated for every spectrum
                self.res[f'j_sc_{self.ctstr}'] = (self.res['irrad_poa']
                                                  *self.loc_data[f'j_sc_norm_spec_{self.ctstr}']
                                                  *self.loss_factor
                                                  )

                # Direct results from NSRDB data without optical model for total irradiance
                #self.res[f'j_sc_{self.ctstr}'] = self.loc_data[f'j_sc_spec_{self.ctstr}']

            else:
                # Usable-band POA irradiance (<= cutoff), normalized by the AM1.5
                # usable-band reference, scaled to STC short-circuit current.
                self.res[f'j_sc_{self.ctstr}'] = (self.res['irrad_poa']
                                                  * self.loc_data[f'pv_frac_{self.ctstr}']
                                                  / self.am15_data[f'irrad_am15_pv_{self.ctstr}']
                                                  * self.loss_factor
                                                  * j_sc_am15
                                                  )
        else:
            self.res[f'j_sc_{self.ctstr}'] = self.res['irrad_poa'] / self.am15_data['irrad_am15'] * self.loss_factor * j_sc_am15

    def get_p_max(self):
        """
        Calculate current-voltage characteristics of PV module and extract maximum power.

        Returns
        -------
        None.
        """
        one_diode = OneDiodeModel(tcJsc=self.pv_dict['tc_j_sc'] / 100,
                                  tcVoc=self.pv_dict['tc_V_oc'] / 100,
                                  R_shunt=self.pv_dict['R_shunt'],
                                  R_series=self.pv_dict['R_series'],
                                  n=self.pv_dict['n'],
                                  j0=self.pv_dict['j_0']
                                  )

        j_sc = self.res[f'j_sc_{self.ctstr}']
        cell_temp = self.res[f'cell_temp_{self.ctstr}']
        iv_paras = one_diode.calc_iv_params(Jsc=j_sc[j_sc > 0.12],
                                            cell_temp=cell_temp[j_sc > 0.12])
        power_max = iv_paras['Pmax'] * 10 / 1000
        self.res[f'power_max_{self.ctstr}'] = power_max
        self.res[f'ey_{self.ctstr}'] = (power_max).sum()

    def get_p_tc(self):
        """
        Calculate power from irradiance and cell temperature using the power
        temperature coefficient ``tc_P``.

        The factor ``j_sc_am15_{ctstr} / j_sc_am15_ref`` accounts for the
        reduced current of the SBR module when its cutoff lies below the
        collection edge. It equals 1 for the reference case and for
        ``wl_sbr_cutoff_nm == wl_jsc_max_ref_nm``. The separate factor
        ``pv_frac_{ctstr} / pv_frac_am15_{ctstr}`` carries only the deviation
        of the actual spectrum from AM1.5, since the case-dependent band
        cancels between numerator and denominator.

        Returns
        -------
        None.
        """
        cell_temp = self.res[f'cell_temp_{self.ctstr}']
        if self.config_dict['spectral']:
            if self.config_dict['calc_j_sc']:
                # The loss factor is included in res[f'j_sc_{self.ctstr}']
                power_tc = (self.pv_dict['PCE'] / 100
                            * self.am15_data['irrad_am15']
                            * self.res[f'j_sc_{self.ctstr}']
                            / self.am15_data['j_sc_am15_ref']
                            * (1 + self.pv_dict['tc_P'] / 100 * (cell_temp - 25))
                            / 1000  # the last /1000 converts from W -> kW
                            )
            else:
                # Usable-band POA power (<= cutoff), normalized by the AM1.5 usable fraction.
                power_tc = (self.pv_dict['PCE'] / 100
                            * self.res['irrad_poa']
                            * self.am15_data[f'j_sc_am15_{self.ctstr}']
                            / self.am15_data['j_sc_am15_ref']
                            * self.loc_data[f'pv_frac_{self.ctstr}']
                            / self.am15_data[f'pv_frac_am15_{self.ctstr}']
                            * self.loss_factor
                            * (1 + self.pv_dict['tc_P'] / 100 * (cell_temp - 25))
                            / 1000  # the last /1000 converts from W -> kW
                            )
        else:
            power_tc = (self.pv_dict['PCE'] / 100
                        * self.res['irrad_poa']
                        * self.am15_data[f'j_sc_am15_{self.ctstr}']
                        / self.am15_data['j_sc_am15_ref']
                        * self.loss_factor
                        * (1 + self.pv_dict['tc_P'] / 100 * (cell_temp - 25))
                        / 1000  # the last /1000 converts from W -> kW
                        )
        self.res[f'power_tc_{self.ctstr}'] = power_tc
        self.res[f'ey_tc_{self.ctstr}'] = (power_tc).sum()

    def get_ey(self):
        """
        Conduct all methods required to calculate annual energy yield using the one-diode model.

        Returns
        -------
        None.
        """
        self.get_irrad_poa()
        for self.coating in (True, False):
            if self.coating:
                self.ctstr = 'sbr'
                self.loss_factor = 1.0 - self.pv_dict['loss']
            else:
                self.ctstr = 'ref'
                self.loss_factor = 1.0
            self.get_irrad_therm()
            self.get_temp()
            self.get_j_sc()
            self.get_p_max()

    def get_ey_tc(self):
        """
        Conduct all methods required to calculate annual energy yield using the tc-based
        linear regression for the power.

        Returns
        -------
        None.
        """
        self.get_irrad_poa()
        # The reference case is evaluated first for readability; get_p_tc()
        # takes j_sc_am15_ref from am15_data, so the order is not critical.
        for self.coating in (False, True):
            if self.coating:
                self.ctstr = 'sbr'
                self.loss_factor = 1.0 - self.pv_dict['loss']
            else:
                self.ctstr = 'ref'
                self.loss_factor = 1.0
            self.get_irrad_therm()
            self.get_temp()
            if self.config_dict['spectral'] and self.config_dict['calc_j_sc']:
                self.get_j_sc()
            self.get_p_tc()

    def print_res(self):
        """
        Print a selection of results.
        Requires prior execution of ``self.get_ey()`` and ``self.get_ey_tc()``.
        """
        
        def irr_weighted(x, w):
            return (x * w).sum() / w.sum()
        
        T_ref = self.res['cell_temp_day_ref']
        T_sbr = self.res['cell_temp_day_sbr']
        dT = T_ref - T_sbr
        ey_ref = self.res['ey_ref']
        ey_sbr = self.res['ey_sbr']
        ey_tc_ref = self.res['ey_tc_ref']
        ey_tc_sbr = self.res['ey_tc_sbr']
        irr = self.res['irrad_poa']
        
        print("\nRESULTS\n=======")

        print(f"Avg. temp ref: {irr_weighted(T_ref,irr):.1f} \N{DEGREE SIGN}C")
        print(f"Avg. temp SBR: {irr_weighted(T_sbr,irr):.1f} \N{DEGREE SIGN}C")
        print(f"Max. dT: {max(T_ref-T_sbr):.1f} \N{DEGREE SIGN}C")
        print(f"Avg. dT: {irr_weighted(dT,irr):.1f} \N{DEGREE SIGN}C")

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
        Requires prior execution of ``self.get_ey()``.
        """
        
        def irr_weighted(x, w):
            return (x * w).sum() / w.sum()
        
        T_ref = self.res['cell_temp_day_ref']
        T_sbr = self.res['cell_temp_day_sbr']
        ey_ref = self.res['ey_ref']
        ey_sbr = self.res['ey_sbr']
        dT = T_ref - T_sbr
        irr = self.res['irrad_poa']

        print(f"{irr_weighted(T_ref,irr):.1f}\t"
              f"{irr_weighted(T_sbr,irr):.1f}\t"
              f"{max(T_ref-T_sbr):.1f}\t"
              f"{irr_weighted(dT,irr):.1f}\t"
              f"{ey_ref:.0f}\t{ey_sbr:.0f}\t"
              f"{ey_sbr-ey_ref:.1f}\t"
              f"{(ey_sbr-ey_ref)/ey_ref*100:.1f}%")
        