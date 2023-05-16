# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
from pv_tandem import utils

class AM15g:
    """
    This class reads the data from the ASTMG173.csv file and stores it as a pandas Series. It also provides a method to interpolate the spectrum at given wavelengths.

    Attributes:
        spec (pd.Series): The global solar spectrum as a function of wavelength in nm.

    Methods:
        interpolate(wavelengths): Returns the interpolated spectrum at the given wavelengths as a pandas Series.
    """

    def __init__(self):
        """Initializes the AM15g class by reading the data from the csv file."""
        csv_file_path = os.path.join(
            os.path.dirname(__file__), "data", "ASTMG173.csv"
        )
        self.spec = pd.read_csv(csv_file_path, sep=";")
        self.spec.columns = ["wavelength", "extra_terra", "global", "direct"]
        self.spec = self.spec.set_index("wavelength")["global"]
        self.spec = self.spec.loc[300:1200]

    def interpolate(self, wavelengths):
        """Interpolates the spectrum at the given wavelengths.

        Args:
            wavelengths (pd.Index): The wavelengths in nm to interpolate the spectrum at.

        Returns:
            spec_return (pd.Series): The interpolated spectrum as a function of wavelength in nm.
        """
        spec_return = pd.Series(
            np.interp(wavelengths, self.spec.index, self.spec),
            index=wavelengths,
        )
        return spec_return
    
    def calc_jph(self, eqe):
        
        eqe = utils.interp_eqe_to_spec(eqe, self.spec.to_frame().T)
        j_ph = utils.calc_current(self.spec, eqe)
        j_ph = pd.Series(j_ph, index=eqe.columns)
        
        return j_ph