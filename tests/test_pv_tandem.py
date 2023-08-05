#!/usr/bin/env python

"""Tests for `pv_tandem` package."""

import pytest
import pandas as pd
import numpy as np


from pv_tandem import utils

from pv_tandem.irradiance_models import AM15g

from pv_tandem import solarcell_models


@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')


def test_content(response):
    """Sample pytest test function with the pytest fixture as an argument."""
    # from bs4 import BeautifulSoup
    # assert 'GitHub' in BeautifulSoup(response.content).title.string


def test_calc_current_single_spec():
    spec = pd.Series(np.ones(181), index=np.arange(300, 1205, 5))
    eqe = pd.Series(np.ones(181) * 0.8, index=np.arange(300, 1205, 5))

    assert round(utils.calc_current(spec, eqe)) == 436


def test_calc_current_ts_spec():
    spec = pd.DataFrame(
        np.ones(shape=(50, 181)), columns=np.arange(300, 1205, 5)
    )
    eqe = pd.Series(np.ones(181) * 0.8, index=np.arange(300, 1205, 5))

    assert all(utils.calc_current(spec, eqe).round() == (np.ones(50) * 436))


def test_one_diode():
    one_diode = solarcell_models.OneDiodeModel(
        tcJsc=-0.001, tcVoc=-0.0013, R_shunt=1000, R_series=4, n=1.1, j0=2e-16
    )

    iv_paras = one_diode.calc_iv_params(np.array(22), np.array([25]))

    iv_paras = one_diode.calc_iv_params(np.array([22]), np.array([25]))

    assert np.allclose(
        iv_paras, np.array([[1.107403, 0.92986, 18.8762, 0.77833, 21.9, 20.3]])
    )

    iv = one_diode.calc_iv(Jsc=22, cell_temp=25, j_arr=np.array([5, 15, 20]))

    assert np.allclose(iv, np.array([1.07969, 1.01181, 0.94097]))

    iv = one_diode.calc_iv(
        Jsc=np.array([22, 18]),
        cell_temp=np.array([25, 35]),
        j_arr=np.array([5, 15, 20]),
    )

    assert np.allclose(
        iv,
        np.array([[1.07969, 1.01181, 0.94097], [1.05678, 0.96356, -2.27431]]),
    )


def test_tandem_2t_stc():
    eqe = pd.DataFrame(index=np.arange(300, 1210, 10), columns=["pero", "si"])
    eqe[:] = 0
    eqe.loc[eqe.index > 800, "pero"] = 0.8
    eqe.loc[eqe.index < 800, "si"] = 0.8

    electrical_parameters = {
        "Rsh": {"pero": 2000, "si": 5000},
        "RsTandem": 3,
        "j0": {"pero": 2.7e-18, "si": 1e-12},
        "n": {"pero": 1.1, "si": 1},
        "Temp": {"pero": 25, "si": 25},
        "noct": {"pero": 48, "si": 48},
        "tcJsc": {"pero": 0.0002, "si": 0.00032},
        "tcVoc": {"pero": -0.002, "si": -0.0041},
    }

    tandem = solarcell_models.TandemSimulator2T(
        eqe=eqe,
        electrical_parameters=electrical_parameters,
        subcell_names=["pero", "si"],
    )

    iv_stc = tandem.calc_IV_stc()

    assert np.allclose(
        iv_stc.max(),
        pd.Series({"pero": 1.21866, "si": 0.78851, "tandem": 2.00718}),
    )


def test_tandem_2t_spec_ts():
    eqe = pd.Series(np.ones(181) * 0.8, index=np.arange(300, 1205, 5))

    am15g = AM15g()

    spec_mock_ts = (
        pd.DataFrame(
            np.tile(am15g.spec.values[None, :], (3, 1)),
            columns=am15g.spec.index,
        )
        * np.array([1, 0.9, 0.8])[:, None]
    )

    eqe = pd.DataFrame(
        index=spec_mock_ts.columns, columns=["pero", "si"]
    ).astype(float)
    eqe[:] = 0
    eqe.loc[eqe.index > 800, "pero"] = 0.8
    eqe.loc[eqe.index < 800, "si"] = 0.8

    spec_mock_ts = utils.interp_spec_to_eqe(eqe, spec_mock_ts)

    electrical_parameters = {
        "Rsh": {"pero": 2000, "si": 5000},
        "RsTandem": 3,
        "j0": {"pero": 2.7e-18, "si": 1e-12},
        "n": {"pero": 1.1, "si": 1},
        "Temp": {"pero": 25, "si": 25},
        "noct": {"pero": 48, "si": 48},
        "tcJsc": {"pero": 0.0002, "si": 0.00032},
        "tcVoc": {"pero": -0.002, "si": -0.0041},
    }

    tandem = solarcell_models.TandemSimulator2T(
        eqe=eqe,
        electrical_parameters=electrical_parameters,
        subcell_names=["pero", "si"],
    )

    mock_temps = pd.DataFrame(
        {"pero": [20.0, 25.0, 30.0], "si": [20.0, 25.0, 30.0]}
    )

    assert np.allclose(
        tandem.calc_power(spec_irrad=spec_mock_ts, cell_temps=mock_temps),
        np.array([26.8205, 23.6803, 20.6176]),
    )


if __name__ == "__main__":
    pass
