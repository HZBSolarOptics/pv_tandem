#!/usr/bin/env python

"""Tests for `pv_tandem` package."""

import pytest
import pandas as pd
import numpy as np

from pv_tandem.utils import calc_current


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
    spec = pd.Series(np.ones(181), index = np.arange(300, 1205, 5))
    eqe = pd.Series(np.ones(181)*0.8, index = np.arange(300, 1205, 5))
    
    assert(round(calc_current(spec, eqe)) == 436)

def test_calc_current_ts_spec():
    spec = pd.DataFrame(np.ones(shape=(50,181)), columns = np.arange(300, 1205, 5))
    eqe = pd.Series(np.ones(181)*0.8, index = np.arange(300, 1205, 5))
    
    assert(all(calc_current(spec, eqe).round() == (np.ones(50)*436)))