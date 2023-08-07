# -*- coding: utf-8 -*-

import pytest
import pandas as pd
import numpy as np

from pv_tandem.bifacial import ViewFactorSimulator, IrradianceSimulator

coord_berlin = dict(latitude =52.5, longitude =13.4)

def test_viewfactors_single():

    vfs = ViewFactorSimulator(module_length=1.92,
        module_tilt=52,
        mount_height=0.5,
        module_spacing=7.3,
        zenith_sun=31.9,
        azimuth_sun=144.1,
        module_steps=4,
        ground_steps=11)
    
    view_factors = vfs.calculate_view_factors()
    
    assert(np.allclose(view_factors['radiance_ground_diffuse_emitted'],
                   np.array([0.225444, 0.140559, 0.178011, 0.222844, 0.253663, 0.270248,
                          0.277571, 0.279531, 0.278113, 0.282878, 0.225444])))
    
    assert(np.allclose(view_factors['radiance_ground_direct_emitted'],
                   np.array([0.270236, 0.      , 0.      , 0.      , 0.270236, 0.270236,
                          0.270236, 0.270236, 0.270236, 0.270236, 0.270236])))
    
    assert(np.allclose(view_factors['irradiance_module_front_sky_direct'],
              np.array([0.859993, 0.859993, 0.859993, 0.859993])))
              
    assert(np.allclose(view_factors['irradiance_module_front_sky_diffuse'],
              np.array([0.719731, 0.748479, 0.774291, 0.797308])))
              
    assert(np.allclose(view_factors['irradiance_module_back_sky_direct'],
              np.array([0., 0., 0., 0.])))
              
    assert(np.allclose(view_factors['irradiance_module_back_sky_diffuse'],
              np.array([0.13417 , 0.148247, 0.1642  , 0.182267])))
              
    assert(np.allclose(view_factors['irradiance_module_front_ground_direct'],
              np.array([0.130184, 0.120346, 0.110021, 0.09758 ])))
              
    assert(np.allclose(view_factors['irradiance_module_front_ground_diffuse'],
              np.array([0.133376, 0.126784, 0.117955, 0.106802])))
              
    assert(np.allclose(view_factors['irradiance_module_back_ground_direct'],
              np.array([0.162969, 0.191026, 0.254374, 0.315642])))
              
    assert(np.allclose(view_factors['irradiance_module_back_ground_diffuse'],
              np.array([0.470705, 0.490032, 0.509953, 0.514866])))
          
def test_viewfactors_ts():
    
    vfs = ViewFactorSimulator(module_length=1.92,
        module_tilt=30,
        mount_height=0.5,
        module_spacing=7.3,
        zenith_sun=np.array([40,30,35]),
        azimuth_sun=np.array([170,180,190]),
        ground_steps=101,
        module_steps=12,
        angle_steps=180,)

    view_factors = vfs.calculate_view_factors()
    
    assert(np.allclose(view_factors['irradiance_module_front_sky_direct_mean'],
              np.array([0.979925, 1.      , 0.991838])))
              
    assert(np.allclose(view_factors['irradiance_module_back_sky_direct_mean'],
              np.array([0., 0., 0.])))
              
    assert(np.allclose(view_factors['irradiance_module_front_sky_diffuse_mean'],
              0.911888))
              
    assert(np.allclose(view_factors['irradiance_module_back_sky_diffuse_mean'],
              0.053632))
              
    assert(np.allclose(view_factors['irradiance_module_front_ground_direct_mean'],
              0.034587))
              
    assert(np.allclose(view_factors['irradiance_module_back_ground_direct_mean'],
              0.233631))
              
    assert(np.allclose(view_factors['irradiance_module_front_ground_diffuse_mean'],
              0.043895))
              
    assert(np.allclose(view_factors['irradiance_module_back_ground_diffuse_mean'],
              0.480027))
    