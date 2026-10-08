# -*- coding: utf-8 -*-
"""
Animation of how the ground irradiance fraction changes over a day.
"""

# %%
# This script is mostly based on the script 'plot_bifacial_basics'.
# The first calculation is repeated for different hours to make an animation.

from pv_tandem.bifacial import ViewFactorSimulator
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Arc
import numpy as np
import pvlib
import pandas as pd
import seaborn as sns

times = pd.date_range(start='2024-06-01 08:00',end='2024-06-1 19:00',freq='h')
coord_berlin = dict(latitude=52.5, longitude=13.4)
solar_pos = pvlib.solarposition.get_solarposition(times, **coord_berlin)

array = np.vstack((times.hour,solar_pos.zenith.to_numpy(),solar_pos.azimuth.to_numpy())).T

for hour, zenith_i, azi_i in array:
    vfs = ViewFactorSimulator(
        module_length=1.92,
        module_tilt=52,
        mount_height=0.5,
        module_spacing=7.3,
        zenith_sun=zenith_i,
        azimuth_sun=azi_i,
        ground_steps=101,
    )
    view_factors = vfs.calculate_view_factors()


    fig, axes = plt.subplots(2,1,dpi=150,figsize=(10,8))
     
    ax = axes[0]
    rect = Rectangle((0,vfs.H),vfs.L,0.2,angle = np.rad2deg(vfs.theta_m_rad), rotation_point = (0,vfs.H))
    ax.add_patch(rect)
    rect = Rectangle((vfs.dist,vfs.H),vfs.L,0.2,angle = np.rad2deg(vfs.theta_m_rad), rotation_point = (vfs.dist,vfs.H))
    ax.add_patch(rect)
    
    arc = Arc((4,0.5),1.5,1.5, theta1 = 90, theta2 = 180, edgecolor='black',facecolor='none',linewidth=2)
    ax.add_patch(arc)
    ax.text(4,0.5+0.85,'90',fontsize=12,color='black',horizontalalignment = 'center',verticalalignment = 'center')
    ax.text(4-0.85,0.5,'0',fontsize=12,color='black',horizontalalignment = 'center',verticalalignment = 'center')
    
    radius_arrow = 0.7
    x_arrow = radius_arrow*np.sin(np.deg2rad(zenith_i))
    y_arrow = radius_arrow*np.cos(np.deg2rad(zenith_i))
    ax.annotate('',xy=(4-x_arrow, 0.5+y_arrow),xytext=(4, 0.5), arrowprops=dict(arrowstyle='->',color='black',linewidth=2))
    
    ax.set_ylim([0,3])
    ax.set_xlim([-0.5,8.3])
    ax.set_title('Time: '+ str(round(hour))+':00',fontsize = 14)
    
    ax = axes[1]

    ax.plot(vfs.x_g_array,view_factors["radiance_ground_diffuse_emitted"] * np.pi * 100)
    ax.plot(vfs.x_g_array,view_factors["radiance_ground_direct_emitted"] * np.pi * 100)

    ax.legend(["fraction of DHI", "fraction of DNI"],fontsize = 12)

    ax.set_ylabel("Ground irradiance fraction (%)",fontsize = 12)
    ax.set_xlabel("Ground array position (m)",fontsize = 12)   
    ax.set_ylim([-1,100])
    ax.set_xlim([-0.5,8.3])

    plt.show()