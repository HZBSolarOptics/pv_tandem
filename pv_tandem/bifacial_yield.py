# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from pv_tandem import geo

class IrradianceSimulator:
    def __init__(
        self,
        illumination_df,
        albedo=0.3,
        module_length=1.96,
        module_height=0.5,
        geo_kw_parameter={},
    ):
        """
        Stil needs docstring
        """

        self.geo_instance = None

        self.dni = illumination_df.loc[:, "DNI"]
        self.dhi = illumination_df.loc[:, "DHI"]
        self.albedo = albedo

        self.input_parameter = dict(
            module_length=module_length, mount_height=module_height
        )
        self.input_parameter.update(geo_kw_parameter)
        self.input_parameter["zenith_sun"] = illumination_df.zenith
        self.input_parameter["azimuth_sun"] = illumination_df.azimuth

    def simulate(self, spacing, tilt, simple_results=True):
        """
        Stil needs docstring
        """
        self.input_parameter["module_tilt"] = tilt
        self.input_parameter["module_spacing"] = spacing

        self.geo_instance = geo.ModuleIllumination(**self.input_parameter)
        
        try:
            diffuse = np.concatenate(
                [
                    self.geo_instance.results["irradiance_module_front_sky_diffuse"],
                    self.geo_instance.results["irradiance_module_back_sky_diffuse"],
                    self.geo_instance.results["irradiance_module_front_ground_diffuse"],
                    self.geo_instance.results["irradiance_module_back_ground_diffuse"],
                ],
            )
            diffuse = np.outer(self.dhi, diffuse)
            
        except:
            diffuse = np.concatenate(
                [
                    self.geo_instance.results["irradiance_module_front_sky_diffuse"],
                    self.geo_instance.results["irradiance_module_back_sky_diffuse"],
                ],
            )
            diffuse = np.tile(diffuse, (len(self.dhi),1))
            diffuse = np.concatenate(
                [
                    self.geo_instance.results["irradiance_module_front_ground_diffuse"],
                    self.geo_instance.results["irradiance_module_back_ground_diffuse"],
                    diffuse
                ],
                axis=1
            )*(self.dhi).values[:,None]
            
        direct = np.concatenate(
            [                
                self.geo_instance.results["irradiance_module_front_sky_direct"],
                self.geo_instance.results["irradiance_module_back_sky_direct"],
                self.geo_instance.results["irradiance_module_front_ground_direct"],
                self.geo_instance.results["irradiance_module_back_ground_direct"],
            ],
            axis=1,
        ) * self.dni.values[:,None]

        #direct_ts = direct#self.dni[:, None] * direct

        column_names = ["front_sky", "back_sky","front_ground", "back_ground"]
        prefixes = ["_diffuse", "_direct"]
        column_names = [name + prefix for prefix in prefixes for name in column_names]

        level_names = ["contribution", "module_position"]
        multi_index = pd.MultiIndex.from_product(
            [column_names, range(self.geo_instance.module_steps)], names=level_names
        )

        results = pd.DataFrame(
            np.concatenate([diffuse, direct], axis=1),
            columns=multi_index,
            index=self.dni.index,
        )
        
        ground_reflected = results.columns.get_level_values(0).str.contains('ground')
        
        results.loc[:,ground_reflected] = results.loc[:,ground_reflected].apply(lambda x: x*self.albedo, raw=True, axis=0)
        
        if simple_results:
            back_columns = results.columns.get_level_values("contribution").str.contains(
                "back"
            )
            front_columns = ~back_columns
            results = pd.concat([
                results.loc[:, front_columns].groupby(level="module_position", axis=1).sum().mean(axis=1),
                results.loc[:, back_columns].groupby(level="module_position", axis=1).sum().mean(axis=1)
                ], keys=['front', 'back'], axis=1)
            
        return results

    def calculate_yield(self, spacing, tilt, module_agg_func='min', bifacial=True,
                        front_eff=0.2, back_eff=0.18):
        """
        Stil needs docstring
        """
        results = self.simulate(spacing, tilt)
        back_columns = results.columns.get_level_values("contribution").str.contains(
            "back"
        )
        front_columns = ~back_columns

        if bifacial:
            results.loc[:, back_columns] *= back_eff
            results.loc[:, front_columns] *= front_eff
        else:
            results = results.loc[:, front_columns]
            results *= front_eff

        if self.tmy_data:
            yearly_yield = (
                results.groupby(level="module_position", axis=1)
                .sum()
                .apply(module_agg_func, axis=1)
                .resample("1H")
                .mean()
                .sum()
            )

        else:
            total_yield = (
                results.groupby(level="module_position", axis=1)
                .sum()
                .apply(module_agg_func, axis=1)
            )
            total_yield = total_yield.resample("1H").mean().resample("1D").sum()
            number_of_days = total_yield.index.normalize().nunique()
            yearly_yield = total_yield.sum() / number_of_days * 365

        return yearly_yield / 1000  # convert to kWh
