# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from pv_tandem.bifacial import IrradianceSimulator

class AngularAnalysis(IrradianceSimulator):
    def __init__(self, illumination_df, module_tilt, module_spacing,
    albedo=0.3,
    module_length=1.96,
    mount_height=0.5, 
    theta_bins="iso", filter_component=None, module_position=5, **numeric_kw_parameter):
        super().__init__(
            illumination_df,
            albedo=albedo,
            module_spacing=module_spacing,
            module_tilt=module_tilt,
            module_length=module_length,
            mount_height=mount_height,
            **numeric_kw_parameter
        )
        self.module_position = module_position
        self.poa_irrad = self.simulate(simple_results=False)
        
    def calculate_angular_dist_ground(self, matrix_name, illumination_name):
        eff_matrix = self.results[matrix_name]
    
        ground_illumination = self.results[illumination_name]
        if illumination_name == "radiance_ground_direct_emitted":
            ground_illumination = (
                ground_illumination * self.dni.values[:, None]
            )
        elif illumination_name == "radiance_ground_diffuse_emitted":
            ground_illumination = (
                ground_illumination * self.dhi.values[:, None]
            )
    
        if len(ground_illumination.shape) > 1:
            ground_illumination = ground_illumination.sum(axis=0)
    
        beta_distribution = (
            np.sin(np.linspace(0, np.pi, 180)) ** 2
            / (np.sin(np.linspace(0, np.pi, 180)) ** 2).sum()
        )
    
        angular_dist_1d = (eff_matrix * ground_illumination).sum(axis=-1)
    
        angular_dist_2d = np.multiply.outer(angular_dist_1d, beta_distribution)
    
        mindex = pd.MultiIndex.from_product(
            [
                range(self.module_steps),
                range(self.angle_steps),
                np.linspace(0, np.pi, 180),
            ],
            names=["module_position", "alpha", "beta"],
        )
    
        df = pd.DataFrame({"int": angular_dist_2d.flatten()}, index=mindex)
        df = df.reset_index()
        df["alpha"] = -np.pi / 2 + df["alpha"] / 180 * np.pi
        df["theta"] = np.arccos(np.sin(df["beta"]) * np.cos(df["alpha"]))
        df["phi"] = np.arctan2(1 / np.tan(df["beta"]), np.sin(df["alpha"]))
        return df


    def sum_binned_ground_data(
        self, theta_bins="iso", filter_component=None
    ):
        ground_direct_back = self.calculate_angular_dist_ground(
            matrix_name="module_back_ground_matrix",
            illumination_name="radiance_ground_direct_emitted",
        )
        ground_direct_back = self.bin_theta_phi(
            ground_direct_back, agg_func=np.sum, theta_bins=theta_bins
        )
    
        ground_diffuse_back = self.calculate_angular_dist_ground(
            matrix_name="module_back_ground_matrix",
            illumination_name="radiance_ground_diffuse_emitted",
        )
        ground_diffuse_back = self.bin_theta_phi(
            ground_diffuse_back, agg_func=np.sum, theta_bins=theta_bins
        )
    
        ground_diffuse_front = self.calculate_angular_dist_ground(
            matrix_name="module_front_ground_matrix",
            illumination_name="radiance_ground_diffuse_emitted",
        )
        ground_diffuse_front = self.bin_theta_phi(
            ground_diffuse_front, agg_func=np.sum, theta_bins=theta_bins
        )
    
        ground_direct_front = self.calculate_angular_dist_ground(
            matrix_name="module_front_ground_matrix",
            illumination_name="radiance_ground_direct_emitted",
        )
        ground_direct_front = self.bin_theta_phi(
            ground_direct_front, agg_func=np.sum, theta_bins=theta_bins
        )
    
        if filter_component is None:
            ground_back = ground_direct_back + ground_diffuse_back
            ground_front = ground_diffuse_front + ground_direct_front
        elif filter_component == "direct":
            ground_back = ground_direct_back
            ground_front = ground_direct_front
        elif filter_component == "diffuse":
            ground_back = ground_diffuse_back
            ground_front = ground_diffuse_front
    
        return {"front": ground_front, "back": ground_back}


    def bin_theta_phi(self, irrad_component, agg_func=np.mean, theta_bins="iso"):
        df = irrad_component.loc[irrad_component["module_position"] == self.module_position]
    
        if theta_bins == "iso":
            theta_bins = np.arccos(np.linspace(0, 1, 21))[::-1]
    
        binned_data = (
            df.groupby(
                [
                    pd.cut(df.theta, theta_bins, labels=False),
                    pd.cut(df.phi, np.linspace(-np.pi, np.pi, 37), labels=False),
                ]
            )["int"]
            .apply(agg_func)
            .unstack("phi")
        )
    
        mindex = pd.MultiIndex.from_product(
            [range(len(theta_bins) - 1), range(36)], names=["theta", "phi"]
        )
        template = pd.DataFrame(index=mindex)
        template["int"] = 0
        template = template["int"].unstack("phi")
    
        binned_data = (template + binned_data).fillna(0)
    
        binned_data.index = theta_bins[:-1] / np.pi * 180
        binned_data.columns = (
            binned_data.columns * 360 / (len(binned_data.columns))
            + 360 / (len(binned_data.columns)) / 2
        )
    
        return binned_data


    def process_sky_direct(self, theta_bins="iso"):
        df = self.poa_irrad[
            ["front_sky_direct", "back_sky_direct"]
        ].copy()  # .groupby(level="contribution", axis=1).mean()
        df["theta"] = self.tmp["theta"].copy()
        df["phi"] = self.tmp["phi"].copy()
        df = df.set_index(["theta", "phi"], append=True)
        df = df.stack("module_position")
    
        df = df.reset_index()
    
        df_front = df[
            ["theta", "phi", "front_sky_direct", "module_position"]
        ].copy()
        df_front = df_front.rename(columns={"front_sky_direct": "int"})
    
        df_back = df[["theta", "phi", "back_sky_direct", "module_position"]].copy()
        df_back["theta"] = np.pi - (df_back["theta"])
        df_back = df_back.rename(columns={"back_sky_direct": "int"})
    
        front = self.bin_theta_phi(df_front, agg_func=np.sum, theta_bins=theta_bins)
    
        if all(df_back.theta > np.pi / 2):
            back = front.copy()
            back[:] = 0
        else:
            back = self.bin_theta_phi(df_back, agg_func=np.sum, theta_bins=theta_bins)
    
        return {"front": front, "back": back}


    def process_sky_diffuse(self, theta_bins="iso"):
        def calc_alpha_dist(alpha_array):
            spacing_alpha = np.linspace(-np.pi / 2, np.pi / 2, 360)
            dist_alpha = np.cos(spacing_alpha)
            dist_alpha = dist_alpha / (dist_alpha).sum()
    
            selector = np.greater.outer(spacing_alpha, alpha_array).T
            dist_alpha = np.tile(
                dist_alpha, (self.module_steps, 1)
            )
            dist_alpha[selector] = 0
            return dist_alpha
    
        def calc_beta_dist(alpha_dist):
            beta_dist = (
                np.sin(np.linspace(0, np.pi, 360)) ** 2
                / (np.sin(np.linspace(0, np.pi, 360)) ** 2).sum()
            )
            return np.multiply.outer(alpha_dist, beta_dist)
    
        alpha_front = self.tmp["alpha_2_front"]
        alpha_front_dist = calc_alpha_dist(alpha_front)
        alpha_front_dist = alpha_front_dist[:, ::-1]
    
        front_dist = calc_beta_dist(alpha_front_dist)
    
        alpha_back = -np.pi / 2 + self.tmp["epsilon_1_back"]
        alpha_back_dist = calc_alpha_dist(alpha_back)
    
        # function is not aware from where to check for shadow
        alpha_back_dist = alpha_back_dist[:, ::-1]
        back_dist = calc_beta_dist(alpha_back_dist)
    
        mindex = pd.MultiIndex.from_product(
            [
                range(self.module_steps),
                np.linspace(0, np.pi, 360),
                np.linspace(0, np.pi, 360),
            ],
            names=["module_position", "alpha", "beta"],
        )
    
        df_front = pd.DataFrame(
            {"int": front_dist.flatten()}, index=mindex
        ).reset_index()
        df_back = pd.DataFrame(
            {"int": back_dist.flatten()}, index=mindex
        ).reset_index()
        df = pd.concat([df_front, df_back], keys=["front", "back"], names=["side"])
        df["alpha"] = -np.pi / 2 + df["alpha"]
        df["theta"] = np.arccos(np.sin(df["beta"]) * np.cos(df["alpha"]))
        df["phi"] = np.arctan2(1 / np.tan(df["beta"]), np.sin(df["alpha"]))
    
        binned_front = (
            self.bin_theta_phi(df.loc["front"], agg_func=np.sum, theta_bins=theta_bins)
            * self.dhi.sum()
        )
        binned_back = (
            self.bin_theta_phi(df.loc["back"], agg_func=np.sum, theta_bins=theta_bins)
            * self.dhi.sum()
        )
    
        return {"front": binned_front, "back": binned_back}


    def calc_front_back(
        self, theta_bins="iso", filter_component=None
    ):
        sky_diffuse = self.process_sky_diffuse(theta_bins=theta_bins)
        sky_direct = self.process_sky_direct(theta_bins=theta_bins)
        ground = self.sum_binned_ground_data(
            theta_bins=theta_bins, filter_component=filter_component
        )
    
        if filter_component == "direct":
            front = sky_direct["front"] + ground["front"] * self.albedo
            back = sky_direct["back"] + ground["back"] * self.albedo
        elif filter_component == "diffuse":
            front = sky_diffuse["front"] + ground["front"] * self.albedo
            back = sky_diffuse["back"] + ground["back"] * self.albedo
        else:
            front = (
                sky_diffuse["front"]
                + sky_direct["front"]
                + ground["front"] * self.albedo
            )
            back = (
                sky_diffuse["back"]
                + sky_direct["back"]
                + ground["back"] * self.albedo
            )
    
        return front, back
    
    def plot_polar_heatmap(self, theta_bins='iso', filepath_front=None, filepath_back=None):
        
        import matplotlib.pyplot as plt
        from pathlib import Path
        
        def _plot_polar(df, filepath=None, prefix=None):
            rad = df.index  # np.sin(np.deg2rad(df.index))
            azi = np.deg2rad(df.columns + 180)
    
            r, th = np.meshgrid(rad, azi)
    
            fig = plt.figure(dpi=200)
            plt.subplot(projection="polar")
    
            im = plt.pcolormesh(th, r, df.T, shading="auto")
            # plt.pcolormesh(th, z, r)
    
            plt.plot(azi, r, color="k", ls="none")
    
            ax1 = plt.gca()
            ax1.set_theta_zero_location("N")
    
            # ax1.set_yticks(np.linspace(20,80,4))
            # import matplotlib as mpl
            # mpl.rcParams['ytick.color'] = 'white'
            plt.yticks(
                ticks=np.linspace(30, 60, 2),
                labels=map(lambda x: str(int(x)) + "°", np.linspace(30, 60, 2)),
            )
    
            cax = fig.add_axes([0.3, 0, 0.4, 0.05])
    
            # cax = divider.append_axes('bottom', size='80%', pad=0.6)
    
            plt.colorbar(im, orientation="horizontal", cax=cax)
            cax.set_xlabel(r"Angle resolved radiant exposure" "\n" r"(kWh/m²/year)")
    
            ax1.grid()  # ax1.grid(axis='y')
    
            rlabels = ax1.get_ymajorticklabels()
            for label in rlabels:
                label.set_color("white")
    
            if filepath is not None:
                path = Path(filepath)
                path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(filepath, format="png", bbox_inches="tight")
    
            plt.show()
            
        front, back = self.calc_front_back(theta_bins=theta_bins)
        
        _plot_polar(front/1000, filepath=filepath_front)
        _plot_polar(back/1000, filepath=filepath_back)

if __name__ == "__main__":
    meta_ts = pd.read_csv(
        "../examples/data/meta_ts_dallas_2020.csv",
        index_col=0,
        parse_dates=True,
    )

    illumination_df = meta_ts
    illumination_df = illumination_df.rename(
        columns={
            "Solar Zenith Angle": "zenith",
            "Solar Azimuth Angle": "azimuth",
        }
    )

    illumination_df = illumination_df[["DNI", "DHI", "zenith", "azimuth"]]
    
    

    simulator = AngularAnalysis(illumination_df, module_tilt=25, module_spacing=6)

    #print(simulator.poa_irrad)

    test = simulator.calc_front_back()
    
    print(test)
    
    simulator.plot_polar_heatmap()

# =============================================================================
#     poa_irrad = simulator.poa_irrad
# 
#     test = calc_front_back(
#         simulator, poa_irrad, theta_bins="iso", filter_component=None
#     )
# =============================================================================
