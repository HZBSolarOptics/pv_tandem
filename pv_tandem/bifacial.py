# -*- coding: utf-8 -*-

import numpy as np
from numpy.linalg import norm
import warnings
import pandas as pd


def section(k_1, d_1, k_2, d_2):  # intersection of two lines
    section_x = (d_1 - d_2) / (k_2 - k_1)
    section_y = k_1 * section_x + d_1
    return (section_x, section_y)


def projection(a, b):
    """calculates the projection between two lines"""
    try:
        projection = (
            np.cross(b, np.cross(a, b) / (norm(b, axis=-1)[:, None]))
            / norm(b, axis=-1)[:, None]
        )
    except:
        projection = np.cross(b, np.cross(a, b) / norm(b)) / norm(b)

    return projection


class ViewFactorSimulator:
    def __init__(
        self,
        module_length=1.92,
        module_tilt=52,
        mount_height=0.5,
        module_spacing=7.1,
        zenith_sun=30,
        azimuth_sun=150,
        ground_steps=101,
        module_steps=12,
        angle_steps=180,
    ):
        """
        Simulation of view factors for illumination for a bifacial solar panel in a periodic
        south facing array.

        Accepts array inputs for sun position for fast time series evaluation.

        Parameters
        ----------
        module_length : numeric
            length of the module used for the array

        module_tilt : numeric
            tilt angle of the module

        mount_height : numeric
            Mounting height of the modules. Difined as distance between the lowest
            point of the module and the ground

        module_distance:
            Distance between modules. Defined as distance between lowest point
            of one module to the same point on the next row.

        zenith_sun : numeric or array-like
            Zenith angle of the sun in degrees

        azimuth_sun : numeric or array-like
            Azimuth angle of the sun in degrees. 180 degrees defined as south.

        ground_steps : int
            Resolution on the ground where the irradiance is evaluated

        module_steps : int
            Resolution on the module where the irradiance is evaluated

        angle_steps : int
            Angular resolution of the ground radiance
        """

        # test variable for development
        self.tmp = {}

        self.L = module_length
        self.theta_m_rad = np.deg2rad(module_tilt)
        self.H = mount_height
        self.dist = module_spacing

        if np.any(zenith_sun > 90):
            warnings.warn(
                "Zenith angle larger then 90 deg was passed to simulation. Zenith angle is truncted to 90."
            )
            zenith_sun[zenith_sun > 90] = 90
        self.theta_S_rad = np.deg2rad(zenith_sun)
        self.phi_S_rad = np.deg2rad(azimuth_sun)
        self.ground_steps = ground_steps
        self.module_steps = module_steps
        self.angle_steps = angle_steps

        # Define variables derived from these base variables
        self.x_g_array = np.linspace(0, self.dist, self.ground_steps)
        self.x_g_distance = self.dist / (
            self.ground_steps - 1
        )  # distance between two points on x_g_array
        # self.l_array = np.linspace(0,self.L,self.module_steps) # OLD, changed on 26 April 2019!
        self.l_array = (
            np.linspace(self.L / self.module_steps, self.L, self.module_steps)
            - 0.5 * self.L / self.module_steps
        )
        # normal angle of the sun
        self.n_S = np.array(
            [
                np.sin(self.theta_S_rad) * np.cos(-self.phi_S_rad),
                np.sin(self.theta_S_rad) * np.sin(-self.phi_S_rad),
                np.cos(self.theta_S_rad),
            ]
        )

    def module(self):  # some functions and values for the PV module
        """
        Helper function to introduce some characteristic points and vectors of the module
        """
        self.H_m = self.L * np.sin(self.theta_m_rad)
        self.e_m = np.array(
            [np.cos(self.theta_m_rad), np.sin(self.theta_m_rad)]
        )  # unit vector along the module
        self.n_m = np.array(
            [-np.sin(self.theta_m_rad), np.cos(self.theta_m_rad)]
        )  # normal to the module
        self.n_m_3D = np.array(
            [self.n_m[0], 0, self.n_m[1]]
        )  # normal to the module
        self.e_m_3D = np.array([self.e_m[0], 0, self.e_m[1]])

    def calculate_view_factors(self):
        # initializing the results dictionary
        self.results = {}

        self.module()
        self.calc_irradiance_module_sky_direct()
        self.calc_irradiance_module_sky_diffuse()
        self.calc_radiance_ground_direct()
        self.calc_radiance_ground_diffuse()
        self.calc_module_ground_matrix()
        self.calc_irradiance_module_ground_direct()
        self.calc_irradiance_module_ground_diffuse()

        self.results["irradiance_module_front"] = (
            self.results["irradiance_module_front_sky_direct"]
            + self.results["irradiance_module_front_sky_diffuse"]
            + self.results["irradiance_module_front_ground_direct"]
            + self.results["irradiance_module_front_ground_diffuse"]
        )

        self.results["irradiance_module_back"] = (
            self.results["irradiance_module_back_sky_direct"]
            + self.results["irradiance_module_back_sky_diffuse"]
            + self.results["irradiance_module_back_ground_direct"]
            + self.results["irradiance_module_back_ground_diffuse"]
        )
        return self.results

    # IRRADIANCE ON MODULE FROM THE SKY
    def calc_irradiance_module_sky_direct(self):
        """
        Calculates the direct irradiance on the module for one or a series of
        solar positions.
        """

        try:
            temp_irrad = np.zeros((self.n_S.shape[1], self.module_steps))
        except:
            temp_irrad = np.zeros(self.module_steps)

        # cosine of angle between Sun and module normal
        self.cos_alpha_mS = np.dot(self.n_S.T, self.n_m_3D)
        # needed for calculating shadow on module
        angle_term = (
            np.cos(self.theta_m_rad)
            - np.sin(self.theta_m_rad) * self.n_S[0] / self.n_S[2]
        )

        proj_module_plane = projection(self.n_S.T, self.n_m_3D)

        # proj_module_plane = projection(self.n_S.T[13952:13983], self.n_m_3D)
        try:
            proj_module_plane = (
                proj_module_plane / norm(proj_module_plane, axis=1)[:, None]
            )
        except:
            proj_module_plane = proj_module_plane / norm(proj_module_plane)

        phi = np.arccos(np.dot(proj_module_plane, self.e_m_3D))
        phi = np.where(self.n_S[1] > 0, phi, -phi)

        self.tmp["phi"] = phi
        self.tmp["theta"] = np.arccos(self.cos_alpha_mS)

        l_shadow = np.where(
            self.cos_alpha_mS > 0,
            self.L - self.dist / angle_term,
            self.L + self.dist / angle_term,
        )

        try:
            temp_irrad[:] = self.cos_alpha_mS[:, None]
            temp_irrad[np.greater.outer(l_shadow, self.l_array)] = 0
            temp_front = np.where(
                (self.cos_alpha_mS > 0)[:, None], temp_irrad, 0
            )
            temp_back = np.where(
                (self.cos_alpha_mS < 0)[:, None], -temp_irrad, 0
            )
        except:
            temp_irrad[:] = self.cos_alpha_mS
            temp_irrad[np.greater.outer(l_shadow, self.l_array)] = 0
            temp_front = np.where((self.cos_alpha_mS > 0), temp_irrad, 0)
            temp_back = np.where((self.cos_alpha_mS < 0), -temp_irrad, 0)

        self.results["irradiance_module_front_sky_direct"] = temp_front
        self.results["irradiance_module_back_sky_direct"] = temp_back
        self.results["irradiance_module_front_sky_direct_mean"] = np.mean(
            temp_front, axis=-1
        )
        self.results["irradiance_module_back_sky_direct_mean"] = np.mean(
            temp_back, axis=-1
        )

    def calc_irradiance_module_sky_diffuse(self):
        """
        Calculates the irradiance of diffuse sky on the module front.
        The result is only depended on the geometrie of the solar panel array.
        """

        vectors_front = np.multiply.outer(
            self.L - self.l_array, self.e_m
        ) - np.array([self.dist, 0])

        vectors_front_normalized = (
            vectors_front / np.linalg.norm(vectors_front, axis=1)[:, None]
        )

        cos_alpha_2 = np.dot(vectors_front, self.n_m) / np.linalg.norm(
            vectors_front, axis=1
        )

        alpha_2_front = np.arctan2(
            np.cross(self.n_m, vectors_front_normalized),
            np.dot(vectors_front_normalized, self.n_m),
        )

        self.tmp["alpha_2_front"] = alpha_2_front

        # np.arctan2(np.cross(vectors_front_normalized, self.n_m),
        #                  np.dot(vectors_front_normalized, self.n_m))

        spacing_alpha = np.linspace(-np.pi / 2, np.pi / 2, 1200)
        dist_alpha = np.cos(spacing_alpha)

        selector = np.greater.outer(spacing_alpha, self.tmp["alpha_2_front"]).T
        dist_alpha = np.tile(dist_alpha, (self.module_steps, 1))
        dist_alpha[selector] = 0

        np.trapz(
            dist_alpha, np.tile(spacing_alpha, (self.module_steps, 1)), axis=1
        ) / 2

        irradiance_front = (np.sin(alpha_2_front) + 1) / 2.0

        vectors_back = np.multiply.outer(
            self.L - self.l_array, self.e_m
        ) + np.array([self.dist, 0])
        cos_epsilon_1 = np.dot(vectors_back, -self.n_m) / norm(
            vectors_back, axis=1
        )

        self.tmp["epsilon_1_back"] = np.pi / 2 - np.arccos(cos_epsilon_1)

        sin_epsilon_2 = (1 - cos_epsilon_1**2) ** 0.5
        irradiance_back = (1 - sin_epsilon_2) / 2

        self.results["irradiance_module_front_sky_diffuse"] = irradiance_front
        self.results[
            "irradiance_module_front_sky_diffuse_mean"
        ] = irradiance_front.mean()

        self.results["irradiance_module_back_sky_diffuse"] = irradiance_back
        self.results[
            "irradiance_module_back_sky_diffuse_mean"
        ] = irradiance_back.mean()

    def calc_shadow_field(self):
        """
        Calculates the start and end position of shadow of direct sunlight.
        Depends on sun position and array  geometry.

        Returns
        -------
        shadow_start : start position of shadow relativ to (0, 0)
        shadow_end : end position of shadow relativ to (0, 0)
        """

        # calculating shadow position for B0
        shadow_B = -self.H / self.n_S[2] * self.n_S[0]

        # calculating shadow for D0
        D0 = self.L * self.e_m + np.array([0, self.H])
        shadow_D = -D0[1] / self.n_S[2] * self.n_S[0] + D0[0]

        # if shadow position of D0 is smaller then B0, positions need to be flipped.
        flipp_mask = shadow_D < shadow_B
        shadow_start, shadow_end = (
            np.where(flipp_mask, shadow_D, shadow_B),
            np.where(flipp_mask, shadow_B, shadow_D),
        )
        return shadow_start, shadow_end

    def calc_radiance_ground_direct(self):
        """
        Calculates the position resolved direct irradiance on the ground.
        """
        shadow_start, shadow_end = self.calc_shadow_field()
        length_shadow = shadow_end - shadow_start

        # reduce such that start and end is in [0, dist] unit cell
        shadow_start_uc = np.remainder(shadow_start, self.dist)
        shadow_end_uc = np.remainder(shadow_end, self.dist)

        # if length_shadow < self.dist: # only in this case direct Sunlight will hit the ground
        length_shadow = shadow_end - shadow_start
        shadow_filter = length_shadow < self.dist
        shadow_start_uc = np.where(shadow_filter, shadow_start_uc, 0)
        shadow_end_uc = np.where(shadow_filter, shadow_end_uc, self.dist)

        # if the ground position is smaller then shadow start OR larger then
        # shadow end it is directly illuminated if shadow start (uc) < shadow end (uc)
        illum_array_1 = np.greater.outer(
            shadow_start_uc,
            self.x_g_array,
        ) | np.less.outer(shadow_end_uc, self.x_g_array)

        # if the ground position is smaller then shadow start AND larger then
        # shadow end it is directly illuminated if shadow start (uc) > shadow end (uc)
        illum_array_2 = np.greater.outer(
            shadow_start_uc,
            self.x_g_array,
        ) & np.less.outer(shadow_end_uc, self.x_g_array)

        # choose appropriet illumination array

        try:
            illum_array_temp = np.where(
                (shadow_end_uc >= shadow_start_uc)[:, None],
                illum_array_1,
                illum_array_2,
            )
            illum_array_temp = (
                illum_array_temp * np.cos(self.theta_S_rad)[:, None]
            )
        except:
            illum_array_temp = np.where(
                (shadow_end_uc >= shadow_start_uc),
                illum_array_1,
                illum_array_2,
            )
            illum_array_temp = illum_array_temp * np.cos(self.theta_S_rad)

        self.results["radiance_ground_direct_emitted"] = (
            illum_array_temp / np.pi
        )

    def calc_sin_B_i(self, i, x_g):
        """
        Calculates cosinus between array of ground positions x_g and point B_i
        with respect to the surface normal
        (lower end of i'th module)

        Parameters
        ----------
        i : int
            Index of corresponding module
        x_g : numpy array
            array of ground positions

        Returns
        -------
        cosinus : array of cosinus for every position in ground position array
        """
        x = np.subtract.outer(i * self.dist, x_g)
        y = self.H * np.ones_like(x)
        return x / norm([x, y], axis=0)

    def calc_sin_D_i(self, i, x_g):
        """
        Calculates cosinus between array of ground positions x_g and point D_i
        with respect to the surface normal
        (lower end of i'th module)

        Parameters
        ----------
        i : int
            Index of corresponding module
        x_g : numpy array
            array of ground positions

        Returns
        -------
        cosinus : array of cosinus for every position in ground position array
        """
        D_0 = self.L * self.e_m
        x = np.subtract.outer(i * self.dist, x_g) + D_0[0]
        y = (self.H + D_0[1]) * np.ones_like(x)
        return x / norm([x, y], axis=0)

    def calc_radiance_ground_diffuse(self):
        """
        Calculates the position resolved diffuse irradiance on the ground.
        """

        # Check how many periods have to be considered in negativ direction
        lower_bound = 0
        while True:
            cos_B_x = self.calc_sin_B_i(lower_bound, self.x_g_array)
            cos_D_x = self.calc_sin_D_i(lower_bound - 1, self.x_g_array)

            # check if sky is visible for any point between B_x and D_x-1
            if all(cos_B_x < cos_D_x):
                break
            lower_bound = lower_bound - 1

        # Check how many periods have to be considered in positive direction
        upper_bond = 0
        while True:
            B0 = self.calc_sin_B_i(upper_bond, self.x_g_array)
            B1 = self.calc_sin_B_i(upper_bond + 1, self.x_g_array)
            D0 = self.calc_sin_D_i(upper_bond, self.x_g_array)
            D1 = self.calc_sin_D_i(upper_bond + 1, self.x_g_array)
            # check if sky is visible between module x and x+1
            if all(np.maximum(B0, D0) > np.minimum(B1, D1)):
                break
            upper_bond = upper_bond + 1

        sin_eta_arr = self.calc_sin_B_i(
            np.arange(lower_bound, upper_bond + 1), self.x_g_array
        )
        sin_zeta_arr = self.calc_sin_D_i(
            np.arange(lower_bound, upper_bond + 1), self.x_g_array
        )

        arr_eta_zeta = np.stack([sin_eta_arr, sin_zeta_arr])
        # sort such that smallest sin is always first in array
        arr_eta_zeta.sort(axis=0)

        # substract lower angle of ith row from higher angle of i+1 'th row
        sky_view_factors = (
            np.roll(arr_eta_zeta[0], -1, axis=0) - arr_eta_zeta[1]
        )
        # set negative sky_view_factors to zero
        sky_view_factors[sky_view_factors < 0] = 0
        # sum over all "windows" between module rows
        illum_array = sky_view_factors.sum(axis=0) / 2

        irradiance_ground_diffuse_received = illum_array

        # division by pi converts irradiance into radiance assuming Lambertian scattering
        self.results["radiance_ground_diffuse_emitted"] = (
            irradiance_ground_diffuse_received / np.pi
        )

    def module_ground_matrix_helper(self, lower_index, upper_index):
        """
        helper function to calculate intensity matrix (ground view factors)
        of visible ground from upper and lower index.
        Parameters
        ----------
        lower_index: int or numpy array
            Ground index of lowest visible point on the module. Can be array to
            calculate intensity matrix for all points on the module at once.

        uppwer_index: int or numpy array
            Ground index of highest visible point on the module. Can be array to
            calculate intensity matrix for all points on the module at once.
        """
        # initialize matrix, containing the 'ground view factors'
        intensity_matrix = np.zeros(
            (self.module_steps, self.angle_steps, self.ground_steps)
        )
        # broadcasting such that upper and lower index have the same length
        indices = np.stack(list(np.broadcast(lower_index, upper_index)))
        for i, l in enumerate(self.l_array):
            lower_index, upper_index = indices[i]
            index_array = np.arange(lower_index, upper_index + 1)

            # calculate the distance for every visible element on the ground
            ground_dist_array = self.x_g_distance * index_array

            # calculates the angle w.r.t. to the module for every visible element on the ground
            x = ground_dist_array - (l * self.e_m)[0]
            y = -self.H - (l * self.e_m)[1]
            vector = np.stack(list(np.broadcast(x, y)))

            # in rare cases intermidiate results can be slightly above 1, thus we divide 1. something
            alpha_array = np.arcsin(
                np.dot(vector, self.e_m) / norm(vector, axis=1) / 1.000001
            )
            # alpha_array = np.abs(alpha_array)

            # calculates the difference between a element and the next element for integration
            sin_alpha = np.sin(alpha_array)
            delta_sin_alpha = np.abs(sin_alpha - np.roll(sin_alpha, 1))
            delta_sin_alpha[0] = delta_sin_alpha[1]

            # convert 'radiance' to 'irradiance'
            irradiance_factor = delta_sin_alpha * np.pi / 2.0

            # filling intensity matrix
            angle_index = (
                np.floor(
                    alpha_array * self.angle_steps / np.pi
                    + self.angle_steps / 2.0
                ).astype(int)
                - 1
            )
            ground_index = np.remainder(index_array, self.ground_steps)
            try:
                for j in range(len(alpha_array)):
                    intensity_matrix[
                        i, angle_index[j], ground_index[j]
                    ] += irradiance_factor[
                        j
                    ]  # np.pi/2.0*cos_alpha[j]*abs(delta_alpha[j])
            except:
                import ipdb

                ipdb.set_trace()

        return intensity_matrix

    def calc_module_ground_matrix(self):
        """
        Calculates matrix that determines distribution of light from ground on the module
        """
        vec_pos_module = np.multiply.outer(self.l_array, self.e_m)

        # intersection of module vector e_m and ground (X prime in manuscript)
        low_view = self.H / self.e_m[1] * (-self.e_m[0])

        # vector from module position to B-1 (lowest point of next module)
        vec_next_front = np.array([-self.dist, 0]) - vec_pos_module
        # vector from module position to B1 (lowest point of module behind)
        vec_next_back = np.array([self.dist, 0]) - vec_pos_module

        # calculates the highest distance on the ground that is visible to
        # a position on the module
        high_view_front = (
            vec_pos_module[:, 0]
            + (vec_pos_module[:, 1] + self.H)
            / (-vec_next_front[:, 1])
            * vec_next_front[:, 0]
        )
        high_view_back = (
            vec_pos_module[:, 0]
            + (vec_pos_module[:, 1] + self.H)
            / (-vec_next_back[:, 1])
            * vec_next_back[:, 0]
        )

        # calculates the upper and lower index on ground for the visible area
        lower_index_front = np.round(
            high_view_front / self.x_g_distance
        ).astype(int)
        upper_index_front = np.round(low_view / self.x_g_distance).astype(int)

        lower_index_back = np.round(low_view / self.x_g_distance).astype(int)
        upper_index_back = np.round(high_view_back / self.x_g_distance).astype(
            int
        )

        intensity_matrix_front = self.module_ground_matrix_helper(
            lower_index_front, upper_index_front
        )
        intensity_matrix_back = self.module_ground_matrix_helper(
            lower_index_back, upper_index_back
        )

        self.results["module_front_ground_matrix"] = intensity_matrix_front
        self.results["module_back_ground_matrix"] = intensity_matrix_back

    def calc_irradiance_module_ground_direct(self):
        """
        irradiance on the module from the ground originating from direct skylight
        """
        for fb in ["front", "back"]:
            field_name = "irradiance_module_" + fb + "_ground_direct"
            matrix = self.results["module_" + fb + "_ground_matrix"]
            # sum over all angles
            ground_view_factor = np.sum(matrix, axis=1)
            # multiply ground radiance with correspoding ground radiance
            # and sum over ground index
            self.results[field_name] = (
                self.results["radiance_ground_direct_emitted"]
                @ ground_view_factor.T
            )
            self.results[field_name + "_mean"] = self.results[
                field_name
            ].mean()

    def calc_irradiance_module_ground_diffuse(self):
        """
        irradiance on the module from the ground originating from diffuse skylight
        """
        for fb in ["front", "back"]:
            field_name = "irradiance_module_" + fb + "_ground_diffuse"
            matrix = self.results["module_" + fb + "_ground_matrix"]
            # sum over all angles
            ground_view_factor = np.sum(matrix, axis=1)
            # multiply ground radiance with correspoding ground radiance
            # and sum over ground index
            try:
                self.results[field_name] = (
                    ground_view_factor
                    * self.results["radiance_ground_diffuse_emitted"]
                ).sum(axis=1)
            except:
                self.results[field_name] = (
                    ground_view_factor
                    @ self.results["radiance_ground_diffuse_emitted"]
                ).T
            self.results[field_name + "_mean"] = self.results[
                field_name
            ].mean()


class IrradianceSimulator(ViewFactorSimulator):
    def __init__(
        self,
        illumination_df,
        albedo=0.3,
        module_length=1.92,
        module_tilt=52,
        mount_height=0.5,
        module_spacing=7.1,
        **numeric_kw_paras
    ):
        """
        Stil needs docstring
        """

        self.dni = illumination_df.loc[:, "DNI"]
        self.dhi = illumination_df.loc[:, "DHI"]
        self.albedo = albedo

        super().__init__(
            module_length,
            module_tilt,
            mount_height,
            module_spacing,
            zenith_sun=illumination_df["zenith"].values,
            azimuth_sun=illumination_df["azimuth"].values,
            **numeric_kw_paras,
        )

        self.view_factors = self.calculate_view_factors()

        self.dni = illumination_df.loc[:, "DNI"]
        self.dhi = illumination_df.loc[:, "DHI"]
        self.albedo = albedo

    def simulate(self, simple_results=True):
        """
        Stil needs docstring
        """

        try:
            diffuse = np.concatenate(
                [
                    self.view_factors["irradiance_module_front_sky_diffuse"],
                    self.view_factors["irradiance_module_back_sky_diffuse"],
                    self.view_factors[
                        "irradiance_module_front_ground_diffuse"
                    ],
                    self.view_factors["irradiance_module_back_ground_diffuse"],
                ],
            )
            diffuse = np.outer(self.dhi, diffuse)

        except:
            diffuse = np.concatenate(
                [
                    self.view_factors["irradiance_module_front_sky_diffuse"],
                    self.view_factors["irradiance_module_back_sky_diffuse"],
                ],
            )
            diffuse = np.tile(diffuse, (len(self.dhi), 1))
            diffuse = (
                np.concatenate(
                    [
                        self.view_factors[
                            "irradiance_module_front_ground_diffuse"
                        ],
                        self.view_factors[
                            "irradiance_module_back_ground_diffuse"
                        ],
                        diffuse,
                    ],
                    axis=1,
                )
                * (self.dhi).values[:, None]
            )

        direct = (
            np.concatenate(
                [
                    self.view_factors["irradiance_module_front_sky_direct"],
                    self.view_factors["irradiance_module_back_sky_direct"],
                    self.view_factors["irradiance_module_front_ground_direct"],
                    self.view_factors["irradiance_module_back_ground_direct"],
                ],
                axis=1,
            )
            * self.dni.values[:, None]
        )

        # direct_ts = direct#self.dni[:, None] * direct

        column_names = ["front_sky", "back_sky", "front_ground", "back_ground"]
        prefixes = ["_diffuse", "_direct"]
        column_names = [
            name + prefix for prefix in prefixes for name in column_names
        ]

        level_names = ["contribution", "module_position"]
        multi_index = pd.MultiIndex.from_product(
            [column_names, range(self.module_steps)],
            names=level_names,
        )

        results = pd.DataFrame(
            np.concatenate([diffuse, direct], axis=1),
            columns=multi_index,
            index=self.dni.index,
        )

        ground_reflected = results.columns.get_level_values(0).str.contains(
            "ground"
        )

        results.loc[:, ground_reflected] = results.loc[
            :, ground_reflected
        ].apply(lambda x: x * self.albedo, raw=True, axis=0)

        if simple_results:
            back_columns = results.columns.get_level_values(
                "contribution"
            ).str.contains("back")
            front_columns = ~back_columns
            results = pd.concat(
                [
                    results.loc[:, front_columns]
                    .groupby(level="module_position", axis=1)
                    .sum()
                    .mean(axis=1),
                    results.loc[:, back_columns]
                    .groupby(level="module_position", axis=1)
                    .sum()
                    .mean(axis=1),
                ],
                keys=["front", "back"],
                axis=1,
            )

        return results
