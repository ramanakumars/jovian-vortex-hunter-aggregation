import numpy as np
import matplotlib.pyplot as plt
import tqdm
import json
from astropy.io import ascii
from .vortex import MultiSubjectVortex
from scipy.interpolate import interp1d
from aggregation.utils import re, rp


class ZonalWind:
    def __init__(self, wind_data, smooth_u=15, smooth_du=20, smooth_dvort=40, plot=False):
        # load the wind data
        ulat_dat = ascii.read(wind_data)

        # get the latitude and wind values
        lat = self.lat = ulat_dat['lat'][:]
        self.u_orig = ulat_dat['v_eastward'][:]

        self.u = np.convolve(self.u_orig, np.ones(smooth_u) / smooth_u, mode='same')

        # get the shape factors for the ellipsoid
        rln = re / np.sqrt(1. + ((rp / re) * np.tan(np.radians(lat)))**2.)
        dy = rln / (np.cos(np.radians(lat)) * ((np.sin(np.radians(lat)))**2. + ((re / rp) * np.cos(np.radians(lat)))**2.))

        # calculate du/d(lat) with latitude in radians
        dudlat = np.gradient(self.u, np.radians(self.lat))
        # smooth this using a `smooth_du` sized box averaging
        dudlat_smoothed = np.convolve(
            dudlat, np.ones(smooth_du) / smooth_du, mode='same')
        # calculate du/dy using the shape factors above
        dudy = dudlat_smoothed / dy

        # calculate vorticity shear d(du/dy)/d(lat)
        uyy_lat = np.gradient(dudy, np.radians(lat))
        # smooth as before with `smooth_dvort`
        uyy_lat_smoothed = np.convolve(uyy_lat, np.ones(
            smooth_dvort) / smooth_dvort, mode='same')
        # convert to d^2 u / dy^2 using the shape factors
        uyy = uyy_lat_smoothed / dy

        # save these as functions that we can call later
        # cubic works well for the u_lat
        self.u_vs_lat = interp1d(self.lat, self.u, kind='cubic', bounds_error=False)
        # but the other two are too noisy so we'll use linear
        self.du_vs_lat = interp1d(lat, dudy, kind='linear', bounds_error=False)
        self.uyy_vs_lat = interp1d(lat, uyy, kind='linear', bounds_error=False)
        # u_vort = np.asarray([u_vs_lat(e.get_center_lonlat()[1]) for e in vortices])

        if plot:
            fig, axs = plt.subplots(1, 3, dpi=150, sharey=True)
            axs[0].plot(self.u, self.lat, '-',
                        color='dodgerblue', linewidth=0.5)
            axs[1].plot(dudy, self.lat, '-', color='dodgerblue', linewidth=0.5)
            axs[2].plot(uyy, self.lat, '-', color='dodgerblue', linewidth=0.5)
            # plt.plot(ulat_dat[:,0], dudlat_smoothed/dy,'r--', linewidth=1)

            plt.show()


class Vortices:
    def __init__(self, vortex_data, zonal_wind, extract_count=4, threshold=0.7):
        self.zonal_wind = zonal_wind

        # open the vortex data and parse the JSON
        with open(vortex_data, 'r') as infile:
            vortex_dict = json.load(infile)

        # loop through each entry and build it into
        # a vortex object
        # go through each vortex and create
        # the parameter data table
        multi_dim_data = []
        vortices = []
        for vort in tqdm.tqdm(vortex_dict, desc='Loading vortices', ascii=True):
            if (len(vort['extracts']) > extract_count) & (np.sqrt(1 - vort['sigma']**2.) > threshold) & ('subject_ids' in vort.keys()):
                ell = MultiSubjectVortex.from_dict(vort)
                ell.set_color()
                lat = ell.get_center_lonlat()[1]
                coriolis_beta = 2. * (1.76e-4) * np.cos(np.radians(lat)) / rp
                rowi = [zonal_wind.u_vs_lat(lat) / 100,
                        zonal_wind.du_vs_lat(lat) * 1.e4,
                        (coriolis_beta - zonal_wind.uyy_vs_lat(lat)) * 5.e10,
                        ell.sx / 5.e6 - 0.5, ell.sx / ell.sy]

                if not np.isnan(rowi[1]):
                    multi_dim_data.append(rowi)
                    vortices.append(ell)

        self.vortices = np.asarray(vortices)
        self.vort_params = np.asarray(multi_dim_data)

    def plot_hist_sizes(self, bin_width=250):
        fig, axs = plt.subplots(2, 5, dpi=150, figsize=(8, 4), sharex=True)
        bins = np.arange(0, 6000, bin_width)

        colors = {'dark': '#ccc', 'red': 'r', 'white': 'white', 'brown': 'brown',
                  'red-brown': 'chocolate', 'red-white': 'mistyrose',
                  'brown-red': 'firebrick', 'brown-white': 'rosybrown',
                  'white-red': 'salmon', 'white-brown': 'peru'}
        for i, key in enumerate(colors.keys()):
            vortex_sub = list(filter(lambda e: e.color == key, self.vortices))

            axi = axs[i // 5, i % 5]

            axi.hist([ell.sx / 1.e3 for ell in vortex_sub],
                     bins=bins, color=colors[key])
            axi.axvline(np.percentile(
                [ell.sx / 1.e3 for ell in vortex_sub], 50), color='black', linestyle='dashed')
        axs[1, 0].set_xlabel(r'Size [km]')
        axs[1, 1].set_xlabel(r'Size [km]')
        axs[0, 0].set_xlim(left=0)
        plt.show()

    def plot_hist_aspect_ratio(self, nbins=20):
        fig, axs = plt.subplots(2, 5, figsize=(8, 4), dpi=150, sharex=True)
        bins = np.linspace(1, 3, nbins)

        colors = {'dark': '#ccc', 'red': 'r', 'white': 'white', 'brown': 'brown',
                  'red-brown': 'chocolate', 'red-white': 'mistyrose',
                  'brown-red': 'firebrick', 'brown-white': 'rosybrown',
                  'white-red': 'salmon', 'white-brown': 'peru'}
        for i, key in enumerate(colors.keys()):
            vortex_sub = list(filter(lambda e: e.color == key, self.vortices))

            axi = axs[i // 5, i % 5]

        #     print(np.asarray([ell.sx/ell.sy for ell in vortex_sub]))
            aspect_ratio = np.asarray([ell.sx / ell.sy for ell in vortex_sub])
            axi.hist(aspect_ratio, bins=bins, color=colors[key])
            axi.axvline(np.percentile(aspect_ratio, 50),
                        color='black', linestyle='dashed')
        axs[1, 0].set_xlabel(r'Aspect ratio')
        axs[1, 1].set_xlabel(r'Aspect ratio')
        plt.show()
