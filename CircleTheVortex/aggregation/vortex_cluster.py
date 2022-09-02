from .utils import pixel_to_lonlat, lonlat_to_pixel
from .shape_funcs import average_shape_IoU, IoU_metric, \
    params_to_shape
import numpy as np
from copy import deepcopy
import tqdm


def cluster_vortices(ellipses):
    '''
        Given a set of ellipses, find groups of vortices
        and cluster them together. Also filter out "lone" ellipses
        that only have one classification
    '''

    # make a copy of the ellipses
    ellipse_queue = np.array(ellipses)

    lone_ellipses = []
    ellipse_groups = []

    # first find the IoU between all the ellipses
    # to find the lone vortices
    with tqdm.tqdm(total=len(ellipse_queue)) as pbar:
        while len(ellipse_queue) > 0:
            nellipses = len(ellipse_queue)
            elli = ellipse_queue[0]

            # get the IoU distance
            IoUs = np.zeros(nellipses)
            IoUs[0] = 1  # IoU wrt itself

            for j in range(1, nellipses):
                # get the IoU between the ellipses in lon/lat space
                IoUs[j] = 1. - IoU_metric(elli.convert_to_lonlat(),
                                          ellipse_queue[j].convert_to_lonlat(),
                                          reshape=False)

            delete_mask = np.where(IoUs > 0)[0]

            # check the IoUs
            if IoUs[1:].sum() == 0:
                # if there are no overlapping vortices
                # then we add this to the lone vortex bin
                lone_ellipses.append(elli)
            else:
                # if not, then we add all overlapping
                # vortices to the group
                ellipse_groups.append(ellipse_queue[delete_mask])

            pbar.update(len(delete_mask))

            # remove the overlapping elements from the group
            ellipse_queue = np.delete(ellipse_queue, delete_mask)

    print(f"{len(lone_ellipses)} vortices; {len(ellipse_groups)} groups")

    return lone_ellipses, ellipse_groups


def average_vortex_cluster(ellipses):
    '''
        Given a cluster of ellipes, find the average ellipse
        This is done in the 'physical' frame and output ellipse is also
        in the physical frame
    '''

    lats_mask = []
    lons_mask = []

    # go through all the subjects in this list
    # and get the center so that we can project
    # everything to that frame
    for i, ell in enumerate(ellipses):
        lats_mask.append(ell.lat0)
        lons_mask.append(ell.lon0)

    # image cluster center
    lat0 = np.mean(lats_mask)
    lon0 = np.mean(lons_mask)

    # now loop through all the ellipses and reproject
    # onto that center
    new_ells = []

    for i, ell in enumerate(ellipses):
        lon, lat = ell.lon0, ell.lat0

        # find the new image center
        x0, y0 = lonlat_to_pixel(lon, lat, lon0, lat0, x0=0, y0=0)

        # update the ellipse to match the new center
        ell_new = deepcopy(ell)

        ell_new.ellipse_params[0] = x0 + ell_new.ellipse_params[0] - 192
        ell_new.ellipse_params[1] = y0 + ell_new.ellipse_params[1] - 192

        new_ells.append(ell_new)

    # get the average of this cluster of ellipses
    avg_shape, sigma = average_shape_IoU(
        [ell.ellipse_params for ell in new_ells],
        [ell.sigma for ell in new_ells])

    # propagate this back to the lat/lon grid
    # based on the fact that (0, 0) corresponds to the
    # (lon0, lat0)
    avg_ellipse = Vortex(avg_shape, sigma, lon0, lat0, x0=0, y0=0)

    return avg_ellipse


class Vortex:
    '''
        Generic class to hold information about the reduced vortex
        Also provides helper functions for transforming between
    '''

    def __init__(self, ellipse_params, sigma, lon0, lat0, x0=192, y0=192):
        self.ellipse_params = ellipse_params
        self.sigma = sigma

        self.lon0 = lon0
        self.lat0 = lat0

        self.x0 = x0
        self.y0 = y0

    @property
    def subject_id(self):
        return self.subject_id_

    @subject_id.setter
    def subject_id(self, subject_id):
        self.subject_id_ = subject_id

    def get_points(self):
        ell = params_to_shape(self.ellipse_params).exterior.xy
        return np.dstack((ell[0], ell[1]))[0, :]

    def convert_to_lonlat(self):
        points = self.get_points()
        xx, yy = points[:, 0], points[:, 1]

        lons, lats = pixel_to_lonlat(xx, yy, self.lon0, self.lat0,
                                     self.x0, self.y0)

        return np.dstack((lons, lats))[0, :]

    def get_center_lonlat(self):
        xc, yc = self.ellipse_params[:2]
        lonc, latc = pixel_to_lonlat(xc, yc, self.lon0, self.lat0,
                                     self.x0, self.y0)

        return (lonc, latc)
