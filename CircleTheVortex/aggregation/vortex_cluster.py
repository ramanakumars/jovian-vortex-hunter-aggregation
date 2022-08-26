from .utils import pixel_to_lonlat, lonlat_to_pixel
import numpy as np
import scipy.optimize
from copy import deepcopy
import tqdm
from shapely.geometry import Point, Polygon
from shapely import affinity


def scale_shape(params, gamma):
    '''
        Scales the ellipse by a factor of gamma
    '''
    return [
        # center is the same
        params[0],
        params[1],
        # radius_x and radius_y scale
        gamma * params[2],
        gamma * params[3],
        # angle does not change
        params[4]
    ]


def get_sigma_shape(params, sigma):
    '''
        Return the plus and minus one sigma shape given the starting parameters
        and sigma value.
        Parameters
        ----------
        params : list
            A list of the parameters for the shape (as defined by PFE)
        shape : string
            The name of the shape these parameters belong to
            (see :meth:`panoptes_to_geometry` for supported shapes)
        sigma : float
            The standard deviation used to scale up and down the input shape
        Returns
        -------
        plus_sigma : list
            A list of shape parameters for the 1-sigma scaled up average
        minus_sigma : list
            A list of shape parameters for the 1-sigma scaled down average
    '''
    gamma = np.sqrt(1 - sigma)
    plus_sigma = scale_shape(params, 1 / gamma)
    minus_sigma = scale_shape(params, gamma)
    return plus_sigma, minus_sigma


def params_to_shape(params):
    '''
        Converts the parameter list for an ellipse
        to a `shapely.geometry.Polygon` object
    '''
    x, y, rx, ry, angle = params
    circ = Point((x, y)).buffer(1)
    ell = affinity.scale(circ, rx, ry)
    ellr = affinity.rotate(ell, -angle)

    return ellr


def average_bounds(params_list):
    '''Find the bounding box for the average shape for each of the shapes
    parameters.
    Parameters
    ----------
    params_list : list
        A list of shape parameters that are being averaged
    shape : string
        The shape these parameters belong to
        (see :meth:`panoptes_to_geometry` for supported shapes)
    Returns
    -------
    bound : list
        This is a list of tuples giving the min and max bounds for
        each shape parameter.
    '''
    geo = params_to_shape(params_list[0])
    # Use the union of all shapes to find the bounding box
    for params in params_list[1:]:
        geo = geo.union(params_to_shape(params))
    # bound on x
    bx = (geo.bounds[0], geo.bounds[2])
    # bound on y
    by = (geo.bounds[1], geo.bounds[3])
    # width of geo
    dx = bx[1] - bx[0]
    # height of geo
    dy = by[1] - by[0]
    # bound is a list of tuples giving (min, max) values
    # for each paramters of the shape
    bound = [bx, by]
    # bound on width or radius_x, min set to 1 pixel
    bound.append((1, dx))
    # bound on height or radius_y, min set to 1 pixel
    bound.append((1, dy))
    # bound on angle (capped at 180 due to symmetry)
    bound.append((0, 180))

    return bound


def average_shape_IoU(params_list, probs, shape='ellipse'):
    '''Find the average shape and standard deviation from a
    list of parameters with respect to the IoU metric.
    Parameters
    ----------
    params_list : list
        A list of shape parameters that are being averaged
    shape : string
        The shape these parameters belong to
        (see :meth:`panoptes_to_geometry` for supported shapes)
    Returns
    -------
    average_shape : list
        A list of shape parameters for the average shape
    sigma : float
        The standard deviation of the input shapes with
        respect to the IoU metric
    '''
    def sum_distance(x):
        return sum([probs[i] * IoU_metric(params_to_shape(x),
                                          params_to_shape(params_list[i]),
                                          reshape=False)**2
                    for i in range(len(params_list))])
    # find shape that minimizes the variance in the IoU metric using bounds
    m = scipy.optimize.shgo(
        sum_distance,
        sampling_method='sobol',
        bounds=average_bounds(params_list)
    )
    # find the 1-sigma value
    sigma = np.sqrt(m.fun / (len(params_list) - 1))
    return list(m.x), sigma


def IoU_metric(params1, params2, reshape=True):
    '''Find the Intersection of Union distance between two shapes.
    Parameters
    ----------
    params1 : list
        A list of the parameters for shape 1 (as defined by PFE)
    params2 : list
        A list of the parameters for shape 2 (as defined by PFE)
    shape : string
        The shape these parameters belong to
        (see :meth:`panoptes_to_geometry` for supported shapes)
    Returns
    -------
    distance : float
        The IoU distance between the two shapes.
        0 means the shapes are the same,
        1 means the shapes don't overlap, values in the middle mean partial
        overlap.
    '''
    if reshape:
        par1 = params1.reshape((65, 2))
        par2 = params2.reshape((65, 2))
    else:
        par1 = params1
        par2 = params2

    geo1 = Polygon(par1)
    geo2 = Polygon(par2)
    intersection = geo1.intersection(geo2).area
    union = geo1.union(geo2).area
    if union == 0:
        # catch divide by zero (i.e. cases when neither shape has an area)
        return np.inf
    return 1 - intersection / union


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

            if IoUs[1:].sum() == 0:
                lone_ellipses.append(elli)
            else:
                ellipse_groups.append(ellipse_queue[delete_mask])

            pbar.update(len(delete_mask))

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
