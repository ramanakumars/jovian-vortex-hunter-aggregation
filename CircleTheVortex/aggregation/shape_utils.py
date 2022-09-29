import numpy as np
import scipy.optimize
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


def average_shape_IoU(params_list, sigma):
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
        The confidence of the input shapes with
        respect to the IoU metric
    '''
    def sum_distance(x):
        return sum([(sigma[i]*IoU_metric(params_to_shape(x),
                                         params_to_shape(params_list[i]),
                                         reshape=False))**2.
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


def get_major_minor_axis(params):
    '''
        Get the points on the major and minor axis in order
        from the positive-x going clockwise
        Parameters
        ----------
        params : list
            A list of parameters for the ellipse

        Returns
        -------
        points : numpy.ndarray
            A set of coordinates for the four
            major/minor axis points
    '''
    x0, y0, rx, ry, a = params

    a = np.radians(a)

    rotation = np.asarray([[np.cos(a), np.sin(a)],
                           [-np.sin(a), np.cos(a)]])

    points = []
    for theta in [0, 1, 2, 3]:
        theta *= np.pi/2

        x, y = rx*np.cos(theta), ry*np.sin(theta)
        points.append([x, y])

    points = np.asarray(points)
    points_rot = np.dot(rotation, points.T).T

    points_rot[:, 0] += x0
    points_rot[:, 1] += y0

    return points_rot
