import numpy as np


flat = 0.06487  # flattening parameter
re = 71492e3  # equatorial radius
rp = re * (1 - flat)  # polar radius
pixscale = 7000e3 / 384  # pixel scale


def pixel_to_lonlat(x, y, lon0, lat0, x0=192, y0=192):
    '''
        Get the (lon, lat) of a pixel given the
        pixel positions and center latitude

    '''

    # find the distance in pixel coordinates from the center
    dx = x - x0
    dy = y0 - y  # opposite to x because of image inversion

    # calculate the shape factors
    rln = re / np.sqrt(1. + ((rp / re) * np.tan(np.radians(lat0)))**2.)
    rlt = rln / (np.cos(np.radians(lat0)) * ((np.sin(np.radians(lat0)))**2. +
                                             ((re / rp) *
                                             np.cos(np.radians(lat0)))**2.))

    # difference between image center to pixel in degrees
    dlat = np.degrees(dy * (pixscale / rlt))
    dlon = np.degrees(dx * (pixscale / rln))

    return lon0 + dlon, lat0 + dlat


def lonlat_to_pixel(lon, lat, lon0, lat0, x0=192, y0=192):
    '''
        Get the pixel positions of (lon, lat)
        given the (lon0, lat0) of the center of the image
    '''
    # calculate the shape factors
    rln = re / np.sqrt(1. + ((rp / re) * np.tan(np.radians(lat)))**2.)
    rlt = rln / (np.cos(np.radians(lat)) * ((np.sin(np.radians(lat)))**2. +
                                            ((re / rp) *
                                            np.cos(np.radians(lat)))**2.))

    # difference between center and requested position
    dlat = lat0 - lat
    dlon = lon - lon0

    # now in pixel coordinates
    dy = np.radians(dlat) / (pixscale / rlt)
    dx = np.radians(dlon) / (pixscale / rln)

    # difference from the image center
    x = dx + x0
    y = dy + y0

    return x, y
