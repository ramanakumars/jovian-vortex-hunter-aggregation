import numpy as np


flat = 0.06487  # flattening parameter
beta = 1 / (1 - flat)  # rp/re
re = 71492e3  # equatorial radius
rp = re / beta  # polar radius
pixscale = 7000e3 / 384  # pixel scale


def lat_pg(lat_pc):
    '''
        Convert planetocentric latitude to planetographic
        Inputs
        ------
        lat_pc : float
            planetocentric latitude in degress

        Returns
        -------
        lat_pg : float
            planetographic latitude in degrees
    '''
    lat_pg = np.degrees(np.arctan(np.tan(np.abs(np.radians(lat_pc))) *
                                  beta * beta))
    return np.sign(lat_pc) * lat_pg


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


def vincenty(p1, p2, MAX_ITER=500):
    '''
        Calculates the vincenty distance between two
        (lon, lat) pairs on Jupiter
    '''

    lon1, lat1 = np.radians(p1)
    lon2, lat2 = np.radians(p2)

    U1 = np.arctan((1 - flat) * np.tan(lat1))
    U2 = np.arctan((1 - flat) * np.tan(lat2))

    L = lon2 - lon1
    lam = L

    for _ in range(MAX_ITER):
        sin_sigma = np.sqrt(
            (np.cos(U1) * np.sin(lam))**2. +
            (np.cos(U1) * np.sin(U2) - np.sin(U1) *
             np.cos(U2) * np.cos(lam))**2.
        )

        cos_sigma = np.sin(U1) * np.sin(U2) + np.cos(U1) * \
            np.cos(U2) * np.cos(lam)

        sigma = np.arctan2(sin_sigma, cos_sigma)

        sin_alpha = np.cos(U1) * np.cos(U2) * np.sin(lam) / np.sin(sigma)
        cossq_alpha = 1. - sin_alpha**2.

        cos_2sigm = np.cos(sigma) - 2. * np.sin(U1) * np.sin(U2) / cossq_alpha

        C = (flat / 16.) * cossq_alpha * (4 + flat * (4 - 3 * cossq_alpha))

        lam_new = L + (1 - C) * flat * sin_alpha *\
            (sigma + C * sin_sigma * (cos_2sigm + C * cos_sigma *
                                      (-1 + 2 * cos_2sigm**2.)
                                      )
             )

        dlam = np.abs(lam - lam_new)

        lam = lam_new

        if dlam < 1.e-10:
            break

    u_sq = cossq_alpha * (re**2. - rp**2.) / (rp**2.)

    A = 1. + u_sq / 16384 * (4096 + u_sq *
                             (-768 + u_sq * (320 - 175 * u_sq))
                             )
    B = u_sq / 1024 * (256 + u_sq * (-128 + u_sq * (74 - 47 * u_sq)))

    dsig = B * sin_sigma * (cos_2sigm + B / 4 *
                            (cos_sigma * (-1 + 2 * cos_2sigm**2.) -
                             (B / 6) * cos_2sigm * (-3 + 4 * sin_sigma**2.) *
                                (-3 + 4 * cos_2sigm**2.))
                            )
    s = rp * A * (sigma - dsig)

    return s, np.degrees(sigma)
