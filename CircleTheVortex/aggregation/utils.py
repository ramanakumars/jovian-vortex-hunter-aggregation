import numpy as np
import matplotlib.pyplot as plt


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


def lat_pc(lat_pg):
    '''
        Convert planetographic latitude to planetocentric
        Inputs
        ------
        lat_pg : float
            planetographic latitude in degress

        Returns
        -------
        lat_pc : float
            planetocentric latitude in degrees
    '''
    lat_pc = np.degrees(np.arctan(np.tan(np.abs(np.radians(lat_pg))) /
                                  (beta * beta)))
    return np.sign(lat_pg) * lat_pc


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


def plot_ellipse(ell):
    fig, ax = plt.subplots(1, 1, dpi=150, figsize=(8, 4))
    # read in the mosaic for this perijove
    jup_map = plt.imread(
        f'/home/rsankar/zooniverse/JuDE/backend/PJimgs/PJ{ell.perijove}/globe_mosaic_highres.png')[:, :, :3]

    # get the points for this ellipse
    center = np.asarray(ell.get_center_lonlat())
    center[1] = lat_pc(center[1])
    points = ell.convert_to_lonlat().T
    points[1, :] = lat_pc(points[1, :])

    # read in the extents for the map for this ellipse
    dlon = max([ell.Lx * 2, 10])
    dlat = max([ell.Ly * 2, 3])
    x0, x1 = max([center[0] - dlon, -180]), min([center[0] + dlon, 180])
    y0, y1 = max([center[1] - dlat, -90]), min([center[1] + dlat, 90])

    # convert this to point coordinates
    sy, ey = 4500 - int((y0 + 90) * 25), 4500 - int((y1 + 90) * 25)
    sx, ex = int((x0 + 180) * 25), int((x1 + 180) * 25)

    # plot the map
    ax.imshow(jup_map[sy:ey:-1, sx:ex],
              extent=(x0, x1, y1, y0), aspect='equal')

    # plot out all the colors for this ellipse
    colors = {'dark': 'k', 'red': 'r', 'white': 'white', 'brown': 'brown',
              'red-brown': 'chocolate', 'red-white': 'mistyrose',
              'brown-red': 'firebrick', 'brown-white': 'rosybrown',
              'white-red': 'salmon', 'white-brown': 'peru'}

    for j, ell_ext in enumerate(ell.extracts):
        points = ell_ext.convert_to_lonlat().T
        points[1, :] = lat_pc(points[1:])
        ax.plot(*points, '--', linewidth=0.15, color=colors[ell_ext.color])

    # plot the cluster
    points = ell.convert_to_lonlat().T
    points[1, :] = lat_pc(points[1:])
    ax.plot(*center, 'x', color=colors[ell.color])
    ax.plot(*points, '-', color=colors[ell.color], linewidth=0.5)

    ax.set_aspect('equal')

    ax.set_xlabel(r'Longitude [$\degree$]')
    ax.set_ylabel(r'Latitude [$\degree$]')

    ax.set_xlim((x0, x1))
    ax.set_ylim((y0, y1))

    plt.tight_layout()
    plt.show()
    plt.close(fig)
