import numpy as np
from .utils import pixel_to_lonlat, vincenty
from .shape_utils import params_to_shape, get_major_minor_axis, IoU_metric


class BaseVortex:
    '''
        Generic class to hold information about the vortex information
        and corresponding metadata. Also provides helper functions
        for transforming between coordinate systems
    '''

    def __init__(self, ellipse_params, lon0, lat0, x0=192, y0=192):
        self.ellipse_params = ellipse_params

        self.lon0 = lon0
        self.lat0 = lat0

        self.x0 = x0
        self.y0 = y0

        self.autorotate()
        self.get_physical_extents()

    @classmethod
    def from_dict(cls, data):
        ellipse_params = []
        for i, key in enumerate(['x', 'y', 'rx', 'ry', 'angle']):
            ellipse_params.append(data[key])

        if 'sigma' in data.keys():
            conf = data['sigma']
        elif 'probability' in data.keys():
            conf = data['probability']

        ell = cls(ellipse_params, conf, data['lon'], data['lat'], data['x'], data['y'])

        if 'subject_id' in data.keys():
            ell.subject_id = data['subject_id']
        elif 'subject_ids' in data.keys():
            ell.subject_ids = data['subject_ids']

        ell.perijove = data['perijove']
        ell.color = data['color']

        if 'center_color' in data.keys():
            ell.center_color = data['center_color']
            ell.edge_color = data['edge_color']

        if 'extracts' in data.keys():
            extracts = []
            for extract in data['extracts']:
                ext_ell = ExtractVortex.from_dict(extract)

                extracts.append(ext_ell)

            ell.extracts = extracts

        return ell

    @property
    def subject_id(self):
        return self.subject_id_

    @subject_id.setter
    def subject_id(self, subject_id):
        self.subject_id_ = subject_id

    @property
    def extracts(self):
        return self.extracts_

    @extracts.setter
    def extracts(self, extracts):
        self.extracts_ = extracts

    @property
    def perijove(self):
        return self.perijove_

    @perijove.setter
    def perijove(self, PJ):
        self.perijove_ = PJ

    @property
    def color(self):
        return self.color_

    @color.setter
    def color(self, color):
        self.color_ = color

    def autorotate(self):
        x0, y0, rx, ry, a = self.ellipse_params

        if ry > rx:
            # switch the semi-major and semi-minor axes
            # if the semi-major < semi-minor
            rx, ry = ry, rx

            # add 90 to the rotation
            # to compensate for the switch
            a += 90

            # modify the rotation so it fits in [-180, 180]
            while a > 180:
                a -= 180
            while a < -180:
                a += 180

        # update the parameters
        self.ellipse_params = [x0, y0, rx, ry, a]

    def get_physical_extents(self):
        corner_points = get_major_minor_axis(self.ellipse_params)
        corner_lons, corner_lats = pixel_to_lonlat(corner_points[:, 0],
                                                   corner_points[:, 1],
                                                   self.lon0, self.lat0,
                                                   self.x0, self.y0)

        corner_points = np.dstack((corner_lons, corner_lats))[0, :]

        self.sx, self.Lx = vincenty(corner_points[0], corner_points[2])
        self.sy, self.Ly = vincenty(corner_points[1], corner_points[3])

    def get_points(self):
        if not hasattr(self, 'points'):
            # cache the calculation when doing this repeatedly
            ell = params_to_shape(self.ellipse_params).exterior.xy
            self.points = np.dstack((ell[0], ell[1]))[0, :]

        return self.points

    def convert_to_lonlat(self):
        if not hasattr(self, 'points_lonlat'):
            # cache the calculation when doing this repeatedly
            points = self.get_points()
            xx, yy = points[:, 0], points[:, 1]

            lons, lats = pixel_to_lonlat(xx, yy, self.lon0, self.lat0,
                                         self.x0, self.y0)

            self.points_lonlat = np.dstack((lons, lats))[0, :]

        return self.points_lonlat.copy()

    def get_center_lonlat(self):
        xc, yc = self.ellipse_params[:2]
        lonc, latc = pixel_to_lonlat(xc, yc, self.lon0, self.lat0,
                                     self.x0, self.y0)

        return (lonc, latc)

    def as_dict(self):
        outdict = {}

        if hasattr(self, 'subject_id_'):
            outdict['subject_id'] = self.subject_id
        elif hasattr(self, 'subject_ids_'):
            outdict['subject_ids'] = self.subject_ids
        else:
            raise ValueError("Please assign subject ID(s) to this vortex")

        if hasattr(self, 'perijove_'):
            outdict['perijove'] = self.perijove
        else:
            raise ValueError("Please assign a perijove to this vortex")

        if hasattr(self, 'color_'):
            outdict['color'] = self.color
        else:
            raise ValueError("Please assign a color label to this vortex")

        outdict['lon'], outdict['lat'] = self.get_center_lonlat()

        for i, key in enumerate(['x', 'y', 'rx', 'ry', 'angle']):
            outdict[key] = self.ellipse_params[i]

        if hasattr(self, 'sigma'):
            outdict['sigma'] = self.sigma
        elif hasattr(self, 'probability'):
            outdict['probability'] = self.probability

        outdict['angular_width'] = self.Lx
        outdict['angular_height'] = self.Ly
        outdict['physical_width'] = self.sx
        outdict['physical_height'] = self.sy

        if hasattr(self, 'center_color'):
            outdict['center_color'] = self.center_color
            outdict['edge_color'] = self.edge_color

        if hasattr(self, 'extracts_'):
            outdict['extracts'] = []

            for extract in self.extracts:
                outdict['extracts'].append(extract.as_dict())

        return outdict


class ExtractVortex(BaseVortex):
    '''
        Extension of the base vortex class to support
        extract information. Vortex confidence is given by
        the cluster probability from the HDBSCAN shape reducer
    '''

    def __init__(self, ellipse_params, prob, lon0, lat0, x0=192, y0=192):
        self.ellipse_params = ellipse_params
        self.probability = prob

        self.lon0 = lon0
        self.lat0 = lat0

        self.x0 = x0
        self.y0 = y0

        self.autorotate()
        self.get_physical_extents()

    def confidence(self):
        return self.probability


class ClusterVortex(BaseVortex):
    '''
        Extension of the base vortex class to support
        reduction information. Vortex confidence is given by
        the average cluster sigma from the shape averaging
        function (see `average_shape_IoU`)
    '''

    def __init__(self, ellipse_params, sigma, lon0, lat0, x0=192, y0=192):
        self.ellipse_params = ellipse_params
        self.sigma = sigma

        self.gamma = np.sqrt(1. - self.sigma**2.)

        self.lon0 = lon0
        self.lat0 = lat0

        self.x0 = x0
        self.y0 = y0

        self.autorotate()
        self.get_physical_extents()

    @property
    def ext_IoUs(self):
        if not hasattr(self, '_ext_IoU'):
            self._ext_IoUs = np.zeros(len(self.extracts))
            for j, ext in enumerate(self.extracts):
                self._ext_IoUs[j] = 1. - IoU_metric(self.convert_to_lonlat(),
                                                    ext.convert_to_lonlat(),
                                                    reshape=False)
        return self._ext_IoUs

    def confidence(self):
        return self.gamma


class MultiSubjectVortex(ClusterVortex):
    '''
        Extension of the cluster vortex class to support
        vortices generated by aggregating ellipses that span
        multiple subjects
    '''
    @property
    def subject_ids(self):
        return self.subject_ids_

    @subject_ids.setter
    def subject_ids(self, subject_ids):
        self.subject_ids_ = subject_ids

    @classmethod
    def from_dict(cls, data):
        obj = super(MultiSubjectVortex, cls).from_dict(data)
        obj.set_color()
        return obj

    def set_color(self):
        exts = self.extracts

        colors = [ext.color for ext in exts]

        un_colors, counts = np.unique(colors, return_counts=True)

        # convert to an agreement fraction
        counts = counts / np.sum(counts)

        # the final color is the one with the most votes
        self.color = un_colors[np.argmax(counts)]
        self.colors = {un_colors[i]: counts[i] for i in range(len(counts))}

    def as_dict(self):
        outdict = super(MultiSubjectVortex, self).as_dict()

        outdict['colors'] = self.colors

        return outdict
