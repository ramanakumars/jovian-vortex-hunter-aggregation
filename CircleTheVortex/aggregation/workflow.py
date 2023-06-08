import numpy as np
from astropy.io import ascii
import matplotlib.pyplot as plt
import ast
import json
import spiceypy as spice
import os
import tqdm
from panoptes_client import Subject
from skimage import io
from .shape_utils import get_sigma_shape, params_to_shape, IoU_metric
from .vortex import ExtractVortex, ClusterVortex
from .subjects import SubjectLoader


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def get_subject_image(subject):
    # get the subject metadata from Panoptes
    subjecti = Subject(int(subject))
    frame0_url = subjecti.raw['locations'][0]['image/png']
    img = io.imread(frame0_url)

    return img


def string_to_np_array(data):
    try:
        return np.asarray(ast.literal_eval(data))
    except ValueError:
        return []


class Aggregator:
    def __init__(self, sub_file, reduction_data=None, autoload=False):
        self.load_subject_data(sub_file)

        if reduction_data is None:
            return

        self.data = ascii.read(reduction_data, format='csv')

        # furnish the spice kernels for converting to SIII coords
        kernel_path = os.path.dirname(os.path.abspath(__file__))
        spice.furnsh(os.path.join(kernel_path, 'jup380s.bsp'))
        spice.furnsh(os.path.join(kernel_path, 'pck00010.tpc'))

        sub_ids = np.asarray(self.data['subject_id'])
        self.subjects = np.unique(sub_ids)

        if autoload:
            self.create_ellipse_data()

    @classmethod
    def from_JSON(cls, sub_file, JSONfile):
        obj = cls(sub_file, None)

        with open(JSONfile, 'r') as indata:
            JSON_data = json.load(indata)

        obj.ellipses = []
        for e in tqdm.tqdm(JSON_data, desc='Getting ellipses', ascii=True):
            obj.ellipses.append(ClusterVortex.from_dict(e))

        obj.ellipses = np.asarray(obj.ellipses)

        obj.subjects = np.unique([d.subject_id for d in obj.ellipses])

        return obj

    def save_JSON(self, outfile):
        with open(outfile, 'w') as outJSON:
            json.dump([e.as_dict() for e in self.ellipses], outJSON, cls=NpEncoder)

        print(f"Saved to {outfile}")

    def load_subject_data(self, sub_file):
        '''
            Load the subject metadata
        '''
        self.subject_data = SubjectLoader(sub_file)

    def create_ellipse_data(self):
        '''
            Create a list of ellipses from the raw CSV import
        '''
        sub_ids = np.asarray(self.data['subject_id'])

        self.ellipses = []

        for subject in tqdm.tqdm(self.subjects, desc='Parsing data', ascii=True):
            datasub = self.data[np.where(sub_ids == subject)[0]]

            dark_ells = self.get_ellipse_data(subject, 'dark', datasub)
            white_ells = self.get_ellipse_data(subject, 'white', datasub)
            red_ells = self.get_ellipse_data(subject, 'red', datasub)
            brown_ells = self.get_ellipse_data(subject, 'brown', datasub)
            multi_color_ells = self.get_ellipse_data(subject, 'multi-color', datasub)

            self.ellipses.extend([*dark_ells, *white_ells, *red_ells, *brown_ells, *multi_color_ells])

        self.ellipses = np.asarray(self.ellipses)

    def get_ellipse_data(self, subject, key, data=None):
        '''
            Get the ellipse data from the CSV reductions import
        '''
        if data is None:
            data = self.data[(self.data['subject_id'] == subject)]

        toolID = {'dark': 3, 'red': 0, 'white': 1, 'brown': 2, 'multi-color': 4}

        clust_data = {}
        ext_data = {}

        toolIDi = toolID[key]
        for subkey in ['x', 'y', 'rx', 'ry', 'angle']:
            try:
                ext_data[subkey] = string_to_np_array(
                    data[
                        f'data.frame0.T0_tool{toolIDi}_ellipse_{subkey}'
                    ][0])
            except KeyError:
                ext_data[subkey] = []

        for subkey in ['x', 'y', 'rx', 'ry', 'angle', 'sigma']:
            try:
                clust_data[subkey] = string_to_np_array(
                    data[
                        f'data.frame0.T0_tool{toolIDi}_clusters_{subkey}'
                    ][0])
            except KeyError:
                clust_data[subkey] = []

        for subkey in ['labels']:
            try:
                clust_data[subkey] = string_to_np_array(
                    data[
                        f'data.frame0.T0_tool{toolIDi}_cluster_{subkey}'
                    ][0])
            except KeyError:
                clust_data[subkey] = []

        if key == 'multi-color':
            try:
                details = ast.literal_eval(data[
                    'data.frame0.T0_tool4_details'
                ][0])
            except (IndexError, ValueError):
                details = []

        colorID = {'0': 'white', '1': 'red', '2': 'brown'}
        lon, lat, PJ = self.subject_data.get_meta(subject)

        re, rp, _ = spice.bodvar(599, 'RADII', 3)
        f = (re - rp) / re

        # convert from the latitudinal coordinate to SIII lon/lat
        lon, lat, _ = np.degrees(spice.recpgr('JUPITER', spice.srfrec(599, *np.radians([lon, lat])), re, f))

        x0 = np.asarray(clust_data['x'])
        y0 = np.asarray(clust_data['y'])
        w = np.asarray(clust_data['rx'])
        h = np.asarray(clust_data['ry'])
        a = np.asarray(clust_data['angle'])

        sigma = np.asarray(clust_data['sigma'])

        ellipses = []
        for i in range(len(x0)):
            ext_inds = np.where(np.asarray(clust_data['labels']) == i)[0]

            ext_x0 = np.asarray(ext_data['x'])[ext_inds]
            ext_y0 = np.asarray(ext_data['y'])[ext_inds]
            ext_w = np.asarray(ext_data['rx'])[ext_inds]
            ext_h = np.asarray(ext_data['ry'])[ext_inds]
            ext_a = np.asarray(ext_data['angle'])[ext_inds]

            if key == 'multi-color':
                ext_details = np.asarray(details)[ext_inds]

            pari = np.asarray([x0[i], y0[i], w[i], h[i], a[i]])

            elli = ClusterVortex(pari, sigma[i], lon, lat)

            elli.subject_id = subject

            if key == 'multi-color':
                assert len(ext_details) == len(ext_x0), \
                    f"Extracts does not match details for multi-color: {ext_x0}  {details}"

            extracts = []
            for j in range(len(ext_x0)):
                par_ext = np.asarray([ext_x0[j], ext_y0[j], ext_w[j], ext_h[j], ext_a[j]])
                # set a dummy probability now and we will calculate
                # it later using the IoU metric
                ext_ellipse = ExtractVortex(par_ext, 1., lon, lat)
                ext_ellipse.subject_id = subject
                ext_ellipse.perijove = PJ
                ext_ellipse.color = key

                prob = IoU_metric(ext_ellipse.get_points(),
                                  elli.get_points(), reshape=False)

                ext_ellipse.probability = 1. - prob

                if key == 'multi-color':
                    center_color = list(ext_details[j][0].keys())[0]
                    edge_color = list(ext_details[j][1].keys())[0]

                    if center_color != edge_color:
                        ext_ellipse.color = f'{colorID[center_color]}-{colorID[edge_color]}'
                    else:
                        ext_ellipse.color = colorID[center_color]

                extracts.append(ext_ellipse)

            if key == 'multi-color':
                colors = [ext.color for ext in extracts]
                unique_colors, counts = np.unique(colors, return_counts=True)

                color = unique_colors[np.argmax(counts)]
            else:
                color = key

            elli.extracts = extracts
            elli.perijove = PJ
            elli.color = color

            ellipses.append(elli)

        return ellipses

    def get_ellipses(self, gamma_cut=0.6):
        ellipses = []

        if not hasattr(self, 'subject_data'):
            print("Please load the subject data using load_subject_data!")

        for i, sub in enumerate(tqdm.tqdm(self.subjects,
                                          desc='Finding vortices', ascii=True)):
            ellipses_i = self.get_ellipse_subject(sub, gamma_cut=gamma_cut)
            ellipses.extend(ellipses_i)

        return np.asarray(ellipses)

    def get_ellipse_subject(self, sub, gamma_cut):
        if not hasattr(self, 'subject_data'):
            print("Please load the subject data using load_subject_data!")

        if not hasattr(self, 'subject_list'):
            self.subject_list = np.asarray([d.subject_id for d in self.ellipses])

        if not hasattr(self, 'ellipse_confidence'):
            self.ellipse_confidence = np.asarray([d.confidence() for d in self.ellipses])

        ellipses = self.ellipses[(self.subject_list == sub) & (self.ellipse_confidence > gamma_cut)]

        return ellipses

    def plot_subject(self, subject, ax=None,
                     keys=['dark', 'red', 'white', 'brown', 'multi_color'], gamma_cut=0.6):

        colors = {'dark': 'k', 'red': 'r', 'white': 'white', 'brown': 'brown',
                  'red-brown': 'chocolate', 'red-white': 'mistyrose',
                  'brown-red': 'firebrick', 'brown-white': 'rosybrown',
                  'white-red': 'salmon', 'white-brown': 'peru'}

        if ax is None:
            fig, ax = plt.subplots(1, 1, dpi=150)
        else:
            fig = None

        img = get_subject_image(subject)
        ax.imshow(img)

        ellipses = self.get_ellipse_subject(subject, gamma_cut)

        for ellipse in ellipses:
            try:
                plus_sigma, minus_sigma = get_sigma_shape(ellipse.ellipse_params, ellipse.sigma)
                avg_minus = params_to_shape(minus_sigma)
                avg_plus = params_to_shape(plus_sigma)

                x_m, y_m = avg_minus.exterior.xy
                x_p, y_p = avg_plus.exterior.xy

                ax.fill(
                    np.append(x_p, x_m[::-1]),
                    np.append(y_p, y_m[::-1]),
                    color=colors[ellipse.color], alpha=0.2)
            except ValueError:
                pass
            ax.plot(*ellipse.get_points().T, '-', color=colors[ellipse.color], linewidth=1)

            for ext in ellipse.extracts:
                ax.plot(*ext.get_points().T, '--',
                        color=colors[ext.color], linewidth=0.35)

        ax.set_xlim((0, 384))
        ax.set_ylim((384, 0))
        ax.axis('off')

        if fig is not None:
            plt.tight_layout(pad=0)
            plt.show()
