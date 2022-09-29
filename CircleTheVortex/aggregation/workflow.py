import numpy as np
from astropy.io import ascii
import matplotlib.pyplot as plt
import ast
import json
import tqdm
from panoptes_client import Subject
from skimage import io
from .shape_utils import get_sigma_shape, params_to_shape, IoU_metric
from .vortex import ExtractVortex, ClusterVortex
from .subjects import SubjectLoader
from .utils import lat_pg


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
    def __init__(self, reduction_data=None, autoload=True):
        if reduction_data is None:
            return

        self.data = ascii.read(reduction_data, format='csv')

        sub_ids = np.asarray(self.data['subject_id'])
        self.subjects = np.unique(sub_ids)

        self.JSON_data = []

        if autoload:
            for subject in tqdm.tqdm(self.subjects, desc='Parsing data'):
                datasub = self.data[np.where(sub_ids == subject)[0]]
                dark_ext, dark_clust = self.get_ellipse_data(
                    subject, 'dark', datasub)
                white_ext, white_clust = self.get_ellipse_data(
                    subject, 'white', datasub)
                red_ext, red_clust = self.get_ellipse_data(
                    subject, 'red', datasub)
                brown_ext, brown_clust = self.get_ellipse_data(
                    subject, 'brown', datasub)

                row = {'subject_id': subject,
                       'dark_extracts': dark_ext,
                       'dark_clusters': dark_clust,
                       'white_extracts': white_ext,
                       'white_clusters': white_clust,
                       'red_extracts': red_ext,
                       'red_clusters': red_clust,
                       'brown_extracts': brown_ext,
                       'brown_clusters': brown_clust
                       }

                self.JSON_data.append(row)

    @classmethod
    def from_JSON(cls, JSONfile):
        obj = cls(None)

        with open(JSONfile, 'r') as indata:
            obj.JSON_data = json.load(indata)

        obj.subjects = np.unique([d['subject_id'] for d in obj.JSON_data])

        return obj

    def save_JSON(self, outfile):
        with open(outfile, 'w') as outJSON:
            json.dump(self.JSON_data, outJSON, cls=NpEncoder)
        print(f"Saved to {outfile}")

    def load_subject_data(self, sub_file):
        self.subject_data = SubjectLoader(sub_file)

    def get_ellipse_data(self, subject, key, data=None):
        if data is None:
            data = self.data[(self.data['subject_id'] == subject)]

        toolID = {'dark': 3, 'red': 0, 'white': 1,
                  'brown': 2, 'multi-color': 4}

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

        return ext_data, clust_data

    def get_ellipses(self, sigma_cut=0.6, prob_cut=0.5):
        ellipses = {'white': [], 'red': [], 'brown': [], 'dark': []}

        if not hasattr(self, 'subject_data'):
            print("Please load the subject data using load_subject_data!")

        for key in ['white', 'red', 'brown', 'dark']:
            extracts = filter(lambda d: (len(d[f'{key}_clusters']['x']) > 0),
                              self.JSON_data)
            extracts = filter(lambda d: (max(d[f'{key}_clusters']['sigma'])
                                         < sigma_cut),
                              extracts)
            best_ids = np.asarray([x['subject_id'] for x in extracts])

            for i, sub in enumerate(tqdm.tqdm(best_ids,
                                              desc=f'{key} vortices')):
                ellipses_i = self.get_ellipse_subject(sub, key,
                                                      sigma_cut=sigma_cut,
                                                      prob_cut=prob_cut)
                ellipses[key].extend(ellipses_i)

            ellipses[key] = np.asarray(ellipses[key])

        return ellipses

    def get_ellipse_subject(self, sub, key, sigma_cut=0.6, prob_cut=0.8):
        if not hasattr(self, 'subject_data'):
            print("Please load the subject data using load_subject_data!")

        datai = next(
            filter(lambda d: d['subject_id'] == sub, self.JSON_data))
        inds = np.where(np.asarray(
            datai[f'{key}_clusters']['sigma']) < sigma_cut)[0]

        lon, lat, PJ = self.subject_data.get_meta(sub)

        lat = lat_pg(lat)

        x0 = np.asarray(datai[f'{key}_clusters']['x'])[inds]
        y0 = np.asarray(datai[f'{key}_clusters']['y'])[inds]

        w = np.asarray(datai[f'{key}_clusters']['rx'])[inds]
        h = np.asarray(datai[f'{key}_clusters']['ry'])[inds]
        a = np.asarray(datai[f'{key}_clusters']['angle'])[inds]

        sigma = np.asarray(datai[f'{key}_clusters']['sigma'])[inds]

        ellipses = []
        for i in range(len(x0)):

            ext_inds = np.where(np.asarray(
                datai[f'{key}_clusters']['labels']) == inds[i])[0]
            ext_x0 = np.asarray(
                datai[f'{key}_extracts']['x'])[ext_inds]
            ext_y0 = np.asarray(
                datai[f'{key}_extracts']['y'])[ext_inds]
            ext_w = np.asarray(
                datai[f'{key}_extracts']['rx'])[ext_inds]
            ext_h = np.asarray(
                datai[f'{key}_extracts']['ry'])[ext_inds]
            ext_a = np.asarray(
                datai[f'{key}_extracts']['angle'])[ext_inds]

            pari = np.asarray([x0[i], y0[i], w[i], h[i], a[i]])

            elli = ClusterVortex(pari, sigma[i], lon, lat)

            elli.subject_id = sub

            extracts = []
            for j in range(len(ext_x0)):
                par_ext = np.asarray([ext_x0[j], ext_y0[j], ext_w[j],
                                      ext_h[j], ext_a[j]])
                # set a dummy probability now and we will calculate 
                # it later using the IoU metric
                ext_ellipse = ExtractVortex(par_ext, 1.,
                                            lon, lat)
                ext_ellipse.subject_id = sub
                ext_ellipse.perijove = PJ
                ext_ellipse.color = key

                prob = IoU_metric(ext_ellipse.get_points(),
                                  elli.get_points(), reshape=False)

                ext_ellipse.probability = 1. - prob

                extracts.append(ext_ellipse)

            elli.extracts = extracts
            elli.perijove = PJ
            elli.color = key

            ellipses.append(elli)

        return ellipses

    def plot_subject(self, subject, ax=None,
                     keys=['dark', 'red', 'white', 'brown'], sigmacut=None):

        colors = {'dark': 'k', 'red': 'r', 'white': 'white', 'brown': 'brown'}

        if ax is None:
            fig, ax = plt.subplots(1, 1, dpi=150)
        else:
            fig = None

        img = get_subject_image(subject)
        ax.imshow(img)

        datai = next(
            filter(lambda d: d['subject_id'] == subject, self.JSON_data))

        for key in keys:
            exti = datai[f'{key}_extracts']
            clusti = datai[f'{key}_clusters']

            for vals in zip(clusti['x'], clusti['y'], clusti['rx'],
                            clusti['ry'], clusti['angle'], clusti['sigma']):
                params = vals[:-1]

                if sigmacut is not None:
                    if vals[-1] > sigmacut:
                        continue

                ellr = params_to_shape(params)

                try:
                    plus_sigma, minus_sigma = get_sigma_shape(params, vals[-1])
                    avg_minus = params_to_shape(minus_sigma)
                    avg_plus = params_to_shape(plus_sigma)

                    x_m, y_m = avg_minus.exterior.xy
                    x_p, y_p = avg_plus.exterior.xy

                    ax.fill(np.append(
                        x_p, x_m[::-1]), np.append(y_p, y_m[::-1]),
                        color=colors[key], alpha=0.2)
                except ValueError:
                    pass
                ax.plot(*ellr.exterior.xy, '-', color=colors[key], linewidth=1)

            for vals in zip(exti['x'], exti['y'], exti['rx'], exti['ry'],
                            exti['angle'], clusti['probabilities']):
                params = vals[:-1]
                prob = vals[-1]

                ellr = params_to_shape(params)

                ax.plot(*ellr.exterior.xy, '--',
                        color=colors[key], linewidth=0.35, alpha=0.25+0.5*prob)

        ax.set_xlim((0, 384))
        ax.set_ylim((384, 0))
        ax.axis('off')

        if fig is not None:
            plt.tight_layout(pad=0)
            plt.show()
