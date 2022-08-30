import numpy as np
from astropy.io import ascii
import matplotlib.pyplot as plt
import ast
import json
import tqdm
from panoptes_client import Subject
from skimage import io
from .vortex_cluster import get_sigma_shape, params_to_shape


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
        subjects = np.unique(sub_ids)

        self.JSON_data = []

        if autoload:
            for subject in tqdm.tqdm(subjects):
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

        return obj

    def save_JSON(self, outfile):
        with open(outfile, 'w') as outJSON:
            json.dump(self.JSON_data, outJSON, cls=NpEncoder)
        print(f"Saved to {outfile}")

    def get_ellipse_data(self, subject, key, data=None):
        if data is None:
            data = self.data[(self.data['subject_id'] == subject)]

        toolID = {'dark': 3, 'red': 0, 'white': 1,
                  'brown': 2, 'multi-color': 4}
        subtaskID = {'white': 0, 'red': 1, 'brown': 2}

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
        for subkey in ['labels', 'probabilities']:
            try:
                clust_data[subkey] = string_to_np_array(
                    data[
                        f'data.frame0.T0_tool{toolIDi}_cluster_{subkey}'
                    ][0])
            except KeyError:
                clust_data[subkey] = []

        if clust_data['probabilities'] == []:
            clust_data['probabilities'] = [1]*len(ext_data['x'])

        return ext_data, clust_data

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