from multiprocessing.sharedctypes import Value
import numpy as np
from astropy.io import ascii
import matplotlib.pyplot as plt
import ast
import json
import tqdm
from panoptes_client import Panoptes, Subject, Workflow
from skimage import io
from shapely.geometry import Polygon, Point
from shapely import affinity

def get_subject_image(subject): 
    # get the subject metadata from Panoptes
    subjecti = Subject(int(subject))
    frame0_url = subjecti.raw['locations'][0]['image/png']
    img = io.imread(frame0_url)

    return img

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
            The name of the shape these parameters belong to (see :meth:`panoptes_to_geometry` for
            supported shapes)
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
    circ = Point((x,y)).buffer(1)
    ell  = affinity.scale(circ, rx, ry)
    ellr = affinity.rotate(ell, -angle)

    return ellr


def string_to_np_array(data):
    try:
        return np.asarray(ast.literal_eval(data))
    except ValueError as e:
        return []


class Aggregator:
    def __init__(self, reduction_data):
        self.data = ascii.read(reduction_data, format='csv')

        subjects = np.unique(self.data['subject_id'])

        self.JSON_data = []

        for subject in tqdm.tqdm(subjects):
            dark_ext, dark_clust   = self.get_ellipse_data(subject, 'dark')
            white_ext, white_clust = self.get_ellipse_data(subject, 'white')
            red_ext, red_clust     = self.get_ellipse_data(subject, 'red')
            brown_ext, brown_clust = self.get_ellipse_data(subject, 'brown')

            row = {'subject_id': subject, 
                   'dark_extracts': dark_ext, 'dark_clusters': dark_clust,
                   'white_extracts': white_ext, 'white_clusters': white_clust,
                   'red_extracts': red_ext, 'red_clusters': red_clust,
                   'brown_extracts': brown_ext, 'brown_clusters': brown_clust
                   }

            self.JSON_data.append(row)


    def get_ellipse_data(self, subject, vort_type):
        '''
        data = {}
        data['dark']  = self.dark_vortex_data[self.dark_vortex_data['subject_id']==subject]
        data['red']   = self.red_vortex_data[self.red_vortex_data['subject_id']==subject]
        data['white'] = self.white_vortex_data[self.white_vortex_data['subject_id']==subject]
        data['brown'] = self.brown_vortex_data[self.brown_vortex_data['subject_id']==subject]
        '''

        data = self.data[(self.data['subject_id']==subject)]#&(self.data['reducer_key']==f'{vort_type}-vortex-cluster')]

        toolID = {'dark': 3, 'red': 0, 'white': 1, 'brown': 2}

        clust_data = {}
        ext_data   = {}
        for j, key in enumerate(['dark', 'red', 'white', 'brown']):
            for subkey in ['x', 'y', 'rx', 'ry', 'angle']:
                try:
                    ext_data[subkey] = string_to_np_array(data[f'data.frame0.T0_tool{toolID[vort_type]}_ellipse_{subkey}'][0])
                except KeyError:
                    ext_data[subkey] = []

            for subkey in ['x', 'y', 'rx', 'ry', 'angle','sigma']:
                try:
                    clust_data[subkey] = string_to_np_array(data[f'data.frame0.T0_tool{toolID[vort_type]}_clusters_{subkey}'][0])
                except KeyError:
                    clust_data[subkey] = []
            for subkey in ['labels', 'probabilities']:
                try:
                    clust_data[subkey] = string_to_np_array(data[f'data.frame0.T0_tool{toolID[vort_type]}_cluster_{subkey}'][0])
                except KeyError:
                    clust_data[subkey] = []

            if clust_data['probabilities'] == []:
                clust_data['probabilities'] = [1]*len(ext_data['x'])

        return ext_data, clust_data

    def plot_subject(self, subject, ax=None, keys=['dark', 'red', 'white', 'brown']):

        colors = {'dark': 'k', 'red': 'r', 'white': 'white', 'brown': 'brown'}


        if ax is None:
            fig, ax = plt.subplots(1,1, dpi=150)
        else:
            fig = None

        img = get_subject_image(subject)
        ax.imshow(img)
        
        datai = next(filter(lambda d: d['subject_id'] == subject, self.JSON_data))

        for key in keys:
            #clusti = clust_data[key]
            #exti   = ext_data[key]

            exti   = datai[f'{key}_extracts']
            clusti = datai[f'{key}_clusters']

            for vals in zip(clusti['x'], clusti['y'], clusti['rx'], clusti['ry'], clusti['angle'], clusti['sigma']):
                params = vals[:-1]
                ellr = params_to_shape(params)

                try:
                    plus_sigma, minus_sigma = get_sigma_shape(params, vals[-1])
                    avg_minus = params_to_shape(minus_sigma)
                    avg_plus  = params_to_shape(plus_sigma)

                    x_m, y_m = avg_minus.exterior.xy
                    x_p, y_p = avg_plus.exterior.xy

                    #ax.fill(np.append(x_p, x_m[::-1]), np.append(y_p, y_m[::-1]), color=colors[key], alpha=0.2)
                except ValueError:
                    pass
                ax.plot(*ellr.exterior.xy, '-', color=colors[key], linewidth=1)
            
            for vals in zip(exti['x'], exti['y'], exti['rx'], exti['ry'], exti['angle'], clusti['probabilities']):
                params = vals[:-1]
                prob   = vals[-1]

                ellr = params_to_shape(params)

                ax.plot(*ellr.exterior.xy, '--', color=colors[key], linewidth=0.35, alpha=0.25+0.5*prob)

        ax.set_xlim((0, 384))
        ax.set_ylim((384, 0))
        ax.axis('off')
        
        if fig is not None:
            plt.tight_layout(pad=0)
            plt.show()
