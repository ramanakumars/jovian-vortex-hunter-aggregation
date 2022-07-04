import numpy as np
from astropy.io import ascii
from astropy.table import Table
import matplotlib.pyplot as plt
import ast
import json
from skimage import io
import tqdm
from panoptes_client import Panoptes, Subject, Workflow

def get_subject_image(subject): 
    # get the subject metadata from Panoptes
    subjecti = Subject(int(subject))
    frame0_url = subjecti.raw['locations'][0]['image/png']
    img = io.imread(frame0_url)

    return img

class Aggregator:
    def __init__(self, reduction_data):
        self.data      = ascii.read(reduction_data, format='csv')

        for j in range(5):
            self.data[f'data.{j}'].fill_value = 0
        self.data = self.data.filled()

        # clean up the data 
        self.consensus =  Table(names=('subject_id', 'vortex.consensus', 
                                       'turbulent.consensus', 'cloudbands.consensus',
                                       'featureless.consensus', 'blurry.consensus', 
                                       'extract_count'), dtype=('i4', 'f4', 'f4', 'f4', 'f4', 'f4', 'i4'))
        
        for subject in tqdm.tqdm(np.unique(self.data['subject_id']), desc='Loading subjects'):
            stats_datai = self.data[(self.data['subject_id']==subject)&(self.data['reducer_key']=='vortex_stats')]
            count_datai = self.data[(self.data['subject_id']==subject)&(self.data['reducer_key']=='question_count')]


            try:
                vortex_consensus      = stats_datai['data.0'][0]/count_datai['data.extracts'][0]
                turbulence_consensus  = stats_datai['data.1'][0]/count_datai['data.extracts'][0]
                cloudbands_consensus  = stats_datai['data.2'][0]/count_datai['data.extracts'][0]
                featureless_consensus = stats_datai['data.3'][0]/count_datai['data.extracts'][0]
                blurry_consensus      = stats_datai['data.4'][0]/count_datai['data.extracts'][0]

                self.consensus.add_row((subject, vortex_consensus, turbulence_consensus, 
                                        cloudbands_consensus, featureless_consensus, blurry_consensus,
                                        count_datai['data.extracts'][0]))
            except Exception as e:
                pass

