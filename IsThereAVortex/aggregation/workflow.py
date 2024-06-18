import numpy as np
from astropy.io import ascii
from astropy.table import Table
from skimage import io
import tqdm
from panoptes_client import Subject


def get_subject_image(subject):
    # get the subject metadata from Panoptes
    subjecti = Subject(int(subject))
    frame0_url = subjecti.raw['locations'][0]['image/png']
    img = io.imread(frame0_url)

    return img


class Aggregator:
    def __init__(self, reduction_data):
        self.data = ascii.read(reduction_data, format='csv')

        for j in range(5):
            self.data[f'data.{j}'].fill_value = 0
        self.data = self.data.filled()

        # clean up the data
        self.consensus = Table(names=('subject_id', 'vortex.consensus',
                                      'turbulent.consensus', 'cloudbands.consensus',
                                      'featureless.consensus', 'blurry.consensus',
                                      'extract_count'), dtype=('i4', 'f4', 'f4', 'f4', 'f4', 'f4', 'i4'))

        subjects = np.asarray(self.data['subject_id'][:])
        reducer_key = np.asarray(self.data['reducer_key'][:])
        stats_data = self.data[np.where(reducer_key == 'vortex_stats')[0]]
        subjects_stats = np.asarray(stats_data['subject_id'][:])
        
        counts_data = self.data[np.where(reducer_key == 'question_count')[0]]
        subjects_counts = np.asarray(counts_data['subject_id'][:])

        for subject in tqdm.tqdm(np.unique(self.data['subject_id']), desc='Loading subjects'):
            stats_datai = stats_data[np.where(subjects_stats == subject)[0]]
            count_datai = counts_data[np.where(subjects_counts == subject)[0]]

            try:
                counts = count_datai['data.classifications'][0]
                vortex_consensus = stats_datai['data.0'][0] / counts
                turbulence_consensus = stats_datai['data.1'][0] / counts
                cloudbands_consensus = stats_datai['data.2'][0] / counts
                featureless_consensus = stats_datai['data.3'][0] / counts
                blurry_consensus = stats_datai['data.4'][0] / counts

                self.consensus.add_row((subject, vortex_consensus, turbulence_consensus,
                                        cloudbands_consensus, featureless_consensus, blurry_consensus,
                                        counts))
            except Exception as e:
                print(e)
                pass
