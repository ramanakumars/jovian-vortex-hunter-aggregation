import numpy as np
from astropy.io import ascii
import tqdm
import ast


class SubjectLoader:

    def __init__(self, manifest):
        self.raw_data = ascii.read(manifest, format='csv')

        # create a clone of the table that we can fill in
        self.data = []

        subject_IDs, ind = np.unique(
            self.raw_data['subject_id'], return_index=True)

        # for each subject, add the metadata from the CSV file
        for i, subject_id in enumerate(tqdm.tqdm(subject_IDs,
                                                 desc='Loading subject data')):
            # get the row corresponding to this subject
            row = self.raw_data[ind[i]]

            # get the metadata and parse it
            meta = ast.literal_eval(row[4])
            try:
                latitude = float(meta['latitude'])
                longitude = float(meta['longitude'])
                perijove = int(meta['perijove'])
            except KeyError:
                # fails for original subject set
                continue

            # parse the url
            location = ast.literal_eval(row[5])["0"]

            # add this to the list
            self.data.append({'subject_id': row[0],
                              'latitude': latitude,
                              'longitude': longitude,
                              'perijove': perijove,
                              'url': location,
                              'classification_count': row[6],
                              'retired_at': row[7],
                              'retirement_reason': row[8]})

    def get_meta(self, subject_id):
        '''
            Get the latitude/latitude/perijove for a given subject
        '''
        rowi = next(filter(lambda d: d['subject_id'] ==
                           subject_id, self.data))
        return rowi['longitude'], rowi['latitude'], rowi['perijove']
