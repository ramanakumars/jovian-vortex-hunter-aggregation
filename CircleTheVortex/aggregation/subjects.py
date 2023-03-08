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

        self.subject_IDs = []
        self.latitudes = []
        self.longitudes = []
        self.perijoves = []
        self.urls = []
        self.classification_counts = []
        self.retired_ats = []
        self.retirement_reasons = []

        # for each subject, add the metadata from the CSV file
        for i, subject_id in enumerate(tqdm.tqdm(subject_IDs,
                                                 desc='Loading subject data', ascii=True)):
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
            self.subject_IDs.append(row[0])
            self.latitudes.append(latitude)
            self.longitudes.append(longitude)
            self.perijoves.append(perijove)
            self.urls.append(location)
            self.classification_counts.append(row[6])
            self.retired_ats.append(row[7])
            self.retirement_reasons.append(row[8])

        self.subject_IDs = np.asarray(self.subject_IDs)
        self.latitudes = np.asarray(self.latitudes)
        self.longitudes = np.asarray(self.longitudes)
        self.perijoves = np.asarray(self.perijoves)
        self.urls = np.asarray(self.urls)
        self.classification_counts = np.asarray(self.classification_counts)
        self.retired_ats = np.asarray(self.retired_ats)
        self.retirement_reasons = np.asarray(self.retirement_reasons)

    def get_meta(self, subject_id):
        '''
            Get the latitude/latitude/perijove for a given subject
        '''
        ind = np.where(self.subject_IDs == subject_id)[0][0]

        return self.longitudes[ind], self.latitudes[ind], self.perijoves[ind]
