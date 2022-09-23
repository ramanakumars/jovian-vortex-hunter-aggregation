import json
import os.path
import sys
import tqdm
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
try:
    from aggregation import Aggregator
    from aggregation.workflow import NpEncoder
    from aggregation.vortex_cluster import (cluster_vortices,
                                            average_vortex_cluster)
except Exception:
    pass

aggregator = Aggregator.from_JSON('reductions/data.json')
aggregator.load_subject_data('../subjects_data.csv')

ellipses = aggregator.get_ellipses(sigma_cut=0.8, prob_cut=0.)

print()

avg_ellipses = []

pbar = tqdm.tqdm(total=(36-13)*4, desc='Finding vortices', position=0)

for PJ in range(13, 36):
    for key in ['white', 'red', 'brown', 'dark']:
        ellipses_i = list(filter(lambda e: e.perijove == PJ,
                                 ellipses[key]))

        # get the clusters and lone vortices
        lone, groups = cluster_vortices(ellipses_i)

        pbar.set_postfix_str(f"{key}: {len(ellipses_i)} => "
                             f"{len(groups)} gr, {len(lone)} ln")

        for ell_group in tqdm.tqdm(groups, position=1, leave=False,
                                   desc='Averaging groups'):
            ells_ext = []
            _ = [ells_ext.extend(ell.extracts) for ell in ell_group]

            avg_ell = average_vortex_cluster(ells_ext)

            if avg_ell.sigma < 0.6:
                avg_ellipses.append(avg_ell)

        for ell in lone:
            avg_ellipses.append(ell)

        pbar.update(1)


with open('reductions/vortices.json', 'w') as outfile:
    json.dump([e.as_dict() for e in avg_ellipses], outfile,
              cls=NpEncoder)
